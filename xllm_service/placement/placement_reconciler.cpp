/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "placement/placement_reconciler.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <string_view>
#include <tuple>

#include <xxhash.h>

namespace xllm_service::placement {
namespace {

using EngineIdentity = std::pair<std::string, std::string>;

bool valid_config(const PlacementReconcileConfig& config) {
  return config.max_operations_per_cycle > 0 &&
         config.max_operations_per_pool > 0 &&
         config.max_create_per_cycle > 0 &&
         config.max_drain_per_cycle > 0 &&
         config.max_create_per_cycle <= config.max_operations_per_cycle &&
         config.max_drain_per_cycle <= config.max_operations_per_cycle;
}

bool valid_action(PlacementOperationAction action) {
  switch (action) {
    case PlacementOperationAction::CREATE:
    case PlacementOperationAction::BEGIN_DRAIN:
    case PlacementOperationAction::CANCEL_DRAIN:
    case PlacementOperationAction::TERMINATE:
      return true;
  }
  return false;
}

bool valid_operation_status(PlacementOperationStatus status) {
  switch (status) {
    case PlacementOperationStatus::PLANNED:
    case PlacementOperationStatus::SUBMITTED:
    case PlacementOperationStatus::IN_PROGRESS:
    case PlacementOperationStatus::UNKNOWN:
    case PlacementOperationStatus::SUCCEEDED:
    case PlacementOperationStatus::FAILED:
    case PlacementOperationStatus::FENCED:
      return true;
  }
  return false;
}

bool managed_state(PlacementLifecycleState state) {
  return state == PlacementLifecycleState::LOADING ||
         state == PlacementLifecycleState::WARMING ||
         state == PlacementLifecycleState::READY;
}

PlacementReconcileResult result(PlacementReconcileStatus status,
                                PlacementReconcileReason reason,
                                uint32_t desired_replicas,
                                uint32_t managed_replicas,
                                uint32_t pending_operations) {
  return PlacementReconcileResult{
      .status = status,
      .reason = reason,
      .desired_replicas = desired_replicas,
      .managed_replicas = managed_replicas,
      .pending_operations = pending_operations,
  };
}

void append_u64(uint64_t value, std::string* output) {
  for (size_t index = 0; index < sizeof(value); ++index) {
    output->push_back(static_cast<char>(value >> (index * 8)));
  }
}

void append_component(std::string_view value, std::string* output) {
  append_u64(value.size(), output);
  output->append(value.data(), value.size());
}

std::string digest_hex(const XXH128_hash_t& digest) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string value;
  value.reserve(32);
  for (size_t index = 0; index < sizeof(digest.low64); ++index) {
    const uint8_t byte = static_cast<uint8_t>(digest.low64 >> (index * 8));
    value.push_back(kHex[byte >> 4]);
    value.push_back(kHex[byte & 0x0f]);
  }
  for (size_t index = 0; index < sizeof(digest.high64); ++index) {
    const uint8_t byte = static_cast<uint8_t>(digest.high64 >> (index * 8));
    value.push_back(kHex[byte >> 4]);
    value.push_back(kHex[byte & 0x0f]);
  }
  return value;
}

}  // namespace

PlacementReconcileResult reconcile_placement_pool(
    const PlacementReconcileConfig& config,
    const PlacementDesiredState& desired,
    const PlacementLeaderIdentity& leader,
    const std::vector<PlacementReplicaFact>& replicas,
    const std::vector<PlacementOperationView>& operations,
    uint64_t now_ms) {
  if (!valid_config(config)) {
    return result(PlacementReconcileStatus::INVALID_INPUT,
                  PlacementReconcileReason::INVALID_CONFIG,
                  desired.desired_replicas,
                  0,
                  0);
  }
  if (!valid_placement_desired_state(desired) ||
      !valid_placement_leader_identity(leader) ||
      desired.leader.address != leader.address ||
      desired.leader.incarnation != leader.incarnation || now_ms == 0) {
    return result(PlacementReconcileStatus::INVALID_INPUT,
                  PlacementReconcileReason::INVALID_DESIRED,
                  desired.desired_replicas,
                  0,
                  0);
  }
  if (operations.size() > config.max_operations_per_pool) {
    return result(PlacementReconcileStatus::HOLD,
                  PlacementReconcileReason::OPERATION_CAPACITY,
                  desired.desired_replicas,
                  0,
                  0);
  }

  uint32_t managed_replicas = 0;
  std::set<EngineIdentity> engine_identities;
  for (const PlacementReplicaFact& replica : replicas) {
    if (!placement_pool_keys_equal(replica.pool, desired.pool) ||
        !valid_placement_identity(replica.engine_uid) ||
        !valid_placement_identity(replica.engine_incarnation) ||
        !valid_placement_lifecycle_state(replica.state) ||
        !std::isfinite(replica.cache_value) || replica.cache_value < 0.0 ||
        replica.observed_at_ms == 0 ||
        replica.observed_at_ms > now_ms || replica.stable_since_ms == 0 ||
        replica.stable_since_ms > replica.observed_at_ms ||
        !engine_identities
             .insert({replica.engine_uid, replica.engine_incarnation})
             .second) {
      return result(PlacementReconcileStatus::INVALID_INPUT,
                    PlacementReconcileReason::INVALID_ACTUAL,
                    desired.desired_replicas,
                    managed_replicas,
                    0);
    }
    if (managed_state(replica.state)) {
      ++managed_replicas;
    }
  }

  uint32_t pending_operations = 0;
  std::set<std::string> operation_ids;
  std::set<EngineIdentity> engines_with_operations;
  for (const PlacementOperationView& operation : operations) {
    const bool has_engine_uid = !operation.engine_uid.empty();
    const bool has_engine_incarnation = !operation.engine_incarnation.empty();
    if (!valid_placement_identity(operation.operation_id) ||
        !valid_action(operation.action) ||
        !valid_operation_status(operation.status) ||
        !placement_pool_keys_equal(operation.pool, desired.pool) ||
        !valid_placement_identity(operation.leader_incarnation) ||
        operation.desired_generation == 0 ||
        !operation_ids.insert(operation.operation_id).second ||
        has_engine_uid != has_engine_incarnation ||
        (has_engine_uid &&
         (!valid_placement_identity(operation.engine_uid) ||
          !valid_placement_identity(operation.engine_incarnation))) ||
        (operation.action != PlacementOperationAction::CREATE &&
         !has_engine_uid)) {
      return result(PlacementReconcileStatus::INVALID_INPUT,
                    PlacementReconcileReason::INVALID_OPERATION,
                    desired.desired_replicas,
                    managed_replicas,
                    pending_operations);
    }
    if (placement_operation_terminal(operation.status)) {
      continue;
    }
    ++pending_operations;
    if (operation.action != PlacementOperationAction::CREATE &&
        !engines_with_operations
             .insert({operation.engine_uid, operation.engine_incarnation})
             .second) {
      return result(PlacementReconcileStatus::INVALID_INPUT,
                    PlacementReconcileReason::INVALID_OPERATION,
                    desired.desired_replicas,
                    managed_replicas,
                    pending_operations);
    }
    if (operation.action == PlacementOperationAction::CREATE) {
      const EngineIdentity created_identity{operation.engine_uid,
                                            operation.engine_incarnation};
      if (!has_engine_uid ||
          engine_identities.find(created_identity) == engine_identities.end()) {
        ++managed_replicas;
      }
    }
  }
  if (pending_operations > 0) {
    return result(PlacementReconcileStatus::HOLD,
                  PlacementReconcileReason::PENDING_OPERATION,
                  desired.desired_replicas,
                  managed_replicas,
                  pending_operations);
  }

  PlacementReconcileResult output =
      result(PlacementReconcileStatus::OK,
             PlacementReconcileReason::CONVERGED,
             desired.desired_replicas,
             managed_replicas,
             0);
  const uint32_t remaining_operation_capacity =
      config.max_operations_per_pool -
      static_cast<uint32_t>(operations.size());
  const uint32_t cycle_capacity = std::min(
      config.max_operations_per_cycle, remaining_operation_capacity);
  if (managed_replicas < desired.desired_replicas) {
    const uint32_t deficit = desired.desired_replicas - managed_replicas;
    const uint32_t create_count =
        std::min({deficit, config.max_create_per_cycle, cycle_capacity});
    if (create_count == 0) {
      output.status = PlacementReconcileStatus::HOLD;
      output.reason = PlacementReconcileReason::OPERATION_CAPACITY;
      return output;
    }
    output.reason = PlacementReconcileReason::SCALE_UP;
    output.intents.reserve(create_count);
    for (uint32_t ordinal = 0; ordinal < create_count; ++ordinal) {
      output.intents.emplace_back(PlacementOperationIntent{
          .operation_id = make_placement_operation_id(
              leader,
              desired.generation,
              desired.pool,
              PlacementOperationAction::CREATE,
              ordinal,
              "",
              ""),
          .action = PlacementOperationAction::CREATE,
          .pool = desired.pool,
          .leader_incarnation = leader.incarnation,
          .desired_generation = desired.generation,
          .ordinal = ordinal,
      });
    }
    return output;
  }
  if (managed_replicas == desired.desired_replicas) {
    return output;
  }

  std::vector<const PlacementReplicaFact*> victims;
  for (const PlacementReplicaFact& replica : replicas) {
    const EngineIdentity identity{replica.engine_uid,
                                  replica.engine_incarnation};
    if (replica.state == PlacementLifecycleState::READY && replica.fresh &&
        replica.drain_capable && replica.active_reservations == 0 &&
        replica.active_transfers == 0 &&
        engines_with_operations.find(identity) == engines_with_operations.end()) {
      victims.push_back(&replica);
    }
  }
  std::sort(victims.begin(),
            victims.end(),
            [](const PlacementReplicaFact* left,
               const PlacementReplicaFact* right) {
              return std::tie(left->cache_value,
                              left->stable_since_ms,
                              left->engine_uid,
                              left->engine_incarnation) <
                     std::tie(right->cache_value,
                              right->stable_since_ms,
                              right->engine_uid,
                              right->engine_incarnation);
            });
  const uint32_t excess = managed_replicas - desired.desired_replicas;
  const uint32_t drain_count = std::min(
      {excess,
       config.max_drain_per_cycle,
       cycle_capacity,
       static_cast<uint32_t>(victims.size())});
  if (drain_count == 0) {
    output.status = PlacementReconcileStatus::HOLD;
    output.reason = PlacementReconcileReason::NO_SAFE_VICTIM;
    return output;
  }
  output.reason = PlacementReconcileReason::SCALE_DOWN;
  output.intents.reserve(drain_count);
  for (uint32_t ordinal = 0; ordinal < drain_count; ++ordinal) {
    const PlacementReplicaFact& victim = *victims[ordinal];
    output.intents.emplace_back(PlacementOperationIntent{
        .operation_id = make_placement_operation_id(
            leader,
            desired.generation,
            desired.pool,
            PlacementOperationAction::BEGIN_DRAIN,
            ordinal,
            victim.engine_uid,
            victim.engine_incarnation),
        .action = PlacementOperationAction::BEGIN_DRAIN,
        .pool = desired.pool,
        .engine_uid = victim.engine_uid,
        .engine_incarnation = victim.engine_incarnation,
        .leader_incarnation = leader.incarnation,
        .desired_generation = desired.generation,
        .ordinal = ordinal,
    });
  }
  return output;
}

std::string make_placement_operation_id(
    const PlacementLeaderIdentity& leader,
    uint64_t desired_generation,
    const PlacementPoolKey& pool,
    PlacementOperationAction action,
    uint32_t ordinal,
    const std::string& engine_uid,
    const std::string& engine_incarnation) {
  if (!valid_placement_leader_identity(leader) || desired_generation == 0 ||
      !valid_placement_pool_key(pool) || !valid_action(action) ||
      (action != PlacementOperationAction::CREATE &&
       (!valid_placement_identity(engine_uid) ||
        !valid_placement_identity(engine_incarnation)))) {
    return "";
  }
  std::string preimage = "xllm-placement-operation-v3";
  append_component(leader.incarnation, &preimage);
  append_u64(desired_generation, &preimage);
  append_component(placement_pool_key_suffix(pool), &preimage);
  append_u64(static_cast<uint8_t>(action), &preimage);
  append_u64(ordinal, &preimage);
  append_component(engine_uid, &preimage);
  append_component(engine_incarnation, &preimage);
  constexpr uint64_t kOperationIdSeed = 0x584c4c4d5633504cULL;
  const XXH128_hash_t digest = XXH3_128bits_withSeed(
      preimage.data(), preimage.size(), kOperationIdSeed);
  return "placement-v3:" + digest_hex(digest);
}

bool placement_operation_terminal(PlacementOperationStatus status) {
  return status == PlacementOperationStatus::SUCCEEDED ||
         status == PlacementOperationStatus::FAILED ||
         status == PlacementOperationStatus::FENCED;
}

const char* placement_operation_action_name(PlacementOperationAction action) {
  switch (action) {
    case PlacementOperationAction::CREATE:
      return "CREATE";
    case PlacementOperationAction::BEGIN_DRAIN:
      return "BEGIN_DRAIN";
    case PlacementOperationAction::CANCEL_DRAIN:
      return "CANCEL_DRAIN";
    case PlacementOperationAction::TERMINATE:
      return "TERMINATE";
  }
  return "UNKNOWN";
}

const char* placement_reconcile_reason_name(PlacementReconcileReason reason) {
  switch (reason) {
    case PlacementReconcileReason::CONVERGED:
      return "CONVERGED";
    case PlacementReconcileReason::SCALE_UP:
      return "SCALE_UP";
    case PlacementReconcileReason::SCALE_DOWN:
      return "SCALE_DOWN";
    case PlacementReconcileReason::PENDING_OPERATION:
      return "PENDING_OPERATION";
    case PlacementReconcileReason::OPERATION_CAPACITY:
      return "OPERATION_CAPACITY";
    case PlacementReconcileReason::NO_SAFE_VICTIM:
      return "NO_SAFE_VICTIM";
    case PlacementReconcileReason::INVALID_CONFIG:
      return "INVALID_CONFIG";
    case PlacementReconcileReason::INVALID_DESIRED:
      return "INVALID_DESIRED";
    case PlacementReconcileReason::INVALID_ACTUAL:
      return "INVALID_ACTUAL";
    case PlacementReconcileReason::INVALID_OPERATION:
      return "INVALID_OPERATION";
    case PlacementReconcileReason::CLOCK_REGRESSION:
      return "CLOCK_REGRESSION";
  }
  return "UNKNOWN";
}

}  // namespace xllm_service::placement
