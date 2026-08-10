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

#include <xxhash.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <string_view>
#include <tuple>

namespace xllm_service::placement {
namespace {

using EngineIdentity = std::pair<std::string, std::string>;

inline constexpr uint32_t kPlacementCycleOrdinalBits = 16;
inline constexpr uint32_t kMaxPlacementCycleOperations =
    (1U << kPlacementCycleOrdinalBits) - 1;

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
    case PlacementOperationStatus::CANCELED:
      return true;
  }
  return false;
}

bool managed_state(PlacementLifecycleState state) {
  return state == PlacementLifecycleState::LOADING ||
         state == PlacementLifecycleState::WARMING ||
         state == PlacementLifecycleState::READY;
}

uint64_t cycle_ordinal(uint64_t now_ms, uint32_t ordinal) {
  return (now_ms << kPlacementCycleOrdinalBits) | ordinal;
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

bool valid_placement_reconcile_config(const PlacementReconcileConfig& config) {
  return config.max_operations_per_cycle > 0 &&
         config.max_operations_per_pool > 0 &&
         config.max_create_per_cycle > 0 && config.max_drain_per_cycle > 0 &&
         config.terminal_visibility_grace_ms > 0 &&
         config.max_operations_per_cycle <= kMaxPlacementCycleOperations &&
         config.max_create_per_cycle <= config.max_operations_per_cycle &&
         config.max_drain_per_cycle <= config.max_operations_per_cycle;
}

PlacementReconcileResult reconcile_placement_pool(
    const PlacementReconcileConfig& config,
    const PlacementDesiredState& desired,
    const PlacementLeaderIdentity& leader,
    const std::vector<PlacementReplicaFact>& replicas,
    const std::vector<PlacementOperationView>& operations,
    uint64_t now_ms) {
  if (!valid_placement_reconcile_config(config)) {
    return result(PlacementReconcileStatus::INVALID_INPUT,
                  PlacementReconcileReason::INVALID_CONFIG,
                  desired.desired_replicas,
                  0,
                  0);
  }
  if (!valid_placement_desired_state(desired) ||
      !valid_placement_leader_identity(leader) ||
      desired.leader.address != leader.address ||
      desired.leader.incarnation != leader.incarnation ||
      desired.leader.epoch != leader.epoch || now_ms == 0 ||
      now_ms > (std::numeric_limits<uint64_t>::max() >>
                kPlacementCycleOrdinalBits)) {
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
  std::set<EngineIdentity> managed_identities;
  std::map<EngineIdentity, PlacementLifecycleState> actual_states;
  for (const PlacementReplicaFact& replica : replicas) {
    if (!placement_pool_keys_equal(replica.pool, desired.pool) ||
        !valid_placement_identity(replica.engine_uid) ||
        !valid_placement_identity(replica.engine_incarnation) ||
        !valid_placement_lifecycle_state(replica.state) ||
        !std::isfinite(replica.cache_value) || replica.cache_value < 0.0 ||
        replica.observed_at_ms == 0 || replica.observed_at_ms > now_ms ||
        replica.stable_since_ms == 0 ||
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
      managed_identities.emplace(replica.engine_uid,
                                 replica.engine_incarnation);
    }
    actual_states.emplace(
        EngineIdentity{replica.engine_uid, replica.engine_incarnation},
        replica.state);
  }

  uint32_t pending_operations = 0;
  std::set<std::string> operation_ids;
  std::map<EngineIdentity, std::set<PlacementOperationAction>>
      engine_operations;
  std::vector<const PlacementOperationView*> pending_drains;
  std::map<EngineIdentity, uint64_t> active_drain_generations;
  std::map<EngineIdentity, uint64_t> succeeded_drain_generations;
  std::map<EngineIdentity, std::pair<uint64_t, uint64_t>>
      succeeded_cancel_proofs;
  std::map<EngineIdentity, uint64_t> terminated_drain_generations;
  bool current_generation_terminal_failure = false;
  for (const PlacementOperationView& operation : operations) {
    const bool has_engine_uid = !operation.engine_uid.empty();
    const bool has_engine_incarnation = !operation.engine_incarnation.empty();
    if (!valid_placement_identity(operation.operation_id) ||
        !valid_action(operation.action) ||
        !valid_operation_status(operation.status) ||
        !placement_pool_keys_equal(operation.pool, desired.pool) ||
        !valid_placement_identity(operation.leader_incarnation) ||
        operation.leader_epoch == 0 || operation.desired_generation == 0 ||
        operation.updated_at_ms == 0 || operation.updated_at_ms > now_ms ||
        !operation_ids.insert(operation.operation_id).second ||
        (operation.status == PlacementOperationStatus::CANCELED &&
         operation.action != PlacementOperationAction::BEGIN_DRAIN) ||
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
      const EngineIdentity identity{operation.engine_uid,
                                    operation.engine_incarnation};
      if (operation.status != PlacementOperationStatus::SUCCEEDED &&
          operation.status != PlacementOperationStatus::CANCELED &&
          operation.desired_generation == desired.generation) {
        current_generation_terminal_failure = true;
      }
      if (operation.status == PlacementOperationStatus::SUCCEEDED &&
          operation.action == PlacementOperationAction::CREATE) {
        if (actual_states.find(identity) == actual_states.end() &&
            now_ms - operation.updated_at_ms <=
                config.terminal_visibility_grace_ms) {
          ++managed_replicas;
        }
      } else if (operation.status == PlacementOperationStatus::SUCCEEDED &&
                 operation.action == PlacementOperationAction::BEGIN_DRAIN) {
        succeeded_drain_generations[identity] =
            std::max(succeeded_drain_generations[identity],
                     operation.desired_generation);
      } else if (operation.status == PlacementOperationStatus::SUCCEEDED &&
                 operation.action == PlacementOperationAction::CANCEL_DRAIN) {
        auto& proof = succeeded_cancel_proofs[identity];
        if (operation.desired_generation > proof.first ||
            (operation.desired_generation == proof.first &&
             operation.updated_at_ms > proof.second)) {
          proof = {operation.desired_generation, operation.updated_at_ms};
        }
      } else if (operation.status == PlacementOperationStatus::SUCCEEDED &&
                 operation.action == PlacementOperationAction::TERMINATE) {
        terminated_drain_generations[identity] =
            std::max(terminated_drain_generations[identity],
                     operation.desired_generation);
      }
      if (operation.action == PlacementOperationAction::BEGIN_DRAIN &&
          operation.status != PlacementOperationStatus::CANCELED) {
        active_drain_generations[identity] = std::max(
            active_drain_generations[identity], operation.desired_generation);
      }
      continue;
    }
    ++pending_operations;
    if (operation.action != PlacementOperationAction::CREATE &&
        !engine_operations[{operation.engine_uid, operation.engine_incarnation}]
             .insert(operation.action)
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
    } else if (operation.action == PlacementOperationAction::BEGIN_DRAIN) {
      pending_drains.push_back(&operation);
      const EngineIdentity identity{operation.engine_uid,
                                    operation.engine_incarnation};
      active_drain_generations[identity] = std::max(
          active_drain_generations[identity], operation.desired_generation);
    }
  }
  for (const auto& [identity, actions] : engine_operations) {
    static_cast<void>(identity);
    if (actions.size() > 2 ||
        (actions.size() == 2 &&
         (actions.find(PlacementOperationAction::BEGIN_DRAIN) ==
              actions.end() ||
          actions.find(PlacementOperationAction::CANCEL_DRAIN) ==
              actions.end()))) {
      return result(PlacementReconcileStatus::INVALID_INPUT,
                    PlacementReconcileReason::INVALID_OPERATION,
                    desired.desired_replicas,
                    managed_replicas,
                    pending_operations);
    }
  }
  for (const auto& [identity, proof] : succeeded_cancel_proofs) {
    const auto active_drain = active_drain_generations.find(identity);
    const bool superseded_by_new_drain =
        active_drain != active_drain_generations.end() &&
        active_drain->second > proof.first;
    const auto committed_drain = succeeded_drain_generations.find(identity);
    const auto terminated = terminated_drain_generations.find(identity);
    const bool has_unterminated_committed_drain =
        committed_drain != succeeded_drain_generations.end() &&
        (terminated == terminated_drain_generations.end() ||
         terminated->second < committed_drain->second);
    const auto actual = actual_states.find(identity);
    const bool registry_can_lag =
        actual == actual_states.end() ||
        actual->second == PlacementLifecycleState::DRAINING;
    if (!superseded_by_new_drain && !has_unterminated_committed_drain &&
        managed_identities.find(identity) == managed_identities.end() &&
        registry_can_lag &&
        now_ms - proof.second <= config.terminal_visibility_grace_ms) {
      ++managed_replicas;
      managed_identities.insert(identity);
    }
  }
  const uint32_t remaining_operation_capacity =
      config.max_operations_per_pool - static_cast<uint32_t>(operations.size());
  if (current_generation_terminal_failure) {
    return result(PlacementReconcileStatus::HOLD,
                  PlacementReconcileReason::TERMINAL_OPERATION,
                  desired.desired_replicas,
                  managed_replicas,
                  pending_operations);
  }
  if (!pending_drains.empty() && desired.desired_replicas > managed_replicas) {
    std::sort(pending_drains.begin(),
              pending_drains.end(),
              [](const PlacementOperationView* left,
                 const PlacementOperationView* right) {
                return std::tie(left->engine_uid, left->engine_incarnation) <
                       std::tie(right->engine_uid, right->engine_incarnation);
              });
    PlacementReconcileResult cancellation =
        result(PlacementReconcileStatus::OK,
               PlacementReconcileReason::CANCEL_DRAIN,
               desired.desired_replicas,
               managed_replicas,
               pending_operations);
    const uint32_t cancel_count =
        std::min({desired.desired_replicas - managed_replicas,
                  config.max_drain_per_cycle,
                  config.max_operations_per_cycle,
                  remaining_operation_capacity,
                  static_cast<uint32_t>(pending_drains.size())});
    for (uint32_t ordinal = 0; ordinal < cancel_count; ++ordinal) {
      const PlacementOperationView& drain = *pending_drains[ordinal];
      if (desired.generation <= drain.desired_generation) {
        continue;
      }
      const uint64_t operation_ordinal = cycle_ordinal(now_ms, ordinal);
      cancellation.intents.emplace_back(PlacementOperationIntent{
          .operation_id = make_placement_operation_id(
              leader,
              desired.generation,
              desired.pool,
              PlacementOperationAction::CANCEL_DRAIN,
              operation_ordinal,
              drain.engine_uid,
              drain.engine_incarnation),
          .action = PlacementOperationAction::CANCEL_DRAIN,
          .pool = desired.pool,
          .engine_uid = drain.engine_uid,
          .engine_incarnation = drain.engine_incarnation,
          .leader_incarnation = leader.incarnation,
          .leader_epoch = leader.epoch,
          .desired_generation = desired.generation,
          .ordinal = operation_ordinal,
      });
    }
    if (!cancellation.intents.empty()) {
      return cancellation;
    }
  }
  if (pending_operations > 0) {
    return result(PlacementReconcileStatus::HOLD,
                  PlacementReconcileReason::PENDING_OPERATION,
                  desired.desired_replicas,
                  managed_replicas,
                  pending_operations);
  }

  std::vector<EngineIdentity> drained_not_completed;
  for (const auto& [identity, drain_generation] : succeeded_drain_generations) {
    const auto terminated = terminated_drain_generations.find(identity);
    if (terminated == terminated_drain_generations.end() ||
        terminated->second < drain_generation) {
      drained_not_completed.push_back(identity);
    }
  }
  if (!drained_not_completed.empty() &&
      desired.desired_replicas <= managed_replicas) {
    PlacementReconcileResult termination =
        result(PlacementReconcileStatus::OK,
               PlacementReconcileReason::TERMINATE_DRAINED,
               desired.desired_replicas,
               managed_replicas,
               0);
    const uint32_t terminate_count =
        std::min({config.max_operations_per_cycle,
                  config.max_drain_per_cycle,
                  remaining_operation_capacity,
                  static_cast<uint32_t>(drained_not_completed.size())});
    if (terminate_count == 0) {
      termination.status = PlacementReconcileStatus::HOLD;
      termination.reason = PlacementReconcileReason::OPERATION_CAPACITY;
      return termination;
    }
    termination.intents.reserve(terminate_count);
    for (uint32_t ordinal = 0; ordinal < terminate_count; ++ordinal) {
      const EngineIdentity& identity = drained_not_completed[ordinal];
      const uint64_t operation_ordinal = cycle_ordinal(now_ms, ordinal);
      termination.intents.emplace_back(PlacementOperationIntent{
          .operation_id =
              make_placement_operation_id(leader,
                                          desired.generation,
                                          desired.pool,
                                          PlacementOperationAction::TERMINATE,
                                          operation_ordinal,
                                          identity.first,
                                          identity.second),
          .action = PlacementOperationAction::TERMINATE,
          .pool = desired.pool,
          .engine_uid = identity.first,
          .engine_incarnation = identity.second,
          .leader_incarnation = leader.incarnation,
          .leader_epoch = leader.epoch,
          .desired_generation = desired.generation,
          .ordinal = operation_ordinal,
      });
    }
    return termination;
  }

  PlacementReconcileResult output = result(PlacementReconcileStatus::OK,
                                           PlacementReconcileReason::CONVERGED,
                                           desired.desired_replicas,
                                           managed_replicas,
                                           0);
  const uint32_t cycle_capacity =
      std::min(config.max_operations_per_cycle, remaining_operation_capacity);
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
      const uint64_t operation_ordinal = cycle_ordinal(now_ms, ordinal);
      output.intents.emplace_back(PlacementOperationIntent{
          .operation_id =
              make_placement_operation_id(leader,
                                          desired.generation,
                                          desired.pool,
                                          PlacementOperationAction::CREATE,
                                          operation_ordinal,
                                          "",
                                          ""),
          .action = PlacementOperationAction::CREATE,
          .pool = desired.pool,
          .leader_incarnation = leader.incarnation,
          .leader_epoch = leader.epoch,
          .desired_generation = desired.generation,
          .ordinal = operation_ordinal,
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
        replica.active_transfers == 0 && replica.cache_value_known &&
        engine_operations.find(identity) == engine_operations.end()) {
      victims.push_back(&replica);
    }
  }
  std::sort(
      victims.begin(),
      victims.end(),
      [](const PlacementReplicaFact* left, const PlacementReplicaFact* right) {
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
  const uint32_t drain_count =
      std::min({excess,
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
    const uint64_t operation_ordinal = cycle_ordinal(now_ms, ordinal);
    output.intents.emplace_back(PlacementOperationIntent{
        .operation_id =
            make_placement_operation_id(leader,
                                        desired.generation,
                                        desired.pool,
                                        PlacementOperationAction::BEGIN_DRAIN,
                                        operation_ordinal,
                                        victim.engine_uid,
                                        victim.engine_incarnation),
        .action = PlacementOperationAction::BEGIN_DRAIN,
        .pool = desired.pool,
        .engine_uid = victim.engine_uid,
        .engine_incarnation = victim.engine_incarnation,
        .leader_incarnation = leader.incarnation,
        .leader_epoch = leader.epoch,
        .desired_generation = desired.generation,
        .ordinal = operation_ordinal,
    });
  }
  return output;
}

std::string make_placement_operation_id(const PlacementLeaderIdentity& leader,
                                        uint64_t desired_generation,
                                        const PlacementPoolKey& pool,
                                        PlacementOperationAction action,
                                        uint64_t ordinal,
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
  append_u64(leader.epoch, &preimage);
  append_u64(desired_generation, &preimage);
  append_component(placement_pool_key_suffix(pool), &preimage);
  append_u64(static_cast<uint8_t>(action), &preimage);
  append_u64(ordinal, &preimage);
  append_component(engine_uid, &preimage);
  append_component(engine_incarnation, &preimage);
  constexpr uint64_t kOperationIdSeed = 0x584c4c4d5633504cULL;
  const XXH128_hash_t digest =
      XXH3_128bits_withSeed(preimage.data(), preimage.size(), kOperationIdSeed);
  return "placement-v3:" + digest_hex(digest);
}

bool placement_operation_terminal(PlacementOperationStatus status) {
  return status == PlacementOperationStatus::SUCCEEDED ||
         status == PlacementOperationStatus::FAILED ||
         status == PlacementOperationStatus::FENCED ||
         status == PlacementOperationStatus::CANCELED;
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
    case PlacementReconcileReason::CANCEL_DRAIN:
      return "CANCEL_DRAIN";
    case PlacementReconcileReason::TERMINATE_DRAINED:
      return "TERMINATE_DRAINED";
    case PlacementReconcileReason::TERMINAL_OPERATION:
      return "TERMINAL_OPERATION";
  }
  return "UNKNOWN";
}

}  // namespace xllm_service::placement
