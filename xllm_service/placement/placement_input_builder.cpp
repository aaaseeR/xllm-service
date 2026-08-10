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

#include "placement/placement_input_builder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <utility>

namespace xllm_service::placement {
namespace {

std::string pool_key(const PlacementPoolKey& pool) {
  std::string key = std::to_string(static_cast<int32_t>(pool.provider_id));
  key.push_back('\0');
  key.append(pool.model_revision);
  key.push_back('\0');
  key.append(std::to_string(static_cast<int32_t>(pool.role)));
  key.push_back('\0');
  key.append(pool.profile_digest);
  return key;
}

bool matches_pool(const xllm::proto::ProviderDescriptor& descriptor,
                  const PlacementPoolKey& pool) {
  return descriptor.identity().provider_id() == pool.provider_id &&
         descriptor.model().model_revision() == pool.model_revision &&
         descriptor.serving().role() == pool.role &&
         descriptor.profile_digest() == pool.profile_digest;
}

bool has_capability(const xllm::proto::ProviderDescriptor& descriptor,
                    xllm::proto::ProviderCapability capability) {
  return std::find(descriptor.capabilities().begin(),
                   descriptor.capabilities().end(),
                   capability) != descriptor.capabilities().end();
}

bool add_saturated(uint64_t value, uint64_t* target) {
  if (value > std::numeric_limits<uint64_t>::max() - *target) {
    *target = std::numeric_limits<uint64_t>::max();
    return false;
  }
  *target += value;
  return true;
}

bool valid_member_identity(
    const provider::EngineRegistryMemberSnapshot& member) {
  const auto& descriptor = member.descriptor;
  if (!xllm::proto::ProviderId_IsValid(descriptor.identity().provider_id()) ||
      descriptor.identity().provider_id() ==
          xllm::proto::PROVIDER_ID_UNSPECIFIED ||
      !valid_placement_identity(descriptor.identity().engine_uid()) ||
      !valid_placement_identity(descriptor.identity().incarnation_id()) ||
      !valid_placement_identity(descriptor.profile_digest()) ||
      !valid_placement_identity(descriptor.model().model_revision()) ||
      !xllm::proto::EngineRole_IsValid(descriptor.serving().role()) ||
      descriptor.serving().role() == xllm::proto::ENGINE_ROLE_UNSPECIFIED) {
    return false;
  }
  if (!member.state.has_value()) {
    return member.state_freshness == provider::EngineStateFreshness::MISSING;
  }
  const xllm::proto::EngineState& state = *member.state;
  return state.engine_uid() == descriptor.identity().engine_uid() &&
         state.incarnation_id() == descriptor.identity().incarnation_id() &&
         state.provider_id() == descriptor.identity().provider_id() &&
         state.profile_digest() == descriptor.profile_digest() &&
         state.model_revision() == descriptor.model().model_revision() &&
         xllm::proto::EngineLifecycle_IsValid(state.lifecycle()) &&
         state.lifecycle() != xllm::proto::ENGINE_LIFECYCLE_UNSPECIFIED &&
         member.state_freshness != provider::EngineStateFreshness::MISSING;
}

PlacementLifecycleState lifecycle_state(
    const provider::EngineRegistryMemberSnapshot& member) {
  if (!member.state.has_value()) {
    // Registry membership is creation proof. Count it while State Stream is
    // cold so the controller cannot emit duplicate CREATE operations.
    return PlacementLifecycleState::LOADING;
  }
  switch (member.state->lifecycle()) {
    case xllm::proto::ENGINE_LIFECYCLE_STARTING:
      return PlacementLifecycleState::WARMING;
    case xllm::proto::ENGINE_LIFECYCLE_READY:
      return member.schedulable ? PlacementLifecycleState::READY
                                : PlacementLifecycleState::FAILED;
    case xllm::proto::ENGINE_LIFECYCLE_DRAINING:
      return PlacementLifecycleState::DRAINING;
    case xllm::proto::ENGINE_LIFECYCLE_FENCED:
    case xllm::proto::ENGINE_LIFECYCLE_MEMBERSHIP_LOST:
    case xllm::proto::ENGINE_LIFECYCLE_UNSPECIFIED:
      return PlacementLifecycleState::FAILED;
    default:
      return PlacementLifecycleState::FAILED;
  }
}

bool populate_activity(const xllm::proto::EngineState& state,
                       uint64_t* active_reservations,
                       uint64_t* active_transfers) {
  bool exact = true;
  if (state.has_drain()) {
    const xllm::proto::ProviderDrainProof& drain = state.drain();
    exact &= add_saturated(drain.prefill_queue(), active_reservations);
    exact &= add_saturated(drain.active_reservations(), active_reservations);
    exact &= add_saturated(drain.decode_sequences(), active_reservations);
    exact &= add_saturated(drain.pending_output(), active_reservations);
    exact &= add_saturated(drain.pending_cleanup(), active_reservations);
    exact &= add_saturated(drain.active_transfers(), active_transfers);
    return exact;
  }
  // READY providers do not need to publish a drain proof. Running and waiting
  // work remains a conservative victim exclusion signal until BEGIN_DRAIN.
  for (const xllm::proto::PerDpEngineState& dp : state.per_dp()) {
    if (dp.has_running()) {
      exact &= add_saturated(dp.running(), active_reservations);
    }
    if (dp.has_waiting_capacity()) {
      exact &= add_saturated(dp.waiting_capacity(), active_reservations);
    }
    if (dp.has_waiting_deferred()) {
      exact &= add_saturated(dp.waiting_deferred(), active_reservations);
    }
  }
  return exact;
}

bool update_kv_pressure(const xllm::proto::EngineState& state,
                        double* kv_used_ratio) {
  for (const xllm::proto::PerDpEngineState& dp : state.per_dp()) {
    if (!dp.has_kv_used_ratio()) {
      continue;
    }
    if (!std::isfinite(dp.kv_used_ratio()) || dp.kv_used_ratio() < 0.0 ||
        dp.kv_used_ratio() > 1.0) {
      return false;
    }
    *kv_used_ratio = std::max(*kv_used_ratio, dp.kv_used_ratio());
  }
  return true;
}

}  // namespace

PlacementInputBuilder::PlacementInputBuilder(PlacementInputBuilderConfig config)
    : config_(std::move(config)),
      valid_(valid_placement_input_builder_config(config_)) {}

PlacementInputBuildStatus PlacementInputBuilder::build(
    const std::vector<PlacementPoolRuntimeSpec>& pools,
    const std::vector<provider::EngineRegistryMemberSnapshot>& members,
    PlacementObservationCollector* observations,
    uint64_t now_monotonic_ms,
    std::vector<PlacementPoolCycleInput>* inputs) const {
  if (inputs != nullptr) {
    inputs->clear();
  }
  if (!valid_ || observations == nullptr || !observations->valid() ||
      now_monotonic_ms == 0 || inputs == nullptr) {
    return PlacementInputBuildStatus::INVALID_INPUT;
  }
  if (pools.size() > config_.max_pools ||
      members.size() > config_.max_members) {
    return PlacementInputBuildStatus::CAPACITY_EXCEEDED;
  }

  std::set<std::string> pool_keys;
  std::set<std::pair<std::string, std::string>> member_identities;
  for (const PlacementPoolRuntimeSpec& pool : pools) {
    if (!valid_placement_capacity_profile(pool.profile) ||
        !valid_placement_identity(pool.config_digest) ||
        !std::isfinite(pool.slo_risk_score) || pool.slo_risk_score < 0.0 ||
        !pool_keys.insert(pool_key(pool.profile.pool)).second) {
      return PlacementInputBuildStatus::INVALID_INPUT;
    }
  }
  for (const provider::EngineRegistryMemberSnapshot& member : members) {
    if (!valid_member_identity(member) ||
        !member_identities
             .insert({member.descriptor.identity().engine_uid(),
                      member.descriptor.identity().incarnation_id()})
             .second) {
      return PlacementInputBuildStatus::INVALID_INPUT;
    }
  }

  inputs->reserve(pools.size());
  for (const PlacementPoolRuntimeSpec& pool : pools) {
    PlacementPoolCycleInput input{
        .profile = pool.profile,
        .priority = pool.priority,
        .slo_risk_score = pool.slo_risk_score,
        .config_digest = pool.config_digest,
    };
    PlacementObservationExternalInputs external = pool.external;
    for (const provider::EngineRegistryMemberSnapshot& member : members) {
      if (!matches_pool(member.descriptor, pool.profile.pool)) {
        continue;
      }
      PlacementReplicaFact replica{
          .pool = pool.profile.pool,
          .engine_uid = member.descriptor.identity().engine_uid(),
          .engine_incarnation = member.descriptor.identity().incarnation_id(),
          .state = lifecycle_state(member),
          .fresh =
              member.state.has_value() &&
              member.state_freshness == provider::EngineStateFreshness::FRESH &&
              member.heartbeat_fresh && member.schedulable,
          .drain_capable = has_capability(
              member.descriptor, xllm::proto::PROVIDER_CAPABILITY_DRAIN),
          .stable_since_ms = member.lifecycle_since_monotonic_ms == 0
                                 ? now_monotonic_ms
                                 : member.lifecycle_since_monotonic_ms,
          .observed_at_ms = now_monotonic_ms,
      };
      if (member.state.has_value()) {
        if (!populate_activity(*member.state,
                               &replica.active_reservations,
                               &replica.active_transfers) ||
            (replica.fresh &&
             !update_kv_pressure(*member.state, &external.kv_used_ratio))) {
          inputs->clear();
          return PlacementInputBuildStatus::CAPACITY_EXCEEDED;
        }
      }
      input.replicas.push_back(std::move(replica));
    }
    const PlacementObservationStatus observation_status =
        observations->snapshot(pool.profile.pool.model_revision,
                               now_monotonic_ms,
                               external,
                               &input.observation);
    if (observation_status != PlacementObservationStatus::OK) {
      inputs->clear();
      return observation_status == PlacementObservationStatus::CAPACITY_EXCEEDED
                 ? PlacementInputBuildStatus::CAPACITY_EXCEEDED
                 : PlacementInputBuildStatus::OBSERVATION_UNAVAILABLE;
    }
    inputs->push_back(std::move(input));
  }
  return PlacementInputBuildStatus::OK;
}

bool PlacementInputBuilder::valid() const { return valid_; }

bool valid_placement_input_builder_config(
    const PlacementInputBuilderConfig& config) {
  return config.max_pools > 0 && config.max_members > 0;
}

const char* placement_input_build_status_name(
    PlacementInputBuildStatus status) {
  switch (status) {
    case PlacementInputBuildStatus::OK:
      return "OK";
    case PlacementInputBuildStatus::INVALID_INPUT:
      return "INVALID_INPUT";
    case PlacementInputBuildStatus::OBSERVATION_UNAVAILABLE:
      return "OBSERVATION_UNAVAILABLE";
    case PlacementInputBuildStatus::CAPACITY_EXCEEDED:
      return "CAPACITY_EXCEEDED";
  }
  return "UNKNOWN";
}

}  // namespace xllm_service::placement
