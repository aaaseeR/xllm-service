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

#include "placement/placement_controller.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

namespace xllm_service::placement {
namespace {

bool valid_mode(PlacementMode mode) {
  switch (mode) {
    case PlacementMode::DISABLED:
    case PlacementMode::SHADOW:
    case PlacementMode::ENFORCED_CREATE_ONLY:
    case PlacementMode::ENFORCED:
      return true;
  }
  return false;
}

bool same_leader(const PlacementLeaderIdentity& left,
                 const PlacementLeaderIdentity& right) {
  return left.address == right.address &&
         left.incarnation == right.incarnation && left.epoch == right.epoch;
}

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

bool managed_replica(PlacementLifecycleState state) {
  return state == PlacementLifecycleState::LOADING ||
         state == PlacementLifecycleState::WARMING ||
         state == PlacementLifecycleState::READY;
}

bool occupies_device(PlacementLifecycleState state) {
  // Admission and serving state are not allocation state. DRAINING,
  // UNLOADING, and FAILED replicas continue to consume their devices until
  // the deployment system proves termination and Registry reports ABSENT.
  return state != PlacementLifecycleState::ABSENT;
}

bool add_occupied_devices(uint64_t replicas,
                          uint64_t devices_per_replica,
                          uint64_t* occupied_devices) {
  if (occupied_devices == nullptr || devices_per_replica == 0 ||
      replicas > std::numeric_limits<uint64_t>::max() / devices_per_replica) {
    return false;
  }
  const uint64_t devices = replicas * devices_per_replica;
  if (*occupied_devices > std::numeric_limits<uint64_t>::max() - devices) {
    return false;
  }
  *occupied_devices += devices;
  return true;
}

void update_actual_state(const PlacementPoolCycleInput& input,
                         const std::vector<PlacementOperationView>& operations,
                         PlacementPoolState* state) {
  state->ready_replicas = 0;
  state->loading_replicas = 0;
  state->warming_replicas = 0;
  state->draining_replicas = 0;
  state->pending_operations = 0;
  for (const PlacementReplicaFact& replica : input.replicas) {
    switch (replica.state) {
      case PlacementLifecycleState::LOADING:
        ++state->loading_replicas;
        break;
      case PlacementLifecycleState::WARMING:
        ++state->warming_replicas;
        break;
      case PlacementLifecycleState::READY:
        ++state->ready_replicas;
        break;
      case PlacementLifecycleState::DRAINING:
        ++state->draining_replicas;
        break;
      case PlacementLifecycleState::ABSENT:
      case PlacementLifecycleState::UNLOADING:
      case PlacementLifecycleState::FAILED:
        break;
    }
  }
  for (const PlacementOperationView& operation : operations) {
    if (placement_pool_keys_equal(operation.pool, input.profile.pool) &&
        !placement_operation_terminal(operation.status)) {
      ++state->pending_operations;
    }
  }
}

uint32_t initial_desired(const PlacementPoolCycleInput& input) {
  uint32_t managed = 0;
  for (const PlacementReplicaFact& replica : input.replicas) {
    if (managed_replica(replica.state)) {
      ++managed;
    }
  }
  return std::clamp(
      managed, input.profile.min_replicas, input.profile.max_replicas);
}

PlacementControllerStatus controller_status(PlacementExecutorStatus status) {
  switch (status) {
    case PlacementExecutorStatus::OK:
      return PlacementControllerStatus::OK;
    case PlacementExecutorStatus::CAPACITY_EXCEEDED:
      return PlacementControllerStatus::CAPACITY_EXCEEDED;
    case PlacementExecutorStatus::INVALID_INPUT:
    case PlacementExecutorStatus::CLOCK_REGRESSION:
      return PlacementControllerStatus::INVALID_INPUT;
    case PlacementExecutorStatus::PERSISTENCE_ERROR:
      return PlacementControllerStatus::PERSISTENCE_ERROR;
    case PlacementExecutorStatus::CORRUPT_SNAPSHOT:
      return PlacementControllerStatus::CORRUPT_SNAPSHOT;
  }
  return PlacementControllerStatus::INVALID_INPUT;
}

}  // namespace

PlacementController::PlacementController(PlacementControllerConfig config,
                                         PlacementDesiredStore* desired_store,
                                         PlacementOperationExecutor* executor)
    : config_(std::move(config)),
      desired_store_(desired_store),
      executor_(executor) {}

PlacementControllerStatus PlacementController::recover(
    const PlacementLeaderIdentity& leader,
    uint64_t now_monotonic_ms) {
  recovered_ = false;
  pools_.clear();
  if (!valid_placement_controller_config(config_) ||
      desired_store_ == nullptr || executor_ == nullptr ||
      !valid_placement_leader_identity(leader) || now_monotonic_ms == 0) {
    return PlacementControllerStatus::INVALID_INPUT;
  }
  std::vector<PlacementDesiredSnapshot> desired;
  const PlacementStoreStatus desired_status = desired_store_->load_snapshot(
      config_.max_pools, config_.max_desired_snapshot_bytes, &desired);
  if (desired_status == PlacementStoreStatus::CAPACITY_EXCEEDED) {
    return PlacementControllerStatus::CAPACITY_EXCEEDED;
  }
  if (desired_status == PlacementStoreStatus::CORRUPT) {
    return PlacementControllerStatus::CORRUPT_SNAPSHOT;
  }
  if (desired_status != PlacementStoreStatus::OK) {
    return PlacementControllerStatus::PERSISTENCE_ERROR;
  }
  for (PlacementDesiredSnapshot& snapshot : desired) {
    const std::string key = pool_key(snapshot.desired.pool);
    RuntimePool runtime;
    runtime.state.desired_replicas = snapshot.desired.desired_replicas;
    runtime.state.last_observation_generation =
        snapshot.desired.observation_generation;
    runtime.persisted = std::move(snapshot);
    runtime.has_persisted = true;
    if (!pools_.emplace(key, std::move(runtime)).second) {
      pools_.clear();
      return PlacementControllerStatus::PERSISTENCE_ERROR;
    }
  }
  const PlacementExecutorStatus operation_status =
      executor_->recover_for_leader(
          leader, now_monotonic_ms, config_.max_operation_snapshot_bytes);
  if (operation_status != PlacementExecutorStatus::OK) {
    pools_.clear();
    return controller_status(operation_status);
  }
  leader_ = leader;
  recovered_ = true;
  return PlacementControllerStatus::OK;
}

PlacementControllerResult PlacementController::run_cycle(
    const PlacementLeaderIdentity& leader,
    const std::vector<PlacementPoolCycleInput>& inputs,
    uint64_t now_monotonic_ms,
    uint64_t now_unix_ms) {
  PlacementControllerResult output{
      .status = PlacementControllerStatus::OK,
      .mode = config_.mode,
  };
  if (!recovered_) {
    output.status = PlacementControllerStatus::NOT_RECOVERED;
    return output;
  }
  if (!same_leader(leader_, leader) || now_monotonic_ms == 0 ||
      now_unix_ms == 0 || inputs.size() > config_.max_pools) {
    output.status = PlacementControllerStatus::INVALID_INPUT;
    return output;
  }
  if (config_.mode == PlacementMode::DISABLED) {
    output.status = PlacementControllerStatus::HOLD;
    return output;
  }

  const std::vector<PlacementOperationView> operation_views =
      executor_->operation_views();
  std::set<std::string> input_keys;
  std::vector<PlacementBudgetCandidate> candidates;
  candidates.reserve(inputs.size());
  output.pools.reserve(inputs.size());
  uint64_t occupied_devices = 0;
  for (const PlacementPoolCycleInput& input : inputs) {
    const std::string key = pool_key(input.profile.pool);
    if (!valid_placement_capacity_profile(input.profile) ||
        !valid_placement_observation(input.observation) ||
        !std::isfinite(input.slo_risk_score) || input.slo_risk_score < 0.0 ||
        !valid_placement_identity(input.config_digest) ||
        !input_keys.insert(key).second) {
      output.status = PlacementControllerStatus::INVALID_INPUT;
      return output;
    }
    std::set<std::pair<std::string, std::string>> observed_identities;
    uint64_t occupied_replicas = 0;
    for (const PlacementReplicaFact& replica : input.replicas) {
      observed_identities.emplace(replica.engine_uid,
                                  replica.engine_incarnation);
      occupied_replicas += occupies_device(replica.state) ? 1 : 0;
    }
    for (const PlacementOperationView& operation : operation_views) {
      if (!placement_pool_keys_equal(operation.pool, input.profile.pool) ||
          operation.action != PlacementOperationAction::CREATE ||
          placement_operation_terminal(operation.status)) {
        continue;
      }
      const bool observed =
          !operation.engine_uid.empty() &&
          observed_identities.find(
              {operation.engine_uid, operation.engine_incarnation}) !=
              observed_identities.end();
      occupied_replicas += observed ? 0 : 1;
    }
    if (!add_occupied_devices(occupied_replicas,
                              input.profile.devices_per_replica,
                              &occupied_devices)) {
      output.status = PlacementControllerStatus::INVALID_INPUT;
      return output;
    }
    const auto [runtime_iterator, inserted] =
        pools_.try_emplace(key, RuntimePool{});
    RuntimePool& runtime = runtime_iterator->second;
    if (inserted) {
      runtime.state.desired_replicas = initial_desired(input);
    }
    update_actual_state(input, operation_views, &runtime.state);
    PlacementRecommendation recommendation =
        plan_placement_pool(config_.planner,
                            input.profile,
                            input.observation,
                            runtime.state,
                            now_monotonic_ms);
    if (recommendation.status == PlacementPlanStatus::INVALID_INPUT) {
      output.status = PlacementControllerStatus::INVALID_INPUT;
      return output;
    }
    candidates.emplace_back(PlacementBudgetCandidate{
        .profile = input.profile,
        .recommendation = recommendation,
        .priority = input.priority,
        .slo_risk_score = input.slo_risk_score,
    });
    output.pools.emplace_back(PlacementPoolCycleReport{
        .pool = input.profile.pool,
        .recommendation = std::move(recommendation),
    });
  }

  if (!candidates.empty()) {
    const PlacementBudgetResult budget =
        allocate_placement_budget(config_.max_devices, candidates);
    if (budget.status != PlacementBudgetStatus::OK &&
        budget.status != PlacementBudgetStatus::OVERCOMMITTED) {
      output.status = PlacementControllerStatus::BUDGET_ERROR;
      return output;
    }
    if (budget.allocations.size() != inputs.size()) {
      output.status = PlacementControllerStatus::BUDGET_ERROR;
      return output;
    }
    for (size_t index = 0; index < inputs.size(); ++index) {
      output.pools[index].allocation = budget.allocations[index];
    }
    if (budget.status == PlacementBudgetStatus::OVERCOMMITTED) {
      output.status = PlacementControllerStatus::HOLD;
    }
  }

  uint32_t remaining_new_operations = config_.max_new_operations_per_cycle;
  for (size_t index = 0; index < inputs.size(); ++index) {
    const PlacementPoolCycleInput& input = inputs[index];
    PlacementPoolCycleReport& report = output.pools[index];
    RuntimePool& runtime = pools_.at(pool_key(input.profile.pool));
    uint32_t approved_desired = report.allocation.approved_desired_replicas;
    if (config_.mode == PlacementMode::ENFORCED_CREATE_ONLY &&
        approved_desired < runtime.state.desired_replicas) {
      approved_desired = runtime.state.desired_replicas;
    }
    report.recommendation.next_state.desired_replicas = approved_desired;
    runtime.state = report.recommendation.next_state;

    if (config_.mode == PlacementMode::SHADOW) {
      continue;
    }

    bool wrote = false;
    const PlacementControllerStatus desired_status =
        persist_desired(leader,
                        input,
                        report.recommendation.reason,
                        approved_desired,
                        now_unix_ms,
                        &runtime,
                        &wrote);
    if (desired_status != PlacementControllerStatus::OK) {
      output.status = desired_status;
      continue;
    }
    report.desired_persisted = wrote;
    report.desired_generation = runtime.persisted.desired.generation;
    output.desired_writes += wrote ? 1 : 0;

    std::vector<PlacementOperationView> pool_operations;
    for (const PlacementOperationView& operation :
         executor_->operation_views()) {
      if (placement_pool_keys_equal(operation.pool, input.profile.pool)) {
        pool_operations.push_back(operation);
      }
    }
    report.reconcile = reconcile_placement_pool(config_.reconcile,
                                                runtime.persisted.desired,
                                                leader,
                                                input.replicas,
                                                pool_operations,
                                                now_monotonic_ms);
    if (report.reconcile.status == PlacementReconcileStatus::INVALID_INPUT) {
      output.status = PlacementControllerStatus::INVALID_INPUT;
      continue;
    }
    if (report.reconcile.status == PlacementReconcileStatus::HOLD) {
      if (output.status == PlacementControllerStatus::OK) {
        output.status = PlacementControllerStatus::HOLD;
      }
      continue;
    }
    if (report.reconcile.intents.size() > remaining_new_operations) {
      report.reconcile.intents.resize(remaining_new_operations);
      output.status = PlacementControllerStatus::HOLD;
    }
    std::vector<PlacementOperationIntent> capacity_allowed;
    capacity_allowed.reserve(report.reconcile.intents.size());
    for (PlacementOperationIntent& intent : report.reconcile.intents) {
      if (intent.action == PlacementOperationAction::CREATE &&
          (occupied_devices >= config_.max_devices ||
           input.profile.devices_per_replica >
               config_.max_devices - occupied_devices)) {
        if (output.status == PlacementControllerStatus::OK) {
          output.status = PlacementControllerStatus::HOLD;
        }
        continue;
      }
      if (intent.action == PlacementOperationAction::CREATE) {
        occupied_devices += input.profile.devices_per_replica;
      }
      capacity_allowed.emplace_back(std::move(intent));
    }
    report.reconcile.intents = std::move(capacity_allowed);
    if (report.reconcile.intents.empty()) {
      continue;
    }
    const PlacementExecutorResult added =
        executor_->add_intents(report.reconcile.intents, now_monotonic_ms);
    if (added.status != PlacementExecutorStatus::OK) {
      output.status = controller_status(added.status);
      continue;
    }
    report.intents_added = added.added;
    output.intents_added += added.added;
    remaining_new_operations -=
        std::min(remaining_new_operations, static_cast<uint32_t>(added.added));
  }

  output.actuator = executor_->drive(now_monotonic_ms,
                                     config_.max_actuator_actions_per_cycle);
  if (output.actuator.status != PlacementExecutorStatus::OK) {
    output.status = controller_status(output.actuator.status);
  }
  return output;
}

PlacementControllerStatus PlacementController::persist_desired(
    const PlacementLeaderIdentity& leader,
    const PlacementPoolCycleInput& input,
    PlacementReason reason,
    uint32_t desired_replicas,
    uint64_t now_unix_ms,
    RuntimePool* runtime,
    bool* wrote) {
  if (runtime == nullptr || wrote == nullptr) {
    return PlacementControllerStatus::INVALID_INPUT;
  }
  *wrote = false;
  PlacementDesiredSnapshot current;
  const PlacementStoreStatus read_status =
      desired_store_->read(input.profile.pool, &current);
  if (read_status == PlacementStoreStatus::OK) {
    const bool previously_known = runtime->has_persisted;
    if (!previously_known ||
        current.mod_revision != runtime->persisted.mod_revision) {
      runtime->persisted = std::move(current);
      runtime->has_persisted = true;
      runtime->state.desired_replicas =
          runtime->persisted.desired.desired_replicas;
      runtime->state.last_observation_generation =
          runtime->persisted.desired.observation_generation;
      return PlacementControllerStatus::HOLD;
    }
    runtime->persisted = std::move(current);
    runtime->has_persisted = true;
  } else if (read_status == PlacementStoreStatus::NOT_FOUND) {
    if (runtime->has_persisted) {
      return PlacementControllerStatus::PERSISTENCE_ERROR;
    }
  } else if (read_status == PlacementStoreStatus::CAPACITY_EXCEEDED) {
    return PlacementControllerStatus::CAPACITY_EXCEEDED;
  } else {
    return PlacementControllerStatus::PERSISTENCE_ERROR;
  }

  if (runtime->has_persisted &&
      same_leader(runtime->persisted.desired.leader, leader) &&
      runtime->persisted.desired.desired_replicas == desired_replicas) {
    runtime->state.desired_replicas = desired_replicas;
    return PlacementControllerStatus::OK;
  }
  const uint64_t previous_generation =
      runtime->has_persisted ? runtime->persisted.desired.generation : 0;
  if (previous_generation == std::numeric_limits<uint64_t>::max()) {
    return PlacementControllerStatus::CAPACITY_EXCEEDED;
  }
  PlacementDesiredState desired{
      .leader = leader,
      .generation = previous_generation + 1,
      .pool = input.profile.pool,
      .desired_replicas = desired_replicas,
      .reason = reason,
      .observation_generation = input.observation.generation,
      .created_at_unix_ms = now_unix_ms,
      .config_digest = input.config_digest,
  };
  const int64_t expected_revision =
      runtime->has_persisted ? runtime->persisted.mod_revision : 0;
  const PlacementStoreStatus write_status =
      desired_store_->compare_and_set(desired, expected_revision, leader);
  if (write_status == PlacementStoreStatus::CAPACITY_EXCEEDED) {
    return PlacementControllerStatus::CAPACITY_EXCEEDED;
  }
  if (write_status != PlacementStoreStatus::OK) {
    return PlacementControllerStatus::PERSISTENCE_ERROR;
  }
  PlacementDesiredSnapshot stored;
  if (desired_store_->read(input.profile.pool, &stored) !=
          PlacementStoreStatus::OK ||
      stored.desired.generation != desired.generation ||
      !same_leader(stored.desired.leader, leader) ||
      stored.desired.desired_replicas != desired_replicas) {
    return PlacementControllerStatus::PERSISTENCE_ERROR;
  }
  runtime->persisted = std::move(stored);
  runtime->has_persisted = true;
  runtime->state.desired_replicas = desired_replicas;
  *wrote = true;
  return PlacementControllerStatus::OK;
}

bool PlacementController::set_mode(PlacementMode mode) {
  if (!valid_mode(mode)) {
    return false;
  }
  config_.mode = mode;
  return true;
}

PlacementMode PlacementController::mode() const { return config_.mode; }

bool PlacementController::recovered() const { return recovered_; }

bool valid_placement_controller_config(
    const PlacementControllerConfig& config) {
  return valid_mode(config.mode) && config.max_pools > 0 &&
         config.max_desired_snapshot_bytes > 0 &&
         config.max_operation_snapshot_bytes > 0 && config.max_devices > 0 &&
         config.max_new_operations_per_cycle > 0 &&
         config.max_actuator_actions_per_cycle > 0 &&
         valid_placement_planner_config(config.planner) &&
         valid_placement_reconcile_config(config.reconcile);
}

const char* placement_mode_name(PlacementMode mode) {
  switch (mode) {
    case PlacementMode::DISABLED:
      return "DISABLED";
    case PlacementMode::SHADOW:
      return "SHADOW";
    case PlacementMode::ENFORCED_CREATE_ONLY:
      return "ENFORCED_CREATE_ONLY";
    case PlacementMode::ENFORCED:
      return "ENFORCED";
  }
  return "UNKNOWN";
}

const char* placement_controller_status_name(PlacementControllerStatus status) {
  switch (status) {
    case PlacementControllerStatus::OK:
      return "OK";
    case PlacementControllerStatus::HOLD:
      return "HOLD";
    case PlacementControllerStatus::INVALID_INPUT:
      return "INVALID_INPUT";
    case PlacementControllerStatus::NOT_RECOVERED:
      return "NOT_RECOVERED";
    case PlacementControllerStatus::PERSISTENCE_ERROR:
      return "PERSISTENCE_ERROR";
    case PlacementControllerStatus::CAPACITY_EXCEEDED:
      return "CAPACITY_EXCEEDED";
    case PlacementControllerStatus::BUDGET_ERROR:
      return "BUDGET_ERROR";
    case PlacementControllerStatus::CORRUPT_SNAPSHOT:
      return "CORRUPT_SNAPSHOT";
  }
  return "UNKNOWN";
}

}  // namespace xllm_service::placement
