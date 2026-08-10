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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "placement/placement_desired_store.h"
#include "placement/placement_lifecycle.h"

namespace xllm_service::placement {

enum class PlacementOperationAction : int8_t {
  CREATE = 0,
  BEGIN_DRAIN = 1,
  CANCEL_DRAIN = 2,
  TERMINATE = 3,
};

enum class PlacementOperationStatus : int8_t {
  PLANNED = 0,
  SUBMITTED = 1,
  IN_PROGRESS = 2,
  UNKNOWN = 3,
  SUCCEEDED = 4,
  FAILED = 5,
  FENCED = 6,
  CANCELED = 7,
};

enum class PlacementReconcileStatus : int8_t {
  OK = 0,
  HOLD = 1,
  INVALID_INPUT = 2,
};

enum class PlacementReconcileReason : int8_t {
  CONVERGED = 0,
  SCALE_UP = 1,
  SCALE_DOWN = 2,
  PENDING_OPERATION = 3,
  OPERATION_CAPACITY = 4,
  NO_SAFE_VICTIM = 5,
  INVALID_CONFIG = 6,
  INVALID_DESIRED = 7,
  INVALID_ACTUAL = 8,
  INVALID_OPERATION = 9,
  CLOCK_REGRESSION = 10,
  CANCEL_DRAIN = 11,
  TERMINATE_DRAINED = 12,
  TERMINAL_OPERATION = 13,
};

struct PlacementReconcileConfig {
  uint32_t max_operations_per_cycle = 0;
  uint32_t max_operations_per_pool = 0;
  uint32_t max_create_per_cycle = 0;
  uint32_t max_drain_per_cycle = 0;
  uint64_t terminal_visibility_grace_ms = 0;
};

struct PlacementReplicaFact {
  PlacementPoolKey pool;
  std::string engine_uid;
  std::string engine_incarnation;
  PlacementLifecycleState state = PlacementLifecycleState::ABSENT;
  bool fresh = false;
  bool drain_capable = false;
  uint64_t active_reservations = 0;
  uint64_t active_transfers = 0;
  double cache_value = 0.0;
  bool cache_value_known = true;
  uint64_t stable_since_ms = 0;
  uint64_t observed_at_ms = 0;
};

struct PlacementOperationView {
  std::string operation_id;
  PlacementOperationAction action = PlacementOperationAction::CREATE;
  PlacementOperationStatus status = PlacementOperationStatus::PLANNED;
  PlacementPoolKey pool;
  std::string engine_uid;
  std::string engine_incarnation;
  std::string leader_incarnation;
  uint64_t leader_epoch = 0;
  uint64_t desired_generation = 0;
  uint64_t updated_at_ms = 0;
  bool visibility_grace_eligible = true;
};

struct PlacementOperationIntent {
  std::string operation_id;
  PlacementOperationAction action = PlacementOperationAction::CREATE;
  PlacementPoolKey pool;
  std::string engine_uid;
  std::string engine_incarnation;
  std::string leader_incarnation;
  uint64_t leader_epoch = 0;
  uint64_t desired_generation = 0;
  uint64_t ordinal = 0;
};

struct PlacementReconcileResult {
  PlacementReconcileStatus status = PlacementReconcileStatus::INVALID_INPUT;
  PlacementReconcileReason reason = PlacementReconcileReason::INVALID_CONFIG;
  uint32_t desired_replicas = 0;
  uint32_t managed_replicas = 0;
  uint32_t pending_operations = 0;
  std::vector<PlacementOperationIntent> intents;
};

PlacementReconcileResult reconcile_placement_pool(
    const PlacementReconcileConfig& config,
    const PlacementDesiredState& desired,
    const PlacementLeaderIdentity& leader,
    const std::vector<PlacementReplicaFact>& replicas,
    const std::vector<PlacementOperationView>& operations,
    uint64_t now_ms);

bool valid_placement_reconcile_config(const PlacementReconcileConfig& config);

std::string make_placement_operation_id(const PlacementLeaderIdentity& leader,
                                        uint64_t desired_generation,
                                        const PlacementPoolKey& pool,
                                        PlacementOperationAction action,
                                        uint64_t ordinal,
                                        const std::string& engine_uid,
                                        const std::string& engine_incarnation);

bool placement_operation_terminal(PlacementOperationStatus status);

const char* placement_operation_action_name(PlacementOperationAction action);

const char* placement_operation_status_name(PlacementOperationStatus status);

const char* placement_reconcile_reason_name(PlacementReconcileReason reason);

}  // namespace xllm_service::placement
