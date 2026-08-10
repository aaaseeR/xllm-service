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

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "placement/placement_actuator.h"
#include "placement/placement_budget_allocator.h"
#include "placement/placement_desired_store.h"
#include "placement/placement_planner.h"

namespace xllm_service::placement {

enum class PlacementMode : int8_t {
  DISABLED = 0,
  SHADOW = 1,
  ENFORCED_CREATE_ONLY = 2,
  ENFORCED = 3,
};

enum class PlacementControllerStatus : int8_t {
  OK = 0,
  HOLD = 1,
  INVALID_INPUT = 2,
  NOT_RECOVERED = 3,
  PERSISTENCE_ERROR = 4,
  CAPACITY_EXCEEDED = 5,
  BUDGET_ERROR = 6,
  CORRUPT_SNAPSHOT = 7,
};

struct PlacementControllerConfig {
  PlacementMode mode = PlacementMode::DISABLED;
  size_t max_pools = 0;
  size_t max_desired_snapshot_bytes = 0;
  size_t max_operation_snapshot_bytes = 0;
  uint64_t max_devices = 0;
  uint32_t max_new_operations_per_cycle = 0;
  uint32_t max_actuator_actions_per_cycle = 0;
  PlacementPlannerConfig planner;
  PlacementReconcileConfig reconcile;
};

struct PlacementPoolCycleInput {
  PlacementCapacityProfile profile;
  PlacementObservation observation;
  std::vector<PlacementReplicaFact> replicas;
  uint32_t priority = 0;
  double slo_risk_score = 0.0;
  std::string config_digest;
};

struct PlacementPoolCycleReport {
  PlacementPoolKey pool;
  PlacementRecommendation recommendation;
  PlacementBudgetAllocation allocation;
  PlacementReconcileResult reconcile;
  uint64_t desired_generation = 0;
  bool desired_persisted = false;
  uint32_t intents_added = 0;
};

struct PlacementControllerResult {
  PlacementControllerStatus status = PlacementControllerStatus::INVALID_INPUT;
  PlacementMode mode = PlacementMode::DISABLED;
  uint32_t desired_writes = 0;
  uint32_t intents_added = 0;
  PlacementExecutorResult actuator;
  std::vector<PlacementPoolCycleReport> pools;
};

// Single-threaded V3 slow-loop coordinator. Callers provide immutable
// observations/Registry facts; this class owns only bounded control state and
// never participates in the request Router critical path.
class PlacementController final {
 public:
  PlacementController(PlacementControllerConfig config,
                      PlacementDesiredStore* desired_store,
                      PlacementOperationExecutor* executor);

  PlacementControllerStatus recover(const PlacementLeaderIdentity& leader,
                                    uint64_t now_monotonic_ms);

  PlacementControllerResult run_cycle(
      const PlacementLeaderIdentity& leader,
      const std::vector<PlacementPoolCycleInput>& inputs,
      uint64_t now_monotonic_ms,
      uint64_t now_unix_ms);

  bool set_mode(PlacementMode mode);
  PlacementMode mode() const;
  bool recovered() const;

 private:
  struct RuntimePool {
    PlacementPoolState state;
    PlacementDesiredSnapshot persisted;
    bool has_persisted = false;
  };

  PlacementControllerStatus persist_desired(
      const PlacementLeaderIdentity& leader,
      const PlacementPoolCycleInput& input,
      PlacementReason reason,
      uint32_t desired_replicas,
      uint64_t now_unix_ms,
      RuntimePool* runtime,
      bool* wrote);

  PlacementControllerConfig config_;
  PlacementDesiredStore* desired_store_ = nullptr;
  PlacementOperationExecutor* executor_ = nullptr;
  PlacementLeaderIdentity leader_;
  bool recovered_ = false;
  std::map<std::string, RuntimePool> pools_;
};

bool valid_placement_controller_config(const PlacementControllerConfig& config);

const char* placement_mode_name(PlacementMode mode);
const char* placement_controller_status_name(PlacementControllerStatus status);

}  // namespace xllm_service::placement
