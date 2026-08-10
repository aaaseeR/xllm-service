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
#include <vector>

#include "placement/placement_types.h"

namespace xllm_service::placement {

enum class PlacementBudgetStatus : int8_t {
  OK = 0,
  OVERCOMMITTED = 1,
  INSUFFICIENT_PROTECTED_BUDGET = 2,
  INVALID_INPUT = 3,
};

enum class PlacementBudgetDecision : int8_t {
  APPROVED = 0,
  PARTIALLY_APPROVED = 1,
  BUDGET_BLOCKED = 2,
};

struct PlacementBudgetCandidate {
  PlacementCapacityProfile profile;
  PlacementRecommendation recommendation;
  uint32_t priority = 0;
  double slo_risk_score = 0.0;
};

struct PlacementBudgetAllocation {
  PlacementPoolKey pool;
  PlacementBudgetDecision decision = PlacementBudgetDecision::BUDGET_BLOCKED;
  uint32_t previous_desired_replicas = 0;
  uint32_t requested_desired_replicas = 0;
  uint32_t approved_desired_replicas = 0;
  uint64_t approved_devices = 0;
};

struct PlacementBudgetResult {
  PlacementBudgetStatus status = PlacementBudgetStatus::INVALID_INPUT;
  uint64_t max_devices = 0;
  uint64_t protected_devices = 0;
  uint64_t approved_devices = 0;
  std::vector<PlacementBudgetAllocation> allocations;
};

PlacementBudgetResult allocate_placement_budget(
    uint64_t max_devices,
    const std::vector<PlacementBudgetCandidate>& candidates);

}  // namespace xllm_service::placement
