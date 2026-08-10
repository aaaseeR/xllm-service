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

#include "placement/placement_budget_allocator.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

namespace xllm_service::placement {
namespace {

PlacementCapacityProfile profile(const std::string& model,
                                 uint32_t devices_per_replica = 2) {
  return PlacementCapacityProfile{
      .pool =
          PlacementPoolKey{
              .provider_id = xllm::proto::PROVIDER_ID_XLLM_NATIVE,
              .model_revision = model,
              .role = xllm::proto::ENGINE_ROLE_PREFILL,
              .profile_digest = "profile-a",
          },
      .devices_per_replica = devices_per_replica,
      .instance_cost_per_hour = 1.0,
      .load_warmup_p99_ms = 1000,
      .prefill_tokens_per_second_under_slo = 100.0,
      .target_utilization = 0.8,
      .min_replicas = 1,
      .max_replicas = 10,
      .failure_headroom_replicas = 1,
      .ttft_slo_ms = 100.0,
      .tpot_slo_ms = 10.0,
  };
}

PlacementRecommendation recommendation(uint32_t previous,
                                       uint32_t desired,
                                       uint32_t safe_required) {
  const PlacementAction action =
      desired > previous ? PlacementAction::SCALE_UP
                         : (desired < previous ? PlacementAction::SCALE_DOWN
                                               : PlacementAction::NONE);
  return PlacementRecommendation{
      .status = action == PlacementAction::NONE ? PlacementPlanStatus::HOLD
                                                : PlacementPlanStatus::OK,
      .action = action,
      .reason = PlacementReason::FORECAST_CAPACITY,
      .previous_desired_replicas = previous,
      .desired_replicas = desired,
      .safe_required_replicas = safe_required,
  };
}

PlacementBudgetCandidate candidate(const std::string& model,
                                   uint32_t previous,
                                   uint32_t desired,
                                   uint32_t safe_required,
                                   uint32_t priority,
                                   double risk) {
  return PlacementBudgetCandidate{
      .profile = profile(model),
      .recommendation = recommendation(previous, desired, safe_required),
      .priority = priority,
      .slo_risk_score = risk,
  };
}

const PlacementBudgetAllocation& find_allocation(
    const PlacementBudgetResult& result,
    const std::string& model) {
  const auto iterator =
      std::find_if(result.allocations.begin(),
                   result.allocations.end(),
                   [&model](const PlacementBudgetAllocation& allocation) {
                     return allocation.pool.model_revision == model;
                   });
  EXPECT_NE(iterator, result.allocations.end());
  return *iterator;
}

TEST(PlacementBudgetAllocatorTest, ApprovesWithinBudget) {
  const std::vector<PlacementBudgetCandidate> candidates = {
      candidate("model-a", 2, 4, 3, 1, 0.5),
      candidate("model-b", 2, 3, 3, 1, 0.5),
  };
  const PlacementBudgetResult result =
      allocate_placement_budget(/*max_devices=*/14, candidates);
  EXPECT_EQ(result.status, PlacementBudgetStatus::OK);
  EXPECT_EQ(result.approved_devices, 14u);
  EXPECT_EQ(find_allocation(result, "model-a").approved_desired_replicas, 4u);
  EXPECT_EQ(find_allocation(result, "model-b").approved_desired_replicas, 3u);
}

TEST(PlacementBudgetAllocatorTest, ProtectsRequiredBeforeDiscretionaryGrowth) {
  const std::vector<PlacementBudgetCandidate> candidates = {
      candidate("model-a", 2, 5, 3, 100, 1.0),
      candidate("model-b", 2, 4, 4, 1, 0.1),
  };
  const PlacementBudgetResult result =
      allocate_placement_budget(/*max_devices=*/14, candidates);
  EXPECT_EQ(result.status, PlacementBudgetStatus::OK);
  EXPECT_EQ(find_allocation(result, "model-a").approved_desired_replicas, 3u);
  EXPECT_EQ(find_allocation(result, "model-b").approved_desired_replicas, 4u);
}

TEST(PlacementBudgetAllocatorTest, RanksDiscretionaryGrowthDeterministically) {
  const std::vector<PlacementBudgetCandidate> candidates = {
      candidate("model-z", 2, 3, 2, 1, 0.5),
      candidate("model-a", 2, 3, 2, 1, 0.5),
  };
  const PlacementBudgetResult result =
      allocate_placement_budget(/*max_devices=*/10, candidates);
  EXPECT_EQ(result.status, PlacementBudgetStatus::OK);
  EXPECT_EQ(find_allocation(result, "model-a").approved_desired_replicas, 3u);
  EXPECT_EQ(find_allocation(result, "model-z").approved_desired_replicas, 2u);
}

TEST(PlacementBudgetAllocatorTest, ScaleDownFreesBudgetInSameCycle) {
  const std::vector<PlacementBudgetCandidate> candidates = {
      candidate("model-a", 4, 3, 2, 1, 0.0),
      candidate("model-b", 2, 3, 3, 1, 1.0),
  };
  const PlacementBudgetResult result =
      allocate_placement_budget(/*max_devices=*/12, candidates);
  EXPECT_EQ(result.status, PlacementBudgetStatus::OK);
  EXPECT_EQ(find_allocation(result, "model-a").approved_desired_replicas, 3u);
  EXPECT_EQ(find_allocation(result, "model-b").approved_desired_replicas, 3u);
}

TEST(PlacementBudgetAllocatorTest, OvercommitNeverForcesUnsafeScaleDown) {
  const std::vector<PlacementBudgetCandidate> candidates = {
      candidate("model-a", 4, 4, 4, 1, 0.0),
      candidate("model-b", 2, 3, 3, 1, 1.0),
  };
  const PlacementBudgetResult result =
      allocate_placement_budget(/*max_devices=*/10, candidates);
  EXPECT_EQ(result.status, PlacementBudgetStatus::OVERCOMMITTED);
  EXPECT_EQ(find_allocation(result, "model-a").approved_desired_replicas, 4u);
  EXPECT_EQ(find_allocation(result, "model-b").approved_desired_replicas, 2u);
}

TEST(PlacementBudgetAllocatorTest, ReportsProtectedBudgetShortfall) {
  const std::vector<PlacementBudgetCandidate> candidates = {
      candidate("model-a", 1, 3, 3, 1, 1.0),
      candidate("model-b", 1, 3, 3, 1, 1.0),
  };
  const PlacementBudgetResult result =
      allocate_placement_budget(/*max_devices=*/8, candidates);
  EXPECT_EQ(result.status,
            PlacementBudgetStatus::INSUFFICIENT_PROTECTED_BUDGET);
  EXPECT_LE(result.approved_devices, result.max_devices);
}

TEST(PlacementBudgetAllocatorTest, RejectsDuplicateInvalidAndOverflowInput) {
  std::vector<PlacementBudgetCandidate> candidates = {
      candidate("model-a", 2, 3, 3, 1, 1.0),
      candidate("model-a", 2, 3, 3, 1, 1.0),
  };
  EXPECT_EQ(allocate_placement_budget(100, candidates).status,
            PlacementBudgetStatus::INVALID_INPUT);

  candidates.resize(1);
  candidates.front().slo_risk_score = std::numeric_limits<double>::infinity();
  EXPECT_EQ(allocate_placement_budget(100, candidates).status,
            PlacementBudgetStatus::INVALID_INPUT);

  candidates.front() = candidate("model-a", 2, 3, 3, 1, 1.0);
  candidates.front().profile.devices_per_replica =
      std::numeric_limits<uint32_t>::max();
  candidates.front().recommendation.previous_desired_replicas = 10;
  candidates.front().recommendation.desired_replicas = 10;
  candidates.front().recommendation.action = PlacementAction::NONE;
  candidates.front().recommendation.status = PlacementPlanStatus::HOLD;
  EXPECT_EQ(allocate_placement_budget(100, candidates).status,
            PlacementBudgetStatus::OVERCOMMITTED);
}

}  // namespace
}  // namespace xllm_service::placement
