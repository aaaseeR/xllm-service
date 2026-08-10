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

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <tuple>
#include <utility>

namespace xllm_service::placement {
namespace {

using PoolIdentity = std::tuple<int32_t, std::string, int32_t, std::string>;

struct PendingScaleUp {
  size_t allocation_index = 0;
  uint32_t priority = 0;
  double slo_risk_score = 0.0;
  uint32_t devices_per_replica = 0;
  uint32_t requested_delta = 0;
  uint32_t protected_delta = 0;
  PoolIdentity identity;
};

PoolIdentity pool_identity(const PlacementPoolKey& pool) {
  return PoolIdentity{static_cast<int32_t>(pool.provider_id),
                      pool.model_revision,
                      static_cast<int32_t>(pool.role),
                      pool.profile_digest};
}

bool multiply_overflows(uint64_t left, uint64_t right) {
  return right != 0 && left > std::numeric_limits<uint64_t>::max() / right;
}

bool add_overflows(uint64_t left, uint64_t right) {
  return left > std::numeric_limits<uint64_t>::max() - right;
}

bool valid_recommendation(const PlacementCapacityProfile& profile,
                          const PlacementRecommendation& recommendation) {
  if (recommendation.status == PlacementPlanStatus::INVALID_INPUT ||
      recommendation.previous_desired_replicas < profile.min_replicas ||
      recommendation.previous_desired_replicas > profile.max_replicas ||
      recommendation.desired_replicas < profile.min_replicas ||
      recommendation.desired_replicas > profile.max_replicas ||
      recommendation.safe_required_replicas > profile.max_replicas) {
    return false;
  }
  switch (recommendation.action) {
    case PlacementAction::NONE:
      return recommendation.desired_replicas ==
             recommendation.previous_desired_replicas;
    case PlacementAction::SCALE_UP:
      return recommendation.status == PlacementPlanStatus::OK &&
             recommendation.desired_replicas >
                 recommendation.previous_desired_replicas;
    case PlacementAction::SCALE_DOWN:
      return recommendation.status == PlacementPlanStatus::OK &&
             recommendation.desired_replicas <
                 recommendation.previous_desired_replicas &&
             recommendation.desired_replicas >=
                 recommendation.safe_required_replicas;
  }
  return false;
}

bool higher_rank(const PendingScaleUp& left, const PendingScaleUp& right) {
  if (left.priority != right.priority) {
    return left.priority > right.priority;
  }
  if (left.slo_risk_score != right.slo_risk_score) {
    return left.slo_risk_score > right.slo_risk_score;
  }
  return left.identity < right.identity;
}

bool reserve_replicas(uint64_t available_devices,
                      uint32_t devices_per_replica,
                      uint32_t requested_replicas,
                      uint32_t* approved_replicas,
                      uint64_t* approved_devices) {
  if (approved_replicas == nullptr || approved_devices == nullptr ||
      devices_per_replica == 0) {
    return false;
  }
  const uint64_t capacity = available_devices / devices_per_replica;
  const uint64_t approved = std::min<uint64_t>(capacity, requested_replicas);
  *approved_replicas = static_cast<uint32_t>(approved);
  *approved_devices = approved * devices_per_replica;
  return true;
}

}  // namespace

PlacementBudgetResult allocate_placement_budget(
    uint64_t max_devices,
    const std::vector<PlacementBudgetCandidate>& candidates) {
  PlacementBudgetResult result{
      .status = PlacementBudgetStatus::OK,
      .max_devices = max_devices,
  };
  if (max_devices == 0 || candidates.empty()) {
    result.status = PlacementBudgetStatus::INVALID_INPUT;
    return result;
  }

  std::set<PoolIdentity> identities;
  std::vector<PendingScaleUp> pending;
  uint64_t baseline_devices = 0;
  for (const PlacementBudgetCandidate& candidate : candidates) {
    const PlacementCapacityProfile& profile = candidate.profile;
    const PlacementRecommendation& recommendation = candidate.recommendation;
    const PoolIdentity identity = pool_identity(profile.pool);
    if (!valid_placement_capacity_profile(profile) ||
        !valid_recommendation(profile, recommendation) ||
        !std::isfinite(candidate.slo_risk_score) ||
        candidate.slo_risk_score < 0.0 || !identities.insert(identity).second) {
      result.status = PlacementBudgetStatus::INVALID_INPUT;
      result.allocations.clear();
      return result;
    }

    const uint32_t baseline_replicas =
        recommendation.action == PlacementAction::SCALE_DOWN
            ? recommendation.desired_replicas
            : recommendation.previous_desired_replicas;
    if (multiply_overflows(baseline_replicas, profile.devices_per_replica)) {
      result.status = PlacementBudgetStatus::INVALID_INPUT;
      result.allocations.clear();
      return result;
    }
    const uint64_t pool_baseline_devices =
        static_cast<uint64_t>(baseline_replicas) * profile.devices_per_replica;
    if (add_overflows(baseline_devices, pool_baseline_devices)) {
      result.status = PlacementBudgetStatus::INVALID_INPUT;
      result.allocations.clear();
      return result;
    }
    baseline_devices += pool_baseline_devices;

    PlacementBudgetAllocation allocation{
        .pool = profile.pool,
        .decision = PlacementBudgetDecision::APPROVED,
        .previous_desired_replicas = recommendation.previous_desired_replicas,
        .requested_desired_replicas = recommendation.desired_replicas,
        .approved_desired_replicas = baseline_replicas,
        .approved_devices = pool_baseline_devices,
    };
    const size_t allocation_index = result.allocations.size();
    result.allocations.emplace_back(std::move(allocation));
    if (recommendation.action == PlacementAction::SCALE_UP) {
      const uint32_t requested_delta =
          recommendation.desired_replicas - baseline_replicas;
      const uint32_t protected_target =
          std::min(recommendation.desired_replicas,
                   std::max(profile.min_replicas,
                            recommendation.safe_required_replicas));
      const uint32_t protected_delta =
          protected_target > baseline_replicas
              ? protected_target - baseline_replicas
              : 0;
      pending.emplace_back(PendingScaleUp{
          .allocation_index = allocation_index,
          .priority = candidate.priority,
          .slo_risk_score = candidate.slo_risk_score,
          .devices_per_replica = profile.devices_per_replica,
          .requested_delta = requested_delta,
          .protected_delta = protected_delta,
          .identity = identity,
      });
      result.allocations.back().decision =
          PlacementBudgetDecision::BUDGET_BLOCKED;
    }
  }

  std::sort(pending.begin(), pending.end(), higher_rank);
  uint64_t available_devices =
      baseline_devices < max_devices ? max_devices - baseline_devices : 0;
  uint64_t protected_devices = 0;
  bool protected_budget_shortfall = false;

  for (PendingScaleUp& item : pending) {
    uint32_t approved_replicas = 0;
    uint64_t approved_devices = 0;
    reserve_replicas(available_devices,
                     item.devices_per_replica,
                     item.protected_delta,
                     &approved_replicas,
                     &approved_devices);
    PlacementBudgetAllocation& allocation =
        result.allocations[item.allocation_index];
    allocation.approved_desired_replicas += approved_replicas;
    allocation.approved_devices += approved_devices;
    item.requested_delta -= approved_replicas;
    available_devices -= approved_devices;
    protected_devices += approved_devices;
    if (approved_replicas < item.protected_delta) {
      protected_budget_shortfall = true;
    }
  }

  if (!protected_budget_shortfall) {
    for (const PendingScaleUp& item : pending) {
      uint32_t approved_replicas = 0;
      uint64_t approved_devices = 0;
      reserve_replicas(available_devices,
                       item.devices_per_replica,
                       item.requested_delta,
                       &approved_replicas,
                       &approved_devices);
      PlacementBudgetAllocation& allocation =
          result.allocations[item.allocation_index];
      allocation.approved_desired_replicas += approved_replicas;
      allocation.approved_devices += approved_devices;
      available_devices -= approved_devices;
    }
  }

  result.protected_devices = protected_devices;
  for (PlacementBudgetAllocation& allocation : result.allocations) {
    if (allocation.requested_desired_replicas ==
        allocation.approved_desired_replicas) {
      allocation.decision = PlacementBudgetDecision::APPROVED;
    } else if (allocation.approved_desired_replicas ==
               allocation.previous_desired_replicas) {
      allocation.decision = PlacementBudgetDecision::BUDGET_BLOCKED;
    } else {
      allocation.decision = PlacementBudgetDecision::PARTIALLY_APPROVED;
    }
  }
  result.approved_devices = max_devices - available_devices;
  if (baseline_devices > max_devices) {
    result.status = PlacementBudgetStatus::OVERCOMMITTED;
    result.approved_devices = baseline_devices;
  } else if (protected_budget_shortfall) {
    result.status = PlacementBudgetStatus::INSUFFICIENT_PROTECTED_BUDGET;
  }
  return result;
}

}  // namespace xllm_service::placement
