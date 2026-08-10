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

#include "placement/placement_planner.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace xllm_service::placement {
namespace {

PlacementRecommendation invalid_recommendation(const PlacementPoolState& state,
                                               PlacementReason reason) {
  return PlacementRecommendation{
      .status = PlacementPlanStatus::INVALID_INPUT,
      .action = PlacementAction::NONE,
      .reason = reason,
      .previous_desired_replicas = state.desired_replicas,
      .desired_replicas = state.desired_replicas,
      .next_state = state,
  };
}

PlacementRecommendation hold_recommendation(
    const PlacementPoolState& previous_state,
    const PlacementPoolState& next_state,
    PlacementReason reason,
    uint32_t safe_required,
    uint32_t warm_spare,
    double scale_down_value = 0.0) {
  return PlacementRecommendation{
      .status = PlacementPlanStatus::HOLD,
      .action = PlacementAction::NONE,
      .reason = reason,
      .previous_desired_replicas = previous_state.desired_replicas,
      .desired_replicas = previous_state.desired_replicas,
      .safe_required_replicas = safe_required,
      .warm_spare_replicas = warm_spare,
      .scale_down_value = scale_down_value,
      .next_state = next_state,
  };
}

bool add_overflows(uint32_t left, uint32_t right) {
  return left > std::numeric_limits<uint32_t>::max() - right;
}

bool elapsed_at_least(uint64_t now_ms, uint64_t since_ms, uint64_t wait_ms) {
  return since_ms != 0 && now_ms >= since_ms && now_ms - since_ms >= wait_ms;
}

bool ceil_to_uint32(double value, uint32_t* result) {
  if (result == nullptr || !std::isfinite(value) || value < 0.0 ||
      value > static_cast<double>(std::numeric_limits<uint32_t>::max())) {
    return false;
  }
  *result = static_cast<uint32_t>(std::ceil(value));
  return true;
}

bool required_replicas(const PlacementCapacityProfile& profile,
                       const PlacementObservation& observation,
                       uint32_t warm_spare,
                       uint32_t* required) {
  double work = 0.0;
  double capacity = 0.0;
  switch (profile.pool.role) {
    case xllm::proto::ENGINE_ROLE_PREFILL:
      work = observation.forecast_request_rate *
             observation.prompt_tokens_per_request;
      capacity = profile.prefill_tokens_per_second_under_slo;
      break;
    case xllm::proto::ENGINE_ROLE_DECODE:
      work = observation.forecast_request_rate *
             observation.output_tokens_per_request;
      capacity = profile.decode_tokens_per_second_under_slo;
      break;
    case xllm::proto::ENGINE_ROLE_AGGREGATED:
      work = observation.forecast_request_rate;
      capacity = profile.requests_per_second_under_slo;
      break;
    default:
      return false;
  }
  const double effective_capacity = capacity * profile.target_utilization;
  uint32_t base = 0;
  if (!std::isfinite(work) || !std::isfinite(effective_capacity) ||
      effective_capacity <= 0.0 ||
      !ceil_to_uint32(work / effective_capacity, &base) ||
      add_overflows(base, profile.failure_headroom_replicas) ||
      add_overflows(base + profile.failure_headroom_replicas, warm_spare)) {
    return false;
  }
  const uint32_t with_headroom =
      base + profile.failure_headroom_replicas + warm_spare;
  *required =
      std::clamp(with_headroom, profile.min_replicas, profile.max_replicas);
  return true;
}

PlacementReason scale_up_reason(const PlacementPlannerConfig& config,
                                const PlacementCapacityProfile& profile,
                                const PlacementObservation& observation,
                                uint32_t safe_required,
                                uint32_t current_desired) {
  if (safe_required > current_desired) {
    return PlacementReason::FORECAST_CAPACITY;
  }
  if (observation.queue_depth >= config.queue_high_watermark) {
    return PlacementReason::QUEUE_HIGH;
  }
  if (observation.admission_reject_rate >=
      config.admission_reject_high_watermark) {
    return PlacementReason::ADMISSION_REJECT_HIGH;
  }
  if (profile.ttft_slo_ms > 0.0 &&
      observation.ttft_p95_ms > profile.ttft_slo_ms) {
    return PlacementReason::TTFT_SLO_VIOLATION;
  }
  if (profile.tpot_slo_ms > 0.0 &&
      observation.tpot_p95_ms > profile.tpot_slo_ms) {
    return PlacementReason::TPOT_SLO_VIOLATION;
  }
  if (observation.kv_used_ratio >= config.kv_high_watermark) {
    return PlacementReason::KV_PRESSURE_HIGH;
  }
  return PlacementReason::NONE;
}

bool low_load(const PlacementPlannerConfig& config,
              const PlacementCapacityProfile& profile,
              const PlacementObservation& observation) {
  return observation.queue_depth <= config.queue_low_watermark &&
         observation.admission_reject_rate <=
             config.admission_reject_low_watermark &&
         observation.kv_used_ratio <= config.kv_low_watermark &&
         (profile.ttft_slo_ms == 0.0 ||
          observation.ttft_p95_ms <= profile.ttft_slo_ms) &&
         (profile.tpot_slo_ms == 0.0 ||
          observation.tpot_p95_ms <= profile.tpot_slo_ms);
}

}  // namespace

PlacementRecommendation plan_placement_pool(
    const PlacementPlannerConfig& config,
    const PlacementCapacityProfile& profile,
    const PlacementObservation& observation,
    const PlacementPoolState& state,
    uint64_t now_ms) {
  if (!valid_placement_planner_config(config)) {
    return invalid_recommendation(state, PlacementReason::INVALID_CONFIG);
  }
  if (!valid_placement_capacity_profile(profile) ||
      state.desired_replicas < profile.min_replicas ||
      state.desired_replicas > profile.max_replicas) {
    return invalid_recommendation(state, PlacementReason::INVALID_PROFILE);
  }
  if (!valid_placement_observation(observation)) {
    return invalid_recommendation(state, PlacementReason::INVALID_OBSERVATION);
  }
  if (now_ms < observation.observed_at_ms ||
      (state.last_scale_at_ms != 0 && now_ms < state.last_scale_at_ms) ||
      (state.high_signal_since_ms != 0 &&
       now_ms < state.high_signal_since_ms) ||
      (state.low_signal_since_ms != 0 && now_ms < state.low_signal_since_ms)) {
    return invalid_recommendation(state, PlacementReason::CLOCK_REGRESSION);
  }
  if (observation.generation <= state.last_observation_generation) {
    return hold_recommendation(state,
                               state,
                               PlacementReason::REPLAYED_OBSERVATION,
                               state.desired_replicas,
                               /*warm_spare=*/0);
  }

  PlacementPoolState next_state = state;
  next_state.last_observation_generation = observation.generation;
  const uint32_t warm_spare =
      observation.forecast_horizon_ms < profile.load_warmup_p99_ms ? 1 : 0;
  uint32_t safe_required = 0;
  if (!required_replicas(profile, observation, warm_spare, &safe_required)) {
    return invalid_recommendation(state, PlacementReason::INVALID_OBSERVATION);
  }

  const PlacementReason up_reason = scale_up_reason(
      config, profile, observation, safe_required, state.desired_replicas);
  if (up_reason != PlacementReason::NONE) {
    next_state.low_signal_since_ms = 0;
    if (next_state.high_signal_since_ms == 0) {
      next_state.high_signal_since_ms = now_ms;
    }
    if (state.pending_operations > 0) {
      return hold_recommendation(state,
                                 next_state,
                                 PlacementReason::PENDING_OPERATION,
                                 safe_required,
                                 warm_spare);
    }
    if (!elapsed_at_least(
            now_ms, next_state.high_signal_since_ms, config.scale_up_hold_ms)) {
      return hold_recommendation(state,
                                 next_state,
                                 PlacementReason::SCALE_UP_HOLD,
                                 safe_required,
                                 warm_spare);
    }
    if (state.desired_replicas == profile.max_replicas) {
      return hold_recommendation(state,
                                 next_state,
                                 PlacementReason::MAX_REPLICAS,
                                 safe_required,
                                 warm_spare);
    }
    const uint32_t reactive_target = state.desired_replicas + 1;
    const uint32_t requested_target = std::max(safe_required, reactive_target);
    const uint32_t step_target =
        state.desired_replicas >
                std::numeric_limits<uint32_t>::max() - config.max_scale_up_step
            ? std::numeric_limits<uint32_t>::max()
            : state.desired_replicas + config.max_scale_up_step;
    const uint32_t desired =
        std::min({requested_target, step_target, profile.max_replicas});
    next_state.desired_replicas = desired;
    next_state.high_signal_since_ms = 0;
    next_state.last_scale_at_ms = now_ms;
    return PlacementRecommendation{
        .status = PlacementPlanStatus::OK,
        .action = PlacementAction::SCALE_UP,
        .reason = up_reason,
        .previous_desired_replicas = state.desired_replicas,
        .desired_replicas = desired,
        .safe_required_replicas = safe_required,
        .warm_spare_replicas = warm_spare,
        .next_state = next_state,
    };
  }

  next_state.high_signal_since_ms = 0;
  if (safe_required >= state.desired_replicas) {
    next_state.low_signal_since_ms = 0;
    const PlacementReason reason =
        state.desired_replicas == profile.min_replicas
            ? PlacementReason::MIN_REPLICAS
            : PlacementReason::STABLE;
    return hold_recommendation(
        state, next_state, reason, safe_required, warm_spare);
  }
  if (state.pending_operations > 0) {
    next_state.low_signal_since_ms = 0;
    return hold_recommendation(state,
                               next_state,
                               PlacementReason::PENDING_OPERATION,
                               safe_required,
                               warm_spare);
  }
  if (observation.out_of_distribution || observation.cold_start) {
    next_state.low_signal_since_ms = 0;
    return hold_recommendation(state,
                               next_state,
                               PlacementReason::OUT_OF_DISTRIBUTION,
                               safe_required,
                               warm_spare);
  }
  if (observation.major_bucket_samples < config.min_scale_down_samples) {
    next_state.low_signal_since_ms = 0;
    return hold_recommendation(state,
                               next_state,
                               PlacementReason::INSUFFICIENT_SAMPLES,
                               safe_required,
                               warm_spare);
  }
  if (!low_load(config, profile, observation)) {
    next_state.low_signal_since_ms = 0;
    return hold_recommendation(
        state, next_state, PlacementReason::STABLE, safe_required, warm_spare);
  }
  if (next_state.low_signal_since_ms == 0) {
    next_state.low_signal_since_ms = now_ms;
  }
  if (!elapsed_at_least(now_ms,
                        next_state.low_signal_since_ms,
                        config.scale_down_stabilization_ms)) {
    return hold_recommendation(state,
                               next_state,
                               PlacementReason::SCALE_DOWN_STABILIZATION,
                               safe_required,
                               warm_spare);
  }
  if (state.last_scale_at_ms != 0 &&
      !elapsed_at_least(now_ms, state.last_scale_at_ms, config.cooldown_ms)) {
    return hold_recommendation(state,
                               next_state,
                               PlacementReason::COOLDOWN,
                               safe_required,
                               warm_spare);
  }

  constexpr double kMillisecondsPerHour = 3600000.0;
  const double instance_saving =
      profile.instance_cost_per_hour *
      static_cast<double>(config.economic_horizon_ms) / kMillisecondsPerHour;
  const double warmup_cost = profile.instance_cost_per_hour *
                             static_cast<double>(profile.load_warmup_p99_ms) /
                             kMillisecondsPerHour;
  const double effective_cache_loss =
      observation.full_cache_loss_cost *
      (1.0 - observation.confirmed_store_coverage);
  const double scale_down_value =
      instance_saving - warmup_cost - effective_cache_loss;
  if (!std::isfinite(scale_down_value) || scale_down_value <= 0.0) {
    return hold_recommendation(state,
                               next_state,
                               PlacementReason::CACHE_LOSS_COST,
                               safe_required,
                               warm_spare,
                               scale_down_value);
  }

  const uint32_t step_target =
      state.desired_replicas > config.max_scale_down_step
          ? state.desired_replicas - config.max_scale_down_step
          : 0;
  const uint32_t desired =
      std::max({safe_required, step_target, profile.min_replicas});
  next_state.desired_replicas = desired;
  next_state.low_signal_since_ms = 0;
  next_state.last_scale_at_ms = now_ms;
  return PlacementRecommendation{
      .status = PlacementPlanStatus::OK,
      .action = PlacementAction::SCALE_DOWN,
      .reason = PlacementReason::FORECAST_CAPACITY,
      .previous_desired_replicas = state.desired_replicas,
      .desired_replicas = desired,
      .safe_required_replicas = safe_required,
      .warm_spare_replicas = warm_spare,
      .scale_down_value = scale_down_value,
      .next_state = next_state,
  };
}

}  // namespace xllm_service::placement
