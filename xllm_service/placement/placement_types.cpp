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

#include "placement/placement_types.h"

#include <algorithm>
#include <cmath>

namespace xllm_service::placement {
namespace {

bool finite_nonnegative(double value) {
  return std::isfinite(value) && value >= 0.0;
}

bool valid_role(xllm::proto::EngineRole role) {
  return role == xllm::proto::ENGINE_ROLE_PREFILL ||
         role == xllm::proto::ENGINE_ROLE_DECODE ||
         role == xllm::proto::ENGINE_ROLE_AGGREGATED;
}

}  // namespace

bool valid_placement_identity(const std::string& value) {
  return !value.empty() && value.size() <= kMaxPlacementIdentityBytes &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return character != 0 && character != '\n' && character != '\r';
         });
}

bool valid_placement_pool_key(const PlacementPoolKey& key) {
  return xllm::proto::ProviderId_IsValid(key.provider_id) &&
         key.provider_id != xllm::proto::PROVIDER_ID_UNSPECIFIED &&
         valid_role(key.role) && valid_placement_identity(key.model_revision) &&
         valid_placement_identity(key.profile_digest);
}

bool placement_pool_keys_equal(const PlacementPoolKey& left,
                               const PlacementPoolKey& right) {
  return left.provider_id == right.provider_id &&
         left.model_revision == right.model_revision &&
         left.role == right.role && left.profile_digest == right.profile_digest;
}

bool valid_placement_capacity_profile(const PlacementCapacityProfile& profile) {
  if (!valid_placement_pool_key(profile.pool) ||
      profile.devices_per_replica == 0 ||
      !finite_nonnegative(profile.instance_cost_per_hour) ||
      profile.load_warmup_p99_ms == 0 ||
      !std::isfinite(profile.target_utilization) ||
      profile.target_utilization <= 0.0 || profile.target_utilization > 1.0 ||
      profile.min_replicas == 0 ||
      profile.max_replicas < profile.min_replicas ||
      profile.failure_headroom_replicas > profile.max_replicas ||
      !finite_nonnegative(profile.ttft_slo_ms) ||
      !finite_nonnegative(profile.tpot_slo_ms)) {
    return false;
  }
  switch (profile.pool.role) {
    case xllm::proto::ENGINE_ROLE_PREFILL:
      return std::isfinite(profile.prefill_tokens_per_second_under_slo) &&
             profile.prefill_tokens_per_second_under_slo > 0.0;
    case xllm::proto::ENGINE_ROLE_DECODE:
      return std::isfinite(profile.decode_tokens_per_second_under_slo) &&
             profile.decode_tokens_per_second_under_slo > 0.0;
    case xllm::proto::ENGINE_ROLE_AGGREGATED:
      return std::isfinite(profile.requests_per_second_under_slo) &&
             profile.requests_per_second_under_slo > 0.0;
    default:
      return false;
  }
}

bool valid_placement_planner_config(const PlacementPlannerConfig& config) {
  return config.scale_down_stabilization_ms > 0 && config.cooldown_ms > 0 &&
         config.economic_horizon_ms > 0 && config.min_scale_down_samples > 0 &&
         config.max_scale_up_step > 0 && config.max_scale_down_step > 0 &&
         finite_nonnegative(config.queue_low_watermark) &&
         finite_nonnegative(config.queue_high_watermark) &&
         config.queue_low_watermark < config.queue_high_watermark &&
         finite_nonnegative(config.admission_reject_low_watermark) &&
         finite_nonnegative(config.admission_reject_high_watermark) &&
         config.admission_reject_low_watermark <
             config.admission_reject_high_watermark &&
         config.admission_reject_high_watermark <= 1.0 &&
         finite_nonnegative(config.kv_low_watermark) &&
         finite_nonnegative(config.kv_high_watermark) &&
         config.kv_low_watermark < config.kv_high_watermark &&
         config.kv_high_watermark <= 1.0;
}

bool valid_placement_observation(const PlacementObservation& observation) {
  return observation.generation > 0 && observation.observed_at_ms > 0 &&
         observation.window_ms > 0 && observation.forecast_horizon_ms > 0 &&
         finite_nonnegative(observation.forecast_request_rate) &&
         finite_nonnegative(observation.prompt_tokens_per_request) &&
         finite_nonnegative(observation.output_tokens_per_request) &&
         finite_nonnegative(observation.queue_depth) &&
         finite_nonnegative(observation.admission_reject_rate) &&
         observation.admission_reject_rate <= 1.0 &&
         finite_nonnegative(observation.ttft_p95_ms) &&
         finite_nonnegative(observation.tpot_p95_ms) &&
         finite_nonnegative(observation.kv_used_ratio) &&
         observation.kv_used_ratio <= 1.0 &&
         finite_nonnegative(observation.full_cache_loss_cost) &&
         finite_nonnegative(observation.confirmed_store_coverage) &&
         observation.confirmed_store_coverage <= 1.0;
}

bool valid_placement_reason(PlacementReason reason) {
  switch (reason) {
    case PlacementReason::NONE:
    case PlacementReason::FORECAST_CAPACITY:
    case PlacementReason::QUEUE_HIGH:
    case PlacementReason::ADMISSION_REJECT_HIGH:
    case PlacementReason::TTFT_SLO_VIOLATION:
    case PlacementReason::TPOT_SLO_VIOLATION:
    case PlacementReason::KV_PRESSURE_HIGH:
    case PlacementReason::SCALE_UP_HOLD:
    case PlacementReason::SCALE_DOWN_STABILIZATION:
    case PlacementReason::COOLDOWN:
    case PlacementReason::INSUFFICIENT_SAMPLES:
    case PlacementReason::OUT_OF_DISTRIBUTION:
    case PlacementReason::PENDING_OPERATION:
    case PlacementReason::CACHE_LOSS_COST:
    case PlacementReason::MIN_REPLICAS:
    case PlacementReason::MAX_REPLICAS:
    case PlacementReason::STABLE:
    case PlacementReason::REPLAYED_OBSERVATION:
    case PlacementReason::INVALID_CONFIG:
    case PlacementReason::INVALID_PROFILE:
    case PlacementReason::INVALID_OBSERVATION:
    case PlacementReason::CLOCK_REGRESSION:
      return true;
  }
  return false;
}

const char* placement_reason_name(PlacementReason reason) {
  switch (reason) {
    case PlacementReason::NONE:
      return "NONE";
    case PlacementReason::FORECAST_CAPACITY:
      return "FORECAST_CAPACITY";
    case PlacementReason::QUEUE_HIGH:
      return "QUEUE_HIGH";
    case PlacementReason::ADMISSION_REJECT_HIGH:
      return "ADMISSION_REJECT_HIGH";
    case PlacementReason::TTFT_SLO_VIOLATION:
      return "TTFT_SLO_VIOLATION";
    case PlacementReason::TPOT_SLO_VIOLATION:
      return "TPOT_SLO_VIOLATION";
    case PlacementReason::KV_PRESSURE_HIGH:
      return "KV_PRESSURE_HIGH";
    case PlacementReason::SCALE_UP_HOLD:
      return "SCALE_UP_HOLD";
    case PlacementReason::SCALE_DOWN_STABILIZATION:
      return "SCALE_DOWN_STABILIZATION";
    case PlacementReason::COOLDOWN:
      return "COOLDOWN";
    case PlacementReason::INSUFFICIENT_SAMPLES:
      return "INSUFFICIENT_SAMPLES";
    case PlacementReason::OUT_OF_DISTRIBUTION:
      return "OUT_OF_DISTRIBUTION";
    case PlacementReason::PENDING_OPERATION:
      return "PENDING_OPERATION";
    case PlacementReason::CACHE_LOSS_COST:
      return "CACHE_LOSS_COST";
    case PlacementReason::MIN_REPLICAS:
      return "MIN_REPLICAS";
    case PlacementReason::MAX_REPLICAS:
      return "MAX_REPLICAS";
    case PlacementReason::STABLE:
      return "STABLE";
    case PlacementReason::REPLAYED_OBSERVATION:
      return "REPLAYED_OBSERVATION";
    case PlacementReason::INVALID_CONFIG:
      return "INVALID_CONFIG";
    case PlacementReason::INVALID_PROFILE:
      return "INVALID_PROFILE";
    case PlacementReason::INVALID_OBSERVATION:
      return "INVALID_OBSERVATION";
    case PlacementReason::CLOCK_REGRESSION:
      return "CLOCK_REGRESSION";
  }
  return "UNKNOWN";
}

}  // namespace xllm_service::placement
