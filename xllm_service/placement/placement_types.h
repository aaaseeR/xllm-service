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
#include <string>

#include "provider.pb.h"

namespace xllm_service::placement {

inline constexpr uint32_t kPlacementSchemaVersion = 1;
inline constexpr size_t kMaxPlacementIdentityBytes = 256;

enum class PlacementPlanStatus : int8_t {
  OK = 0,
  HOLD = 1,
  INVALID_INPUT = 2,
};

enum class PlacementAction : int8_t {
  NONE = 0,
  SCALE_UP = 1,
  SCALE_DOWN = 2,
};

enum class PlacementReason : int8_t {
  NONE = 0,
  FORECAST_CAPACITY = 1,
  QUEUE_HIGH = 2,
  ADMISSION_REJECT_HIGH = 3,
  TTFT_SLO_VIOLATION = 4,
  TPOT_SLO_VIOLATION = 5,
  KV_PRESSURE_HIGH = 6,
  SCALE_UP_HOLD = 7,
  SCALE_DOWN_STABILIZATION = 8,
  COOLDOWN = 9,
  INSUFFICIENT_SAMPLES = 10,
  OUT_OF_DISTRIBUTION = 11,
  PENDING_OPERATION = 12,
  CACHE_LOSS_COST = 13,
  MIN_REPLICAS = 14,
  MAX_REPLICAS = 15,
  STABLE = 16,
  REPLAYED_OBSERVATION = 17,
  INVALID_CONFIG = 18,
  INVALID_PROFILE = 19,
  INVALID_OBSERVATION = 20,
  CLOCK_REGRESSION = 21,
};

struct PlacementPoolKey {
  xllm::proto::ProviderId provider_id =
      xllm::proto::PROVIDER_ID_UNSPECIFIED;
  std::string model_revision;
  xllm::proto::EngineRole role = xllm::proto::ENGINE_ROLE_UNSPECIFIED;
  std::string profile_digest;
};

struct PlacementCapacityProfile {
  PlacementPoolKey pool;
  uint32_t devices_per_replica = 0;
  double instance_cost_per_hour = 0.0;
  uint64_t load_warmup_p99_ms = 0;
  double prefill_tokens_per_second_under_slo = 0.0;
  double decode_tokens_per_second_under_slo = 0.0;
  double requests_per_second_under_slo = 0.0;
  double target_utilization = 0.0;
  uint32_t min_replicas = 0;
  uint32_t max_replicas = 0;
  uint32_t failure_headroom_replicas = 0;
  double ttft_slo_ms = 0.0;
  double tpot_slo_ms = 0.0;
};

struct PlacementPlannerConfig {
  uint64_t scale_up_hold_ms = 0;
  uint64_t scale_down_stabilization_ms = 0;
  uint64_t cooldown_ms = 0;
  uint64_t economic_horizon_ms = 0;
  uint64_t min_scale_down_samples = 0;
  uint32_t max_scale_up_step = 0;
  uint32_t max_scale_down_step = 0;
  double queue_high_watermark = 0.0;
  double queue_low_watermark = 0.0;
  double admission_reject_high_watermark = 0.0;
  double admission_reject_low_watermark = 0.0;
  double kv_high_watermark = 0.0;
  double kv_low_watermark = 0.0;
};

struct PlacementObservation {
  uint64_t generation = 0;
  uint64_t observed_at_ms = 0;
  uint64_t window_ms = 0;
  uint64_t forecast_horizon_ms = 0;
  double forecast_request_rate = 0.0;
  double prompt_tokens_per_request = 0.0;
  double output_tokens_per_request = 0.0;
  double queue_depth = 0.0;
  double admission_reject_rate = 0.0;
  double ttft_p95_ms = 0.0;
  double tpot_p95_ms = 0.0;
  double kv_used_ratio = 0.0;
  uint64_t major_bucket_samples = 0;
  bool out_of_distribution = false;
  bool cold_start = false;
  double full_cache_loss_cost = 0.0;
  double confirmed_store_coverage = 0.0;
};

struct PlacementPoolState {
  uint32_t desired_replicas = 0;
  uint32_t ready_replicas = 0;
  uint32_t loading_replicas = 0;
  uint32_t warming_replicas = 0;
  uint32_t draining_replicas = 0;
  uint32_t pending_operations = 0;
  uint64_t high_signal_since_ms = 0;
  uint64_t low_signal_since_ms = 0;
  uint64_t last_scale_at_ms = 0;
  uint64_t last_observation_generation = 0;
};

struct PlacementRecommendation {
  PlacementPlanStatus status = PlacementPlanStatus::INVALID_INPUT;
  PlacementAction action = PlacementAction::NONE;
  PlacementReason reason = PlacementReason::INVALID_CONFIG;
  uint32_t previous_desired_replicas = 0;
  uint32_t desired_replicas = 0;
  uint32_t safe_required_replicas = 0;
  uint32_t warm_spare_replicas = 0;
  double scale_down_value = 0.0;
  PlacementPoolState next_state;
};

bool valid_placement_identity(const std::string& value);

bool valid_placement_pool_key(const PlacementPoolKey& key);

bool placement_pool_keys_equal(const PlacementPoolKey& left,
                               const PlacementPoolKey& right);

bool valid_placement_capacity_profile(
    const PlacementCapacityProfile& profile);

bool valid_placement_planner_config(const PlacementPlannerConfig& config);

bool valid_placement_observation(const PlacementObservation& observation);

bool valid_placement_reason(PlacementReason reason);

const char* placement_reason_name(PlacementReason reason);

}  // namespace xllm_service::placement
