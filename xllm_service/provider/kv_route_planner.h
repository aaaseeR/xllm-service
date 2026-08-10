/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm-service/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "provider.pb.h"
#include "provider/kv_shadow_index.h"

namespace xllm_service::provider {

enum class KVRouteMode : int8_t {
  DISABLED = 0,
  SHADOW = 1,
  ENFORCED = 2,
};

enum class KVRouteFallback : int8_t {
  NONE = 0,
  DISABLED = 1,
  INVALID_INPUT = 2,
  CANDIDATE_LIMIT = 3,
  KV_UNAVAILABLE = 4,
  BELOW_MARGIN = 5,
};

KVRouteMode select_kv_route_mode(KVRouteMode configured_mode,
                                 bool enforced_gate_open,
                                 uint32_t enforced_bucket_permyriad,
                                 uint64_t request_hash);

struct KVRoutePlannerConfig {
  size_t max_candidate_plans = 16384;
  size_t least_load_shortlist = 8;
  size_t top_prefix_shortlist = 8;
  uint64_t prefill_queue_cost_us = 1000;
  uint64_t decode_request_cost_us = 1000;
  uint64_t prefill_token_cost_us = 10;
  double transfer_byte_cost_us = 0.001;
  uint64_t decode_headroom_cost_us = 1000;
  uint64_t prefill_reserve_blocks = 1;
  uint64_t kv_routing_margin_us = 100;
  uint64_t near_equal_cost_us = 10;
};

struct KVRouteRequest {
  uint64_t prompt_tokens = 0;
  uint64_t block_size = 0;
  uint64_t kv_bytes_per_token = 0;
  uint64_t request_hash = 0;
};

struct KVRouteEngineCandidate {
  std::string engine_uid;
  xllm::proto::ProviderEngineKey engine_key;
  std::string model_revision;
  std::string kv_namespace;
  uint64_t hash_seed = 0;
  std::string cache_group;
  uint64_t block_size = 0;
  xllm::proto::EngineRole role = xllm::proto::ENGINE_ROLE_UNSPECIFIED;
  bool load_known = false;
  uint64_t waiting_requests = 0;
  uint64_t running_requests = 0;
  uint64_t local_pending_requests = 0;
  uint64_t local_pending_tokens = 0;
  double kv_used_ratio = 0.0;
  uint64_t kv_free_blocks = 0;
  KVShadowHealth kv_health = KVShadowHealth::UNKNOWN;
  uint64_t hbm_prefix_blocks = 0;
  uint64_t host_prefix_blocks = 0;
  double residence_probability = 1.0;
};

struct KVRoutePlanCandidate {
  KVRouteEngineCandidate prefill;
  std::optional<KVRouteEngineCandidate> decode;
};

struct KVRouteEvaluation {
  size_t candidate_index = 0;
  uint64_t predicted_prefill_hit_tokens = 0;
  uint64_t predicted_decode_hit_tokens = 0;
  uint64_t effective_prefill_tokens = 0;
  uint64_t effective_transfer_bytes = 0;
  double survival_credit = 0.0;
  double load_only_cost_us = 0.0;
  double kv_cost_us = 0.0;
  bool in_shortlist = false;
};

struct KVRouteDecision {
  bool valid = false;
  size_t selected_index = 0;
  size_t load_only_index = 0;
  size_t kv_preferred_index = 0;
  KVRouteFallback fallback = KVRouteFallback::INVALID_INPUT;
  std::vector<KVRouteEvaluation> evaluations;
};

struct KVRouteObservation {
  KVRouteMode mode = KVRouteMode::DISABLED;
  KVRouteFallback fallback = KVRouteFallback::INVALID_INPUT;
  std::string selected_prefill;
  std::string selected_decode;
  std::string load_only_prefill;
  std::string load_only_decode;
  std::string kv_preferred_prefill;
  std::string kv_preferred_decode;
  uint64_t predicted_prefill_hit_tokens = 0;
  uint64_t predicted_decode_hit_tokens = 0;
  uint64_t predicted_effective_prefill_tokens = 0;
  uint64_t predicted_transfer_bytes = 0;
  uint64_t shadow_prefill_host_hit_tokens_ub = 0;
  uint64_t shadow_decode_host_hit_tokens_ub = 0;
  uint64_t kv_bytes_per_token = 0;
  double load_only_cost_us = 0.0;
  double kv_cost_us = 0.0;
};

struct LowerTierShadowCredit {
  uint64_t prefill_host_hit_tokens_ub = 0;
  uint64_t decode_host_hit_tokens_ub = 0;
};

// Reports only bounded hit-token upper bounds for future tier-cost
// calibration. HOST/SSD/STORE never affect V2 route selection or admission.
LowerTierShadowCredit lower_tier_shadow_credit(
    const KVRouteRequest& request,
    const std::vector<KVRoutePlanCandidate>& candidates);

// Pure, bounded K1 planner. Inputs have already crossed the shared Provider
// hard-filter boundary; the planner can rank but can never restore a rejected
// Engine or P/D link. All costs are expressed in microseconds.
class KVRoutePlanner final {
 public:
  explicit KVRoutePlanner(KVRoutePlannerConfig config);

  bool valid() const;
  KVRouteDecision select(const KVRouteRequest& request,
                         const std::vector<KVRoutePlanCandidate>& candidates,
                         KVRouteMode mode,
                         bool candidate_limit_exceeded = false) const;

 private:
  static uint64_t stable_tie_key(const KVRouteRequest& request,
                                 const KVRoutePlanCandidate& candidate);
  static uint64_t saturated_multiply(uint64_t left, uint64_t right);
  static double clamp_ratio(double value);

  KVRouteEvaluation evaluate(size_t candidate_index,
                             const KVRouteRequest& request,
                             const KVRoutePlanCandidate& candidate) const;
  size_t pick_lower_cost(const KVRouteRequest& request,
                         const std::vector<KVRoutePlanCandidate>& candidates,
                         const std::vector<KVRouteEvaluation>& evaluations,
                         const std::vector<size_t>& candidate_indices,
                         bool use_kv_cost) const;

  KVRoutePlannerConfig config_;
  bool config_valid_ = false;
};

}  // namespace xllm_service::provider
