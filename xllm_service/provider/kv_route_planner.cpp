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

#include "provider/kv_route_planner.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <utility>

namespace xllm_service::provider {
namespace {

constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

uint64_t hash_bytes(uint64_t hash, const std::string& value) {
  for (unsigned char byte : value) {
    hash ^= static_cast<uint64_t>(byte);
    hash *= kFnvPrime;
  }
  return hash;
}

double safe_ratio(double value) {
  if (!std::isfinite(value)) {
    return 1.0;
  }
  return std::clamp(value, 0.0, 1.0);
}

double candidate_load_cost(const KVRoutePlannerConfig& config,
                           const KVRouteEngineCandidate& candidate,
                           bool is_decode) {
  const uint64_t queued =
      candidate.waiting_requests > std::numeric_limits<uint64_t>::max() -
                                       candidate.local_pending_requests
          ? std::numeric_limits<uint64_t>::max()
          : candidate.waiting_requests + candidate.local_pending_requests;
  const double queued_cost =
      static_cast<double>(queued) *
      static_cast<double>(is_decode ? config.decode_request_cost_us
                                    : config.prefill_queue_cost_us);
  const double running_cost =
      is_decode ? static_cast<double>(candidate.running_requests) *
                      static_cast<double>(config.decode_request_cost_us)
                : 0.0;
  const double pending_token_cost =
      is_decode ? 0.0
                : static_cast<double>(candidate.local_pending_tokens) *
                      static_cast<double>(config.prefill_token_cost_us);
  const double headroom_cost =
      is_decode ? safe_ratio(candidate.kv_used_ratio) *
                      static_cast<double>(config.decode_headroom_cost_us)
                : 0.0;
  return queued_cost + running_cost + pending_token_cost + headroom_cost;
}

}  // namespace

KVRoutePlanner::KVRoutePlanner(KVRoutePlannerConfig config)
    : config_(std::move(config)) {
  config_valid_ =
      config_.max_candidate_plans > 0 && config_.least_load_shortlist > 0 &&
      config_.top_prefix_shortlist > 0 && config_.prefill_queue_cost_us > 0 &&
      config_.decode_request_cost_us > 0 && config_.prefill_token_cost_us > 0 &&
      std::isfinite(config_.transfer_byte_cost_us) &&
      config_.transfer_byte_cost_us >= 0.0 &&
      config_.decode_headroom_cost_us > 0;
}

bool KVRoutePlanner::valid() const { return config_valid_; }

uint64_t KVRoutePlanner::stable_tie_key(const KVRouteRequest& request,
                                        const KVRoutePlanCandidate& candidate) {
  uint64_t hash = kFnvOffset ^ request.request_hash;
  hash = hash_bytes(hash, candidate.prefill.engine_uid);
  if (candidate.decode.has_value()) {
    hash = hash_bytes(hash, candidate.decode->engine_uid);
  }
  return hash;
}

uint64_t KVRoutePlanner::saturated_multiply(uint64_t left, uint64_t right) {
  if (left == 0 || right == 0) {
    return 0;
  }
  if (left > std::numeric_limits<uint64_t>::max() / right) {
    return std::numeric_limits<uint64_t>::max();
  }
  return left * right;
}

double KVRoutePlanner::clamp_ratio(double value) { return safe_ratio(value); }

KVRouteEvaluation KVRoutePlanner::evaluate(
    size_t candidate_index,
    const KVRouteRequest& request,
    const KVRoutePlanCandidate& candidate) const {
  KVRouteEvaluation result;
  result.candidate_index = candidate_index;
  result.effective_prefill_tokens = request.prompt_tokens;
  result.effective_transfer_bytes =
      saturated_multiply(request.prompt_tokens, request.kv_bytes_per_token);

  const double prefill_load =
      candidate_load_cost(config_, candidate.prefill, false);
  const double decode_load =
      candidate.decode.has_value()
          ? candidate_load_cost(config_, *candidate.decode, true)
          : 0.0;
  const double full_prefill_cost =
      static_cast<double>(request.prompt_tokens) *
      static_cast<double>(config_.prefill_token_cost_us);
  const double full_transfer_cost =
      candidate.decode.has_value()
          ? static_cast<double>(result.effective_transfer_bytes) *
                config_.transfer_byte_cost_us
          : 0.0;
  result.load_only_cost_us =
      prefill_load + decode_load + full_prefill_cost + full_transfer_cost;

  const uint64_t required_blocks =
      request.prompt_tokens / request.block_size +
      (request.prompt_tokens % request.block_size == 0 ? 0 : 1);
  if (candidate.prefill.kv_health == KVShadowHealth::READY &&
      required_blocks > 0) {
    const uint64_t retainable_blocks =
        candidate.prefill.kv_free_blocks > config_.prefill_reserve_blocks
            ? candidate.prefill.kv_free_blocks - config_.prefill_reserve_blocks
            : 0;
    result.survival_credit =
        std::min(1.0,
                 static_cast<double>(retainable_blocks) /
                     static_cast<double>(required_blocks)) *
        clamp_ratio(candidate.prefill.residence_probability);
    const uint64_t hit_tokens =
        std::min(request.prompt_tokens,
                 saturated_multiply(candidate.prefill.hbm_prefix_blocks,
                                    request.block_size));
    result.predicted_prefill_hit_tokens = static_cast<uint64_t>(
        std::floor(static_cast<double>(hit_tokens) * result.survival_credit));
    result.effective_prefill_tokens =
        request.prompt_tokens - result.predicted_prefill_hit_tokens;
  }

  if (candidate.decode.has_value() &&
      candidate.decode->kv_health == KVShadowHealth::READY) {
    result.predicted_decode_hit_tokens =
        std::min(request.prompt_tokens,
                 saturated_multiply(candidate.decode->hbm_prefix_blocks,
                                    request.block_size));
    result.effective_transfer_bytes = saturated_multiply(
        request.prompt_tokens - result.predicted_decode_hit_tokens,
        request.kv_bytes_per_token);
  }

  result.kv_cost_us =
      prefill_load + decode_load +
      static_cast<double>(result.effective_prefill_tokens) *
          static_cast<double>(config_.prefill_token_cost_us) +
      (candidate.decode.has_value()
           ? static_cast<double>(result.effective_transfer_bytes) *
                 config_.transfer_byte_cost_us
           : 0.0);
  return result;
}

size_t KVRoutePlanner::pick_lower_cost(
    const KVRouteRequest& request,
    const std::vector<KVRoutePlanCandidate>& candidates,
    const std::vector<KVRouteEvaluation>& evaluations,
    const std::vector<size_t>& candidate_indices,
    bool use_kv_cost) const {
  size_t best = candidate_indices.front();
  for (size_t index : candidate_indices) {
    const double cost = use_kv_cost ? evaluations[index].kv_cost_us
                                    : evaluations[index].load_only_cost_us;
    const double best_cost = use_kv_cost ? evaluations[best].kv_cost_us
                                         : evaluations[best].load_only_cost_us;
    const double difference = std::abs(cost - best_cost);
    if (cost + static_cast<double>(config_.near_equal_cost_us) < best_cost ||
        (difference <= static_cast<double>(config_.near_equal_cost_us) &&
         stable_tie_key(request, candidates[index]) <
             stable_tie_key(request, candidates[best]))) {
      best = index;
    }
  }
  return best;
}

KVRouteDecision KVRoutePlanner::select(
    const KVRouteRequest& request,
    const std::vector<KVRoutePlanCandidate>& candidates,
    KVRouteMode mode,
    bool candidate_limit_exceeded) const {
  KVRouteDecision decision;
  if (!config_valid_ || request.prompt_tokens == 0 || request.block_size == 0 ||
      candidates.empty() || candidates.size() > config_.max_candidate_plans) {
    return decision;
  }

  decision.evaluations.reserve(candidates.size());
  std::vector<size_t> all_indices(candidates.size());
  std::iota(all_indices.begin(), all_indices.end(), 0);
  for (size_t index : all_indices) {
    if (candidates[index].prefill.engine_uid.empty() ||
        !candidates[index].prefill.load_known ||
        (candidates[index].decode.has_value() &&
         (candidates[index].decode->engine_uid.empty() ||
          !candidates[index].decode->load_known))) {
      return decision;
    }
    decision.evaluations.emplace_back(
        evaluate(index, request, candidates[index]));
  }
  decision.load_only_index = pick_lower_cost(
      request, candidates, decision.evaluations, all_indices, false);
  decision.kv_preferred_index = decision.load_only_index;
  decision.selected_index = decision.load_only_index;
  decision.valid = true;

  if (candidate_limit_exceeded) {
    decision.fallback = KVRouteFallback::CANDIDATE_LIMIT;
    return decision;
  }
  if (mode == KVRouteMode::DISABLED) {
    decision.fallback = KVRouteFallback::DISABLED;
    return decision;
  }

  std::vector<size_t> by_load = all_indices;
  std::sort(by_load.begin(), by_load.end(), [&](size_t left, size_t right) {
    return decision.evaluations[left].load_only_cost_us <
           decision.evaluations[right].load_only_cost_us;
  });
  std::vector<size_t> by_prefix = all_indices;
  std::sort(by_prefix.begin(), by_prefix.end(), [&](size_t left, size_t right) {
    const double left_blocks =
        static_cast<double>(candidates[left].prefill.hbm_prefix_blocks) +
        (candidates[left].decode.has_value()
             ? static_cast<double>(candidates[left].decode->hbm_prefix_blocks)
             : 0.0);
    const double right_blocks =
        static_cast<double>(candidates[right].prefill.hbm_prefix_blocks) +
        (candidates[right].decode.has_value()
             ? static_cast<double>(candidates[right].decode->hbm_prefix_blocks)
             : 0.0);
    return left_blocks > right_blocks;
  });

  std::vector<size_t> shortlist;
  std::unordered_set<size_t> included;
  const auto append = [&](const std::vector<size_t>& source, size_t limit) {
    const size_t count = std::min(limit, source.size());
    for (size_t offset = 0; offset < count; ++offset) {
      if (included.insert(source[offset]).second) {
        shortlist.emplace_back(source[offset]);
      }
    }
  };
  append(by_load, config_.least_load_shortlist);
  append(by_prefix, config_.top_prefix_shortlist);
  for (size_t index : shortlist) {
    decision.evaluations[index].in_shortlist = true;
  }

  const bool any_kv_ready =
      std::any_of(shortlist.begin(), shortlist.end(), [&](size_t index) {
        return candidates[index].prefill.kv_health == KVShadowHealth::READY ||
               (candidates[index].decode.has_value() &&
                candidates[index].decode->kv_health == KVShadowHealth::READY);
      });
  if (!any_kv_ready) {
    decision.fallback = KVRouteFallback::KV_UNAVAILABLE;
    return decision;
  }

  decision.kv_preferred_index = pick_lower_cost(
      request, candidates, decision.evaluations, shortlist, true);
  const double load_only_cost =
      decision.evaluations[decision.load_only_index].load_only_cost_us;
  const double kv_cost =
      decision.evaluations[decision.kv_preferred_index].kv_cost_us;
  if (kv_cost + static_cast<double>(config_.kv_routing_margin_us) >=
      load_only_cost) {
    decision.fallback = KVRouteFallback::BELOW_MARGIN;
    return decision;
  }

  decision.fallback = KVRouteFallback::NONE;
  if (mode == KVRouteMode::ENFORCED) {
    decision.selected_index = decision.kv_preferred_index;
  }
  return decision;
}

}  // namespace xllm_service::provider
