/* Copyright 2025-2026 The xLLM Authors.

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

#include "cache_aware_routing.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "common/hash_util.h"

namespace xllm_service {
namespace {

provider::KVRoutePlannerConfig planner_config(const Options& options) {
  return provider::KVRoutePlannerConfig{
      .max_candidate_plans = options.kv_route_max_candidate_plans(),
      .least_load_shortlist = options.kv_route_least_load_shortlist(),
      .top_prefix_shortlist = options.kv_route_top_prefix_shortlist(),
      .prefill_queue_cost_us = options.kv_route_prefill_queue_cost_us(),
      .decode_request_cost_us = options.kv_route_decode_request_cost_us(),
      .prefill_token_cost_us = options.kv_route_prefill_token_cost_us(),
      .transfer_byte_cost_us = options.kv_route_transfer_byte_cost_us(),
      .decode_headroom_cost_us = options.kv_route_decode_headroom_cost_us(),
      .prefill_reserve_blocks = options.kv_route_prefill_reserve_blocks(),
      .kv_routing_margin_us = options.kv_route_margin_us(),
      .near_equal_cost_us = options.kv_route_near_equal_cost_us(),
  };
}

provider::KVRouteMode route_mode(const std::string& value) {
  if (value == "SHADOW") {
    return provider::KVRouteMode::SHADOW;
  }
  if (value == "ENFORCED") {
    return provider::KVRouteMode::ENFORCED;
  }
  return provider::KVRouteMode::DISABLED;
}

uint64_t stable_request_hash(const Request& request) {
  constexpr uint64_t kOffset = 1469598103934665603ULL;
  constexpr uint64_t kPrime = 1099511628211ULL;
  uint64_t hash = kOffset;
  const std::string& identity = request.correlation.request_uid().empty()
                                    ? request.model
                                    : request.correlation.request_uid();
  for (unsigned char byte : identity) {
    hash ^= static_cast<uint64_t>(byte);
    hash *= kPrime;
  }
  return hash;
}

std::vector<std::string> build_block_hashes(
    const std::string& kv_namespace,
    uint64_t hash_seed,
    const std::vector<int32_t>& token_ids,
    size_t block_size) {
  std::vector<std::string> hashes;
  if (kv_namespace.empty() || block_size == 0) {
    return hashes;
  }
  const size_t block_count = token_ids.size() / block_size;
  hashes.reserve(block_count);
  XXH3Key previous{};
  const uint8_t* parent = nullptr;
  const Slice<int32_t> tokens(token_ids.data(), token_ids.size());
  for (size_t block = 0; block < block_count; ++block) {
    XXH3Key current{};
    xxh3_128bits_hash(
        kv_namespace,
        hash_seed,
        parent,
        tokens.slice(block * block_size, (block + 1) * block_size),
        /*block_extra=*/{},
        current.data);
    hashes.emplace_back(current.to_string());
    previous = current;
    parent = previous.data;
  }
  return hashes;
}

provider::KVPrefixMatch observe_prefix_tier(
    provider::KVShadowIndex* index,
    const std::vector<std::string>& block_hashes,
    const provider::KVRouteEngineCandidate& candidate,
    xllm::proto::KVCacheTier tier) {
  if (index == nullptr || candidate.engine_key.engine_uid().empty() ||
      candidate.model_revision.empty() || candidate.kv_namespace.empty() ||
      candidate.cache_group.empty()) {
    return {};
  }
  return index->match_current_contiguous_prefix(candidate.engine_key,
                                                candidate.model_revision,
                                                candidate.kv_namespace,
                                                block_hashes,
                                                candidate.cache_group,
                                                tier);
}

void observe_prefix(provider::KVShadowIndex* index,
                    const std::vector<std::string>& block_hashes,
                    provider::KVRouteEngineCandidate* candidate) {
  if (candidate == nullptr) {
    return;
  }
  const provider::KVPrefixMatch match = observe_prefix_tier(
      index, block_hashes, *candidate, xllm::proto::KV_CACHE_TIER_HBM);
  candidate->kv_health = match.health;
  candidate->hbm_prefix_blocks = match.contiguous_blocks;
}

void observe_lower_tier_prefix(provider::KVShadowIndex* index,
                               const std::vector<std::string>& block_hashes,
                               provider::KVRouteEngineCandidate* candidate) {
  if (candidate == nullptr) {
    return;
  }
  const auto observe_shadow_blocks = [&](xllm::proto::KVCacheTier tier) {
    const provider::KVPrefixMatch tier_match =
        observe_prefix_tier(index, block_hashes, *candidate, tier);
    return tier_match.health == provider::KVShadowHealth::READY
               ? tier_match.contiguous_blocks
               : 0;
  };
  candidate->host_prefix_blocks =
      observe_shadow_blocks(xllm::proto::KV_CACHE_TIER_HOST);
  candidate->ssd_prefix_blocks =
      observe_shadow_blocks(xllm::proto::KV_CACHE_TIER_SSD);
  candidate->store_prefix_blocks =
      observe_shadow_blocks(xllm::proto::KV_CACHE_TIER_STORE);
}

std::string decode_name(const provider::KVRoutePlanCandidate& candidate) {
  return candidate.decode.has_value() ? candidate.decode->engine_uid : "";
}

}  // namespace

CacheAwareRouting::CacheAwareRouting(const Options& options,
                                     std::shared_ptr<InstanceMgr> instance_mgr,
                                     provider::KVShadowIndex* kv_shadow_index)
    : LoadBalancePolicy(std::move(instance_mgr)),
      options_(options),
      planner_(planner_config(options)),
      mode_(route_mode(options.kv_route_mode())),
      kv_shadow_index_(kv_shadow_index) {}

bool CacheAwareRouting::fallback_load_only(
    const std::shared_ptr<Request>& request) const {
  request->kv_route_observation.reset();
  return instance_mgr_->get_next_instance_pair(
      &request->routing, request->provider_id, request->model);
}

bool CacheAwareRouting::select_instances_pair(
    std::shared_ptr<Request> request) {
  if (request == nullptr || request->token_ids.empty() || !planner_.valid() ||
      !request->kv_isolation_reusable) {
    return request != nullptr && fallback_load_only(request);
  }
  const std::string model_revision =
      request->canonical_request.has_value()
          ? request->canonical_request->model_revision()
          : request->model;
  std::vector<provider::KVRoutePlanCandidate> candidates;
  bool truncated = false;
  if (!instance_mgr_->get_kv_route_candidates(
          request->provider_id,
          model_revision,
          options_.kv_route_max_candidate_plans(),
          &candidates,
          &truncated)) {
    return fallback_load_only(request);
  }

  const uint64_t block_size = candidates.front().prefill.block_size;
  const std::string& kv_namespace = candidates.front().prefill.kv_namespace;
  const uint64_t hash_seed = candidates.front().prefill.hash_seed;
  const bool incompatible_hash_contract =
      block_size == 0 || kv_namespace.empty() || hash_seed == 0 ||
      std::any_of(candidates.begin(), candidates.end(), [&](const auto& plan) {
        return plan.prefill.block_size != block_size ||
               plan.prefill.kv_namespace != kv_namespace ||
               plan.prefill.hash_seed != hash_seed ||
               (plan.decode.has_value() &&
                (plan.decode->block_size != block_size ||
                 plan.decode->kv_namespace != kv_namespace ||
                 plan.decode->hash_seed != hash_seed));
      });
  if (incompatible_hash_contract) {
    return fallback_load_only(request);
  }

  const std::string request_kv_namespace = derive_request_kv_namespace(
      kv_namespace, request->kv_isolation_domain, /*adapter_identity=*/"");
  if (request_kv_namespace.empty()) {
    return fallback_load_only(request);
  }
  const std::vector<std::string> block_hashes = build_block_hashes(
      request_kv_namespace, hash_seed, request->token_ids, block_size);
  for (provider::KVRoutePlanCandidate& candidate : candidates) {
    observe_prefix(kv_shadow_index_, block_hashes, &candidate.prefill);
    if (candidate.decode.has_value()) {
      observe_prefix(kv_shadow_index_, block_hashes, &candidate.decode.value());
    }
  }

  const provider::KVRouteRequest route_request{
      .prompt_tokens = request->token_ids.size(),
      .block_size = block_size,
      .kv_bytes_per_token = options_.kv_route_bytes_per_token(),
      .request_hash = stable_request_hash(*request),
  };
  const provider::KVRouteMode effective_mode = provider::select_kv_route_mode(
      mode_,
      options_.kv_route_enforced_gate_open(),
      options_.kv_route_enforced_bucket_permyriad(),
      route_request.request_hash);
  const provider::KVRouteDecision decision =
      planner_.select(route_request, candidates, effective_mode, truncated);
  if (!decision.valid || decision.selected_index >= candidates.size() ||
      decision.load_only_index >= candidates.size() ||
      decision.kv_preferred_index >= candidates.size() ||
      decision.evaluations.size() != candidates.size()) {
    return fallback_load_only(request);
  }

  // Lower-tier cache state is V2 observation-only. Query only the bounded
  // planner shortlist after the HBM decision has been frozen, so HOST/SSD/
  // STORE cannot influence routing and do not add O(all candidates) work.
  for (size_t index = 0; index < candidates.size(); ++index) {
    if (!decision.evaluations[index].in_shortlist) {
      continue;
    }
    observe_lower_tier_prefix(
        kv_shadow_index_, block_hashes, &candidates[index].prefill);
    if (candidates[index].decode.has_value()) {
      observe_lower_tier_prefix(
          kv_shadow_index_, block_hashes, &candidates[index].decode.value());
    }
  }

  const provider::KVRoutePlanCandidate& selected =
      candidates[decision.selected_index];
  request->routing.prefill_name = selected.prefill.engine_uid;
  request->routing.decode_name = decode_name(selected);

  const provider::KVRoutePlanCandidate& load_only =
      candidates[decision.load_only_index];
  const provider::KVRoutePlanCandidate& kv_preferred =
      candidates[decision.kv_preferred_index];
  const provider::KVRouteEvaluation& kv_evaluation =
      decision.evaluations[decision.kv_preferred_index];
  const provider::LowerTierShadowCredit lower_tier_credit =
      provider::lower_tier_shadow_credit(route_request, candidates);
  request->kv_route_observation = provider::KVRouteObservation{
      .mode = effective_mode,
      .fallback = decision.fallback,
      .selected_prefill = selected.prefill.engine_uid,
      .selected_decode = decode_name(selected),
      .load_only_prefill = load_only.prefill.engine_uid,
      .load_only_decode = decode_name(load_only),
      .kv_preferred_prefill = kv_preferred.prefill.engine_uid,
      .kv_preferred_decode = decode_name(kv_preferred),
      .predicted_prefill_hit_tokens =
          kv_evaluation.predicted_prefill_hit_tokens,
      .predicted_decode_hit_tokens = kv_evaluation.predicted_decode_hit_tokens,
      .predicted_effective_prefill_tokens =
          kv_evaluation.effective_prefill_tokens,
      .predicted_transfer_bytes = kv_evaluation.effective_transfer_bytes,
      .shadow_prefill_host_hit_tokens_ub =
          lower_tier_credit.prefill_host_hit_tokens_ub,
      .shadow_prefill_ssd_hit_tokens_ub =
          lower_tier_credit.prefill_ssd_hit_tokens_ub,
      .shadow_prefill_store_hit_tokens_ub =
          lower_tier_credit.prefill_store_hit_tokens_ub,
      .shadow_decode_host_hit_tokens_ub =
          lower_tier_credit.decode_host_hit_tokens_ub,
      .shadow_decode_ssd_hit_tokens_ub =
          lower_tier_credit.decode_ssd_hit_tokens_ub,
      .shadow_decode_store_hit_tokens_ub =
          lower_tier_credit.decode_store_hit_tokens_ub,
      .kv_bytes_per_token = route_request.kv_bytes_per_token,
      .load_only_cost_us =
          decision.evaluations[decision.load_only_index].load_only_cost_us,
      .kv_cost_us = kv_evaluation.kv_cost_us,
  };
  return true;
}

}  // namespace xllm_service
