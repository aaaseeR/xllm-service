/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace xllm_service {
namespace {

constexpr double kDramOverlapWeight = 0.5;
constexpr double kSsdOverlapWeight = 0.25;
constexpr double kLoadPenaltyWeight = 1.0;
constexpr double kSoftmaxTemperature = 0.2;

double get_score(const std::unordered_map<std::string, uint32_t>& scores,
                 const std::string& instance_name) {
  const auto it = scores.find(instance_name);
  if (it == scores.end()) {
    return 0.0;
  }
  return static_cast<double>(it->second);
}

double weighted_overlap_blocks(const OverlapScores& overlap_scores,
                               const std::string& instance_name) {
  return get_score(overlap_scores.hbm_instance_score, instance_name) +
         kDramOverlapWeight *
             get_score(overlap_scores.dram_instance_score, instance_name) +
         kSsdOverlapWeight *
             get_score(overlap_scores.ssd_instance_score, instance_name);
}

}  // namespace

bool CacheAwareRouting::select_instances_pair(
    std::shared_ptr<Request> request) {
  LoadBalanceInfos lb_infos;
  if (!request->token_ids.empty()) {
    Slice<int32_t> token_ids(request->token_ids.data(),
                             request->token_ids.size());
    global_kvcache_mgr_->match(token_ids, &lb_infos.overlap_scores);
    DLOG(INFO) << lb_infos.debug_string();
  }

  instance_mgr_->get_load_metrics(&lb_infos);
  DLOG(INFO) << lb_infos.debug_string();

  if (lb_infos.prefill_load_metrics.size() == 0) {
    LOG(INFO) << "No node available!";
    return false;
  }

  // find preifll
  cost_function(lb_infos.overlap_scores,
                lb_infos.prefill_load_metrics,
                lb_infos.prefill_max_waiting_requests_num,
                &request->routing.prefill_endpoint);

  // find decode
  if (lb_infos.decode_load_metrics.size()) {
    cost_function(lb_infos.overlap_scores,
                  lb_infos.decode_load_metrics,
                  lb_infos.decode_max_waiting_requests_num,
                  &request->routing.decode_endpoint);
  }

  return true;
}

void CacheAwareRouting::cost_function(
    const OverlapScores& overlap_scores,
    const std::unordered_map<std::string, LoadMetrics>& load_metrics,
    const uint64_t& max_waiting_requests_num,
    std::string* best_choice) {
  if (best_choice == nullptr || load_metrics.empty()) {
    return;
  }

  struct Candidate {
    std::string instance_name;
    double score = 0.0;
  };
  std::vector<Candidate> candidates;
  candidates.reserve(load_metrics.size());
  double max_score = -std::numeric_limits<double>::infinity();

  for (const auto& it : load_metrics) {
    const double max_blocks =
        std::max<double>(1.0, overlap_scores.max_block_num);
    const double overlap =
        std::min(weighted_overlap_blocks(overlap_scores, it.first), max_blocks);
    const double prefill_blocks_after_reuse = max_blocks - overlap;
    const double waiting_ratio =
        max_waiting_requests_num == 0
            ? 0.0
            : static_cast<double>(it.second.waiting_requests_num) /
                  static_cast<double>(max_waiting_requests_num);
    const double load_penalty =
        static_cast<double>(it.second.gpu_cache_usage_perc) + waiting_ratio;
    const double score =
        -prefill_blocks_after_reuse - kLoadPenaltyWeight * load_penalty;
    candidates.push_back({it.first, score});
    max_score = std::max(max_score, score);
  }

  if (candidates.empty()) {
    return;
  }

  thread_local std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<double> unit_dist(0.0, 1.0);
  double sum = 0.0;
  std::vector<double> weights;
  weights.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    const double weight =
        std::exp((candidate.score - max_score) / kSoftmaxTemperature);
    weights.push_back(weight);
    sum += weight;
  }

  if (sum <= 0.0 || !std::isfinite(sum)) {
    *best_choice = candidates.front().instance_name;
    return;
  }

  double threshold = unit_dist(rng) * sum;
  for (size_t i = 0; i < candidates.size(); ++i) {
    threshold -= weights[i];
    if (threshold <= 0.0) {
      *best_choice = candidates[i].instance_name;
      return;
    }
  }
  *best_choice = candidates.back().instance_name;
}

}  // namespace xllm_service
