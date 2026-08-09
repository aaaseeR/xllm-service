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

#include "provider/kv_route_metrics.h"

#include <limits>

namespace xllm_service::provider {

void KVRouteMetrics::saturated_add(std::atomic<uint64_t>* target,
                                   uint64_t value) {
  uint64_t current = target->load(std::memory_order_relaxed);
  while (true) {
    const uint64_t next = value > std::numeric_limits<uint64_t>::max() - current
                              ? std::numeric_limits<uint64_t>::max()
                              : current + value;
    if (target->compare_exchange_weak(current,
                                      next,
                                      std::memory_order_relaxed,
                                      std::memory_order_relaxed)) {
      return;
    }
  }
}

void KVRouteMetrics::record_decision(const KVRouteObservation& observation) {
  saturated_add(&decisions_, 1);
  if (observation.mode == KVRouteMode::SHADOW) {
    saturated_add(&shadow_decisions_, 1);
  } else if (observation.mode == KVRouteMode::ENFORCED) {
    saturated_add(&enforced_decisions_, 1);
  }
  saturated_add(&predicted_prefill_hit_tokens_,
                observation.predicted_prefill_hit_tokens);
  saturated_add(&predicted_decode_hit_tokens_,
                observation.predicted_decode_hit_tokens);
  saturated_add(&predicted_effective_prefill_tokens_,
                observation.predicted_effective_prefill_tokens);
  saturated_add(&predicted_transfer_bytes_,
                observation.predicted_transfer_bytes);
  const size_t fallback = static_cast<size_t>(observation.fallback);
  if (fallback < fallbacks_.size()) {
    saturated_add(&fallbacks_[fallback], 1);
  }
}

void KVRouteMetrics::record_actual(const KVRouteObservation& observation,
                                   const KVRouteActual& actual) {
  if (actual.prefix_state == PrefixMetricState::DISABLED) {
    saturated_add(&disabled_prefix_observations_, 1);
    return;
  }
  if (actual.prefix_state == PrefixMetricState::MISSING ||
      !actual.actual_hit_tokens.has_value()) {
    saturated_add(&missing_prefix_observations_, 1);
  } else {
    saturated_add(&actual_hit_tokens_, *actual.actual_hit_tokens);
    const uint64_t predicted = observation.predicted_prefill_hit_tokens;
    if (predicted > *actual.actual_hit_tokens) {
      saturated_add(&overpredicted_requests_, 1);
    }
  }
  if (actual.actual_prefill_tokens.has_value()) {
    saturated_add(&actual_prefill_tokens_, *actual.actual_prefill_tokens);
  } else {
    saturated_add(&missing_prefill_observations_, 1);
  }
  if (actual.skipped_transfer_bytes.has_value()) {
    saturated_add(&skipped_transfer_bytes_, *actual.skipped_transfer_bytes);
  } else {
    saturated_add(&missing_transfer_observations_, 1);
  }
  if (actual.admission_conflict) {
    saturated_add(&admission_conflicts_, 1);
  }
}

KVRouteMetricsSnapshot KVRouteMetrics::snapshot() const {
  KVRouteMetricsSnapshot current;
  current.decisions = decisions_.load(std::memory_order_relaxed);
  current.shadow_decisions = shadow_decisions_.load(std::memory_order_relaxed);
  current.enforced_decisions =
      enforced_decisions_.load(std::memory_order_relaxed);
  current.predicted_prefill_hit_tokens =
      predicted_prefill_hit_tokens_.load(std::memory_order_relaxed);
  current.predicted_decode_hit_tokens =
      predicted_decode_hit_tokens_.load(std::memory_order_relaxed);
  current.predicted_effective_prefill_tokens =
      predicted_effective_prefill_tokens_.load(std::memory_order_relaxed);
  current.predicted_transfer_bytes =
      predicted_transfer_bytes_.load(std::memory_order_relaxed);
  current.actual_hit_tokens =
      actual_hit_tokens_.load(std::memory_order_relaxed);
  current.actual_prefill_tokens =
      actual_prefill_tokens_.load(std::memory_order_relaxed);
  current.skipped_transfer_bytes =
      skipped_transfer_bytes_.load(std::memory_order_relaxed);
  current.disabled_prefix_observations =
      disabled_prefix_observations_.load(std::memory_order_relaxed);
  current.missing_prefix_observations =
      missing_prefix_observations_.load(std::memory_order_relaxed);
  current.missing_prefill_observations =
      missing_prefill_observations_.load(std::memory_order_relaxed);
  current.missing_transfer_observations =
      missing_transfer_observations_.load(std::memory_order_relaxed);
  current.overpredicted_requests =
      overpredicted_requests_.load(std::memory_order_relaxed);
  current.admission_conflicts =
      admission_conflicts_.load(std::memory_order_relaxed);
  for (size_t index = 0; index < current.fallbacks.size(); ++index) {
    current.fallbacks[index] =
        fallbacks_[index].load(std::memory_order_relaxed);
  }
  return current;
}

}  // namespace xllm_service::provider
