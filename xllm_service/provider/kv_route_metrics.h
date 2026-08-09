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

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "provider/kv_route_planner.h"

namespace xllm_service::provider {

enum class PrefixMetricState : int8_t {
  DISABLED = 0,
  MISSING = 1,
  VALID_ZERO = 2,
  VALID_NONZERO = 3,
};

struct KVRouteActual {
  PrefixMetricState prefix_state = PrefixMetricState::MISSING;
  PrefixMetricState decode_prefix_state = PrefixMetricState::MISSING;
  std::optional<uint64_t> actual_hit_tokens;
  std::optional<uint64_t> actual_decode_hit_tokens;
  std::optional<uint64_t> actual_prefill_tokens;
  std::optional<uint64_t> skipped_transfer_bytes;
  bool admission_conflict = false;
};

struct KVRouteMetricsSnapshot {
  uint64_t decisions = 0;
  uint64_t shadow_decisions = 0;
  uint64_t enforced_decisions = 0;
  uint64_t predicted_prefill_hit_tokens = 0;
  uint64_t predicted_decode_hit_tokens = 0;
  uint64_t predicted_effective_prefill_tokens = 0;
  uint64_t predicted_transfer_bytes = 0;
  uint64_t actual_hit_tokens = 0;
  uint64_t actual_decode_hit_tokens = 0;
  uint64_t actual_prefill_tokens = 0;
  uint64_t skipped_transfer_bytes = 0;
  uint64_t disabled_prefix_observations = 0;
  uint64_t missing_prefix_observations = 0;
  uint64_t missing_decode_prefix_observations = 0;
  uint64_t missing_prefill_observations = 0;
  uint64_t missing_transfer_observations = 0;
  uint64_t overpredicted_requests = 0;
  uint64_t decode_overpredicted_requests = 0;
  uint64_t admission_conflicts = 0;
  std::array<uint64_t, 6> fallbacks{};
};

std::optional<uint64_t> logical_skipped_transfer_bytes(
    uint64_t decode_hit_tokens,
    uint64_t kv_bytes_per_token);

// Fixed-size lock-free counters for the K1 shadow gate. No request identity,
// prompt text or unbounded label is retained.
class KVRouteMetrics final {
 public:
  void record_decision(const KVRouteObservation& observation);
  void record_actual(const KVRouteObservation& observation,
                     const KVRouteActual& actual);
  KVRouteMetricsSnapshot snapshot() const;

 private:
  static void saturated_add(std::atomic<uint64_t>* target, uint64_t value);

  std::atomic<uint64_t> decisions_ = 0;
  std::atomic<uint64_t> shadow_decisions_ = 0;
  std::atomic<uint64_t> enforced_decisions_ = 0;
  std::atomic<uint64_t> predicted_prefill_hit_tokens_ = 0;
  std::atomic<uint64_t> predicted_decode_hit_tokens_ = 0;
  std::atomic<uint64_t> predicted_effective_prefill_tokens_ = 0;
  std::atomic<uint64_t> predicted_transfer_bytes_ = 0;
  std::atomic<uint64_t> actual_hit_tokens_ = 0;
  std::atomic<uint64_t> actual_decode_hit_tokens_ = 0;
  std::atomic<uint64_t> actual_prefill_tokens_ = 0;
  std::atomic<uint64_t> skipped_transfer_bytes_ = 0;
  std::atomic<uint64_t> disabled_prefix_observations_ = 0;
  std::atomic<uint64_t> missing_prefix_observations_ = 0;
  std::atomic<uint64_t> missing_decode_prefix_observations_ = 0;
  std::atomic<uint64_t> missing_prefill_observations_ = 0;
  std::atomic<uint64_t> missing_transfer_observations_ = 0;
  std::atomic<uint64_t> overpredicted_requests_ = 0;
  std::atomic<uint64_t> decode_overpredicted_requests_ = 0;
  std::atomic<uint64_t> admission_conflicts_ = 0;
  std::array<std::atomic<uint64_t>, 6> fallbacks_{};
};

}  // namespace xllm_service::provider
