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

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

namespace xllm_service::provider {
namespace {

KVRouteObservation observation() {
  return KVRouteObservation{
      .mode = KVRouteMode::SHADOW,
      .fallback = KVRouteFallback::KV_UNAVAILABLE,
      .predicted_prefill_hit_tokens = 80,
      .predicted_decode_hit_tokens = 32,
      .predicted_effective_prefill_tokens = 80,
      .predicted_transfer_bytes = 4096,
  };
}

TEST(KVRouteMetricsTest, SeparatesMissingFromValidZero) {
  KVRouteMetrics metrics;
  const KVRouteObservation predicted = observation();
  metrics.record_decision(predicted);
  metrics.record_actual(predicted,
                        KVRouteActual{
                            .prefix_state = PrefixMetricState::MISSING,
                        });
  metrics.record_actual(predicted,
                        KVRouteActual{
                            .prefix_state = PrefixMetricState::VALID_ZERO,
                            .actual_hit_tokens = 0,
                            .actual_prefill_tokens = 160,
                            .skipped_transfer_bytes = 0,
                        });

  const KVRouteMetricsSnapshot snapshot = metrics.snapshot();
  EXPECT_EQ(snapshot.decisions, 1u);
  EXPECT_EQ(snapshot.shadow_decisions, 1u);
  EXPECT_EQ(snapshot.missing_prefix_observations, 1u);
  EXPECT_EQ(snapshot.actual_hit_tokens, 0u);
  EXPECT_EQ(snapshot.actual_prefill_tokens, 160u);
  EXPECT_EQ(snapshot.missing_prefill_observations, 1u);
  EXPECT_EQ(snapshot.missing_transfer_observations, 1u);
  EXPECT_EQ(snapshot.overpredicted_requests, 1u);
  EXPECT_EQ(
      snapshot.fallbacks[static_cast<size_t>(KVRouteFallback::KV_UNAVAILABLE)],
      1u);
}

TEST(KVRouteMetricsTest, RecordsActualHitAndAdmissionConflict) {
  KVRouteMetrics metrics;
  const KVRouteObservation predicted = observation();
  metrics.record_decision(predicted);
  metrics.record_actual(predicted,
                        KVRouteActual{
                            .prefix_state = PrefixMetricState::VALID_NONZERO,
                            .actual_hit_tokens = 64,
                            .actual_prefill_tokens = 96,
                            .skipped_transfer_bytes = 2048,
                            .admission_conflict = true,
                        });

  const KVRouteMetricsSnapshot snapshot = metrics.snapshot();
  EXPECT_EQ(snapshot.actual_hit_tokens, 64u);
  EXPECT_EQ(snapshot.actual_prefill_tokens, 96u);
  EXPECT_EQ(snapshot.skipped_transfer_bytes, 2048u);
  EXPECT_EQ(snapshot.overpredicted_requests, 1u);
  EXPECT_EQ(snapshot.admission_conflicts, 1u);
}

TEST(KVRouteMetricsTest, DisabledObservationIsNotReportedAsMissing) {
  KVRouteMetrics metrics;
  KVRouteObservation predicted = observation();
  predicted.mode = KVRouteMode::DISABLED;
  metrics.record_decision(predicted);
  metrics.record_actual(predicted,
                        KVRouteActual{
                            .prefix_state = PrefixMetricState::DISABLED,
                        });

  const KVRouteMetricsSnapshot snapshot = metrics.snapshot();
  EXPECT_EQ(snapshot.disabled_prefix_observations, 1u);
  EXPECT_EQ(snapshot.missing_prefix_observations, 0u);
  EXPECT_EQ(snapshot.missing_prefill_observations, 0u);
  EXPECT_EQ(snapshot.missing_transfer_observations, 0u);
}

TEST(KVRouteMetricsTest, ConcurrentUpdatesRemainLinearizable) {
  KVRouteMetrics metrics;
  const KVRouteObservation predicted = observation();
  std::vector<std::thread> workers;
  for (size_t worker = 0; worker < 8; ++worker) {
    workers.emplace_back([&] {
      for (size_t iteration = 0; iteration < 1000; ++iteration) {
        metrics.record_decision(predicted);
      }
    });
  }
  for (std::thread& worker : workers) {
    worker.join();
  }

  const KVRouteMetricsSnapshot snapshot = metrics.snapshot();
  EXPECT_EQ(snapshot.decisions, 8000u);
  EXPECT_EQ(snapshot.predicted_prefill_hit_tokens, 640000u);
  EXPECT_EQ(snapshot.predicted_decode_hit_tokens, 256000u);
}

TEST(KVRouteMetricsTest, CountersSaturateInsteadOfWrapping) {
  KVRouteMetrics metrics;
  KVRouteObservation predicted = observation();
  predicted.predicted_transfer_bytes = std::numeric_limits<uint64_t>::max();
  metrics.record_decision(predicted);
  metrics.record_decision(predicted);

  EXPECT_EQ(metrics.snapshot().predicted_transfer_bytes,
            std::numeric_limits<uint64_t>::max());
}

}  // namespace
}  // namespace xllm_service::provider
