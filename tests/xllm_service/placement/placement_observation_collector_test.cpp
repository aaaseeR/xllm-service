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

#include "placement/placement_observation_collector.h"

#include <gtest/gtest.h>

#include <atomic>
#include <limits>
#include <thread>
#include <vector>

namespace xllm_service::placement {
namespace {

PlacementObservationCollectorConfig config() {
  return PlacementObservationCollectorConfig{
      .max_models = 2,
      .bucket_count = 3,
      .bucket_width_ms = 1000,
      .max_latency_samples_per_bucket = 8,
      .forecast_horizon_ms = 5000,
      .forecast_headroom = 1.25,
  };
}

TEST(PlacementObservationCollectorTest, AggregatesStableWindowAndQuantiles) {
  PlacementObservationCollector collector(config());
  ASSERT_TRUE(collector.valid());
  ASSERT_EQ(collector.record_ingress("model-a", 100, 50, 1000),
            PlacementObservationStatus::OK);
  ASSERT_EQ(collector.record_ingress("model-a", 300, 100, 1500),
            PlacementObservationStatus::OK);
  ASSERT_EQ(collector.record_admission_reject("model-a", 1500),
            PlacementObservationStatus::OK);
  ASSERT_EQ(collector.record_terminal("model-a", 20, 100.0, 10.0, 1600),
            PlacementObservationStatus::OK);
  ASSERT_EQ(collector.record_terminal("model-a", 40, 200.0, 20.0, 1700),
            PlacementObservationStatus::OK);

  PlacementObservation observation;
  ASSERT_EQ(collector.snapshot("model-a",
                               2000,
                               PlacementObservationExternalInputs{
                                   .queue_depth = 3.0,
                                   .kv_used_ratio = 0.7,
                                   .full_cache_loss_cost = 4.0,
                                   .confirmed_store_coverage = 0.25,
                               },
                               &observation),
            PlacementObservationStatus::OK);
  EXPECT_EQ(observation.generation, 1u);
  EXPECT_EQ(observation.window_ms, 2000u);
  EXPECT_DOUBLE_EQ(observation.forecast_request_rate, 1.25);
  EXPECT_DOUBLE_EQ(observation.prompt_tokens_per_request, 200.0);
  EXPECT_DOUBLE_EQ(observation.output_tokens_per_request, 30.0);
  EXPECT_NEAR(observation.admission_reject_rate, 1.0 / 3.0, 1e-12);
  EXPECT_DOUBLE_EQ(observation.ttft_p95_ms, 200.0);
  EXPECT_DOUBLE_EQ(observation.tpot_p95_ms, 20.0);
  EXPECT_EQ(observation.major_bucket_samples, 2u);
  EXPECT_FALSE(observation.cold_start);
  EXPECT_FALSE(observation.out_of_distribution);
}

TEST(PlacementObservationCollectorTest, RotatesOldBucketsDeterministically) {
  PlacementObservationCollector collector(config());
  ASSERT_EQ(collector.record_ingress("model-a", 100, 10, 1000),
            PlacementObservationStatus::OK);
  ASSERT_EQ(collector.record_terminal("model-a", 5, 10.0, 1.0, 1000),
            PlacementObservationStatus::OK);
  ASSERT_EQ(collector.record_ingress("model-a", 300, 30, 4000),
            PlacementObservationStatus::OK);
  ASSERT_EQ(collector.record_terminal("model-a", 15, 30.0, 3.0, 4000),
            PlacementObservationStatus::OK);

  PlacementObservation observation;
  ASSERT_EQ(
      collector.snapshot(
          "model-a", 4000, PlacementObservationExternalInputs{}, &observation),
      PlacementObservationStatus::OK);
  EXPECT_DOUBLE_EQ(observation.prompt_tokens_per_request, 300.0);
  EXPECT_DOUBLE_EQ(observation.output_tokens_per_request, 15.0);
  EXPECT_DOUBLE_EQ(observation.forecast_request_rate, 1.25);
  EXPECT_EQ(observation.major_bucket_samples, 1u);
}

TEST(PlacementObservationCollectorTest, FailsClosedOnBoundsAndOldClock) {
  PlacementObservationCollector collector(config());
  EXPECT_EQ(collector.record_ingress("model-a", 1, 1, 5000),
            PlacementObservationStatus::OK);
  EXPECT_EQ(collector.record_ingress("model-a", 1, 1, 1000),
            PlacementObservationStatus::CLOCK_REGRESSION);
  EXPECT_EQ(collector.record_ingress("model-b", 1, 1, 5000),
            PlacementObservationStatus::OK);
  EXPECT_EQ(collector.record_ingress("model-c", 1, 1, 5000),
            PlacementObservationStatus::CAPACITY_EXCEEDED);

  PlacementObservation observation;
  EXPECT_EQ(
      collector.snapshot(
          "missing", 5000, PlacementObservationExternalInputs{}, &observation),
      PlacementObservationStatus::NOT_FOUND);
  EXPECT_EQ(collector.snapshot(
                "model-a",
                5000,
                PlacementObservationExternalInputs{.kv_used_ratio = 2.0},
                &observation),
            PlacementObservationStatus::INVALID_INPUT);
}

TEST(PlacementObservationCollectorTest, TruncationMarksObservationOod) {
  PlacementObservationCollectorConfig small = config();
  small.max_latency_samples_per_bucket = 1;
  PlacementObservationCollector collector(small);
  ASSERT_EQ(collector.record_ingress("model-a", 1, 1, 1000),
            PlacementObservationStatus::OK);
  ASSERT_EQ(collector.record_terminal("model-a", 1, 1.0, 1.0, 1000),
            PlacementObservationStatus::OK);
  ASSERT_EQ(collector.record_terminal("model-a", 1, 2.0, 2.0, 1000),
            PlacementObservationStatus::OK);

  PlacementObservation observation;
  ASSERT_EQ(
      collector.snapshot(
          "model-a", 1000, PlacementObservationExternalInputs{}, &observation),
      PlacementObservationStatus::OK);
  EXPECT_TRUE(observation.out_of_distribution);
  EXPECT_DOUBLE_EQ(observation.ttft_p95_ms, 1.0);
}

TEST(PlacementObservationCollectorTest, ConcurrentProducersRemainExact) {
  PlacementObservationCollectorConfig wide = config();
  wide.max_latency_samples_per_bucket = 1000;
  PlacementObservationCollector collector(wide);
  constexpr size_t kThreads = 8;
  constexpr size_t kEvents = 100;
  std::vector<std::thread> threads;
  std::atomic<bool> failed = false;
  for (size_t thread = 0; thread < kThreads; ++thread) {
    threads.emplace_back([&collector, &failed]() {
      for (size_t event = 0; event < kEvents; ++event) {
        if (collector.record_ingress("model-a", 10, 5, 1000) !=
                PlacementObservationStatus::OK ||
            collector.record_terminal("model-a", 2, 10.0, 1.0, 1000) !=
                PlacementObservationStatus::OK) {
          failed.store(true, std::memory_order_relaxed);
        }
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  EXPECT_FALSE(failed.load(std::memory_order_relaxed));
  PlacementObservation observation;
  ASSERT_EQ(
      collector.snapshot(
          "model-a", 1000, PlacementObservationExternalInputs{}, &observation),
      PlacementObservationStatus::OK);
  EXPECT_EQ(observation.major_bucket_samples, kThreads * kEvents);
  EXPECT_DOUBLE_EQ(observation.prompt_tokens_per_request, 10.0);
  EXPECT_DOUBLE_EQ(observation.output_tokens_per_request, 2.0);
}

TEST(PlacementObservationCollectorTest, RejectsInvalidConfiguration) {
  PlacementObservationCollectorConfig invalid = config();
  invalid.forecast_headroom = 0.5;
  EXPECT_FALSE(valid_placement_observation_collector_config(invalid));
  invalid = config();
  invalid.bucket_count = std::numeric_limits<size_t>::max();
  EXPECT_FALSE(valid_placement_observation_collector_config(invalid));
}

}  // namespace
}  // namespace xllm_service::placement
