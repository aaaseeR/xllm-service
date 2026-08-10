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

#include "placement/placement_planner.h"

#include <gtest/gtest.h>

#include <limits>
#include <string>

namespace xllm_service::placement {
namespace {

PlacementPlannerConfig config() {
  return PlacementPlannerConfig{
      .scale_up_hold_ms = 1000,
      .scale_down_stabilization_ms = 10000,
      .cooldown_ms = 5000,
      .economic_horizon_ms = 3600000,
      .min_scale_down_samples = 100,
      .max_scale_up_step = 2,
      .max_scale_down_step = 1,
      .queue_high_watermark = 10.0,
      .queue_low_watermark = 1.0,
      .admission_reject_high_watermark = 0.05,
      .admission_reject_low_watermark = 0.01,
      .kv_high_watermark = 0.90,
      .kv_low_watermark = 0.60,
  };
}

PlacementCapacityProfile profile(
    xllm::proto::EngineRole role = xllm::proto::ENGINE_ROLE_PREFILL) {
  return PlacementCapacityProfile{
      .pool =
          PlacementPoolKey{
              .provider_id = xllm::proto::PROVIDER_ID_XLLM_NATIVE,
              .model_revision = "model-r1",
              .role = role,
              .profile_digest = "profile-a",
          },
      .devices_per_replica = 8,
      .instance_cost_per_hour = 1.0,
      .load_warmup_p99_ms = 60000,
      .prefill_tokens_per_second_under_slo = 1000.0,
      .decode_tokens_per_second_under_slo = 500.0,
      .requests_per_second_under_slo = 5.0,
      .target_utilization = 0.8,
      .min_replicas = 2,
      .max_replicas = 10,
      .failure_headroom_replicas = 1,
      .ttft_slo_ms = 500.0,
      .tpot_slo_ms = 50.0,
  };
}

PlacementObservation observation(uint64_t generation = 1) {
  return PlacementObservation{
      .generation = generation,
      .observed_at_ms = 1000,
      .window_ms = 60000,
      .forecast_horizon_ms = 120000,
      .forecast_request_rate = 1.0,
      .prompt_tokens_per_request = 100.0,
      .output_tokens_per_request = 20.0,
      .queue_depth = 0.0,
      .admission_reject_rate = 0.0,
      .ttft_p95_ms = 100.0,
      .tpot_p95_ms = 10.0,
      .kv_used_ratio = 0.20,
      .major_bucket_samples = 1000,
      .full_cache_loss_cost = 0.10,
  };
}

PlacementPoolState state(uint32_t desired = 2) {
  return PlacementPoolState{
      .desired_replicas = desired,
      .ready_replicas = desired,
  };
}

TEST(PlacementTypesTest, RejectsInvalidAndDelimitedPoolIdentity) {
  PlacementCapacityProfile value = profile();
  EXPECT_TRUE(valid_placement_pool_key(value.pool));
  value.pool.model_revision = "bad\nmodel";
  EXPECT_FALSE(valid_placement_pool_key(value.pool));
  value = profile();
  value.pool.profile_digest =
      std::string(kMaxPlacementIdentityBytes + 1, 'p');
  EXPECT_FALSE(valid_placement_pool_key(value.pool));
  value = profile(xllm::proto::ENGINE_ROLE_UNSPECIFIED);
  EXPECT_FALSE(valid_placement_capacity_profile(value));
}

TEST(PlacementPlannerTest, RejectsInvalidConfigProfileAndObservation) {
  PlacementPlannerConfig invalid_config = config();
  invalid_config.max_scale_up_step = 0;
  EXPECT_EQ(plan_placement_pool(invalid_config,
                                profile(),
                                observation(),
                                state(),
                                /*now_ms=*/2000)
                .reason,
            PlacementReason::INVALID_CONFIG);

  PlacementCapacityProfile invalid_profile = profile();
  invalid_profile.prefill_tokens_per_second_under_slo = 0.0;
  EXPECT_EQ(plan_placement_pool(config(),
                                invalid_profile,
                                observation(),
                                state(),
                                /*now_ms=*/2000)
                .reason,
            PlacementReason::INVALID_PROFILE);

  PlacementObservation invalid_observation = observation();
  invalid_observation.forecast_request_rate =
      std::numeric_limits<double>::infinity();
  EXPECT_EQ(plan_placement_pool(config(),
                                profile(),
                                invalid_observation,
                                state(),
                                /*now_ms=*/2000)
                .reason,
            PlacementReason::INVALID_OBSERVATION);
}

TEST(PlacementPlannerTest, ComputesPrefillDecodeAndAggregatedIndependently) {
  PlacementPlannerConfig planner_config = config();
  planner_config.scale_up_hold_ms = 0;
  PlacementObservation input = observation();
  input.forecast_request_rate = 10.0;
  input.prompt_tokens_per_request = 400.0;
  input.output_tokens_per_request = 100.0;

  const PlacementRecommendation prefill = plan_placement_pool(
      planner_config, profile(), input, state(), /*now_ms=*/2000);
  const PlacementRecommendation decode = plan_placement_pool(
      planner_config,
      profile(xllm::proto::ENGINE_ROLE_DECODE),
      input,
      state(),
      /*now_ms=*/2000);
  const PlacementRecommendation aggregated = plan_placement_pool(
      planner_config,
      profile(xllm::proto::ENGINE_ROLE_AGGREGATED),
      input,
      state(),
      /*now_ms=*/2000);

  EXPECT_EQ(prefill.safe_required_replicas, 6u);
  EXPECT_EQ(decode.safe_required_replicas, 4u);
  EXPECT_EQ(aggregated.safe_required_replicas, 4u);
  EXPECT_EQ(prefill.desired_replicas, 4u);
  EXPECT_EQ(decode.desired_replicas, 4u);
  EXPECT_EQ(aggregated.desired_replicas, 4u);
}

TEST(PlacementPlannerTest, ShortForecastKeepsWarmSpare) {
  PlacementObservation input = observation();
  input.forecast_horizon_ms = 1000;
  const PlacementRecommendation result = plan_placement_pool(
      config(), profile(), input, state(), /*now_ms=*/2000);
  EXPECT_EQ(result.warm_spare_replicas, 1u);
  EXPECT_EQ(result.safe_required_replicas, 3u);
}

TEST(PlacementPlannerTest, ScaleUpRequiresHoldAndRespectsStep) {
  PlacementObservation first_input = observation();
  first_input.forecast_request_rate = 100.0;
  PlacementRecommendation first = plan_placement_pool(
      config(), profile(), first_input, state(), /*now_ms=*/2000);
  ASSERT_EQ(first.status, PlacementPlanStatus::HOLD);
  EXPECT_EQ(first.reason, PlacementReason::SCALE_UP_HOLD);
  EXPECT_EQ(first.next_state.high_signal_since_ms, 2000u);

  PlacementObservation second_input = first_input;
  second_input.generation = 2;
  second_input.observed_at_ms = 2000;
  const PlacementRecommendation second = plan_placement_pool(
      config(), profile(), second_input, first.next_state, /*now_ms=*/3000);
  EXPECT_EQ(second.status, PlacementPlanStatus::OK);
  EXPECT_EQ(second.action, PlacementAction::SCALE_UP);
  EXPECT_EQ(second.reason, PlacementReason::FORECAST_CAPACITY);
  EXPECT_EQ(second.previous_desired_replicas, 2u);
  EXPECT_EQ(second.desired_replicas, 4u);
}

TEST(PlacementPlannerTest, ReactiveSignalsScaleWithoutForecastUnderestimate) {
  PlacementPlannerConfig planner_config = config();
  planner_config.scale_up_hold_ms = 0;
  PlacementObservation input = observation();
  input.queue_depth = 10.0;
  const PlacementRecommendation queue_result = plan_placement_pool(
      planner_config, profile(), input, state(), /*now_ms=*/2000);
  EXPECT_EQ(queue_result.reason, PlacementReason::QUEUE_HIGH);
  EXPECT_EQ(queue_result.desired_replicas, 3u);

  input = observation(2);
  input.admission_reject_rate = 0.10;
  const PlacementRecommendation reject_result = plan_placement_pool(
      planner_config, profile(), input, state(), /*now_ms=*/2000);
  EXPECT_EQ(reject_result.reason, PlacementReason::ADMISSION_REJECT_HIGH);

  input = observation(3);
  input.kv_used_ratio = 0.95;
  const PlacementRecommendation kv_result = plan_placement_pool(
      planner_config, profile(), input, state(), /*now_ms=*/2000);
  EXPECT_EQ(kv_result.reason, PlacementReason::KV_PRESSURE_HIGH);
}

TEST(PlacementPlannerTest, PendingOperationBlocksAnotherScaleAction) {
  PlacementPlannerConfig planner_config = config();
  planner_config.scale_up_hold_ms = 0;
  PlacementObservation input = observation();
  input.queue_depth = 20.0;
  PlacementPoolState current = state();
  current.pending_operations = 1;
  const PlacementRecommendation result = plan_placement_pool(
      planner_config, profile(), input, current, /*now_ms=*/2000);
  EXPECT_EQ(result.status, PlacementPlanStatus::HOLD);
  EXPECT_EQ(result.reason, PlacementReason::PENDING_OPERATION);
}

TEST(PlacementPlannerTest, MaxReplicaLimitFailsClosed) {
  PlacementPlannerConfig planner_config = config();
  planner_config.scale_up_hold_ms = 0;
  PlacementObservation input = observation();
  input.queue_depth = 20.0;
  const PlacementRecommendation result = plan_placement_pool(
      planner_config, profile(), input, state(10), /*now_ms=*/2000);
  EXPECT_EQ(result.status, PlacementPlanStatus::HOLD);
  EXPECT_EQ(result.reason, PlacementReason::MAX_REPLICAS);
  EXPECT_EQ(result.desired_replicas, 10u);
}

TEST(PlacementPlannerTest, ReplayedObservationIsIdempotent) {
  PlacementPoolState current = state();
  current.last_observation_generation = 7;
  PlacementObservation input = observation(7);
  const PlacementRecommendation result = plan_placement_pool(
      config(), profile(), input, current, /*now_ms=*/2000);
  EXPECT_EQ(result.status, PlacementPlanStatus::HOLD);
  EXPECT_EQ(result.reason, PlacementReason::REPLAYED_OBSERVATION);
  EXPECT_EQ(result.next_state.last_observation_generation, 7u);
}

TEST(PlacementPlannerTest, ScaleDownRequiresSamplesAndStableWindow) {
  PlacementObservation input = observation();
  input.major_bucket_samples = 99;
  PlacementRecommendation result = plan_placement_pool(
      config(), profile(), input, state(4), /*now_ms=*/2000);
  EXPECT_EQ(result.reason, PlacementReason::INSUFFICIENT_SAMPLES);

  input.major_bucket_samples = 1000;
  result = plan_placement_pool(
      config(), profile(), input, state(4), /*now_ms=*/2000);
  ASSERT_EQ(result.reason, PlacementReason::SCALE_DOWN_STABILIZATION);
  EXPECT_EQ(result.next_state.low_signal_since_ms, 2000u);

  PlacementObservation next_input = input;
  next_input.generation = 2;
  next_input.observed_at_ms = 2000;
  result = plan_placement_pool(config(),
                               profile(),
                               next_input,
                               result.next_state,
                               /*now_ms=*/12000);
  EXPECT_EQ(result.status, PlacementPlanStatus::OK);
  EXPECT_EQ(result.action, PlacementAction::SCALE_DOWN);
  EXPECT_EQ(result.desired_replicas, 3u);
  EXPECT_GT(result.scale_down_value, 0.0);
}

TEST(PlacementPlannerTest, OodColdStartAndCooldownBlockScaleDown) {
  PlacementObservation input = observation();
  input.out_of_distribution = true;
  EXPECT_EQ(plan_placement_pool(
                config(), profile(), input, state(4), /*now_ms=*/2000)
                .reason,
            PlacementReason::OUT_OF_DISTRIBUTION);

  PlacementPoolState current = state(4);
  current.low_signal_since_ms = 1000;
  current.last_scale_at_ms = 9000;
  input = observation(2);
  input.observed_at_ms = 10000;
  EXPECT_EQ(plan_placement_pool(
                config(), profile(), input, current, /*now_ms=*/11000)
                .reason,
            PlacementReason::COOLDOWN);
}

TEST(PlacementPlannerTest, ConfirmedStoreCoverageCanUnlockScaleDown) {
  PlacementPoolState current = state(4);
  current.low_signal_since_ms = 1000;
  PlacementObservation input = observation();
  input.full_cache_loss_cost = 2.0;
  input.observed_at_ms = 10000;
  PlacementRecommendation result = plan_placement_pool(
      config(), profile(), input, current, /*now_ms=*/12000);
  EXPECT_EQ(result.reason, PlacementReason::CACHE_LOSS_COST);
  EXPECT_LT(result.scale_down_value, 0.0);

  input.generation = 2;
  input.confirmed_store_coverage = 1.0;
  result = plan_placement_pool(
      config(), profile(), input, current, /*now_ms=*/12000);
  EXPECT_EQ(result.status, PlacementPlanStatus::OK);
  EXPECT_EQ(result.action, PlacementAction::SCALE_DOWN);
}

TEST(PlacementPlannerTest, ClockRegressionFailsClosed) {
  PlacementObservation input = observation();
  input.observed_at_ms = 3000;
  const PlacementRecommendation result = plan_placement_pool(
      config(), profile(), input, state(), /*now_ms=*/2000);
  EXPECT_EQ(result.status, PlacementPlanStatus::INVALID_INPUT);
  EXPECT_EQ(result.reason, PlacementReason::CLOCK_REGRESSION);
}

}  // namespace
}  // namespace xllm_service::placement
