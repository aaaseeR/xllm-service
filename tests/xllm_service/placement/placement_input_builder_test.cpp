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

#include "placement/placement_input_builder.h"

#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <vector>

namespace xllm_service::placement {
namespace {

PlacementCapacityProfile profile(std::string digest = "profile-a") {
  return PlacementCapacityProfile{
      .pool =
          PlacementPoolKey{
              .provider_id = xllm::proto::PROVIDER_ID_XLLM_NATIVE,
              .model_revision = "model-r1",
              .role = xllm::proto::ENGINE_ROLE_PREFILL,
              .profile_digest = std::move(digest),
          },
      .devices_per_replica = 1,
      .instance_cost_per_hour = 1.0,
      .load_warmup_p99_ms = 1000,
      .prefill_tokens_per_second_under_slo = 100.0,
      .decode_tokens_per_second_under_slo = 100.0,
      .requests_per_second_under_slo = 10.0,
      .target_utilization = 0.8,
      .min_replicas = 1,
      .max_replicas = 8,
      .ttft_slo_ms = 500.0,
      .tpot_slo_ms = 50.0,
  };
}

PlacementPoolRuntimeSpec spec(std::string digest = "profile-a") {
  return PlacementPoolRuntimeSpec{
      .profile = profile(std::move(digest)),
      .priority = 3,
      .slo_risk_score = 0.5,
      .config_digest = "config-a",
      .external =
          PlacementObservationExternalInputs{
              .queue_depth = 2.0,
              .kv_used_ratio = 0.2,
          },
  };
}

PlacementObservationCollectorConfig observation_config() {
  return PlacementObservationCollectorConfig{
      .max_models = 4,
      .bucket_count = 3,
      .bucket_width_ms = 1000,
      .max_latency_samples_per_bucket = 16,
      .forecast_horizon_ms = 5000,
      .forecast_headroom = 1.2,
  };
}

provider::EngineRegistryMemberSnapshot member(
    std::string uid,
    std::string incarnation,
    xllm::proto::EngineLifecycle lifecycle,
    bool schedulable = true,
    std::string profile_digest = "profile-a") {
  provider::EngineRegistryMemberSnapshot value;
  value.descriptor.mutable_identity()->set_provider_id(
      xllm::proto::PROVIDER_ID_XLLM_NATIVE);
  value.descriptor.mutable_identity()->set_engine_uid(std::move(uid));
  value.descriptor.mutable_identity()->set_incarnation_id(
      std::move(incarnation));
  value.descriptor.mutable_model()->set_model_revision("model-r1");
  value.descriptor.mutable_serving()->set_role(
      xllm::proto::ENGINE_ROLE_PREFILL);
  value.descriptor.set_profile_digest(std::move(profile_digest));
  value.descriptor.add_capabilities(xllm::proto::PROVIDER_CAPABILITY_DRAIN);
  xllm::proto::EngineState state;
  state.set_engine_uid(value.descriptor.identity().engine_uid());
  state.set_incarnation_id(value.descriptor.identity().incarnation_id());
  state.set_provider_id(value.descriptor.identity().provider_id());
  state.set_profile_digest(value.descriptor.profile_digest());
  state.set_model_revision(value.descriptor.model().model_revision());
  state.set_lifecycle(lifecycle);
  value.state = std::move(state);
  value.state_freshness = provider::EngineStateFreshness::FRESH;
  value.heartbeat_fresh = true;
  value.schedulable = schedulable;
  value.lifecycle_since_monotonic_ms = 1000;
  return value;
}

PlacementInputBuilder builder(size_t max_pools = 4, size_t max_members = 16) {
  return PlacementInputBuilder(PlacementInputBuilderConfig{
      .max_pools = max_pools,
      .max_members = max_members,
  });
}

void seed(PlacementObservationCollector* collector) {
  ASSERT_EQ(collector->record_ingress("model-r1", 100, 50, 2000),
            PlacementObservationStatus::OK);
}

TEST(PlacementInputBuilderTest, MapsLifecycleAndAggregatesRuntimePressure) {
  PlacementObservationCollector observations(observation_config());
  seed(&observations);
  std::vector<provider::EngineRegistryMemberSnapshot> members;
  members.push_back(
      member("engine-ready", "inc-ready", xllm::proto::ENGINE_LIFECYCLE_READY));
  xllm::proto::PerDpEngineState* dp = members.back().state->add_per_dp();
  dp->set_running(2);
  dp->set_waiting_capacity(3);
  dp->set_kv_used_ratio(0.8);
  members.push_back(member("engine-starting",
                           "inc-starting",
                           xllm::proto::ENGINE_LIFECYCLE_STARTING));
  members.push_back(member("engine-draining",
                           "inc-draining",
                           xllm::proto::ENGINE_LIFECYCLE_DRAINING));
  members.back().state->mutable_drain()->set_active_transfers(4);
  members.back().state->mutable_drain()->set_pending_cleanup(1);
  members.push_back(member(
      "engine-fenced", "inc-fenced", xllm::proto::ENGINE_LIFECYCLE_FENCED));

  std::vector<PlacementPoolCycleInput> inputs;
  ASSERT_EQ(builder().build({spec()}, members, &observations, 3000, &inputs),
            PlacementInputBuildStatus::OK);
  ASSERT_EQ(inputs.size(), 1u);
  ASSERT_EQ(inputs[0].replicas.size(), 4u);
  EXPECT_EQ(inputs[0].replicas[0].state, PlacementLifecycleState::READY);
  EXPECT_TRUE(inputs[0].replicas[0].fresh);
  EXPECT_EQ(inputs[0].replicas[0].active_reservations, 5u);
  EXPECT_EQ(inputs[0].replicas[1].state, PlacementLifecycleState::WARMING);
  EXPECT_EQ(inputs[0].replicas[2].state, PlacementLifecycleState::DRAINING);
  EXPECT_EQ(inputs[0].replicas[2].active_reservations, 1u);
  EXPECT_EQ(inputs[0].replicas[2].active_transfers, 4u);
  EXPECT_EQ(inputs[0].replicas[3].state, PlacementLifecycleState::FAILED);
  EXPECT_DOUBLE_EQ(inputs[0].observation.kv_used_ratio, 0.8);
}

TEST(PlacementInputBuilderTest, MissingStateCountsAsLoading) {
  PlacementObservationCollector observations(observation_config());
  seed(&observations);
  provider::EngineRegistryMemberSnapshot missing = member(
      "engine-loading", "inc-loading", xllm::proto::ENGINE_LIFECYCLE_STARTING);
  missing.state.reset();
  missing.state_freshness = provider::EngineStateFreshness::MISSING;
  missing.heartbeat_fresh = false;
  missing.schedulable = false;
  missing.lifecycle_since_monotonic_ms = 0;

  std::vector<PlacementPoolCycleInput> inputs;
  ASSERT_EQ(builder().build({spec()}, {missing}, &observations, 3000, &inputs),
            PlacementInputBuildStatus::OK);
  ASSERT_EQ(inputs[0].replicas.size(), 1u);
  EXPECT_EQ(inputs[0].replicas[0].state, PlacementLifecycleState::LOADING);
  EXPECT_FALSE(inputs[0].replicas[0].fresh);
  EXPECT_EQ(inputs[0].replicas[0].stable_since_ms, 3000u);
}

TEST(PlacementInputBuilderTest, StaleOrUnschedulableReadyFailsClosed) {
  PlacementObservationCollector observations(observation_config());
  seed(&observations);
  auto stale =
      member("engine-stale", "inc-stale", xllm::proto::ENGINE_LIFECYCLE_READY);
  stale.state_freshness = provider::EngineStateFreshness::SOFT_STALE;
  stale.schedulable = false;

  std::vector<PlacementPoolCycleInput> inputs;
  ASSERT_EQ(builder().build({spec()}, {stale}, &observations, 3000, &inputs),
            PlacementInputBuildStatus::OK);
  EXPECT_EQ(inputs[0].replicas[0].state, PlacementLifecycleState::FAILED);
  EXPECT_FALSE(inputs[0].replicas[0].fresh);
}

TEST(PlacementInputBuilderTest, IgnoresMembersFromAnotherExactPool) {
  PlacementObservationCollector observations(observation_config());
  seed(&observations);
  auto other = member("engine-other",
                      "inc-other",
                      xllm::proto::ENGINE_LIFECYCLE_READY,
                      true,
                      "profile-other");

  std::vector<PlacementPoolCycleInput> inputs;
  ASSERT_EQ(builder().build({spec()}, {other}, &observations, 3000, &inputs),
            PlacementInputBuildStatus::OK);
  EXPECT_TRUE(inputs[0].replicas.empty());
}

TEST(PlacementInputBuilderTest, RejectsDuplicatePoolsAndMemberIdentity) {
  PlacementObservationCollector observations(observation_config());
  seed(&observations);
  std::vector<PlacementPoolCycleInput> inputs;
  EXPECT_EQ(builder().build({spec(), spec()}, {}, &observations, 3000, &inputs),
            PlacementInputBuildStatus::INVALID_INPUT);
  auto duplicate =
      member("engine-a", "inc-a", xllm::proto::ENGINE_LIFECYCLE_READY);
  EXPECT_EQ(builder().build(
                {spec()}, {duplicate, duplicate}, &observations, 3000, &inputs),
            PlacementInputBuildStatus::INVALID_INPUT);
}

TEST(PlacementInputBuilderTest, FailsClosedOnMissingObservationAndBounds) {
  PlacementObservationCollector observations(observation_config());
  std::vector<PlacementPoolCycleInput> inputs;
  EXPECT_EQ(builder().build({spec()}, {}, &observations, 3000, &inputs),
            PlacementInputBuildStatus::OBSERVATION_UNAVAILABLE);
  seed(&observations);
  EXPECT_EQ(
      builder(1, 1).build(
          {spec()},
          {member("engine-a", "inc-a", xllm::proto::ENGINE_LIFECYCLE_READY),
           member("engine-b", "inc-b", xllm::proto::ENGINE_LIFECYCLE_READY)},
          &observations,
          3000,
          &inputs),
      PlacementInputBuildStatus::CAPACITY_EXCEEDED);
}

TEST(PlacementInputBuilderTest, SaturationCannotProduceDrainCandidate) {
  PlacementObservationCollector observations(observation_config());
  seed(&observations);
  auto overflow = member("engine-overflow",
                         "inc-overflow",
                         xllm::proto::ENGINE_LIFECYCLE_DRAINING);
  overflow.state->mutable_drain()->set_prefill_queue(
      std::numeric_limits<uint64_t>::max());
  overflow.state->mutable_drain()->set_pending_cleanup(1);
  std::vector<PlacementPoolCycleInput> inputs;
  EXPECT_EQ(builder().build({spec()}, {overflow}, &observations, 3000, &inputs),
            PlacementInputBuildStatus::CAPACITY_EXCEEDED);
}

}  // namespace
}  // namespace xllm_service::placement
