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

#include "provider/readiness_controller.h"

#include <gtest/gtest.h>

#include <optional>
#include <string>

namespace xllm_service::provider {
namespace {

ObservationSnapshot observation(ObservationMode mode, bool within_grace) {
  return ObservationSnapshot{
      .mode = mode,
      .mode_entered_monotonic_ms = 0,
      .hard_stale_ratio = 0.0,
      .within_grace = within_grace,
  };
}

ReadinessInput healthy_input() {
  return ReadinessInput{
      .is_leader = true,
      .has_accepted_full_snapshot = true,
      .has_compatible_capacity = true,
      .draining = false,
      .observation = observation(ObservationMode::NORMAL, false),
  };
}

ReadinessSnapshot update(ReadinessController* controller,
                         const ReadinessInput& input,
                         uint64_t now_monotonic_ms) {
  std::string error;
  const std::optional<ReadinessSnapshot> snapshot =
      controller->update(input, now_monotonic_ms, &error);
  EXPECT_TRUE(snapshot.has_value()) << error;
  return snapshot.value_or(ReadinessSnapshot{});
}

TEST(ReadinessControllerTest, FollowerFailsClosedAndPromotionUsesHold) {
  ReadinessController controller(
      ReadinessControllerConfig{.recovery_hold_ms = 10});
  ReadinessInput follower = healthy_input();
  follower.is_leader = false;

  ReadinessSnapshot snapshot = update(&controller, follower, 100);
  EXPECT_FALSE(snapshot.accepting_new_requests);
  EXPECT_EQ(snapshot.reason, ReadinessReason::NOT_LEADER);

  snapshot = update(&controller, healthy_input(), 101);
  EXPECT_FALSE(snapshot.accepting_new_requests);
  EXPECT_EQ(snapshot.reason, ReadinessReason::RECOVERY_HOLD);
  snapshot = update(&controller, healthy_input(), 111);
  EXPECT_TRUE(snapshot.accepting_new_requests);
  EXPECT_EQ(snapshot.reason, ReadinessReason::READY);
}

TEST(ReadinessControllerTest, RejectsInvalidConfigAndClockRegression) {
  ReadinessController invalid(ReadinessControllerConfig{.recovery_hold_ms = 0});
  EXPECT_FALSE(invalid.valid());
  std::string error;
  EXPECT_FALSE(invalid.update(healthy_input(), 100, &error).has_value());

  ReadinessController controller(
      ReadinessControllerConfig{.recovery_hold_ms = 10});
  update(&controller, healthy_input(), 100);
  EXPECT_FALSE(controller.update(healthy_input(), 99, &error).has_value());
  EXPECT_FALSE(error.empty());
}

TEST(ReadinessControllerTest, ColdAndEmptyReplicaFailClosed) {
  ReadinessController controller(
      ReadinessControllerConfig{.recovery_hold_ms = 10});
  ReadinessInput cold = healthy_input();
  cold.has_accepted_full_snapshot = false;
  ReadinessSnapshot snapshot = update(&controller, cold, 100);
  EXPECT_FALSE(snapshot.accepting_new_requests);
  EXPECT_EQ(snapshot.reason, ReadinessReason::STARTING_NO_FULL);
  EXPECT_EQ(snapshot.changed_monotonic_ms, 100);

  ReadinessInput empty = healthy_input();
  empty.has_compatible_capacity = false;
  snapshot = update(&controller, empty, 101);
  EXPECT_FALSE(snapshot.accepting_new_requests);
  EXPECT_EQ(snapshot.reason, ReadinessReason::STARTING_NO_COMPATIBLE_CAPACITY);
}

TEST(ReadinessControllerTest, RecoveryRequiresStableHold) {
  ReadinessController controller(
      ReadinessControllerConfig{.recovery_hold_ms = 10});
  ReadinessSnapshot snapshot = update(&controller, healthy_input(), 100);
  EXPECT_FALSE(snapshot.accepting_new_requests);
  EXPECT_EQ(snapshot.reason, ReadinessReason::RECOVERY_HOLD);
  EXPECT_FALSE(
      update(&controller, healthy_input(), 109).accepting_new_requests);
  snapshot = update(&controller, healthy_input(), 110);
  EXPECT_TRUE(snapshot.accepting_new_requests);
  EXPECT_EQ(snapshot.reason, ReadinessReason::READY);
}

TEST(ReadinessControllerTest, NormalCapacitySpikeDoesNotFlapReady) {
  ReadinessController controller(
      ReadinessControllerConfig{.recovery_hold_ms = 10});
  update(&controller, healthy_input(), 100);
  ASSERT_TRUE(update(&controller, healthy_input(), 110).accepting_new_requests);
  ReadinessInput no_capacity = healthy_input();
  no_capacity.has_compatible_capacity = false;
  const ReadinessSnapshot snapshot = update(&controller, no_capacity, 111);
  EXPECT_TRUE(snapshot.accepting_new_requests);
  EXPECT_EQ(snapshot.reason, ReadinessReason::READY);
}

TEST(ReadinessControllerTest, StateBlindCapacityLossIsImmediate) {
  ReadinessController controller(
      ReadinessControllerConfig{.recovery_hold_ms = 10});
  update(&controller, healthy_input(), 100);
  ASSERT_TRUE(update(&controller, healthy_input(), 110).accepting_new_requests);
  ReadinessInput blind = healthy_input();
  blind.observation = observation(ObservationMode::STATE_BLIND, false);
  blind.has_compatible_capacity = false;
  ReadinessSnapshot snapshot = update(&controller, blind, 111);
  EXPECT_FALSE(snapshot.accepting_new_requests);
  EXPECT_EQ(snapshot.reason, ReadinessReason::STATE_BLIND_NO_DIRECT_CAPACITY);

  EXPECT_EQ(update(&controller, healthy_input(), 112).reason,
            ReadinessReason::RECOVERY_HOLD);
  EXPECT_TRUE(update(&controller, healthy_input(), 122).accepting_new_requests);
}

TEST(ReadinessControllerTest, RegistryBlindAndDrainUseStableReasons) {
  ReadinessController controller(
      ReadinessControllerConfig{.recovery_hold_ms = 10});
  update(&controller, healthy_input(), 100);
  ASSERT_TRUE(update(&controller, healthy_input(), 110).accepting_new_requests);

  ReadinessInput blind = healthy_input();
  blind.observation = observation(ObservationMode::REGISTRY_BLIND, true);
  EXPECT_TRUE(update(&controller, blind, 111).accepting_new_requests);
  blind.observation = observation(ObservationMode::REGISTRY_BLIND, false);
  ReadinessSnapshot snapshot = update(&controller, blind, 112);
  EXPECT_FALSE(snapshot.accepting_new_requests);
  EXPECT_EQ(snapshot.reason, ReadinessReason::REGISTRY_BLIND_GRACE_EXPIRED);

  ReadinessInput draining = healthy_input();
  draining.draining = true;
  snapshot = update(&controller, draining, 113);
  EXPECT_FALSE(snapshot.accepting_new_requests);
  EXPECT_EQ(snapshot.reason, ReadinessReason::DRAINING);
  EXPECT_STREQ(readiness_reason_name(snapshot.reason), "DRAINING");
}

}  // namespace
}  // namespace xllm_service::provider
