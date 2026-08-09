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

#include "provider/observation_controller.h"

#include <gtest/gtest.h>

#include <optional>
#include <string>

namespace xllm_service::provider {
namespace {

ObservationControllerConfig test_config() {
  return ObservationControllerConfig{
      .state_blind_enter_ratio = 0.5,
      .state_blind_exit_ratio = 0.25,
      .state_blind_enter_hold_ms = 10,
      .state_blind_exit_hold_ms = 20,
      .state_blind_grace_ms = 30,
      .registry_blind_grace_ms = 5,
  };
}

ObservationInput healthy_input() {
  return ObservationInput{
      .registry_known = true,
      .has_usable_state_snapshot = true,
      .has_current_full_snapshot = true,
      .member_count = 4,
      .hard_stale_member_count = 0,
  };
}

ObservationSnapshot update(ObservationController* controller,
                           const ObservationInput& input,
                           uint64_t now_monotonic_ms) {
  std::string error;
  const std::optional<ObservationSnapshot> snapshot =
      controller->update(input, now_monotonic_ms, &error);
  EXPECT_TRUE(snapshot.has_value()) << error;
  return snapshot.value_or(ObservationSnapshot{});
}

TEST(ObservationControllerTest, RejectsInvalidConfigurationAndInput) {
  ObservationControllerConfig config = test_config();
  config.state_blind_exit_ratio = config.state_blind_enter_ratio;
  ObservationController invalid(config);
  EXPECT_FALSE(invalid.valid());
  std::string error;
  EXPECT_FALSE(invalid.update(healthy_input(), 1, &error).has_value());

  ObservationController controller(test_config());
  ObservationInput input = healthy_input();
  input.hard_stale_member_count = 5;
  EXPECT_FALSE(controller.update(input, 1, &error).has_value());
}

TEST(ObservationControllerTest, EntersStateBlindOnlyAfterHold) {
  ObservationController controller(test_config());
  EXPECT_EQ(update(&controller, healthy_input(), 100).mode,
            ObservationMode::NORMAL);

  ObservationInput stale = healthy_input();
  stale.hard_stale_member_count = 2;
  EXPECT_EQ(update(&controller, stale, 101).mode, ObservationMode::NORMAL);
  EXPECT_EQ(update(&controller, stale, 110).mode, ObservationMode::NORMAL);
  ObservationSnapshot blind = update(&controller, stale, 111);
  EXPECT_EQ(blind.mode, ObservationMode::STATE_BLIND);
  EXPECT_TRUE(blind.within_grace);
  EXPECT_FALSE(update(&controller, stale, 141).within_grace);
}

TEST(ObservationControllerTest, ExitThresholdAndHoldPreventFlapping) {
  ObservationController controller(test_config());
  ObservationInput stale = healthy_input();
  stale.hard_stale_member_count = 4;
  update(&controller, healthy_input(), 100);
  update(&controller, stale, 101);
  ASSERT_EQ(update(&controller, stale, 111).mode, ObservationMode::STATE_BLIND);

  ObservationInput middle = healthy_input();
  middle.hard_stale_member_count = 2;
  EXPECT_EQ(update(&controller, middle, 120).mode,
            ObservationMode::STATE_BLIND);
  update(&controller, healthy_input(), 121);
  EXPECT_EQ(update(&controller, healthy_input(), 140).mode,
            ObservationMode::STATE_BLIND);
  EXPECT_EQ(update(&controller, healthy_input(), 141).mode,
            ObservationMode::NORMAL);
}

TEST(ObservationControllerTest, RegistryBlindIsImmediateAndHasShortGrace) {
  ObservationController controller(test_config());
  update(&controller, healthy_input(), 100);
  ObservationInput blind = healthy_input();
  blind.registry_known = false;
  ObservationSnapshot snapshot = update(&controller, blind, 101);
  EXPECT_EQ(snapshot.mode, ObservationMode::REGISTRY_BLIND);
  EXPECT_TRUE(snapshot.within_grace);
  EXPECT_FALSE(update(&controller, blind, 106).within_grace);

  EXPECT_EQ(update(&controller, healthy_input(), 107).mode,
            ObservationMode::NORMAL);
}

TEST(ObservationControllerTest, ColdReplicaWaitsForUsableState) {
  ObservationController controller(test_config());
  ObservationInput cold = healthy_input();
  cold.has_usable_state_snapshot = false;
  EXPECT_EQ(update(&controller, cold, 100).mode, ObservationMode::STATE_BLIND);
  EXPECT_EQ(update(&controller, healthy_input(), 101).mode,
            ObservationMode::STATE_BLIND);
  EXPECT_EQ(update(&controller, healthy_input(), 121).mode,
            ObservationMode::NORMAL);
}

TEST(ObservationControllerTest, InitialStaleSnapshotStartsStateBlind) {
  ObservationController controller(test_config());
  ObservationInput stale = healthy_input();
  stale.hard_stale_member_count = 2;
  EXPECT_EQ(update(&controller, stale, 100).mode, ObservationMode::STATE_BLIND);
}

TEST(ObservationControllerTest, RecoveryRequiresCurrentMasterFull) {
  ObservationController controller(test_config());
  ObservationInput stale = healthy_input();
  stale.hard_stale_member_count = 4;
  update(&controller, stale, 100);

  ObservationInput old_master_snapshot = healthy_input();
  old_master_snapshot.has_current_full_snapshot = false;
  EXPECT_EQ(update(&controller, old_master_snapshot, 101).mode,
            ObservationMode::STATE_BLIND);
  EXPECT_EQ(update(&controller, old_master_snapshot, 150).mode,
            ObservationMode::STATE_BLIND);

  update(&controller, healthy_input(), 151);
  EXPECT_EQ(update(&controller, healthy_input(), 171).mode,
            ObservationMode::NORMAL);
}

TEST(ObservationControllerTest, MonotonicClockRegressionFailsClosed) {
  ObservationController controller(test_config());
  update(&controller, healthy_input(), 100);
  std::string error;
  EXPECT_FALSE(controller.update(healthy_input(), 99, &error).has_value());
  EXPECT_FALSE(error.empty());
}

}  // namespace
}  // namespace xllm_service::provider
