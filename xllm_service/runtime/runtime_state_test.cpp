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

#include "runtime/runtime_state.h"

#include <gtest/gtest.h>

namespace xllm_service {
namespace {

TEST(RuntimeStateTest, StartsLiveButNotReady) {
  RuntimeState state;

  RuntimeHealthSnapshot snapshot = state.health_snapshot();

  EXPECT_EQ(snapshot.phase, RuntimePhase::STARTING);
  EXPECT_TRUE(snapshot.live);
  EXPECT_FALSE(snapshot.ready);
  EXPECT_EQ(snapshot.reason, "starting");
}

TEST(RuntimeStateTest, BecomesReadyOnlyAfterRuntimeStarts) {
  RuntimeState state;
  state.set_backend_ready(true);

  EXPECT_FALSE(state.health_snapshot().ready);

  state.mark_running();
  RuntimeHealthSnapshot snapshot = state.health_snapshot();
  EXPECT_EQ(snapshot.phase, RuntimePhase::RUNNING);
  EXPECT_TRUE(snapshot.live);
  EXPECT_TRUE(snapshot.ready);
  EXPECT_TRUE(snapshot.reason.empty());
}

TEST(RuntimeStateTest, PublishesBackendFailureReason) {
  RuntimeState state;
  state.mark_running();
  state.set_backend_ready(false, "model is not ready");

  RuntimeHealthSnapshot snapshot = state.health_snapshot();

  EXPECT_TRUE(snapshot.live);
  EXPECT_FALSE(snapshot.ready);
  EXPECT_EQ(snapshot.reason, "model is not ready");
}

TEST(RuntimeStateTest, DrainingStateCannotBecomeReadyAgain) {
  RuntimeState state;
  state.set_backend_ready(true);
  state.mark_running();
  state.begin_draining();
  state.set_backend_ready(true);

  RuntimeHealthSnapshot snapshot = state.health_snapshot();

  EXPECT_EQ(snapshot.phase, RuntimePhase::DRAINING);
  EXPECT_TRUE(snapshot.live);
  EXPECT_FALSE(snapshot.ready);
  EXPECT_EQ(snapshot.reason, "draining");
}

TEST(RuntimeStateTest, StoppedRuntimeIsNotLive) {
  RuntimeState state;
  state.mark_running();
  state.mark_stopped();

  RuntimeHealthSnapshot snapshot = state.health_snapshot();

  EXPECT_EQ(snapshot.phase, RuntimePhase::STOPPED);
  EXPECT_FALSE(snapshot.live);
  EXPECT_FALSE(snapshot.ready);
  EXPECT_EQ(snapshot.reason, "stopped");
}

}  // namespace
}  // namespace xllm_service
