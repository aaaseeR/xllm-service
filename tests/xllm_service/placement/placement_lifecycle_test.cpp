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

#include "placement/placement_lifecycle.h"

#include <gtest/gtest.h>

#include <limits>
#include <string>

namespace xllm_service::placement {
namespace {

PlacementLifecycleRecord absent_record() {
  return PlacementLifecycleRecord{
      .pool =
          PlacementPoolKey{
              .provider_id = xllm::proto::PROVIDER_ID_XLLM_NATIVE,
              .model_revision = "model-r1",
              .role = xllm::proto::ENGINE_ROLE_PREFILL,
              .profile_digest = "profile-a",
          },
  };
}

PlacementLifecycleCommand command(PlacementLifecycleEvent event,
                                  const std::string& operation_id,
                                  uint64_t desired_generation,
                                  uint64_t observed_at_ms,
                                  const std::string& incarnation = "inc-1") {
  return PlacementLifecycleCommand{
      .event = event,
      .engine_uid = "engine-1",
      .engine_incarnation = incarnation,
      .operation_id = operation_id,
      .desired_generation = desired_generation,
      .observed_at_ms = observed_at_ms,
  };
}

PlacementTransitionResult apply(const PlacementLifecycleRecord& record,
                                PlacementLifecycleEvent event,
                                const std::string& operation_id,
                                uint64_t desired_generation,
                                uint64_t observed_at_ms,
                                const std::string& incarnation = "inc-1") {
  return apply_placement_lifecycle_event(record,
                                         command(event,
                                                 operation_id,
                                                 desired_generation,
                                                 observed_at_ms,
                                                 incarnation));
}

PlacementLifecycleRecord ready_record() {
  PlacementLifecycleRecord record = absent_record();
  record =
      apply(
          record, PlacementLifecycleEvent::CREATE_ACCEPTED, "create-1", 1, 1000)
          .record;
  record =
      apply(
          record, PlacementLifecycleEvent::LOAD_COMPLETED, "create-1", 1, 2000)
          .record;
  return apply(record,
               PlacementLifecycleEvent::WARMUP_COMPLETED,
               "create-1",
               1,
               3000)
      .record;
}

TEST(PlacementLifecycleTest, AppliesFullCreateAndDrainLifecycle) {
  PlacementLifecycleRecord record = ready_record();
  ASSERT_EQ(record.state, PlacementLifecycleState::READY);
  EXPECT_EQ(record.state_generation, 3u);

  PlacementTransitionResult result =
      apply(record,
            PlacementLifecycleEvent::BEGIN_DRAIN_ACCEPTED,
            "drain-2",
            2,
            4000);
  ASSERT_EQ(result.status, PlacementTransitionStatus::APPLIED);
  EXPECT_EQ(result.record.state, PlacementLifecycleState::DRAINING);
  EXPECT_FALSE(result.record.drain_committed);

  result = apply(result.record,
                 PlacementLifecycleEvent::DRAIN_COMPLETED,
                 "drain-2",
                 2,
                 5000);
  ASSERT_EQ(result.status, PlacementTransitionStatus::APPLIED);
  EXPECT_EQ(result.record.state, PlacementLifecycleState::UNLOADING);
  EXPECT_TRUE(result.record.drain_committed);

  result = apply(result.record,
                 PlacementLifecycleEvent::TERMINATE_COMPLETED,
                 "drain-2",
                 2,
                 6000);
  EXPECT_EQ(result.status, PlacementTransitionStatus::APPLIED);
  EXPECT_EQ(result.record.state, PlacementLifecycleState::ABSENT);
}

TEST(PlacementLifecycleTest, ReplaysAnyAppliedPhaseWithoutRegression) {
  PlacementLifecycleRecord record = ready_record();
  const PlacementTransitionResult replay = apply(
      record, PlacementLifecycleEvent::CREATE_ACCEPTED, "create-1", 1, 1000);
  EXPECT_EQ(replay.status, PlacementTransitionStatus::REPLAYED);
  EXPECT_EQ(replay.record.state, PlacementLifecycleState::READY);
  EXPECT_EQ(replay.record.state_generation, 3u);

  const PlacementTransitionResult stale_incarnation =
      apply(record,
            PlacementLifecycleEvent::CREATE_ACCEPTED,
            "create-1",
            1,
            1000,
            "stale-inc");
  EXPECT_EQ(stale_incarnation.status, PlacementTransitionStatus::FENCED);
}

TEST(PlacementLifecycleTest, CancelDrainRequiresNewGenerationBeforeCommit) {
  PlacementLifecycleRecord record = ready_record();
  record = apply(record,
                 PlacementLifecycleEvent::BEGIN_DRAIN_ACCEPTED,
                 "drain-2",
                 2,
                 4000)
               .record;
  const PlacementTransitionResult cancel =
      apply(record,
            PlacementLifecycleEvent::CANCEL_DRAIN_ACCEPTED,
            "cancel-3",
            3,
            5000);
  EXPECT_EQ(cancel.status, PlacementTransitionStatus::APPLIED);
  EXPECT_EQ(cancel.record.state, PlacementLifecycleState::READY);
  EXPECT_FALSE(cancel.record.drain_committed);
}

TEST(PlacementLifecycleTest, DrainCommitCannotReturnToReady) {
  PlacementLifecycleRecord record = ready_record();
  record = apply(record,
                 PlacementLifecycleEvent::BEGIN_DRAIN_ACCEPTED,
                 "drain-2",
                 2,
                 4000)
               .record;
  record =
      apply(
          record, PlacementLifecycleEvent::DRAIN_COMPLETED, "drain-2", 2, 5000)
          .record;
  const PlacementTransitionResult cancel =
      apply(record,
            PlacementLifecycleEvent::CANCEL_DRAIN_ACCEPTED,
            "cancel-3",
            3,
            6000);
  EXPECT_EQ(cancel.status, PlacementTransitionStatus::INVALID_TRANSITION);
  EXPECT_EQ(cancel.record.state, PlacementLifecycleState::UNLOADING);
}

TEST(PlacementLifecycleTest, NewCreateRequiresNewIncarnation) {
  PlacementLifecycleRecord record = ready_record();
  record = apply(record,
                 PlacementLifecycleEvent::BEGIN_DRAIN_ACCEPTED,
                 "drain-2",
                 2,
                 4000)
               .record;
  record =
      apply(
          record, PlacementLifecycleEvent::DRAIN_COMPLETED, "drain-2", 2, 5000)
          .record;
  record = apply(record,
                 PlacementLifecycleEvent::TERMINATE_COMPLETED,
                 "drain-2",
                 2,
                 6000)
               .record;

  EXPECT_EQ(
      apply(
          record, PlacementLifecycleEvent::CREATE_ACCEPTED, "create-3", 3, 7000)
          .status,
      PlacementTransitionStatus::FENCED);
  const PlacementTransitionResult recreated =
      apply(record,
            PlacementLifecycleEvent::CREATE_ACCEPTED,
            "create-3",
            3,
            7000,
            "inc-2");
  EXPECT_EQ(recreated.status, PlacementTransitionStatus::APPLIED);
  EXPECT_EQ(recreated.record.state, PlacementLifecycleState::LOADING);
  EXPECT_EQ(recreated.record.engine_incarnation, "inc-2");
}

TEST(PlacementLifecycleTest, FailureRequiresExplicitCleanupProof) {
  PlacementLifecycleRecord record = absent_record();
  record =
      apply(
          record, PlacementLifecycleEvent::CREATE_ACCEPTED, "create-1", 1, 1000)
          .record;
  const PlacementTransitionResult failed = apply(
      record, PlacementLifecycleEvent::OPERATION_FAILED, "create-1", 1, 2000);
  ASSERT_EQ(failed.status, PlacementTransitionStatus::APPLIED);
  EXPECT_EQ(failed.record.state, PlacementLifecycleState::FAILED);

  const PlacementTransitionResult cleaned =
      apply(failed.record,
            PlacementLifecycleEvent::CLEANUP_COMPLETED,
            "cleanup-2",
            2,
            3000);
  EXPECT_EQ(cleaned.status, PlacementTransitionStatus::APPLIED);
  EXPECT_EQ(cleaned.record.state, PlacementLifecycleState::ABSENT);
}

TEST(PlacementLifecycleTest, FencesOldGenerationIncarnationAndConflict) {
  const PlacementLifecycleRecord record = ready_record();
  EXPECT_EQ(apply(record,
                  PlacementLifecycleEvent::BEGIN_DRAIN_ACCEPTED,
                  "drain-old",
                  0,
                  4000)
                .status,
            PlacementTransitionStatus::INVALID_INPUT);
  EXPECT_EQ(apply(record,
                  PlacementLifecycleEvent::BEGIN_DRAIN_ACCEPTED,
                  "drain-old",
                  1,
                  4000)
                .status,
            PlacementTransitionStatus::OPERATION_CONFLICT);
  EXPECT_EQ(apply(record,
                  PlacementLifecycleEvent::BEGIN_DRAIN_ACCEPTED,
                  "drain-2",
                  2,
                  4000,
                  "stale-inc")
                .status,
            PlacementTransitionStatus::FENCED);
}

TEST(PlacementLifecycleTest, RejectsForbiddenTransitionsAndClockRegression) {
  const PlacementLifecycleRecord record = ready_record();
  EXPECT_EQ(apply(record,
                  PlacementLifecycleEvent::TERMINATE_COMPLETED,
                  "terminate-2",
                  2,
                  4000)
                .status,
            PlacementTransitionStatus::OPERATION_CONFLICT);
  EXPECT_EQ(apply(record,
                  PlacementLifecycleEvent::BEGIN_DRAIN_ACCEPTED,
                  "drain-2",
                  2,
                  2000)
                .status,
            PlacementTransitionStatus::CLOCK_REGRESSION);
}

TEST(PlacementLifecycleTest, RejectsGenerationExhaustionAndInvalidRecord) {
  PlacementLifecycleRecord record = ready_record();
  record.state_generation = std::numeric_limits<uint64_t>::max() - 1;
  EXPECT_EQ(apply(record,
                  PlacementLifecycleEvent::BEGIN_DRAIN_ACCEPTED,
                  "drain-2",
                  2,
                  4000)
                .status,
            PlacementTransitionStatus::GENERATION_EXHAUSTED);

  record.state_generation = std::numeric_limits<uint64_t>::max();
  EXPECT_FALSE(valid_placement_lifecycle_record(record));
  EXPECT_EQ(apply(record,
                  PlacementLifecycleEvent::BEGIN_DRAIN_ACCEPTED,
                  "drain-2",
                  2,
                  4000)
                .status,
            PlacementTransitionStatus::INVALID_INPUT);

  record = ready_record();
  record.applied_event_mask |= uint64_t{1} << 63;
  EXPECT_FALSE(valid_placement_lifecycle_record(record));
}

}  // namespace
}  // namespace xllm_service::placement
