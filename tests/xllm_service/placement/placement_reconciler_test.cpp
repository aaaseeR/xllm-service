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

#include "placement/placement_reconciler.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace xllm_service::placement {
namespace {

PlacementReconcileConfig config() {
  return PlacementReconcileConfig{
      .max_operations_per_cycle = 4,
      .max_operations_per_pool = 16,
      .max_create_per_cycle = 2,
      .max_drain_per_cycle = 2,
  };
}

PlacementLeaderIdentity leader() {
  return PlacementLeaderIdentity{
      .address = "service-1:2888",
      .incarnation = "leader-1",
      .epoch = 10,
  };
}

PlacementPoolKey pool() {
  return PlacementPoolKey{
      .provider_id = xllm::proto::PROVIDER_ID_XLLM_NATIVE,
      .model_revision = "model-r1",
      .role = xllm::proto::ENGINE_ROLE_PREFILL,
      .profile_digest = "profile-a",
  };
}

PlacementDesiredState desired(uint32_t replicas, uint64_t generation = 1) {
  return PlacementDesiredState{
      .leader = leader(),
      .generation = generation,
      .pool = pool(),
      .desired_replicas = replicas,
      .reason = PlacementReason::FORECAST_CAPACITY,
      .observation_generation = generation,
      .created_at_unix_ms = 1000,
      .config_digest = "config-a",
  };
}

PlacementReplicaFact replica(const std::string& uid,
                             PlacementLifecycleState state,
                             double cache_value = 1.0) {
  return PlacementReplicaFact{
      .pool = pool(),
      .engine_uid = uid,
      .engine_incarnation = uid + "-inc",
      .state = state,
      .fresh = true,
      .drain_capable = true,
      .cache_value = cache_value,
      .stable_since_ms = 1000,
      .observed_at_ms = 2000,
  };
}

PlacementOperationView operation(const std::string& operation_id,
                                 PlacementOperationAction action,
                                 PlacementOperationStatus status) {
  return PlacementOperationView{
      .operation_id = operation_id,
      .action = action,
      .status = status,
      .pool = pool(),
      .engine_uid =
          action == PlacementOperationAction::CREATE ? "" : "engine-1",
      .engine_incarnation =
          action == PlacementOperationAction::CREATE ? "" : "engine-1-inc",
      .leader_incarnation = "leader-1",
      .leader_epoch = 10,
      .desired_generation = 1,
  };
}

TEST(PlacementReconcilerTest, CreatesBoundedDeterministicDeficit) {
  const std::vector<PlacementReplicaFact> replicas = {
      replica("engine-1", PlacementLifecycleState::READY),
  };
  const PlacementReconcileResult first = reconcile_placement_pool(
      config(), desired(4), leader(), replicas, {}, /*now_ms=*/3000);
  const PlacementReconcileResult second = reconcile_placement_pool(
      config(), desired(4), leader(), replicas, {}, /*now_ms=*/3000);
  ASSERT_EQ(first.status, PlacementReconcileStatus::OK);
  ASSERT_EQ(first.reason, PlacementReconcileReason::SCALE_UP);
  ASSERT_EQ(first.intents.size(), 2u);
  EXPECT_EQ(first.managed_replicas, 1u);
  EXPECT_EQ(first.intents[0].action, PlacementOperationAction::CREATE);
  EXPECT_EQ(first.intents[0].operation_id, second.intents[0].operation_id);
  EXPECT_NE(first.intents[0].operation_id, first.intents[1].operation_id);
}

TEST(PlacementReconcilerTest, UnknownCacheValueCannotBecomeDrainVictim) {
  std::vector<PlacementReplicaFact> replicas = {
      replica("engine-unknown", PlacementLifecycleState::READY, 0.0),
      replica("engine-known", PlacementLifecycleState::READY, 10.0),
  };
  replicas[0].cache_value_known = false;
  const PlacementReconcileResult result = reconcile_placement_pool(
      config(), desired(1), leader(), replicas, {}, /*now_ms=*/3000);
  ASSERT_EQ(result.status, PlacementReconcileStatus::OK);
  ASSERT_EQ(result.reason, PlacementReconcileReason::SCALE_DOWN);
  ASSERT_EQ(result.intents.size(), 1u);
  EXPECT_EQ(result.intents[0].engine_uid, "engine-known");
}

TEST(PlacementReconcilerTest, LoadingAndWarmingCountTowardDesired) {
  const std::vector<PlacementReplicaFact> replicas = {
      replica("engine-1", PlacementLifecycleState::READY),
      replica("engine-2", PlacementLifecycleState::LOADING),
      replica("engine-3", PlacementLifecycleState::WARMING),
  };
  const PlacementReconcileResult result = reconcile_placement_pool(
      config(), desired(3), leader(), replicas, {}, /*now_ms=*/3000);
  EXPECT_EQ(result.reason, PlacementReconcileReason::CONVERGED);
  EXPECT_TRUE(result.intents.empty());
}

TEST(PlacementReconcilerTest, PendingOperationBlocksNewSideEffects) {
  std::vector<PlacementOperationView> operations = {
      operation("create-1",
                PlacementOperationAction::CREATE,
                PlacementOperationStatus::UNKNOWN),
  };
  const PlacementReconcileResult result = reconcile_placement_pool(
      config(), desired(3), leader(), {}, operations, /*now_ms=*/3000);
  EXPECT_EQ(result.status, PlacementReconcileStatus::HOLD);
  EXPECT_EQ(result.reason, PlacementReconcileReason::PENDING_OPERATION);
  EXPECT_EQ(result.managed_replicas, 1u);
  EXPECT_TRUE(result.intents.empty());

  PlacementReplicaFact loading =
      replica("engine-new", PlacementLifecycleState::LOADING);
  operations.front().engine_uid = loading.engine_uid;
  operations.front().engine_incarnation = loading.engine_incarnation;
  const PlacementReconcileResult observed = reconcile_placement_pool(
      config(), desired(3), leader(), {loading}, operations, /*now_ms=*/3000);
  EXPECT_EQ(observed.managed_replicas, 1u);
}

TEST(PlacementReconcilerTest, SelectsLowestCacheStableSafeVictim) {
  PlacementReplicaFact unsafe =
      replica("engine-unsafe", PlacementLifecycleState::READY, 0.0);
  unsafe.active_transfers = 1;
  PlacementReplicaFact high =
      replica("engine-high", PlacementLifecycleState::READY, 10.0);
  PlacementReplicaFact low =
      replica("engine-low", PlacementLifecycleState::READY, 1.0);
  const PlacementReconcileResult result =
      reconcile_placement_pool(config(),
                               desired(1, 2),
                               leader(),
                               {unsafe, high, low},
                               {},
                               /*now_ms=*/3000);
  ASSERT_EQ(result.reason, PlacementReconcileReason::SCALE_DOWN);
  ASSERT_EQ(result.intents.size(), 2u);
  EXPECT_EQ(result.intents[0].engine_uid, "engine-low");
  EXPECT_EQ(result.intents[1].engine_uid, "engine-high");
}

TEST(PlacementReconcilerTest, CommittedDrainAdvancesToTerminate) {
  PlacementOperationView drained =
      operation("drain-1",
                PlacementOperationAction::BEGIN_DRAIN,
                PlacementOperationStatus::SUCCEEDED);
  PlacementReplicaFact draining =
      replica("engine-1", PlacementLifecycleState::DRAINING);
  const PlacementReconcileResult result = reconcile_placement_pool(
      config(),
      desired(1, 2),
      leader(),
      {draining, replica("engine-2", PlacementLifecycleState::READY)},
      {drained},
      /*now_ms=*/3000);
  ASSERT_EQ(result.reason, PlacementReconcileReason::TERMINATE_DRAINED);
  ASSERT_EQ(result.intents.size(), 1u);
  EXPECT_EQ(result.intents[0].action, PlacementOperationAction::TERMINATE);
  EXPECT_EQ(result.intents[0].engine_uid, "engine-1");
  EXPECT_EQ(result.intents[0].leader_epoch, leader().epoch);
}

TEST(PlacementReconcilerTest, NewGenerationCanCancelUncommittedDrain) {
  PlacementOperationView draining_operation =
      operation("drain-pending",
                PlacementOperationAction::BEGIN_DRAIN,
                PlacementOperationStatus::IN_PROGRESS);
  PlacementReplicaFact draining =
      replica("engine-1", PlacementLifecycleState::DRAINING);
  const PlacementReconcileResult result = reconcile_placement_pool(
      config(),
      desired(2, 2),
      leader(),
      {draining, replica("engine-2", PlacementLifecycleState::READY)},
      {draining_operation},
      /*now_ms=*/3000);
  ASSERT_EQ(result.reason, PlacementReconcileReason::CANCEL_DRAIN);
  ASSERT_EQ(result.intents.size(), 1u);
  EXPECT_EQ(result.intents[0].action, PlacementOperationAction::CANCEL_DRAIN);
  EXPECT_EQ(result.intents[0].desired_generation, 2u);
}

TEST(PlacementReconcilerTest, CurrentGenerationTerminalFailureHolds) {
  const PlacementOperationView failed =
      operation("create-failed",
                PlacementOperationAction::CREATE,
                PlacementOperationStatus::FAILED);
  const PlacementReconcileResult result = reconcile_placement_pool(
      config(), desired(2), leader(), {}, {failed}, /*now_ms=*/3000);
  EXPECT_EQ(result.status, PlacementReconcileStatus::HOLD);
  EXPECT_EQ(result.reason, PlacementReconcileReason::TERMINAL_OPERATION);
  EXPECT_TRUE(result.intents.empty());
}

TEST(PlacementReconcilerTest, HoldsWhenNoVictimHasDrainProofBoundary) {
  PlacementReplicaFact value =
      replica("engine-1", PlacementLifecycleState::READY);
  value.drain_capable = false;
  const PlacementReconcileResult result = reconcile_placement_pool(
      config(),
      desired(1),
      leader(),
      {value, replica("engine-2", PlacementLifecycleState::READY)},
      {},
      /*now_ms=*/3000);
  ASSERT_EQ(result.reason, PlacementReconcileReason::SCALE_DOWN);
  ASSERT_EQ(result.intents.size(), 1u);

  PlacementReplicaFact second =
      replica("engine-2", PlacementLifecycleState::READY);
  second.active_reservations = 1;
  const PlacementReconcileResult blocked = reconcile_placement_pool(
      config(), desired(1), leader(), {value, second}, {}, /*now_ms=*/3000);
  EXPECT_EQ(blocked.status, PlacementReconcileStatus::HOLD);
  EXPECT_EQ(blocked.reason, PlacementReconcileReason::NO_SAFE_VICTIM);
}

TEST(PlacementReconcilerTest, FailsClosedOnInvalidActualAndOperation) {
  PlacementReplicaFact future =
      replica("engine-1", PlacementLifecycleState::READY);
  future.observed_at_ms = 4000;
  EXPECT_EQ(reconcile_placement_pool(
                config(), desired(1), leader(), {future}, {}, /*now_ms=*/3000)
                .reason,
            PlacementReconcileReason::INVALID_ACTUAL);

  std::vector<PlacementOperationView> duplicate = {
      operation("same",
                PlacementOperationAction::CREATE,
                PlacementOperationStatus::SUCCEEDED),
      operation("same",
                PlacementOperationAction::CREATE,
                PlacementOperationStatus::FAILED),
  };
  EXPECT_EQ(reconcile_placement_pool(
                config(), desired(1), leader(), {}, duplicate, /*now_ms=*/3000)
                .reason,
            PlacementReconcileReason::INVALID_OPERATION);
}

TEST(PlacementReconcilerTest, OperationIdFencesGenerationActionAndEngine) {
  const std::string base =
      make_placement_operation_id(leader(),
                                  1,
                                  pool(),
                                  PlacementOperationAction::BEGIN_DRAIN,
                                  0,
                                  "engine-1",
                                  "inc-1");
  ASSERT_FALSE(base.empty());
  EXPECT_NE(base,
            make_placement_operation_id(leader(),
                                        2,
                                        pool(),
                                        PlacementOperationAction::BEGIN_DRAIN,
                                        0,
                                        "engine-1",
                                        "inc-1"));
  EXPECT_NE(base,
            make_placement_operation_id(leader(),
                                        1,
                                        pool(),
                                        PlacementOperationAction::TERMINATE,
                                        0,
                                        "engine-1",
                                        "inc-1"));
  EXPECT_NE(base,
            make_placement_operation_id(leader(),
                                        1,
                                        pool(),
                                        PlacementOperationAction::BEGIN_DRAIN,
                                        0,
                                        "engine-1",
                                        "inc-2"));
  PlacementLeaderIdentity next_epoch = leader();
  ++next_epoch.epoch;
  EXPECT_NE(base,
            make_placement_operation_id(next_epoch,
                                        1,
                                        pool(),
                                        PlacementOperationAction::BEGIN_DRAIN,
                                        0,
                                        "engine-1",
                                        "inc-1"));
}

}  // namespace
}  // namespace xllm_service::placement
