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

#include "placement/placement_actuator.h"

#include <gtest/gtest.h>

#include <deque>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace xllm_service::placement {
namespace {

class FakePlacementActuator final : public PlacementActuator {
 public:
  PlacementActuatorResponse execute(
      const PlacementOperationIntent& intent) override {
    ++execute_calls[intent.operation_id];
    return pop(intent.operation_id, &execute_responses);
  }

  PlacementActuatorResponse query(
      const PlacementOperationIntent& intent) override {
    ++query_calls[intent.operation_id];
    return pop(intent.operation_id, &query_responses);
  }

  std::map<std::string, std::deque<PlacementActuatorResponse>>
      execute_responses;
  std::map<std::string, std::deque<PlacementActuatorResponse>> query_responses;
  std::map<std::string, uint32_t> execute_calls;
  std::map<std::string, uint32_t> query_calls;

 private:
  static PlacementActuatorResponse pop(
      const std::string& operation_id,
      std::map<std::string, std::deque<PlacementActuatorResponse>>* responses) {
    auto iterator = responses->find(operation_id);
    if (iterator == responses->end() || iterator->second.empty()) {
      return PlacementActuatorResponse{
          .code = PlacementActuatorCode::UNKNOWN,
      };
    }
    PlacementActuatorResponse response = std::move(iterator->second.front());
    iterator->second.pop_front();
    return response;
  }
};

PlacementLeaderIdentity leader() {
  return PlacementLeaderIdentity{
      .address = "service-1:2888",
      .incarnation = "leader-1",
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

PlacementOperationIntent intent(
    PlacementOperationAction action = PlacementOperationAction::CREATE,
    uint32_t ordinal = 0) {
  const std::string engine_uid =
      action == PlacementOperationAction::CREATE ? "" : "engine-1";
  const std::string engine_incarnation =
      action == PlacementOperationAction::CREATE ? "" : "inc-1";
  return PlacementOperationIntent{
      .operation_id = make_placement_operation_id(leader(),
                                                  1,
                                                  pool(),
                                                  action,
                                                  ordinal,
                                                  engine_uid,
                                                  engine_incarnation),
      .action = action,
      .pool = pool(),
      .engine_uid = engine_uid,
      .engine_incarnation = engine_incarnation,
      .leader_incarnation = leader().incarnation,
      .desired_generation = 1,
      .ordinal = ordinal,
  };
}

PlacementOperationExecutorConfig config(size_t max_records = 8) {
  return PlacementOperationExecutorConfig{
      .max_records = max_records,
      .max_message_bytes = 64,
      .operation_timeout_ms = 1000,
  };
}

PlacementActuatorResponse create_success() {
  return PlacementActuatorResponse{
      .code = PlacementActuatorCode::SUCCEEDED,
      .engine_uid = "engine-new",
      .engine_incarnation = "inc-new",
      .lifecycle_state = PlacementLifecycleState::READY,
      .ready_proven = true,
  };
}

PlacementActuatorResponse drain_response(PlacementActuatorCode code,
                                         uint64_t pending_output = 0) {
  return PlacementActuatorResponse{
      .code = code,
      .engine_uid = "engine-1",
      .engine_incarnation = "inc-1",
      .lifecycle_state = PlacementLifecycleState::DRAINING,
      .drain =
          PlacementDrainProof{
              .admission_closed = true,
              .pending_output = pending_output,
          },
  };
}

TEST(PlacementOperationExecutorTest, UnknownExecuteIsOnlyQueriedAfterward) {
  FakePlacementActuator actuator;
  PlacementOperationExecutor executor(config(), &actuator);
  const PlacementOperationIntent create = intent();
  actuator.execute_responses[create.operation_id].push_back(
      PlacementActuatorResponse{.code = PlacementActuatorCode::UNKNOWN});
  actuator.query_responses[create.operation_id].push_back(
      PlacementActuatorResponse{
          .code = PlacementActuatorCode::IN_PROGRESS,
          .engine_uid = "engine-new",
          .engine_incarnation = "inc-new",
          .lifecycle_state = PlacementLifecycleState::LOADING,
      });
  actuator.query_responses[create.operation_id].push_back(create_success());

  ASSERT_EQ(executor.add_intents({create}, 1000).status,
            PlacementExecutorStatus::OK);
  EXPECT_EQ(executor.drive(1100, 1).unknown, 1u);
  EXPECT_EQ(executor.drive(1200, 1).unknown, 0u);
  EXPECT_EQ(executor.drive(1300, 1).terminal, 1u);
  EXPECT_EQ(actuator.execute_calls[create.operation_id], 1u);
  EXPECT_EQ(actuator.query_calls[create.operation_id], 2u);
  const std::vector<PlacementOperationRecord> snapshot = executor.snapshot();
  ASSERT_EQ(snapshot.size(), 1u);
  EXPECT_EQ(snapshot[0].status, PlacementOperationStatus::SUCCEEDED);
  EXPECT_EQ(snapshot[0].engine_uid, "engine-new");
}

TEST(PlacementOperationExecutorTest, DrainRequiresCompleteTerminalProof) {
  FakePlacementActuator actuator;
  PlacementOperationExecutor executor(config(), &actuator);
  const PlacementOperationIntent drain =
      intent(PlacementOperationAction::BEGIN_DRAIN);
  actuator.execute_responses[drain.operation_id].push_back(
      drain_response(PlacementActuatorCode::SUCCEEDED,
                     /*pending_output=*/1));
  ASSERT_EQ(executor.add_intents({drain}, 1000).added, 1u);
  ASSERT_EQ(executor.drive(1100, 1).terminal, 1u);
  EXPECT_EQ(executor.snapshot()[0].status, PlacementOperationStatus::FAILED);
  EXPECT_EQ(executor.snapshot()[0].last_code, PlacementActuatorCode::INVALID);
}

TEST(PlacementOperationExecutorTest, CompleteDrainAndTerminationProofSucceed) {
  FakePlacementActuator actuator;
  PlacementOperationExecutor executor(config(), &actuator);
  const PlacementOperationIntent drain =
      intent(PlacementOperationAction::BEGIN_DRAIN);
  actuator.execute_responses[drain.operation_id].push_back(
      drain_response(PlacementActuatorCode::SUCCEEDED));
  ASSERT_EQ(executor.add_intents({drain}, 1000).added, 1u);
  EXPECT_EQ(executor.drive(1100, 1).terminal, 1u);
  EXPECT_EQ(executor.snapshot()[0].status,
            PlacementOperationStatus::SUCCEEDED);

  FakePlacementActuator terminate_actuator;
  PlacementOperationExecutor terminate_executor(config(),
                                                 &terminate_actuator);
  const PlacementOperationIntent terminate =
      intent(PlacementOperationAction::TERMINATE);
  terminate_actuator.execute_responses[terminate.operation_id].push_back(
      PlacementActuatorResponse{
          .code = PlacementActuatorCode::SUCCEEDED,
          .engine_uid = "engine-1",
          .engine_incarnation = "inc-1",
          .lifecycle_state = PlacementLifecycleState::ABSENT,
          .termination_proven = true,
      });
  ASSERT_EQ(terminate_executor.add_intents({terminate}, 1000).added, 1u);
  EXPECT_EQ(terminate_executor.drive(1100, 1).terminal, 1u);
  EXPECT_EQ(terminate_executor.snapshot()[0].status,
            PlacementOperationStatus::SUCCEEDED);
}

TEST(PlacementOperationExecutorTest, RejectsMissingProofAndWrongIncarnation) {
  FakePlacementActuator actuator;
  PlacementOperationExecutor executor(config(), &actuator);
  const PlacementOperationIntent create = intent();
  PlacementActuatorResponse response = create_success();
  response.ready_proven = false;
  actuator.execute_responses[create.operation_id].push_back(response);
  ASSERT_EQ(executor.add_intents({create}, 1000).added, 1u);
  executor.drive(1100, 1);
  EXPECT_EQ(executor.snapshot()[0].status, PlacementOperationStatus::FAILED);

  FakePlacementActuator drain_actuator;
  PlacementOperationExecutor drain_executor(config(), &drain_actuator);
  const PlacementOperationIntent drain =
      intent(PlacementOperationAction::BEGIN_DRAIN);
  response = drain_response(PlacementActuatorCode::ACCEPTED);
  response.engine_incarnation = "stale-inc";
  drain_actuator.execute_responses[drain.operation_id].push_back(response);
  ASSERT_EQ(drain_executor.add_intents({drain}, 1000).added, 1u);
  drain_executor.drive(1100, 1);
  EXPECT_EQ(drain_executor.snapshot()[0].status,
            PlacementOperationStatus::FAILED);
}

TEST(PlacementOperationExecutorTest, IsBoundedIdempotentAndFenced) {
  FakePlacementActuator actuator;
  PlacementOperationExecutor executor(config(/*max_records=*/1), &actuator);
  const PlacementOperationIntent first = intent();
  const PlacementOperationIntent second = intent(
      PlacementOperationAction::CREATE, /*ordinal=*/1);
  PlacementExecutorResult result =
      executor.add_intents({first, first}, 1000);
  EXPECT_EQ(result.added, 1u);
  EXPECT_EQ(result.replayed, 1u);
  EXPECT_EQ(executor.add_intents({second}, 1000).status,
            PlacementExecutorStatus::CAPACITY_EXCEEDED);

  actuator.execute_responses[first.operation_id].push_back(
      PlacementActuatorResponse{.code = PlacementActuatorCode::FENCED});
  EXPECT_EQ(executor.drive(1100, 1).terminal, 1u);
  EXPECT_EQ(executor.snapshot()[0].status, PlacementOperationStatus::FENCED);
}

TEST(PlacementOperationExecutorTest, TimeoutStaysQueryableAndClockFailsClosed) {
  FakePlacementActuator actuator;
  PlacementOperationExecutor executor(config(), &actuator);
  const PlacementOperationIntent create = intent();
  actuator.execute_responses[create.operation_id].push_back(
      PlacementActuatorResponse{.code = PlacementActuatorCode::ACCEPTED});
  actuator.query_responses[create.operation_id].push_back(
      PlacementActuatorResponse{.code = PlacementActuatorCode::UNKNOWN});
  ASSERT_EQ(executor.add_intents({create}, 1000).added, 1u);
  executor.drive(1100, 1);
  EXPECT_EQ(executor.drive(2500, 1).unknown, 1u);
  EXPECT_TRUE(executor.snapshot()[0].timed_out);
  EXPECT_EQ(actuator.execute_calls[create.operation_id], 1u);
  EXPECT_EQ(actuator.query_calls[create.operation_id], 1u);
  EXPECT_EQ(executor.drive(900, 1).status,
            PlacementExecutorStatus::CLOCK_REGRESSION);
}

}  // namespace
}  // namespace xllm_service::placement
