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

#include "placement/placement_controller.h"

#include <gtest/gtest.h>

#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "placement/placement_operation_store.h"

namespace xllm_service::placement {
namespace {

class FakeFencedKv final : public PlacementFencedKv {
 public:
  explicit FakeFencedKv(PlacementLeaderIdentity leader)
      : leader_(std::move(leader)) {}

  PlacementStoreStatus read(const std::string& key,
                            std::string* value,
                            int64_t* revision) override {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iterator = values_.find(key);
    if (iterator == values_.end()) {
      return PlacementStoreStatus::NOT_FOUND;
    }
    *value = iterator->second.value;
    *revision = iterator->second.revision;
    return PlacementStoreStatus::OK;
  }

  PlacementStoreStatus list(const std::string& prefix,
                            std::vector<PlacementRawValue>* values) override {
    std::lock_guard<std::mutex> lock(mutex_);
    values->clear();
    for (const auto& [key, stored] : values_) {
      if (key.compare(0, prefix.size(), prefix) == 0) {
        values->emplace_back(PlacementRawValue{
            .key_suffix = key.substr(prefix.size()),
            .value = stored.value,
            .mod_revision = stored.revision,
        });
      }
    }
    return PlacementStoreStatus::OK;
  }

  PlacementStoreStatus compare_and_set(
      const std::string& key,
      const std::string& value,
      int64_t expected_revision,
      const PlacementLeaderIdentity& leader) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!same_leader(leader, leader_)) {
      return PlacementStoreStatus::FENCED;
    }
    const auto iterator = values_.find(key);
    const int64_t revision =
        iterator == values_.end() ? 0 : iterator->second.revision;
    if (revision != expected_revision) {
      return PlacementStoreStatus::REVISION_CONFLICT;
    }
    values_.insert_or_assign(
        key, Stored{.value = value, .revision = next_revision_++});
    ++writes_;
    return PlacementStoreStatus::OK;
  }

  void set_leader(PlacementLeaderIdentity leader) {
    std::lock_guard<std::mutex> lock(mutex_);
    leader_ = std::move(leader);
  }

  void set_desired(const PlacementDesiredState& desired) {
    std::string value;
    ASSERT_TRUE(serialize_placement_desired_state(desired, &value));
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string key = std::string(kPlacementDesiredPrefix) +
                            placement_pool_key_suffix(desired.pool);
    values_.insert_or_assign(
        key, Stored{.value = std::move(value), .revision = next_revision_++});
  }

  uint32_t writes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return writes_;
  }

 private:
  struct Stored {
    std::string value;
    int64_t revision = 0;
  };

  static bool same_leader(const PlacementLeaderIdentity& left,
                          const PlacementLeaderIdentity& right) {
    return left.address == right.address &&
           left.incarnation == right.incarnation && left.epoch == right.epoch;
  }

  mutable std::mutex mutex_;
  PlacementLeaderIdentity leader_;
  std::map<std::string, Stored> values_;
  int64_t next_revision_ = 1;
  uint32_t writes_ = 0;
};

class FakeActuator final : public PlacementActuator {
 public:
  PlacementActuatorResponse execute(
      const PlacementOperationIntent& intent) override {
    ++execute_calls;
    executed_actions.push_back(intent.action);
    if (accept_execute) {
      return PlacementActuatorResponse{
          .code = PlacementActuatorCode::ACCEPTED,
          .lifecycle_state = PlacementLifecycleState::LOADING,
      };
    }
    return success(intent);
  }

  PlacementActuatorResponse query(
      const PlacementOperationIntent& intent) override {
    ++query_calls;
    return success(intent);
  }

  PlacementActuatorResponse success(
      const PlacementOperationIntent& intent) const {
    if (intent.action == PlacementOperationAction::CREATE) {
      return PlacementActuatorResponse{
          .code = PlacementActuatorCode::SUCCEEDED,
          .engine_uid = "created-engine-" + std::to_string(intent.ordinal),
          .engine_incarnation = "created-incarnation",
          .lifecycle_state = PlacementLifecycleState::READY,
          .ready_proven = true,
      };
    }
    return PlacementActuatorResponse{
        .code = PlacementActuatorCode::SUCCEEDED,
        .engine_uid = intent.engine_uid,
        .engine_incarnation = intent.engine_incarnation,
        .lifecycle_state = PlacementLifecycleState::DRAINING,
        .drain = PlacementDrainProof{.admission_closed = true},
    };
  }

  bool accept_execute = false;
  uint32_t execute_calls = 0;
  uint32_t query_calls = 0;
  std::vector<PlacementOperationAction> executed_actions;
};

PlacementLeaderIdentity leader(uint64_t epoch = 17) {
  return PlacementLeaderIdentity{
      .address = "service-1:2888",
      .incarnation = "leader-" + std::to_string(epoch),
      .epoch = epoch,
  };
}

PlacementCapacityProfile profile() {
  return PlacementCapacityProfile{
      .pool =
          PlacementPoolKey{
              .provider_id = xllm::proto::PROVIDER_ID_XLLM_NATIVE,
              .model_revision = "model-r1",
              .role = xllm::proto::ENGINE_ROLE_PREFILL,
              .profile_digest = "profile-a",
          },
      .devices_per_replica = 1,
      .instance_cost_per_hour = 1.0,
      .load_warmup_p99_ms = 1000,
      .prefill_tokens_per_second_under_slo = 100.0,
      .decode_tokens_per_second_under_slo = 100.0,
      .requests_per_second_under_slo = 10.0,
      .target_utilization = 0.8,
      .min_replicas = 1,
      .max_replicas = 4,
      .ttft_slo_ms = 500.0,
      .tpot_slo_ms = 50.0,
  };
}

PlacementObservation observation(uint64_t generation = 1) {
  return PlacementObservation{
      .generation = generation,
      .observed_at_ms = generation * 1000,
      .window_ms = 1000,
      .forecast_horizon_ms = 2000,
      .forecast_request_rate = 0.1,
      .prompt_tokens_per_request = 10.0,
      .output_tokens_per_request = 10.0,
      .ttft_p95_ms = 100.0,
      .tpot_p95_ms = 10.0,
      .kv_used_ratio = 0.1,
      .major_bucket_samples = 100,
  };
}

PlacementReplicaFact ready_replica(uint32_t ordinal) {
  return PlacementReplicaFact{
      .pool = profile().pool,
      .engine_uid = "engine-" + std::to_string(ordinal),
      .engine_incarnation = "incarnation-" + std::to_string(ordinal),
      .state = PlacementLifecycleState::READY,
      .fresh = true,
      .drain_capable = true,
      .stable_since_ms = 1,
      .observed_at_ms = 1000,
  };
}

PlacementPoolCycleInput input(uint64_t generation = 1) {
  return PlacementPoolCycleInput{
      .profile = profile(),
      .observation = observation(generation),
      .priority = 1,
      .slo_risk_score = 0.1,
      .config_digest = "config-a",
  };
}

PlacementControllerConfig controller_config(PlacementMode mode) {
  return PlacementControllerConfig{
      .mode = mode,
      .max_pools = 8,
      .max_desired_snapshot_bytes = 65536,
      .max_operation_snapshot_bytes = 65536,
      .max_devices = 16,
      .max_new_operations_per_cycle = 4,
      .max_actuator_actions_per_cycle = 4,
      .planner =
          PlacementPlannerConfig{
              .scale_up_hold_ms = 0,
              .scale_down_stabilization_ms = 1,
              .cooldown_ms = 1,
              .economic_horizon_ms = 3600000,
              .min_scale_down_samples = 1,
              .max_scale_up_step = 2,
              .max_scale_down_step = 1,
              .queue_high_watermark = 10.0,
              .queue_low_watermark = 1.0,
              .admission_reject_high_watermark = 0.1,
              .admission_reject_low_watermark = 0.01,
              .kv_high_watermark = 0.9,
              .kv_low_watermark = 0.5,
          },
      .reconcile =
          PlacementReconcileConfig{
              .max_operations_per_cycle = 4,
              .max_operations_per_pool = 16,
              .max_create_per_cycle = 4,
              .max_drain_per_cycle = 4,
              .terminal_visibility_grace_ms = 1000,
          },
  };
}

PlacementOperationExecutorConfig executor_config() {
  return PlacementOperationExecutorConfig{
      .max_records = 32,
      .max_message_bytes = 256,
      .operation_timeout_ms = 10000,
      .terminal_retention_ms = 1000,
      .max_terminal_compactions_per_cycle = 4,
  };
}

TEST(PlacementControllerTest, ShadowPlansWithoutPersistingOrActuating) {
  FakeFencedKv backend(leader());
  PlacementDesiredStore desired_store(&backend);
  PlacementOperationStore operation_store(&backend);
  FakeActuator actuator;
  PlacementOperationExecutor executor(
      executor_config(), &actuator, &operation_store, leader());
  PlacementController controller(
      controller_config(PlacementMode::SHADOW), &desired_store, &executor);

  ASSERT_EQ(controller.recover(leader(), 1000), PlacementControllerStatus::OK);
  const PlacementControllerResult result =
      controller.run_cycle(leader(), {input()}, 2000, 100000);

  EXPECT_EQ(result.status, PlacementControllerStatus::OK);
  ASSERT_EQ(result.pools.size(), 1u);
  EXPECT_EQ(result.pools[0].allocation.approved_desired_replicas, 1u);
  EXPECT_EQ(result.desired_writes, 0u);
  EXPECT_EQ(result.intents_added, 0u);
  EXPECT_EQ(backend.writes(), 0u);
  EXPECT_EQ(actuator.execute_calls, 0u);
}

TEST(PlacementControllerTest, EnforcedPersistsReconcilesAndExecutesCreate) {
  FakeFencedKv backend(leader());
  PlacementDesiredStore desired_store(&backend);
  PlacementOperationStore operation_store(&backend);
  FakeActuator actuator;
  PlacementOperationExecutor executor(
      executor_config(), &actuator, &operation_store, leader());
  PlacementController controller(
      controller_config(PlacementMode::ENFORCED), &desired_store, &executor);

  ASSERT_EQ(controller.recover(leader(), 1000), PlacementControllerStatus::OK);
  const PlacementControllerResult result =
      controller.run_cycle(leader(), {input()}, 2000, 100000);

  EXPECT_EQ(result.status, PlacementControllerStatus::OK);
  EXPECT_EQ(result.desired_writes, 1u);
  EXPECT_EQ(result.intents_added, 1u);
  EXPECT_EQ(result.actuator.driven, 1u);
  EXPECT_EQ(actuator.execute_calls, 1u);
  ASSERT_EQ(executor.snapshot().size(), 1u);
  EXPECT_EQ(executor.snapshot()[0].status, PlacementOperationStatus::SUCCEEDED);
}

TEST(PlacementControllerTest, HardDeviceLimitCountsDrainingReplica) {
  FakeFencedKv backend(leader());
  PlacementDesiredStore desired_store(&backend);
  PlacementOperationStore operation_store(&backend);
  FakeActuator actuator;
  PlacementOperationExecutor executor(
      executor_config(), &actuator, &operation_store, leader());
  PlacementControllerConfig config = controller_config(PlacementMode::ENFORCED);
  config.max_devices = 1;
  PlacementController controller(config, &desired_store, &executor);
  PlacementPoolCycleInput cycle_input = input();
  PlacementReplicaFact draining = ready_replica(1);
  draining.state = PlacementLifecycleState::DRAINING;
  cycle_input.replicas = {draining};

  ASSERT_EQ(controller.recover(leader(), 1000), PlacementControllerStatus::OK);
  const PlacementControllerResult result =
      controller.run_cycle(leader(), {cycle_input}, 2000, 100000);

  EXPECT_EQ(result.status, PlacementControllerStatus::HOLD);
  EXPECT_EQ(result.intents_added, 0u);
  EXPECT_EQ(actuator.execute_calls, 0u);
  ASSERT_EQ(result.pools.size(), 1u);
  EXPECT_EQ(result.pools[0].reconcile.reason,
            PlacementReconcileReason::SCALE_UP);
  EXPECT_TRUE(result.pools[0].reconcile.intents.empty());
}

TEST(PlacementControllerTest, CreateOnlyNeverReducesDesiredOrBeginsDrain) {
  FakeFencedKv backend(leader());
  PlacementDesiredStore desired_store(&backend);
  PlacementOperationStore operation_store(&backend);
  FakeActuator actuator;
  PlacementOperationExecutor executor(
      executor_config(), &actuator, &operation_store, leader());
  PlacementController controller(
      controller_config(PlacementMode::ENFORCED_CREATE_ONLY),
      &desired_store,
      &executor);
  PlacementPoolCycleInput first = input();
  first.replicas = {ready_replica(1), ready_replica(2)};

  ASSERT_EQ(controller.recover(leader(), 1000), PlacementControllerStatus::OK);
  ASSERT_EQ(controller.run_cycle(leader(), {first}, 2000, 100000).status,
            PlacementControllerStatus::OK);
  PlacementPoolCycleInput second = input(2);
  second.replicas = first.replicas;
  const PlacementControllerResult result =
      controller.run_cycle(leader(), {second}, 3000, 101000);

  EXPECT_EQ(result.status, PlacementControllerStatus::OK);
  EXPECT_EQ(result.pools[0].allocation.requested_desired_replicas, 1u);
  EXPECT_EQ(result.pools[0].desired_generation, 1u);
  EXPECT_EQ(result.desired_writes, 0u);
  EXPECT_EQ(result.intents_added, 0u);
  EXPECT_EQ(actuator.execute_calls, 0u);
}

TEST(PlacementControllerTest, NewLeaderRecoversAndQueriesWithoutReexecute) {
  FakeFencedKv backend(leader());
  PlacementDesiredStore desired_store(&backend);
  PlacementOperationStore operation_store(&backend);
  FakeActuator first_actuator;
  first_actuator.accept_execute = true;
  PlacementOperationExecutor first_executor(
      executor_config(), &first_actuator, &operation_store, leader());
  PlacementController first_controller(
      controller_config(PlacementMode::ENFORCED),
      &desired_store,
      &first_executor);
  ASSERT_EQ(first_controller.recover(leader(), 1000),
            PlacementControllerStatus::OK);
  ASSERT_EQ(first_controller.run_cycle(leader(), {input()}, 2000, 100000)
                .actuator.driven,
            1u);
  ASSERT_EQ(first_actuator.execute_calls, 1u);

  const PlacementLeaderIdentity next_leader = leader(18);
  backend.set_leader(next_leader);
  FakeActuator recovered_actuator;
  PlacementOperationExecutor recovered_executor(
      executor_config(), &recovered_actuator, &operation_store, next_leader);
  PlacementController recovered_controller(
      controller_config(PlacementMode::ENFORCED),
      &desired_store,
      &recovered_executor);
  ASSERT_EQ(recovered_controller.recover(next_leader, 100),
            PlacementControllerStatus::OK);
  PlacementPoolCycleInput recovered_input = input(2);
  recovered_input.observation.observed_at_ms = 100;
  const PlacementControllerResult result = recovered_controller.run_cycle(
      next_leader, {recovered_input}, 101, 101000);

  EXPECT_EQ(result.status, PlacementControllerStatus::HOLD);
  EXPECT_EQ(result.desired_writes, 1u);
  EXPECT_EQ(recovered_actuator.execute_calls, 0u);
  EXPECT_EQ(recovered_actuator.query_calls, 1u);
  ASSERT_EQ(recovered_executor.snapshot().size(), 1u);
  EXPECT_EQ(recovered_executor.snapshot()[0].status,
            PlacementOperationStatus::SUCCEEDED);
}

TEST(PlacementControllerTest, UnknownExternalDesiredIsAdoptedBeforeAnyWrite) {
  FakeFencedKv backend(leader());
  PlacementDesiredStore desired_store(&backend);
  PlacementOperationStore operation_store(&backend);
  FakeActuator actuator;
  PlacementOperationExecutor executor(
      executor_config(), &actuator, &operation_store, leader());
  PlacementController controller(
      controller_config(PlacementMode::ENFORCED), &desired_store, &executor);
  ASSERT_EQ(controller.recover(leader(), 1000), PlacementControllerStatus::OK);
  backend.set_desired(PlacementDesiredState{
      .leader = leader(),
      .generation = 7,
      .pool = profile().pool,
      .desired_replicas = 2,
      .reason = PlacementReason::FORECAST_CAPACITY,
      .observation_generation = 7,
      .created_at_unix_ms = 99999,
      .config_digest = "external-config",
  });
  const uint32_t writes_before = backend.writes();

  const PlacementControllerResult result =
      controller.run_cycle(leader(), {input()}, 2000, 100000);

  EXPECT_EQ(result.status, PlacementControllerStatus::HOLD);
  EXPECT_EQ(result.desired_writes, 0u);
  EXPECT_EQ(result.intents_added, 0u);
  EXPECT_EQ(backend.writes(), writes_before);
  EXPECT_EQ(actuator.execute_calls, 0u);
}

}  // namespace
}  // namespace xllm_service::placement
