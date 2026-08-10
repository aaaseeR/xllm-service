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

#include "placement/placement_operation_store.h"

#include <gtest/gtest.h>

#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

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
    if (leader.address != leader_.address ||
        leader.incarnation != leader_.incarnation ||
        leader.epoch != leader_.epoch) {
      return PlacementStoreStatus::FENCED;
    }
    const auto iterator = values_.find(key);
    const int64_t revision =
        iterator == values_.end() ? 0 : iterator->second.revision;
    if (revision != expected_revision) {
      if (iterator != values_.end() && iterator->second.value == value) {
        return PlacementStoreStatus::OK;
      }
      return PlacementStoreStatus::REVISION_CONFLICT;
    }
    values_.insert_or_assign(
        key, Stored{.value = value, .revision = next_revision_++});
    return PlacementStoreStatus::OK;
  }

  PlacementStoreStatus compare_and_delete_pair(
      const std::string& first_key,
      int64_t first_expected_revision,
      const std::string& second_key,
      int64_t second_expected_revision,
      const PlacementLeaderIdentity& leader) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (leader.address != leader_.address ||
        leader.incarnation != leader_.incarnation ||
        leader.epoch != leader_.epoch) {
      return PlacementStoreStatus::FENCED;
    }
    const auto first = values_.find(first_key);
    const auto second = values_.find(second_key);
    if (first == values_.end() && second == values_.end()) {
      return PlacementStoreStatus::OK;
    }
    if (first == values_.end() || second == values_.end() ||
        first->second.revision != first_expected_revision ||
        second->second.revision != second_expected_revision) {
      return PlacementStoreStatus::REVISION_CONFLICT;
    }
    values_.erase(first);
    values_.erase(second);
    return PlacementStoreStatus::OK;
  }

  void set_raw(std::string key, std::string value) {
    std::lock_guard<std::mutex> lock(mutex_);
    values_.insert_or_assign(
        std::move(key),
        Stored{.value = std::move(value), .revision = next_revision_++});
  }

 private:
  struct Stored {
    std::string value;
    int64_t revision = 0;
  };

  std::mutex mutex_;
  PlacementLeaderIdentity leader_;
  std::map<std::string, Stored> values_;
  int64_t next_revision_ = 1;
};

class RecordingActuator final : public PlacementActuator {
 public:
  PlacementActuatorResponse execute(
      const PlacementOperationIntent& intent) override {
    ++execute_calls;
    return PlacementActuatorResponse{
        .code = PlacementActuatorCode::ACCEPTED,
        .engine_uid = intent.engine_uid,
        .engine_incarnation = intent.engine_incarnation,
        .lifecycle_state = PlacementLifecycleState::DRAINING,
        .drain = PlacementDrainProof{.admission_closed = true},
    };
  }

  PlacementActuatorResponse query(
      const PlacementOperationIntent& intent) override {
    ++query_calls;
    return PlacementActuatorResponse{
        .code = PlacementActuatorCode::SUCCEEDED,
        .engine_uid = intent.engine_uid,
        .engine_incarnation = intent.engine_incarnation,
        .lifecycle_state = PlacementLifecycleState::DRAINING,
        .drain = PlacementDrainProof{.admission_closed = true},
    };
  }

  uint32_t execute_calls = 0;
  uint32_t query_calls = 0;
};

PlacementLeaderIdentity leader() {
  return PlacementLeaderIdentity{
      .address = "service-1:2888",
      .incarnation = "leader-1",
      .epoch = 17,
  };
}

PlacementPoolKey pool() {
  return PlacementPoolKey{
      .provider_id = xllm::proto::PROVIDER_ID_XLLM_NATIVE,
      .model_revision = "model-r1",
      .role = xllm::proto::ENGINE_ROLE_DECODE,
      .profile_digest = "profile-a",
  };
}

PlacementOperationIntent drain_intent() {
  return PlacementOperationIntent{
      .operation_id =
          make_placement_operation_id(leader(),
                                      3,
                                      pool(),
                                      PlacementOperationAction::BEGIN_DRAIN,
                                      0,
                                      "engine-1",
                                      "inc-1"),
      .action = PlacementOperationAction::BEGIN_DRAIN,
      .pool = pool(),
      .engine_uid = "engine-1",
      .engine_incarnation = "inc-1",
      .leader_incarnation = leader().incarnation,
      .leader_epoch = leader().epoch,
      .desired_generation = 3,
  };
}

PlacementOperationExecutorConfig executor_config() {
  return PlacementOperationExecutorConfig{
      .max_records = 8,
      .max_message_bytes = 128,
      .operation_timeout_ms = 10000,
      .terminal_retention_ms = 1000,
      .max_terminal_compactions_per_cycle = 4,
  };
}

TEST(PlacementOperationStoreTest, StrictIntentAndStatusRoundtrip) {
  const PlacementOperationIntent intent = drain_intent();
  std::string value;
  ASSERT_TRUE(serialize_placement_operation_intent(intent, &value));
  PlacementOperationIntent parsed_intent;
  ASSERT_TRUE(parse_placement_operation_intent(value, &parsed_intent));
  EXPECT_TRUE(placement_operation_intents_equal(intent, parsed_intent));

  PlacementOperationRecord record{
      .intent = intent,
      .status = PlacementOperationStatus::UNKNOWN,
      .last_code = PlacementActuatorCode::UNKNOWN,
      .execute_attempts = 1,
      .created_at_ms = 1000,
      .updated_at_ms = 1100,
      .message = "query required",
  };
  ASSERT_TRUE(serialize_placement_operation_record(record, &value));
  PlacementOperationRecord parsed_record;
  ASSERT_TRUE(parse_placement_operation_record(value, &parsed_record));
  EXPECT_EQ(parsed_record.status, PlacementOperationStatus::UNKNOWN);
  EXPECT_EQ(parsed_record.execute_attempts, 1u);

  record.status = PlacementOperationStatus::CANCELED;
  record.last_code = PlacementActuatorCode::SUCCEEDED;
  record.message = "superseded by successful cancel drain";
  ASSERT_TRUE(serialize_placement_operation_record(record, &value));
  ASSERT_TRUE(parse_placement_operation_record(value, &parsed_record));
  EXPECT_EQ(parsed_record.status, PlacementOperationStatus::CANCELED);

  record.last_code = PlacementActuatorCode::UNKNOWN;
  EXPECT_FALSE(serialize_placement_operation_record(record, &value));
}

TEST(PlacementOperationStoreTest, MissingStatusRecoversAsUnknown) {
  FakeFencedKv backend(leader());
  PlacementOperationStore store(&backend);
  int64_t revision = 0;
  ASSERT_EQ(store.create_command(drain_intent(), leader(), &revision),
            PlacementStoreStatus::OK);
  ASSERT_GT(revision, 0);
  int64_t replayed_revision = 0;
  ASSERT_EQ(store.create_command(drain_intent(), leader(), &replayed_revision),
            PlacementStoreStatus::OK);
  EXPECT_EQ(replayed_revision, revision);

  std::vector<PlacementPersistedOperation> snapshot;
  ASSERT_EQ(store.load_snapshot(8, 65536, &snapshot), PlacementStoreStatus::OK);
  ASSERT_EQ(snapshot.size(), 1u);
  EXPECT_EQ(snapshot[0].record.status, PlacementOperationStatus::UNKNOWN);
  EXPECT_EQ(snapshot[0].record.execute_attempts, 1u);
  EXPECT_EQ(snapshot[0].status_revision, 0);
}

TEST(PlacementOperationStoreTest, DeletesTerminalPairAtomicallyAndFenced) {
  FakeFencedKv backend(leader());
  PlacementOperationStore store(&backend);
  PlacementOperationRecord record{
      .intent = drain_intent(),
      .status = PlacementOperationStatus::SUCCEEDED,
      .last_code = PlacementActuatorCode::SUCCEEDED,
      .engine_uid = "engine-1",
      .engine_incarnation = "inc-1",
      .created_at_ms = 1000,
      .updated_at_ms = 1100,
  };
  ASSERT_EQ(
      store.create_command(record.intent, leader(), &record.command_revision),
      PlacementStoreStatus::OK);
  ASSERT_EQ(store.compare_and_set_status(
                record, 0, leader(), &record.status_revision),
            PlacementStoreStatus::OK);

  PlacementLeaderIdentity stale = leader();
  --stale.epoch;
  EXPECT_EQ(store.delete_terminal(record, stale), PlacementStoreStatus::FENCED);
  ASSERT_EQ(store.delete_terminal(record, leader()), PlacementStoreStatus::OK);
  EXPECT_EQ(store.delete_terminal(record, leader()), PlacementStoreStatus::OK);
  std::vector<PlacementPersistedOperation> snapshot;
  ASSERT_EQ(store.load_snapshot(8, 65536, &snapshot), PlacementStoreStatus::OK);
  EXPECT_TRUE(snapshot.empty());
}

TEST(PlacementOperationStoreTest, DurableRecoveryQueriesWithoutReexecute) {
  FakeFencedKv backend(leader());
  PlacementOperationStore store(&backend);
  RecordingActuator first_actuator;
  PlacementOperationExecutor first(
      executor_config(), &first_actuator, &store, leader());
  ASSERT_EQ(first.add_intents({drain_intent()}, 1000).status,
            PlacementExecutorStatus::OK);
  ASSERT_EQ(first.drive(1100, 1).status, PlacementExecutorStatus::OK);
  EXPECT_EQ(first_actuator.execute_calls, 1u);

  RecordingActuator recovered_actuator;
  PlacementOperationExecutor recovered(
      executor_config(), &recovered_actuator, &store, leader());
  ASSERT_EQ(recovered.recover(100, 65536), PlacementExecutorStatus::OK);
  ASSERT_EQ(recovered.drive(101, 1).status, PlacementExecutorStatus::OK);
  EXPECT_EQ(recovered_actuator.execute_calls, 0u);
  EXPECT_EQ(recovered_actuator.query_calls, 1u);
  EXPECT_EQ(recovered.snapshot()[0].status,
            PlacementOperationStatus::SUCCEEDED);
}

TEST(PlacementOperationStoreTest, OrphanStatusAndEpochChangeFailClosed) {
  FakeFencedKv backend(leader());
  PlacementOperationStore store(&backend);
  PlacementOperationRecord record{
      .intent = drain_intent(),
      .created_at_ms = 1000,
      .updated_at_ms = 1000,
  };
  std::string value;
  ASSERT_TRUE(serialize_placement_operation_record(record, &value));
  backend.set_raw(
      std::string(kPlacementStatusPrefix) + record.intent.operation_id, value);
  std::vector<PlacementPersistedOperation> snapshot;
  EXPECT_EQ(store.load_snapshot(8, 65536, &snapshot),
            PlacementStoreStatus::CORRUPT);

  PlacementLeaderIdentity stale = leader();
  stale.epoch--;
  int64_t revision = 0;
  EXPECT_EQ(store.create_command(drain_intent(), stale, &revision),
            PlacementStoreStatus::INVALID_INPUT);
}

}  // namespace
}  // namespace xllm_service::placement
