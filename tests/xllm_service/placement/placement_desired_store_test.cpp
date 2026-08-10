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

#include "placement/placement_desired_store.h"

#include <gtest/gtest.h>

#include <atomic>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace xllm_service::placement {
namespace {

class FakePlacementFencedKv final : public PlacementFencedKv {
 public:
  explicit FakePlacementFencedKv(PlacementLeaderIdentity leader)
      : leader_(std::move(leader)) {}

  PlacementStoreStatus read(const std::string& logical_key,
                            std::string* value,
                            int64_t* mod_revision) override {
    if (logical_key.empty() || value == nullptr || mod_revision == nullptr) {
      return PlacementStoreStatus::INVALID_INPUT;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iterator = entries_.find(logical_key);
    if (iterator == entries_.end()) {
      return PlacementStoreStatus::NOT_FOUND;
    }
    *value = iterator->second.value;
    *mod_revision = iterator->second.mod_revision;
    return PlacementStoreStatus::OK;
  }

  PlacementStoreStatus list(const std::string& logical_prefix,
                            std::vector<PlacementRawValue>* values) override {
    if (logical_prefix.empty() || values == nullptr) {
      return PlacementStoreStatus::INVALID_INPUT;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    values->clear();
    for (const auto& [key, entry] : entries_) {
      if (key.compare(0, logical_prefix.size(), logical_prefix) != 0) {
        continue;
      }
      values->emplace_back(PlacementRawValue{
          .key_suffix = key.substr(logical_prefix.size()),
          .value = entry.value,
          .mod_revision = entry.mod_revision,
      });
    }
    return PlacementStoreStatus::OK;
  }

  PlacementStoreStatus compare_and_set(
      const std::string& logical_key,
      const std::string& value,
      int64_t expected_mod_revision,
      const PlacementLeaderIdentity& leader) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (leader.address != leader_.address ||
        leader.incarnation != leader_.incarnation ||
        leader.epoch != leader_.epoch) {
      return PlacementStoreStatus::FENCED;
    }
    const auto iterator = entries_.find(logical_key);
    const int64_t current_revision =
        iterator == entries_.end() ? 0 : iterator->second.mod_revision;
    if (current_revision != expected_mod_revision) {
      return PlacementStoreStatus::REVISION_CONFLICT;
    }
    entries_.insert_or_assign(logical_key,
                              Entry{
                                  .value = value,
                                  .mod_revision = next_revision_++,
                              });
    return PlacementStoreStatus::OK;
  }

  void set_leader(PlacementLeaderIdentity leader) {
    std::lock_guard<std::mutex> lock(mutex_);
    leader_ = std::move(leader);
  }

  void set_raw(const std::string& logical_key,
               std::string value,
               int64_t revision) {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.insert_or_assign(logical_key,
                              Entry{
                                  .value = std::move(value),
                                  .mod_revision = revision,
                              });
  }

 private:
  struct Entry {
    std::string value;
    int64_t mod_revision = 0;
  };

  std::mutex mutex_;
  PlacementLeaderIdentity leader_;
  std::map<std::string, Entry> entries_;
  int64_t next_revision_ = 1;
};

PlacementLeaderIdentity leader(const std::string& incarnation = "leader-1") {
  return PlacementLeaderIdentity{
      .address = "service-1:2888",
      .incarnation = incarnation,
      .epoch = incarnation == "leader-1" ? 10u : 20u,
  };
}

PlacementPoolKey pool(const std::string& model = "model/r1") {
  return PlacementPoolKey{
      .provider_id = xllm::proto::PROVIDER_ID_XLLM_NATIVE,
      .model_revision = model,
      .role = xllm::proto::ENGINE_ROLE_DECODE,
      .profile_digest = "tp8%profile",
  };
}

PlacementDesiredState desired(uint64_t generation = 1,
                              const std::string& model = "model/r1") {
  return PlacementDesiredState{
      .leader = leader(),
      .generation = generation,
      .pool = pool(model),
      .desired_replicas = 3,
      .reason = PlacementReason::FORECAST_CAPACITY,
      .observation_generation = generation,
      .created_at_unix_ms = 1000 + generation,
      .config_digest = "config-a",
  };
}

TEST(PlacementDesiredStateTest, RoundTripsStrictSchemaAndEscapedKey) {
  const PlacementDesiredState input = desired();
  std::string value;
  ASSERT_TRUE(serialize_placement_desired_state(input, &value));
  PlacementDesiredState parsed;
  ASSERT_TRUE(parse_placement_desired_state(value, &parsed));
  EXPECT_TRUE(placement_pool_keys_equal(input.pool, parsed.pool));
  EXPECT_EQ(parsed.leader.address, input.leader.address);
  EXPECT_EQ(parsed.leader.epoch, input.leader.epoch);
  EXPECT_EQ(parsed.generation, input.generation);
  EXPECT_EQ(parsed.desired_replicas, input.desired_replicas);
  EXPECT_EQ(parsed.reason, input.reason);
  EXPECT_EQ(placement_pool_key_suffix(input.pool),
            "1/3/model%2Fr1/tp8%25profile");
}

TEST(PlacementDesiredStateTest, RejectsMalformedNumericAndOversizedState) {
  std::string value;
  ASSERT_TRUE(serialize_placement_desired_state(desired(), &value));
  nlohmann::json json = nlohmann::json::parse(value);

  json["generation"] = -1;
  EXPECT_FALSE(parse_placement_desired_state(json.dump(), nullptr));
  PlacementDesiredState parsed;
  EXPECT_FALSE(parse_placement_desired_state(json.dump(), &parsed));

  json = nlohmann::json::parse(value);
  json["desired_replicas"] = 1.5;
  EXPECT_FALSE(parse_placement_desired_state(json.dump(), &parsed));

  json = nlohmann::json::parse(value);
  json["reason"] = 999;
  EXPECT_FALSE(parse_placement_desired_state(json.dump(), &parsed));

  EXPECT_FALSE(parse_placement_desired_state(
      std::string(kMaxPlacementDesiredBytes + 1, 'x'), &parsed));
}

TEST(PlacementDesiredStoreTest, CreatesReadsAndRequiresMonotonicGeneration) {
  FakePlacementFencedKv backend(leader());
  PlacementDesiredStore store(&backend);
  PlacementDesiredState first = desired();
  ASSERT_EQ(store.compare_and_set(first, 0, leader()),
            PlacementStoreStatus::OK);

  PlacementDesiredSnapshot snapshot;
  ASSERT_EQ(store.read(first.pool, &snapshot), PlacementStoreStatus::OK);
  EXPECT_EQ(snapshot.mod_revision, 1);
  EXPECT_EQ(snapshot.desired.generation, 1u);

  EXPECT_EQ(store.compare_and_set(first, snapshot.mod_revision, leader()),
            PlacementStoreStatus::REVISION_CONFLICT);
  PlacementDesiredState second = desired(2);
  EXPECT_EQ(store.compare_and_set(second, snapshot.mod_revision, leader()),
            PlacementStoreStatus::OK);
}

TEST(PlacementDesiredStoreTest, FencesOldLeaderAndRevision) {
  FakePlacementFencedKv backend(leader());
  PlacementDesiredStore store(&backend);
  ASSERT_EQ(store.compare_and_set(desired(), 0, leader()),
            PlacementStoreStatus::OK);

  PlacementLeaderIdentity next_leader = leader("leader-2");
  backend.set_leader(next_leader);
  PlacementDesiredState next = desired(2);
  EXPECT_EQ(store.compare_and_set(next, 1, leader()),
            PlacementStoreStatus::FENCED);

  next.leader = next_leader;
  EXPECT_EQ(store.compare_and_set(next, 99, next_leader),
            PlacementStoreStatus::REVISION_CONFLICT);
}

TEST(PlacementDesiredStoreTest, LoadsStableBoundedSnapshot) {
  FakePlacementFencedKv backend(leader());
  PlacementDesiredStore store(&backend);
  ASSERT_EQ(store.compare_and_set(desired(1, "model-z"), 0, leader()),
            PlacementStoreStatus::OK);
  ASSERT_EQ(store.compare_and_set(desired(1, "model-a"), 0, leader()),
            PlacementStoreStatus::OK);

  std::vector<PlacementDesiredSnapshot> snapshot;
  ASSERT_EQ(store.load_snapshot(2, 2 * kMaxPlacementDesiredBytes, &snapshot),
            PlacementStoreStatus::OK);
  ASSERT_EQ(snapshot.size(), 2u);
  EXPECT_EQ(snapshot[0].desired.pool.model_revision, "model-a");
  EXPECT_EQ(snapshot[1].desired.pool.model_revision, "model-z");
  EXPECT_EQ(store.load_snapshot(1, 2 * kMaxPlacementDesiredBytes, &snapshot),
            PlacementStoreStatus::CAPACITY_EXCEEDED);
  EXPECT_EQ(store.load_snapshot(2, 1, &snapshot),
            PlacementStoreStatus::CAPACITY_EXCEEDED);
}

TEST(PlacementDesiredStoreTest, CorruptSnapshotFailsClosed) {
  FakePlacementFencedKv backend(leader());
  PlacementDesiredStore store(&backend);
  backend.set_raw(
      std::string(kPlacementDesiredPrefix) + placement_pool_key_suffix(pool()),
      "{broken-json",
      1);
  std::vector<PlacementDesiredSnapshot> snapshot;
  EXPECT_EQ(store.load_snapshot(10, 10000, &snapshot),
            PlacementStoreStatus::CORRUPT);
}

TEST(PlacementDesiredStoreTest, ConcurrentCasHasSingleWinner) {
  FakePlacementFencedKv backend(leader());
  PlacementDesiredStore store(&backend);
  ASSERT_EQ(store.compare_and_set(desired(), 0, leader()),
            PlacementStoreStatus::OK);
  std::atomic<uint32_t> winners = 0;
  std::vector<std::thread> threads;
  for (uint32_t index = 0; index < 8; ++index) {
    threads.emplace_back([&store, &winners, index]() {
      PlacementDesiredState next = desired(2);
      next.desired_replicas += index;
      if (store.compare_and_set(next, 1, leader()) ==
          PlacementStoreStatus::OK) {
        winners.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(winners.load(std::memory_order_relaxed), 1u);
}

}  // namespace
}  // namespace xllm_service::placement
