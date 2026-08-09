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

#include "provider/state_stream_outbox.h"

#include <gtest/gtest.h>

#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "provider/provider_contract.h"

namespace xllm_service::provider {
namespace {

xllm::proto::EngineState make_state(std::string engine_uid,
                                    uint64_t state_seq) {
  xllm::proto::EngineState state;
  state.set_engine_uid(std::move(engine_uid));
  state.set_incarnation_id("inc");
  state.set_provider_id(xllm::proto::PROVIDER_ID_XLLM_NATIVE);
  state.set_profile_digest("profile");
  state.set_model_revision("model");
  state.set_state_seq(state_seq);
  state.set_heartbeat_age_ms_at_publish(0);
  state.set_state_age_ms_at_publish(0);
  return state;
}

xllm::proto::ProviderEngineKey make_key(std::string engine_uid) {
  xllm::proto::ProviderEngineKey key;
  key.set_engine_uid(std::move(engine_uid));
  key.set_incarnation_id("inc");
  key.set_provider_id(xllm::proto::PROVIDER_ID_XLLM_NATIVE);
  key.set_profile_digest("profile");
  return key;
}

xllm::proto::StateBatch make_batch(xllm::proto::StateBatchKind kind) {
  xllm::proto::StateBatch batch;
  batch.set_contract_version(kProviderContractVersion);
  batch.set_master_incarnation("master-inc");
  batch.set_snapshot_seq(1);
  batch.set_kind(kind);
  return batch;
}

StateStreamOutbox make_outbox(size_t max_pending = 4) {
  return StateStreamOutbox(
      StateStreamOutboxConfig{
          .max_subscribers = 4,
          .max_pending_engine_states = max_pending,
          .max_pending_link_states = 4,
      },
      "master-inc");
}

TEST(StateStreamOutboxTest, ResolvesOnlyValidatedServiceMembers) {
  const std::unordered_map<std::string, std::string> members = {
      {"MASTER", "10.0.0.1:9000"},
      {"10.0.0.1:9000", "10.0.0.1:9000"},
      {"10.0.0.2:9000", "10.0.0.2:9000"},
      {"alias", "10.0.0.3:9000"},
      {"bad", "not-an-address"},
  };
  std::vector<std::string> subscribers;
  ASSERT_TRUE(resolve_state_stream_subscribers(
                  members, "10.0.0.1:9000", 4, &subscribers)
                  .ok());
  ASSERT_EQ(subscribers.size(), 1u);
  EXPECT_EQ(subscribers[0], "10.0.0.2:9000");
  EXPECT_EQ(resolve_state_stream_subscribers(
                members, "10.0.0.1:9000", 0, &subscribers)
                .error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE);
}

TEST(StateStreamOutboxTest, KeepsOneInflightAndMergesLatestDelta) {
  StateStreamOutbox outbox = make_outbox();
  ASSERT_TRUE(outbox.replace_subscribers({"10.0.0.2:9000"}).ok());
  xllm::proto::StateBatch delta =
      make_batch(xllm::proto::STATE_BATCH_KIND_DELTA);
  *delta.add_engine_states() = make_state("engine", 1);
  ASSERT_TRUE(outbox.enqueue_delta(delta, 100).ok());

  xllm::proto::StateBatch full = make_batch(xllm::proto::STATE_BATCH_KIND_FULL);
  *full.add_engine_states() = make_state("engine", 1);
  xllm::proto::StateBatch delivery;
  ASSERT_TRUE(outbox.begin_delivery("10.0.0.2:9000", full, 1, 100, &delivery));
  EXPECT_EQ(delivery.kind(), xllm::proto::STATE_BATCH_KIND_FULL);

  delta.mutable_engine_states(0)->set_state_seq(2);
  ASSERT_TRUE(outbox.enqueue_delta(delta, 101).ok());
  EXPECT_FALSE(outbox.begin_delivery("10.0.0.2:9000", full, 2, 101, &delivery));
  ASSERT_TRUE(outbox.complete_delivery("10.0.0.2:9000", true));

  ASSERT_TRUE(outbox.begin_delivery("10.0.0.2:9000", full, 2, 111, &delivery));
  ASSERT_EQ(delivery.kind(), xllm::proto::STATE_BATCH_KIND_DELTA);
  ASSERT_EQ(delivery.engine_states_size(), 1);
  EXPECT_EQ(delivery.engine_states(0).state_seq(), 2u);
  EXPECT_EQ(delivery.engine_states(0).state_age_ms_at_publish(), 10u);
}

TEST(StateStreamOutboxTest, FailureRequiresFreshFull) {
  StateStreamOutbox outbox = make_outbox();
  ASSERT_TRUE(outbox.replace_subscribers({"10.0.0.2:9000"}).ok());
  xllm::proto::StateBatch full = make_batch(xllm::proto::STATE_BATCH_KIND_FULL);
  xllm::proto::StateBatch delivery;
  ASSERT_TRUE(outbox.begin_delivery("10.0.0.2:9000", full, 1, 100, &delivery));
  ASSERT_TRUE(outbox.complete_delivery("10.0.0.2:9000", true));

  xllm::proto::StateBatch delta =
      make_batch(xllm::proto::STATE_BATCH_KIND_DELTA);
  *delta.add_engine_states() = make_state("engine", 1);
  ASSERT_TRUE(outbox.enqueue_delta(delta, 100).ok());
  ASSERT_TRUE(outbox.begin_delivery("10.0.0.2:9000", full, 2, 100, &delivery));
  EXPECT_EQ(delivery.kind(), xllm::proto::STATE_BATCH_KIND_DELTA);
  ASSERT_TRUE(outbox.complete_delivery("10.0.0.2:9000", false));

  ASSERT_TRUE(outbox.begin_delivery("10.0.0.2:9000", full, 3, 101, &delivery));
  EXPECT_EQ(delivery.kind(), xllm::proto::STATE_BATCH_KIND_FULL);
}

TEST(StateStreamOutboxTest, OverflowStaysBoundedAndRequiresFull) {
  StateStreamOutbox outbox = make_outbox(1);
  ASSERT_TRUE(outbox.replace_subscribers({"10.0.0.2:9000"}).ok());
  xllm::proto::StateBatch full = make_batch(xllm::proto::STATE_BATCH_KIND_FULL);
  xllm::proto::StateBatch delivery;
  ASSERT_TRUE(outbox.begin_delivery("10.0.0.2:9000", full, 1, 100, &delivery));
  ASSERT_TRUE(outbox.complete_delivery("10.0.0.2:9000", true));

  xllm::proto::StateBatch delta =
      make_batch(xllm::proto::STATE_BATCH_KIND_DELTA);
  *delta.add_engine_states() = make_state("engine-1", 1);
  *delta.add_engine_states() = make_state("engine-2", 1);
  ASSERT_TRUE(outbox.enqueue_delta(delta, 100).ok());
  EXPECT_EQ(outbox.pending_entry_count("10.0.0.2:9000"), 0u);
  ASSERT_TRUE(outbox.begin_delivery("10.0.0.2:9000", full, 2, 100, &delivery));
  EXPECT_EQ(delivery.kind(), xllm::proto::STATE_BATCH_KIND_FULL);
}

TEST(StateStreamOutboxTest, RemovalSupersedesStateAndRelatedLink) {
  StateStreamOutbox outbox = make_outbox();
  ASSERT_TRUE(outbox.replace_subscribers({"10.0.0.2:9000"}).ok());
  xllm::proto::StateBatch full = make_batch(xllm::proto::STATE_BATCH_KIND_FULL);
  xllm::proto::StateBatch delivery;
  ASSERT_TRUE(outbox.begin_delivery("10.0.0.2:9000", full, 1, 100, &delivery));
  ASSERT_TRUE(outbox.complete_delivery("10.0.0.2:9000", true));

  xllm::proto::StateBatch delta =
      make_batch(xllm::proto::STATE_BATCH_KIND_DELTA);
  *delta.add_engine_states() = make_state("engine", 1);
  xllm::proto::LinkState* link = delta.add_link_states();
  *link->mutable_prefill() = make_key("engine");
  *link->mutable_decode() = make_key("decode");
  ASSERT_TRUE(outbox.enqueue_delta(delta, 100).ok());

  delta.clear_engine_states();
  delta.clear_link_states();
  *delta.add_removed_engines() = make_key("engine");
  ASSERT_TRUE(outbox.enqueue_delta(delta, 101).ok());
  ASSERT_TRUE(outbox.begin_delivery("10.0.0.2:9000", full, 2, 101, &delivery));
  EXPECT_EQ(delivery.engine_states_size(), 0);
  EXPECT_EQ(delivery.link_states_size(), 0);
  ASSERT_EQ(delivery.removed_engines_size(), 1);
  EXPECT_EQ(delivery.removed_engines(0).engine_uid(), "engine");
}

TEST(StateStreamOutboxTest, ConcurrentUpdatesKeepHighestStateSequence) {
  StateStreamOutbox outbox = make_outbox();
  ASSERT_TRUE(outbox.replace_subscribers({"10.0.0.2:9000"}).ok());
  xllm::proto::StateBatch full = make_batch(xllm::proto::STATE_BATCH_KIND_FULL);
  xllm::proto::StateBatch delivery;
  ASSERT_TRUE(outbox.begin_delivery("10.0.0.2:9000", full, 1, 100, &delivery));
  ASSERT_TRUE(outbox.complete_delivery("10.0.0.2:9000", true));

  constexpr uint64_t kUpdateCount = 64;
  std::vector<std::thread> threads;
  threads.reserve(kUpdateCount);
  for (uint64_t sequence = 1; sequence <= kUpdateCount; ++sequence) {
    threads.emplace_back([sequence, &outbox]() {
      xllm::proto::StateBatch delta =
          make_batch(xllm::proto::STATE_BATCH_KIND_DELTA);
      *delta.add_engine_states() = make_state("engine", sequence);
      EXPECT_TRUE(outbox.enqueue_delta(delta, 100 + sequence).ok());
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(outbox.pending_entry_count("10.0.0.2:9000"), 1u);
  ASSERT_TRUE(outbox.begin_delivery("10.0.0.2:9000", full, 2, 200, &delivery));
  ASSERT_EQ(delivery.engine_states_size(), 1);
  EXPECT_EQ(delivery.engine_states(0).state_seq(), kUpdateCount);
}

}  // namespace
}  // namespace xllm_service::provider
