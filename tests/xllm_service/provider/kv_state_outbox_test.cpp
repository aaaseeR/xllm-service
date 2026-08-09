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

#include "provider/kv_state_outbox.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace xllm_service::provider {
namespace {

xllm::proto::KVEventBatch make_batch(uint64_t event_seq, uint64_t age_ms = 0) {
  xllm::proto::KVEventBatch batch;
  batch.set_contract_version(kProviderContractVersion);
  batch.mutable_identity()->mutable_engine()->set_provider_id(
      xllm::proto::PROVIDER_ID_XLLM_NATIVE);
  batch.mutable_identity()->mutable_engine()->set_profile_digest("profile");
  batch.mutable_identity()->mutable_engine()->set_engine_uid("engine");
  batch.mutable_identity()->mutable_engine()->set_incarnation_id("incarnation");
  batch.mutable_identity()->set_model_revision("model");
  batch.mutable_identity()->set_kv_namespace("namespace");
  batch.mutable_identity()->set_cache_epoch(1);
  batch.set_last_event_seq(event_seq);
  batch.set_batch_age_ms_at_publish(age_ms);
  if (event_seq == 0) {
    return batch;
  }
  xllm::proto::KVEvent* event = batch.add_events();
  event->set_event_seq(event_seq);
  event->set_kind(xllm::proto::KV_EVENT_KIND_STORED);
  std::array<uint8_t, 16> hash{};
  hash[0] = static_cast<uint8_t>(event_seq);
  event->mutable_block()->set_block_hash(hash.data(), hash.size());
  event->mutable_block()->set_token_begin((event_seq - 1) * 16);
  event->mutable_block()->set_token_end(event_seq * 16);
  event->mutable_block()->set_cache_group("group:0");
  event->mutable_block()->set_tier(xllm::proto::KV_CACHE_TIER_HBM);
  event->set_reason(xllm::proto::KV_EVENT_REASON_PREFIX_PUBLISHED);
  event->set_event_age_ms_at_publish(age_ms);
  return batch;
}

KVStateOutboxConfig make_config() {
  return KVStateOutboxConfig{
      .max_subscribers = 4,
      .max_pending_batches_per_subscriber = 8,
      .max_pending_events_per_subscriber = 8,
      .max_pending_bytes_per_subscriber = 64 * 1024,
      .max_delivery_batches = 4,
      .max_delivery_bytes = 16 * 1024,
  };
}

TEST(KVStateOutboxTest, RetriesExactDeliveryIndependentlyPerSubscriber) {
  KVStateOutbox outbox(make_config(), "master-a");
  ASSERT_TRUE(
      outbox.replace_subscribers({"127.0.0.1:8001", "127.0.0.1:8002"}).ok());
  ASSERT_TRUE(outbox.enqueue(make_batch(1), 100).ok());

  xllm::proto::KVStateBatch first;
  ASSERT_TRUE(outbox.begin_delivery("127.0.0.1:8001", 101, &first));
  EXPECT_EQ(first.stream_seq(), 1);
  EXPECT_GT(first.stream_epoch(), 0);
  EXPECT_EQ(first.engine_batches_size(), 1);
  EXPECT_TRUE(outbox.complete_delivery("127.0.0.1:8001", false));

  xllm::proto::KVStateBatch retry;
  ASSERT_TRUE(outbox.begin_delivery("127.0.0.1:8001", 500, &retry));
  EXPECT_EQ(retry.SerializeAsString(), first.SerializeAsString());
  EXPECT_TRUE(outbox.complete_delivery("127.0.0.1:8001", true));

  xllm::proto::KVStateBatch other;
  ASSERT_TRUE(outbox.begin_delivery("127.0.0.1:8002", 102, &other));
  EXPECT_EQ(other.stream_seq(), 1);
  EXPECT_TRUE(outbox.complete_delivery("127.0.0.1:8002", true));
}

TEST(KVStateOutboxTest, ReaddedSubscriberGetsNewStreamEpoch) {
  KVStateOutbox outbox(make_config(), "master-a");
  ASSERT_TRUE(outbox.replace_subscribers({"127.0.0.1:8001"}).ok());
  ASSERT_TRUE(outbox.enqueue(make_batch(1), 100).ok());
  xllm::proto::KVStateBatch first;
  ASSERT_TRUE(outbox.begin_delivery("127.0.0.1:8001", 101, &first));
  ASSERT_TRUE(outbox.complete_delivery("127.0.0.1:8001", true));

  ASSERT_TRUE(outbox.replace_subscribers({}).ok());
  ASSERT_TRUE(outbox.replace_subscribers({"127.0.0.1:8001"}).ok());
  ASSERT_TRUE(outbox.enqueue(make_batch(2), 102).ok());
  xllm::proto::KVStateBatch replacement;
  ASSERT_TRUE(outbox.begin_delivery("127.0.0.1:8001", 103, &replacement));
  EXPECT_EQ(replacement.stream_seq(), 1);
  EXPECT_GT(replacement.stream_epoch(), first.stream_epoch());
}

TEST(KVStateOutboxTest, OverflowSkipsReplicaSequenceAndRemainsBounded) {
  KVStateOutboxConfig config = make_config();
  config.max_pending_batches_per_subscriber = 1;
  KVStateOutbox outbox(config, "master-a");
  ASSERT_TRUE(outbox.replace_subscribers({"127.0.0.1:8001"}).ok());
  ASSERT_TRUE(outbox.enqueue(make_batch(1), 100).ok());
  ASSERT_TRUE(outbox.enqueue(make_batch(2), 101).ok());
  const KVStateOutboxStats before = outbox.stats();
  EXPECT_EQ(before.pending_batches, 1);
  EXPECT_EQ(before.forced_stream_gaps, 1);

  xllm::proto::KVStateBatch delivery;
  ASSERT_TRUE(outbox.begin_delivery("127.0.0.1:8001", 102, &delivery));
  EXPECT_EQ(delivery.stream_seq(), 2);
  ASSERT_EQ(delivery.engine_batches_size(), 1);
  EXPECT_EQ(delivery.engine_batches(0).last_event_seq(), 2);
}

TEST(KVStateOutboxTest, AccumulatesQueueAgeAcrossServiceHop) {
  KVStateOutbox outbox(make_config(), "master-a");
  ASSERT_TRUE(outbox.replace_subscribers({"127.0.0.1:8001"}).ok());
  ASSERT_TRUE(outbox.enqueue(make_batch(1, 7), 100).ok());

  xllm::proto::KVStateBatch delivery;
  ASSERT_TRUE(outbox.begin_delivery("127.0.0.1:8001", 130, &delivery));
  ASSERT_EQ(delivery.engine_batches_size(), 1);
  EXPECT_EQ(delivery.engine_batches(0).batch_age_ms_at_publish(), 37);
  EXPECT_EQ(delivery.engine_batches(0).events(0).event_age_ms_at_publish(), 37);
}

TEST(KVStateOutboxTest, OversizedBatchBecomesFailClosedGapMarker) {
  KVStateOutboxConfig config = make_config();
  config.max_delivery_bytes = 512;
  KVStateOutbox outbox(config, "master-a");
  ASSERT_TRUE(outbox.replace_subscribers({"127.0.0.1:8001"}).ok());
  xllm::proto::KVEventBatch batch = make_batch(1);
  batch.mutable_events(0)->mutable_block()->set_cache_group(
      std::string(1024, 'x'));
  ASSERT_TRUE(outbox.enqueue(batch, 100).ok());

  xllm::proto::KVStateBatch delivery;
  ASSERT_TRUE(outbox.begin_delivery("127.0.0.1:8001", 101, &delivery));
  EXPECT_EQ(delivery.stream_seq(), 2);
  ASSERT_EQ(delivery.engine_batches_size(), 1);
  EXPECT_TRUE(delivery.engine_batches(0).gap_before_events());
  EXPECT_TRUE(delivery.engine_batches(0).events().empty());
}

}  // namespace
}  // namespace xllm_service::provider
