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

#include "provider/kv_state_replica.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include "provider/provider_contract.h"

namespace xllm_service::provider {
namespace {

xllm::proto::KVStreamIdentity make_identity() {
  xllm::proto::KVStreamIdentity identity;
  identity.mutable_engine()->set_provider_id(
      xllm::proto::PROVIDER_ID_XLLM_NATIVE);
  identity.mutable_engine()->set_profile_digest("profile");
  identity.mutable_engine()->set_engine_uid("engine");
  identity.mutable_engine()->set_incarnation_id("incarnation");
  identity.set_model_revision("model");
  identity.set_kv_namespace("namespace");
  identity.set_cache_epoch(1);
  return identity;
}

xllm::proto::KVEventBatch make_event_batch(uint64_t sequence) {
  xllm::proto::KVEventBatch batch;
  batch.set_contract_version(kProviderContractVersion);
  *batch.mutable_identity() = make_identity();
  batch.set_last_event_seq(sequence);
  batch.set_batch_age_ms_at_publish(0);
  xllm::proto::KVEvent* event = batch.add_events();
  event->set_event_seq(sequence);
  event->set_kind(xllm::proto::KV_EVENT_KIND_STORED);
  std::array<uint8_t, 16> hash{};
  hash[0] = static_cast<uint8_t>(sequence);
  event->mutable_block()->set_block_hash(hash.data(), hash.size());
  event->mutable_block()->set_token_begin((sequence - 1) * 16);
  event->mutable_block()->set_token_end(sequence * 16);
  event->mutable_block()->set_cache_group("group:0");
  event->mutable_block()->set_tier(xllm::proto::KV_CACHE_TIER_HBM);
  event->set_reason(xllm::proto::KV_EVENT_REASON_PREFIX_PUBLISHED);
  event->set_event_age_ms_at_publish(0);
  return batch;
}

xllm::proto::KVStateBatch make_state_batch(uint64_t stream_sequence,
                                           uint64_t event_sequence,
                                           uint64_t stream_epoch = 1) {
  xllm::proto::KVStateBatch batch;
  batch.set_contract_version(kProviderContractVersion);
  batch.set_master_incarnation("master-a");
  batch.set_stream_seq(stream_sequence);
  batch.set_stream_epoch(stream_epoch);
  *batch.add_engine_batches() = make_event_batch(event_sequence);
  return batch;
}

KVShadowIndexConfig shadow_config() {
  return KVShadowIndexConfig{
      .max_index_entries = 128,
      .max_index_bytes = 1024 * 1024,
      .max_recovery_events_per_engine = 32,
      .max_recovery_bytes_per_engine = 64 * 1024,
      .max_snapshot_entries_per_engine = 128,
      .max_snapshot_bytes_per_engine = 1024 * 1024,
      .event_ttl_ms = 1000,
      .recovery_timeout_ms = 1000,
  };
}

TEST(KVStateReplicaTest, NewStreamEpochFencesOldSequenceAndCredit) {
  KVShadowIndex index(shadow_config());
  KVStateReplica replica(KVStateReplicaConfig{}, &index);
  ASSERT_TRUE(replica.set_master("master-a"));
  ASSERT_EQ(replica.apply(make_state_batch(1, 1), 100).code,
            KVApplyCode::APPLIED);

  const KVApplyResult replacement =
      replica.apply(make_state_batch(1, 1, 2), 101);
  EXPECT_EQ(replacement.code, KVApplyCode::APPLIED);
  EXPECT_EQ(replica.stream_epoch(), 2);
  EXPECT_EQ(replica.last_stream_seq(), 1);
  EXPECT_EQ(index.resident_entries(make_identity()), 1);

  EXPECT_EQ(replica.apply(make_state_batch(2, 2, 1), 102).code,
            KVApplyCode::DUPLICATE);
  EXPECT_EQ(replica.stream_epoch(), 2);
  EXPECT_EQ(replica.last_stream_seq(), 1);
}

TEST(KVStateReplicaTest, AppliesInOrderAndDeduplicatesServiceBatch) {
  KVShadowIndex index(shadow_config());
  KVStateReplica replica(KVStateReplicaConfig{}, &index);
  ASSERT_TRUE(replica.set_master("master-a"));

  const xllm::proto::KVStateBatch first = make_state_batch(1, 1);
  EXPECT_EQ(replica.apply(first, 100).code, KVApplyCode::APPLIED);
  EXPECT_EQ(replica.apply(first, 101).code, KVApplyCode::DUPLICATE);
  EXPECT_EQ(replica.last_stream_seq(), 1);
  EXPECT_EQ(index.resident_entries(make_identity()), 1);
}

TEST(KVStateReplicaTest, StreamGapDropsPriorCreditBeforeApplyingCurrentBatch) {
  KVShadowIndex index(shadow_config());
  KVStateReplica replica(KVStateReplicaConfig{}, &index);
  ASSERT_TRUE(replica.set_master("master-a"));
  ASSERT_EQ(replica.apply(make_state_batch(1, 1), 100).code,
            KVApplyCode::APPLIED);

  const KVApplyResult gap = replica.apply(make_state_batch(3, 2), 101);
  EXPECT_EQ(gap.code, KVApplyCode::SNAPSHOT_REQUIRED);
  EXPECT_EQ(index.health(make_identity()), KVShadowHealth::UNKNOWN);
  EXPECT_EQ(index.resident_entries(make_identity()), 0);
  EXPECT_EQ(replica.last_stream_seq(), 3);
}

TEST(KVStateReplicaTest, MasterChangeFencesOldShadowAndStalePush) {
  KVShadowIndex index(shadow_config());
  KVStateReplica replica(KVStateReplicaConfig{}, &index);
  ASSERT_TRUE(replica.set_master("master-a"));
  ASSERT_EQ(replica.apply(make_state_batch(1, 1), 100).code,
            KVApplyCode::APPLIED);
  ASSERT_TRUE(replica.set_master("master-b"));
  EXPECT_EQ(index.stats().engine_streams, 0);
  EXPECT_EQ(replica.last_stream_seq(), 0);
  EXPECT_EQ(replica.apply(make_state_batch(2, 2), 101).code,
            KVApplyCode::REJECTED);
  EXPECT_EQ(index.stats().engine_streams, 0);
}

TEST(KVStateReplicaTest, MalformedEngineBatchFailsClosedAfterAckBoundary) {
  KVShadowIndex index(shadow_config());
  KVStateReplica replica(KVStateReplicaConfig{}, &index);
  ASSERT_TRUE(replica.set_master("master-a"));
  xllm::proto::KVStateBatch malformed = make_state_batch(1, 1);
  malformed.mutable_engine_batches(0)->clear_batch_age_ms_at_publish();
  EXPECT_EQ(replica.apply(malformed, 100).code, KVApplyCode::SNAPSHOT_REQUIRED);
  EXPECT_EQ(replica.last_stream_seq(), 1);
  EXPECT_EQ(index.stats().engine_streams, 0);
}

}  // namespace
}  // namespace xllm_service::provider
