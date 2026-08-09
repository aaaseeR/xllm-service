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

#include "provider/kv_shadow_index.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace xllm_service::provider {
namespace {

inline constexpr uint32_t kContractVersion = 1;
inline constexpr size_t kHashBytes = 16;

xllm::proto::KVStreamIdentity make_identity(
    std::string incarnation = "incarnation-a",
    std::string model = "model-a",
    std::string kv_namespace = "namespace-a",
    uint64_t epoch = 1) {
  xllm::proto::KVStreamIdentity identity;
  identity.mutable_engine()->set_provider_id(
      xllm::proto::PROVIDER_ID_XLLM_NATIVE);
  identity.mutable_engine()->set_profile_digest("profile-a");
  identity.mutable_engine()->set_engine_uid("engine-a");
  identity.mutable_engine()->set_incarnation_id(std::move(incarnation));
  identity.set_model_revision(std::move(model));
  identity.set_kv_namespace(std::move(kv_namespace));
  identity.set_cache_epoch(epoch);
  return identity;
}

xllm::proto::KVBlockEntry make_block(uint64_t value, uint64_t block_index = 0) {
  std::array<uint8_t, kHashBytes> hash{};
  for (size_t index = 0; index < hash.size(); ++index) {
    hash[index] = static_cast<uint8_t>((value >> ((index % 8) * 8)) ^ index);
  }
  xllm::proto::KVBlockEntry block;
  block.set_block_hash(hash.data(), hash.size());
  if (block_index > 0) {
    std::array<uint8_t, kHashBytes> parent = hash;
    parent[0] ^= 0xff;
    block.set_parent_hash(parent.data(), parent.size());
  }
  block.set_token_begin(block_index * 16);
  block.set_token_end((block_index + 1) * 16);
  block.set_cache_group("group:0");
  block.set_tier(xllm::proto::KV_CACHE_TIER_HBM);
  return block;
}

xllm::proto::KVEvent make_event(uint64_t sequence,
                                xllm::proto::KVEventKind kind,
                                const xllm::proto::KVBlockEntry& block = {}) {
  xllm::proto::KVEvent event;
  event.set_event_seq(sequence);
  event.set_kind(kind);
  if (kind != xllm::proto::KV_EVENT_KIND_CLEARED) {
    *event.mutable_block() = block;
  }
  event.set_reason(kind == xllm::proto::KV_EVENT_KIND_STORED
                       ? xllm::proto::KV_EVENT_REASON_PREFIX_PUBLISHED
                   : kind == xllm::proto::KV_EVENT_KIND_REMOVED
                       ? xllm::proto::KV_EVENT_REASON_CAPACITY_EVICTED
                       : xllm::proto::KV_EVENT_REASON_CACHE_RESET);
  return event;
}

xllm::proto::KVEventBatch make_batch(
    const xllm::proto::KVStreamIdentity& identity,
    std::vector<xllm::proto::KVEvent> events,
    bool gap = false,
    uint64_t high_watermark = 0) {
  xllm::proto::KVEventBatch batch;
  batch.set_contract_version(kContractVersion);
  *batch.mutable_identity() = identity;
  batch.set_gap_before_events(gap);
  batch.set_batch_age_ms_at_publish(0);
  for (xllm::proto::KVEvent& event : events) {
    high_watermark = std::max(high_watermark, event.event_seq());
    *batch.add_events() = std::move(event);
  }
  batch.set_last_event_seq(high_watermark);
  return batch;
}

xllm::proto::KVCacheSnapshotPage make_snapshot_page(
    const xllm::proto::KVStreamIdentity& identity,
    std::string snapshot_id,
    uint64_t base_sequence,
    uint64_t next_cursor,
    bool done,
    std::vector<xllm::proto::KVBlockEntry> entries) {
  xllm::proto::KVCacheSnapshotPage page;
  page.set_contract_version(kContractVersion);
  page.set_status(xllm::proto::KV_SNAPSHOT_STATUS_OK);
  *page.mutable_identity() = identity;
  page.set_snapshot_id(std::move(snapshot_id));
  page.set_base_event_seq(base_sequence);
  page.set_next_cursor(next_cursor);
  page.set_done(done);
  for (xllm::proto::KVBlockEntry& entry : entries) {
    *page.add_entries() = std::move(entry);
  }
  return page;
}

KVShadowIndexConfig make_config() {
  return KVShadowIndexConfig{
      .max_engine_streams = 64,
      .max_index_entries = 1024,
      .max_index_bytes = 1024 * 1024,
      .max_recovery_events_per_engine = 32,
      .max_recovery_bytes_per_engine = 64 * 1024,
      .max_snapshot_entries_per_engine = 1024,
      .max_snapshot_bytes_per_engine = 1024 * 1024,
      .event_ttl_ms = 1000,
      .recovery_timeout_ms = 1000,
  };
}

TEST(KVShadowIndexTest, AppliesDuplicateStoreAndRemoveIdempotently) {
  KVShadowIndex index(make_config());
  const xllm::proto::KVStreamIdentity identity = make_identity();
  const xllm::proto::KVBlockEntry block = make_block(1);
  const xllm::proto::KVEventBatch stored = make_batch(
      identity, {make_event(1, xllm::proto::KV_EVENT_KIND_STORED, block)});

  EXPECT_EQ(index.apply_event_batch(stored, 100).code, KVApplyCode::APPLIED);
  EXPECT_EQ(index.apply_event_batch(stored, 101).code, KVApplyCode::DUPLICATE);
  EXPECT_TRUE(index.contains(
      identity, block.block_hash(), block.cache_group(), block.tier()));

  const xllm::proto::KVEventBatch removed = make_batch(
      identity, {make_event(2, xllm::proto::KV_EVENT_KIND_REMOVED, block)});
  EXPECT_EQ(index.apply_event_batch(removed, 102).code, KVApplyCode::APPLIED);
  EXPECT_FALSE(index.contains(
      identity, block.block_hash(), block.cache_group(), block.tier()));
}

TEST(KVShadowIndexTest, MatchesContinuousPrefixAndCountsDuplicateLocations) {
  KVShadowIndex index(make_config());
  const xllm::proto::KVStreamIdentity identity = make_identity();
  xllm::proto::KVBlockEntry first_rank_zero = make_block(1, 0);
  xllm::proto::KVBlockEntry first_rank_one = first_rank_zero;
  first_rank_one.set_dp_rank(1);
  const xllm::proto::KVBlockEntry second = make_block(2, 1);
  const xllm::proto::KVBlockEntry missing = make_block(3, 2);

  EXPECT_EQ(
      index
          .apply_event_batch(
              make_batch(
                  identity,
                  {make_event(
                       1, xllm::proto::KV_EVENT_KIND_STORED, first_rank_zero),
                   make_event(
                       2, xllm::proto::KV_EVENT_KIND_STORED, first_rank_one),
                   make_event(3, xllm::proto::KV_EVENT_KIND_STORED, second)}),
              100)
          .code,
      KVApplyCode::APPLIED);

  const KVPrefixMatch full = index.match_contiguous_prefix(
      identity,
      {first_rank_zero.block_hash(), second.block_hash()},
      first_rank_zero.cache_group(),
      first_rank_zero.tier());
  EXPECT_EQ(full.health, KVShadowHealth::READY);
  EXPECT_EQ(full.contiguous_blocks, 2u);

  const KVPrefixMatch stopped = index.match_contiguous_prefix(
      identity,
      {first_rank_zero.block_hash(), missing.block_hash(), second.block_hash()},
      first_rank_zero.cache_group(),
      first_rank_zero.tier());
  EXPECT_EQ(stopped.contiguous_blocks, 1u);

  EXPECT_EQ(index
                .apply_event_batch(
                    make_batch(identity,
                               {make_event(4,
                                           xllm::proto::KV_EVENT_KIND_REMOVED,
                                           first_rank_zero)}),
                    101)
                .code,
            KVApplyCode::APPLIED);
  EXPECT_TRUE(index.contains(identity,
                             first_rank_zero.block_hash(),
                             first_rank_zero.cache_group(),
                             first_rank_zero.tier()));
  EXPECT_EQ(index
                .apply_event_batch(
                    make_batch(identity,
                               {make_event(5,
                                           xllm::proto::KV_EVENT_KIND_REMOVED,
                                           first_rank_one)}),
                    102)
                .code,
            KVApplyCode::APPLIED);
  EXPECT_FALSE(index.contains(identity,
                              first_rank_zero.block_hash(),
                              first_rank_zero.cache_group(),
                              first_rank_zero.tier()));
}

TEST(KVShadowIndexTest, GapAndOutOfOrderFailClosed) {
  KVShadowIndex index(make_config());
  const xllm::proto::KVStreamIdentity identity = make_identity();
  EXPECT_EQ(index
                .apply_event_batch(
                    make_batch(identity,
                               {make_event(1,
                                           xllm::proto::KV_EVENT_KIND_STORED,
                                           make_block(1))}),
                    100)
                .code,
            KVApplyCode::APPLIED);

  const KVApplyResult gap = index.apply_event_batch(
      make_batch(
          identity,
          {make_event(2, xllm::proto::KV_EVENT_KIND_STORED, make_block(2, 1))},
          /*gap=*/true),
      101);
  EXPECT_EQ(gap.code, KVApplyCode::SNAPSHOT_REQUIRED);
  EXPECT_EQ(index.health(identity), KVShadowHealth::UNKNOWN);
  EXPECT_EQ(index.resident_entries(identity), 0);

  KVShadowIndex reordered(make_config());
  const KVApplyResult out_of_order = reordered.apply_event_batch(
      make_batch(
          identity,
          {make_event(2, xllm::proto::KV_EVENT_KIND_STORED, make_block(2, 1))}),
      100);
  EXPECT_EQ(out_of_order.code, KVApplyCode::SNAPSHOT_REQUIRED);
  EXPECT_EQ(reordered.health(identity), KVShadowHealth::UNKNOWN);
}

TEST(KVShadowIndexTest, ClearEpochAndNewIncarnationFenceOldLocations) {
  KVShadowIndex index(make_config());
  const xllm::proto::KVStreamIdentity first = make_identity();
  ASSERT_EQ(index
                .apply_event_batch(
                    make_batch(first,
                               {make_event(1,
                                           xllm::proto::KV_EVENT_KIND_STORED,
                                           make_block(1))}),
                    100)
                .code,
            KVApplyCode::APPLIED);

  const xllm::proto::KVStreamIdentity epoch_two =
      make_identity("incarnation-a", "model-a", "namespace-a", 2);
  EXPECT_EQ(
      index
          .apply_event_batch(
              make_batch(epoch_two,
                         {make_event(1, xllm::proto::KV_EVENT_KIND_CLEARED)}),
              101)
          .code,
      KVApplyCode::APPLIED);
  EXPECT_EQ(index.health(first), KVShadowHealth::UNKNOWN);
  EXPECT_EQ(index.health(epoch_two), KVShadowHealth::READY);
  EXPECT_EQ(index.resident_entries(epoch_two), 0);

  const xllm::proto::KVStreamIdentity replacement =
      make_identity("incarnation-b");
  EXPECT_EQ(index
                .apply_event_batch(
                    make_batch(replacement,
                               {make_event(1,
                                           xllm::proto::KV_EVENT_KIND_STORED,
                                           make_block(9))}),
                    102)
                .code,
            KVApplyCode::APPLIED);
  EXPECT_EQ(index.health(epoch_two), KVShadowHealth::UNKNOWN);
  EXPECT_EQ(index.health(replacement), KVShadowHealth::READY);
  EXPECT_EQ(index.stats().index_entries, 1);
}

TEST(KVShadowIndexTest, SnapshotReplaysIncrementAfterDeterministicCutover) {
  KVShadowIndex index(make_config());
  const xllm::proto::KVStreamIdentity identity = make_identity();
  ASSERT_EQ(index.begin_recovery(identity, 100).code, KVApplyCode::RECOVERING);

  const xllm::proto::KVBlockEntry third = make_block(3, 2);
  EXPECT_EQ(
      index
          .apply_event_batch(
              make_batch(
                  identity,
                  {make_event(3, xllm::proto::KV_EVENT_KIND_STORED, third)}),
              101)
          .code,
      KVApplyCode::RECOVERING);

  const xllm::proto::KVCacheSnapshotPage first =
      make_snapshot_page(identity,
                         "snapshot-a",
                         /*base_sequence=*/2,
                         /*next_cursor=*/1,
                         /*done=*/false,
                         {make_block(1)});
  EXPECT_EQ(index.apply_snapshot_page(first, 102).code,
            KVApplyCode::RECOVERING);
  const xllm::proto::KVCacheSnapshotPage second =
      make_snapshot_page(identity,
                         "snapshot-a",
                         /*base_sequence=*/2,
                         /*next_cursor=*/2,
                         /*done=*/true,
                         {make_block(2, 1)});
  const KVApplyResult completed = index.apply_snapshot_page(second, 103);
  EXPECT_EQ(completed.code, KVApplyCode::APPLIED);
  EXPECT_EQ(completed.accepted_through_event_seq, 3);
  EXPECT_EQ(index.health(identity), KVShadowHealth::READY);
  EXPECT_EQ(index.resident_entries(identity), 3);
  EXPECT_TRUE(index.contains(
      identity, third.block_hash(), third.cache_group(), third.tier()));
}

TEST(KVShadowIndexTest, SnapshotReplayGapRestartsRecovery) {
  KVShadowIndex index(make_config());
  const xllm::proto::KVStreamIdentity identity = make_identity();
  ASSERT_EQ(index.begin_recovery(identity, 100).code, KVApplyCode::RECOVERING);
  ASSERT_EQ(index
                .apply_event_batch(
                    make_batch(identity,
                               {make_event(4,
                                           xllm::proto::KV_EVENT_KIND_STORED,
                                           make_block(4, 3))}),
                    101)
                .code,
            KVApplyCode::RECOVERING);

  const KVApplyResult completed =
      index.apply_snapshot_page(make_snapshot_page(identity,
                                                   "snapshot-a",
                                                   /*base_sequence=*/2,
                                                   /*next_cursor=*/1,
                                                   /*done=*/true,
                                                   {make_block(1)}),
                                102);
  EXPECT_EQ(completed.code, KVApplyCode::SNAPSHOT_REQUIRED);
  EXPECT_EQ(index.health(identity), KVShadowHealth::UNKNOWN);
  EXPECT_EQ(index.resident_entries(identity), 0);
}

TEST(KVShadowIndexTest, AbortedRecoveryDropsStagingAndBufferedEvents) {
  KVShadowIndex index(make_config());
  const xllm::proto::KVStreamIdentity identity = make_identity();
  ASSERT_EQ(index.begin_recovery(identity, 100).code, KVApplyCode::RECOVERING);
  ASSERT_EQ(index
                .apply_event_batch(
                    make_batch(identity,
                               {make_event(2,
                                           xllm::proto::KV_EVENT_KIND_STORED,
                                           make_block(2, 1))}),
                    101)
                .code,
            KVApplyCode::RECOVERING);
  ASSERT_EQ(index
                .apply_snapshot_page(make_snapshot_page(identity,
                                                        "snapshot-a",
                                                        /*base_sequence=*/1,
                                                        /*next_cursor=*/1,
                                                        /*done=*/false,
                                                        {make_block(1)}),
                                     102)
                .code,
            KVApplyCode::RECOVERING);

  index.abort_recovery(identity);

  EXPECT_EQ(index.health(identity), KVShadowHealth::UNKNOWN);
  EXPECT_EQ(index.resident_entries(identity), 0);
  const KVShadowIndexStats stats = index.stats();
  EXPECT_EQ(stats.index_entries, 0);
  EXPECT_EQ(stats.recovery_events, 0);
  EXPECT_EQ(stats.recovery_bytes, 0);
}

TEST(KVShadowIndexTest, RecoveryAndIndexCapacityRemainBounded) {
  KVShadowIndexConfig recovery_config = make_config();
  recovery_config.max_recovery_events_per_engine = 1;
  KVShadowIndex recovering(recovery_config);
  const xllm::proto::KVStreamIdentity identity = make_identity();
  ASSERT_EQ(recovering.begin_recovery(identity, 100).code,
            KVApplyCode::RECOVERING);
  EXPECT_EQ(recovering
                .apply_event_batch(
                    make_batch(identity,
                               {make_event(2,
                                           xllm::proto::KV_EVENT_KIND_STORED,
                                           make_block(2, 1))}),
                    101)
                .code,
            KVApplyCode::RECOVERING);
  EXPECT_EQ(recovering
                .apply_event_batch(
                    make_batch(identity,
                               {make_event(3,
                                           xllm::proto::KV_EVENT_KIND_STORED,
                                           make_block(3, 2))}),
                    102)
                .code,
            KVApplyCode::SNAPSHOT_REQUIRED);
  EXPECT_EQ(recovering.stats().recovery_events, 0);

  KVShadowIndexConfig index_config = make_config();
  index_config.max_index_entries = 1;
  KVShadowIndex bounded(index_config);
  const KVApplyResult capacity = bounded.apply_event_batch(
      make_batch(
          identity,
          {make_event(1, xllm::proto::KV_EVENT_KIND_STORED, make_block(1)),
           make_event(2, xllm::proto::KV_EVENT_KIND_STORED, make_block(2, 1))}),
      100);
  EXPECT_EQ(capacity.code, KVApplyCode::SNAPSHOT_REQUIRED);
  EXPECT_EQ(bounded.stats().index_entries, 0);
}

TEST(KVShadowIndexTest, KeepaliveRefreshesTtlAndExposesSilentLoss) {
  KVShadowIndexConfig config = make_config();
  config.event_ttl_ms = 10;
  KVShadowIndex index(config);
  const xllm::proto::KVStreamIdentity identity = make_identity();
  ASSERT_EQ(index
                .apply_event_batch(
                    make_batch(identity,
                               {make_event(1,
                                           xllm::proto::KV_EVENT_KIND_STORED,
                                           make_block(1))}),
                    100)
                .code,
            KVApplyCode::APPLIED);
  EXPECT_EQ(
      index.apply_event_batch(make_batch(identity, {}, /*gap=*/false, 1), 105)
          .code,
      KVApplyCode::DUPLICATE);
  index.expire(114);
  EXPECT_EQ(index.health(identity), KVShadowHealth::READY);
  index.expire(116);
  EXPECT_EQ(index.health(identity), KVShadowHealth::UNKNOWN);

  KVShadowIndex silent_loss(config);
  ASSERT_EQ(silent_loss
                .apply_event_batch(
                    make_batch(identity,
                               {make_event(1,
                                           xllm::proto::KV_EVENT_KIND_STORED,
                                           make_block(1))}),
                    100)
                .code,
            KVApplyCode::APPLIED);
  EXPECT_EQ(
      silent_loss
          .apply_event_batch(make_batch(identity, {}, /*gap=*/false, 2), 101)
          .code,
      KVApplyCode::SNAPSHOT_REQUIRED);
}

TEST(KVShadowIndexTest, StaleTransportObservationNeverReceivesCredit) {
  KVShadowIndexConfig config = make_config();
  config.event_ttl_ms = 10;
  KVShadowIndex index(config);
  const xllm::proto::KVStreamIdentity identity = make_identity();
  xllm::proto::KVEventBatch stale = make_batch(
      identity,
      {make_event(1, xllm::proto::KV_EVENT_KIND_STORED, make_block(1))});
  stale.set_batch_age_ms_at_publish(11);
  EXPECT_EQ(index.apply_event_batch(stale, 100).code,
            KVApplyCode::SNAPSHOT_REQUIRED);
  EXPECT_EQ(index.health(identity), KVShadowHealth::UNKNOWN);
  EXPECT_EQ(index.resident_entries(identity), 0);
}

TEST(KVShadowIndexTest, ModelAndNamespaceAreStrictIsolationKeys) {
  KVShadowIndex index(make_config());
  const xllm::proto::KVBlockEntry block = make_block(1);
  const xllm::proto::KVStreamIdentity first = make_identity();
  const xllm::proto::KVStreamIdentity second =
      make_identity("incarnation-a", "model-b", "namespace-b");
  ASSERT_EQ(
      index
          .apply_event_batch(
              make_batch(
                  first,
                  {make_event(1, xllm::proto::KV_EVENT_KIND_STORED, block)}),
              100)
          .code,
      KVApplyCode::APPLIED);
  ASSERT_EQ(
      index
          .apply_event_batch(
              make_batch(
                  second,
                  {make_event(1, xllm::proto::KV_EVENT_KIND_STORED, block)}),
              101)
          .code,
      KVApplyCode::APPLIED);
  EXPECT_EQ(index.stats().engine_streams, 2);
  EXPECT_TRUE(index.contains(
      first, block.block_hash(), block.cache_group(), block.tier()));
  EXPECT_TRUE(index.contains(
      second, block.block_hash(), block.cache_group(), block.tier()));
}

TEST(KVShadowIndexTest, EngineStreamCountIsBoundedAndIncarnationFreesSpace) {
  KVShadowIndexConfig config = make_config();
  config.max_engine_streams = 1;
  KVShadowIndex index(config);
  const xllm::proto::KVStreamIdentity first = make_identity();
  ASSERT_EQ(index
                .apply_event_batch(
                    make_batch(first,
                               {make_event(1,
                                           xllm::proto::KV_EVENT_KIND_STORED,
                                           make_block(1))}),
                    100)
                .code,
            KVApplyCode::APPLIED);

  const xllm::proto::KVStreamIdentity second_namespace =
      make_identity("incarnation-a", "model-a", "namespace-b");
  EXPECT_EQ(index
                .apply_event_batch(
                    make_batch(second_namespace,
                               {make_event(1,
                                           xllm::proto::KV_EVENT_KIND_STORED,
                                           make_block(2))}),
                    101)
                .code,
            KVApplyCode::REJECTED);
  EXPECT_EQ(index.stats().engine_streams, 1);

  const xllm::proto::KVStreamIdentity replacement =
      make_identity("incarnation-b");
  EXPECT_EQ(index
                .apply_event_batch(
                    make_batch(replacement,
                               {make_event(1,
                                           xllm::proto::KV_EVENT_KIND_STORED,
                                           make_block(3))}),
                    102)
                .code,
            KVApplyCode::APPLIED);
  EXPECT_EQ(index.stats().engine_streams, 1);
  EXPECT_EQ(index.health(first), KVShadowHealth::UNKNOWN);
  EXPECT_EQ(index.health(replacement), KVShadowHealth::READY);
}

TEST(KVShadowIndexTest, ConcurrentDuplicateDeliveryIsLinearizable) {
  KVShadowIndex index(make_config());
  const xllm::proto::KVStreamIdentity identity = make_identity();
  const xllm::proto::KVEventBatch batch = make_batch(
      identity,
      {make_event(1, xllm::proto::KV_EVENT_KIND_STORED, make_block(1))});
  std::vector<std::thread> threads;
  threads.reserve(8);
  for (uint64_t index_value = 0; index_value < 8; ++index_value) {
    threads.emplace_back([&index, &batch, index_value]() {
      const KVApplyResult applied =
          index.apply_event_batch(batch, 100 + index_value);
      EXPECT_TRUE(applied.code == KVApplyCode::APPLIED ||
                  applied.code == KVApplyCode::DUPLICATE);
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  const KVShadowIndexStats stats = index.stats();
  EXPECT_EQ(stats.ready_streams, 1);
  EXPECT_EQ(stats.index_entries, 1);
}

}  // namespace
}  // namespace xllm_service::provider
