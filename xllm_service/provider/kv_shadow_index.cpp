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

#include <algorithm>
#include <utility>

namespace xllm_service::provider {
namespace {

inline constexpr uint32_t kKVEventContractVersion = 1;
inline constexpr size_t kBlockHashBytes = 16;
inline constexpr size_t kEntryAccountingOverhead = 96;
inline constexpr size_t kEventAccountingOverhead = 64;

bool valid_identity(const xllm::proto::KVStreamIdentity& identity) {
  const xllm::proto::ProviderEngineKey& engine = identity.engine();
  return xllm::proto::ProviderId_IsValid(engine.provider_id()) &&
         engine.provider_id() != xllm::proto::PROVIDER_ID_UNSPECIFIED &&
         !engine.profile_digest().empty() && !engine.engine_uid().empty() &&
         !engine.incarnation_id().empty() &&
         !identity.model_revision().empty() &&
         !identity.kv_namespace().empty() && identity.cache_epoch() > 0;
}

bool valid_block(const xllm::proto::KVBlockEntry& block) {
  return block.block_hash().size() == kBlockHashBytes &&
         (block.parent_hash().empty() ||
          block.parent_hash().size() == kBlockHashBytes) &&
         block.token_end() > block.token_begin() &&
         !block.cache_group().empty() &&
         xllm::proto::KVCacheTier_IsValid(block.tier()) &&
         block.tier() != xllm::proto::KV_CACHE_TIER_UNSPECIFIED;
}

bool valid_event(const xllm::proto::KVEvent& event) {
  if (event.event_seq() == 0 ||
      !xllm::proto::KVEventKind_IsValid(event.kind()) ||
      event.kind() == xllm::proto::KV_EVENT_KIND_UNSPECIFIED ||
      !xllm::proto::KVEventReason_IsValid(event.reason()) ||
      event.reason() == xllm::proto::KV_EVENT_REASON_UNSPECIFIED) {
    return false;
  }
  if (event.kind() == xllm::proto::KV_EVENT_KIND_CLEARED) {
    return !event.has_block();
  }
  return event.has_block() && valid_block(event.block());
}

bool same_stream(const xllm::proto::KVStreamIdentity& left,
                 const xllm::proto::KVStreamIdentity& right) {
  return left.engine().SerializeAsString() ==
             right.engine().SerializeAsString() &&
         left.model_revision() == right.model_revision() &&
         left.kv_namespace() == right.kv_namespace();
}

}  // namespace

KVShadowIndex::KVShadowIndex(KVShadowIndexConfig config)
    : config_(std::move(config)) {
  config_valid_ = config_.max_engine_streams > 0 &&
                  config_.max_index_entries > 0 &&
                  config_.max_index_bytes > 0 &&
                  config_.max_recovery_events_per_engine > 0 &&
                  config_.max_recovery_bytes_per_engine > 0 &&
                  config_.max_snapshot_entries_per_engine > 0 &&
                  config_.max_snapshot_bytes_per_engine > 0 &&
                  config_.event_ttl_ms > 0 && config_.recovery_timeout_ms > 0;
}

std::string KVShadowIndex::stream_key(
    const xllm::proto::KVStreamIdentity& identity) {
  xllm::proto::KVStreamIdentity ordering_domain = identity;
  ordering_domain.set_cache_epoch(0);
  return ordering_domain.SerializeAsString();
}

std::string KVShadowIndex::entry_key(const xllm::proto::KVBlockEntry& entry) {
  return entry.SerializeAsString();
}

size_t KVShadowIndex::entry_bytes(const xllm::proto::KVBlockEntry& entry) {
  return entry.ByteSizeLong() + kEntryAccountingOverhead;
}

KVApplyResult KVShadowIndex::result(KVApplyCode code,
                                    std::string reason,
                                    uint64_t accepted_through_event_seq) {
  return KVApplyResult{
      .code = code,
      .reason = std::move(reason),
      .accepted_through_event_seq = accepted_through_event_seq,
  };
}

void KVShadowIndex::fence_old_incarnations_locked(
    const xllm::proto::KVStreamIdentity& identity) {
  for (auto it = streams_.begin(); it != streams_.end();) {
    const xllm::proto::ProviderEngineKey& existing =
        it->second.identity.engine();
    const xllm::proto::ProviderEngineKey& incoming = identity.engine();
    if (existing.provider_id() == incoming.provider_id() &&
        existing.engine_uid() == incoming.engine_uid() &&
        existing.incarnation_id() != incoming.incarnation_id()) {
      index_entries_ -= it->second.live.size();
      index_bytes_ -= it->second.live_bytes;
      it = streams_.erase(it);
    } else {
      ++it;
    }
  }
}

KVShadowIndex::EngineShadow* KVShadowIndex::find_or_create_stream_locked(
    const xllm::proto::KVStreamIdentity& identity,
    uint64_t now_monotonic_ms) {
  fence_old_incarnations_locked(identity);
  const std::string key = stream_key(identity);
  auto found = streams_.find(key);
  if (found != streams_.end()) {
    return &found->second;
  }
  if (streams_.size() >= config_.max_engine_streams) {
    return nullptr;
  }
  EngineShadow created;
  created.identity = identity;
  created.last_confirmed_monotonic_ms = now_monotonic_ms;
  return &streams_.emplace(key, std::move(created)).first->second;
}

void KVShadowIndex::make_unknown_locked(EngineShadow* shadow) {
  index_entries_ -= shadow->live.size();
  index_bytes_ -= shadow->live_bytes;
  shadow->live.clear();
  shadow->live_bytes = 0;
  shadow->health = KVShadowHealth::UNKNOWN;
  shadow->recovery.reset();
}

bool KVShadowIndex::index_capacity_available_locked(size_t new_entries,
                                                    size_t new_bytes) const {
  return new_entries <= config_.max_index_entries - index_entries_ &&
         new_bytes <= config_.max_index_bytes - index_bytes_;
}

bool KVShadowIndex::apply_event_locked(EngineShadow* shadow,
                                       const xllm::proto::KVEvent& event) {
  if (event.kind() == xllm::proto::KV_EVENT_KIND_CLEARED) {
    return false;
  }
  const std::string key = entry_key(event.block());
  const auto existing = shadow->live.find(key);
  if (event.kind() == xllm::proto::KV_EVENT_KIND_STORED) {
    if (existing != shadow->live.end()) {
      return true;
    }
    const size_t bytes = entry_bytes(event.block());
    if (!index_capacity_available_locked(/*new_entries=*/1, bytes)) {
      return false;
    }
    shadow->live.emplace(key, event.block());
    shadow->live_bytes += bytes;
    ++index_entries_;
    index_bytes_ += bytes;
    return true;
  }
  if (event.kind() != xllm::proto::KV_EVENT_KIND_REMOVED) {
    return false;
  }
  if (existing != shadow->live.end()) {
    const size_t bytes = entry_bytes(existing->second);
    shadow->live_bytes -= bytes;
    index_bytes_ -= bytes;
    --index_entries_;
    shadow->live.erase(existing);
  }
  return true;
}

bool KVShadowIndex::buffer_recovery_event_locked(
    EngineShadow* shadow,
    const xllm::proto::KVEvent& event) {
  RecoveryState& recovery = *shadow->recovery;
  if (event.event_seq() <= recovery.base_event_seq ||
      recovery.buffered.find(event.event_seq()) != recovery.buffered.end()) {
    return true;
  }
  const size_t bytes = event.ByteSizeLong() + kEventAccountingOverhead;
  if (recovery.buffered.size() >= config_.max_recovery_events_per_engine ||
      bytes > config_.max_recovery_bytes_per_engine ||
      recovery.buffered_bytes > config_.max_recovery_bytes_per_engine - bytes) {
    return false;
  }
  recovery.buffered.emplace(event.event_seq(), event);
  recovery.buffered_bytes += bytes;
  return true;
}

KVApplyResult KVShadowIndex::apply_event_batch(
    const xllm::proto::KVEventBatch& batch,
    uint64_t received_monotonic_ms) {
  if (!config_valid_ || received_monotonic_ms == 0 ||
      batch.contract_version() != kKVEventContractVersion ||
      !valid_identity(batch.identity()) ||
      !batch.has_batch_age_ms_at_publish()) {
    return result(KVApplyCode::REJECTED, "KV event batch is invalid", 0);
  }
  uint64_t previous_sequence = 0;
  bool observation_expired =
      batch.batch_age_ms_at_publish() > config_.event_ttl_ms;
  for (const xllm::proto::KVEvent& event : batch.events()) {
    if (!valid_event(event) || event.event_seq() <= previous_sequence) {
      return result(
          KVApplyCode::REJECTED, "KV event batch ordering is invalid", 0);
    }
    if (event.has_event_age_ms_at_publish() &&
        event.event_age_ms_at_publish() > config_.event_ttl_ms) {
      observation_expired = true;
    }
    previous_sequence = event.event_seq();
  }
  if (previous_sequence > batch.last_event_seq()) {
    return result(
        KVApplyCode::REJECTED, "KV event batch high watermark is invalid", 0);
  }

  std::lock_guard<std::mutex> lock(mutex_);
  const std::string key = stream_key(batch.identity());
  const bool is_new_stream = streams_.find(key) == streams_.end();
  EngineShadow* shadow =
      find_or_create_stream_locked(batch.identity(), received_monotonic_ms);
  if (shadow == nullptr) {
    return result(
        KVApplyCode::REJECTED, "KV shadow stream capacity is exhausted", 0);
  }
  if (observation_expired) {
    make_unknown_locked(shadow);
    shadow->last_confirmed_monotonic_ms = received_monotonic_ms;
    return result(KVApplyCode::SNAPSHOT_REQUIRED,
                  "KV event observation exceeded its TTL",
                  shadow->last_event_seq);
  }
  if (batch.identity().cache_epoch() < shadow->identity.cache_epoch()) {
    return result(KVApplyCode::DUPLICATE,
                  "KV event epoch is stale",
                  shadow->last_event_seq);
  }

  if (batch.identity().cache_epoch() > shadow->identity.cache_epoch()) {
    const bool valid_clear =
        !batch.gap_before_events() && !batch.events().empty() &&
        batch.events(0).event_seq() == 1 &&
        batch.events(0).kind() == xllm::proto::KV_EVENT_KIND_CLEARED;
    make_unknown_locked(shadow);
    shadow->identity = batch.identity();
    shadow->last_event_seq = 0;
    if (!valid_clear) {
      shadow->last_confirmed_monotonic_ms = received_monotonic_ms;
      return result(KVApplyCode::SNAPSHOT_REQUIRED,
                    "KV cache epoch changed without a complete clear",
                    0);
    }
    shadow->health = KVShadowHealth::READY;
  }

  if (batch.gap_before_events()) {
    make_unknown_locked(shadow);
    shadow->last_confirmed_monotonic_ms = received_monotonic_ms;
    return result(KVApplyCode::SNAPSHOT_REQUIRED,
                  "KV event lane reported a gap",
                  shadow->last_event_seq);
  }

  if (is_new_stream) {
    if (batch.events().empty() || batch.events(0).event_seq() != 1) {
      make_unknown_locked(shadow);
      return result(KVApplyCode::SNAPSHOT_REQUIRED,
                    "KV stream did not start at sequence one",
                    0);
    }
    shadow->health = KVShadowHealth::READY;
  }

  if (shadow->health == KVShadowHealth::UNKNOWN) {
    return result(KVApplyCode::SNAPSHOT_REQUIRED,
                  "KV shadow is unknown",
                  shadow->last_event_seq);
  }
  if (shadow->health == KVShadowHealth::RECOVERING) {
    for (const xllm::proto::KVEvent& event : batch.events()) {
      if (!buffer_recovery_event_locked(shadow, event)) {
        make_unknown_locked(shadow);
        return result(KVApplyCode::SNAPSHOT_REQUIRED,
                      "KV recovery event buffer overflowed",
                      shadow->last_event_seq);
      }
    }
    shadow->last_confirmed_monotonic_ms = received_monotonic_ms;
    return result(KVApplyCode::RECOVERING,
                  "KV events buffered during snapshot recovery",
                  shadow->last_event_seq);
  }

  bool applied = false;
  for (const xllm::proto::KVEvent& event : batch.events()) {
    if (event.event_seq() <= shadow->last_event_seq) {
      continue;
    }
    if (event.event_seq() != shadow->last_event_seq + 1 ||
        (event.kind() == xllm::proto::KV_EVENT_KIND_CLEARED &&
         event.event_seq() != 1) ||
        (event.kind() != xllm::proto::KV_EVENT_KIND_CLEARED &&
         !apply_event_locked(shadow, event))) {
      make_unknown_locked(shadow);
      return result(KVApplyCode::SNAPSHOT_REQUIRED,
                    "KV event sequence or capacity is not recoverable",
                    shadow->last_event_seq);
    }
    shadow->last_event_seq = event.event_seq();
    applied = true;
  }
  if (batch.events().empty() &&
      batch.last_event_seq() > shadow->last_event_seq) {
    make_unknown_locked(shadow);
    return result(KVApplyCode::SNAPSHOT_REQUIRED,
                  "KV keepalive exposed an event sequence gap",
                  shadow->last_event_seq);
  }
  shadow->last_confirmed_monotonic_ms = received_monotonic_ms;
  return result(
      applied ? KVApplyCode::APPLIED : KVApplyCode::DUPLICATE,
      applied ? "KV event batch applied" : "KV event batch was already applied",
      shadow->last_event_seq);
}

KVApplyResult KVShadowIndex::begin_recovery(
    const xllm::proto::KVStreamIdentity& identity,
    uint64_t started_monotonic_ms) {
  if (!config_valid_ || started_monotonic_ms == 0 ||
      !valid_identity(identity)) {
    return result(KVApplyCode::REJECTED, "KV recovery identity is invalid", 0);
  }
  std::lock_guard<std::mutex> lock(mutex_);
  EngineShadow* shadow =
      find_or_create_stream_locked(identity, started_monotonic_ms);
  if (shadow == nullptr) {
    return result(
        KVApplyCode::REJECTED, "KV shadow stream capacity is exhausted", 0);
  }
  if (identity.cache_epoch() < shadow->identity.cache_epoch()) {
    return result(KVApplyCode::REJECTED,
                  "KV recovery epoch is stale",
                  shadow->last_event_seq);
  }
  if (identity.cache_epoch() > shadow->identity.cache_epoch()) {
    make_unknown_locked(shadow);
    shadow->identity = identity;
    shadow->last_event_seq = 0;
  } else {
    make_unknown_locked(shadow);
  }
  shadow->health = KVShadowHealth::RECOVERING;
  shadow->last_confirmed_monotonic_ms = started_monotonic_ms;
  shadow->recovery = RecoveryState{
      .started_monotonic_ms = started_monotonic_ms,
  };
  return result(KVApplyCode::RECOVERING,
                "KV snapshot recovery started",
                shadow->last_event_seq);
}

bool KVShadowIndex::replace_live_from_recovery_locked(EngineShadow* shadow) {
  RecoveryState& recovery = *shadow->recovery;
  if (!index_capacity_available_locked(recovery.staging.size(),
                                       recovery.staging_bytes)) {
    return false;
  }
  shadow->live = std::move(recovery.staging);
  shadow->live_bytes = recovery.staging_bytes;
  index_entries_ += shadow->live.size();
  index_bytes_ += shadow->live_bytes;
  shadow->last_event_seq = recovery.base_event_seq;
  shadow->health = KVShadowHealth::READY;
  return true;
}

KVApplyResult KVShadowIndex::apply_snapshot_page(
    const xllm::proto::KVCacheSnapshotPage& page,
    uint64_t received_monotonic_ms) {
  if (!config_valid_ || received_monotonic_ms == 0 ||
      page.contract_version() != kKVEventContractVersion ||
      page.status() != xllm::proto::KV_SNAPSHOT_STATUS_OK ||
      !valid_identity(page.identity()) || page.snapshot_id().empty()) {
    return result(KVApplyCode::REJECTED, "KV snapshot page is invalid", 0);
  }

  std::lock_guard<std::mutex> lock(mutex_);
  auto found = streams_.find(stream_key(page.identity()));
  if (found == streams_.end() ||
      found->second.health != KVShadowHealth::RECOVERING ||
      !found->second.recovery.has_value() ||
      !same_stream(found->second.identity, page.identity()) ||
      found->second.identity.cache_epoch() != page.identity().cache_epoch()) {
    return result(
        KVApplyCode::REJECTED, "KV snapshot has no matching recovery", 0);
  }
  EngineShadow* shadow = &found->second;
  RecoveryState& recovery = *shadow->recovery;
  const bool is_first_page = recovery.snapshot_id.empty();
  if (recovery.snapshot_id.empty()) {
    recovery.snapshot_id = page.snapshot_id();
    recovery.base_event_seq = page.base_event_seq();
  } else if (recovery.snapshot_id != page.snapshot_id() ||
             recovery.base_event_seq != page.base_event_seq()) {
    make_unknown_locked(shadow);
    return result(KVApplyCode::SNAPSHOT_REQUIRED,
                  "KV snapshot identity changed between pages",
                  shadow->last_event_seq);
  }

  if (!is_first_page && page.next_cursor() <= recovery.next_cursor) {
    return result(KVApplyCode::DUPLICATE,
                  "KV snapshot page was already applied",
                  shadow->last_event_seq);
  }
  const uint64_t expected_cursor =
      recovery.next_cursor + static_cast<uint64_t>(page.entries_size());
  if (page.next_cursor() != expected_cursor) {
    make_unknown_locked(shadow);
    return result(KVApplyCode::SNAPSHOT_REQUIRED,
                  "KV snapshot cursor is discontinuous",
                  shadow->last_event_seq);
  }

  for (const xllm::proto::KVBlockEntry& entry : page.entries()) {
    if (!valid_block(entry)) {
      make_unknown_locked(shadow);
      return result(KVApplyCode::SNAPSHOT_REQUIRED,
                    "KV snapshot contains an invalid block",
                    shadow->last_event_seq);
    }
    const std::string key = entry_key(entry);
    if (recovery.staging.find(key) != recovery.staging.end()) {
      continue;
    }
    const size_t bytes = entry_bytes(entry);
    if (recovery.staging.size() >= config_.max_snapshot_entries_per_engine ||
        bytes > config_.max_snapshot_bytes_per_engine ||
        recovery.staging_bytes >
            config_.max_snapshot_bytes_per_engine - bytes) {
      make_unknown_locked(shadow);
      return result(KVApplyCode::SNAPSHOT_REQUIRED,
                    "KV snapshot staging capacity is exhausted",
                    shadow->last_event_seq);
    }
    recovery.staging.emplace(key, entry);
    recovery.staging_bytes += bytes;
  }
  recovery.next_cursor = page.next_cursor();
  shadow->last_confirmed_monotonic_ms = received_monotonic_ms;
  if (!page.done()) {
    return result(KVApplyCode::RECOVERING,
                  "KV snapshot page applied",
                  shadow->last_event_seq);
  }

  const std::map<uint64_t, xllm::proto::KVEvent> buffered = recovery.buffered;
  if (!replace_live_from_recovery_locked(shadow)) {
    make_unknown_locked(shadow);
    return result(KVApplyCode::SNAPSHOT_REQUIRED,
                  "KV index capacity cannot hold the snapshot",
                  shadow->last_event_seq);
  }
  for (const auto& [sequence, event] : buffered) {
    if (sequence <= shadow->last_event_seq) {
      continue;
    }
    if (sequence != shadow->last_event_seq + 1 ||
        !apply_event_locked(shadow, event)) {
      make_unknown_locked(shadow);
      return result(KVApplyCode::SNAPSHOT_REQUIRED,
                    "KV recovery increment replay has a gap",
                    shadow->last_event_seq);
    }
    shadow->last_event_seq = sequence;
  }
  shadow->recovery.reset();
  return result(KVApplyCode::APPLIED,
                "KV snapshot recovery completed",
                shadow->last_event_seq);
}

void KVShadowIndex::abort_recovery(
    const xllm::proto::KVStreamIdentity& identity) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto found = streams_.find(stream_key(identity));
  if (found != streams_.end() &&
      found->second.identity.cache_epoch() == identity.cache_epoch() &&
      found->second.health == KVShadowHealth::RECOVERING) {
    make_unknown_locked(&found->second);
  }
}

void KVShadowIndex::reset_all() {
  std::lock_guard<std::mutex> lock(mutex_);
  streams_.clear();
  index_entries_ = 0;
  index_bytes_ = 0;
}

void KVShadowIndex::expire(uint64_t now_monotonic_ms) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& [key, shadow] : streams_) {
    static_cast<void>(key);
    const bool clock_regressed =
        now_monotonic_ms < shadow.last_confirmed_monotonic_ms;
    const uint64_t age =
        clock_regressed ? config_.event_ttl_ms + 1
                        : now_monotonic_ms - shadow.last_confirmed_monotonic_ms;
    const bool recovery_timed_out =
        shadow.health == KVShadowHealth::RECOVERING &&
        shadow.recovery.has_value() &&
        (now_monotonic_ms < shadow.recovery->started_monotonic_ms ||
         now_monotonic_ms - shadow.recovery->started_monotonic_ms >
             config_.recovery_timeout_ms);
    if (age > config_.event_ttl_ms || recovery_timed_out) {
      make_unknown_locked(&shadow);
    }
  }
}

KVShadowHealth KVShadowIndex::health(
    const xllm::proto::KVStreamIdentity& identity) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto found = streams_.find(stream_key(identity));
  if (found == streams_.end() ||
      found->second.identity.cache_epoch() != identity.cache_epoch()) {
    return KVShadowHealth::UNKNOWN;
  }
  return found->second.health;
}

bool KVShadowIndex::contains(const xllm::proto::KVStreamIdentity& identity,
                             const std::string& block_hash,
                             const std::string& cache_group,
                             xllm::proto::KVCacheTier tier) const {
  if (block_hash.size() != kBlockHashBytes || cache_group.empty()) {
    return false;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  const auto found = streams_.find(stream_key(identity));
  if (found == streams_.end() ||
      found->second.identity.cache_epoch() != identity.cache_epoch() ||
      found->second.health != KVShadowHealth::READY) {
    return false;
  }
  for (const auto& [key, entry] : found->second.live) {
    static_cast<void>(key);
    if (entry.block_hash() == block_hash &&
        entry.cache_group() == cache_group && entry.tier() == tier) {
      return true;
    }
  }
  return false;
}

size_t KVShadowIndex::resident_entries(
    const xllm::proto::KVStreamIdentity& identity) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto found = streams_.find(stream_key(identity));
  return found == streams_.end() ? 0 : found->second.live.size();
}

std::vector<xllm::proto::KVStreamIdentity> KVShadowIndex::snapshot_required()
    const {
  std::vector<xllm::proto::KVStreamIdentity> identities;
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& [key, shadow] : streams_) {
    static_cast<void>(key);
    if (shadow.health == KVShadowHealth::UNKNOWN) {
      identities.emplace_back(shadow.identity);
    }
  }
  std::sort(identities.begin(),
            identities.end(),
            [](const xllm::proto::KVStreamIdentity& left,
               const xllm::proto::KVStreamIdentity& right) {
              return left.SerializeAsString() < right.SerializeAsString();
            });
  return identities;
}

KVShadowIndexStats KVShadowIndex::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  KVShadowIndexStats current;
  current.engine_streams = streams_.size();
  current.index_entries = index_entries_;
  current.index_bytes = index_bytes_;
  for (const auto& [key, shadow] : streams_) {
    static_cast<void>(key);
    if (shadow.health == KVShadowHealth::READY) {
      ++current.ready_streams;
    } else if (shadow.health == KVShadowHealth::RECOVERING) {
      ++current.recovering_streams;
    } else {
      ++current.unknown_streams;
    }
    if (shadow.recovery.has_value()) {
      current.recovery_events += shadow.recovery->buffered.size();
      current.recovery_bytes += shadow.recovery->buffered_bytes;
    }
  }
  return current;
}

}  // namespace xllm_service::provider
