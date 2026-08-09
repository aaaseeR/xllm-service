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

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "provider.pb.h"

namespace xllm_service::provider {

enum class KVShadowHealth : int8_t {
  UNKNOWN = 0,
  RECOVERING = 1,
  READY = 2,
};

enum class KVApplyCode : int8_t {
  APPLIED = 0,
  DUPLICATE = 1,
  SNAPSHOT_REQUIRED = 2,
  RECOVERING = 3,
  REJECTED = 4,
};

struct KVShadowIndexConfig {
  size_t max_engine_streams = 4096;
  size_t max_index_entries = 1048576;
  size_t max_index_bytes = 512 * 1024 * 1024;
  size_t max_recovery_events_per_engine = 8192;
  size_t max_recovery_bytes_per_engine = 8 * 1024 * 1024;
  size_t max_snapshot_entries_per_engine = 262144;
  size_t max_snapshot_bytes_per_engine = 128 * 1024 * 1024;
  uint64_t event_ttl_ms = 30000;
  uint64_t recovery_timeout_ms = 30000;
};

struct KVApplyResult {
  KVApplyCode code = KVApplyCode::REJECTED;
  std::string reason;
  uint64_t accepted_through_event_seq = 0;
};

struct KVShadowIndexStats {
  size_t engine_streams = 0;
  size_t ready_streams = 0;
  size_t recovering_streams = 0;
  size_t unknown_streams = 0;
  size_t index_entries = 0;
  size_t index_bytes = 0;
  size_t recovery_events = 0;
  size_t recovery_bytes = 0;
};

struct KVPrefixMatch {
  KVShadowHealth health = KVShadowHealth::UNKNOWN;
  size_t contiguous_blocks = 0;
};

// Bounded, fail-closed Service-side KV observation. It is deliberately not an
// allocator or admission ledger: UNKNOWN and RECOVERING always mean zero KV
// routing credit while normal load routing remains available.
class KVShadowIndex final {
 public:
  explicit KVShadowIndex(KVShadowIndexConfig config);

  KVApplyResult apply_event_batch(const xllm::proto::KVEventBatch& batch,
                                  uint64_t received_monotonic_ms);
  KVApplyResult begin_recovery(const xllm::proto::KVStreamIdentity& identity,
                               uint64_t started_monotonic_ms);
  KVApplyResult apply_snapshot_page(
      const xllm::proto::KVCacheSnapshotPage& page,
      uint64_t received_monotonic_ms);
  void abort_recovery(const xllm::proto::KVStreamIdentity& identity);

  void reset_all();
  void expire(uint64_t now_monotonic_ms);

  KVShadowHealth health(const xllm::proto::KVStreamIdentity& identity) const;
  bool contains(const xllm::proto::KVStreamIdentity& identity,
                const std::string& block_hash,
                const std::string& cache_group,
                xllm::proto::KVCacheTier tier) const;
  KVPrefixMatch match_contiguous_prefix(
      const xllm::proto::KVStreamIdentity& identity,
      const std::vector<std::string>& block_hashes,
      const std::string& cache_group,
      xllm::proto::KVCacheTier tier) const;
  KVPrefixMatch match_current_contiguous_prefix(
      const xllm::proto::ProviderEngineKey& engine,
      const std::string& model_revision,
      const std::string& kv_namespace,
      const std::vector<std::string>& block_hashes,
      const std::string& cache_group,
      xllm::proto::KVCacheTier tier) const;
  size_t resident_entries(const xllm::proto::KVStreamIdentity& identity) const;
  std::vector<xllm::proto::KVStreamIdentity> snapshot_required() const;
  KVShadowIndexStats stats() const;

 private:
  struct RecoveryState {
    std::string snapshot_id;
    uint64_t base_event_seq = 0;
    uint64_t next_cursor = 0;
    uint64_t started_monotonic_ms = 0;
    size_t staging_bytes = 0;
    size_t buffered_bytes = 0;
    std::unordered_map<std::string, xllm::proto::KVBlockEntry> staging;
    std::map<uint64_t, xllm::proto::KVEvent> buffered;
  };

  struct EngineShadow {
    xllm::proto::KVStreamIdentity identity;
    KVShadowHealth health = KVShadowHealth::UNKNOWN;
    uint64_t last_event_seq = 0;
    uint64_t last_confirmed_monotonic_ms = 0;
    size_t live_bytes = 0;
    std::unordered_map<std::string, xllm::proto::KVBlockEntry> live;
    std::unordered_map<std::string, size_t> lookup_counts;
    std::optional<RecoveryState> recovery;
  };

  static std::string stream_key(const xllm::proto::KVStreamIdentity& identity);
  static std::string entry_key(const xllm::proto::KVBlockEntry& entry);
  static std::string lookup_key(const std::string& block_hash,
                                const std::string& cache_group,
                                xllm::proto::KVCacheTier tier);
  static size_t entry_bytes(const xllm::proto::KVBlockEntry& entry);

  EngineShadow* find_or_create_stream_locked(
      const xllm::proto::KVStreamIdentity& identity,
      uint64_t now_monotonic_ms);
  void fence_old_incarnations_locked(
      const xllm::proto::KVStreamIdentity& identity);
  void make_unknown_locked(EngineShadow* shadow);
  bool buffer_recovery_event_locked(EngineShadow* shadow,
                                    const xllm::proto::KVEvent& event);
  bool apply_event_locked(EngineShadow* shadow,
                          const xllm::proto::KVEvent& event);
  bool replace_live_from_recovery_locked(EngineShadow* shadow);
  bool index_capacity_available_locked(size_t new_entries,
                                       size_t new_bytes) const;
  static KVApplyResult result(KVApplyCode code,
                              std::string reason,
                              uint64_t accepted_through_event_seq);

  KVShadowIndexConfig config_;
  bool config_valid_ = false;
  mutable std::mutex mutex_;
  size_t index_entries_ = 0;
  size_t index_bytes_ = 0;
  std::unordered_map<std::string, EngineShadow> streams_;
};

}  // namespace xllm_service::provider
