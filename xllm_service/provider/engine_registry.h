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
#include <memory>
#include <optional>
#include <set>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "provider.pb.h"
#include "provider/observation_controller.h"
#include "provider/provider_contract.h"

namespace xllm_service::provider {

struct EngineRegistryConfig {
  size_t max_members = 4096;
  size_t max_links = 16384;
  uint64_t state_soft_ttl_ms = 3000;
  uint64_t state_hard_ttl_ms = 10000;
  uint64_t heartbeat_hard_ttl_ms = 10000;
  uint64_t link_hard_ttl_ms = 10000;
  uint64_t direct_evidence_ttl_ms = 3000;
  ObservationControllerConfig observation;
};

enum class EngineStateFreshness : int8_t {
  MISSING = 0,
  FRESH = 1,
  SOFT_STALE = 2,
  HARD_STALE = 3,
};

struct EngineKVCapacitySnapshot {
  size_t reporting_engines = 0;
  size_t reporting_dp_ranks = 0;
  double max_used_ratio = 0.0;
  uint64_t min_free_blocks = 0;
  uint64_t total_free_blocks = 0;
  bool has_used_ratio = false;
  bool has_free_blocks = false;
};

struct EngineRegistryMemberSnapshot {
  xllm::proto::ProviderDescriptor descriptor;
  std::optional<xllm::proto::EngineState> state;
  EngineStateFreshness state_freshness = EngineStateFreshness::MISSING;
  bool heartbeat_fresh = false;
  bool schedulable = false;
  uint64_t lifecycle_since_monotonic_ms = 0;
};

// Immutable request-path view. Registry ingestion/readiness refresh builds the
// shared Data once and atomically publishes it; copying this handle and all
// P/D candidate/link lookups avoid the Registry lock and protobuf
// serialization.
class EngineRegistryRoutingSnapshot final {
 public:
  bool is_schedulable(const xllm::proto::ProviderEngineKey& key) const;
  bool is_link_ready(const xllm::proto::ProviderEngineKey& prefill,
                     const xllm::proto::ProviderEngineKey& decode) const;

 private:
  friend class EngineRegistry;

  struct EngineIdentity {
    int provider_id = 0;
    std::string profile_digest;
    std::string engine_uid;
    std::string incarnation_id;
  };

  struct EngineIdentityLess {
    using is_transparent = void;

    bool operator()(const EngineIdentity& left,
                    const EngineIdentity& right) const;
    bool operator()(const EngineIdentity& left,
                    const xllm::proto::ProviderEngineKey& right) const;
    bool operator()(const xllm::proto::ProviderEngineKey& left,
                    const EngineIdentity& right) const;
  };

  struct Data {
    std::map<EngineIdentity, uint32_t, EngineIdentityLess> engine_indices;
    // Exclusive per-entry expiry prevents one stale member or link from
    // invalidating unrelated healthy routes between readiness refreshes.
    std::vector<uint64_t> schedulable_until_monotonic_ms;
    std::unordered_map<uint64_t, uint64_t> ready_links_until_monotonic_ms;
  };

  explicit EngineRegistryRoutingSnapshot(std::shared_ptr<const Data> data,
                                         uint64_t receiver_monotonic_ms)
      : data_(std::move(data)), receiver_monotonic_ms_(receiver_monotonic_ms) {}

  std::shared_ptr<const Data> data_;
  uint64_t receiver_monotonic_ms_ = 0;
};

xllm::proto::ProviderEngineKey make_provider_engine_key(
    const xllm::proto::ProviderDescriptor& descriptor);

// Holds the authoritative Registry membership separately from soft State
// Stream observations. State updates can never create or resurrect a member.
class EngineRegistry final {
 public:
  explicit EngineRegistry(EngineRegistryConfig config);

  ContractResult upsert_member(
      const xllm::proto::ProviderDescriptor& descriptor);
  bool remove_member(const xllm::proto::ProviderEngineKey& key);

  // Registry visibility and State Stream master identity are independent.
  // A master change retains the last legal snapshot for conservative routing
  // but requires a new FULL before subsequent DELTA batches are accepted.
  ContractResult set_registry_visibility(bool registry_known);
  ContractResult set_state_stream_master(std::string master_incarnation);
  // Master-side ingestion. These methods update only soft observations for an
  // existing Registry member and report whether a newer sequence was stored.
  ContractResult record_engine_state(const xllm::proto::EngineState& state,
                                     uint64_t receiver_monotonic_ms,
                                     bool* applied);
  ContractResult record_link_state(const xllm::proto::LinkState& state,
                                   uint64_t receiver_monotonic_ms,
                                   bool* applied);
  ContractResult build_full_state_batch(const std::string& master_incarnation,
                                        uint64_t snapshot_seq,
                                        uint64_t publish_monotonic_ms,
                                        xllm::proto::StateBatch* batch) const;
  ContractResult apply_state_batch(const xllm::proto::StateBatch& batch,
                                   uint64_t receiver_monotonic_ms,
                                   bool* applied);
  ContractResult record_direct_evidence(
      const xllm::proto::ProviderEngineKey& key,
      bool success,
      uint64_t receiver_monotonic_ms);

  std::optional<xllm::proto::ProviderDescriptor> find_member(
      const xllm::proto::ProviderEngineKey& key) const;
  std::optional<xllm::proto::EngineState> find_state(
      const xllm::proto::ProviderEngineKey& key) const;
  EngineStateFreshness state_freshness(
      const xllm::proto::ProviderEngineKey& key,
      uint64_t receiver_monotonic_ms) const;
  bool is_schedulable(const xllm::proto::ProviderEngineKey& key,
                      uint64_t receiver_monotonic_ms) const;
  bool is_link_ready(const xllm::proto::ProviderEngineKey& prefill,
                     const xllm::proto::ProviderEngineKey& decode,
                     uint64_t receiver_monotonic_ms) const;
  // Slow-side refresh. Request paths call routing_snapshot(), which performs
  // only an atomic shared_ptr load and an expiry comparison.
  void refresh_routing_snapshot(uint64_t receiver_monotonic_ms) const;
  EngineRegistryRoutingSnapshot routing_snapshot(
      uint64_t receiver_monotonic_ms) const;
  std::optional<ObservationSnapshot> observation_snapshot(
      uint64_t receiver_monotonic_ms) const;
  EngineKVCapacitySnapshot kv_capacity_snapshot(
      uint64_t receiver_monotonic_ms) const;
  // Stable, bounded membership/state view for the V3 slow loop. The Registry
  // lock is held only while copying protobuf values; callers never retain
  // pointers into routing state.
  ContractResult snapshot_members(
      uint64_t receiver_monotonic_ms,
      size_t max_members,
      std::vector<EngineRegistryMemberSnapshot>* snapshot) const;

  bool registry_known() const;
  bool has_accepted_full_snapshot() const;
  bool has_current_full_snapshot() const;
  size_t member_count() const;
  size_t state_count() const;
  size_t link_count() const;

 private:
  struct EngineKey {
    int provider_id = 0;
    std::string profile_digest;
    std::string incarnation_id;

    bool operator<(const EngineKey& other) const;
  };

  struct LinkKey {
    EngineKey prefill;
    EngineKey decode;

    bool operator<(const LinkKey& other) const;
  };

  struct CachedEngineState {
    xllm::proto::EngineState state;
    uint64_t received_monotonic_ms = 0;
    uint64_t lifecycle_since_monotonic_ms = 0;
  };

  struct CachedLinkState {
    xllm::proto::LinkState state;
    uint64_t received_monotonic_ms = 0;
  };

  struct DirectEvidence {
    std::optional<uint64_t> last_success_monotonic_ms;
    std::optional<uint64_t> last_failure_monotonic_ms;
  };

  static std::optional<EngineKey> to_engine_key(
      const xllm::proto::ProviderEngineKey& key);
  static bool same_engine_key(const EngineKey& left, const EngineKey& right);
  static uint64_t effective_age_ms(uint64_t age_at_publish_ms,
                                   uint64_t received_monotonic_ms,
                                   uint64_t now_monotonic_ms);

  ContractResult validate_link_state_locked(
      const xllm::proto::LinkState& state) const;
  ObservationInput observation_input_locked(
      uint64_t receiver_monotonic_ms) const;
  uint64_t normalize_observation_time_locked(
      uint64_t receiver_monotonic_ms) const;
  std::optional<ObservationSnapshot> update_observation_locked(
      uint64_t receiver_monotonic_ms) const;
  bool has_usable_state_snapshot_locked() const;
  bool is_schedulable_locked(const EngineKey& key,
                             const std::string& engine_uid,
                             uint64_t receiver_monotonic_ms,
                             const ObservationSnapshot& observation) const;
  bool is_link_ready_locked(const EngineKey& prefill_key,
                            const std::string& prefill_engine_uid,
                            const EngineKey& decode_key,
                            const std::string& decode_engine_uid,
                            uint64_t receiver_monotonic_ms,
                            const ObservationSnapshot& observation) const;
  bool has_unrefuted_cached_state_locked(const EngineKey& key,
                                         const std::string& engine_uid) const;
  bool has_recent_direct_success_locked(const EngineKey& key,
                                        uint64_t receiver_monotonic_ms) const;
  void invalidate_routing_snapshot_locked() const;
  void remove_from_routing_snapshot_locked(
      const xllm::proto::ProviderEngineKey& key) const;
  void publish_routing_snapshot_locked(uint64_t receiver_monotonic_ms) const;

  EngineRegistryConfig config_;
  bool config_valid_ = false;
  mutable std::shared_mutex mutex_;
  mutable ObservationController observation_controller_;
  mutable uint64_t last_observation_monotonic_ms_ = 0;
  mutable std::shared_ptr<const EngineRegistryRoutingSnapshot::Data>
      routing_snapshot_data_;
  std::map<EngineKey, xllm::proto::ProviderDescriptor> members_;
  std::map<std::string, EngineKey> current_by_engine_uid_;
  std::map<EngineKey, CachedEngineState> states_;
  std::map<LinkKey, CachedLinkState> links_;
  std::map<EngineKey, DirectEvidence> direct_evidence_;
  bool registry_known_ = false;
  bool has_accepted_full_snapshot_ = false;
  std::string master_incarnation_;
  std::string full_snapshot_master_incarnation_;
  uint64_t last_snapshot_seq_ = 0;
};

}  // namespace xllm_service::provider
