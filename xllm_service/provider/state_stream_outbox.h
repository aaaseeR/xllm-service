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
#include <string>
#include <unordered_map>
#include <vector>

#include "provider.pb.h"
#include "provider/provider_contract.h"

namespace xllm_service::provider {

struct StateStreamOutboxConfig {
  size_t max_subscribers = 256;
  size_t max_pending_engine_states = 4096;
  size_t max_pending_link_states = 16384;
};

// Converts the existing etcd service member map into validated, deduplicated
// State Stream targets. The MASTER entry is excluded before address parsing.
ContractResult resolve_state_stream_subscribers(
    const std::unordered_map<std::string, std::string>& service_members,
    const std::string& local_service,
    size_t max_subscribers,
    std::vector<std::string>* subscribers);

// Per-subscriber bounded latest-map. At most one batch can be in flight for a
// subscriber. A failed or overflowed DELTA is never replayed; the subscriber
// must receive a fresh authoritative FULL before DELTAs resume.
class StateStreamOutbox final {
 public:
  StateStreamOutbox(StateStreamOutboxConfig config,
                    std::string master_incarnation);

  ContractResult replace_subscribers(
      const std::vector<std::string>& subscribers);
  ContractResult enqueue_delta(const xllm::proto::StateBatch& delta,
                               uint64_t received_monotonic_ms);
  void require_full_for_all();

  std::vector<std::string> ready_subscribers() const;
  bool begin_delivery(const std::string& subscriber,
                      const xllm::proto::StateBatch& authoritative_full,
                      uint64_t snapshot_seq,
                      uint64_t publish_monotonic_ms,
                      xllm::proto::StateBatch* delivery);
  bool complete_delivery(const std::string& subscriber, bool success);

  size_t subscriber_count() const;
  size_t pending_entry_count(const std::string& subscriber) const;

 private:
  struct PendingChanges {
    struct EngineState {
      xllm::proto::EngineState state;
      uint64_t received_monotonic_ms = 0;
    };
    struct LinkState {
      xllm::proto::LinkState state;
      uint64_t received_monotonic_ms = 0;
    };
    std::map<std::string, EngineState> engine_states;
    std::map<std::string, LinkState> link_states;
    std::map<std::string, xllm::proto::ProviderEngineKey> removals;
  };

  struct SubscriberState {
    bool in_flight = false;
    bool in_flight_was_full = false;
    uint64_t full_requirement_generation = 1;
    uint64_t completed_full_generation = 0;
    uint64_t in_flight_full_generation = 0;
    PendingChanges pending;
  };

  bool requires_full(const SubscriberState& state) const;
  void force_full(SubscriberState* state);
  void merge_delta(const xllm::proto::StateBatch& delta,
                   uint64_t received_monotonic_ms,
                   SubscriberState* state);

  StateStreamOutboxConfig config_;
  bool config_valid_ = false;
  std::string master_incarnation_;
  mutable std::mutex mutex_;
  std::map<std::string, SubscriberState> subscribers_;
  uint64_t last_issued_snapshot_seq_ = 0;
};

}  // namespace xllm_service::provider
