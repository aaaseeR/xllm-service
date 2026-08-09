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
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "provider.pb.h"
#include "provider/provider_contract.h"

namespace xllm_service::provider {

struct KVStateOutboxConfig {
  size_t max_subscribers = 256;
  size_t max_pending_batches_per_subscriber = 4096;
  size_t max_pending_events_per_subscriber = 16384;
  size_t max_pending_bytes_per_subscriber = 16 * 1024 * 1024;
  size_t max_delivery_batches = 64;
  size_t max_delivery_bytes = 1024 * 1024;
};

struct KVStateOutboxStats {
  size_t subscribers = 0;
  size_t pending_batches = 0;
  size_t pending_events = 0;
  size_t pending_bytes = 0;
  size_t in_flight_subscribers = 0;
  size_t forced_stream_gaps = 0;
};

// Independent, bounded master-to-replica KV lane. Queue overflow skips one
// replica stream sequence, which makes that replica discard every KV credit
// before it processes later observations. Health/load StateBatch delivery is
// never coupled to this queue.
class KVStateOutbox final {
 public:
  KVStateOutbox(KVStateOutboxConfig config, std::string master_incarnation);

  ContractResult replace_subscribers(
      const std::vector<std::string>& subscribers);
  ContractResult enqueue(const xllm::proto::KVEventBatch& batch,
                         uint64_t received_monotonic_ms);

  std::vector<std::string> ready_subscribers() const;
  bool begin_delivery(const std::string& subscriber,
                      uint64_t publish_monotonic_ms,
                      xllm::proto::KVStateBatch* delivery);
  bool complete_delivery(const std::string& subscriber, bool success);

  KVStateOutboxStats stats() const;

 private:
  struct PendingBatch {
    xllm::proto::KVEventBatch batch;
    uint64_t received_monotonic_ms = 0;
    size_t bytes = 0;
  };

  struct SubscriberState {
    uint64_t stream_epoch = 0;
    uint64_t last_issued_stream_seq = 0;
    bool force_stream_gap = false;
    size_t pending_events = 0;
    size_t pending_bytes = 0;
    std::deque<PendingBatch> pending;
    std::optional<xllm::proto::KVStateBatch> in_flight;
  };

  bool append_locked(const xllm::proto::KVEventBatch& batch,
                     uint64_t received_monotonic_ms,
                     SubscriberState* state);
  void force_gap_locked(SubscriberState* state);

  KVStateOutboxConfig config_;
  bool config_valid_ = false;
  std::string master_incarnation_;
  uint64_t next_stream_epoch_ = 1;
  mutable std::mutex mutex_;
  std::map<std::string, SubscriberState> subscribers_;
};

}  // namespace xllm_service::provider
