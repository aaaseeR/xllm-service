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

#include <algorithm>
#include <charconv>
#include <limits>
#include <set>
#include <string_view>
#include <utility>

namespace xllm_service::provider {
namespace {

inline constexpr size_t kStateEnvelopeBytes = 128;

size_t batch_bytes(const xllm::proto::KVEventBatch& batch) {
  return batch.ByteSizeLong();
}

uint64_t add_age(uint64_t initial_age,
                 uint64_t received_monotonic_ms,
                 uint64_t publish_monotonic_ms) {
  if (publish_monotonic_ms < received_monotonic_ms) {
    return std::numeric_limits<uint64_t>::max();
  }
  const uint64_t elapsed = publish_monotonic_ms - received_monotonic_ms;
  if (initial_age > std::numeric_limits<uint64_t>::max() - elapsed) {
    return std::numeric_limits<uint64_t>::max();
  }
  return initial_age + elapsed;
}

bool valid_batch(const xllm::proto::KVEventBatch& batch) {
  return batch.contract_version() == kProviderContractVersion &&
         batch.has_identity() &&
         !batch.identity().engine().engine_uid().empty() &&
         !batch.identity().engine().incarnation_id().empty() &&
         batch.has_batch_age_ms_at_publish();
}

bool valid_rpc_address(const std::string& address) {
  if (address.empty() ||
      address.find_first_of(" \t\r\n") != std::string::npos) {
    return false;
  }
  const size_t separator = address.rfind(':');
  if (separator == std::string::npos || separator == 0 ||
      separator + 1 >= address.size()) {
    return false;
  }
  uint32_t port = 0;
  const std::string_view port_text(address.data() + separator + 1,
                                   address.size() - separator - 1);
  const auto parsed = std::from_chars(
      port_text.data(), port_text.data() + port_text.size(), port);
  return parsed.ec == std::errc() &&
         parsed.ptr == port_text.data() + port_text.size() && port > 0 &&
         port <= 65535;
}

}  // namespace

KVStateOutbox::KVStateOutbox(KVStateOutboxConfig config,
                             std::string master_incarnation)
    : config_(std::move(config)),
      master_incarnation_(std::move(master_incarnation)) {
  config_valid_ = config_.max_subscribers > 0 &&
                  config_.max_pending_batches_per_subscriber > 0 &&
                  config_.max_pending_events_per_subscriber > 0 &&
                  config_.max_pending_bytes_per_subscriber > 0 &&
                  config_.max_delivery_batches > 0 &&
                  config_.max_delivery_batches <=
                      static_cast<size_t>(std::numeric_limits<int>::max()) &&
                  config_.max_delivery_bytes > kStateEnvelopeBytes &&
                  !master_incarnation_.empty();
}

ContractResult KVStateOutbox::replace_subscribers(
    const std::vector<std::string>& subscribers) {
  if (!config_valid_) {
    return ContractResult::failure(
        xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
        "KV State Outbox configuration is invalid");
  }
  std::set<std::string> desired;
  for (const std::string& subscriber : subscribers) {
    if (!valid_rpc_address(subscriber)) {
      return ContractResult::failure(
          xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD,
          "KV State subscriber address is invalid");
    }
    desired.insert(subscriber);
  }
  if (desired.size() > config_.max_subscribers) {
    return ContractResult::failure(
        xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
        "KV State subscriber capacity is exhausted");
  }

  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = subscribers_.begin(); it != subscribers_.end();) {
    if (desired.find(it->first) == desired.end()) {
      it = subscribers_.erase(it);
    } else {
      ++it;
    }
  }
  for (const std::string& subscriber : desired) {
    if (subscribers_.find(subscriber) != subscribers_.end()) {
      continue;
    }
    if (next_stream_epoch_ == std::numeric_limits<uint64_t>::max()) {
      return ContractResult::failure(
          xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
          "KV State stream epoch space is exhausted");
    }
    SubscriberState state;
    state.stream_epoch = next_stream_epoch_++;
    subscribers_.emplace(subscriber, std::move(state));
  }
  return ContractResult::success();
}

void KVStateOutbox::force_gap_locked(SubscriberState* state) {
  state->pending.clear();
  state->pending_events = 0;
  state->pending_bytes = 0;
  state->force_stream_gap = true;
}

bool KVStateOutbox::append_locked(const xllm::proto::KVEventBatch& batch,
                                  uint64_t received_monotonic_ms,
                                  SubscriberState* state) {
  const size_t bytes = batch_bytes(batch);
  const size_t events = static_cast<size_t>(batch.events_size());
  if (bytes > config_.max_delivery_bytes - kStateEnvelopeBytes ||
      bytes > config_.max_pending_bytes_per_subscriber ||
      events > config_.max_pending_events_per_subscriber ||
      state->pending.size() >= config_.max_pending_batches_per_subscriber ||
      state->pending_events >
          config_.max_pending_events_per_subscriber - events ||
      state->pending_bytes > config_.max_pending_bytes_per_subscriber - bytes) {
    return false;
  }
  state->pending.emplace_back(PendingBatch{
      .batch = batch,
      .received_monotonic_ms = received_monotonic_ms,
      .bytes = bytes,
  });
  state->pending_events += events;
  state->pending_bytes += bytes;
  return true;
}

ContractResult KVStateOutbox::enqueue(const xllm::proto::KVEventBatch& batch,
                                      uint64_t received_monotonic_ms) {
  if (!config_valid_ || received_monotonic_ms == 0 || !valid_batch(batch)) {
    return ContractResult::failure(
        xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
        "KV State event batch is invalid");
  }

  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& [subscriber, state] : subscribers_) {
    static_cast<void>(subscriber);
    if (append_locked(batch, received_monotonic_ms, &state)) {
      continue;
    }
    force_gap_locked(&state);
    if (append_locked(batch, received_monotonic_ms, &state)) {
      continue;
    }
    xllm::proto::KVEventBatch marker;
    marker.set_contract_version(kProviderContractVersion);
    *marker.mutable_identity() = batch.identity();
    marker.set_gap_before_events(true);
    marker.set_last_event_seq(batch.last_event_seq());
    marker.set_batch_age_ms_at_publish(batch.batch_age_ms_at_publish());
    if (!append_locked(marker, received_monotonic_ms, &state)) {
      return ContractResult::failure(
          xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
          "KV State gap marker exceeds configured capacity");
    }
  }
  return ContractResult::success();
}

std::vector<std::string> KVStateOutbox::ready_subscribers() const {
  std::vector<std::string> ready;
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& [subscriber, state] : subscribers_) {
    if (state.in_flight.has_value() || !state.pending.empty()) {
      ready.emplace_back(subscriber);
    }
  }
  return ready;
}

bool KVStateOutbox::begin_delivery(const std::string& subscriber,
                                   uint64_t publish_monotonic_ms,
                                   xllm::proto::KVStateBatch* delivery) {
  if (publish_monotonic_ms == 0 || delivery == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  auto found = subscribers_.find(subscriber);
  if (found == subscribers_.end()) {
    return false;
  }
  SubscriberState& state = found->second;
  if (state.in_flight.has_value()) {
    *delivery = *state.in_flight;
    return true;
  }
  if (state.pending.empty() ||
      state.last_issued_stream_seq == std::numeric_limits<uint64_t>::max()) {
    return false;
  }

  uint64_t stream_seq = state.last_issued_stream_seq + 1;
  if (state.force_stream_gap) {
    if (stream_seq == std::numeric_limits<uint64_t>::max()) {
      return false;
    }
    ++stream_seq;
  }
  xllm::proto::KVStateBatch candidate;
  candidate.set_contract_version(kProviderContractVersion);
  candidate.set_master_incarnation(master_incarnation_);
  candidate.set_stream_seq(stream_seq);
  candidate.set_stream_epoch(state.stream_epoch);
  while (!state.pending.empty() &&
         candidate.engine_batches_size() <
             static_cast<int>(config_.max_delivery_batches)) {
    const PendingBatch& pending = state.pending.front();
    xllm::proto::KVEventBatch adjusted = pending.batch;
    adjusted.set_batch_age_ms_at_publish(
        add_age(adjusted.batch_age_ms_at_publish(),
                pending.received_monotonic_ms,
                publish_monotonic_ms));
    for (xllm::proto::KVEvent& event : *adjusted.mutable_events()) {
      if (event.has_event_age_ms_at_publish()) {
        event.set_event_age_ms_at_publish(
            add_age(event.event_age_ms_at_publish(),
                    pending.received_monotonic_ms,
                    publish_monotonic_ms));
      }
    }
    *candidate.add_engine_batches() = std::move(adjusted);
    if (candidate.ByteSizeLong() > config_.max_delivery_bytes) {
      candidate.mutable_engine_batches()->RemoveLast();
      if (candidate.engine_batches().empty()) {
        force_gap_locked(&state);
      }
      break;
    }
    state.pending_events -= static_cast<size_t>(pending.batch.events_size());
    state.pending_bytes -= pending.bytes;
    state.pending.pop_front();
  }
  if (candidate.engine_batches().empty()) {
    return false;
  }
  state.force_stream_gap = false;
  state.last_issued_stream_seq = stream_seq;
  state.in_flight = candidate;
  *delivery = std::move(candidate);
  return true;
}

bool KVStateOutbox::complete_delivery(const std::string& subscriber,
                                      bool success) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto found = subscribers_.find(subscriber);
  if (found == subscribers_.end() || !found->second.in_flight.has_value()) {
    return false;
  }
  if (success) {
    found->second.in_flight.reset();
  }
  return true;
}

KVStateOutboxStats KVStateOutbox::stats() const {
  KVStateOutboxStats current;
  std::lock_guard<std::mutex> lock(mutex_);
  current.subscribers = subscribers_.size();
  for (const auto& [subscriber, state] : subscribers_) {
    static_cast<void>(subscriber);
    current.pending_batches += state.pending.size();
    current.pending_events += state.pending_events;
    current.pending_bytes += state.pending_bytes;
    if (state.in_flight.has_value()) {
      ++current.in_flight_subscribers;
    }
    if (state.force_stream_gap) {
      ++current.forced_stream_gaps;
    }
  }
  return current;
}

}  // namespace xllm_service::provider
