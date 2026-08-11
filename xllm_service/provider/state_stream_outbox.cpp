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

#include <algorithm>
#include <charconv>
#include <limits>
#include <set>
#include <string_view>
#include <utility>

#include "provider/identity_key.h"

namespace xllm_service::provider {
namespace {

ContractResult fail(xllm::proto::ProviderContractError error,
                    std::string message) {
  return ContractResult::failure(error, std::move(message));
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

std::string engine_key(const xllm::proto::ProviderEngineKey& key) {
  return provider_engine_identity_key(key);
}

std::string engine_key(const xllm::proto::EngineState& state) {
  xllm::proto::ProviderEngineKey key;
  key.set_provider_id(state.provider_id());
  key.set_profile_digest(state.profile_digest());
  key.set_engine_uid(state.engine_uid());
  key.set_incarnation_id(state.incarnation_id());
  return engine_key(key);
}

std::string link_key(const xllm::proto::LinkState& state) {
  return provider_link_identity_key(state.prefill(), state.decode());
}

bool link_references_engine(const xllm::proto::LinkState& link,
                            const std::string& removed_key) {
  return engine_key(link.prefill()) == removed_key ||
         engine_key(link.decode()) == removed_key;
}

uint64_t effective_age_ms(uint64_t age_at_receive_ms,
                          uint64_t received_monotonic_ms,
                          uint64_t publish_monotonic_ms) {
  if (publish_monotonic_ms < received_monotonic_ms) {
    return std::numeric_limits<uint64_t>::max();
  }
  const uint64_t elapsed_ms = publish_monotonic_ms - received_monotonic_ms;
  if (age_at_receive_ms > std::numeric_limits<uint64_t>::max() - elapsed_ms) {
    return std::numeric_limits<uint64_t>::max();
  }
  return age_at_receive_ms + elapsed_ms;
}

}  // namespace

ContractResult resolve_state_stream_subscribers(
    const std::unordered_map<std::string, std::string>& service_members,
    const std::string& local_service,
    size_t max_subscribers,
    std::vector<std::string>* subscribers) {
  if (subscribers == nullptr || max_subscribers == 0) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "State Stream subscriber output or capacity is invalid");
  }
  subscribers->clear();
  std::set<std::string> unique;
  for (const auto& [member_name, member_value] : service_members) {
    if (member_name == "MASTER") {
      continue;
    }
    if (member_name != member_value || member_value == local_service ||
        !valid_rpc_address(member_value)) {
      continue;
    }
    unique.insert(member_value);
  }
  if (unique.size() > max_subscribers) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "State Stream subscriber capacity is exhausted");
  }
  subscribers->assign(unique.begin(), unique.end());
  return ContractResult::success();
}

StateStreamOutbox::StateStreamOutbox(StateStreamOutboxConfig config,
                                     std::string master_incarnation)
    : config_(std::move(config)),
      master_incarnation_(std::move(master_incarnation)) {
  config_valid_ =
      config_.max_subscribers > 0 && config_.max_pending_engine_states > 0 &&
      config_.max_pending_link_states > 0 && !master_incarnation_.empty();
}

bool StateStreamOutbox::requires_full(const SubscriberState& state) const {
  return state.completed_full_generation < state.full_requirement_generation;
}

void StateStreamOutbox::force_full(SubscriberState* state) {
  ++state->full_requirement_generation;
  if (state->full_requirement_generation == 0) {
    state->full_requirement_generation = 1;
    state->completed_full_generation = 0;
  }
  state->pending = PendingChanges{};
}

ContractResult StateStreamOutbox::replace_subscribers(
    const std::vector<std::string>& subscribers) {
  if (!config_valid_) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "State Stream Outbox configuration is invalid");
  }
  std::set<std::string> desired;
  for (const std::string& subscriber : subscribers) {
    if (!valid_rpc_address(subscriber)) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD,
                  "State Stream subscriber address is invalid");
    }
    desired.insert(subscriber);
  }
  if (desired.size() > config_.max_subscribers) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "State Stream subscriber capacity is exhausted");
  }

  std::lock_guard lock(mutex_);
  for (auto it = subscribers_.begin(); it != subscribers_.end();) {
    if (desired.find(it->first) == desired.end()) {
      it = subscribers_.erase(it);
    } else {
      ++it;
    }
  }
  for (const std::string& subscriber : desired) {
    subscribers_.try_emplace(subscriber);
  }
  return ContractResult::success();
}

void StateStreamOutbox::merge_delta(const xllm::proto::StateBatch& delta,
                                    uint64_t received_monotonic_ms,
                                    SubscriberState* state) {
  for (const xllm::proto::EngineState& engine_state : delta.engine_states()) {
    const std::string key = engine_key(engine_state);
    if (state->pending.removals.find(key) != state->pending.removals.end()) {
      continue;
    }
    const auto existing = state->pending.engine_states.find(key);
    if (existing == state->pending.engine_states.end() ||
        engine_state.state_seq() > existing->second.state.state_seq()) {
      state->pending.engine_states.insert_or_assign(
          key,
          PendingChanges::EngineState{
              .state = engine_state,
              .received_monotonic_ms = received_monotonic_ms,
          });
    }
  }
  for (const xllm::proto::ProviderEngineKey& removed :
       delta.removed_engines()) {
    const std::string key = engine_key(removed);
    state->pending.engine_states.erase(key);
    for (auto link = state->pending.link_states.begin();
         link != state->pending.link_states.end();) {
      if (link_references_engine(link->second.state, key)) {
        link = state->pending.link_states.erase(link);
      } else {
        ++link;
      }
    }
    state->pending.removals.insert_or_assign(key, removed);
  }
  for (const xllm::proto::LinkState& link : delta.link_states()) {
    const std::string prefill_key = engine_key(link.prefill());
    const std::string decode_key = engine_key(link.decode());
    if (state->pending.removals.find(prefill_key) !=
            state->pending.removals.end() ||
        state->pending.removals.find(decode_key) !=
            state->pending.removals.end()) {
      continue;
    }
    const std::string key = link_key(link);
    const auto existing = state->pending.link_states.find(key);
    if (existing == state->pending.link_states.end() ||
        link.state_seq() > existing->second.state.state_seq()) {
      state->pending.link_states.insert_or_assign(
          key,
          PendingChanges::LinkState{
              .state = link,
              .received_monotonic_ms = received_monotonic_ms,
          });
    }
  }
  if (state->pending.engine_states.size() + state->pending.removals.size() >
      config_.max_pending_engine_states) {
    force_full(state);
  } else if (state->pending.link_states.size() >
             config_.max_pending_link_states) {
    force_full(state);
  }
}

ContractResult StateStreamOutbox::enqueue_delta(
    const xllm::proto::StateBatch& delta,
    uint64_t received_monotonic_ms) {
  if (!config_valid_) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "State Stream Outbox configuration is invalid");
  }
  if (delta.contract_version() != kProviderContractVersion ||
      delta.master_incarnation() != master_incarnation_ ||
      delta.kind() != xllm::proto::STATE_BATCH_KIND_DELTA) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
                "State Stream delta identity or kind is invalid");
  }
  std::lock_guard lock(mutex_);
  for (auto& [subscriber, state] : subscribers_) {
    static_cast<void>(subscriber);
    merge_delta(delta, received_monotonic_ms, &state);
  }
  return ContractResult::success();
}

void StateStreamOutbox::require_full_for_all() {
  std::lock_guard lock(mutex_);
  for (auto& [subscriber, state] : subscribers_) {
    static_cast<void>(subscriber);
    force_full(&state);
  }
}

std::vector<std::string> StateStreamOutbox::ready_subscribers() const {
  std::vector<std::string> ready;
  std::lock_guard lock(mutex_);
  for (const auto& [subscriber, state] : subscribers_) {
    const bool has_pending = !state.pending.engine_states.empty() ||
                             !state.pending.link_states.empty() ||
                             !state.pending.removals.empty();
    if (!state.in_flight && (requires_full(state) || has_pending)) {
      ready.emplace_back(subscriber);
    }
  }
  return ready;
}

bool StateStreamOutbox::begin_delivery(
    const std::string& subscriber,
    const xllm::proto::StateBatch& authoritative_full,
    uint64_t snapshot_seq,
    uint64_t publish_monotonic_ms,
    xllm::proto::StateBatch* delivery) {
  if (delivery == nullptr || snapshot_seq == 0) {
    return false;
  }
  std::lock_guard lock(mutex_);
  auto found = subscribers_.find(subscriber);
  if (found == subscribers_.end() || found->second.in_flight ||
      snapshot_seq <= last_issued_snapshot_seq_) {
    return false;
  }
  SubscriberState& state = found->second;
  const bool needs_full = requires_full(state);
  const bool has_pending = !state.pending.engine_states.empty() ||
                           !state.pending.link_states.empty() ||
                           !state.pending.removals.empty();
  if (!needs_full && !has_pending) {
    return false;
  }

  delivery->Clear();
  if (needs_full) {
    if (authoritative_full.contract_version() != kProviderContractVersion ||
        authoritative_full.master_incarnation() != master_incarnation_ ||
        authoritative_full.kind() != xllm::proto::STATE_BATCH_KIND_FULL ||
        !authoritative_full.removed_engines().empty()) {
      return false;
    }
    *delivery = authoritative_full;
    state.in_flight_was_full = true;
    state.in_flight_full_generation = state.full_requirement_generation;
  } else {
    delivery->set_contract_version(kProviderContractVersion);
    delivery->set_master_incarnation(master_incarnation_);
    delivery->set_kind(xllm::proto::STATE_BATCH_KIND_DELTA);
    for (const auto& [key, pending] : state.pending.engine_states) {
      static_cast<void>(key);
      xllm::proto::EngineState* engine_state = delivery->add_engine_states();
      *engine_state = pending.state;
      if (engine_state->has_heartbeat_age_ms_at_publish()) {
        engine_state->set_heartbeat_age_ms_at_publish(
            effective_age_ms(engine_state->heartbeat_age_ms_at_publish(),
                             pending.received_monotonic_ms,
                             publish_monotonic_ms));
      }
      if (engine_state->has_state_age_ms_at_publish()) {
        engine_state->set_state_age_ms_at_publish(
            effective_age_ms(engine_state->state_age_ms_at_publish(),
                             pending.received_monotonic_ms,
                             publish_monotonic_ms));
      }
    }
    for (const auto& [key, pending] : state.pending.link_states) {
      static_cast<void>(key);
      xllm::proto::LinkState* link_state = delivery->add_link_states();
      *link_state = pending.state;
      if (link_state->has_age_ms_at_publish()) {
        link_state->set_age_ms_at_publish(
            effective_age_ms(link_state->age_ms_at_publish(),
                             pending.received_monotonic_ms,
                             publish_monotonic_ms));
      }
    }
    for (const auto& [key, removed] : state.pending.removals) {
      static_cast<void>(key);
      *delivery->add_removed_engines() = removed;
    }
    state.pending = PendingChanges{};
    state.in_flight_was_full = false;
    state.in_flight_full_generation = 0;
  }
  delivery->set_snapshot_seq(snapshot_seq);
  state.in_flight = true;
  last_issued_snapshot_seq_ = snapshot_seq;
  return true;
}

bool StateStreamOutbox::complete_delivery(const std::string& subscriber,
                                          bool success) {
  std::lock_guard lock(mutex_);
  auto found = subscribers_.find(subscriber);
  if (found == subscribers_.end() || !found->second.in_flight) {
    return false;
  }
  SubscriberState& state = found->second;
  if (success && state.in_flight_was_full) {
    state.completed_full_generation = std::max(state.completed_full_generation,
                                               state.in_flight_full_generation);
  } else if (!success) {
    force_full(&state);
  }
  state.in_flight = false;
  state.in_flight_was_full = false;
  state.in_flight_full_generation = 0;
  return true;
}

size_t StateStreamOutbox::subscriber_count() const {
  std::lock_guard lock(mutex_);
  return subscribers_.size();
}

size_t StateStreamOutbox::pending_entry_count(
    const std::string& subscriber) const {
  std::lock_guard lock(mutex_);
  const auto found = subscribers_.find(subscriber);
  if (found == subscribers_.end()) {
    return 0;
  }
  return found->second.pending.engine_states.size() +
         found->second.pending.link_states.size() +
         found->second.pending.removals.size();
}

}  // namespace xllm_service::provider
