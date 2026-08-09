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

#include "provider/link_reconciler.h"

#include <algorithm>
#include <limits>
#include <set>
#include <utility>

namespace xllm_service::provider {
namespace {

ContractResult fail(xllm::proto::ProviderContractError error,
                    std::string message) {
  return ContractResult::failure(error, std::move(message));
}

uint64_t saturating_add(uint64_t left, uint64_t right) {
  if (left > std::numeric_limits<uint64_t>::max() - right) {
    return std::numeric_limits<uint64_t>::max();
  }
  return left + right;
}

xllm::proto::ProviderEngineKey provider_engine_key(
    const xllm::proto::ProviderDescriptor& descriptor) {
  xllm::proto::ProviderEngineKey key;
  key.set_provider_id(descriptor.identity().provider_id());
  key.set_profile_digest(descriptor.profile_digest());
  key.set_engine_uid(descriptor.identity().engine_uid());
  key.set_incarnation_id(descriptor.identity().incarnation_id());
  return key;
}

}  // namespace

LinkReconciler::LinkReconciler(LinkReconcilerConfig config)
    : config_(std::move(config)) {
  config_valid_ = config_.max_links > 0 && config_.retry_initial_ms > 0 &&
                  config_.retry_max_ms >= config_.retry_initial_ms &&
                  config_.ready_recheck_ms > 0;
}

std::string LinkReconciler::link_key(
    const xllm::proto::ProviderEngineKey& prefill,
    const xllm::proto::ProviderEngineKey& decode) {
  const std::string prefill_wire = prefill.SerializeAsString();
  const std::string decode_wire = decode.SerializeAsString();
  std::string key;
  key.reserve(sizeof(uint64_t) + prefill_wire.size() + decode_wire.size());
  const uint64_t prefill_size = static_cast<uint64_t>(prefill_wire.size());
  key.append(reinterpret_cast<const char*>(&prefill_size),
             sizeof(prefill_size));
  key.append(prefill_wire);
  key.append(decode_wire);
  return key;
}

uint64_t LinkReconciler::retry_delay_ms(uint32_t consecutive_failures) const {
  uint64_t delay = config_.retry_initial_ms;
  const uint32_t shifts = std::min<uint32_t>(consecutive_failures - 1, 63);
  for (uint32_t shift = 0; shift < shifts; ++shift) {
    if (delay >= config_.retry_max_ms ||
        delay > std::numeric_limits<uint64_t>::max() / 2) {
      return config_.retry_max_ms;
    }
    delay *= 2;
  }
  return std::min(delay, config_.retry_max_ms);
}

ContractResult LinkReconciler::replace_desired(
    const std::vector<DesiredProviderLink>& desired,
    uint64_t now_monotonic_ms,
    std::vector<xllm::proto::LinkState>* state_changes) {
  if (state_changes == nullptr) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD,
                "Link reconciler state change output must not be null");
  }
  state_changes->clear();
  if (!config_valid_) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "Link reconciler configuration is invalid");
  }
  if (desired.size() > config_.max_links) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "Desired Provider link capacity is exhausted");
  }

  std::map<std::string, Entry> incoming;
  for (const DesiredProviderLink& link : desired) {
    std::string compatibility_proof;
    ContractResult compatible = validate_remote_pd_compatibility(
        link.prefill, link.decode, &compatibility_proof);
    if (!compatible.ok()) {
      return compatible;
    }
    const xllm::proto::ProviderEngineKey prefill_key =
        provider_engine_key(link.prefill);
    const xllm::proto::ProviderEngineKey decode_key =
        provider_engine_key(link.decode);
    const std::string key = link_key(prefill_key, decode_key);
    Entry entry;
    *entry.state.mutable_prefill() = prefill_key;
    *entry.state.mutable_decode() = decode_key;
    entry.state.set_lifecycle(xllm::proto::LINK_LIFECYCLE_PENDING);
    entry.state.set_connector(link.prefill.kv().connector());
    entry.state.set_connector_version(link.prefill.kv().connector_version());
    entry.state.set_transfer_mode(xllm::proto::TRANSFER_MODE_LAYERWISE_PUSH);
    entry.state.set_compatibility_proof(std::move(compatibility_proof));
    entry.state.set_last_handshake_result("pending");
    entry.state.set_state_seq(1);
    entry.state.set_age_ms_at_publish(0);
    entry.next_attempt_ms = now_monotonic_ms;
    if (!incoming.emplace(key, std::move(entry)).second) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DUPLICATE_ENTRY,
                  "Desired Provider links contain a duplicate pair");
    }
  }

  std::lock_guard lock(mutex_);
  for (auto& [key, entry] : incoming) {
    auto existing = entries_.find(key);
    if (existing == entries_.end()) {
      state_changes->emplace_back(entry.state);
      continue;
    }
    if (existing->second.state.compatibility_proof() !=
            entry.state.compatibility_proof() ||
        existing->second.state.connector() != entry.state.connector() ||
        existing->second.state.connector_version() !=
            entry.state.connector_version()) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
                  "Provider link identity collides with a new contract");
    }
    entry = existing->second;
  }
  entries_ = std::move(incoming);
  return ContractResult::success();
}

std::vector<ProviderLinkAttempt> LinkReconciler::begin_due_attempts(
    uint64_t now_monotonic_ms,
    size_t max_attempts) {
  std::vector<ProviderLinkAttempt> attempts;
  if (!config_valid_ || max_attempts == 0) {
    return attempts;
  }
  std::lock_guard lock(mutex_);
  attempts.reserve(std::min(max_attempts, entries_.size()));
  for (auto& [key, entry] : entries_) {
    static_cast<void>(key);
    if (attempts.size() == max_attempts) {
      break;
    }
    if (entry.in_flight || entry.next_attempt_ms > now_monotonic_ms) {
      continue;
    }
    entry.in_flight = true;
    attempts.emplace_back(ProviderLinkAttempt{
        .prefill = entry.state.prefill(),
        .decode = entry.state.decode(),
    });
  }
  return attempts;
}

ContractResult LinkReconciler::complete_attempt(
    const ProviderLinkAttempt& attempt,
    bool success,
    std::string handshake_result,
    uint64_t now_monotonic_ms,
    xllm::proto::LinkState* state_change) {
  if (state_change == nullptr) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD,
                "Link attempt state change output must not be null");
  }
  state_change->Clear();
  std::lock_guard lock(mutex_);
  const auto found = entries_.find(link_key(attempt.prefill, attempt.decode));
  if (found == entries_.end() || !found->second.in_flight) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "Link attempt is stale or was not started");
  }
  Entry& entry = found->second;
  if (entry.state.state_seq() == std::numeric_limits<uint64_t>::max()) {
    entry.in_flight = false;
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "Link state sequence is exhausted");
  }
  entry.in_flight = false;
  entry.state.set_state_seq(entry.state.state_seq() + 1);
  entry.state.set_age_ms_at_publish(0);
  if (handshake_result.empty()) {
    handshake_result = success ? "ready" : "link-rpc-failed";
  }
  constexpr size_t kMaxHandshakeResultBytes = 256;
  if (handshake_result.size() > kMaxHandshakeResultBytes) {
    handshake_result.resize(kMaxHandshakeResultBytes);
  }
  entry.state.set_last_handshake_result(std::move(handshake_result));
  if (success) {
    entry.state.set_lifecycle(xllm::proto::LINK_LIFECYCLE_READY);
    entry.consecutive_failures = 0;
    entry.next_attempt_ms =
        saturating_add(now_monotonic_ms, config_.ready_recheck_ms);
  } else {
    entry.state.set_lifecycle(xllm::proto::LINK_LIFECYCLE_DEGRADED);
    if (entry.consecutive_failures < std::numeric_limits<uint32_t>::max()) {
      ++entry.consecutive_failures;
    }
    entry.next_attempt_ms = saturating_add(
        now_monotonic_ms, retry_delay_ms(entry.consecutive_failures));
  }
  *state_change = entry.state;
  return ContractResult::success();
}

void LinkReconciler::require_recheck() {
  std::lock_guard lock(mutex_);
  for (auto& [key, entry] : entries_) {
    static_cast<void>(key);
    entry.next_attempt_ms = 0;
  }
}

size_t LinkReconciler::size() const {
  std::lock_guard lock(mutex_);
  return entries_.size();
}

}  // namespace xllm_service::provider
