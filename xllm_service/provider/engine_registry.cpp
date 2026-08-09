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

#include "provider/engine_registry.h"

#include <google/protobuf/util/message_differencer.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <set>
#include <tuple>
#include <utility>

namespace xllm_service::provider {
namespace {

ContractResult fail(xllm::proto::ProviderContractError error,
                    std::string message) {
  return ContractResult::failure(error, std::move(message));
}

bool has_capability(const xllm::proto::ProviderDescriptor& descriptor,
                    xllm::proto::ProviderCapability capability) {
  return std::find(descriptor.capabilities().begin(),
                   descriptor.capabilities().end(),
                   capability) != descriptor.capabilities().end();
}

}  // namespace

xllm::proto::ProviderEngineKey make_provider_engine_key(
    const xllm::proto::ProviderDescriptor& descriptor) {
  xllm::proto::ProviderEngineKey key;
  key.set_provider_id(descriptor.identity().provider_id());
  key.set_profile_digest(descriptor.profile_digest());
  key.set_engine_uid(descriptor.identity().engine_uid());
  key.set_incarnation_id(descriptor.identity().incarnation_id());
  return key;
}

EngineRegistry::EngineRegistry(EngineRegistryConfig config)
    : config_(std::move(config)) {
  config_valid_ =
      config_.max_members > 0 && config_.max_links > 0 &&
      config_.state_soft_ttl_ms > 0 && config_.state_hard_ttl_ms > 0 &&
      config_.state_soft_ttl_ms <= config_.state_hard_ttl_ms &&
      config_.heartbeat_hard_ttl_ms > 0 && config_.link_hard_ttl_ms > 0;
}

bool EngineRegistry::EngineKey::operator<(const EngineKey& other) const {
  return std::tie(provider_id, profile_digest, incarnation_id) <
         std::tie(
             other.provider_id, other.profile_digest, other.incarnation_id);
}

bool EngineRegistry::LinkKey::operator<(const LinkKey& other) const {
  return std::tie(prefill, decode) < std::tie(other.prefill, other.decode);
}

std::optional<EngineRegistry::EngineKey> EngineRegistry::to_engine_key(
    const xllm::proto::ProviderEngineKey& key) {
  if (!xllm::proto::ProviderId_IsValid(key.provider_id()) ||
      key.provider_id() == xllm::proto::PROVIDER_ID_UNSPECIFIED ||
      key.profile_digest().empty() || key.engine_uid().empty() ||
      key.incarnation_id().empty()) {
    return std::nullopt;
  }
  return EngineKey{
      .provider_id = static_cast<int>(key.provider_id()),
      .profile_digest = key.profile_digest(),
      .incarnation_id = key.incarnation_id(),
  };
}

bool EngineRegistry::same_engine_key(const EngineKey& left,
                                     const EngineKey& right) {
  return !(left < right) && !(right < left);
}

uint64_t EngineRegistry::effective_age_ms(uint64_t age_at_publish_ms,
                                          uint64_t received_monotonic_ms,
                                          uint64_t now_monotonic_ms) {
  if (now_monotonic_ms < received_monotonic_ms) {
    return std::numeric_limits<uint64_t>::max();
  }
  const uint64_t elapsed_ms = now_monotonic_ms - received_monotonic_ms;
  if (age_at_publish_ms > std::numeric_limits<uint64_t>::max() - elapsed_ms) {
    return std::numeric_limits<uint64_t>::max();
  }
  return age_at_publish_ms + elapsed_ms;
}

ContractResult EngineRegistry::upsert_member(
    const xllm::proto::ProviderDescriptor& descriptor) {
  if (!config_valid_) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "Engine Registry configuration is invalid");
  }
  ContractResult validation = validate_provider_descriptor(descriptor);
  if (!validation.ok()) {
    return validation;
  }
  const xllm::proto::ProviderEngineKey wire_key =
      make_provider_engine_key(descriptor);
  const std::optional<EngineKey> key = to_engine_key(wire_key);
  if (!key.has_value()) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD,
                "Engine Registry key is incomplete");
  }

  std::unique_lock lock(mutex_);
  const auto same_key = members_.find(key.value());
  if (same_key != members_.end()) {
    if (!google::protobuf::util::MessageDifferencer::Equivalent(
            same_key->second, descriptor)) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
                  "Engine Registry key collides with another Descriptor");
    }
    return ContractResult::success();
  }

  const std::string& engine_uid = descriptor.identity().engine_uid();
  const auto current = current_by_engine_uid_.find(engine_uid);
  const bool replaces_current = current != current_by_engine_uid_.end();
  if (!replaces_current && members_.size() >= config_.max_members) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "Engine Registry member capacity is exhausted");
  }
  if (replaces_current) {
    const EngineKey old_key = current->second;
    members_.erase(old_key);
    states_.erase(old_key);
    for (auto link = links_.begin(); link != links_.end();) {
      if (same_engine_key(link->first.prefill, old_key) ||
          same_engine_key(link->first.decode, old_key)) {
        link = links_.erase(link);
      } else {
        ++link;
      }
    }
  }

  members_.emplace(key.value(), descriptor);
  current_by_engine_uid_.insert_or_assign(engine_uid, key.value());
  full_snapshot_master_incarnation_.clear();
  return ContractResult::success();
}

bool EngineRegistry::remove_member(
    const xllm::proto::ProviderEngineKey& wire_key) {
  const std::optional<EngineKey> key = to_engine_key(wire_key);
  if (!key.has_value()) {
    return false;
  }

  std::unique_lock lock(mutex_);
  const auto member = members_.find(key.value());
  if (member == members_.end() ||
      member->second.identity().engine_uid() != wire_key.engine_uid()) {
    return false;
  }
  const std::string engine_uid = member->second.identity().engine_uid();
  members_.erase(member);
  states_.erase(key.value());
  const auto current = current_by_engine_uid_.find(engine_uid);
  if (current != current_by_engine_uid_.end() &&
      same_engine_key(current->second, key.value())) {
    current_by_engine_uid_.erase(current);
  }
  for (auto link = links_.begin(); link != links_.end();) {
    if (same_engine_key(link->first.prefill, key.value()) ||
        same_engine_key(link->first.decode, key.value())) {
      link = links_.erase(link);
    } else {
      ++link;
    }
  }
  full_snapshot_master_incarnation_.clear();
  return true;
}

ContractResult EngineRegistry::set_registry_view(
    bool registry_known,
    std::string master_incarnation) {
  if (registry_known && master_incarnation.empty()) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD,
                "Known Registry view must identify the current master");
  }
  if (!config_valid_) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "Engine Registry configuration is invalid");
  }
  std::unique_lock lock(mutex_);
  registry_known_ = registry_known;
  if (!registry_known) {
    return ContractResult::success();
  }
  if (master_incarnation_ != master_incarnation) {
    master_incarnation_ = std::move(master_incarnation);
    full_snapshot_master_incarnation_.clear();
    last_snapshot_seq_ = 0;
  }
  return ContractResult::success();
}

ContractResult EngineRegistry::validate_link_state_locked(
    const xllm::proto::LinkState& state) const {
  const std::optional<EngineKey> prefill_key = to_engine_key(state.prefill());
  const std::optional<EngineKey> decode_key = to_engine_key(state.decode());
  if (!prefill_key.has_value() || !decode_key.has_value()) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD,
                "LinkState endpoint key is incomplete");
  }
  const auto prefill = members_.find(prefill_key.value());
  const auto decode = members_.find(decode_key.value());
  if (prefill == members_.end() || decode == members_.end() ||
      prefill->second.identity().engine_uid() != state.prefill().engine_uid() ||
      decode->second.identity().engine_uid() != state.decode().engine_uid()) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
                "LinkState endpoint is not a current Registry member");
  }
  if (!xllm::proto::LinkLifecycle_IsValid(state.lifecycle()) ||
      state.lifecycle() == xllm::proto::LINK_LIFECYCLE_UNSPECIFIED ||
      !xllm::proto::TransferMode_IsValid(state.transfer_mode()) ||
      state.transfer_mode() == xllm::proto::TRANSFER_MODE_UNSPECIFIED) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_UNKNOWN_ENUM_VALUE,
                "LinkState contains an unknown lifecycle or transfer mode");
  }
  if (state.state_seq() == 0 || !state.has_age_ms_at_publish() ||
      state.connector().empty() || state.connector_version().empty() ||
      state.compatibility_proof().empty()) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD,
                "LinkState is missing version, age, or compatibility fields");
  }
  if (prefill->second.serving().role() != xllm::proto::ENGINE_ROLE_PREFILL ||
      decode->second.serving().role() != xllm::proto::ENGINE_ROLE_DECODE ||
      prefill->second.kv().connector() != state.connector() ||
      decode->second.kv().connector() != state.connector() ||
      prefill->second.kv().connector_version() != state.connector_version() ||
      decode->second.kv().connector_version() != state.connector_version()) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
                "LinkState role or Connector does not match its members");
  }
  const auto supports_transfer =
      [&state](const xllm::proto::ProviderDescriptor& descriptor) {
        return std::any_of(
            descriptor.serving().execution_modes().begin(),
            descriptor.serving().execution_modes().end(),
            [&state](const xllm::proto::ExecutionModeSpec& mode) {
              return mode.mode() == xllm::proto::EXECUTION_MODE_REMOTE_PD &&
                     mode.transfer_mode() == state.transfer_mode();
            });
      };
  if (!supports_transfer(prefill->second) ||
      !supports_transfer(decode->second)) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
                "LinkState transfer mode does not match its members");
  }
  std::string expected_proof;
  ContractResult compatibility = validate_remote_pd_compatibility(
      prefill->second, decode->second, &expected_proof);
  if (!compatibility.ok()) {
    return compatibility;
  }
  if (expected_proof != state.compatibility_proof()) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
                "LinkState compatibility proof does not match descriptors");
  }
  return ContractResult::success();
}

ContractResult EngineRegistry::apply_state_batch(
    const xllm::proto::StateBatch& batch,
    uint64_t receiver_monotonic_ms,
    bool* applied) {
  if (applied == nullptr) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD,
                "StateBatch applied output must not be null");
  }
  *applied = false;
  if (!config_valid_) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "Engine Registry configuration is invalid");
  }
  if (batch.contract_version() != kProviderContractVersion) {
    return fail(
        xllm::proto::PROVIDER_CONTRACT_ERROR_UNSUPPORTED_CONTRACT_VERSION,
        "StateBatch contract version is unsupported");
  }
  if (batch.master_incarnation().empty() || batch.snapshot_seq() == 0) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD,
                "StateBatch version, master, and sequence are required");
  }
  if (!xllm::proto::StateBatchKind_IsValid(batch.kind()) ||
      batch.kind() == xllm::proto::STATE_BATCH_KIND_UNSPECIFIED) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_UNKNOWN_ENUM_VALUE,
                "StateBatch kind is unknown");
  }
  if (batch.engine_states_size() > static_cast<int>(config_.max_members) ||
      batch.removed_engines_size() > static_cast<int>(config_.max_members) ||
      batch.link_states_size() > static_cast<int>(config_.max_links)) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "StateBatch exceeds configured capacity");
  }

  std::unique_lock lock(mutex_);
  if (!registry_known_ || batch.master_incarnation() != master_incarnation_) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
                "StateBatch was not published by the Registry current master");
  }
  if (batch.snapshot_seq() <= last_snapshot_seq_) {
    return ContractResult::success();
  }
  if (batch.kind() == xllm::proto::STATE_BATCH_KIND_DELTA &&
      full_snapshot_master_incarnation_ != master_incarnation_) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "StateBatch DELTA arrived before the current master FULL");
  }
  if (batch.kind() == xllm::proto::STATE_BATCH_KIND_FULL &&
      !batch.removed_engines().empty()) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "StateBatch FULL must not contain removed engines");
  }

  std::map<EngineKey, CachedEngineState> incoming_states;
  for (const xllm::proto::EngineState& state : batch.engine_states()) {
    xllm::proto::ProviderEngineKey wire_key;
    wire_key.set_provider_id(state.provider_id());
    wire_key.set_profile_digest(state.profile_digest());
    wire_key.set_engine_uid(state.engine_uid());
    wire_key.set_incarnation_id(state.incarnation_id());
    const std::optional<EngineKey> key = to_engine_key(wire_key);
    if (!key.has_value() || !state.has_heartbeat_age_ms_at_publish() ||
        !state.has_state_age_ms_at_publish()) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD,
                  "EngineState identity and publish ages are required");
    }
    const auto member = members_.find(key.value());
    if (member == members_.end()) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
                  "EngineState does not match a current Registry member");
    }
    ContractResult validation = validate_engine_state(member->second, state);
    if (!validation.ok()) {
      return validation;
    }
    if (!incoming_states
             .emplace(key.value(),
                      CachedEngineState{
                          .state = state,
                          .received_monotonic_ms = receiver_monotonic_ms,
                      })
             .second) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DUPLICATE_ENTRY,
                  "StateBatch contains a duplicate EngineState");
    }
  }

  std::set<EngineKey> removed;
  for (const xllm::proto::ProviderEngineKey& wire_key :
       batch.removed_engines()) {
    const std::optional<EngineKey> key = to_engine_key(wire_key);
    const auto member =
        key.has_value() ? members_.find(key.value()) : members_.end();
    if (!key.has_value() || member == members_.end() ||
        member->second.identity().engine_uid() != wire_key.engine_uid()) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
                  "StateBatch removal is not a current Registry member");
    }
    if (incoming_states.find(key.value()) != incoming_states.end() ||
        !removed.insert(key.value()).second) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DUPLICATE_ENTRY,
                  "StateBatch duplicates an EngineState update or removal");
    }
  }

  std::map<LinkKey, CachedLinkState> incoming_links;
  for (const xllm::proto::LinkState& state : batch.link_states()) {
    ContractResult validation = validate_link_state_locked(state);
    if (!validation.ok()) {
      return validation;
    }
    const LinkKey key{
        .prefill = to_engine_key(state.prefill()).value(),
        .decode = to_engine_key(state.decode()).value(),
    };
    if (!incoming_links
             .emplace(key,
                      CachedLinkState{
                          .state = state,
                          .received_monotonic_ms = receiver_monotonic_ms,
                      })
             .second) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DUPLICATE_ENTRY,
                  "StateBatch contains a duplicate LinkState");
    }
  }

  if (batch.kind() == xllm::proto::STATE_BATCH_KIND_FULL) {
    if (incoming_states.size() != members_.size()) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                  "StateBatch FULL does not cover every Registry member");
    }
    states_ = std::move(incoming_states);
    links_ = std::move(incoming_links);
    full_snapshot_master_incarnation_ = master_incarnation_;
  } else {
    for (auto& [key, incoming] : incoming_states) {
      const auto existing = states_.find(key);
      if (existing == states_.end() ||
          incoming.state.state_seq() > existing->second.state.state_seq()) {
        states_.insert_or_assign(key, std::move(incoming));
      }
    }
    for (const EngineKey& key : removed) {
      states_.erase(key);
      for (auto link = links_.begin(); link != links_.end();) {
        if (same_engine_key(link->first.prefill, key) ||
            same_engine_key(link->first.decode, key)) {
          link = links_.erase(link);
        } else {
          ++link;
        }
      }
    }
    for (auto& [key, incoming] : incoming_links) {
      const auto existing = links_.find(key);
      if (existing == links_.end() ||
          incoming.state.state_seq() > existing->second.state.state_seq()) {
        links_.insert_or_assign(key, std::move(incoming));
      }
    }
  }

  last_snapshot_seq_ = batch.snapshot_seq();
  *applied = true;
  return ContractResult::success();
}

std::optional<xllm::proto::ProviderDescriptor> EngineRegistry::find_member(
    const xllm::proto::ProviderEngineKey& wire_key) const {
  const std::optional<EngineKey> key = to_engine_key(wire_key);
  if (!key.has_value()) {
    return std::nullopt;
  }
  std::shared_lock lock(mutex_);
  const auto member = members_.find(key.value());
  if (member == members_.end() ||
      member->second.identity().engine_uid() != wire_key.engine_uid()) {
    return std::nullopt;
  }
  return member->second;
}

std::optional<xllm::proto::EngineState> EngineRegistry::find_state(
    const xllm::proto::ProviderEngineKey& wire_key) const {
  const std::optional<EngineKey> key = to_engine_key(wire_key);
  if (!key.has_value()) {
    return std::nullopt;
  }
  std::shared_lock lock(mutex_);
  const auto state = states_.find(key.value());
  if (state == states_.end() ||
      state->second.state.engine_uid() != wire_key.engine_uid()) {
    return std::nullopt;
  }
  return state->second.state;
}

EngineStateFreshness EngineRegistry::state_freshness(
    const xllm::proto::ProviderEngineKey& wire_key,
    uint64_t receiver_monotonic_ms) const {
  const std::optional<EngineKey> key = to_engine_key(wire_key);
  if (!key.has_value()) {
    return EngineStateFreshness::MISSING;
  }
  std::shared_lock lock(mutex_);
  const auto state = states_.find(key.value());
  if (state == states_.end() ||
      state->second.state.engine_uid() != wire_key.engine_uid() ||
      !state->second.state.has_state_age_ms_at_publish()) {
    return EngineStateFreshness::MISSING;
  }
  const uint64_t age =
      effective_age_ms(state->second.state.state_age_ms_at_publish(),
                       state->second.received_monotonic_ms,
                       receiver_monotonic_ms);
  if (age > config_.state_hard_ttl_ms) {
    return EngineStateFreshness::HARD_STALE;
  }
  if (age > config_.state_soft_ttl_ms) {
    return EngineStateFreshness::SOFT_STALE;
  }
  return EngineStateFreshness::FRESH;
}

bool EngineRegistry::is_schedulable(
    const xllm::proto::ProviderEngineKey& wire_key,
    uint64_t receiver_monotonic_ms) const {
  const std::optional<EngineKey> key = to_engine_key(wire_key);
  if (!key.has_value()) {
    return false;
  }
  std::shared_lock lock(mutex_);
  if (!registry_known_ ||
      full_snapshot_master_incarnation_ != master_incarnation_) {
    return false;
  }
  const auto member = members_.find(key.value());
  const auto state = states_.find(key.value());
  if (member == members_.end() || state == states_.end() ||
      member->second.identity().engine_uid() != wire_key.engine_uid() ||
      state->second.state.engine_uid() != wire_key.engine_uid()) {
    return false;
  }
  const xllm::proto::EngineState& engine_state = state->second.state;
  if (engine_state.lifecycle() != xllm::proto::ENGINE_LIFECYCLE_READY ||
      engine_state.ownership() != xllm::proto::ENGINE_OWNERSHIP_OWNED ||
      engine_state.shallow_health() != xllm::proto::HEALTH_STATUS_HEALTHY ||
      engine_state.state_quality() == xllm::proto::STATE_QUALITY_STALE ||
      !engine_state.has_heartbeat_age_ms_at_publish() ||
      !engine_state.has_state_age_ms_at_publish()) {
    return false;
  }
  if (has_capability(member->second,
                     xllm::proto::PROVIDER_CAPABILITY_DEEP_HEALTH) &&
      engine_state.deep_health() != xllm::proto::HEALTH_STATUS_HEALTHY) {
    return false;
  }
  const uint64_t state_age =
      effective_age_ms(engine_state.state_age_ms_at_publish(),
                       state->second.received_monotonic_ms,
                       receiver_monotonic_ms);
  const uint64_t heartbeat_age =
      effective_age_ms(engine_state.heartbeat_age_ms_at_publish(),
                       state->second.received_monotonic_ms,
                       receiver_monotonic_ms);
  return state_age <= config_.state_hard_ttl_ms &&
         heartbeat_age <= config_.heartbeat_hard_ttl_ms;
}

bool EngineRegistry::is_link_ready(
    const xllm::proto::ProviderEngineKey& prefill,
    const xllm::proto::ProviderEngineKey& decode,
    uint64_t receiver_monotonic_ms) const {
  const std::optional<EngineKey> prefill_key = to_engine_key(prefill);
  const std::optional<EngineKey> decode_key = to_engine_key(decode);
  if (!prefill_key.has_value() || !decode_key.has_value()) {
    return false;
  }
  std::shared_lock lock(mutex_);
  if (!registry_known_ ||
      full_snapshot_master_incarnation_ != master_incarnation_) {
    return false;
  }
  const auto link = links_.find(
      LinkKey{.prefill = prefill_key.value(), .decode = decode_key.value()});
  if (link == links_.end() ||
      link->second.state.prefill().engine_uid() != prefill.engine_uid() ||
      link->second.state.decode().engine_uid() != decode.engine_uid() ||
      link->second.state.lifecycle() != xllm::proto::LINK_LIFECYCLE_READY ||
      !link->second.state.has_age_ms_at_publish()) {
    return false;
  }
  const uint64_t age = effective_age_ms(link->second.state.age_ms_at_publish(),
                                        link->second.received_monotonic_ms,
                                        receiver_monotonic_ms);
  return age <= config_.link_hard_ttl_ms;
}

bool EngineRegistry::registry_known() const {
  std::shared_lock lock(mutex_);
  return registry_known_;
}

bool EngineRegistry::has_current_full_snapshot() const {
  std::shared_lock lock(mutex_);
  return registry_known_ &&
         full_snapshot_master_incarnation_ == master_incarnation_;
}

size_t EngineRegistry::member_count() const {
  std::shared_lock lock(mutex_);
  return members_.size();
}

size_t EngineRegistry::state_count() const {
  std::shared_lock lock(mutex_);
  return states_.size();
}

size_t EngineRegistry::link_count() const {
  std::shared_lock lock(mutex_);
  return links_.size();
}

}  // namespace xllm_service::provider
