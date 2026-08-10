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

#include "provider/provider_registry.h"

#include <google/protobuf/util/message_differencer.h>

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <utility>

namespace xllm_service::provider {
namespace {

ContractResult validate_adapter(const ProviderAdapter* adapter) {
  if (adapter == nullptr) {
    return ContractResult::failure(
        xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD,
        "provider adapter must not be null");
  }
  ContractResult validation = validate_provider_descriptor(adapter->describe());
  if (!validation.ok()) {
    return validation;
  }

  const xllm::proto::ProviderDescriptor& descriptor = adapter->describe();
  const bool native_dispatch =
      adapter->dispatch_kind() == ProviderDispatchKind::XLLM_NATIVE_RPC;
  const bool native_descriptor = descriptor.identity().provider_id() ==
                                 xllm::proto::PROVIDER_ID_XLLM_NATIVE;
  if (native_dispatch != native_descriptor) {
    return ContractResult::failure(
        xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
        "provider descriptor does not match Adapter dispatch kind");
  }
  return ContractResult::success();
}

bool same_adapter_contract(const xllm::proto::ProviderDescriptor& left,
                           const xllm::proto::ProviderDescriptor& right) {
  // An Adapter is cached at provider-profile scope and is deliberately not
  // bound to an Engine incarnation.  Engine UID, incarnation and dial address
  // differ between replicas of the same immutable profile; every other field
  // remains part of the collision check.  Keeping transport/runtime/model/KV/
  // scheduler/capabilities in the comparison preserves fail-closed behavior
  // if a producer incorrectly reuses a profile digest.
  xllm::proto::ProviderDescriptor left_contract = left;
  xllm::proto::ProviderDescriptor right_contract = right;
  left_contract.mutable_identity()->clear_engine_uid();
  left_contract.mutable_identity()->clear_incarnation_id();
  left_contract.mutable_endpoint()->clear_address();
  right_contract.mutable_identity()->clear_engine_uid();
  right_contract.mutable_identity()->clear_incarnation_id();
  right_contract.mutable_endpoint()->clear_address();
  return google::protobuf::util::MessageDifferencer::Equivalent(left_contract,
                                                                right_contract);
}

bool same_adapter_contract(const ProviderAdapter& left,
                           const ProviderAdapter& right) {
  if (left.dispatch_kind() != right.dispatch_kind()) {
    return false;
  }
  return same_adapter_contract(left.describe(), right.describe());
}

}  // namespace

ContractResult ProviderAdapterRegistry::register_adapter(
    std::unique_ptr<ProviderAdapter> adapter) {
  ContractResult validation = validate_adapter(adapter.get());
  if (!validation.ok()) {
    return validation;
  }

  const auto& descriptor = adapter->describe();
  Key key{static_cast<int>(descriptor.identity().provider_id()),
          descriptor.profile_digest()};
  std::unique_lock lock(mutex_);
  if (adapters_.find(key) != adapters_.end()) {
    return ContractResult::failure(
        xllm::proto::PROVIDER_CONTRACT_ERROR_DUPLICATE_ADAPTER,
        "provider adapter key is already registered");
  }
  adapters_.emplace(std::move(key), std::move(adapter));
  return ContractResult::success();
}

ContractResult ProviderAdapterRegistry::find_or_register_adapter(
    std::unique_ptr<ProviderAdapter> adapter,
    const ProviderAdapter** registered_adapter) {
  if (registered_adapter == nullptr) {
    return ContractResult::failure(
        xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD,
        "registered Adapter output must not be null");
  }
  *registered_adapter = nullptr;
  ContractResult validation = validate_adapter(adapter.get());
  if (!validation.ok()) {
    return validation;
  }

  const xllm::proto::ProviderDescriptor& descriptor = adapter->describe();
  Key key{static_cast<int>(descriptor.identity().provider_id()),
          descriptor.profile_digest()};
  std::unique_lock lock(mutex_);
  const auto existing = adapters_.find(key);
  if (existing != adapters_.end()) {
    if (!same_adapter_contract(*existing->second, *adapter)) {
      return ContractResult::failure(
          xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
          "Provider Adapter key collides with a different Descriptor");
    }
    *registered_adapter = existing->second.get();
    return ContractResult::success();
  }

  auto [inserted, did_insert] =
      adapters_.emplace(std::move(key), std::move(adapter));
  if (!did_insert) {
    return ContractResult::failure(
        xllm::proto::PROVIDER_CONTRACT_ERROR_DUPLICATE_ADAPTER,
        "Provider Adapter could not be installed");
  }
  *registered_adapter = inserted->second.get();
  return ContractResult::success();
}

ContractResult ProviderAdapterRegistry::find_compatible_adapter(
    const xllm::proto::ProviderDescriptor& descriptor,
    const ProviderAdapter** adapter) const {
  if (adapter == nullptr) {
    return ContractResult::failure(
        xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD,
        "compatible Adapter output must not be null");
  }
  *adapter = nullptr;
  ContractResult validation = validate_provider_descriptor(descriptor);
  if (!validation.ok()) {
    return validation;
  }

  std::shared_lock lock(mutex_);
  const auto existing =
      adapters_.find(Key{static_cast<int>(descriptor.identity().provider_id()),
                         descriptor.profile_digest()});
  if (existing == adapters_.end()) {
    return ContractResult::success();
  }
  if (!same_adapter_contract(existing->second->describe(), descriptor)) {
    return ContractResult::failure(
        xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
        "Provider Adapter key collides with a different Descriptor");
  }
  *adapter = existing->second.get();
  return ContractResult::success();
}

const ProviderAdapter* ProviderAdapterRegistry::find(
    xllm::proto::ProviderId provider_id,
    const std::string& profile_digest) const {
  std::shared_lock lock(mutex_);
  const auto it =
      adapters_.find(Key{static_cast<int>(provider_id), profile_digest});
  return it == adapters_.end() ? nullptr : it->second.get();
}

size_t ProviderAdapterRegistry::size() const {
  std::shared_lock lock(mutex_);
  return adapters_.size();
}

}  // namespace xllm_service::provider
