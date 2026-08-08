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

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <utility>

namespace xllm_service::provider {

ContractResult ProviderAdapterRegistry::register_adapter(
    std::unique_ptr<ProviderAdapter> adapter) {
  if (adapter == nullptr) {
    return ContractResult::failure(
        xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD,
        "provider adapter must not be null");
  }
  ContractResult validation = validate_provider_descriptor(adapter->describe());
  if (!validation.ok()) {
    return validation;
  }

  const auto& descriptor = adapter->describe();
  const bool native_dispatch =
      adapter->dispatch_kind() == ProviderDispatchKind::XLLM_NATIVE_RPC;
  const bool native_descriptor = descriptor.identity().provider_id() ==
                                 xllm::proto::PROVIDER_ID_XLLM_NATIVE;
  if (native_dispatch != native_descriptor) {
    return ContractResult::failure(
        xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
        "provider descriptor does not match Adapter dispatch kind");
  }
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
