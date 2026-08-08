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
#include <map>
#include <memory>
#include <shared_mutex>
#include <string>
#include <utility>

#include "provider/provider_adapter.h"

namespace xllm_service::provider {

class ProviderAdapterRegistry {
 public:
  ContractResult register_adapter(std::unique_ptr<ProviderAdapter> adapter);

  const ProviderAdapter* find(xllm::proto::ProviderId provider_id,
                              const std::string& profile_digest) const;
  size_t size() const;

 private:
  using Key = std::pair<int, std::string>;

  mutable std::shared_mutex mutex_;
  std::map<Key, std::unique_ptr<ProviderAdapter>> adapters_;
};

}  // namespace xllm_service::provider
