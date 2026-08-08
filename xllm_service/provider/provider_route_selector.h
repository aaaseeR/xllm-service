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

#include <cstdint>
#include <string>
#include <vector>

#include "provider.pb.h"

namespace xllm_service::provider {

struct ProviderRouteCandidate {
  std::string engine_uid;
  xllm::proto::ProviderId provider_id = xllm::proto::PROVIDER_ID_UNSPECIFIED;
  xllm::proto::EngineRole role = xllm::proto::ENGINE_ROLE_UNSPECIFIED;
  bool schedulable = false;
};

struct ProviderRouteSelection {
  std::string prefill_engine_uid;
  std::string decode_engine_uid;
  xllm::proto::ProviderId provider_id = xllm::proto::PROVIDER_ID_UNSPECIFIED;
  uint64_t next_prefill_index = 0;
  uint64_t next_decode_index = 0;
};

class ProviderRouteSelector final {
 public:
  static bool select(
      const std::vector<ProviderRouteCandidate>& prefill_candidates,
      const std::vector<ProviderRouteCandidate>& decode_candidates,
      xllm::proto::ProviderId required_provider_id,
      uint64_t prefill_start_index,
      uint64_t decode_start_index,
      ProviderRouteSelection* selection);
};

}  // namespace xllm_service::provider
