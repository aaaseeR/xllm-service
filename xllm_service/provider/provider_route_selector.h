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
#include <string>
#include <vector>

#include "provider.pb.h"

namespace xllm_service::provider {

struct ProviderRouteCandidate {
  std::string engine_uid;
  xllm::proto::ProviderId provider_id = xllm::proto::PROVIDER_ID_UNSPECIFIED;
  xllm::proto::EngineRole role = xllm::proto::ENGINE_ROLE_UNSPECIFIED;
  bool schedulable = false;
  std::string model_revision;
  // Non-owning immutable view valid for the duration of select(). nullptr is
  // the explicit BEST_EFFORT legacy registration path.
  const xllm::proto::ProviderDescriptor* descriptor = nullptr;
  // When a current State Stream FULL exists, a strict Native P candidate
  // lists only Decode peers whose incarnation-scoped LinkState is READY.
  bool link_state_required = false;
  std::vector<std::string> ready_peer_engine_uids;
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
      ProviderRouteSelection* selection,
      const std::string& required_model_revision = "");

  // Enumerates only complete plans that pass the same Provider, Descriptor,
  // lifecycle/capability and incarnation-scoped LinkState hard filters as
  // select(). The output is bounded; truncation is explicit so callers can
  // fail closed to their load-only path instead of ranking an incomplete set.
  static bool select_candidates(
      const std::vector<ProviderRouteCandidate>& prefill_candidates,
      const std::vector<ProviderRouteCandidate>& decode_candidates,
      xllm::proto::ProviderId required_provider_id,
      size_t max_selections,
      std::vector<ProviderRouteSelection>* selections,
      bool* truncated,
      const std::string& required_model_revision = "");
};

}  // namespace xllm_service::provider
