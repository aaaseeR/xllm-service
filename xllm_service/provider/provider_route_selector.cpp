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

#include "provider/provider_route_selector.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace xllm_service::provider {
namespace {

bool is_supported_provider(xllm::proto::ProviderId provider_id) {
  return provider_id == xllm::proto::PROVIDER_ID_XLLM_NATIVE ||
         provider_id == xllm::proto::PROVIDER_ID_VLLM_ASCEND;
}

bool provider_matches(xllm::proto::ProviderId candidate_provider_id,
                      xllm::proto::ProviderId required_provider_id) {
  return required_provider_id == xllm::proto::PROVIDER_ID_UNSPECIFIED ||
         candidate_provider_id == required_provider_id;
}

}  // namespace

bool ProviderRouteSelector::select(
    const std::vector<ProviderRouteCandidate>& prefill_candidates,
    const std::vector<ProviderRouteCandidate>& decode_candidates,
    xllm::proto::ProviderId required_provider_id,
    uint64_t prefill_start_index,
    uint64_t decode_start_index,
    ProviderRouteSelection* selection) {
  if (selection == nullptr || prefill_candidates.empty()) {
    return false;
  }

  const uint64_t prefill_size =
      static_cast<uint64_t>(prefill_candidates.size());
  const uint64_t prefill_begin = prefill_start_index % prefill_size;
  for (uint64_t prefill_offset = 0; prefill_offset < prefill_size;
       ++prefill_offset) {
    const uint64_t prefill_index =
        (prefill_begin + prefill_offset) % prefill_size;
    const ProviderRouteCandidate& prefill = prefill_candidates[prefill_index];
    if (!prefill.schedulable || prefill.engine_uid.empty() ||
        !is_supported_provider(prefill.provider_id) ||
        !provider_matches(prefill.provider_id, required_provider_id)) {
      continue;
    }

    if (prefill.role == xllm::proto::ENGINE_ROLE_AGGREGATED) {
      selection->prefill_engine_uid = prefill.engine_uid;
      selection->decode_engine_uid.clear();
      selection->provider_id = prefill.provider_id;
      selection->next_prefill_index = prefill_index + 1;
      selection->next_decode_index = decode_start_index;
      return true;
    }

    if (prefill.role != xllm::proto::ENGINE_ROLE_PREFILL ||
        prefill.provider_id != xllm::proto::PROVIDER_ID_XLLM_NATIVE ||
        decode_candidates.empty()) {
      continue;
    }

    const uint64_t decode_size =
        static_cast<uint64_t>(decode_candidates.size());
    const uint64_t decode_begin = decode_start_index % decode_size;
    for (uint64_t decode_offset = 0; decode_offset < decode_size;
         ++decode_offset) {
      const uint64_t decode_index =
          (decode_begin + decode_offset) % decode_size;
      const ProviderRouteCandidate& decode = decode_candidates[decode_index];
      if (!decode.schedulable || decode.engine_uid.empty() ||
          decode.role != xllm::proto::ENGINE_ROLE_DECODE ||
          decode.provider_id != prefill.provider_id) {
        continue;
      }

      selection->prefill_engine_uid = prefill.engine_uid;
      selection->decode_engine_uid = decode.engine_uid;
      selection->provider_id = prefill.provider_id;
      selection->next_prefill_index = prefill_index + 1;
      selection->next_decode_index = decode_index + 1;
      return true;
    }
  }
  return false;
}

}  // namespace xllm_service::provider
