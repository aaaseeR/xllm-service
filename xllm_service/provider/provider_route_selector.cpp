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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "provider/provider_contract.h"

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

bool descriptor_matches_candidate(const ProviderRouteCandidate& candidate) {
  if (candidate.descriptor == nullptr) {
    return true;
  }
  const ContractResult validation =
      validate_provider_descriptor(*candidate.descriptor);
  return validation.ok() &&
         candidate.descriptor->identity().provider_id() ==
             candidate.provider_id &&
         candidate.descriptor->identity().engine_uid() ==
             candidate.engine_uid &&
         candidate.descriptor->serving().role() == candidate.role &&
         (candidate.model_revision.empty() ||
          candidate.descriptor->model().model_revision() ==
              candidate.model_revision);
}

bool model_matches(const ProviderRouteCandidate& candidate,
                   const std::string& required_model_revision) {
  if (required_model_revision.empty()) {
    return true;
  }
  if (candidate.descriptor != nullptr) {
    return candidate.descriptor->model().model_revision() ==
           required_model_revision;
  }
  // Contract-v0 registrations do not publish a model identity and retain
  // their historical BEST_EFFORT behavior. V2 multi-model guarantees apply
  // only to immutable Descriptor-backed pools.
  return candidate.model_revision.empty() ||
         candidate.model_revision == required_model_revision;
}

bool remote_pd_compatible(const ProviderRouteCandidate& prefill,
                          const ProviderRouteCandidate& decode) {
  if (prefill.descriptor == nullptr && decode.descriptor == nullptr) {
    return true;
  }
  if (prefill.descriptor == nullptr || decode.descriptor == nullptr) {
    return false;
  }
  if (prefill.link_state_required &&
      std::find(prefill.ready_peer_engine_uids.begin(),
                prefill.ready_peer_engine_uids.end(),
                decode.engine_uid) == prefill.ready_peer_engine_uids.end()) {
    return false;
  }
  std::string compatibility_proof;
  return validate_remote_pd_compatibility(
             *prefill.descriptor, *decode.descriptor, &compatibility_proof)
      .ok();
}

}  // namespace

bool ProviderRouteSelector::select(
    const std::vector<ProviderRouteCandidate>& prefill_candidates,
    const std::vector<ProviderRouteCandidate>& decode_candidates,
    xllm::proto::ProviderId required_provider_id,
    uint64_t prefill_start_index,
    uint64_t decode_start_index,
    ProviderRouteSelection* selection,
    const std::string& required_model_revision) {
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
        !provider_matches(prefill.provider_id, required_provider_id) ||
        !model_matches(prefill, required_model_revision) ||
        !descriptor_matches_candidate(prefill)) {
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
          decode.provider_id != prefill.provider_id ||
          !model_matches(decode, required_model_revision) ||
          !descriptor_matches_candidate(decode) ||
          !remote_pd_compatible(prefill, decode)) {
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

bool ProviderRouteSelector::select_candidates(
    const std::vector<ProviderRouteCandidate>& prefill_candidates,
    const std::vector<ProviderRouteCandidate>& decode_candidates,
    xllm::proto::ProviderId required_provider_id,
    size_t max_selections,
    std::vector<ProviderRouteSelection>* selections,
    bool* truncated,
    const std::string& required_model_revision) {
  if (selections == nullptr || truncated == nullptr || max_selections == 0) {
    return false;
  }
  selections->clear();
  *truncated = false;

  const auto append = [&](const ProviderRouteCandidate& prefill,
                          const ProviderRouteCandidate* decode) {
    if (selections->size() == max_selections) {
      *truncated = true;
      return false;
    }
    selections->emplace_back(ProviderRouteSelection{
        .prefill_engine_uid = prefill.engine_uid,
        .decode_engine_uid = decode == nullptr ? "" : decode->engine_uid,
        .provider_id = prefill.provider_id,
    });
    return true;
  };

  for (const ProviderRouteCandidate& prefill : prefill_candidates) {
    if (!prefill.schedulable || prefill.engine_uid.empty() ||
        !is_supported_provider(prefill.provider_id) ||
        !provider_matches(prefill.provider_id, required_provider_id) ||
        !model_matches(prefill, required_model_revision) ||
        !descriptor_matches_candidate(prefill)) {
      continue;
    }
    if (prefill.role == xllm::proto::ENGINE_ROLE_AGGREGATED) {
      if (!append(prefill, nullptr)) {
        return true;
      }
      continue;
    }
    if (prefill.role != xllm::proto::ENGINE_ROLE_PREFILL ||
        prefill.provider_id != xllm::proto::PROVIDER_ID_XLLM_NATIVE) {
      continue;
    }
    for (const ProviderRouteCandidate& decode : decode_candidates) {
      if (!decode.schedulable || decode.engine_uid.empty() ||
          decode.role != xllm::proto::ENGINE_ROLE_DECODE ||
          decode.provider_id != prefill.provider_id ||
          !model_matches(decode, required_model_revision) ||
          !descriptor_matches_candidate(decode) ||
          !remote_pd_compatible(prefill, decode)) {
        continue;
      }
      if (!append(prefill, &decode)) {
        return true;
      }
    }
  }
  return !selections->empty();
}

}  // namespace xllm_service::provider
