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

#include "routing/pd_compatibility.h"

namespace xllm_service {
namespace {

PdCompatibilityResult make_error(PdCompatibilityError error,
                                 const std::string& message) {
  return {error, message};
}

bool is_prefill_role(const InstanceMetaInfo& instance) {
  return instance.type == InstanceType::PREFILL ||
         (instance.type == InstanceType::MIX &&
          instance.current_type == InstanceType::PREFILL);
}

bool is_decode_role(const InstanceMetaInfo& instance) {
  return instance.type == InstanceType::DECODE ||
         (instance.type == InstanceType::MIX &&
          instance.current_type == InstanceType::DECODE);
}

}  // namespace

bool pd_compatibility_result_ok(const PdCompatibilityResult& result) {
  return result.error == PdCompatibilityError::NONE;
}

PdCompatibilityResult validate_external_pd_compatibility(
    const InstanceMetaInfo& prefill,
    const InstanceMetaInfo& decode,
    int32_t expected_block_size,
    uint32_t expected_hash_seed) {
  if (prefill.name.empty() || prefill.name == decode.name) {
    return make_error(PdCompatibilityError::SAME_ENDPOINT,
                      "P/D endpoints must be distinct");
  }
  if (!is_prefill_role(prefill)) {
    return make_error(PdCompatibilityError::INVALID_PREFILL_ROLE,
                      "Selected prefill endpoint is not in prefill role");
  }
  if (!is_decode_role(decode)) {
    return make_error(PdCompatibilityError::INVALID_DECODE_ROLE,
                      "Selected decode endpoint is not in decode role");
  }
  if (prefill.block_size <= 0 || decode.block_size <= 0 ||
      prefill.xxh3_128bits_seed == 0 || decode.xxh3_128bits_seed == 0 ||
      prefill.kv_split_size <= 0 || decode.kv_split_size <= 0) {
    return make_error(
        PdCompatibilityError::MISSING_CACHE_CONTRACT,
        "P/D endpoints must publish cache compatibility metadata");
  }
  if (prefill.block_size != decode.block_size ||
      prefill.block_size != expected_block_size) {
    return make_error(PdCompatibilityError::BLOCK_SIZE_MISMATCH,
                      "P/D block size does not match the adapter contract");
  }
  if (prefill.xxh3_128bits_seed != decode.xxh3_128bits_seed ||
      prefill.xxh3_128bits_seed != expected_hash_seed) {
    return make_error(PdCompatibilityError::HASH_SEED_MISMATCH,
                      "P/D hash seed does not match the adapter contract");
  }
  if (prefill.kv_split_size != decode.kv_split_size) {
    return make_error(PdCompatibilityError::KV_SPLIT_SIZE_MISMATCH,
                      "P/D kv_split_size values do not match");
  }
  return {};
}

}  // namespace xllm_service
