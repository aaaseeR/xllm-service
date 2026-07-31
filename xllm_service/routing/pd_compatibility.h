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

#include "common/types.h"

namespace xllm_service {

enum class PdCompatibilityError : uint8_t {
  NONE = 0,
  SAME_ENDPOINT = 1,
  INVALID_PREFILL_ROLE = 2,
  INVALID_DECODE_ROLE = 3,
  MISSING_CACHE_CONTRACT = 4,
  BLOCK_SIZE_MISMATCH = 5,
  HASH_SEED_MISMATCH = 6,
  KV_SPLIT_SIZE_MISMATCH = 7,
};

struct PdCompatibilityResult {
  PdCompatibilityError error = PdCompatibilityError::NONE;
  std::string message;
};

bool pd_compatibility_result_ok(const PdCompatibilityResult& result);
PdCompatibilityResult validate_external_pd_compatibility(
    const InstanceMetaInfo& prefill,
    const InstanceMetaInfo& decode,
    int32_t expected_block_size,
    uint32_t expected_hash_seed);

}  // namespace xllm_service
