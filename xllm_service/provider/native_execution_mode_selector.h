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

#include "provider.pb.h"
#include "provider/provider_contract.h"

namespace xllm_service::provider {

struct NativeExecutionModeConfig final {
  bool local_prefill_decode_enabled = false;
  uint32_t local_prefill_decode_bucket_permyriad = 0;
  uint64_t local_prefill_decode_prompt_token_cap = 0;
  bool prefill_only_enabled = false;
  uint64_t prefill_only_output_token_cap = 0;
};

struct NativeExecutionModeInput final {
  uint64_t stable_request_hash = 0;
  uint64_t prompt_tokens = 0;
  uint64_t output_tokens = 0;
  const xllm::proto::ProviderDescriptor* prefill = nullptr;
  const xllm::proto::ProviderDescriptor* decode = nullptr;
};

struct NativeExecutionModeDecision final {
  xllm::proto::ExecutionMode mode = xllm::proto::EXECUTION_MODE_UNSPECIFIED;
  std::string reason_code;
};

// Selects only descriptor-open Native modes. P-only has precedence over the
// local D canary; REMOTE_PD is the fail-closed baseline.
ContractResult select_native_execution_mode(
    const NativeExecutionModeConfig& config,
    const NativeExecutionModeInput& input,
    NativeExecutionModeDecision* decision);

}  // namespace xllm_service::provider
