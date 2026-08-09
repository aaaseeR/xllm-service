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

#include "provider/native_execution_mode_selector.h"

#include <unordered_set>

namespace xllm_service::provider {
namespace {

bool has_capabilities(
    const xllm::proto::ProviderDescriptor& descriptor,
    std::initializer_list<xllm::proto::ProviderCapability> required) {
  std::unordered_set<int32_t> capabilities;
  capabilities.reserve(descriptor.capabilities_size());
  for (const int32_t capability : descriptor.capabilities()) {
    capabilities.insert(capability);
  }
  for (const xllm::proto::ProviderCapability capability : required) {
    if (capabilities.find(static_cast<int32_t>(capability)) ==
        capabilities.end()) {
      return false;
    }
  }
  return true;
}

bool opens_mode(const xllm::proto::ProviderDescriptor& descriptor,
                xllm::proto::ExecutionMode mode) {
  if (descriptor.identity().provider_id() !=
      xllm::proto::PROVIDER_ID_XLLM_NATIVE) {
    return false;
  }
  for (const xllm::proto::ExecutionModeSpec& spec :
       descriptor.serving().execution_modes()) {
    if (spec.mode() != mode || spec.p_selection_delegated()) {
      continue;
    }
    switch (mode) {
      case xllm::proto::EXECUTION_MODE_REMOTE_PD:
        return spec.transfer_mode() ==
                   xllm::proto::TRANSFER_MODE_LAYERWISE_PUSH &&
               spec.selection_order() == xllm::proto::SELECTION_ORDER_P_FIRST &&
               spec.binding_stage() ==
                   xllm::proto::BINDING_STAGE_BEFORE_PREFILL &&
               has_capabilities(
                   descriptor,
                   {xllm::proto::PROVIDER_CAPABILITY_REMOTE_PD_LAYERWISE_PUSH,
                    xllm::proto::PROVIDER_CAPABILITY_NATIVE_RESERVATION});
      case xllm::proto::EXECUTION_MODE_LOCAL_PREFILL_DECODE:
        return descriptor.serving().role() == xllm::proto::ENGINE_ROLE_DECODE &&
               spec.transfer_mode() == xllm::proto::TRANSFER_MODE_NONE &&
               spec.selection_order() == xllm::proto::SELECTION_ORDER_D_ONLY &&
               spec.binding_stage() == xllm::proto::BINDING_STAGE_AT_SUBMIT &&
               has_capabilities(
                   descriptor,
                   {xllm::proto::PROVIDER_CAPABILITY_LOCAL_PREFILL_DECODE,
                    xllm::proto::
                        PROVIDER_CAPABILITY_MIXED_PREFILL_DECODE_ACCOUNTING,
                    xllm::proto::PROVIDER_CAPABILITY_STRUCTURED_ADMISSION});
      case xllm::proto::EXECUTION_MODE_PREFILL_ONLY:
        return descriptor.serving().role() ==
                   xllm::proto::ENGINE_ROLE_PREFILL &&
               spec.transfer_mode() == xllm::proto::TRANSFER_MODE_NONE &&
               spec.selection_order() == xllm::proto::SELECTION_ORDER_P_ONLY &&
               spec.binding_stage() == xllm::proto::BINDING_STAGE_AT_SUBMIT &&
               has_capabilities(
                   descriptor,
                   {xllm::proto::PROVIDER_CAPABILITY_PREFILL_ONLY,
                    xllm::proto::PROVIDER_CAPABILITY_STRUCTURED_ADMISSION});
      default:
        return false;
    }
  }
  return false;
}

ContractResult invalid(const std::string& message) {
  return ContractResult::failure(
      xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE, message);
}

}  // namespace

ContractResult select_native_execution_mode(
    const NativeExecutionModeConfig& config,
    const NativeExecutionModeInput& input,
    NativeExecutionModeDecision* decision) {
  if (decision == nullptr) {
    return ContractResult::failure(
        xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD,
        "Native execution mode decision must not be null");
  }
  decision->mode = xllm::proto::EXECUTION_MODE_UNSPECIFIED;
  decision->reason_code.clear();
  if (input.prefill == nullptr || input.decode == nullptr) {
    return invalid("Native execution mode selection requires a P/D pair");
  }
  if (config.local_prefill_decode_bucket_permyriad > 10000 ||
      (config.local_prefill_decode_enabled &&
       config.local_prefill_decode_prompt_token_cap == 0) ||
      (config.prefill_only_enabled &&
       config.prefill_only_output_token_cap == 0)) {
    return invalid("Native execution mode selector configuration is invalid");
  }

  if (config.prefill_only_enabled &&
      input.output_tokens <= config.prefill_only_output_token_cap &&
      opens_mode(*input.prefill, xllm::proto::EXECUTION_MODE_PREFILL_ONLY)) {
    decision->mode = xllm::proto::EXECUTION_MODE_PREFILL_ONLY;
    decision->reason_code = "native-prefill-only";
    return ContractResult::success();
  }

  const bool in_local_bucket =
      config.local_prefill_decode_bucket_permyriad == 10000 ||
      input.stable_request_hash % 10000 <
          config.local_prefill_decode_bucket_permyriad;
  if (config.local_prefill_decode_enabled && in_local_bucket &&
      input.prompt_tokens <= config.local_prefill_decode_prompt_token_cap &&
      opens_mode(*input.decode,
                 xllm::proto::EXECUTION_MODE_LOCAL_PREFILL_DECODE)) {
    decision->mode = xllm::proto::EXECUTION_MODE_LOCAL_PREFILL_DECODE;
    decision->reason_code = "native-local-prefill-allowlist";
    return ContractResult::success();
  }

  if (!opens_mode(*input.prefill, xllm::proto::EXECUTION_MODE_REMOTE_PD) ||
      !opens_mode(*input.decode, xllm::proto::EXECUTION_MODE_REMOTE_PD)) {
    return invalid("Native P/D pair does not open REMOTE_PD");
  }
  decision->mode = xllm::proto::EXECUTION_MODE_REMOTE_PD;
  decision->reason_code = "native-remote-pd-baseline";
  return ContractResult::success();
}

}  // namespace xllm_service::provider
