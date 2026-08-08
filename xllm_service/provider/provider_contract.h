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

inline constexpr uint32_t kProviderContractVersion = 1;

class ContractResult {
 public:
  static ContractResult success();
  static ContractResult failure(xllm::proto::ProviderContractError error,
                                std::string message);

  bool ok() const { return error_ == xllm::proto::PROVIDER_CONTRACT_ERROR_OK; }
  xllm::proto::ProviderContractError error() const { return error_; }
  const std::string& message() const { return message_; }

 private:
  ContractResult(xllm::proto::ProviderContractError error, std::string message);

  xllm::proto::ProviderContractError error_;
  std::string message_;
};

struct ModeRequirements {
  std::vector<xllm::proto::ProviderCapability> required_capabilities;
};

ContractResult resolve_mode_requirements(xllm::proto::ExecutionMode mode,
                                         xllm::proto::TransferMode transfer,
                                         ModeRequirements* requirements);

ContractResult validate_v2_open_mode(xllm::proto::ProviderId provider_id,
                                     xllm::proto::ExecutionMode mode,
                                     xllm::proto::TransferMode transfer,
                                     xllm::proto::SelectionOrder order,
                                     xllm::proto::BindingStage binding_stage,
                                     bool p_selection_delegated);

ContractResult validate_provider_descriptor(
    const xllm::proto::ProviderDescriptor& descriptor);
ContractResult validate_remote_pd_compatibility(
    const xllm::proto::ProviderDescriptor& prefill,
    const xllm::proto::ProviderDescriptor& decode,
    std::string* compatibility_proof);
ContractResult validate_canonical_request(
    const xllm::proto::CanonicalRequest& request);
ContractResult validate_encoded_request(
    const xllm::proto::ProviderDescriptor& descriptor,
    const xllm::proto::CanonicalRequest& canonical,
    const xllm::proto::EncodedRequest& request);
ContractResult validate_engine_state(
    const xllm::proto::ProviderDescriptor& descriptor,
    const xllm::proto::EngineState& state);
ContractResult validate_execution_plan(
    const xllm::proto::ProviderDescriptor& descriptor,
    const xllm::proto::ExecutionPlan& plan);

}  // namespace xllm_service::provider
