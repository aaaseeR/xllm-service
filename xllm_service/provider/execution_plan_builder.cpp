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

#include "provider/execution_plan_builder.h"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace xllm_service::provider {
namespace {

const xllm::proto::ExecutionModeSpec* find_open_mode(
    const xllm::proto::ProviderDescriptor& descriptor) {
  const xllm::proto::ProviderId provider_id =
      descriptor.identity().provider_id();
  for (const xllm::proto::ExecutionModeSpec& spec :
       descriptor.serving().execution_modes()) {
    const bool native_remote_pd =
        provider_id == xllm::proto::PROVIDER_ID_XLLM_NATIVE &&
        spec.mode() == xllm::proto::EXECUTION_MODE_REMOTE_PD &&
        spec.transfer_mode() == xllm::proto::TRANSFER_MODE_LAYERWISE_PUSH &&
        spec.selection_order() == xllm::proto::SELECTION_ORDER_P_FIRST &&
        spec.binding_stage() == xllm::proto::BINDING_STAGE_BEFORE_PREFILL &&
        !spec.p_selection_delegated();
    const bool vllm_aggregated =
        provider_id == xllm::proto::PROVIDER_ID_VLLM_ASCEND &&
        spec.mode() == xllm::proto::EXECUTION_MODE_AGGREGATED &&
        spec.transfer_mode() == xllm::proto::TRANSFER_MODE_NONE &&
        spec.selection_order() == xllm::proto::SELECTION_ORDER_SINGLE &&
        spec.binding_stage() == xllm::proto::BINDING_STAGE_AT_SUBMIT &&
        !spec.p_selection_delegated();
    if (native_remote_pd || vllm_aggregated) {
      return &spec;
    }
  }
  return nullptr;
}

void add_selected_role(const xllm::proto::ProviderDescriptor& descriptor,
                       uint32_t order_index,
                       xllm::proto::ExecutionPlan* plan) {
  xllm::proto::SelectedEngineRole* selected = plan->add_selected_roles();
  selected->set_role(descriptor.serving().role());
  selected->set_engine_uid(descriptor.identity().engine_uid());
  selected->set_incarnation_id(descriptor.identity().incarnation_id());
  selected->set_order_index(order_index);
}

ContractResult set_resource_estimate(
    const xllm::proto::CanonicalRequest& canonical,
    const xllm::proto::EncodedRequest& encoded,
    const xllm::proto::ProviderDescriptor& descriptor,
    xllm::proto::ExecutionPlan* plan) {
  xllm::proto::PlanResourceEstimate* estimate =
      plan->mutable_resource_estimate();
  estimate->set_prompt_tokens(encoded.prompt_tokens());
  estimate->set_output_tokens(canonical.effective_max_new_tokens());
  if (encoded.token_count_quality() ==
      xllm::proto::TOKEN_COUNT_QUALITY_UNKNOWN) {
    estimate->set_kv_blocks(0);
    plan->add_reason_codes("prompt-token-count-unknown");
    return ContractResult::success();
  }

  if (encoded.prompt_tokens() > std::numeric_limits<uint64_t>::max() -
                                    canonical.effective_max_new_tokens()) {
    return ContractResult::failure(
        xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
        "execution plan token estimate overflows");
  }
  const uint64_t total_tokens =
      encoded.prompt_tokens() + canonical.effective_max_new_tokens();
  const uint64_t block_size = descriptor.kv().block_size();
  if (block_size == 0 ||
      (total_tokens > 0 &&
       total_tokens - 1 > std::numeric_limits<uint64_t>::max() - block_size)) {
    return ContractResult::failure(
        xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
        "execution plan KV block estimate overflows");
  }

  estimate->set_kv_blocks(
      total_tokens == 0 ? 0 : (total_tokens + block_size - 1) / block_size);
  return ContractResult::success();
}

}  // namespace

ContractResult build_execution_plan(
    const xllm::proto::CanonicalRequest& canonical,
    const xllm::proto::EncodedRequest& encoded,
    const xllm::proto::ProviderDescriptor& primary,
    const xllm::proto::ProviderDescriptor* decode,
    xllm::proto::ExecutionPlan* plan) {
  if (plan == nullptr) {
    return ContractResult::failure(
        xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD,
        "execution plan output must not be null");
  }
  plan->Clear();
  ContractResult encoded_validation =
      validate_encoded_request(primary, canonical, encoded);
  if (!encoded_validation.ok()) {
    return encoded_validation;
  }
  const xllm::proto::ExecutionModeSpec* spec = find_open_mode(primary);
  if (spec == nullptr) {
    return ContractResult::failure(
        xllm::proto::PROVIDER_CONTRACT_ERROR_MODE_NOT_OPEN,
        "selected Provider has no open V2 execution mode");
  }

  std::string compatibility_proof;
  if (primary.identity().provider_id() ==
      xllm::proto::PROVIDER_ID_XLLM_NATIVE) {
    if (decode == nullptr) {
      return ContractResult::failure(
          xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD,
          "xLLM Native REMOTE_PD plan requires a Decode descriptor");
    }
    ContractResult compatibility = validate_remote_pd_compatibility(
        primary, *decode, &compatibility_proof);
    if (!compatibility.ok()) {
      return compatibility;
    }
  } else {
    if (decode != nullptr ||
        primary.serving().role() != xllm::proto::ENGINE_ROLE_AGGREGATED) {
      return ContractResult::failure(
          xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_SELECTED_ROLES,
          "vLLM-Ascend AGGREGATED plan requires exactly one Engine");
    }
    compatibility_proof = "aggregated-v1|profile=" + primary.profile_digest() +
                          "|incarnation=" + primary.identity().incarnation_id();
  }

  xllm::proto::ExecutionPlan candidate;
  candidate.set_contract_version(kProviderContractVersion);
  candidate.set_request_uid(canonical.request_uid());
  candidate.set_attempt_seq(canonical.attempt_seq());
  candidate.set_provider_id(primary.identity().provider_id());
  candidate.set_mode(spec->mode());
  candidate.set_transfer_mode(spec->transfer_mode());
  candidate.set_selection_order(spec->selection_order());
  candidate.set_p_selection_delegated(spec->p_selection_delegated());
  candidate.set_binding_stage(spec->binding_stage());
  candidate.set_compatibility_proof(std::move(compatibility_proof));
  candidate.set_provider_payload(encoded.provider_payload());
  candidate.set_score(0.0);
  candidate.add_reason_codes("v2-contract-hard-filter");

  add_selected_role(primary, 0, &candidate);
  if (decode != nullptr) {
    add_selected_role(*decode, 1, &candidate);
  }

  ModeRequirements requirements;
  ContractResult resolved = resolve_mode_requirements(
      spec->mode(), spec->transfer_mode(), &requirements);
  if (!resolved.ok()) {
    return resolved;
  }
  for (const xllm::proto::ProviderCapability capability :
       requirements.required_capabilities) {
    candidate.add_required_capabilities(capability);
  }

  xllm::proto::DeadlineBudget* budget = candidate.mutable_deadline_budget();
  budget->set_remaining_ms(canonical.remaining_deadline_ms());
  budget->set_submit_ms(0);
  budget->set_handoff_ms(0);
  budget->set_output_ms(canonical.remaining_deadline_ms());

  ContractResult estimate =
      set_resource_estimate(canonical, encoded, primary, &candidate);
  if (!estimate.ok()) {
    return estimate;
  }
  ContractResult validation = validate_execution_plan(primary, candidate);
  if (!validation.ok()) {
    return validation;
  }
  *plan = std::move(candidate);
  return ContractResult::success();
}

}  // namespace xllm_service::provider
