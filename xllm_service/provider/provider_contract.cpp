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

#include "provider/provider_contract.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace xllm_service::provider {
namespace {

using xllm::proto::BindingStage;
using xllm::proto::EngineRole;
using xllm::proto::ExecutionMode;
using xllm::proto::ProviderCapability;
using xllm::proto::ProviderContractError;
using xllm::proto::ProviderDescriptor;
using xllm::proto::ProviderId;
using xllm::proto::SelectionOrder;
using xllm::proto::TransferMode;

ContractResult fail(ProviderContractError error, std::string message) {
  return ContractResult::failure(error, std::move(message));
}

ContractResult missing(std::string field) {
  return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD,
              "missing required field: " + std::move(field));
}

template <typename Repeated, typename Validator>
ContractResult validate_unique_enum_list(const Repeated& values,
                                         Validator validator,
                                         const std::string& field) {
  std::unordered_set<int> seen;
  for (const auto value : values) {
    const int raw = static_cast<int>(value);
    if (!validator(raw) || raw == 0) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_UNKNOWN_ENUM_VALUE,
                  field + " contains an unknown or unspecified enum");
    }
    if (!seen.insert(raw).second) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DUPLICATE_ENTRY,
                  field + " contains a duplicate enum");
    }
  }
  return ContractResult::success();
}

template <typename Repeated>
ContractResult validate_unique_strings(const Repeated& values,
                                       const std::string& field,
                                       bool allow_empty_list) {
  if (!allow_empty_list && values.empty()) {
    return missing(field);
  }
  std::unordered_set<std::string> seen;
  for (const std::string& value : values) {
    if (value.empty()) {
      return missing(field + " entry");
    }
    if (!seen.insert(value).second) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DUPLICATE_ENTRY,
                  field + " contains a duplicate entry");
    }
  }
  return ContractResult::success();
}

bool has_capability(const ProviderDescriptor& descriptor,
                    ProviderCapability capability) {
  for (const int published : descriptor.capabilities()) {
    if (published == static_cast<int>(capability)) {
      return true;
    }
  }
  return false;
}

bool has_transfer_mode(const ProviderDescriptor& descriptor,
                       TransferMode transfer) {
  for (const int published : descriptor.kv().transfer_modes()) {
    if (published == static_cast<int>(transfer)) {
      return true;
    }
  }
  return false;
}

bool has_remote_pd_mode(const ProviderDescriptor& descriptor) {
  for (const xllm::proto::ExecutionModeSpec& spec :
       descriptor.serving().execution_modes()) {
    if (spec.mode() == xllm::proto::EXECUTION_MODE_REMOTE_PD &&
        spec.transfer_mode() == xllm::proto::TRANSFER_MODE_LAYERWISE_PUSH &&
        spec.selection_order() == xllm::proto::SELECTION_ORDER_P_FIRST &&
        spec.binding_stage() == xllm::proto::BINDING_STAGE_BEFORE_PREFILL &&
        !spec.p_selection_delegated()) {
      return true;
    }
  }
  return false;
}

template <typename Repeated>
std::vector<std::string> sorted_strings(const Repeated& values) {
  std::vector<std::string> result(values.begin(), values.end());
  std::sort(result.begin(), result.end());
  return result;
}

ContractResult validate_contract_version(uint32_t version,
                                         const std::string& field) {
  if (version != kProviderContractVersion) {
    return fail(
        xllm::proto::PROVIDER_CONTRACT_ERROR_UNSUPPORTED_CONTRACT_VERSION,
        field + " must equal " + std::to_string(kProviderContractVersion));
  }
  return ContractResult::success();
}

ContractResult validate_mode_spec_role(EngineRole role, ExecutionMode mode) {
  bool valid = false;
  switch (mode) {
    case xllm::proto::EXECUTION_MODE_AGGREGATED:
      valid = role == xllm::proto::ENGINE_ROLE_AGGREGATED;
      break;
    case xllm::proto::EXECUTION_MODE_REMOTE_PD:
      valid = role == xllm::proto::ENGINE_ROLE_PREFILL ||
              role == xllm::proto::ENGINE_ROLE_DECODE;
      break;
    case xllm::proto::EXECUTION_MODE_LOCAL_PREFILL_DECODE:
      valid = role == xllm::proto::ENGINE_ROLE_DECODE;
      break;
    case xllm::proto::EXECUTION_MODE_PREFILL_ONLY:
      valid = role == xllm::proto::ENGINE_ROLE_PREFILL;
      break;
    default:
      break;
  }
  if (!valid) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_MODE_SPEC,
                "serving role is incompatible with execution mode");
  }
  return ContractResult::success();
}

ContractResult validate_prediction(const xllm::proto::PlanPrediction& value) {
  const auto valid_nonnegative = [](double candidate) {
    return std::isfinite(candidate) && candidate >= 0.0;
  };
  if ((value.has_ttft_upper_bound_ms() &&
       !valid_nonnegative(value.ttft_upper_bound_ms())) ||
      (value.has_tpot_upper_bound_ms() &&
       !valid_nonnegative(value.tpot_upper_bound_ms())) ||
      (value.has_completion_upper_bound_ms() &&
       !valid_nonnegative(value.completion_upper_bound_ms()))) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "plan prediction contains an invalid upper bound");
  }
  if (value.has_uncertainty() &&
      (!std::isfinite(value.uncertainty()) || value.uncertainty() < 0.0 ||
       value.uncertainty() > 1.0)) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "plan prediction uncertainty must be in [0, 1]");
  }
  return ContractResult::success();
}

}  // namespace

ContractResult::ContractResult(ProviderContractError error, std::string message)
    : error_(error), message_(std::move(message)) {}

ContractResult ContractResult::success() {
  return ContractResult(xllm::proto::PROVIDER_CONTRACT_ERROR_OK, "");
}

ContractResult ContractResult::failure(ProviderContractError error,
                                       std::string message) {
  return ContractResult(error, std::move(message));
}

ContractResult resolve_mode_requirements(ExecutionMode mode,
                                         TransferMode transfer,
                                         ModeRequirements* requirements) {
  if (requirements == nullptr) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD,
                "requirements output must not be null");
  }
  requirements->required_capabilities.clear();

  if (!xllm::proto::ExecutionMode_IsValid(static_cast<int>(mode)) ||
      mode == xllm::proto::EXECUTION_MODE_UNSPECIFIED ||
      !xllm::proto::TransferMode_IsValid(static_cast<int>(transfer)) ||
      transfer == xllm::proto::TRANSFER_MODE_UNSPECIFIED) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_UNKNOWN_ENUM_VALUE,
                "mode or transfer contains an unknown or unspecified enum");
  }

  auto& required = requirements->required_capabilities;
  switch (mode) {
    case xllm::proto::EXECUTION_MODE_AGGREGATED:
      if (transfer != xllm::proto::TRANSFER_MODE_NONE) {
        return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_MODE_SPEC,
                    "aggregated mode requires transfer NONE");
      }
      required = {xllm::proto::PROVIDER_CAPABILITY_AGGREGATED};
      break;
    case xllm::proto::EXECUTION_MODE_REMOTE_PD:
      if (transfer == xllm::proto::TRANSFER_MODE_LAYERWISE_PUSH) {
        required = {xllm::proto::PROVIDER_CAPABILITY_REMOTE_PD_LAYERWISE_PUSH,
                    xllm::proto::PROVIDER_CAPABILITY_NATIVE_RESERVATION};
      } else if (transfer == xllm::proto::TRANSFER_MODE_PULL) {
        required = {xllm::proto::PROVIDER_CAPABILITY_REMOTE_PD_PULL,
                    xllm::proto::PROVIDER_CAPABILITY_NATIVE_RESERVATION};
      } else {
        return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_MODE_SPEC,
                    "remote PD requires PULL or LAYERWISE_PUSH");
      }
      break;
    case xllm::proto::EXECUTION_MODE_LOCAL_PREFILL_DECODE:
      if (transfer != xllm::proto::TRANSFER_MODE_NONE) {
        return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_MODE_SPEC,
                    "local prefill/decode requires transfer NONE");
      }
      required = {
          xllm::proto::PROVIDER_CAPABILITY_LOCAL_PREFILL_DECODE,
          xllm::proto::PROVIDER_CAPABILITY_MIXED_PREFILL_DECODE_ACCOUNTING,
          xllm::proto::PROVIDER_CAPABILITY_STRUCTURED_ADMISSION};
      break;
    case xllm::proto::EXECUTION_MODE_PREFILL_ONLY:
      if (transfer != xllm::proto::TRANSFER_MODE_NONE) {
        return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_MODE_SPEC,
                    "prefill-only mode requires transfer NONE");
      }
      required = {xllm::proto::PROVIDER_CAPABILITY_PREFILL_ONLY,
                  xllm::proto::PROVIDER_CAPABILITY_STRUCTURED_ADMISSION};
      break;
    default:
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_UNKNOWN_ENUM_VALUE,
                  "execution mode is not classified");
  }

  required.insert(required.end(),
                  {xllm::proto::PROVIDER_CAPABILITY_ATTEMPT_QUERY,
                   xllm::proto::PROVIDER_CAPABILITY_CANCEL_FENCE,
                   xllm::proto::PROVIDER_CAPABILITY_ENGINE_LOCAL_DEADLINE,
                   xllm::proto::PROVIDER_CAPABILITY_SELF_FENCING,
                   xllm::proto::PROVIDER_CAPABILITY_DRAIN});
  return ContractResult::success();
}

ContractResult validate_v2_open_mode(ProviderId provider_id,
                                     ExecutionMode mode,
                                     TransferMode transfer,
                                     SelectionOrder order,
                                     BindingStage binding_stage,
                                     bool p_selection_delegated) {
  if (!xllm::proto::ProviderId_IsValid(static_cast<int>(provider_id)) ||
      provider_id == xllm::proto::PROVIDER_ID_UNSPECIFIED ||
      !xllm::proto::SelectionOrder_IsValid(static_cast<int>(order)) ||
      order == xllm::proto::SELECTION_ORDER_UNSPECIFIED ||
      !xllm::proto::BindingStage_IsValid(static_cast<int>(binding_stage)) ||
      binding_stage == xllm::proto::BINDING_STAGE_UNSPECIFIED) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_UNKNOWN_ENUM_VALUE,
                "provider mode binding contains an unknown enum");
  }
  ModeRequirements ignored;
  ContractResult resolved = resolve_mode_requirements(mode, transfer, &ignored);
  if (!resolved.ok()) {
    return resolved;
  }

  if (provider_id == xllm::proto::PROVIDER_ID_XLLM_NATIVE) {
    const bool remote =
        mode == xllm::proto::EXECUTION_MODE_REMOTE_PD &&
        transfer == xllm::proto::TRANSFER_MODE_LAYERWISE_PUSH &&
        order == xllm::proto::SELECTION_ORDER_P_FIRST &&
        binding_stage == xllm::proto::BINDING_STAGE_BEFORE_PREFILL &&
        !p_selection_delegated;
    const bool local =
        mode == xllm::proto::EXECUTION_MODE_LOCAL_PREFILL_DECODE &&
        transfer == xllm::proto::TRANSFER_MODE_NONE &&
        order == xllm::proto::SELECTION_ORDER_D_ONLY &&
        binding_stage == xllm::proto::BINDING_STAGE_AT_SUBMIT &&
        !p_selection_delegated;
    const bool prefill_only =
        mode == xllm::proto::EXECUTION_MODE_PREFILL_ONLY &&
        transfer == xllm::proto::TRANSFER_MODE_NONE &&
        order == xllm::proto::SELECTION_ORDER_P_ONLY &&
        binding_stage == xllm::proto::BINDING_STAGE_AT_SUBMIT &&
        !p_selection_delegated;
    if (remote || local || prefill_only) {
      return ContractResult::success();
    }
  } else if (provider_id == xllm::proto::PROVIDER_ID_VLLM_ASCEND) {
    const bool aggregated =
        mode == xllm::proto::EXECUTION_MODE_AGGREGATED &&
        transfer == xllm::proto::TRANSFER_MODE_NONE &&
        order == xllm::proto::SELECTION_ORDER_SINGLE &&
        binding_stage == xllm::proto::BINDING_STAGE_AT_SUBMIT &&
        !p_selection_delegated;
    if (aggregated) {
      return ContractResult::success();
    }
    const bool closed_d_first =
        mode == xllm::proto::EXECUTION_MODE_REMOTE_PD &&
        transfer == xllm::proto::TRANSFER_MODE_LAYERWISE_PUSH &&
        order == xllm::proto::SELECTION_ORDER_D_FIRST &&
        binding_stage == xllm::proto::BINDING_STAGE_BEFORE_PREFILL &&
        p_selection_delegated;
    if (closed_d_first) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_MODE_NOT_OPEN,
                  "vLLM-Ascend D_FIRST is represented but closed in V2");
    }
  }
  return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_MODE_NOT_OPEN,
              "provider mode binding is not open in V2");
}

ContractResult validate_provider_descriptor(
    const ProviderDescriptor& descriptor) {
  ContractResult version = validate_contract_version(
      descriptor.contract_version(), "descriptor.contract_version");
  if (!version.ok()) {
    return version;
  }
  if (!descriptor.has_identity() || !descriptor.has_endpoint() ||
      !descriptor.has_serving() || !descriptor.has_model() ||
      !descriptor.has_topology() || !descriptor.has_kv() ||
      !descriptor.has_scheduler()) {
    return missing("descriptor section");
  }

  const auto& identity = descriptor.identity();
  if (identity.engine_uid().empty() || identity.incarnation_id().empty() ||
      identity.runtime_family().empty() || identity.runtime_version().empty() ||
      identity.plugin_version().empty() ||
      identity.hardware_runtime_version().empty()) {
    return missing("descriptor.identity field");
  }
  if (!xllm::proto::ProviderId_IsValid(identity.provider_id()) ||
      identity.provider_id() == xllm::proto::PROVIDER_ID_UNSPECIFIED) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_UNKNOWN_ENUM_VALUE,
                "descriptor provider_id is unknown");
  }
  if (identity.protocol_version() != kProviderContractVersion) {
    return fail(
        xllm::proto::PROVIDER_CONTRACT_ERROR_UNSUPPORTED_CONTRACT_VERSION,
        "identity.protocol_version is unsupported");
  }
  std::string expected_runtime;
  switch (identity.provider_id()) {
    case xllm::proto::PROVIDER_ID_XLLM_NATIVE:
      expected_runtime = "xllm";
      break;
    case xllm::proto::PROVIDER_ID_VLLM_ASCEND:
      expected_runtime = "vllm";
      break;
    default:
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_UNKNOWN_ENUM_VALUE,
                  "descriptor provider_id is not classified");
  }
  if (identity.runtime_family() != expected_runtime) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
                "runtime_family does not match provider_id");
  }

  const auto& endpoint = descriptor.endpoint();
  if (endpoint.control_transport().empty() ||
      endpoint.data_transport().empty() || endpoint.address().empty()) {
    return missing("descriptor.endpoint field");
  }
  if (!xllm::proto::EngineRole_IsValid(descriptor.serving().role()) ||
      descriptor.serving().role() == xllm::proto::ENGINE_ROLE_UNSPECIFIED) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_UNKNOWN_ENUM_VALUE,
                "descriptor serving role is unknown");
  }
  if (descriptor.serving().execution_modes().empty()) {
    return missing("descriptor.serving.execution_modes");
  }
  ContractResult features = validate_unique_strings(
      descriptor.serving().api_features(), "serving.api_features", false);
  if (!features.ok()) {
    return features;
  }

  ContractResult capabilities =
      validate_unique_enum_list(descriptor.capabilities(),
                                xllm::proto::ProviderCapability_IsValid,
                                "descriptor.capabilities");
  if (!capabilities.ok()) {
    return capabilities;
  }
  if (descriptor.capabilities().empty()) {
    return missing("descriptor.capabilities");
  }

  std::unordered_set<uint64_t> mode_keys;
  for (const auto& spec : descriptor.serving().execution_modes()) {
    const uint64_t key = (static_cast<uint64_t>(spec.mode()) << 32) |
                         static_cast<uint32_t>(spec.transfer_mode());
    if (!mode_keys.insert(key).second) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DUPLICATE_ENTRY,
                  "descriptor contains a duplicate execution mode");
    }
    ModeRequirements requirements;
    ContractResult resolved = resolve_mode_requirements(
        spec.mode(), spec.transfer_mode(), &requirements);
    if (!resolved.ok()) {
      return resolved;
    }
    ContractResult role =
        validate_mode_spec_role(descriptor.serving().role(), spec.mode());
    if (!role.ok()) {
      return role;
    }
    ContractResult open = validate_v2_open_mode(identity.provider_id(),
                                                spec.mode(),
                                                spec.transfer_mode(),
                                                spec.selection_order(),
                                                spec.binding_stage(),
                                                spec.p_selection_delegated());
    if (!open.ok()) {
      return open;
    }
    for (const ProviderCapability required :
         requirements.required_capabilities) {
      if (!has_capability(descriptor, required)) {
        return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_CAPABILITY,
                    "descriptor is missing a required mode capability");
      }
    }
    if (!has_transfer_mode(descriptor, spec.transfer_mode())) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
                  "mode transfer is absent from KV descriptor");
    }
  }

  const auto& model = descriptor.model();
  if (model.model_revision().empty() || model.tokenizer_revision().empty() ||
      model.chat_template_digest().empty() || model.quantization().empty() ||
      model.renderer_digest().empty()) {
    return missing("descriptor.model field");
  }
  const auto& topology = descriptor.topology();
  if (topology.soc().empty() || topology.device_count() == 0 ||
      topology.tp() == 0 || topology.dp() == 0 || topology.pp() == 0 ||
      topology.ep() == 0 || topology.cp() == 0) {
    return missing("descriptor.topology field");
  }
  const auto& kv = descriptor.kv();
  if (kv.kv_layout_digest().empty() || kv.cache_dtype().empty() ||
      kv.block_size() == 0 || kv.head_shard_mapping_digest().empty() ||
      kv.connector().empty() || kv.connector_version().empty()) {
    return missing("descriptor.kv field");
  }
  ContractResult cache_groups =
      validate_unique_strings(kv.cache_groups(), "kv.cache_groups", false);
  if (!cache_groups.ok()) {
    return cache_groups;
  }
  ContractResult transfers =
      validate_unique_enum_list(kv.transfer_modes(),
                                xllm::proto::TransferMode_IsValid,
                                "kv.transfer_modes");
  if (!transfers.ok()) {
    return transfers;
  }
  if (kv.transfer_modes().empty()) {
    return missing("kv.transfer_modes");
  }
  if (kv.has_storage() &&
      (kv.storage().storage_kv_layout_digest().empty() ||
       kv.storage().store_serialization_version().empty())) {
    return missing("kv.storage field");
  }
  const auto& scheduler = descriptor.scheduler();
  if (scheduler.scheduler_class().empty() || scheduler.max_num_seqs() == 0 ||
      scheduler.max_num_batched_tokens() == 0 ||
      scheduler.scheduler_policy_digest().empty()) {
    return missing("descriptor.scheduler field");
  }
  if (descriptor.profile_digest().empty()) {
    return missing("descriptor.profile_digest");
  }
  return ContractResult::success();
}

ContractResult validate_remote_pd_compatibility(
    const ProviderDescriptor& prefill,
    const ProviderDescriptor& decode,
    std::string* compatibility_proof) {
  if (compatibility_proof == nullptr) {
    return missing("remote P/D compatibility proof output");
  }
  compatibility_proof->clear();

  ContractResult prefill_validation = validate_provider_descriptor(prefill);
  if (!prefill_validation.ok()) {
    return prefill_validation;
  }
  ContractResult decode_validation = validate_provider_descriptor(decode);
  if (!decode_validation.ok()) {
    return decode_validation;
  }
  const xllm::proto::ProviderIdentity& prefill_identity = prefill.identity();
  const xllm::proto::ProviderIdentity& decode_identity = decode.identity();
  const bool identity_compatible =
      prefill_identity.provider_id() == xllm::proto::PROVIDER_ID_XLLM_NATIVE &&
      decode_identity.provider_id() == prefill_identity.provider_id() &&
      prefill_identity.runtime_family() == decode_identity.runtime_family() &&
      prefill_identity.runtime_version() == decode_identity.runtime_version() &&
      prefill_identity.plugin_version() == decode_identity.plugin_version() &&
      prefill_identity.hardware_runtime_version() ==
          decode_identity.hardware_runtime_version() &&
      prefill_identity.protocol_version() == decode_identity.protocol_version();
  if (!identity_compatible ||
      prefill.serving().role() != xllm::proto::ENGINE_ROLE_PREFILL ||
      decode.serving().role() != xllm::proto::ENGINE_ROLE_DECODE ||
      !has_remote_pd_mode(prefill) || !has_remote_pd_mode(decode)) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
                "remote P/D Provider identity, role, or mode is incompatible");
  }

  const xllm::proto::ModelDescriptor& prefill_model = prefill.model();
  const xllm::proto::ModelDescriptor& decode_model = decode.model();
  if (prefill_model.model_revision() != decode_model.model_revision() ||
      prefill_model.quantization() != decode_model.quantization()) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
                "remote P/D model revision or quantization is incompatible");
  }

  const xllm::proto::KVDescriptor& prefill_kv = prefill.kv();
  const xllm::proto::KVDescriptor& decode_kv = decode.kv();
  const bool kv_compatible =
      prefill_kv.kv_layout_digest() == decode_kv.kv_layout_digest() &&
      prefill_kv.cache_dtype() == decode_kv.cache_dtype() &&
      prefill_kv.block_size() == decode_kv.block_size() &&
      sorted_strings(prefill_kv.cache_groups()) ==
          sorted_strings(decode_kv.cache_groups()) &&
      prefill_kv.head_shard_mapping_digest() ==
          decode_kv.head_shard_mapping_digest() &&
      prefill_kv.connector() == decode_kv.connector() &&
      prefill_kv.connector_version() == decode_kv.connector_version() &&
      has_transfer_mode(prefill, xllm::proto::TRANSFER_MODE_LAYERWISE_PUSH) &&
      has_transfer_mode(decode, xllm::proto::TRANSFER_MODE_LAYERWISE_PUSH);
  if (!kv_compatible) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
                "remote P/D KV or Connector contract is incompatible");
  }

  const xllm::proto::TopologyDescriptor& prefill_topology = prefill.topology();
  const xllm::proto::TopologyDescriptor& decode_topology = decode.topology();
  const bool topology_compatible =
      prefill_topology.soc() == decode_topology.soc() &&
      prefill_topology.device_count() == decode_topology.device_count() &&
      prefill_topology.tp() == decode_topology.tp() &&
      prefill_topology.dp() == decode_topology.dp() &&
      prefill_topology.pp() == decode_topology.pp() &&
      prefill_topology.ep() == decode_topology.ep() &&
      prefill_topology.cp() == decode_topology.cp();
  if (!topology_compatible) {
    return fail(
        xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
        "remote P/D topology requires an unavailable topology transform");
  }

  *compatibility_proof = "remote-pd-v1|p=" + prefill.profile_digest() +
                         "|d=" + decode.profile_digest() +
                         "|model=" + prefill_model.model_revision() +
                         "|kv=" + prefill_kv.kv_layout_digest() +
                         "|connector=" + prefill_kv.connector() + "/" +
                         prefill_kv.connector_version() +
                         "|transfer=layerwise-push";
  return ContractResult::success();
}

ContractResult validate_canonical_request(
    const xllm::proto::CanonicalRequest& request) {
  ContractResult version = validate_contract_version(
      request.contract_version(), "request.contract_version");
  if (!version.ok()) {
    return version;
  }
  if (request.global_request_id().empty() || request.trace_id().empty() ||
      request.request_uid().empty() || !request.has_attempt_seq() ||
      request.model_revision().empty() ||
      request.canonical_payload_schema().empty() ||
      request.canonical_payload().empty()) {
    return missing("canonical request identity or payload");
  }
  if (!xllm::proto::ApiKind_IsValid(request.api_kind()) ||
      request.api_kind() == xllm::proto::API_KIND_UNSPECIFIED) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_UNKNOWN_ENUM_VALUE,
                "canonical request api_kind is unknown");
  }
  if (request.effective_max_new_tokens() == 0 || request.n() == 0 ||
      request.best_of() < request.n() || request.remaining_deadline_ms() == 0) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "canonical request limits are invalid");
  }
  ContractResult capabilities =
      validate_unique_enum_list(request.required_capabilities(),
                                xllm::proto::ProviderCapability_IsValid,
                                "request.required_capabilities");
  if (!capabilities.ok()) {
    return capabilities;
  }
  return validate_unique_strings(
      request.required_api_features(), "request.required_api_features", true);
}

ContractResult validate_encoded_request(
    const ProviderDescriptor& descriptor,
    const xllm::proto::CanonicalRequest& canonical,
    const xllm::proto::EncodedRequest& request) {
  ContractResult canonical_validation = validate_canonical_request(canonical);
  if (!canonical_validation.ok()) {
    return canonical_validation;
  }
  if (canonical.model_revision() != descriptor.model().model_revision()) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
                "canonical request model does not match descriptor");
  }
  for (const int required : canonical.required_capabilities()) {
    if (!has_capability(descriptor,
                        static_cast<ProviderCapability>(required))) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_CAPABILITY,
                  "canonical request requires an unpublished capability");
    }
  }
  for (const std::string& required : canonical.required_api_features()) {
    bool found = false;
    for (const std::string& published : descriptor.serving().api_features()) {
      found = found || required == published;
    }
    if (!found) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
                  "canonical request requires an unpublished API feature");
    }
  }
  if (request.provider_id() != descriptor.identity().provider_id()) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
                "encoded request provider does not match descriptor");
  }
  if (!xllm::proto::TokenCountQuality_IsValid(request.token_count_quality()) ||
      request.token_count_quality() ==
          xllm::proto::TOKEN_COUNT_QUALITY_UNSPECIFIED) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_UNKNOWN_ENUM_VALUE,
                "encoded request token count quality is unknown");
  }
  if (request.renderer_digest().empty() || request.provider_payload().empty()) {
    return missing("encoded request renderer or payload");
  }
  if (canonical.strict() &&
      request.renderer_digest() != descriptor.model().renderer_digest()) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
                "strict request renderer does not match descriptor");
  }
  if (request.token_count_quality() == xllm::proto::TOKEN_COUNT_QUALITY_EXACT &&
      request.prompt_tokens_upper_bound() != request.prompt_tokens()) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "exact token count must equal its upper bound");
  }
  if (request.token_count_quality() ==
          xllm::proto::TOKEN_COUNT_QUALITY_BOUNDED &&
      request.prompt_tokens_upper_bound() < request.prompt_tokens()) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "bounded token upper bound is below the estimate");
  }
  if (request.token_count_quality() ==
          xllm::proto::TOKEN_COUNT_QUALITY_UNKNOWN &&
      (request.prompt_tokens() != 0 ||
       request.prompt_tokens_upper_bound() != 0)) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "unknown token counts must not contain numeric estimates");
  }
  return ContractResult::success();
}

ContractResult validate_engine_state(const ProviderDescriptor& descriptor,
                                     const xllm::proto::EngineState& state) {
  if (state.engine_uid() != descriptor.identity().engine_uid() ||
      state.incarnation_id() != descriptor.identity().incarnation_id()) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
                "engine state identity does not match descriptor");
  }
  if (state.state_seq() == 0 || state.observed_at_unix_ms() == 0) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "engine state sequence and timestamp must be nonzero");
  }
  if (!xllm::proto::EngineLifecycle_IsValid(state.lifecycle()) ||
      state.lifecycle() == xllm::proto::ENGINE_LIFECYCLE_UNSPECIFIED ||
      !xllm::proto::EngineOwnership_IsValid(state.ownership()) ||
      state.ownership() == xllm::proto::ENGINE_OWNERSHIP_UNSPECIFIED ||
      !xllm::proto::HealthStatus_IsValid(state.shallow_health()) ||
      !xllm::proto::HealthStatus_IsValid(state.deep_health()) ||
      !xllm::proto::StateQuality_IsValid(state.state_quality()) ||
      state.state_quality() == xllm::proto::STATE_QUALITY_UNSPECIFIED) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_UNKNOWN_ENUM_VALUE,
                "engine state contains an unknown enum");
  }

  const bool per_dp_capability =
      has_capability(descriptor, xllm::proto::PROVIDER_CAPABILITY_PER_DP_STATE);
  if (!per_dp_capability && !state.per_dp().empty()) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_CAPABILITY,
                "per-DP state was published without capability");
  }
  if (per_dp_capability &&
      state.state_quality() == xllm::proto::STATE_QUALITY_FULL &&
      state.per_dp_size() != static_cast<int>(descriptor.topology().dp())) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "full state must contain every DP rank");
  }
  std::unordered_set<uint32_t> dp_ranks;
  for (const auto& dp : state.per_dp()) {
    if (dp.dp_rank() >= descriptor.topology().dp() ||
        !dp_ranks.insert(dp.dp_rank()).second) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                  "per-DP state has a duplicate or out-of-range rank");
    }
    if (dp.has_kv_used_ratio() &&
        (!std::isfinite(dp.kv_used_ratio()) || dp.kv_used_ratio() < 0.0 ||
         dp.kv_used_ratio() > 1.0)) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                  "KV used ratio must be in [0, 1]");
    }
  }
  if (!has_capability(descriptor,
                      xllm::proto::PROVIDER_CAPABILITY_DEEP_HEALTH) &&
      state.deep_health() != xllm::proto::HEALTH_STATUS_UNKNOWN) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_CAPABILITY,
                "deep health was published without capability");
  }
  if (state.has_throughput_tokens_per_second() &&
      (!std::isfinite(state.throughput_tokens_per_second()) ||
       state.throughput_tokens_per_second() < 0.0)) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "throughput must be finite and nonnegative");
  }
  double previous_upper_bound = 0.0;
  for (const auto& bucket : state.latency_histogram_delta()) {
    if (!std::isfinite(bucket.upper_bound()) ||
        bucket.upper_bound() <= previous_upper_bound) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                  "histogram upper bounds must be strictly increasing");
    }
    previous_upper_bound = bucket.upper_bound();
  }
  std::unordered_set<std::string> failure_reasons;
  for (const auto& counter : state.failure_counters()) {
    if (counter.reason().empty()) {
      return missing("engine state failure reason");
    }
    if (!failure_reasons.insert(counter.reason()).second) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DUPLICATE_ENTRY,
                  "engine state contains a duplicate failure reason");
    }
  }
  return ContractResult::success();
}

ContractResult validate_execution_plan(const ProviderDescriptor& descriptor,
                                       const xllm::proto::ExecutionPlan& plan) {
  ContractResult version = validate_contract_version(plan.contract_version(),
                                                     "plan.contract_version");
  if (!version.ok()) {
    return version;
  }
  if (plan.request_uid().empty() || !plan.has_attempt_seq() ||
      plan.compatibility_proof().empty() || plan.provider_payload().empty()) {
    return missing("execution plan identity, proof, or payload");
  }
  if (plan.provider_id() != descriptor.identity().provider_id()) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
                "execution plan provider does not match descriptor");
  }

  ModeRequirements requirements;
  ContractResult resolved = resolve_mode_requirements(
      plan.mode(), plan.transfer_mode(), &requirements);
  if (!resolved.ok()) {
    return resolved;
  }
  ContractResult open = validate_v2_open_mode(plan.provider_id(),
                                              plan.mode(),
                                              plan.transfer_mode(),
                                              plan.selection_order(),
                                              plan.binding_stage(),
                                              plan.p_selection_delegated());
  if (!open.ok()) {
    return open;
  }

  bool descriptor_mode_found = false;
  for (const auto& spec : descriptor.serving().execution_modes()) {
    if (spec.mode() == plan.mode() &&
        spec.transfer_mode() == plan.transfer_mode() &&
        spec.selection_order() == plan.selection_order() &&
        spec.binding_stage() == plan.binding_stage() &&
        spec.p_selection_delegated() == plan.p_selection_delegated()) {
      descriptor_mode_found = true;
      break;
    }
  }
  if (!descriptor_mode_found) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
                "execution plan mode is absent from descriptor");
  }
  ContractResult plan_capabilities =
      validate_unique_enum_list(plan.required_capabilities(),
                                xllm::proto::ProviderCapability_IsValid,
                                "plan.required_capabilities");
  if (!plan_capabilities.ok()) {
    return plan_capabilities;
  }
  for (const ProviderCapability required : requirements.required_capabilities) {
    bool found = false;
    for (const int planned : plan.required_capabilities()) {
      found = found || planned == static_cast<int>(required);
    }
    if (!found || !has_capability(descriptor, required)) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_CAPABILITY,
                  "execution plan omits a required capability");
    }
  }
  for (const int planned : plan.required_capabilities()) {
    if (!has_capability(descriptor, static_cast<ProviderCapability>(planned))) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_CAPABILITY,
                  "execution plan requires an unpublished capability");
    }
  }

  if (plan.selected_roles().empty()) {
    return missing("plan.selected_roles");
  }
  std::unordered_set<std::string> selected_engines;
  for (int i = 0; i < plan.selected_roles_size(); ++i) {
    const auto& selected = plan.selected_roles(i);
    if (!xllm::proto::EngineRole_IsValid(selected.role()) ||
        selected.role() == xllm::proto::ENGINE_ROLE_UNSPECIFIED ||
        selected.engine_uid().empty() || selected.incarnation_id().empty() ||
        selected.order_index() != static_cast<uint32_t>(i)) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_SELECTED_ROLES,
                  "selected role is invalid or not in canonical order");
    }
    if (!selected_engines
             .insert(selected.engine_uid() + "\n" + selected.incarnation_id())
             .second) {
      return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_DUPLICATE_ENTRY,
                  "execution plan selects an engine twice");
    }
  }

  bool role_shape_valid = false;
  if (plan.selection_order() == xllm::proto::SELECTION_ORDER_SINGLE) {
    role_shape_valid =
        plan.selected_roles_size() == 1 &&
        plan.selected_roles(0).role() == xllm::proto::ENGINE_ROLE_AGGREGATED;
  } else if (plan.selection_order() == xllm::proto::SELECTION_ORDER_P_FIRST) {
    role_shape_valid =
        plan.selected_roles_size() >= 2 &&
        plan.selected_roles(0).role() == xllm::proto::ENGINE_ROLE_PREFILL;
    for (int i = 1; role_shape_valid && i < plan.selected_roles_size(); ++i) {
      role_shape_valid =
          plan.selected_roles(i).role() == xllm::proto::ENGINE_ROLE_DECODE;
    }
  } else if (plan.selection_order() == xllm::proto::SELECTION_ORDER_D_ONLY) {
    role_shape_valid = true;
    for (const auto& selected : plan.selected_roles()) {
      role_shape_valid = role_shape_valid &&
                         selected.role() == xllm::proto::ENGINE_ROLE_DECODE;
    }
  } else if (plan.selection_order() == xllm::proto::SELECTION_ORDER_P_ONLY) {
    role_shape_valid =
        plan.selected_roles_size() == 1 &&
        plan.selected_roles(0).role() == xllm::proto::ENGINE_ROLE_PREFILL;
  }
  if (!role_shape_valid) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_SELECTED_ROLES,
                "selected roles do not match selection order");
  }

  const auto& budget = plan.deadline_budget();
  const bool budget_overflow =
      budget.remaining_ms() == 0 ||
      budget.submit_ms() > budget.remaining_ms() ||
      budget.handoff_ms() > budget.remaining_ms() - budget.submit_ms() ||
      budget.output_ms() >
          budget.remaining_ms() - budget.submit_ms() - budget.handoff_ms();
  if (budget_overflow) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "execution plan deadline budget is invalid");
  }
  if (!std::isfinite(plan.score())) {
    return fail(xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
                "execution plan score must be finite");
  }
  return validate_prediction(plan.prediction());
}

}  // namespace xllm_service::provider
