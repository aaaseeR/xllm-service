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

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "provider/canonical_request_builder.h"
#include "provider/execution_plan_builder.h"
#include "provider/provider_registry.h"
#include "provider/provider_route_selector.h"

namespace xllm_service::provider {
namespace {

using xllm::proto::BindingStage;
using xllm::proto::EngineRole;
using xllm::proto::ExecutionMode;
using xllm::proto::ExecutionPlan;
using xllm::proto::ProviderCapability;
using xllm::proto::ProviderDescriptor;
using xllm::proto::ProviderId;
using xllm::proto::SelectionOrder;
using xllm::proto::TransferMode;

struct ModeCase {
  ProviderId provider_id;
  ExecutionMode mode;
  TransferMode transfer;
  SelectionOrder order;
  BindingStage binding;
  EngineRole role;
};

const std::vector<ModeCase> kOpenModeCases = {
    {xllm::proto::PROVIDER_ID_XLLM_NATIVE,
     xllm::proto::EXECUTION_MODE_REMOTE_PD,
     xllm::proto::TRANSFER_MODE_LAYERWISE_PUSH,
     xllm::proto::SELECTION_ORDER_P_FIRST,
     xllm::proto::BINDING_STAGE_BEFORE_PREFILL,
     xllm::proto::ENGINE_ROLE_PREFILL},
    {xllm::proto::PROVIDER_ID_XLLM_NATIVE,
     xllm::proto::EXECUTION_MODE_LOCAL_PREFILL_DECODE,
     xllm::proto::TRANSFER_MODE_NONE,
     xllm::proto::SELECTION_ORDER_D_ONLY,
     xllm::proto::BINDING_STAGE_AT_SUBMIT,
     xllm::proto::ENGINE_ROLE_DECODE},
    {xllm::proto::PROVIDER_ID_XLLM_NATIVE,
     xllm::proto::EXECUTION_MODE_PREFILL_ONLY,
     xllm::proto::TRANSFER_MODE_NONE,
     xllm::proto::SELECTION_ORDER_P_ONLY,
     xllm::proto::BINDING_STAGE_AT_SUBMIT,
     xllm::proto::ENGINE_ROLE_PREFILL},
    {xllm::proto::PROVIDER_ID_VLLM_ASCEND,
     xllm::proto::EXECUTION_MODE_AGGREGATED,
     xllm::proto::TRANSFER_MODE_NONE,
     xllm::proto::SELECTION_ORDER_SINGLE,
     xllm::proto::BINDING_STAGE_AT_SUBMIT,
     xllm::proto::ENGINE_ROLE_AGGREGATED},
};

ProviderDescriptor make_descriptor(const ModeCase& mode_case) {
  ProviderDescriptor descriptor;
  descriptor.set_contract_version(kProviderContractVersion);
  auto* identity = descriptor.mutable_identity();
  identity->set_engine_uid("engine-1");
  identity->set_incarnation_id("incarnation-1");
  identity->set_provider_id(mode_case.provider_id);
  identity->set_runtime_family(
      mode_case.provider_id == xllm::proto::PROVIDER_ID_XLLM_NATIVE ? "xllm"
                                                                    : "vllm");
  identity->set_runtime_version("runtime-v1");
  identity->set_plugin_version("plugin-v1");
  identity->set_hardware_runtime_version("hardware-v1");
  identity->set_protocol_version(kProviderContractVersion);

  descriptor.mutable_endpoint()->set_control_transport("grpc");
  descriptor.mutable_endpoint()->set_data_transport("grpc");
  descriptor.mutable_endpoint()->set_address("127.0.0.1:8000");
  descriptor.mutable_serving()->set_role(mode_case.role);
  auto* mode = descriptor.mutable_serving()->add_execution_modes();
  mode->set_mode(mode_case.mode);
  mode->set_transfer_mode(mode_case.transfer);
  mode->set_selection_order(mode_case.order);
  mode->set_binding_stage(mode_case.binding);
  mode->set_p_selection_delegated(false);
  descriptor.mutable_serving()->add_api_features("chat");

  descriptor.mutable_model()->set_model_revision("model-r1");
  descriptor.mutable_model()->set_tokenizer_revision("tokenizer-r1");
  descriptor.mutable_model()->set_chat_template_digest("template-sha256");
  descriptor.mutable_model()->set_quantization("none");
  descriptor.mutable_model()->set_renderer_digest("renderer-sha256");

  auto* topology = descriptor.mutable_topology();
  topology->set_soc("cpu-test");
  topology->set_device_count(1);
  topology->set_tp(1);
  topology->set_dp(2);
  topology->set_pp(1);
  topology->set_ep(1);
  topology->set_cp(1);

  auto* kv = descriptor.mutable_kv();
  kv->set_kv_layout_digest("kv-sha256");
  kv->set_cache_dtype("bf16");
  kv->set_block_size(16);
  kv->set_kv_namespace("kv-namespace-sha256");
  kv->set_hash_version(1);
  kv->set_hash_seed(1024);
  kv->add_cache_groups("full-attention");
  kv->set_head_shard_mapping_digest("head-shard-sha256");
  kv->set_connector("native");
  kv->set_connector_version("1");
  kv->add_transfer_modes(mode_case.transfer);

  auto* scheduler = descriptor.mutable_scheduler();
  scheduler->set_scheduler_class("default");
  scheduler->set_max_num_seqs(64);
  scheduler->set_max_num_batched_tokens(8192);
  scheduler->set_scheduler_policy_digest("scheduler-sha256");

  ModeRequirements requirements;
  const ContractResult resolved = resolve_mode_requirements(
      mode_case.mode, mode_case.transfer, &requirements);
  if (!resolved.ok()) {
    return descriptor;
  }
  for (const ProviderCapability capability :
       requirements.required_capabilities) {
    descriptor.add_capabilities(capability);
  }
  descriptor.set_profile_digest("profile-sha256");
  return descriptor;
}

std::pair<ProviderDescriptor, ProviderDescriptor> make_remote_pd_descriptors() {
  ProviderDescriptor prefill = make_descriptor(kOpenModeCases[0]);
  prefill.mutable_identity()->set_engine_uid("engine-p");
  prefill.mutable_identity()->set_incarnation_id("incarnation-p");
  prefill.set_profile_digest("profile-p");

  ProviderDescriptor decode = prefill;
  decode.mutable_identity()->set_engine_uid("engine-d");
  decode.mutable_identity()->set_incarnation_id("incarnation-d");
  decode.mutable_serving()->set_role(xllm::proto::ENGINE_ROLE_DECODE);
  decode.set_profile_digest("profile-d");
  return std::make_pair(std::move(prefill), std::move(decode));
}

xllm::proto::CanonicalRequest make_request() {
  xllm::proto::CanonicalRequest request;
  request.set_contract_version(kProviderContractVersion);
  request.set_global_request_id("global-1");
  request.set_trace_id("trace-1");
  request.set_request_uid("request-1");
  request.set_attempt_seq(0);
  request.set_api_kind(xllm::proto::API_KIND_CHAT_COMPLETIONS);
  request.set_model_revision("model-r1");
  request.set_strict(true);
  request.set_canonical_payload_schema("json-v1");
  request.set_canonical_payload("{}");
  request.set_effective_max_new_tokens(128);
  request.set_n(1);
  request.set_best_of(1);
  request.set_remaining_deadline_ms(1000);
  return request;
}

xllm::proto::EncodedRequest make_encoded_request(
    const ProviderDescriptor& descriptor) {
  xllm::proto::EncodedRequest request;
  request.set_provider_id(descriptor.identity().provider_id());
  request.set_token_count_quality(xllm::proto::TOKEN_COUNT_QUALITY_EXACT);
  request.set_prompt_tokens(4);
  request.set_prompt_tokens_upper_bound(4);
  request.set_renderer_digest("renderer-sha256");
  request.set_provider_payload("encoded");
  return request;
}

ExecutionPlan make_plan(const ProviderDescriptor& descriptor) {
  const auto& spec = descriptor.serving().execution_modes(0);
  ExecutionPlan plan;
  plan.set_contract_version(kProviderContractVersion);
  plan.set_request_uid("request-1");
  plan.set_attempt_seq(0);
  plan.set_provider_id(descriptor.identity().provider_id());
  plan.set_mode(spec.mode());
  plan.set_transfer_mode(spec.transfer_mode());
  plan.set_selection_order(spec.selection_order());
  plan.set_binding_stage(spec.binding_stage());
  plan.set_p_selection_delegated(spec.p_selection_delegated());
  plan.set_compatibility_proof("compatibility-sha256");
  plan.set_provider_payload("encoded");
  plan.mutable_deadline_budget()->set_remaining_ms(1000);
  plan.mutable_deadline_budget()->set_submit_ms(100);
  plan.mutable_deadline_budget()->set_handoff_ms(200);
  plan.mutable_deadline_budget()->set_output_ms(600);
  plan.set_score(1.0);

  ModeRequirements requirements;
  EXPECT_TRUE(resolve_mode_requirements(
                  spec.mode(), spec.transfer_mode(), &requirements)
                  .ok());
  for (const ProviderCapability capability :
       requirements.required_capabilities) {
    plan.add_required_capabilities(capability);
  }

  auto add_role = [&plan](EngineRole role, const std::string& suffix) {
    auto* selected = plan.add_selected_roles();
    selected->set_role(role);
    selected->set_engine_uid("engine-" + suffix);
    selected->set_incarnation_id("incarnation-" + suffix);
    selected->set_order_index(plan.selected_roles_size() - 1);
  };
  switch (spec.selection_order()) {
    case xllm::proto::SELECTION_ORDER_SINGLE:
      add_role(xllm::proto::ENGINE_ROLE_AGGREGATED, "a");
      break;
    case xllm::proto::SELECTION_ORDER_P_FIRST:
      add_role(xllm::proto::ENGINE_ROLE_PREFILL, "p");
      add_role(xllm::proto::ENGINE_ROLE_DECODE, "d0");
      add_role(xllm::proto::ENGINE_ROLE_DECODE, "d1");
      break;
    case xllm::proto::SELECTION_ORDER_D_ONLY:
      add_role(xllm::proto::ENGINE_ROLE_DECODE, "d0");
      add_role(xllm::proto::ENGINE_ROLE_DECODE, "d1");
      break;
    case xllm::proto::SELECTION_ORDER_P_ONLY:
      add_role(xllm::proto::ENGINE_ROLE_PREFILL, "p");
      break;
    default:
      break;
  }
  if (!plan.selected_roles().empty()) {
    plan.mutable_selected_roles(0)->set_engine_uid(
        descriptor.identity().engine_uid());
    plan.mutable_selected_roles(0)->set_incarnation_id(
        descriptor.identity().incarnation_id());
  }
  return plan;
}

xllm::proto::EngineState make_state(const ProviderDescriptor& descriptor) {
  xllm::proto::EngineState state;
  state.set_engine_uid(descriptor.identity().engine_uid());
  state.set_incarnation_id(descriptor.identity().incarnation_id());
  state.set_provider_id(descriptor.identity().provider_id());
  state.set_profile_digest(descriptor.profile_digest());
  state.set_model_revision(descriptor.model().model_revision());
  state.set_state_seq(1);
  state.set_observed_at_unix_ms(1000);
  state.set_lifecycle(xllm::proto::ENGINE_LIFECYCLE_READY);
  state.set_ownership(xllm::proto::ENGINE_OWNERSHIP_OWNED);
  state.set_shallow_health(xllm::proto::HEALTH_STATUS_HEALTHY);
  state.set_deep_health(xllm::proto::HEALTH_STATUS_UNKNOWN);
  state.set_state_quality(xllm::proto::STATE_QUALITY_PARTIAL);
  return state;
}

class TestCodec final : public RequestCodec {
 public:
  explicit TestCodec(const ProviderDescriptor& descriptor)
      : descriptor_(descriptor) {}

  ContractResult encode(const xllm::proto::CanonicalRequest& request,
                        const RequestEncodingContext& context,
                        xllm::proto::EncodedRequest* encoded) const override {
    static_cast<void>(context);
    ContractResult validation = validate_canonical_request(request);
    if (!validation.ok()) {
      return validation;
    }
    if (encoded == nullptr) {
      return ContractResult::failure(
          xllm::proto::PROVIDER_CONTRACT_ERROR_ENCODING_FAILED,
          "encoded output must not be null");
    }
    *encoded = make_encoded_request(descriptor_);
    return ContractResult::success();
  }

 private:
  const ProviderDescriptor& descriptor_;
};

class TestAdapter final : public ProviderAdapter {
 public:
  explicit TestAdapter(ProviderDescriptor descriptor)
      : descriptor_(std::move(descriptor)), codec_(descriptor_) {}

  const ProviderDescriptor& describe() const override { return descriptor_; }
  const RequestCodec& request_codec() const override { return codec_; }
  ProviderDispatchKind dispatch_kind() const override {
    return ProviderDispatchKind::XLLM_NATIVE_RPC;
  }

 private:
  ProviderDescriptor descriptor_;
  TestCodec codec_;
};

class TestNativeRenderer final : public XllmNativeRequestRenderer {
 public:
  ContractResult render(const xllm::proto::CanonicalRequest& request,
                        const RequestEncodingContext& context,
                        std::string* provider_payload,
                        uint64_t* prompt_tokens,
                        std::string* renderer_digest) const override {
    static_cast<void>(context);
    if (provider_payload == nullptr || prompt_tokens == nullptr ||
        renderer_digest == nullptr) {
      return ContractResult::failure(
          xllm::proto::PROVIDER_CONTRACT_ERROR_ENCODING_FAILED,
          "test renderer output is null");
    }
    *provider_payload = "native:" + request.canonical_payload();
    *prompt_tokens = 4;
    *renderer_digest = "renderer-sha256";
    return ContractResult::success();
  }
};

TEST(ProviderContractTest, ResolvesEveryCompleteV2CapabilityRow) {
  struct RequirementCase {
    ExecutionMode mode;
    TransferMode transfer;
    size_t count;
    ProviderCapability distinguishing_capability;
  };
  const std::vector<RequirementCase> cases = {
      {xllm::proto::EXECUTION_MODE_AGGREGATED,
       xllm::proto::TRANSFER_MODE_NONE,
       6,
       xllm::proto::PROVIDER_CAPABILITY_AGGREGATED},
      {xllm::proto::EXECUTION_MODE_REMOTE_PD,
       xllm::proto::TRANSFER_MODE_PULL,
       7,
       xllm::proto::PROVIDER_CAPABILITY_REMOTE_PD_PULL},
      {xllm::proto::EXECUTION_MODE_REMOTE_PD,
       xllm::proto::TRANSFER_MODE_LAYERWISE_PUSH,
       7,
       xllm::proto::PROVIDER_CAPABILITY_REMOTE_PD_LAYERWISE_PUSH},
      {xllm::proto::EXECUTION_MODE_LOCAL_PREFILL_DECODE,
       xllm::proto::TRANSFER_MODE_NONE,
       8,
       xllm::proto::PROVIDER_CAPABILITY_MIXED_PREFILL_DECODE_ACCOUNTING},
      {xllm::proto::EXECUTION_MODE_PREFILL_ONLY,
       xllm::proto::TRANSFER_MODE_NONE,
       7,
       xllm::proto::PROVIDER_CAPABILITY_PREFILL_ONLY},
  };
  for (const auto& test_case : cases) {
    ModeRequirements requirements;
    ContractResult result = resolve_mode_requirements(
        test_case.mode, test_case.transfer, &requirements);
    ASSERT_TRUE(result.ok()) << result.message();
    EXPECT_EQ(requirements.required_capabilities.size(), test_case.count);
    EXPECT_NE(std::find(requirements.required_capabilities.begin(),
                        requirements.required_capabilities.end(),
                        test_case.distinguishing_capability),
              requirements.required_capabilities.end());
  }
}

TEST(ProviderContractTest, UnknownAndUnclassifiedModesFailClosed) {
  ModeRequirements requirements;
  ContractResult unknown =
      resolve_mode_requirements(static_cast<ExecutionMode>(99),
                                xllm::proto::TRANSFER_MODE_NONE,
                                &requirements);
  EXPECT_EQ(unknown.error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_UNKNOWN_ENUM_VALUE);

  ContractResult invalid_pair =
      resolve_mode_requirements(xllm::proto::EXECUTION_MODE_AGGREGATED,
                                xllm::proto::TRANSFER_MODE_PULL,
                                &requirements);
  EXPECT_EQ(invalid_pair.error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_MODE_SPEC);
}

TEST(ProviderContractTest, AcceptsEveryOpenProviderModeBinding) {
  for (const auto& test_case : kOpenModeCases) {
    ContractResult result = validate_v2_open_mode(test_case.provider_id,
                                                  test_case.mode,
                                                  test_case.transfer,
                                                  test_case.order,
                                                  test_case.binding,
                                                  false);
    EXPECT_TRUE(result.ok()) << result.message();
    EXPECT_TRUE(validate_provider_descriptor(make_descriptor(test_case)).ok());
  }
}

TEST(ProviderContractTest, VllmDfirstIsRepresentableButNotOpen) {
  ContractResult result =
      validate_v2_open_mode(xllm::proto::PROVIDER_ID_VLLM_ASCEND,
                            xllm::proto::EXECUTION_MODE_REMOTE_PD,
                            xllm::proto::TRANSFER_MODE_LAYERWISE_PUSH,
                            xllm::proto::SELECTION_ORDER_D_FIRST,
                            xllm::proto::BINDING_STAGE_BEFORE_PREFILL,
                            /*p_selection_delegated=*/true);
  EXPECT_EQ(result.error(), xllm::proto::PROVIDER_CONTRACT_ERROR_MODE_NOT_OPEN);
}

TEST(ProviderContractTest, DescriptorRejectsMissingCapability) {
  ProviderDescriptor descriptor = make_descriptor(kOpenModeCases[1]);
  descriptor.mutable_capabilities()->RemoveLast();
  ContractResult result = validate_provider_descriptor(descriptor);
  EXPECT_EQ(result.error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_CAPABILITY);
}

TEST(ProviderContractTest, DescriptorRejectsProviderRuntimeAlias) {
  ProviderDescriptor descriptor = make_descriptor(kOpenModeCases[3]);
  descriptor.mutable_identity()->set_runtime_family("vllm-ascend");
  ContractResult result = validate_provider_descriptor(descriptor);
  EXPECT_EQ(result.error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);
}

TEST(ProviderContractTest, DescriptorRejectsPartialKVHashContract) {
  ProviderDescriptor descriptor = make_descriptor(kOpenModeCases[0]);
  descriptor.mutable_kv()->clear_hash_version();
  EXPECT_EQ(validate_provider_descriptor(descriptor).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);

  descriptor = make_descriptor(kOpenModeCases[0]);
  descriptor.mutable_kv()->clear_kv_namespace();
  EXPECT_EQ(validate_provider_descriptor(descriptor).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);

  descriptor = make_descriptor(kOpenModeCases[0]);
  descriptor.mutable_kv()->clear_hash_seed();
  EXPECT_EQ(validate_provider_descriptor(descriptor).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);
}

TEST(ProviderContractTest, DescriptorRejectsDuplicateCapabilitiesAndModes) {
  ProviderDescriptor descriptor = make_descriptor(kOpenModeCases[1]);
  descriptor.add_capabilities(descriptor.capabilities(0));
  EXPECT_EQ(validate_provider_descriptor(descriptor).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DUPLICATE_ENTRY);

  descriptor = make_descriptor(kOpenModeCases[1]);
  *descriptor.mutable_serving()->add_execution_modes() =
      descriptor.serving().execution_modes(0);
  EXPECT_EQ(validate_provider_descriptor(descriptor).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DUPLICATE_ENTRY);

  descriptor = make_descriptor(kOpenModeCases[1]);
  descriptor.add_capabilities(static_cast<ProviderCapability>(999));
  EXPECT_EQ(validate_provider_descriptor(descriptor).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_UNKNOWN_ENUM_VALUE);
}

TEST(ProviderContractTest, CanonicalAndEncodedRequestsValidate) {
  ProviderDescriptor descriptor = make_descriptor(kOpenModeCases[1]);
  EXPECT_TRUE(validate_canonical_request(make_request()).ok());
  EXPECT_TRUE(validate_encoded_request(
                  descriptor, make_request(), make_encoded_request(descriptor))
                  .ok());

  auto request = make_request();
  request.set_api_kind(static_cast<xllm::proto::ApiKind>(99));
  EXPECT_EQ(validate_canonical_request(request).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_UNKNOWN_ENUM_VALUE);

  request = make_request();
  request.clear_attempt_seq();
  EXPECT_EQ(validate_canonical_request(request).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD);

  auto encoded = make_encoded_request(descriptor);
  encoded.set_prompt_tokens_upper_bound(3);
  EXPECT_EQ(
      validate_encoded_request(descriptor, make_request(), encoded).error(),
      xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE);

  encoded = make_encoded_request(descriptor);
  encoded.set_renderer_digest("other-renderer");
  EXPECT_EQ(
      validate_encoded_request(descriptor, make_request(), encoded).error(),
      xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);
  auto non_strict = make_request();
  non_strict.set_strict(false);
  EXPECT_TRUE(validate_encoded_request(descriptor, non_strict, encoded).ok());

  auto unsupported = make_request();
  unsupported.add_required_capabilities(
      xllm::proto::PROVIDER_CAPABILITY_DEEP_HEALTH);
  EXPECT_EQ(validate_encoded_request(descriptor, unsupported, encoded).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_CAPABILITY);
}

TEST(ProviderContractTest, EngineStateNeedsNoPlaceholderConnectorSignal) {
  ProviderDescriptor descriptor = make_descriptor(kOpenModeCases[1]);
  xllm::proto::EngineState state = make_state(descriptor);
  ContractResult result = validate_engine_state(descriptor, state);
  EXPECT_TRUE(result.ok()) << result.message();
  EXPECT_EQ(state.deep_health(), xllm::proto::HEALTH_STATUS_UNKNOWN);
}

TEST(ProviderContractTest, SharedWireSchemaReservesRetiredV2Placeholders) {
  const google::protobuf::Descriptor* admission =
      xllm::proto::AdmissionResult::descriptor();
  const google::protobuf::Descriptor* state =
      xllm::proto::EngineState::descriptor();
  const google::protobuf::Descriptor* plan =
      xllm::proto::ExecutionPlan::descriptor();
  ASSERT_NE(admission, nullptr);
  ASSERT_NE(state, nullptr);
  ASSERT_NE(plan, nullptr);

  EXPECT_TRUE(admission->IsReservedNumber(6));
  EXPECT_TRUE(admission->IsReservedName("per_rank"));
  EXPECT_TRUE(state->IsReservedNumber(10));
  EXPECT_TRUE(state->IsReservedNumber(11));
  EXPECT_TRUE(state->IsReservedNumber(12));
  EXPECT_TRUE(state->IsReservedNumber(13));
  EXPECT_TRUE(state->IsReservedName("latency_histogram_delta"));
  EXPECT_TRUE(state->IsReservedName("throughput_tokens_per_second"));
  EXPECT_TRUE(state->IsReservedName("connector_state"));
  EXPECT_TRUE(state->IsReservedName("failure_counters"));
  EXPECT_TRUE(plan->IsReservedNumber(13));
  EXPECT_TRUE(plan->IsReservedName("prediction"));

  xllm::proto::AdmissionResult result;
  result.set_disposition(xllm::proto::ADMISSION_DISPOSITION_ACCEPTED);
  result.set_reason(xllm::proto::ADMISSION_REASON_NONE);
  result.set_state(xllm::proto::ATTEMPT_LIFECYCLE_STATE_RESERVED);
  result.set_reservation_ttl_ms(100);
  const std::string golden =
      "\x08\x01"
      "\x10\x01"
      "\x18\x02"
      "\x28\x64";
  EXPECT_EQ(result.SerializeAsString(), golden);
}

TEST(ProviderContractTest, FullPerDpStateRequiresEveryUniqueRank) {
  ProviderDescriptor descriptor = make_descriptor(kOpenModeCases[1]);
  descriptor.add_capabilities(xllm::proto::PROVIDER_CAPABILITY_PER_DP_STATE);
  xllm::proto::EngineState state = make_state(descriptor);
  state.set_state_quality(xllm::proto::STATE_QUALITY_FULL);
  state.add_per_dp()->set_dp_rank(0);
  EXPECT_EQ(validate_engine_state(descriptor, state).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE);

  state.add_per_dp()->set_dp_rank(1);
  EXPECT_TRUE(validate_engine_state(descriptor, state).ok());
  state.mutable_per_dp(1)->set_dp_rank(0);
  EXPECT_EQ(validate_engine_state(descriptor, state).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE);
}

TEST(ProviderContractTest, EngineStateRejectsInvalidRatios) {
  ProviderDescriptor descriptor = make_descriptor(kOpenModeCases[1]);
  descriptor.add_capabilities(xllm::proto::PROVIDER_CAPABILITY_PER_DP_STATE);
  xllm::proto::EngineState state = make_state(descriptor);
  state.add_per_dp()->set_dp_rank(0);
  state.mutable_per_dp(0)->set_kv_used_ratio(1.1);
  EXPECT_EQ(validate_engine_state(descriptor, state).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE);
}

TEST(ProviderContractTest, ExecutionPlanValidatesEveryOpenRoleShape) {
  for (const auto& test_case : kOpenModeCases) {
    ProviderDescriptor descriptor = make_descriptor(test_case);
    ContractResult result =
        validate_execution_plan(descriptor, make_plan(descriptor));
    EXPECT_TRUE(result.ok()) << result.message();
  }
}

TEST(ProviderContractTest, ExecutionPlanRejectsWrongRoleAndCapability) {
  ProviderDescriptor descriptor = make_descriptor(kOpenModeCases[0]);
  ExecutionPlan plan = make_plan(descriptor);
  plan.mutable_selected_roles(0)->set_role(xllm::proto::ENGINE_ROLE_DECODE);
  EXPECT_EQ(validate_execution_plan(descriptor, plan).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_SELECTED_ROLES);

  plan = make_plan(descriptor);
  plan.mutable_required_capabilities()->RemoveLast();
  EXPECT_EQ(validate_execution_plan(descriptor, plan).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_CAPABILITY);

  plan = make_plan(descriptor);
  plan.clear_attempt_seq();
  EXPECT_EQ(validate_execution_plan(descriptor, plan).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD);

  plan = make_plan(descriptor);
  plan.mutable_selected_roles(0)->set_incarnation_id("other-incarnation");
  EXPECT_EQ(validate_execution_plan(descriptor, plan).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);
}

TEST(ProviderContractTest, ExecutionPlanRejectsOverflowingDeadlineBudget) {
  ProviderDescriptor descriptor = make_descriptor(kOpenModeCases[1]);
  ExecutionPlan plan = make_plan(descriptor);
  plan.mutable_deadline_budget()->set_remaining_ms(
      std::numeric_limits<uint64_t>::max());
  plan.mutable_deadline_budget()->set_submit_ms(
      std::numeric_limits<uint64_t>::max());
  plan.mutable_deadline_budget()->set_handoff_ms(1);
  plan.mutable_deadline_budget()->set_output_ms(0);
  EXPECT_EQ(validate_execution_plan(descriptor, plan).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE);
}

TEST(ProviderContractTest, RegistryOwnsAndFindsValidatedAdapters) {
  ProviderAdapterRegistry registry;
  ProviderDescriptor descriptor = make_descriptor(kOpenModeCases[1]);
  ContractResult result =
      registry.register_adapter(std::make_unique<TestAdapter>(descriptor));
  ASSERT_TRUE(result.ok()) << result.message();
  EXPECT_EQ(registry.size(), 1u);

  const ProviderAdapter* adapter = registry.find(
      descriptor.identity().provider_id(), descriptor.profile_digest());
  ASSERT_NE(adapter, nullptr);
  xllm::proto::EncodedRequest encoded;
  EXPECT_TRUE(
      adapter->request_codec().encode(make_request(), {}, &encoded).ok());
  EXPECT_TRUE(
      validate_encoded_request(descriptor, make_request(), encoded).ok());
  EXPECT_EQ(registry.find(xllm::proto::PROVIDER_ID_VLLM_ASCEND,
                          descriptor.profile_digest()),
            nullptr);
}

TEST(ProviderContractTest, RegistryRejectsDuplicateAndInvalidAdapters) {
  ProviderAdapterRegistry registry;
  ProviderDescriptor descriptor = make_descriptor(kOpenModeCases[1]);
  const ProviderAdapter* registered = nullptr;
  EXPECT_EQ(registry.find_or_register_adapter(nullptr, &registered).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD);
  EXPECT_EQ(registry
                .find_or_register_adapter(
                    std::make_unique<TestAdapter>(descriptor), nullptr)
                .error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD);
  EXPECT_EQ(registry.find_compatible_adapter(descriptor, nullptr).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD);
  ASSERT_TRUE(
      registry.register_adapter(std::make_unique<TestAdapter>(descriptor))
          .ok());
  EXPECT_EQ(registry.register_adapter(std::make_unique<TestAdapter>(descriptor))
                .error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DUPLICATE_ADAPTER);

  ProviderDescriptor invalid = make_descriptor(kOpenModeCases[1]);
  invalid.clear_profile_digest();
  EXPECT_EQ(
      registry.register_adapter(std::make_unique<TestAdapter>(invalid)).error(),
      xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD);
  EXPECT_EQ(registry.size(), 1u);
}

TEST(ProviderContractTest, RegistryLazilyReusesIdenticalAdapter) {
  ProviderAdapterRegistry registry;
  ProviderDescriptor descriptor = make_descriptor(kOpenModeCases[1]);
  const ProviderAdapter* first = nullptr;
  const ProviderAdapter* second = nullptr;

  ASSERT_TRUE(registry
                  .find_or_register_adapter(
                      std::make_unique<TestAdapter>(descriptor), &first)
                  .ok());
  ASSERT_TRUE(registry
                  .find_or_register_adapter(
                      std::make_unique<TestAdapter>(descriptor), &second)
                  .ok());
  EXPECT_NE(first, nullptr);
  EXPECT_EQ(first, second);
  EXPECT_EQ(registry.size(), 1u);

  const ProviderAdapter* found = nullptr;
  ASSERT_TRUE(registry.find_compatible_adapter(descriptor, &found).ok());
  EXPECT_EQ(found, first);
}

TEST(ProviderContractTest, RegistryReusesProfileAcrossEngineReplicas) {
  ProviderAdapterRegistry registry;
  ProviderDescriptor first_descriptor = make_descriptor(kOpenModeCases[1]);
  ProviderDescriptor replica_descriptor = first_descriptor;
  replica_descriptor.mutable_identity()->set_engine_uid("engine-replica-2");
  replica_descriptor.mutable_identity()->set_incarnation_id("incarnation-2");
  replica_descriptor.mutable_endpoint()->set_address("127.0.0.1:19001");
  const ProviderAdapter* first = nullptr;
  const ProviderAdapter* replica = nullptr;

  ASSERT_TRUE(registry
                  .find_or_register_adapter(
                      std::make_unique<TestAdapter>(first_descriptor), &first)
                  .ok());
  ASSERT_TRUE(
      registry
          .find_or_register_adapter(
              std::make_unique<TestAdapter>(replica_descriptor), &replica)
          .ok());
  EXPECT_NE(first, nullptr);
  EXPECT_EQ(replica, first);
  EXPECT_EQ(registry.size(), 1u);

  const ProviderAdapter* found = nullptr;
  ASSERT_TRUE(
      registry.find_compatible_adapter(replica_descriptor, &found).ok());
  EXPECT_EQ(found, first);
}

TEST(ProviderContractTest, RegistryStillRejectsProfileRuntimeCollision) {
  ProviderAdapterRegistry registry;
  ProviderDescriptor descriptor = make_descriptor(kOpenModeCases[1]);
  const ProviderAdapter* registered = nullptr;
  ASSERT_TRUE(registry
                  .find_or_register_adapter(
                      std::make_unique<TestAdapter>(descriptor), &registered)
                  .ok());

  ProviderDescriptor collision = descriptor;
  collision.mutable_identity()->set_runtime_version("different-runtime");
  const ProviderAdapter* conflicting = nullptr;
  EXPECT_EQ(registry.find_compatible_adapter(collision, &conflicting).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);
  EXPECT_EQ(conflicting, nullptr);
}

TEST(ProviderContractTest, RegistryRejectsProfileDigestCollision) {
  ProviderAdapterRegistry registry;
  ProviderDescriptor descriptor = make_descriptor(kOpenModeCases[1]);
  const ProviderAdapter* registered = nullptr;
  ASSERT_TRUE(registry
                  .find_or_register_adapter(
                      std::make_unique<TestAdapter>(descriptor), &registered)
                  .ok());

  ProviderDescriptor collision = descriptor;
  collision.mutable_model()->set_model_revision("other-model");
  const ProviderAdapter* conflicting = nullptr;
  EXPECT_EQ(registry
                .find_or_register_adapter(
                    std::make_unique<TestAdapter>(collision), &conflicting)
                .error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);
  EXPECT_EQ(registry.find_compatible_adapter(collision, &conflicting).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);
  EXPECT_EQ(conflicting, nullptr);
  EXPECT_EQ(registry.size(), 1u);
}

TEST(ProviderContractTest, RegistrySupportsConcurrentRegistrationAndLookup) {
  ProviderAdapterRegistry registry;
  constexpr size_t kAdapterCount = 8;
  std::vector<int> registered(kAdapterCount, 0);
  std::vector<std::thread> threads;
  threads.reserve(kAdapterCount);
  for (size_t i = 0; i < kAdapterCount; ++i) {
    threads.emplace_back([i, &registered, &registry]() {
      ProviderDescriptor descriptor = make_descriptor(kOpenModeCases[1]);
      descriptor.set_profile_digest("profile-" + std::to_string(i));
      const bool inserted =
          registry.register_adapter(std::make_unique<TestAdapter>(descriptor))
              .ok();
      const bool found = registry.find(xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                                       descriptor.profile_digest()) != nullptr;
      registered[i] = inserted && found;
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(registry.size(), kAdapterCount);
  for (size_t i = 0; i < kAdapterCount; ++i) {
    EXPECT_TRUE(registered[i]);
    EXPECT_NE(registry.find(xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                            "profile-" + std::to_string(i)),
              nullptr);
  }
}

TEST(ProviderContractTest, RegistryCoalescesConcurrentLazyRegistration) {
  ProviderAdapterRegistry registry;
  constexpr size_t kThreadCount = 8;
  std::vector<const ProviderAdapter*> adapters(kThreadCount, nullptr);
  std::vector<int> registered(kThreadCount, 0);
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (size_t i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([i, &adapters, &registered, &registry]() {
      ProviderDescriptor descriptor = make_descriptor(kOpenModeCases[1]);
      registered[i] =
          registry
              .find_or_register_adapter(
                  std::make_unique<TestAdapter>(descriptor), &adapters[i])
              .ok();
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  ASSERT_NE(adapters[0], nullptr);
  EXPECT_EQ(registry.size(), 1u);
  for (size_t i = 0; i < kThreadCount; ++i) {
    EXPECT_TRUE(registered[i]);
    EXPECT_EQ(adapters[i], adapters[0]);
  }
}

TEST(ProviderContractTest, XllmNativeAdapterUsesExactRenderedRequest) {
  ProviderDescriptor descriptor = make_descriptor(kOpenModeCases[0]);
  XllmNativeAdapter adapter(descriptor, std::make_unique<TestNativeRenderer>());
  xllm::proto::EncodedRequest encoded;
  ContractResult result =
      adapter.request_codec().encode(make_request(), {}, &encoded);
  ASSERT_TRUE(result.ok()) << result.message();
  EXPECT_EQ(adapter.dispatch_kind(), ProviderDispatchKind::XLLM_NATIVE_RPC);
  EXPECT_EQ(encoded.provider_id(), xllm::proto::PROVIDER_ID_XLLM_NATIVE);
  EXPECT_EQ(encoded.token_count_quality(),
            xllm::proto::TOKEN_COUNT_QUALITY_EXACT);
  EXPECT_EQ(encoded.prompt_tokens(), 4u);
  EXPECT_EQ(encoded.prompt_tokens_upper_bound(), 4u);
  EXPECT_EQ(encoded.provider_payload(), "native:{}");
}

TEST(ProviderContractTest, ProductionAdapterFactorySupportsKnownProviders) {
  std::unique_ptr<ProviderAdapter> adapter;
  ProviderDescriptor descriptor = make_descriptor(kOpenModeCases[0]);
  EXPECT_EQ(create_provider_adapter(descriptor, nullptr).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD);
  ASSERT_TRUE(create_provider_adapter(descriptor, &adapter).ok());
  ASSERT_NE(adapter, nullptr);
  EXPECT_EQ(adapter->dispatch_kind(), ProviderDispatchKind::XLLM_NATIVE_RPC);

  descriptor = make_descriptor(kOpenModeCases[3]);
  ASSERT_TRUE(create_provider_adapter(descriptor, &adapter).ok());
  ASSERT_NE(adapter, nullptr);
  EXPECT_EQ(adapter->dispatch_kind(), ProviderDispatchKind::OPENAI_HTTP);

  descriptor.mutable_identity()->set_provider_id(
      xllm::proto::PROVIDER_ID_UNSPECIFIED);
  EXPECT_EQ(create_provider_adapter(descriptor, &adapter).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_UNKNOWN_ENUM_VALUE);
  EXPECT_EQ(adapter, nullptr);
}

TEST(ProviderContractTest, XllmNativeAdapterFailsClosedWithoutRenderer) {
  XllmNativeAdapter adapter(make_descriptor(kOpenModeCases[0]), nullptr);
  xllm::proto::EncodedRequest encoded;
  EXPECT_EQ(
      adapter.request_codec().encode(make_request(), {}, &encoded).error(),
      xllm::proto::PROVIDER_CONTRACT_ERROR_ENCODING_FAILED);
}

TEST(ProviderContractTest, PreparedNativeRendererPreservesCanonicalAndTokens) {
  const std::vector<int32_t> token_ids = {11, 22, 33};
  xllm::proto::CanonicalRequest canonical = make_request();
  XllmNativeAdapter adapter(
      make_descriptor(kOpenModeCases[0]),
      std::make_unique<XllmNativePreparedRequestRenderer>());
  RequestEncodingContext context{
      .native_token_ids = &token_ids,
      .request_uid = "request-1",
      .attempt_seq = 0,
      .native_renderer_digest = "renderer-sha256",
  };
  xllm::proto::EncodedRequest encoded;

  ContractResult result =
      adapter.request_codec().encode(canonical, context, &encoded);
  ASSERT_TRUE(result.ok()) << result.message();
  EXPECT_EQ(encoded.prompt_tokens(), 3u);
  const nlohmann::json payload =
      nlohmann::json::parse(encoded.provider_payload());
  EXPECT_EQ(payload.at("canonical_payload_schema"), "json-v1");
  EXPECT_EQ(payload.at("canonical_request"), nlohmann::json::object());
  EXPECT_EQ(payload.at("token_ids"), token_ids);

  canonical.set_attempt_seq(1);
  EXPECT_EQ(
      adapter.request_codec().encode(canonical, context, &encoded).error(),
      xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);

  const std::vector<int32_t> retry_token_ids = {44, 55};
  context.native_token_ids = &retry_token_ids;
  context.attempt_seq = 1;
  ASSERT_TRUE(
      adapter.request_codec().encode(canonical, context, &encoded).ok());
  EXPECT_EQ(encoded.prompt_tokens(), 2u);
  EXPECT_EQ(nlohmann::json::parse(encoded.provider_payload()).at("token_ids"),
            retry_token_ids);

  XllmNativeAdapter missing_digest_adapter(
      make_descriptor(kOpenModeCases[0]),
      std::make_unique<XllmNativePreparedRequestRenderer>());
  canonical.set_attempt_seq(0);
  context.attempt_seq = 0;
  context.native_renderer_digest.clear();
  encoded.set_provider_payload("stale-payload");
  EXPECT_EQ(missing_digest_adapter.request_codec()
                .encode(canonical, context, &encoded)
                .error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_ENCODING_FAILED);
  EXPECT_EQ(encoded.ByteSizeLong(), 0u);
}

TEST(ProviderContractTest, VllmAscendAdapterPreservesCanonicalPayload) {
  ProviderDescriptor descriptor = make_descriptor(kOpenModeCases[3]);
  VllmAscendAdapter adapter(descriptor);
  xllm::proto::CanonicalRequest request = make_request();
  request.set_canonical_payload_schema(kOpenAiHttpJsonSchema);
  request.set_canonical_payload("{\"stream\":true}");
  xllm::proto::EncodedRequest encoded;
  ContractResult result = adapter.request_codec().encode(request, {}, &encoded);
  ASSERT_TRUE(result.ok()) << result.message();
  EXPECT_EQ(adapter.dispatch_kind(), ProviderDispatchKind::OPENAI_HTTP);
  EXPECT_EQ(encoded.provider_id(), xllm::proto::PROVIDER_ID_VLLM_ASCEND);
  EXPECT_EQ(encoded.token_count_quality(),
            xllm::proto::TOKEN_COUNT_QUALITY_UNKNOWN);
  EXPECT_EQ(encoded.prompt_tokens(), 0u);
  EXPECT_EQ(encoded.prompt_tokens_upper_bound(), 0u);
  EXPECT_EQ(encoded.provider_payload(), request.canonical_payload());
}

TEST(ProviderContractTest, VllmAscendAdapterRejectsAmbiguousPayloadSchema) {
  VllmAscendAdapter adapter(make_descriptor(kOpenModeCases[3]));
  xllm::proto::EncodedRequest encoded;
  encoded.set_provider_payload("stale-payload");
  EXPECT_EQ(
      adapter.request_codec().encode(make_request(), {}, &encoded).error(),
      xllm::proto::PROVIDER_CONTRACT_ERROR_ENCODING_FAILED);
  EXPECT_EQ(encoded.ByteSizeLong(), 0u);
}

TEST(ProviderContractTest, ProductionAdaptersRejectWrongProviderDescriptor) {
  ProviderDescriptor descriptor = make_descriptor(kOpenModeCases[3]);
  XllmNativeAdapter native_adapter(descriptor,
                                   std::make_unique<TestNativeRenderer>());
  xllm::proto::EncodedRequest encoded;
  EXPECT_EQ(native_adapter.request_codec()
                .encode(make_request(), {}, &encoded)
                .error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);

  descriptor = make_descriptor(kOpenModeCases[0]);
  VllmAscendAdapter vllm_adapter(descriptor);
  xllm::proto::CanonicalRequest vllm_request = make_request();
  vllm_request.set_canonical_payload_schema(kOpenAiHttpJsonSchema);
  EXPECT_EQ(
      vllm_adapter.request_codec().encode(vllm_request, {}, &encoded).error(),
      xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);

  ProviderAdapterRegistry registry;
  EXPECT_EQ(registry
                .register_adapter(std::make_unique<XllmNativeAdapter>(
                    make_descriptor(kOpenModeCases[3]),
                    std::make_unique<TestNativeRenderer>()))
                .error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);
  EXPECT_EQ(registry
                .register_adapter(std::make_unique<VllmAscendAdapter>(
                    make_descriptor(kOpenModeCases[0])))
                .error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);
}

TEST(ProviderContractTest, DispatchKindResolvesOnlyKnownProviders) {
  EXPECT_EQ(
      resolve_provider_dispatch_kind(xllm::proto::PROVIDER_ID_XLLM_NATIVE),
      ProviderDispatchKind::XLLM_NATIVE_RPC);
  EXPECT_EQ(
      resolve_provider_dispatch_kind(xllm::proto::PROVIDER_ID_VLLM_ASCEND),
      ProviderDispatchKind::OPENAI_HTTP);
  EXPECT_FALSE(
      resolve_provider_dispatch_kind(xllm::proto::PROVIDER_ID_UNSPECIFIED)
          .has_value());
}

TEST(ProviderContractTest, RemotePdCompatibilityAcceptsDistinctProfiles) {
  auto [prefill, decode] = make_remote_pd_descriptors();
  std::string proof;

  const ContractResult result =
      validate_remote_pd_compatibility(prefill, decode, &proof);
  ASSERT_TRUE(result.ok()) << result.message();
  EXPECT_NE(proof.find("profile-p"), std::string::npos);
  EXPECT_NE(proof.find("profile-d"), std::string::npos);
}

TEST(ProviderContractTest, RemotePdCompatibilityRejectsHardMismatches) {
  auto [prefill, baseline_decode] = make_remote_pd_descriptors();
  std::string proof;

  ProviderDescriptor decode = baseline_decode;
  decode.mutable_model()->set_model_revision("other-model");
  EXPECT_EQ(validate_remote_pd_compatibility(prefill, decode, &proof).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);

  decode = baseline_decode;
  decode.mutable_kv()->set_connector_version("other-connector-version");
  EXPECT_EQ(validate_remote_pd_compatibility(prefill, decode, &proof).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);

  decode = baseline_decode;
  decode.mutable_kv()->set_kv_namespace("other-kv-namespace");
  EXPECT_EQ(validate_remote_pd_compatibility(prefill, decode, &proof).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);

  decode = baseline_decode;
  decode.mutable_kv()->set_hash_seed(2048);
  EXPECT_EQ(validate_remote_pd_compatibility(prefill, decode, &proof).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);

  decode = baseline_decode;
  decode.mutable_topology()->set_tp(2);
  EXPECT_EQ(validate_remote_pd_compatibility(prefill, decode, &proof).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);

  decode = baseline_decode;
  decode.mutable_identity()->set_runtime_version("other-runtime");
  EXPECT_EQ(validate_remote_pd_compatibility(prefill, decode, &proof).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);
}

TEST(ProviderContractTest, StrictRouteSelectorUsesCompatibilityMatrix) {
  auto [prefill_descriptor, decode_descriptor] = make_remote_pd_descriptors();
  ProviderRouteCandidate prefill{
      .engine_uid = "engine-p",
      .provider_id = xllm::proto::PROVIDER_ID_XLLM_NATIVE,
      .role = xllm::proto::ENGINE_ROLE_PREFILL,
      .schedulable = true,
      .descriptor = &prefill_descriptor,
  };
  ProviderRouteCandidate decode{
      .engine_uid = "engine-d",
      .provider_id = xllm::proto::PROVIDER_ID_XLLM_NATIVE,
      .role = xllm::proto::ENGINE_ROLE_DECODE,
      .schedulable = true,
      .descriptor = &decode_descriptor,
  };
  ProviderRouteSelection selection;

  ASSERT_TRUE(
      ProviderRouteSelector::select({prefill},
                                    {decode},
                                    xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                                    0,
                                    0,
                                    &selection));

  decode_descriptor.mutable_kv()->set_kv_layout_digest("other-layout");
  EXPECT_FALSE(
      ProviderRouteSelector::select({prefill},
                                    {decode},
                                    xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                                    0,
                                    0,
                                    &selection));

  decode_descriptor = make_remote_pd_descriptors().second;
  decode.descriptor = nullptr;
  EXPECT_FALSE(
      ProviderRouteSelector::select({prefill},
                                    {decode},
                                    xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                                    0,
                                    0,
                                    &selection));

  decode.descriptor = &decode_descriptor;
  prefill.link_state_required = true;
  EXPECT_FALSE(
      ProviderRouteSelector::select({prefill},
                                    {decode},
                                    xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                                    0,
                                    0,
                                    &selection));

  prefill.ready_peer_engine_uids.emplace_back(decode.engine_uid);
  EXPECT_TRUE(
      ProviderRouteSelector::select({prefill},
                                    {decode},
                                    xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                                    0,
                                    0,
                                    &selection));
}

TEST(ProviderContractTest, CanonicalBuilderPreservesIngressSemantics) {
  CanonicalRequestInput input;
  input.correlation.set_global_request_id("global-1");
  input.correlation.set_trace_id("trace-1");
  input.correlation.set_request_uid("request-1");
  input.correlation.set_attempt_seq(0);
  input.api_kind = xllm::proto::API_KIND_CHAT_COMPLETIONS;
  input.model_revision = "model-r1";
  input.payload_schema = kOpenAiHttpJsonSchema;
  input.payload = R"({"model":"model-r1","messages":[]})";
  input.effective_max_new_tokens = 128;
  input.n = 1;
  input.best_of = 2;
  input.priority = 3;
  input.remaining_deadline_ms = 1000;
  input.ttft_slo_ms = 100;

  xllm::proto::CanonicalRequest request;
  ASSERT_TRUE(build_canonical_request(input, &request).ok());
  EXPECT_TRUE(request.strict());
  EXPECT_EQ(request.attempt_seq(), 0u);
  EXPECT_EQ(request.best_of(), 2u);
  ASSERT_TRUE(request.has_ttft_slo_ms());
  EXPECT_EQ(request.ttft_slo_ms(), 100u);

  input.correlation.clear_attempt_seq();
  EXPECT_EQ(build_canonical_request(input, &request).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD);
  input.correlation.set_attempt_seq(0);
  input.best_of = 0;
  EXPECT_EQ(build_canonical_request(input, &request).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE);
}

TEST(ProviderContractTest, ExecutionPlanBuilderBuildsRemotePdPlan) {
  auto [prefill, decode] = make_remote_pd_descriptors();
  const xllm::proto::CanonicalRequest canonical = make_request();
  const xllm::proto::EncodedRequest encoded = make_encoded_request(prefill);
  xllm::proto::ExecutionPlan plan;

  ContractResult result =
      build_execution_plan(canonical, encoded, prefill, &decode, &plan);
  ASSERT_TRUE(result.ok()) << result.message();
  ASSERT_EQ(plan.selected_roles_size(), 2);
  EXPECT_EQ(plan.selected_roles(0).engine_uid(), "engine-p");
  EXPECT_EQ(plan.selected_roles(1).engine_uid(), "engine-d");
  EXPECT_EQ(plan.mode(), xllm::proto::EXECUTION_MODE_REMOTE_PD);
  EXPECT_EQ(plan.resource_estimate().prompt_tokens(), 4u);
  EXPECT_EQ(plan.resource_estimate().output_tokens(), 128u);
  EXPECT_EQ(plan.resource_estimate().kv_blocks(), 9u);
  EXPECT_NE(plan.compatibility_proof().find("remote-pd-v1"), std::string::npos);

  decode.mutable_identity()->set_runtime_version("incompatible");
  EXPECT_EQ(
      build_execution_plan(canonical, encoded, prefill, &decode, &plan).error(),
      xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);
  EXPECT_EQ(plan.ByteSizeLong(), 0u);
}

TEST(ProviderContractTest, ExecutionPlanBuilderBuildsNativeSingleEngineModes) {
  for (size_t index : {size_t{1}, size_t{2}}) {
    ProviderDescriptor descriptor = make_descriptor(kOpenModeCases[index]);
    const xllm::proto::CanonicalRequest canonical = make_request();
    const xllm::proto::EncodedRequest encoded =
        make_encoded_request(descriptor);
    xllm::proto::ExecutionPlan plan;
    const xllm::proto::ExecutionMode mode = kOpenModeCases[index].mode;
    const ContractResult result = build_execution_plan(
        canonical, encoded, descriptor, nullptr, &plan, mode);
    ASSERT_TRUE(result.ok()) << result.message();
    ASSERT_EQ(plan.selected_roles_size(), 1);
    EXPECT_EQ(plan.selected_roles(0).role(), kOpenModeCases[index].role);
    EXPECT_EQ(plan.mode(), mode);
    EXPECT_EQ(plan.transfer_mode(), xllm::proto::TRANSFER_MODE_NONE);
    EXPECT_FALSE(plan.compatibility_proof().empty());
  }

  auto [prefill, decode] = make_remote_pd_descriptors();
  const xllm::proto::CanonicalRequest canonical = make_request();
  const xllm::proto::EncodedRequest encoded = make_encoded_request(prefill);
  xllm::proto::ExecutionPlan plan;
  EXPECT_EQ(
      build_execution_plan(canonical,
                           encoded,
                           prefill,
                           nullptr,
                           &plan,
                           xllm::proto::EXECUTION_MODE_LOCAL_PREFILL_DECODE)
          .error(),
      xllm::proto::PROVIDER_CONTRACT_ERROR_MODE_NOT_OPEN);
}

TEST(ProviderContractTest, ExecutionPlanBuilderBuildsAggregatedPlan) {
  ProviderDescriptor descriptor = make_descriptor(kOpenModeCases[3]);
  xllm::proto::CanonicalRequest canonical = make_request();
  canonical.set_canonical_payload_schema(kOpenAiHttpJsonSchema);
  xllm::proto::EncodedRequest encoded;
  VllmAscendAdapter adapter(descriptor);
  ASSERT_TRUE(adapter.request_codec().encode(canonical, {}, &encoded).ok());
  xllm::proto::ExecutionPlan plan;

  ContractResult result =
      build_execution_plan(canonical, encoded, descriptor, nullptr, &plan);
  ASSERT_TRUE(result.ok()) << result.message();
  ASSERT_EQ(plan.selected_roles_size(), 1);
  EXPECT_EQ(plan.selected_roles(0).role(), xllm::proto::ENGINE_ROLE_AGGREGATED);
  EXPECT_EQ(plan.mode(), xllm::proto::EXECUTION_MODE_AGGREGATED);
  EXPECT_EQ(plan.provider_payload(), canonical.canonical_payload());
  EXPECT_EQ(plan.resource_estimate().kv_blocks(), 0u);
  EXPECT_EQ(plan.reason_codes(plan.reason_codes_size() - 1),
            "prompt-token-count-unknown");
  EXPECT_NE(plan.compatibility_proof().find("aggregated-v1"),
            std::string::npos);

  EXPECT_EQ(
      build_execution_plan(canonical, encoded, descriptor, &descriptor, &plan)
          .error(),
      xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_SELECTED_ROLES);
}

}  // namespace
}  // namespace xllm_service::provider
