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

#include <gtest/gtest.h>

namespace xllm_service::provider {
namespace {

xllm::proto::ProviderDescriptor make_descriptor(xllm::proto::EngineRole role) {
  xllm::proto::ProviderDescriptor descriptor;
  descriptor.mutable_identity()->set_provider_id(
      xllm::proto::PROVIDER_ID_XLLM_NATIVE);
  descriptor.mutable_serving()->set_role(role);
  xllm::proto::ExecutionModeSpec* remote =
      descriptor.mutable_serving()->add_execution_modes();
  remote->set_mode(xllm::proto::EXECUTION_MODE_REMOTE_PD);
  remote->set_transfer_mode(xllm::proto::TRANSFER_MODE_LAYERWISE_PUSH);
  remote->set_selection_order(xllm::proto::SELECTION_ORDER_P_FIRST);
  remote->set_binding_stage(xllm::proto::BINDING_STAGE_BEFORE_PREFILL);
  descriptor.add_capabilities(
      xllm::proto::PROVIDER_CAPABILITY_REMOTE_PD_LAYERWISE_PUSH);
  descriptor.add_capabilities(
      xllm::proto::PROVIDER_CAPABILITY_NATIVE_RESERVATION);
  if (role == xllm::proto::ENGINE_ROLE_PREFILL) {
    xllm::proto::ExecutionModeSpec* prefill_only =
        descriptor.mutable_serving()->add_execution_modes();
    prefill_only->set_mode(xllm::proto::EXECUTION_MODE_PREFILL_ONLY);
    prefill_only->set_transfer_mode(xllm::proto::TRANSFER_MODE_NONE);
    prefill_only->set_selection_order(xllm::proto::SELECTION_ORDER_P_ONLY);
    prefill_only->set_binding_stage(xllm::proto::BINDING_STAGE_AT_SUBMIT);
    descriptor.add_capabilities(xllm::proto::PROVIDER_CAPABILITY_PREFILL_ONLY);
  } else {
    xllm::proto::ExecutionModeSpec* local =
        descriptor.mutable_serving()->add_execution_modes();
    local->set_mode(xllm::proto::EXECUTION_MODE_LOCAL_PREFILL_DECODE);
    local->set_transfer_mode(xllm::proto::TRANSFER_MODE_NONE);
    local->set_selection_order(xllm::proto::SELECTION_ORDER_D_ONLY);
    local->set_binding_stage(xllm::proto::BINDING_STAGE_AT_SUBMIT);
    descriptor.add_capabilities(
        xllm::proto::PROVIDER_CAPABILITY_LOCAL_PREFILL_DECODE);
    descriptor.add_capabilities(
        xllm::proto::PROVIDER_CAPABILITY_MIXED_PREFILL_DECODE_ACCOUNTING);
  }
  descriptor.add_capabilities(
      xllm::proto::PROVIDER_CAPABILITY_STRUCTURED_ADMISSION);
  return descriptor;
}

NativeExecutionModeConfig all_modes_config() {
  return NativeExecutionModeConfig{
      .local_prefill_decode_enabled = true,
      .local_prefill_decode_bucket_permyriad = 10000,
      .local_prefill_decode_prompt_token_cap = 512,
      .prefill_only_enabled = true,
      .prefill_only_output_token_cap = 1,
  };
}

TEST(NativeExecutionModeSelectorTest, PrefillOnlyHasPrecedence) {
  xllm::proto::ProviderDescriptor prefill =
      make_descriptor(xllm::proto::ENGINE_ROLE_PREFILL);
  xllm::proto::ProviderDescriptor decode =
      make_descriptor(xllm::proto::ENGINE_ROLE_DECODE);
  NativeExecutionModeDecision decision;
  EXPECT_TRUE(select_native_execution_mode(
                  all_modes_config(),
                  NativeExecutionModeInput{.stable_request_hash = 0,
                                           .prompt_tokens = 32,
                                           .output_tokens = 1,
                                           .prefill = &prefill,
                                           .decode = &decode},
                  &decision)
                  .ok());
  EXPECT_EQ(decision.mode, xllm::proto::EXECUTION_MODE_PREFILL_ONLY);
  EXPECT_EQ(decision.reason_code, "native-prefill-only");
}

TEST(NativeExecutionModeSelectorTest, LocalRespectsBucketAndTokenCap) {
  xllm::proto::ProviderDescriptor prefill =
      make_descriptor(xllm::proto::ENGINE_ROLE_PREFILL);
  xllm::proto::ProviderDescriptor decode =
      make_descriptor(xllm::proto::ENGINE_ROLE_DECODE);
  NativeExecutionModeConfig config = all_modes_config();
  config.prefill_only_enabled = false;
  config.local_prefill_decode_bucket_permyriad = 100;
  NativeExecutionModeDecision decision;
  EXPECT_TRUE(select_native_execution_mode(
                  config,
                  NativeExecutionModeInput{.stable_request_hash = 99,
                                           .prompt_tokens = 512,
                                           .output_tokens = 8,
                                           .prefill = &prefill,
                                           .decode = &decode},
                  &decision)
                  .ok());
  EXPECT_EQ(decision.mode, xllm::proto::EXECUTION_MODE_LOCAL_PREFILL_DECODE);

  EXPECT_TRUE(select_native_execution_mode(
                  config,
                  NativeExecutionModeInput{.stable_request_hash = 100,
                                           .prompt_tokens = 512,
                                           .output_tokens = 8,
                                           .prefill = &prefill,
                                           .decode = &decode},
                  &decision)
                  .ok());
  EXPECT_EQ(decision.mode, xllm::proto::EXECUTION_MODE_REMOTE_PD);

  EXPECT_TRUE(select_native_execution_mode(
                  config,
                  NativeExecutionModeInput{.stable_request_hash = 0,
                                           .prompt_tokens = 513,
                                           .output_tokens = 8,
                                           .prefill = &prefill,
                                           .decode = &decode},
                  &decision)
                  .ok());
  EXPECT_EQ(decision.mode, xllm::proto::EXECUTION_MODE_REMOTE_PD);
}

TEST(NativeExecutionModeSelectorTest, MissingModeCapabilityFallsBackRemote) {
  xllm::proto::ProviderDescriptor prefill =
      make_descriptor(xllm::proto::ENGINE_ROLE_PREFILL);
  xllm::proto::ProviderDescriptor decode =
      make_descriptor(xllm::proto::ENGINE_ROLE_DECODE);
  decode.clear_capabilities();
  decode.add_capabilities(
      xllm::proto::PROVIDER_CAPABILITY_REMOTE_PD_LAYERWISE_PUSH);
  decode.add_capabilities(xllm::proto::PROVIDER_CAPABILITY_NATIVE_RESERVATION);
  NativeExecutionModeConfig config = all_modes_config();
  config.prefill_only_enabled = false;
  NativeExecutionModeDecision decision;
  EXPECT_TRUE(select_native_execution_mode(
                  config,
                  NativeExecutionModeInput{.stable_request_hash = 0,
                                           .prompt_tokens = 1,
                                           .output_tokens = 2,
                                           .prefill = &prefill,
                                           .decode = &decode},
                  &decision)
                  .ok());
  EXPECT_EQ(decision.mode, xllm::proto::EXECUTION_MODE_REMOTE_PD);
}

TEST(NativeExecutionModeSelectorTest, RejectsNoFailClosedRemoteBaseline) {
  xllm::proto::ProviderDescriptor prefill =
      make_descriptor(xllm::proto::ENGINE_ROLE_PREFILL);
  xllm::proto::ProviderDescriptor decode =
      make_descriptor(xllm::proto::ENGINE_ROLE_DECODE);
  decode.mutable_serving()->clear_execution_modes();
  NativeExecutionModeDecision decision;
  EXPECT_FALSE(select_native_execution_mode(
                   NativeExecutionModeConfig{},
                   NativeExecutionModeInput{.stable_request_hash = 0,
                                            .prompt_tokens = 1,
                                            .output_tokens = 2,
                                            .prefill = &prefill,
                                            .decode = &decode},
                   &decision)
                   .ok());
  EXPECT_EQ(decision.mode, xllm::proto::EXECUTION_MODE_UNSPECIFIED);
}

TEST(NativeExecutionModeSelectorTest, RejectsInvalidConfigurationAndInput) {
  NativeExecutionModeDecision decision;
  NativeExecutionModeConfig config;
  config.local_prefill_decode_enabled = true;
  config.local_prefill_decode_bucket_permyriad = 10001;
  EXPECT_FALSE(select_native_execution_mode(
                   config, NativeExecutionModeInput{}, &decision)
                   .ok());
  EXPECT_FALSE(
      select_native_execution_mode(config, NativeExecutionModeInput{}, nullptr)
          .ok());
}

}  // namespace
}  // namespace xllm_service::provider
