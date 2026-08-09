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

#include "provider.pb.h"
#include "provider/provider_contract.h"

namespace xllm_service::provider {

// Builds an execution shape opened by the immutable Provider Descriptor. An
// unspecified requested mode preserves the Provider's default open mode.
ContractResult build_execution_plan(
    const xllm::proto::CanonicalRequest& canonical,
    const xllm::proto::EncodedRequest& encoded,
    const xllm::proto::ProviderDescriptor& primary,
    const xllm::proto::ProviderDescriptor* decode,
    xllm::proto::ExecutionPlan* plan,
    xllm::proto::ExecutionMode requested_mode =
        xllm::proto::EXECUTION_MODE_UNSPECIFIED);

}  // namespace xllm_service::provider
