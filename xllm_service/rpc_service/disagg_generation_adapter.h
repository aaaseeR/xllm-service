/* Copyright 2025-2026 The xLLM Authors.

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

#include <cstddef>
#include <optional>

#include "common/xllm/output.h"
#include "common/xllm/status.h"
#include "disagg_pd.pb.h"
#include "provider.pb.h"
#include "xllm_rpc_service.pb.h"

namespace xllm_service {

struct RequestOutputConversionResult {
  llm::Status status;
  std::optional<llm::RequestOutput> output;
};

RequestOutputConversionResult request_output_from_disagg_generation(
    const proto::DisaggStreamGeneration& generation);

// Validates a D Query response and reconstructs the exact P-side seq=0 event
// retained at GenerationCommit. No token decoding or parser replay occurs.
RequestOutputConversionResult first_output_from_query_response(
    const xllm::proto::AttemptControlResponse& response,
    const xllm::proto::ExecutionAttemptId& expected_attempt,
    const xllm::proto::ExecutionHolder& expected_decode,
    const xllm::proto::ExecutionHolder& expected_prefill,
    size_t max_payload_bytes);

}  // namespace xllm_service
