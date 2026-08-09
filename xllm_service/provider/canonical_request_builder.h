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
#include <optional>
#include <string>

#include "observability.pb.h"
#include "provider.pb.h"
#include "provider/provider_contract.h"

namespace xllm_service::provider {

inline constexpr char kOpenAiHttpJsonSchema[] = "openai.http.json.v1";
inline constexpr char kAnthropicHttpJsonSchema[] = "anthropic.http.json.v1";

struct CanonicalRequestInput {
  xllm::proto::RequestCorrelation correlation;
  xllm::proto::ApiKind api_kind = xllm::proto::API_KIND_UNSPECIFIED;
  std::string model_revision;
  std::string payload_schema;
  std::string payload;
  uint64_t effective_max_new_tokens = 0;
  uint32_t n = 0;
  uint32_t best_of = 0;
  int32_t priority = 0;
  uint64_t remaining_deadline_ms = 0;
  std::optional<uint64_t> ttft_slo_ms;
  std::optional<uint64_t> tpot_slo_ms;
  bool strict = true;
};

ContractResult build_canonical_request(const CanonicalRequestInput& input,
                                       xllm::proto::CanonicalRequest* request);

}  // namespace xllm_service::provider
