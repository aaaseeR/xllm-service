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

#include "provider/canonical_request_builder.h"

namespace xllm_service::provider {

ContractResult build_canonical_request(const CanonicalRequestInput& input,
                                       xllm::proto::CanonicalRequest* request) {
  if (request == nullptr) {
    return ContractResult::failure(
        xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD,
        "canonical request output must not be null");
  }

  request->Clear();
  request->set_contract_version(kProviderContractVersion);
  request->set_global_request_id(input.correlation.global_request_id());
  request->set_trace_id(input.correlation.trace_id());
  request->set_request_uid(input.correlation.request_uid());
  if (input.correlation.has_attempt_seq()) {
    request->set_attempt_seq(input.correlation.attempt_seq());
  }
  request->set_api_kind(input.api_kind);
  request->set_model_revision(input.model_revision);
  request->set_strict(input.strict);
  request->set_canonical_payload_schema(input.payload_schema);
  request->set_canonical_payload(input.payload);
  request->set_effective_max_new_tokens(input.effective_max_new_tokens);
  request->set_n(input.n);
  request->set_best_of(input.best_of);
  request->set_priority(input.priority);
  request->set_remaining_deadline_ms(input.remaining_deadline_ms);
  if (input.ttft_slo_ms.has_value()) {
    request->set_ttft_slo_ms(*input.ttft_slo_ms);
  }
  if (input.tpot_slo_ms.has_value()) {
    request->set_tpot_slo_ms(*input.tpot_slo_ms);
  }
  return validate_canonical_request(*request);
}

}  // namespace xllm_service::provider
