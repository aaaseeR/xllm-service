/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "rpc_service/first_event_recovery_client.h"

#include <brpc/controller.h>

#include <limits>
#include <string>
#include <utility>

#include "disagg_pd.pb.h"
#include "rpc_service/disagg_generation_adapter.h"

namespace xllm_service {
namespace {

class QueryContext final {
 public:
  xllm::proto::AttemptControlRequest request;
  xllm::proto::AttemptControlResponse response;
  brpc::Controller controller;
  bool issued = false;
};

llm::Status invalid_query(std::string message) {
  return llm::Status(llm::StatusCode::INVALID_ARGUMENT, std::move(message));
}

llm::Status validate_query(const FirstEventRecoveryQuery& query) {
  if (query.channel == nullptr) {
    return llm::Status(llm::StatusCode::UNAVAILABLE,
                       "Decode QueryRequest channel is unavailable");
  }
  if (query.timeout_ms == 0) {
    return llm::Status(llm::StatusCode::DEADLINE_EXCEEDED,
                       "Decode QueryRequest has no remaining time");
  }
  if (query.timeout_ms >
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return invalid_query("Decode QueryRequest timeout exceeds int32 range");
  }
  if (query.max_payload_bytes == 0 || query.attempt.request_uid().empty() ||
      !query.attempt.has_attempt_seq() || query.decode.engine_uid().empty() ||
      query.decode.incarnation_id().empty() ||
      query.prefill.engine_uid().empty() ||
      query.prefill.incarnation_id().empty()) {
    return invalid_query("Decode QueryRequest identity is incomplete");
  }
  return llm::Status();
}

}  // namespace

std::vector<FirstEventRecoveryResult> query_first_output_events(
    const std::vector<FirstEventRecoveryQuery>& queries) {
  std::vector<std::unique_ptr<QueryContext>> contexts;
  contexts.reserve(queries.size());
  std::vector<FirstEventRecoveryResult> results(queries.size());

  for (size_t index = 0; index < queries.size(); ++index) {
    const FirstEventRecoveryQuery& query = queries[index];
    results[index].status = validate_query(query);

    auto context = std::make_unique<QueryContext>();
    if (results[index].status.ok()) {
      xllm::proto::RequestAttemptKey* key = context->request.mutable_key();
      key->set_request_uid(query.attempt.request_uid());
      key->set_attempt_seq(query.attempt.attempt_seq());
      key->set_incarnation_id(query.decode.incarnation_id());
      context->controller.set_timeout_ms(
          static_cast<int32_t>(query.timeout_ms));

      xllm::proto::DisaggPDService_Stub stub(query.channel.get());
      stub.QueryRequest(&context->controller,
                        &context->request,
                        &context->response,
                        brpc::DoNothing());
      context->issued = true;
    }
    contexts.emplace_back(std::move(context));
  }

  for (const std::unique_ptr<QueryContext>& context : contexts) {
    if (context->issued) {
      brpc::Join(context->controller.call_id());
    }
  }

  for (size_t index = 0; index < queries.size(); ++index) {
    const FirstEventRecoveryQuery& query = queries[index];
    const QueryContext& context = *contexts[index];
    if (!context.issued) {
      continue;
    }
    if (context.controller.Failed()) {
      const bool timed_out =
          context.controller.ErrorCode() == brpc::ERPCTIMEDOUT;
      results[index].status =
          llm::Status(timed_out ? llm::StatusCode::DEADLINE_EXCEEDED
                                : llm::StatusCode::UNAVAILABLE,
                      std::string("Decode QueryRequest failed: ") +
                          context.controller.ErrorText());
      continue;
    }

    RequestOutputConversionResult conversion =
        first_output_from_query_response(context.response,
                                         query.attempt,
                                         query.decode,
                                         query.prefill,
                                         query.max_payload_bytes);
    results[index].status = std::move(conversion.status);
    results[index].output = std::move(conversion.output);
  }
  return results;
}

}  // namespace xllm_service
