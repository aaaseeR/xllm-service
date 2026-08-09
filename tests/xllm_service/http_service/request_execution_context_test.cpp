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

#include "http_service/request_execution_context.h"

#include <gtest/gtest.h>

#include "chat.pb.h"
#include "completion.pb.h"

namespace xllm_service {
namespace {

void initialize_request_context(Request* request) {
  request->correlation.set_global_request_id("global-request");
  request->correlation.set_trace_id("trace-id");
  request->correlation.set_request_uid("request-uid");
  request->correlation.set_attempt_seq(0);
  request->request_deadline = xllm::RequestDeadline::from_remaining_ms(10000);
  request->first_event_retry_policy =
      xllm::FirstEventRetryPolicy::from_durations_ms(700, 200);
}

TEST(RequestExecutionContextTest, CopiesStrictPlanIntoCompletionRequest) {
  Request request;
  initialize_request_context(&request);
  request.execution_plan.emplace();
  request.execution_plan->set_contract_version(1);
  request.execution_plan->set_request_uid("request-uid");
  request.execution_plan->set_attempt_seq(0);
  xllm::proto::CompletionRequest request_pb;

  ASSERT_TRUE(set_request_execution_context(&request_pb, request));
  EXPECT_EQ(request_pb.service_request_id(), "request-uid");
  EXPECT_EQ(request_pb.correlation().request_uid(), "request-uid");
  ASSERT_TRUE(request_pb.has_execution_plan());
  EXPECT_EQ(request_pb.execution_plan().contract_version(), 1u);
  EXPECT_TRUE(request_pb.execution_plan().has_attempt_seq());
  EXPECT_EQ(request_pb.execution_plan().attempt_seq(), 0u);
  EXPECT_GT(request_pb.remaining_deadline_ms(), 0u);
  EXPECT_LE(request_pb.remaining_deadline_ms(), 10000u);
}

TEST(RequestExecutionContextTest, ClearsUntrustedPlanOnLegacyChatRequest) {
  Request request;
  initialize_request_context(&request);
  xllm::proto::ChatRequest request_pb;
  request_pb.mutable_execution_plan()->set_request_uid("client-plan");

  ASSERT_TRUE(set_request_execution_context(&request_pb, request));
  EXPECT_FALSE(request_pb.has_execution_plan());
}

TEST(RequestExecutionContextTest, RejectsMissingRetryPolicy) {
  Request request;
  initialize_request_context(&request);
  request.first_event_retry_policy.reset();
  xllm::proto::CompletionRequest request_pb;

  EXPECT_FALSE(set_request_execution_context(&request_pb, request));
}

}  // namespace
}  // namespace xllm_service
