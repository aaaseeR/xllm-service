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

#include <brpc/controller.h>
#include <brpc/http_status_code.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common/call_data.h"
#include "common/options.h"
#include "completion.pb.h"
#include "http_service/service.h"
#include "runtime/runtime_state.h"
#include "scheduler/schedule_result.h"
#include "xllm_http_service.pb.h"

namespace xllm_service {
namespace {

class HealthServiceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    options_.num_threads(1);
    service_ = std::make_unique<XllmHttpServiceImpl>(
        options_, /*scheduler=*/nullptr, runtime_state_);
  }

  Options options_;
  RuntimeState runtime_state_;
  std::unique_ptr<XllmHttpServiceImpl> service_;
};

class RecordingClosure final : public google::protobuf::Closure {
 public:
  explicit RecordingClosure(bool* called) : called_(called) {}

  void Run() override {
    *called_ = true;
    delete this;
  }

 private:
  bool* called_;
};

TEST_F(HealthServiceTest, StartingRuntimeIsLiveButNotReady) {
  proto::HttpRequest request;
  proto::HttpResponse response;
  brpc::Controller liveness_controller;
  brpc::Controller readiness_controller;

  service_->Liveness(
      &liveness_controller, &request, &response, /*done=*/nullptr);
  service_->Readiness(
      &readiness_controller, &request, &response, /*done=*/nullptr);

  EXPECT_EQ(liveness_controller.http_response().status_code(),
            brpc::HTTP_STATUS_OK);
  EXPECT_EQ(liveness_controller.response_attachment().to_string(), "ok\n");
  EXPECT_EQ(readiness_controller.http_response().status_code(),
            brpc::HTTP_STATUS_SERVICE_UNAVAILABLE);
  EXPECT_EQ(readiness_controller.response_attachment().to_string(),
            "starting\n");
}

TEST_F(HealthServiceTest, ReadyRuntimeSupportsCompatibilityHealthEndpoint) {
  runtime_state_.set_backend_ready(true);
  runtime_state_.mark_running();

  proto::HttpRequest request;
  proto::HttpResponse response;
  brpc::Controller health_controller;
  brpc::Controller readiness_controller;

  service_->Health(&health_controller, &request, &response, /*done=*/nullptr);
  service_->Readiness(
      &readiness_controller, &request, &response, /*done=*/nullptr);

  EXPECT_EQ(health_controller.http_response().status_code(),
            brpc::HTTP_STATUS_OK);
  EXPECT_EQ(health_controller.response_attachment().to_string(), "ok\n");
  EXPECT_EQ(readiness_controller.http_response().status_code(),
            brpc::HTTP_STATUS_OK);
  EXPECT_EQ(readiness_controller.response_attachment().to_string(), "ready\n");
}

TEST_F(HealthServiceTest, DrainingRuntimeRejectsNewInferenceRequests) {
  runtime_state_.set_backend_ready(true);
  runtime_state_.mark_running();
  runtime_state_.begin_draining();

  proto::HttpRequest request;
  proto::HttpResponse response;
  brpc::Controller controller;
  controller.request_attachment().append(
      R"({"model":"test","prompt":"hello"})");

  service_->Completions(&controller, &request, &response, /*done=*/nullptr);

  EXPECT_EQ(controller.http_response().status_code(),
            brpc::HTTP_STATUS_SERVICE_UNAVAILABLE);
  const std::string body = controller.response_attachment().to_string();
  EXPECT_NE(body.find("service_unavailable"), std::string::npos);
  EXPECT_NE(body.find("draining"), std::string::npos);
}

TEST(CallDataTest, StreamingRetryableFailureCommits503BeforeFirstToken) {
  brpc::Controller controller;
  xllm::proto::CompletionRequest request;
  xllm::proto::CompletionResponse response;
  bool done_called = false;
  auto call_data =
      std::make_unique<CompletionCallData>(&controller,
                                           /*stream=*/true,
                                           new RecordingClosure(&done_called),
                                           &request,
                                           &response);

  EXPECT_FALSE(done_called);
  EXPECT_TRUE(
      call_data->finish_with_error("stale routing decision",
                                   brpc::HTTP_STATUS_SERVICE_UNAVAILABLE,
                                   "stale_routing_decision",
                                   /*retryable=*/true));

  EXPECT_TRUE(done_called);
  EXPECT_EQ(controller.http_response().status_code(),
            brpc::HTTP_STATUS_SERVICE_UNAVAILABLE);
  const auto* error_code =
      controller.http_response().GetHeader("x-llm-d-error-code");
  ASSERT_NE(error_code, nullptr);
  EXPECT_EQ(*error_code, "stale_routing_decision");
  const auto* retryable =
      controller.http_response().GetHeader("x-llm-d-retryable");
  ASSERT_NE(retryable, nullptr);
  EXPECT_EQ(*retryable, "true");
  EXPECT_EQ(controller.response_attachment().to_string(),
            "stale routing decision");
}

TEST(ScheduleFailureTest, StaleDecisionIsExplicitlyRetryable) {
  brpc::Controller controller;
  const ScheduleResult result = {ScheduleError::STALE_ROUTING_DECISION,
                                 "routing decision is stale"};

  apply_schedule_failure(&controller, result);

  EXPECT_EQ(controller.http_response().status_code(),
            brpc::HTTP_STATUS_SERVICE_UNAVAILABLE);
  const auto* error_code =
      controller.http_response().GetHeader("x-llm-d-error-code");
  ASSERT_NE(error_code, nullptr);
  EXPECT_EQ(*error_code, "stale_routing_decision");
  const auto* retryable =
      controller.http_response().GetHeader("x-llm-d-retryable");
  ASSERT_NE(retryable, nullptr);
  EXPECT_EQ(*retryable, "true");
  const auto* retry_after = controller.http_response().GetHeader("Retry-After");
  ASSERT_NE(retry_after, nullptr);
  EXPECT_EQ(*retry_after, "0");
}

TEST(ScheduleFailureTest, InvalidDirectiveIsNotRetryable) {
  brpc::Controller controller;
  const ScheduleResult result = {ScheduleError::INVALID_ROUTING_DIRECTIVE,
                                 "routing directive is invalid"};

  apply_schedule_failure(&controller, result);

  EXPECT_EQ(controller.http_response().status_code(),
            brpc::HTTP_STATUS_BAD_REQUEST);
  const auto* error_code =
      controller.http_response().GetHeader("x-llm-d-error-code");
  ASSERT_NE(error_code, nullptr);
  EXPECT_EQ(*error_code, "invalid_routing_directive");
  EXPECT_EQ(controller.http_response().GetHeader("x-llm-d-retryable"), nullptr);
}

}  // namespace
}  // namespace xllm_service
