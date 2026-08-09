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

#include "provider/attempt_control_client.h"

#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <brpc/server.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "xllm_http_service.pb.h"

namespace xllm_service::provider {
namespace {

class FakeAgentService final : public proto::XllmHttpService {
 public:
  void Models(google::protobuf::RpcController* controller,
              const proto::HttpRequest*,
              proto::HttpResponse*,
              google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    auto* brpc_controller = static_cast<brpc::Controller*>(controller);
    last_path_ = brpc_controller->http_request().uri().path();
    last_body_ = brpc_controller->request_attachment().to_string();
    brpc_controller->http_response().set_status_code(200);
    if (last_path_.find("/query") != std::string::npos) {
      brpc_controller->response_attachment().append(
          R"({"accepted":true,"request_uid":"request-1","attempt_seq":7,"incarnation_id":"incarnation-1","state":"ATTEMPT_LIFECYCLE_STATE_DONE"})");
    } else {
      brpc_controller->response_attachment().append(
          R"({"accepted":true,"request_uid":"request-1","attempt_seq":7,"incarnation_id":"incarnation-1","state":"ATTEMPT_LIFECYCLE_STATE_CANCELLED_BEFORE_CREATE"})");
    }
  }

  const std::string& last_path() const { return last_path_; }
  const std::string& last_body() const { return last_body_; }

 private:
  std::string last_path_;
  std::string last_body_;
};

std::string agent_response(const std::string& state,
                           const std::string& request_uid = "request-1",
                           uint64_t attempt_seq = 7,
                           const std::string& incarnation_id = "incarnation-1",
                           bool accepted = true) {
  return "{\"accepted\":" + std::string(accepted ? "true" : "false") +
         ",\"request_uid\":\"" + request_uid +
         "\",\"attempt_seq\":" + std::to_string(attempt_seq) +
         ",\"incarnation_id\":\"" + incarnation_id + "\",\"state\":\"" + state +
         "\"}";
}

class AttemptControlClientLoopbackTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(server_.AddService(&service_,
                                 brpc::SERVER_DOESNT_OWN_SERVICE,
                                 "/v1/internal/attempt/query => Models,"
                                 "/v1/internal/attempt/cancel => Models"),
              0);
    ASSERT_EQ(server_.Start("127.0.0.1:0", nullptr), 0);
    channel_ = std::make_shared<brpc::Channel>();
    brpc::ChannelOptions options;
    options.protocol = "http";
    ASSERT_EQ(
        channel_->Init("127.0.0.1", server_.listen_address().port, &options),
        0);

    hold_.mutable_attempt()->set_request_uid("request-1");
    hold_.mutable_attempt()->set_attempt_seq(7);
    holder_.set_engine_uid("127.0.0.1:" +
                           std::to_string(server_.listen_address().port));
    holder_.set_incarnation_id("incarnation-1");
  }

  void TearDown() override {
    ASSERT_EQ(server_.Stop(0), 0);
    ASSERT_EQ(server_.Join(), 0);
  }

  FakeAgentService service_;
  brpc::Server server_;
  std::shared_ptr<brpc::Channel> channel_;
  xllm::proto::ExecutionResourceHold hold_;
  xllm::proto::ExecutionHolder holder_;
};

TEST(AttemptControlClientTest, AcceptsOnlyTerminalAgentStatesAsProof) {
  const std::vector<std::string> terminal_states = {
      "ATTEMPT_LIFECYCLE_STATE_DONE",
      "ATTEMPT_LIFECYCLE_STATE_CANCELLED",
      "ATTEMPT_LIFECYCLE_STATE_EXPIRED",
      "ATTEMPT_LIFECYCLE_STATE_FAILED",
      "ATTEMPT_LIFECYCLE_STATE_CANCELLED_BEFORE_CREATE",
  };
  for (const std::string& state : terminal_states) {
    const AttemptControlResult result = parse_vllm_agent_attempt_response(
        200, agent_response(state), "request-1", 7, "incarnation-1");
    EXPECT_TRUE(result.direct_success) << state;
    EXPECT_TRUE(result.terminal_proof) << state;
    EXPECT_NE(result.state, xllm::proto::ATTEMPT_LIFECYCLE_STATE_UNSPECIFIED)
        << state;
  }

  const std::vector<std::string> nonterminal_states = {
      "ATTEMPT_LIFECYCLE_STATE_ABSENT",
      "ATTEMPT_LIFECYCLE_STATE_RESERVED",
      "ATTEMPT_LIFECYCLE_STATE_RECEIVING",
      "ATTEMPT_LIFECYCLE_STATE_GENERATION_COMMITTED",
      "ATTEMPT_LIFECYCLE_STATE_RUNNING",
      "ATTEMPT_LIFECYCLE_STATE_LOCAL_GENERATION_COMMITTED",
  };
  for (const std::string& state : nonterminal_states) {
    const AttemptControlResult result = parse_vllm_agent_attempt_response(
        200, agent_response(state), "request-1", 7, "incarnation-1");
    EXPECT_TRUE(result.direct_success) << state;
    EXPECT_FALSE(result.terminal_proof) << state;
  }
}

TEST(AttemptControlClientTest, MalformedOrFailedResponseNeverProvesTerminal) {
  const std::vector<std::string> invalid_bodies = {
      "",
      "[]",
      "{}",
      "{\"state\":1}",
      R"({"request_uid":"request-1","attempt_seq":7,"incarnation_id":"incarnation-1","state":"ATTEMPT_LIFECYCLE_STATE_DONE"})",
      "{\"state\":\"ATTEMPT_LIFECYCLE_STATE_NEW_UNKNOWN\"}",
  };
  for (const std::string& body : invalid_bodies) {
    const AttemptControlResult result = parse_vllm_agent_attempt_response(
        200, body, "request-1", 7, "incarnation-1");
    EXPECT_FALSE(result.direct_success) << body;
    EXPECT_FALSE(result.terminal_proof) << body;
    EXPECT_EQ(result.state, xllm::proto::ATTEMPT_LIFECYCLE_STATE_UNSPECIFIED)
        << body;
  }

  const AttemptControlResult failed = parse_vllm_agent_attempt_response(
      409,
      agent_response("ATTEMPT_LIFECYCLE_STATE_DONE"),
      "request-1",
      7,
      "incarnation-1");
  EXPECT_FALSE(failed.terminal_proof);
  EXPECT_FALSE(failed.direct_success);
  EXPECT_EQ(failed.state, xllm::proto::ATTEMPT_LIFECYCLE_STATE_UNSPECIFIED);

  const std::vector<std::string> mismatched_bodies = {
      agent_response("ATTEMPT_LIFECYCLE_STATE_DONE", "other-request"),
      agent_response("ATTEMPT_LIFECYCLE_STATE_DONE", "request-1", 8),
      agent_response(
          "ATTEMPT_LIFECYCLE_STATE_DONE", "request-1", 7, "other-incarnation"),
  };
  for (const std::string& body : mismatched_bodies) {
    const AttemptControlResult mismatch = parse_vllm_agent_attempt_response(
        200, body, "request-1", 7, "incarnation-1");
    EXPECT_FALSE(mismatch.direct_success) << body;
    EXPECT_FALSE(mismatch.terminal_proof) << body;
    EXPECT_EQ(mismatch.state, xllm::proto::ATTEMPT_LIFECYCLE_STATE_UNSPECIFIED)
        << body;
  }
}

TEST(AttemptControlClientTest, RejectedCancelCannotProveFenceInstallation) {
  const AttemptControlResult rejected = parse_vllm_agent_attempt_response(
      200,
      agent_response("ATTEMPT_LIFECYCLE_STATE_FAILED",
                     "request-1",
                     7,
                     "incarnation-1",
                     false),
      "request-1",
      7,
      "incarnation-1");
  EXPECT_TRUE(rejected.direct_success);
  EXPECT_FALSE(rejected.terminal_proof);
  EXPECT_EQ(rejected.state, xllm::proto::ATTEMPT_LIFECYCLE_STATE_FAILED);
}

TEST(AttemptControlClientTest, BindsSubmitHeadersToExactIncarnation) {
  brpc::Controller controller;
  ASSERT_TRUE(set_vllm_agent_attempt_headers(
      &controller, "request-1", 7, "incarnation-1", 1234));
  ASSERT_NE(controller.http_request().GetHeader("X-Request-UID"), nullptr);
  EXPECT_EQ(*controller.http_request().GetHeader("X-Request-UID"), "request-1");
  ASSERT_NE(controller.http_request().GetHeader("X-Attempt-Seq"), nullptr);
  EXPECT_EQ(*controller.http_request().GetHeader("X-Attempt-Seq"), "7");
  ASSERT_NE(controller.http_request().GetHeader("X-Incarnation-ID"), nullptr);
  EXPECT_EQ(*controller.http_request().GetHeader("X-Incarnation-ID"),
            "incarnation-1");
  ASSERT_NE(controller.http_request().GetHeader("X-Remaining-Deadline-Ms"),
            nullptr);
  EXPECT_EQ(*controller.http_request().GetHeader("X-Remaining-Deadline-Ms"),
            "1234");

  EXPECT_FALSE(set_vllm_agent_attempt_headers(
      nullptr, "request-1", 7, "incarnation-1", 1234));
  EXPECT_FALSE(set_vllm_agent_attempt_headers(
      &controller, "", 7, "incarnation-1", 1234));
  EXPECT_FALSE(
      set_vllm_agent_attempt_headers(&controller, "request-1", 7, "", 1234));
  EXPECT_FALSE(set_vllm_agent_attempt_headers(
      &controller, "request-1", 7, "incarnation-1", 0));
}

TEST_F(AttemptControlClientLoopbackTest, CallsAgentQueryAndCancelEndpoints) {
  const AttemptControlResult query =
      call_provider_attempt_control(xllm::proto::PROVIDER_ID_VLLM_ASCEND,
                                    channel_,
                                    hold_,
                                    holder_,
                                    AttemptControlOperation::QUERY,
                                    200);
  EXPECT_TRUE(query.direct_success);
  EXPECT_TRUE(query.terminal_proof);
  EXPECT_EQ(query.state, xllm::proto::ATTEMPT_LIFECYCLE_STATE_DONE);
  EXPECT_EQ(service_.last_path(), "/v1/internal/attempt/query");
  EXPECT_NE(service_.last_body().find(R"("request_uid":"request-1")"),
            std::string::npos);
  EXPECT_NE(service_.last_body().find(R"("attempt_seq":7)"), std::string::npos);
  EXPECT_NE(service_.last_body().find(R"("incarnation_id":"incarnation-1")"),
            std::string::npos);

  const AttemptControlResult cancel =
      call_provider_attempt_control(xllm::proto::PROVIDER_ID_VLLM_ASCEND,
                                    channel_,
                                    hold_,
                                    holder_,
                                    AttemptControlOperation::CANCEL,
                                    200);
  EXPECT_TRUE(cancel.direct_success);
  EXPECT_TRUE(cancel.terminal_proof);
  EXPECT_EQ(cancel.state,
            xllm::proto::ATTEMPT_LIFECYCLE_STATE_CANCELLED_BEFORE_CREATE);
  EXPECT_EQ(service_.last_path(), "/v1/internal/attempt/cancel");
}

}  // namespace
}  // namespace xllm_service::provider
