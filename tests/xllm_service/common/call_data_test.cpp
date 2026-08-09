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

#include "common/call_data.h"

#include <brpc/controller.h>
#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

namespace xllm_service {
namespace {

class TestClosure : public google::protobuf::Closure {
 public:
  void Run() override { ++run_count; }
  int run_count = 0;
};

class MemoryStreamSink : public StreamOutputSink {
 public:
  int write(const butil::IOBuf& attachment) override {
    if (write_status != 0) {
      return write_status;
    }
    chunks.push_back(attachment.to_string());
    return 0;
  }

  void notify_on_stopped(google::protobuf::Closure* callback) override {
    stopped_callback = callback;
  }

  int write_status = 0;
  std::vector<std::string> chunks;
  google::protobuf::Closure* stopped_callback = nullptr;
};

std::string join_chunks(const MemoryStreamSink& sink) {
  std::string result;
  for (const auto& chunk : sink.chunks) {
    result.append(chunk);
  }
  return result;
}

size_t count_substrings(const std::string& text, const std::string& needle) {
  size_t count = 0;
  size_t offset = 0;
  while ((offset = text.find(needle, offset)) != std::string::npos) {
    ++count;
    offset += needle.size();
  }
  return count;
}

TEST(CallDataTest, NonStreamWriteAndFinishTracesAttachment) {
  brpc::Controller controller;
  xllm::proto::ChatRequest request;
  xllm::proto::ChatResponse response;
  TestClosure done;
  std::vector<std::string> trace_chunks;

  {
    ChatCallData call_data(
        &controller,
        /*stream=*/false,
        &done,
        &request,
        &response,
        [&](const std::string& chunk) { trace_chunks.push_back(chunk); });

    ASSERT_EQ(done.run_count, 0);
    ASSERT_TRUE(call_data.write_and_finish("{\"id\":\"chatcmpl-test\"}"));

    ASSERT_EQ(trace_chunks.size(), 1);
    EXPECT_EQ(trace_chunks[0], "{\"id\":\"chatcmpl-test\"}");
    EXPECT_EQ(controller.response_attachment().to_string(),
              "{\"id\":\"chatcmpl-test\"}");
  }

  EXPECT_EQ(done.run_count, 1);
}

TEST(CallDataTest, NonStreamFinishWithErrorMarksControllerFailed) {
  brpc::Controller controller;
  xllm::proto::ChatRequest request;
  xllm::proto::ChatResponse response;
  TestClosure done;

  {
    ChatCallData call_data(&controller,
                           /*stream=*/false,
                           &done,
                           &request,
                           &response);

    ASSERT_TRUE(call_data.finish_with_error("encode failed"));
    EXPECT_TRUE(controller.Failed());
    EXPECT_EQ(controller.ErrorText(), "encode failed");
  }

  EXPECT_EQ(done.run_count, 1);
}

TEST(CallDataTest, OpenAIStreamWritesOneExplicitTerminalFrame) {
  brpc::Controller controller;
  xllm::proto::ChatRequest request;
  xllm::proto::ChatResponse response;
  TestClosure done;
  auto sink = std::make_shared<MemoryStreamSink>();

  ChatCallData call_data(&controller,
                         /*stream=*/true,
                         &done,
                         &request,
                         &response,
                         /*trace_callback=*/nullptr,
                         sink);

  EXPECT_EQ(done.run_count, 1);
  EXPECT_TRUE(call_data.write(
      "data: {\"text\":\"payload contains data: [DONE]\"}\n\n"));
  EXPECT_FALSE(call_data.finished());
  EXPECT_TRUE(call_data.finish());
  EXPECT_TRUE(call_data.finish());
  EXPECT_FALSE(call_data.write("data: late\n\n"));

  const std::string output = join_chunks(*sink);
  EXPECT_EQ(count_substrings(output, "data: [DONE]\n\n"), 1);
  EXPECT_NE(output.find("payload contains data: [DONE]"), std::string::npos);
}

TEST(CallDataTest, OpenAIStreamErrorIsEscapedAndTerminatesOnce) {
  brpc::Controller controller;
  xllm::proto::ChatRequest request;
  xllm::proto::ChatResponse response;
  TestClosure done;
  auto sink = std::make_shared<MemoryStreamSink>();

  ChatCallData call_data(&controller,
                         /*stream=*/true,
                         &done,
                         &request,
                         &response,
                         /*trace_callback=*/nullptr,
                         sink);
  EXPECT_TRUE(call_data.finish_with_error("bad \"request\"\nline"));
  EXPECT_TRUE(call_data.finish());

  const std::string output = join_chunks(*sink);
  EXPECT_NE(output.find("data: {\"error\":"), std::string::npos);
  EXPECT_NE(output.find("bad \\\"request\\\"\\nline"), std::string::npos);
  EXPECT_EQ(count_substrings(output, "data: [DONE]\n\n"), 1);
}

TEST(CallDataTest, AnthropicStreamDoesNotEmitOpenAITerminalFrame) {
  brpc::Controller controller;
  xllm::proto::ChatRequest request;
  xllm::proto::AnthropicMessagesResponse response;
  TestClosure done;
  auto sink = std::make_shared<MemoryStreamSink>();

  AnthropicCallData call_data(&controller,
                              /*stream=*/true,
                              &done,
                              &request,
                              &response,
                              /*trace_callback=*/nullptr,
                              sink);
  EXPECT_TRUE(call_data.write(
      "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"));
  EXPECT_TRUE(call_data.finish());

  const std::string output = join_chunks(*sink);
  EXPECT_EQ(output.find("[DONE]"), std::string::npos);
  EXPECT_NE(output.find("event: message_stop"), std::string::npos);
}

TEST(CallDataTest, AnthropicStreamErrorUsesAnthropicErrorEvent) {
  brpc::Controller controller;
  xllm::proto::ChatRequest request;
  xllm::proto::AnthropicMessagesResponse response;
  TestClosure done;
  auto sink = std::make_shared<MemoryStreamSink>();

  AnthropicCallData call_data(&controller,
                              /*stream=*/true,
                              &done,
                              &request,
                              &response,
                              /*trace_callback=*/nullptr,
                              sink);
  EXPECT_TRUE(call_data.finish_with_error("provider failed"));

  const std::string output = join_chunks(*sink);
  EXPECT_NE(output.find("event: error\n"), std::string::npos);
  EXPECT_NE(output.find("\"type\":\"api_error\""), std::string::npos);
  EXPECT_EQ(output.find("[DONE]"), std::string::npos);
}

TEST(CallDataTest, MissingStreamSinkFailsWithoutDereference) {
  brpc::Controller controller;
  xllm::proto::ChatRequest request;
  xllm::proto::ChatResponse response;
  TestClosure done;

  ChatCallData call_data(&controller,
                         /*stream=*/true,
                         &done,
                         &request,
                         &response,
                         /*trace_callback=*/nullptr,
                         std::shared_ptr<StreamOutputSink>(),
                         StreamProtocol::kOpenAI,
                         /*create_progressive_attachment=*/false);

  // If attachment creation fails, construction still completes, done runs
  // once, and writes fail safely.
  EXPECT_EQ(done.run_count, 1);
  EXPECT_TRUE(controller.Failed());
  EXPECT_TRUE(call_data.is_disconnected());
  EXPECT_FALSE(call_data.write("data: unavailable\n\n"));
}

}  // namespace
}  // namespace xllm_service
