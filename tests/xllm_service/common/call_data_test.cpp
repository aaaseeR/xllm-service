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

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
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

class BlockingStreamSink : public StreamOutputSink {
 public:
  int write(const butil::IOBuf& attachment) override {
    size_t call_index = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      call_index = ++write_calls_;
      ++active_writes_;
      peak_active_writes_ = std::max(peak_active_writes_, active_writes_);
      if (call_index == 1) {
        first_write_entered_ = true;
      }
      condition_.notify_all();
    }

    if (call_index == 1) {
      std::unique_lock<std::mutex> lock(mutex_);
      condition_.wait(lock, [this] { return release_first_write_; });
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      chunks_.push_back(attachment.to_string());
      --active_writes_;
      condition_.notify_all();
    }
    return 0;
  }

  void notify_on_stopped(google::protobuf::Closure* callback) override {
    stopped_callback_ = callback;
  }

  void wait_for_first_write() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return first_write_entered_; });
  }

  bool wait_for_concurrent_write(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return condition_.wait_for(
        lock, timeout, [this] { return peak_active_writes_ > 1; });
  }

  void release_first_write() {
    std::lock_guard<std::mutex> lock(mutex_);
    release_first_write_ = true;
    condition_.notify_all();
  }

  size_t peak_active_writes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return peak_active_writes_;
  }

  std::string output() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string result;
    for (const auto& chunk : chunks_) {
      result.append(chunk);
    }
    return result;
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  size_t write_calls_ = 0;
  size_t active_writes_ = 0;
  size_t peak_active_writes_ = 0;
  bool first_write_entered_ = false;
  bool release_first_write_ = false;
  std::vector<std::string> chunks_;
  google::protobuf::Closure* stopped_callback_ = nullptr;
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

TEST(CallDataTest, NonStreamFirstTerminalResponseWins) {
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

    ASSERT_TRUE(call_data.finish_with_error("deadline exceeded"));
    EXPECT_FALSE(call_data.write_and_finish("{\"status\":\"late\"}"));
    EXPECT_TRUE(controller.Failed());
    EXPECT_EQ(controller.ErrorText(), "deadline exceeded");
    EXPECT_TRUE(controller.response_attachment().empty());
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

TEST(CallDataTest, StreamTerminalWaitsForInFlightPayloadWrite) {
  brpc::Controller controller;
  xllm::proto::ChatRequest request;
  xllm::proto::ChatResponse response;
  TestClosure done;
  auto sink = std::make_shared<BlockingStreamSink>();

  ChatCallData call_data(&controller,
                         /*stream=*/true,
                         &done,
                         &request,
                         &response,
                         /*trace_callback=*/nullptr,
                         sink);

  bool payload_result = false;
  std::thread payload_thread([&] {
    payload_result = call_data.write("data: {\"token\":\"first\"}\n\n");
  });
  sink->wait_for_first_write();

  bool terminal_result = false;
  std::thread terminal_thread([&] {
    terminal_result = call_data.finish_with_error("deadline exceeded");
  });
  const auto terminal_claim_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (!call_data.finished() &&
         std::chrono::steady_clock::now() < terminal_claim_deadline) {
    std::this_thread::yield();
  }

  bool late_payload_result = true;
  std::thread late_payload_thread([&] {
    late_payload_result = call_data.write("data: {\"token\":\"late\"}\n\n");
  });
  EXPECT_FALSE(sink->wait_for_concurrent_write(std::chrono::milliseconds(50)));
  sink->release_first_write();
  payload_thread.join();
  terminal_thread.join();
  late_payload_thread.join();

  EXPECT_TRUE(call_data.finished());
  EXPECT_TRUE(payload_result);
  EXPECT_FALSE(late_payload_result);
  EXPECT_TRUE(terminal_result);
  EXPECT_EQ(sink->peak_active_writes(), 1);
  const std::string output = sink->output();
  EXPECT_LT(output.find("first"), output.find("deadline exceeded"));
  EXPECT_EQ(output.find("late"), std::string::npos);
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
