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

#include "execution/request_session_registry.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace xllm_service {
namespace {

using namespace std::chrono_literals;

std::shared_ptr<Request> make_registry_request(const std::string& id) {
  auto request = std::make_shared<Request>();
  request->service_request_id = id;
  request->routing.prefill_endpoint = "prefill-0";
  request->routing.prefill_incarnation = "prefill-incarnation";
  return request;
}

llm::RequestOutput make_registry_output(const std::string& id,
                                        const std::string& text,
                                        bool finished) {
  llm::RequestOutput output;
  output.service_request_id = id;
  output.finished = finished;
  llm::SequenceOutput sequence;
  sequence.index = 0;
  sequence.text = text;
  output.outputs.push_back(std::move(sequence));
  return output;
}

TEST(RequestSessionRegistryTest, SerializesOutputAndRemovesTerminalSession) {
  std::mutex mutex;
  std::condition_variable completed;
  std::vector<std::string> observed_outputs;
  std::vector<RequestTerminalReason> terminal_reasons;
  RequestSessionRegistry registry(
      {},
      [&mutex, &completed, &terminal_reasons](
          const std::shared_ptr<Request>&,
          RequestTerminalReason reason) {
        {
          std::lock_guard<std::mutex> lock(mutex);
          terminal_reasons.push_back(reason);
        }
        completed.notify_one();
      },
      2);

  auto request = make_registry_request("request-1");
  ASSERT_TRUE(registry.register_request(
      request,
      [&mutex, &observed_outputs](llm::RequestOutput output) {
        std::lock_guard<std::mutex> lock(mutex);
        observed_outputs.push_back(output.outputs.front().text);
        return true;
      },
      []() { return false; }));
  EXPECT_FALSE(registry.register_request(
      request, [](llm::RequestOutput) { return true; }, []() { return false; }));

  EXPECT_EQ(registry.on_generation(
                make_registry_output("request-1", "first", false)),
            GenerationDispatchResult::ACCEPTED);
  EXPECT_EQ(registry.on_generation(
                make_registry_output("request-1", "second", false)),
            GenerationDispatchResult::ACCEPTED);
  EXPECT_EQ(registry.on_generation(
                make_registry_output("request-1", "third", true)),
            GenerationDispatchResult::ACCEPTED);

  std::unique_lock<std::mutex> lock(mutex);
  ASSERT_TRUE(completed.wait_for(lock, 2s, [&terminal_reasons]() {
    return !terminal_reasons.empty();
  }));
  EXPECT_EQ(observed_outputs,
            (std::vector<std::string>{"first", "second", "third"}));
  EXPECT_EQ(terminal_reasons,
            (std::vector<RequestTerminalReason>{
                RequestTerminalReason::COMPLETED}));
  lock.unlock();

  registry.close();
  EXPECT_EQ(registry.size(), 0);
}

TEST(RequestSessionRegistryTest, RunsTerminalObserverOutsideRegistryLock) {
  std::mutex mutex;
  std::condition_variable completed;
  size_t size_during_callback = 0;
  RequestSessionRegistry* registry_ptr = nullptr;
  RequestSessionRegistry registry(
      {},
      [&mutex, &completed, &size_during_callback, &registry_ptr](
          const std::shared_ptr<Request>&,
          RequestTerminalReason) {
        const size_t registry_size = registry_ptr->size();
        {
          std::lock_guard<std::mutex> lock(mutex);
          size_during_callback = registry_size;
        }
        completed.notify_one();
      },
      1);
  registry_ptr = &registry;

  ASSERT_TRUE(registry.register_request(
      make_registry_request("request-2"),
      [](llm::RequestOutput) { return true; },
      []() { return false; }));
  EXPECT_EQ(registry.on_instance_failure(
                {"prefill-0",
                 "prefill-incarnation",
                 InstanceType::PREFILL}),
            1);

  std::unique_lock<std::mutex> lock(mutex);
  ASSERT_EQ(completed.wait_for(lock, 2s), std::cv_status::no_timeout);
  EXPECT_EQ(size_during_callback, 1);
  lock.unlock();

  registry.close();
  EXPECT_EQ(registry.size(), 0);
}

TEST(RequestSessionRegistryTest, CloseDrainsEventsAndRejectsNewWork) {
  int output_count = 0;
  int terminal_count = 0;
  RequestSessionRegistry registry(
      {},
      [&terminal_count](const std::shared_ptr<Request>&,
                        RequestTerminalReason) { ++terminal_count; },
      1);
  ASSERT_TRUE(registry.register_request(
      make_registry_request("request-3"),
      [&output_count](llm::RequestOutput) {
        ++output_count;
        return true;
      },
      []() { return false; }));
  ASSERT_EQ(registry.on_generation(
                make_registry_output("request-3", "final", true)),
            GenerationDispatchResult::ACCEPTED);

  registry.close();

  EXPECT_EQ(output_count, 1);
  EXPECT_EQ(terminal_count, 1);
  EXPECT_EQ(registry.size(), 0);
  EXPECT_EQ(registry.on_generation(
                make_registry_output("request-3", "late", true)),
            GenerationDispatchResult::REGISTRY_CLOSED);
  EXPECT_FALSE(registry.register_request(
      make_registry_request("request-4"),
      [](llm::RequestOutput) { return true; },
      []() { return false; }));
}

TEST(RequestSessionRegistryTest, CloseCancelsActiveSessions) {
  std::vector<llm::StatusCode> status_codes;
  std::vector<RequestTerminalReason> terminal_reasons;
  RequestSessionRegistry registry(
      {},
      [&terminal_reasons](const std::shared_ptr<Request>&,
                          RequestTerminalReason reason) {
        terminal_reasons.push_back(reason);
      },
      1);
  ASSERT_TRUE(registry.register_request(
      make_registry_request("request-active"),
      [&status_codes](llm::RequestOutput output) {
        status_codes.push_back(output.status->code());
        return true;
      },
      []() { return false; }));

  registry.close();

  EXPECT_EQ(status_codes,
            (std::vector<llm::StatusCode>{llm::StatusCode::CANCELLED}));
  EXPECT_EQ(terminal_reasons,
            (std::vector<RequestTerminalReason>{
                RequestTerminalReason::RUNTIME_CANCELLED}));
  EXPECT_EQ(registry.size(), 0);
}

TEST(RequestSessionRegistryTest, ReportsClientDisconnectWithoutOutput) {
  int32_t output_count = 0;
  RequestTerminalReason terminal_reason = RequestTerminalReason::COMPLETED;
  RequestSessionRegistry registry(
      {},
      [&terminal_reason](const std::shared_ptr<Request>&,
                         RequestTerminalReason reason) {
        terminal_reason = reason;
      },
      1);
  ASSERT_TRUE(registry.register_request(
      make_registry_request("request-4"),
      [&output_count](llm::RequestOutput) {
        ++output_count;
        return true;
      },
      []() { return true; }));

  EXPECT_EQ(registry.on_generation(
                make_registry_output("request-4", "ignored", false)),
            GenerationDispatchResult::CLIENT_DISCONNECTED);
  registry.close();

  EXPECT_EQ(output_count, 0);
  EXPECT_EQ(terminal_reason, RequestTerminalReason::CLIENT_DISCONNECTED);
}

TEST(RequestSessionRegistryTest, ConcurrentTerminalEventsCompleteOnce) {
  std::atomic<int32_t> output_count{0};
  std::atomic<int32_t> terminal_count{0};
  RequestSessionRegistry registry(
      {},
      [&terminal_count](const std::shared_ptr<Request>&,
                        RequestTerminalReason) { ++terminal_count; },
      1);
  ASSERT_TRUE(registry.register_request(
      make_registry_request("request-5"),
      [&output_count](llm::RequestOutput) {
        ++output_count;
        return true;
      },
      []() { return false; }));

  std::atomic<bool> start{false};
  auto await_start = [&start]() {
    while (!start.load()) {
      std::this_thread::yield();
    }
  };
  std::thread generation_thread([&registry, &await_start]() {
    await_start();
    registry.on_generation(
        make_registry_output("request-5", "final", true));
  });
  std::thread transport_thread([&registry, &await_start]() {
    await_start();
    registry.on_transport_failure("request-5",
                                  TransportFailureStage::BEFORE_FIRST_TOKEN,
                                  "connection reset");
  });
  std::thread instance_thread([&registry, &await_start]() {
    await_start();
    registry.on_instance_failure(
        {"prefill-0", "prefill-incarnation", InstanceType::PREFILL});
  });

  start.store(true);
  generation_thread.join();
  transport_thread.join();
  instance_thread.join();
  registry.close();

  EXPECT_EQ(output_count.load(), 1);
  EXPECT_EQ(terminal_count.load(), 1);
  EXPECT_EQ(registry.size(), 0);
}

}  // namespace
}  // namespace xllm_service
