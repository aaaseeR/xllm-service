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

#include "dispatcher/dispatcher.h"

#include <brpc/server.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/options.h"
#include "xllm_service.pb.h"

namespace xllm_service {
namespace {

class DelayedCompletionService final
    : public xllm::proto::XllmAPIService {
 public:
  void Completions(google::protobuf::RpcController*,
                   const xllm::proto::CompletionRequest* request,
                   xllm::proto::CompletionResponse*,
                   google::protobuf::Closure* done) override {
    std::lock_guard<std::mutex> lock(mutex_);
    request_id_ = request->service_request_id();
    done_ = done;
    condition_.notify_all();
  }

  bool wait_until_received() {
    std::unique_lock<std::mutex> lock(mutex_);
    return condition_.wait_for(
        lock, std::chrono::seconds(5), [this]() { return done_ != nullptr; });
  }

  std::string request_id() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return request_id_;
  }

  void finish() {
    google::protobuf::Closure* done = nullptr;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      done = done_;
      done_ = nullptr;
    }
    if (done != nullptr) {
      done->Run();
    }
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::string request_id_;
  google::protobuf::Closure* done_ = nullptr;
};

TEST(DispatcherTest, ReportsUnavailableGenerationChannel) {
  auto pool = std::make_shared<ChannelPool>([](const std::string&) {
    return std::shared_ptr<brpc::Channel>();
  });
  std::vector<TransportResult> results;
  Dispatcher dispatcher(
      pool,
      [&results](const TransportResult& result) { results.push_back(result); });

  xllm::proto::CompletionRequest request;
  EXPECT_TRUE(dispatcher.dispatch_completion("127.0.0.1:8000",
                                             "incarnation-1",
                                             "request-1",
                                             request));

  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].request_id, "request-1");
  EXPECT_EQ(results[0].code, TransportResultCode::CHANNEL_UNAVAILABLE);
  EXPECT_EQ(results[0].failure_stage,
            TransportFailureStage::BEFORE_FIRST_TOKEN);
  EXPECT_EQ(dispatcher.stats().inflight, 0);
  EXPECT_EQ(dispatcher.stats().transport_failure_total, 1);
}

TEST(DispatcherTest, ReportsUnavailableModelsChannelToCaller) {
  auto pool = std::make_shared<ChannelPool>([](const std::string&) {
    return std::shared_ptr<brpc::Channel>();
  });
  Dispatcher dispatcher(pool, {});
  TransportResult observed;
  int callback_count = 0;

  xllm::proto::ModelListRequest request;
  EXPECT_TRUE(dispatcher.dispatch_models(
      "127.0.0.1:8000",
      "incarnation-1",
      request,
      [&observed, &callback_count](
          const TransportResult& result,
          const xllm::proto::ModelListResponse&) {
        observed = result;
        ++callback_count;
      }));

  EXPECT_EQ(callback_count, 1);
  EXPECT_EQ(observed.code, TransportResultCode::CHANNEL_UNAVAILABLE);
  EXPECT_EQ(dispatcher.stats().inflight, 0);
}

TEST(DispatcherTest, RejectsDispatchAfterClose) {
  auto pool = std::make_shared<ChannelPool>([](const std::string&) {
    return std::shared_ptr<brpc::Channel>();
  });
  int observer_count = 0;
  Dispatcher dispatcher(pool, [&observer_count](const TransportResult&) {
    ++observer_count;
  });
  dispatcher.close();

  xllm::proto::ChatRequest request;
  EXPECT_FALSE(dispatcher.dispatch_chat("127.0.0.1:8000",
                                       "incarnation-1",
                                       "request-1",
                                       request));
  EXPECT_EQ(observer_count, 0);
}

TEST(DispatcherTest, CloseWaitsForInflightRpcCompletion) {
  DelayedCompletionService service;
  brpc::Server server;
  ASSERT_EQ(server.AddService(
                &service, brpc::SERVER_DOESNT_OWN_SERVICE),
            0);
  ASSERT_EQ(server.Start("127.0.0.1",
                         brpc::PortRange(19000, 19999),
                         nullptr),
            0);

  const std::string endpoint =
      "127.0.0.1:" + std::to_string(server.listen_address().port);
  Options options;
  options.timeout_ms(5000).connect_timeout_ms(1000);
  auto pool = std::make_shared<ChannelPool>(options);
  ASSERT_TRUE(pool->activate(endpoint, "incarnation-1"));

  std::atomic<int32_t> observer_count{0};
  Dispatcher dispatcher(pool, [&observer_count](const TransportResult& result) {
    EXPECT_EQ(result.code, TransportResultCode::SUCCESS);
    ++observer_count;
  });
  xllm::proto::CompletionRequest request;
  request.set_service_request_id("request-owned-by-dispatcher");
  ASSERT_TRUE(dispatcher.dispatch_completion(endpoint,
                                             "incarnation-1",
                                             "request-1",
                                             request));
  const bool request_received = service.wait_until_received();
  EXPECT_TRUE(request_received);
  if (!request_received) {
    server.Stop(0);
    server.Join();
    dispatcher.close();
    return;
  }
  EXPECT_EQ(service.request_id(), "request-owned-by-dispatcher");

  std::atomic<bool> close_returned{false};
  std::thread close_thread([&dispatcher, &close_returned]() {
    dispatcher.close();
    close_returned.store(true);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(close_returned.load());

  service.finish();
  close_thread.join();

  EXPECT_TRUE(close_returned.load());
  EXPECT_EQ(observer_count.load(), 1);
  EXPECT_EQ(dispatcher.stats().inflight, 0);
  server.Stop(0);
  server.Join();
}

}  // namespace
}  // namespace xllm_service
