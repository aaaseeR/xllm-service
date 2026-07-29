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

#include <brpc/controller.h>
#include <glog/logging.h>
#include <google/protobuf/message.h>
#include <google/protobuf/stubs/callback.h>

#include <condition_variable>
#include <exception>
#include <mutex>
#include <utility>

#include "xllm_service.pb.h"

namespace xllm_service {

class DispatcherState final {
 public:
  explicit DispatcherState(Dispatcher::TransportObserver transport_observer)
      : transport_observer_(std::move(transport_observer)) {}

  bool begin_dispatch() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return false;
    }
    ++inflight_;
    return true;
  }

  void complete_generation(const TransportResult& result) {
    complete(result, [this, &result]() {
      if (transport_observer_) {
        transport_observer_(result);
      }
    });
  }

  void complete_models(const TransportResult& result,
                       const xllm::proto::ModelListResponse& response,
                       const Dispatcher::ModelResultCallback& callback) {
    complete(result, [&result, &response, &callback]() {
      if (callback) {
        callback(result, response);
      }
    });
  }

  void close() {
    std::unique_lock<std::mutex> lock(mutex_);
    closed_ = true;
    condition_.wait(lock, [this]() { return inflight_ == 0; });
  }

  DispatcherStats stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    DispatcherStats result;
    result.inflight = inflight_;
    result.transport_failure_total = transport_failure_total_;
    return result;
  }

 private:
  template <typename Callback>
  void complete(const TransportResult& result, Callback callback) {
    if (result.code != TransportResultCode::SUCCESS) {
      std::lock_guard<std::mutex> lock(mutex_);
      ++transport_failure_total_;
    }

    try {
      callback();
    } catch (const std::exception& error) {
      LOG(ERROR) << "Transport completion callback failed: " << error.what();
    } catch (...) {
      LOG(ERROR) << "Transport completion callback failed with unknown error";
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (inflight_ > 0) {
      --inflight_;
    }
    if (inflight_ == 0) {
      condition_.notify_all();
    }
  }

  const Dispatcher::TransportObserver transport_observer_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  bool closed_ = false;
  uint64_t inflight_ = 0;
  uint64_t transport_failure_total_ = 0;
};

namespace {

TransportResult make_failure(const std::string& request_id,
                             TransportResultCode code,
                             const std::string& message) {
  TransportResult result;
  result.request_id = request_id;
  result.code = code;
  result.failure_stage = TransportFailureStage::BEFORE_FIRST_TOKEN;
  result.message = message;
  return result;
}

class GenerationRpcClosure final : public google::protobuf::Closure {
 public:
  GenerationRpcClosure(std::shared_ptr<DispatcherState> state,
                       std::shared_ptr<brpc::Channel> channel,
                       std::shared_ptr<const google::protobuf::Message> request,
                       std::string request_id)
      : state_(std::move(state)),
        channel_(std::move(channel)),
        request_(std::move(request)),
        request_id_(std::move(request_id)) {}

  brpc::Controller* controller() { return &controller_; }

  void Run() override {
    TransportResult result;
    result.request_id = request_id_;
    if (controller_.Failed()) {
      result = make_failure(request_id_,
                            TransportResultCode::RPC_FAILURE,
                            controller_.ErrorText());
    }
    state_->complete_generation(result);
    delete this;
  }

 private:
  const std::shared_ptr<DispatcherState> state_;
  const std::shared_ptr<brpc::Channel> channel_;
  const std::shared_ptr<const google::protobuf::Message> request_;
  const std::string request_id_;
  brpc::Controller controller_;
};

class ModelsRpcClosure final : public google::protobuf::Closure {
 public:
  ModelsRpcClosure(std::shared_ptr<DispatcherState> state,
                   std::shared_ptr<brpc::Channel> channel,
                   std::shared_ptr<const xllm::proto::ModelListRequest> request,
                   Dispatcher::ModelResultCallback callback)
      : state_(std::move(state)),
        channel_(std::move(channel)),
        request_(std::move(request)),
        callback_(std::move(callback)) {}

  brpc::Controller* controller() { return &controller_; }
  xllm::proto::ModelListResponse* response() { return &response_; }

  void Run() override {
    TransportResult result;
    if (controller_.Failed()) {
      result = make_failure("",
                            TransportResultCode::RPC_FAILURE,
                            controller_.ErrorText());
    }
    state_->complete_models(result, response_, callback_);
    delete this;
  }

 private:
  const std::shared_ptr<DispatcherState> state_;
  const std::shared_ptr<brpc::Channel> channel_;
  const std::shared_ptr<const xllm::proto::ModelListRequest> request_;
  const Dispatcher::ModelResultCallback callback_;
  brpc::Controller controller_;
  xllm::proto::ModelListResponse response_;
};

template <typename Request, typename RpcInvoker>
bool dispatch_generation(
    const std::shared_ptr<ChannelPool>& channel_pool,
    const std::shared_ptr<DispatcherState>& state,
    const std::string& endpoint,
    const std::string& incarnation_id,
    const std::string& request_id,
    const Request& request,
    RpcInvoker invoke) {
  if (!state->begin_dispatch()) {
    return false;
  }

  const auto channel = channel_pool->get_or_create(endpoint, incarnation_id);
  if (channel == nullptr) {
    state->complete_generation(make_failure(
        request_id,
        TransportResultCode::CHANNEL_UNAVAILABLE,
        "Backend channel is unavailable for endpoint " + endpoint));
    return true;
  }

  auto owned_request = std::make_shared<Request>(request);
  auto* closure = new GenerationRpcClosure(
      state, channel, owned_request, request_id);
  xllm::proto::XllmAPIService_Stub stub(channel.get());
  invoke(stub, closure->controller(), owned_request.get(), closure);
  return true;
}

}  // namespace

Dispatcher::Dispatcher(std::shared_ptr<ChannelPool> channel_pool,
                       TransportObserver transport_observer)
    : channel_pool_(std::move(channel_pool)),
      state_(std::make_shared<DispatcherState>(
          std::move(transport_observer))) {}

Dispatcher::~Dispatcher() { close(); }

bool Dispatcher::dispatch_completion(
    const std::string& endpoint,
    const std::string& incarnation_id,
    const std::string& request_id,
    const xllm::proto::CompletionRequest& request) {
  return dispatch_generation(
      channel_pool_,
      state_,
      endpoint,
      incarnation_id,
      request_id,
      request,
      [](xllm::proto::XllmAPIService_Stub& stub,
         brpc::Controller* controller,
         const xllm::proto::CompletionRequest* owned_request,
         google::protobuf::Closure* closure) {
        stub.Completions(controller, owned_request, nullptr, closure);
      });
}

bool Dispatcher::dispatch_chat(
    const std::string& endpoint,
    const std::string& incarnation_id,
    const std::string& request_id,
    const xllm::proto::ChatRequest& request) {
  return dispatch_generation(
      channel_pool_,
      state_,
      endpoint,
      incarnation_id,
      request_id,
      request,
      [](xllm::proto::XllmAPIService_Stub& stub,
         brpc::Controller* controller,
         const xllm::proto::ChatRequest* owned_request,
         google::protobuf::Closure* closure) {
        stub.ChatCompletions(controller, owned_request, nullptr, closure);
      });
}

bool Dispatcher::dispatch_models(
    const std::string& endpoint,
    const std::string& incarnation_id,
    const xllm::proto::ModelListRequest& request,
    ModelResultCallback callback) {
  if (!state_->begin_dispatch()) {
    return false;
  }

  const auto channel =
      channel_pool_->get_or_create(endpoint, incarnation_id);
  if (channel == nullptr) {
    const xllm::proto::ModelListResponse response;
    state_->complete_models(
        make_failure("",
                     TransportResultCode::CHANNEL_UNAVAILABLE,
                     "Backend channel is unavailable for endpoint " +
                         endpoint),
        response,
        callback);
    return true;
  }

  auto owned_request =
      std::make_shared<xllm::proto::ModelListRequest>(request);
  auto* closure = new ModelsRpcClosure(
      state_, channel, owned_request, std::move(callback));
  xllm::proto::XllmAPIService_Stub stub(channel.get());
  stub.Models(closure->controller(),
              owned_request.get(),
              closure->response(),
              closure);
  return true;
}

void Dispatcher::close() { state_->close(); }

DispatcherStats Dispatcher::stats() const {
  DispatcherStats result = state_->stats();
  result.channel_count = channel_pool_->channel_count();
  result.endpoint_count = channel_pool_->endpoint_count();
  return result;
}

}  // namespace xllm_service
