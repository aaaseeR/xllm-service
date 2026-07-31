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

#include <algorithm>
#include <utility>

namespace xllm_service {

RequestSessionRegistry::RequestSessionRegistry(
    RequestSession::GenerationObserver generation_observer,
    RequestSession::TerminalObserver terminal_observer,
    size_t executor_count)
    : generation_observer_(std::move(generation_observer)),
      terminal_observer_(std::move(terminal_observer)) {
  executor_count = std::max<size_t>(1, executor_count);
  executors_.reserve(executor_count);
  for (size_t i = 0; i < executor_count; ++i) {
    executors_.emplace_back(std::make_unique<ThreadPool>(1));
  }
}

RequestSessionRegistry::~RequestSessionRegistry() { close(); }

bool RequestSessionRegistry::register_request(
    std::shared_ptr<Request> request,
    OutputCallback output_callback,
    RequestSession::DisconnectCheck disconnect_check) {
  if (request == nullptr || request->service_request_id.empty()) {
    return false;
  }

  const std::string request_id = request->service_request_id;
  auto terminal_observer = [this](
                               const std::shared_ptr<Request>& terminal_request,
                               RequestTerminalReason reason) {
    handle_terminal(terminal_request, reason);
  };
  auto session = std::make_shared<RequestSession>(request,
                                                  std::move(output_callback),
                                                  std::move(disconnect_check),
                                                  generation_observer_,
                                                  terminal_observer);
  session->on_dispatched();

  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_ || sessions_.find(request_id) != sessions_.end()) {
    return false;
  }
  const size_t executor_index = next_executor_index_;
  next_executor_index_ = (next_executor_index_ + 1) % executors_.size();
  sessions_.emplace(request_id, Entry{std::move(session), executor_index});
  return true;
}

GenerationDispatchResult RequestSessionRegistry::on_generation(
    const llm::RequestOutput& output) {
  std::shared_ptr<RequestSession> session;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return GenerationDispatchResult::REGISTRY_CLOSED;
    }
    auto it = sessions_.find(output.service_request_id);
    if (it == sessions_.end()) {
      return GenerationDispatchResult::SESSION_NOT_FOUND;
    }
    session = it->second.session;
  }

  const bool disconnected = session->client_disconnected();

  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) {
    return GenerationDispatchResult::REGISTRY_CLOSED;
  }
  auto it = sessions_.find(output.service_request_id);
  if (it == sessions_.end() || it->second.session != session) {
    return GenerationDispatchResult::SESSION_NOT_FOUND;
  }
  if (disconnected) {
    executors_[it->second.executor_index]->schedule(
        [session]() { session->on_client_disconnect(); });
    return GenerationDispatchResult::CLIENT_DISCONNECTED;
  }
  executors_[it->second.executor_index]->schedule([session, output]() mutable {
    session->on_generation(std::move(output));
  });
  return GenerationDispatchResult::ACCEPTED;
}

bool RequestSessionRegistry::on_transport_failure(
    const TransportResult& result) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) {
    return false;
  }
  auto it = sessions_.find(result.request_id);
  if (it == sessions_.end()) {
    return false;
  }
  const auto session = it->second.session;
  executors_[it->second.executor_index]->schedule(
      [session, result]() { session->on_transport_failure(result); });
  return true;
}

bool RequestSessionRegistry::on_transport_failure(const std::string& request_id,
                                                  TransportFailureStage stage,
                                                  const std::string& message) {
  TransportResult result;
  result.request_id = request_id;
  result.code = TransportResultCode::RPC_FAILURE;
  result.failure_stage = stage;
  result.message = message;
  return on_transport_failure(result);
}

size_t RequestSessionRegistry::on_instance_failure(
    const InstanceFailure& failure) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) {
    return 0;
  }
  for (const auto& item : sessions_) {
    const Entry& entry = item.second;
    const auto session = entry.session;
    executors_[entry.executor_index]->schedule(
        [session, failure]() { session->on_instance_failure(failure); });
  }
  return sessions_.size();
}

size_t RequestSessionRegistry::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return sessions_.size();
}

void RequestSessionRegistry::close() {
  std::vector<Entry> active_sessions;
  std::vector<std::unique_ptr<ThreadPool>> executors;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return;
    }
    closed_ = true;
    active_sessions.reserve(sessions_.size());
    for (const auto& item : sessions_) {
      active_sessions.push_back(item.second);
    }
  }

  for (const Entry& entry : active_sessions) {
    const auto session = entry.session;
    executors_[entry.executor_index]->schedule([session]() {
      session->on_runtime_cancel("Runtime is shutting down");
    });
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    executors.swap(executors_);
  }

  executors.clear();

  std::lock_guard<std::mutex> lock(mutex_);
  sessions_.clear();
}

void RequestSessionRegistry::handle_terminal(
    const std::shared_ptr<Request>& request,
    RequestTerminalReason reason) {
  if (terminal_observer_) {
    terminal_observer_(request, reason);
  }

  std::lock_guard<std::mutex> lock(mutex_);
  sessions_.erase(request->service_request_id);
}

}  // namespace xllm_service
