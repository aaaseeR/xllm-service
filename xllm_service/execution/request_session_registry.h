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

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/threadpool.h"
#include "execution/request_session.h"

namespace xllm_service {

enum class GenerationDispatchResult : uint8_t {
  ACCEPTED = 0,
  CLIENT_DISCONNECTED = 1,
  SESSION_NOT_FOUND = 2,
  REGISTRY_CLOSED = 3,
};

class RequestSessionRegistry final {
 public:
  static constexpr size_t kDefaultExecutorCount = 128;

  explicit RequestSessionRegistry(
      RequestSession::GenerationObserver generation_observer,
      RequestSession::TerminalObserver terminal_observer,
      size_t executor_count = kDefaultExecutorCount);
  ~RequestSessionRegistry();

  RequestSessionRegistry(const RequestSessionRegistry&) = delete;
  RequestSessionRegistry& operator=(const RequestSessionRegistry&) = delete;

  bool register_request(std::shared_ptr<Request> request,
                        OutputCallback output_callback,
                        RequestSession::DisconnectCheck disconnect_check);
  GenerationDispatchResult on_generation(const llm::RequestOutput& output);
  bool on_transport_failure(const std::string& request_id,
                            TransportFailureStage stage,
                            const std::string& message);
  size_t on_instance_failure(const InstanceFailure& failure);

  size_t size() const;
  void close();

 private:
  struct Entry {
    std::shared_ptr<RequestSession> session;
    size_t executor_index = 0;
  };

  void handle_terminal(const std::shared_ptr<Request>& request,
                       RequestTerminalReason reason);

  const RequestSession::GenerationObserver generation_observer_;
  const RequestSession::TerminalObserver terminal_observer_;

  mutable std::mutex mutex_;
  bool closed_ = false;
  size_t next_executor_index_ = 0;
  std::unordered_map<std::string, Entry> sessions_;
  std::vector<std::unique_ptr<ThreadPool>> executors_;
};

}  // namespace xllm_service
