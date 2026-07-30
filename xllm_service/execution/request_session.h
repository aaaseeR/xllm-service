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

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "common/xllm/output.h"
#include "request/request.h"
#include "transport/types.h"

namespace xllm_service {

enum class RequestSessionState : uint8_t {
  CREATED = 0,
  DISPATCHED = 1,
  PREFILL_RUNNING = 2,
  DECODE_RUNNING = 3,
  COMPLETED = 4,
  FAILED = 5,
  CANCELLED = 6,
};

enum class RequestTerminalReason : uint8_t {
  COMPLETED = 0,
  GENERATION_FAILURE = 1,
  OUTPUT_FAILURE = 2,
  TRANSPORT_FAILURE_BEFORE_FIRST_TOKEN = 3,
  TRANSPORT_FAILURE_AFTER_FIRST_TOKEN = 4,
  CLIENT_DISCONNECTED = 5,
  INSTANCE_FAILURE = 6,
  RUNTIME_CANCELLED = 7,
};

const char* request_terminal_reason_name(RequestTerminalReason reason);

struct InstanceFailure {
  std::string instance_name;
  std::string incarnation_id;
  InstanceType type = InstanceType::DEFAULT;
};

class RequestSession final {
 public:
  using DisconnectCheck = std::function<bool()>;
  using GenerationObserver = std::function<void(
      const std::shared_ptr<Request>&, const llm::RequestOutput&)>;
  using TerminalObserver = std::function<void(
      const std::shared_ptr<Request>&, RequestTerminalReason)>;

  RequestSession(std::shared_ptr<Request> request,
                 OutputCallback output_callback,
                 DisconnectCheck disconnect_check,
                 GenerationObserver generation_observer,
                 TerminalObserver terminal_observer);

  RequestSession(const RequestSession&) = delete;
  RequestSession& operator=(const RequestSession&) = delete;

  // State-changing methods must be invoked by the same serial executor.
  bool on_dispatched();
  bool on_generation(llm::RequestOutput output);
  bool on_transport_failure(TransportFailureStage stage,
                            const std::string& message);
  bool on_instance_failure(const InstanceFailure& failure);
  bool on_runtime_cancel(const std::string& message);
  bool on_client_disconnect();

  RequestSessionState state() const { return state_.load(); }
  bool client_disconnected() const;
  const std::shared_ptr<Request>& request() const { return request_; }

 private:
  bool matches(const InstanceFailure& failure) const;
  bool deliver_output(llm::RequestOutput output);
  bool finish(RequestSessionState state, RequestTerminalReason reason);
  bool is_terminal() const;

  const std::shared_ptr<Request> request_;
  const OutputCallback output_callback_;
  const DisconnectCheck disconnect_check_;
  const GenerationObserver generation_observer_;
  const TerminalObserver terminal_observer_;
  std::atomic<RequestSessionState> state_{RequestSessionState::CREATED};
};

}  // namespace xllm_service
