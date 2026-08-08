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

#pragma once

#include <absl/time/time.h>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>

#include "chat_template/jinja_chat_template.h"
#include "common/call_data.h"
#include "common/types.h"
#include "common/xllm/output.h"
#include "core/framework/request/first_event_retry_policy.h"
#include "core/framework/request/request_deadline.h"
#include "observability.pb.h"
#include "provider/execution_hold.h"
#include "request/first_output_retry_budget.h"
#include "request/output_event_sequencer.h"

namespace xllm_service {

// Store request-related data
struct Request {
  // model name
  std::string model;

  // V2 observation identity. request_uid is also the execution key carried by
  // the legacy service_request_id wire field; no second execution ID exists.
  xllm::proto::RequestCorrelation correlation;

  // The business duration is converted once at Service ingress. Every
  // downstream hop receives only the locally recomputed remaining duration.
  bool request_deadline_present = false;
  std::optional<xllm::RequestDeadline> request_deadline;

  // Service owns this timer relationship and forwards it as durations. P
  // starts its local retry clock only when GenerationCommit begins.
  std::optional<xllm::FirstEventRetryPolicy> first_event_retry_policy;

  // whether to stream the response
  bool stream = false;

  // whether to return usage
  bool include_usage = false;

  bool offline = false;

  // input prompt
  std::string prompt;

  // input messages
  ChatMessages messages;

  // tool definitions for function/tool calling
  std::vector<JsonTool> tools;

  // controls tool usage behavior, e.g. auto/none/required
  std::string tool_choice = "auto";

  // extra template context such as {"enable_thinking": false}
  nlohmann::json chat_template_kwargs = nlohmann::json::object();

  // token ids of prompt
  std::vector<int32_t> token_ids;

  // instance routing
  xllm::proto::ProviderId provider_id = xllm::proto::PROVIDER_ID_UNSPECIFIED;
  Routing routing;
  std::string prefill_incarnation_id;
  std::string decode_incarnation_id;

  // At most one outcome-unknown execution resource holder is permitted for
  // the current attempt. Its cleanup-capacity token is reserved before the
  // request is dispatched.
  provider::RequestExecutionHold execution_hold;

  // Serializes cross-sender sequencing with affinity-thread dispatch. Native
  // REMOTE_PD installs the bounded sequencer before the first RPC is sent.
  std::mutex output_dispatch_mutex;
  std::unique_ptr<OutputEventSequencer> output_event_sequencer;
  bool output_dispatch_closed = false;

  // Set directly by the brpc connection/progressive-attachment stop callback;
  // the bounded watchdog consumes the corresponding monitor notification.
  std::atomic<bool> client_disconnected{false};

  // Changes only after the output callback has successfully crossed the
  // Service response boundary. It is the hard no-replacement fence.
  std::atomic<bool> first_token_emitted{false};

  std::unique_ptr<FirstOutputRetryBudget> first_output_retry_budget;

  // Re-encodes only attempt/routing/deadline fields into the already owned
  // request protobuf and starts an asynchronous native dispatch. It must not
  // retain a shared_ptr back to this Request.
  std::function<bool(const Request&)> retry_dispatch_callback;

  // prefill stage finished
  std::atomic<bool> prefill_stage_finished{false};

  // the number of generated tokens
  int64_t num_generated_tokens = 0;

  // the estimated TTFT obtained from the TTFT predictor
  int64_t estimated_ttft = 0;

  // output callback
  OutputCallback output_callback;

  std::shared_ptr<CallData> call_data;

  // trace callback
  std::function<void(const std::string&)> trace_callback = nullptr;

  // latest token generate time
  absl::Time latest_generate_time;
};

}  // namespace xllm_service
