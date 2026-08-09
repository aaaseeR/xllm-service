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

#include <cstdint>

#include "request/request.h"

namespace xllm_service {

// Copies the attempt-scoped V2 envelope immediately before an xLLM Native
// dispatch. Client-supplied plan fields are always cleared; only the plan
// built and retained by Scheduler may cross the Engine boundary.
template <typename RequestProto>
bool set_request_execution_context(RequestProto* request_pb,
                                   const Request& request) {
  request_pb->set_service_request_id(request.correlation.request_uid());
  *request_pb->mutable_correlation() = request.correlation;
  request_pb->clear_execution_plan();
  if (request.execution_plan.has_value()) {
    *request_pb->mutable_execution_plan() = *request.execution_plan;
  }
  if (request.request_deadline.has_value()) {
    const uint64_t remaining_ms = request.request_deadline->remaining_ms();
    if (remaining_ms == 0) {
      return false;
    }
    request_pb->set_remaining_deadline_ms(remaining_ms);
  }
  if (!request.first_event_retry_policy.has_value()) {
    return false;
  }
  request_pb->set_first_event_retry_budget_ms(
      request.first_event_retry_policy->retry_budget_ms());
  request_pb->set_first_event_dispatch_margin_ms(
      request.first_event_retry_policy->dispatch_margin_ms());
  return true;
}

}  // namespace xllm_service
