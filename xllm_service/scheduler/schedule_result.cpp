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

#include "scheduler/schedule_result.h"

namespace xllm_service {

bool schedule_result_ok(const ScheduleResult& result) {
  return result.error == ScheduleError::NONE;
}

bool schedule_result_is_retryable(const ScheduleResult& result) {
  return result.error == ScheduleError::STALE_ROUTING_DECISION;
}

const char* schedule_error_code(ScheduleError error) {
  switch (error) {
    case ScheduleError::NONE:
      return "none";
    case ScheduleError::INVALID_REQUEST:
      return "invalid_request";
    case ScheduleError::INVALID_ROUTING_DIRECTIVE:
      return "invalid_routing_directive";
    case ScheduleError::BACKEND_UNAVAILABLE:
      return "backend_unavailable";
    case ScheduleError::STALE_ROUTING_DECISION:
      return "stale_routing_decision";
    case ScheduleError::INTERNAL_ERROR:
      return "internal_error";
  }
  return "unknown";
}

}  // namespace xllm_service
