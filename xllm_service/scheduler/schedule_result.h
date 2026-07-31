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
#include <string>

namespace xllm_service {

enum class ScheduleError : uint8_t {
  NONE = 0,
  INVALID_REQUEST = 1,
  INVALID_ROUTING_DIRECTIVE = 2,
  BACKEND_UNAVAILABLE = 3,
  STALE_ROUTING_DECISION = 4,
  INTERNAL_ERROR = 5,
};

struct ScheduleResult {
  ScheduleError error = ScheduleError::NONE;
  std::string message;
};

bool schedule_result_ok(const ScheduleResult& result);
bool schedule_result_is_retryable(const ScheduleResult& result);
const char* schedule_error_code(ScheduleError error);

}  // namespace xllm_service
