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

#include "http_service/health_response.h"

namespace xllm_service {

HealthResponse make_liveness_response() {
  return HealthResponse{
      .status_code = 200,
      .body = "{\"status\":\"live\"}",
  };
}

HealthResponse make_readiness_response(
    const provider::ReadinessSnapshot& readiness) {
  const std::string reason = provider::readiness_reason_name(readiness.reason);
  return HealthResponse{
      .status_code = readiness.accepting_new_requests ? 200 : 503,
      .body = "{\"status\":\"" +
              std::string(readiness.accepting_new_requests ? "ready"
                                                           : "not_ready") +
              "\",\"reason\":\"" + reason + "\"}",
  };
}

HealthResponse make_service_not_ready_response(
    const provider::ReadinessSnapshot& readiness) {
  const std::string reason = provider::readiness_reason_name(readiness.reason);
  return HealthResponse{
      .status_code = 503,
      .body = "{\"code\":\"SERVICE_NOT_READY\",\"reason\":\"" + reason + "\"}",
  };
}

}  // namespace xllm_service
