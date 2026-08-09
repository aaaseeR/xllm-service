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

#include <brpc/channel.h>
#include <brpc/controller.h>

#include <cstdint>
#include <memory>
#include <string>

#include "provider.pb.h"

namespace xllm_service::provider {

enum class AttemptControlOperation : int8_t {
  QUERY = 0,
  CANCEL = 1,
};

struct AttemptControlResult {
  bool direct_success = false;
  bool terminal_proof = false;
  xllm::proto::AttemptLifecycleState state =
      xllm::proto::ATTEMPT_LIFECYCLE_STATE_UNSPECIFIED;
};

bool constant_time_internal_token_equal(const std::string& expected,
                                        const std::string& provided);
bool valid_vllm_agent_internal_token(const std::string& internal_api_token);

bool set_vllm_agent_internal_token(brpc::Controller* controller,
                                   const std::string& internal_api_token);

bool set_vllm_agent_attempt_headers(brpc::Controller* controller,
                                    const std::string& request_uid,
                                    uint64_t attempt_seq,
                                    const std::string& incarnation_id,
                                    uint64_t remaining_deadline_ms);

AttemptControlResult call_provider_attempt_control(
    xllm::proto::ProviderId provider_id,
    const std::shared_ptr<brpc::Channel>& channel,
    const xllm::proto::ExecutionResourceHold& hold,
    const xllm::proto::ExecutionHolder& holder,
    AttemptControlOperation operation,
    const std::string& internal_api_token,
    int32_t timeout_ms);

AttemptControlResult parse_vllm_agent_attempt_response(
    int32_t http_status_code,
    const std::string& response_body,
    const std::string& expected_request_uid,
    uint64_t expected_attempt_seq,
    const std::string& expected_incarnation_id);

}  // namespace xllm_service::provider
