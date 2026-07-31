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

#include "transport/types.h"

namespace xllm_service {

const char* transport_result_code_name(TransportResultCode code) {
  switch (code) {
    case TransportResultCode::SUCCESS:
      return "success";
    case TransportResultCode::CHANNEL_UNAVAILABLE:
      return "channel_unavailable";
    case TransportResultCode::RPC_FAILURE:
      return "rpc_failure";
    case TransportResultCode::DISPATCHER_CLOSED:
      return "dispatcher_closed";
    case TransportResultCode::INVALID_ROUTING_DECISION:
      return "invalid_routing_decision";
    case TransportResultCode::STALE_ROUTING_DECISION:
      return "stale_routing_decision";
  }
  return "unknown";
}

const char* transport_failure_stage_name(TransportFailureStage stage) {
  switch (stage) {
    case TransportFailureStage::BEFORE_FIRST_TOKEN:
      return "before_first_token";
    case TransportFailureStage::AFTER_FIRST_TOKEN:
      return "after_first_token";
  }
  return "unknown";
}

const char* transport_retryability_name(TransportRetryability retryability) {
  switch (retryability) {
    case TransportRetryability::NOT_RETRYABLE:
      return "not_retryable";
    case TransportRetryability::RETRYABLE_BEFORE_FIRST_TOKEN:
      return "retryable_before_first_token";
  }
  return "unknown";
}

bool transport_result_is_retryable(const TransportResult& result) {
  return result.retryability ==
             TransportRetryability::RETRYABLE_BEFORE_FIRST_TOKEN &&
         result.failure_stage == TransportFailureStage::BEFORE_FIRST_TOKEN;
}

}  // namespace xllm_service
