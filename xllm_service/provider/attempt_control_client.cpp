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

#include "provider/attempt_control_client.h"

#include <brpc/controller.h>
#include <glog/logging.h>
#include <openssl/crypto.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>

#include "disagg_pd.pb.h"

namespace xllm_service::provider {
namespace {

constexpr size_t kMaxIdentityBytes = 256;
constexpr size_t kMaxInternalTokenBytes = 4096;

bool valid_identity(const std::string& value) {
  return !value.empty() && value.size() <= kMaxIdentityBytes;
}

bool terminal_state(xllm::proto::AttemptLifecycleState state) {
  switch (state) {
    case xllm::proto::ATTEMPT_LIFECYCLE_STATE_DONE:
    case xllm::proto::ATTEMPT_LIFECYCLE_STATE_CANCELLED:
    case xllm::proto::ATTEMPT_LIFECYCLE_STATE_EXPIRED:
    case xllm::proto::ATTEMPT_LIFECYCLE_STATE_FAILED:
    case xllm::proto::ATTEMPT_LIFECYCLE_STATE_CANCELLED_BEFORE_CREATE:
      return true;
    case xllm::proto::ATTEMPT_LIFECYCLE_STATE_UNSPECIFIED:
    case xllm::proto::ATTEMPT_LIFECYCLE_STATE_ABSENT:
    case xllm::proto::ATTEMPT_LIFECYCLE_STATE_RESERVED:
    case xllm::proto::ATTEMPT_LIFECYCLE_STATE_RECEIVING:
    case xllm::proto::ATTEMPT_LIFECYCLE_STATE_GENERATION_COMMITTED:
    case xllm::proto::ATTEMPT_LIFECYCLE_STATE_RUNNING:
    case xllm::proto::ATTEMPT_LIFECYCLE_STATE_LOCAL_GENERATION_COMMITTED:
    default:
      return false;
  }
}

std::optional<xllm::proto::AttemptLifecycleState> parse_state_name(
    const std::string& name) {
  xllm::proto::AttemptLifecycleState state;
  if (!xllm::proto::AttemptLifecycleState_Parse(name, &state)) {
    return std::nullopt;
  }
  return state;
}

bool same_attempt_key(const xllm::proto::RequestAttemptKey& key,
                      const xllm::proto::ExecutionResourceHold& hold,
                      const xllm::proto::ExecutionHolder& holder) {
  return key.request_uid() == hold.attempt().request_uid() &&
         key.has_attempt_seq() && hold.attempt().has_attempt_seq() &&
         key.attempt_seq() == hold.attempt().attempt_seq() &&
         key.incarnation_id() == holder.incarnation_id();
}

AttemptControlResult call_vllm_agent(
    const std::shared_ptr<brpc::Channel>& channel,
    const xllm::proto::ExecutionResourceHold& hold,
    const xllm::proto::ExecutionHolder& holder,
    AttemptControlOperation operation,
    const std::string& internal_api_token,
    int32_t timeout_ms) {
  nlohmann::json body = {
      {"request_uid", hold.attempt().request_uid()},
      {"attempt_seq", hold.attempt().attempt_seq()},
      {"incarnation_id", holder.incarnation_id()},
  };
  brpc::Controller controller;
  controller.http_request().uri() = "http://" + holder.engine_uid() +
                                    (operation == AttemptControlOperation::QUERY
                                         ? "/v1/internal/attempt/query"
                                         : "/v1/internal/attempt/cancel");
  controller.http_request().set_method(brpc::HTTP_METHOD_POST);
  controller.http_request().SetHeader("Content-Type", "application/json");
  if (!set_vllm_agent_internal_token(&controller, internal_api_token)) {
    return AttemptControlResult{};
  }
  controller.request_attachment().append(body.dump());
  if (timeout_ms > 0) {
    controller.set_timeout_ms(timeout_ms);
  }
  channel->CallMethod(nullptr, &controller, nullptr, nullptr, nullptr);
  if (controller.Failed()) {
    return AttemptControlResult{};
  }
  return parse_vllm_agent_attempt_response(
      controller.http_response().status_code(),
      controller.response_attachment().to_string(),
      hold.attempt().request_uid(),
      hold.attempt().attempt_seq(),
      holder.incarnation_id());
}

AttemptControlResult call_xllm_native(
    const std::shared_ptr<brpc::Channel>& channel,
    const xllm::proto::ExecutionResourceHold& hold,
    const xllm::proto::ExecutionHolder& holder,
    AttemptControlOperation operation,
    int32_t timeout_ms) {
  xllm::proto::AttemptControlRequest request;
  xllm::proto::RequestAttemptKey* key = request.mutable_key();
  key->set_request_uid(hold.attempt().request_uid());
  if (hold.attempt().has_attempt_seq()) {
    key->set_attempt_seq(hold.attempt().attempt_seq());
  }
  key->set_incarnation_id(holder.incarnation_id());

  xllm::proto::AttemptControlResponse response;
  brpc::Controller controller;
  if (timeout_ms > 0) {
    controller.set_timeout_ms(timeout_ms);
  }
  xllm::proto::DisaggPDService_Stub stub(channel.get());
  if (operation == AttemptControlOperation::QUERY) {
    stub.QueryRequest(&controller, &request, &response, nullptr);
  } else {
    stub.CancelRequest(&controller, &request, &response, nullptr);
  }

  AttemptControlResult result;
  if (controller.Failed() || !response.ok() ||
      !same_attempt_key(response.status().key(), hold, holder)) {
    return result;
  }
  result.direct_success = true;
  result.state = response.status().state();
  result.terminal_proof = terminal_state(result.state);
  return result;
}

}  // namespace

bool constant_time_internal_token_equal(const std::string& expected,
                                        const std::string& provided) {
  return expected.size() == provided.size() && !expected.empty() &&
         CRYPTO_memcmp(expected.data(), provided.data(), expected.size()) == 0;
}

bool valid_vllm_agent_internal_token(const std::string& internal_api_token) {
  if (internal_api_token.empty() ||
      internal_api_token.size() > kMaxInternalTokenBytes) {
    return false;
  }
  return std::all_of(internal_api_token.begin(),
                     internal_api_token.end(),
                     [](unsigned char character) {
                       return character >= '!' && character <= '~';
                     });
}

bool set_vllm_agent_internal_token(brpc::Controller* controller,
                                   const std::string& internal_api_token) {
  if (controller == nullptr) {
    LOG_EVERY_N(ERROR, 100) << "Provider Agent controller is null";
    return false;
  }
  if (!valid_vllm_agent_internal_token(internal_api_token)) {
    LOG_EVERY_N(ERROR, 100)
        << "Provider Agent internal token is missing, oversized, or invalid";
    return false;
  }
  controller->http_request().SetHeader("X-Internal-Token", internal_api_token);
  return true;
}

bool set_vllm_agent_attempt_headers(brpc::Controller* controller,
                                    const std::string& request_uid,
                                    uint64_t attempt_seq,
                                    const std::string& incarnation_id,
                                    uint64_t remaining_deadline_ms) {
  if (controller == nullptr || !valid_identity(request_uid) ||
      !valid_identity(incarnation_id) || remaining_deadline_ms == 0) {
    return false;
  }
  controller->http_request().SetHeader("X-Request-UID", request_uid);
  controller->http_request().SetHeader("X-Attempt-Seq",
                                       std::to_string(attempt_seq));
  controller->http_request().SetHeader("X-Incarnation-ID", incarnation_id);
  controller->http_request().SetHeader("X-Remaining-Deadline-Ms",
                                       std::to_string(remaining_deadline_ms));
  return true;
}

std::optional<int32_t> bounded_vllm_request_timeout_ms(
    uint64_t remaining_deadline_ms,
    int32_t provider_timeout_ms) {
  if (remaining_deadline_ms == 0 || provider_timeout_ms <= 0) {
    return std::nullopt;
  }
  const uint64_t bounded = std::min<uint64_t>(
      remaining_deadline_ms, static_cast<uint64_t>(provider_timeout_ms));
  return static_cast<int32_t>(std::min<uint64_t>(
      bounded, static_cast<uint64_t>(std::numeric_limits<int32_t>::max())));
}

AttemptControlResult call_provider_attempt_control(
    xllm::proto::ProviderId provider_id,
    const std::shared_ptr<brpc::Channel>& channel,
    const xllm::proto::ExecutionResourceHold& hold,
    const xllm::proto::ExecutionHolder& holder,
    AttemptControlOperation operation,
    const std::string& internal_api_token,
    int32_t timeout_ms) {
  if (channel == nullptr || !hold.has_attempt() ||
      !hold.attempt().has_attempt_seq() || holder.engine_uid().empty() ||
      holder.incarnation_id().empty()) {
    return AttemptControlResult{};
  }
  switch (provider_id) {
    case xllm::proto::PROVIDER_ID_XLLM_NATIVE:
      return call_xllm_native(channel, hold, holder, operation, timeout_ms);
    case xllm::proto::PROVIDER_ID_VLLM_ASCEND:
      return call_vllm_agent(
          channel, hold, holder, operation, internal_api_token, timeout_ms);
    case xllm::proto::PROVIDER_ID_UNSPECIFIED:
    default:
      return AttemptControlResult{};
  }
}

AttemptControlResult parse_vllm_agent_attempt_response(
    int32_t http_status_code,
    const std::string& response_body,
    const std::string& expected_request_uid,
    uint64_t expected_attempt_seq,
    const std::string& expected_incarnation_id) {
  AttemptControlResult result;
  if (http_status_code != 200) {
    return result;
  }
  const nlohmann::json response =
      nlohmann::json::parse(response_body, nullptr, /*allow_exceptions=*/false);
  if (response.is_discarded() || !response.is_object() ||
      !response.contains("state") || !response.at("state").is_string() ||
      !response.contains("request_uid") ||
      !response.at("request_uid").is_string() ||
      !response.contains("attempt_seq") ||
      !response.at("attempt_seq").is_number_unsigned() ||
      !response.contains("incarnation_id") ||
      !response.at("incarnation_id").is_string() ||
      !response.contains("accepted") || !response.at("accepted").is_boolean() ||
      response.at("request_uid").get<std::string>() != expected_request_uid ||
      response.at("attempt_seq").get<uint64_t>() != expected_attempt_seq ||
      response.at("incarnation_id").get<std::string>() !=
          expected_incarnation_id) {
    return result;
  }
  const std::optional<xllm::proto::AttemptLifecycleState> state =
      parse_state_name(response.at("state").get<std::string>());
  if (!state.has_value()) {
    return result;
  }
  result.direct_success = true;
  result.state = *state;
  result.terminal_proof =
      response.at("accepted").get<bool>() && terminal_state(*state);
  return result;
}

bool is_retryable_vllm_agent_drain_rejection(
    int32_t http_status_code,
    const std::string& response_body,
    const std::string& expected_request_uid,
    uint64_t expected_attempt_seq,
    const std::string& expected_incarnation_id) {
  if (http_status_code != 503) {
    return false;
  }
  const nlohmann::json response =
      nlohmann::json::parse(response_body, nullptr, /*allow_exceptions=*/false);
  return !response.is_discarded() && response.is_object() &&
         response.contains("accepted") &&
         response.at("accepted").is_boolean() &&
         !response.at("accepted").get<bool>() &&
         response.contains("replayed") &&
         response.at("replayed").is_boolean() &&
         !response.at("replayed").get<bool>() && response.contains("state") &&
         response.at("state").is_string() &&
         response.at("state").get<std::string>() ==
             "ATTEMPT_LIFECYCLE_STATE_FAILED" &&
         response.contains("reason") && response.at("reason").is_string() &&
         response.at("reason").get<std::string>() ==
             "ADMISSION_REASON_ENGINE_DRAINING" &&
         response.contains("request_uid") &&
         response.at("request_uid").is_string() &&
         response.at("request_uid").get<std::string>() ==
             expected_request_uid &&
         response.contains("attempt_seq") &&
         response.at("attempt_seq").is_number_unsigned() &&
         response.at("attempt_seq").get<uint64_t>() == expected_attempt_seq &&
         response.contains("incarnation_id") &&
         response.at("incarnation_id").is_string() &&
         response.at("incarnation_id").get<std::string>() ==
             expected_incarnation_id;
}

}  // namespace xllm_service::provider
