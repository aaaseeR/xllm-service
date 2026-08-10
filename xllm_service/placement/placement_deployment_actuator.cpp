/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "placement/placement_deployment_actuator.h"

#include <brpc/controller.h>

#include <algorithm>
#include <chrono>
#include <nlohmann/json.hpp>
#include <utility>

namespace xllm_service::placement {
namespace {

constexpr uint32_t kDeploymentSchemaVersion = 1;
constexpr size_t kMaxInternalTokenBytes = 4096;

bool valid_token(const std::string& value) {
  return !value.empty() && value.size() <= kMaxInternalTokenBytes &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return character >= '!' && character <= '~';
         });
}

bool exact_fields(const nlohmann::json& object,
                  std::initializer_list<const char*> fields) {
  if (!object.is_object() || object.size() != fields.size()) {
    return false;
  }
  return std::all_of(
      fields.begin(), fields.end(), [&object](const char* field) {
        return object.contains(field);
      });
}

bool string_field(const nlohmann::json& object, const char* name) {
  return object.at(name).is_string();
}

bool unsigned_field(const nlohmann::json& object, const char* name) {
  return object.at(name).is_number_unsigned();
}

nlohmann::json pool_json(const PlacementPoolKey& pool) {
  return nlohmann::json{
      {"provider_id", static_cast<int32_t>(pool.provider_id)},
      {"model_revision", pool.model_revision},
      {"role", static_cast<int32_t>(pool.role)},
      {"profile_digest", pool.profile_digest},
  };
}

bool pool_matches(const nlohmann::json& value, const PlacementPoolKey& pool) {
  if (!exact_fields(
          value, {"provider_id", "model_revision", "role", "profile_digest"}) ||
      !value.at("provider_id").is_number_integer() ||
      !string_field(value, "model_revision") ||
      !value.at("role").is_number_integer() ||
      !string_field(value, "profile_digest")) {
    return false;
  }
  return value.at("provider_id").get<int32_t>() ==
             static_cast<int32_t>(pool.provider_id) &&
         value.at("model_revision").get<std::string>() == pool.model_revision &&
         value.at("role").get<int32_t>() == static_cast<int32_t>(pool.role) &&
         value.at("profile_digest").get<std::string>() == pool.profile_digest;
}

bool parse_code(const std::string& value, PlacementActuatorCode* code) {
  for (PlacementActuatorCode candidate :
       {PlacementActuatorCode::NOT_FOUND,
        PlacementActuatorCode::ACCEPTED,
        PlacementActuatorCode::IN_PROGRESS,
        PlacementActuatorCode::SUCCEEDED,
        PlacementActuatorCode::RETRYABLE_ERROR,
        PlacementActuatorCode::TERMINAL_ERROR,
        PlacementActuatorCode::UNKNOWN,
        PlacementActuatorCode::FENCED,
        PlacementActuatorCode::CONFLICT,
        PlacementActuatorCode::INVALID}) {
    if (value == placement_actuator_code_name(candidate)) {
      *code = candidate;
      return true;
    }
  }
  return false;
}

bool parse_lifecycle(const std::string& value,
                     PlacementLifecycleState* lifecycle) {
  for (PlacementLifecycleState candidate : {PlacementLifecycleState::ABSENT,
                                            PlacementLifecycleState::LOADING,
                                            PlacementLifecycleState::WARMING,
                                            PlacementLifecycleState::READY,
                                            PlacementLifecycleState::DRAINING,
                                            PlacementLifecycleState::UNLOADING,
                                            PlacementLifecycleState::FAILED}) {
    if (value == placement_lifecycle_state_name(candidate)) {
      *lifecycle = candidate;
      return true;
    }
  }
  return false;
}

PlacementActuatorResponse error_response(const PlacementOperationIntent& intent,
                                         PlacementActuatorCode code,
                                         std::string message) {
  return PlacementActuatorResponse{
      .code = code,
      .engine_uid = intent.engine_uid,
      .engine_incarnation = intent.engine_incarnation,
      .lifecycle_state = PlacementLifecycleState::FAILED,
      .message = std::move(message),
  };
}

bool descriptor_matches(const xllm::proto::ProviderDescriptor& descriptor,
                        const PlacementOperationIntent& intent,
                        const PlacementActuatorResponse& response) {
  return descriptor.identity().provider_id() == intent.pool.provider_id &&
         descriptor.identity().engine_uid() == response.engine_uid &&
         descriptor.identity().incarnation_id() ==
             response.engine_incarnation &&
         descriptor.profile_digest() == intent.pool.profile_digest &&
         descriptor.model().model_revision() == intent.pool.model_revision &&
         descriptor.serving().role() == intent.pool.role;
}

xllm::proto::ProviderEngineKey engine_key(
    const PlacementOperationIntent& intent,
    const PlacementActuatorResponse& response) {
  xllm::proto::ProviderEngineKey key;
  key.set_provider_id(intent.pool.provider_id);
  key.set_profile_digest(intent.pool.profile_digest);
  key.set_engine_uid(response.engine_uid);
  key.set_incarnation_id(response.engine_incarnation);
  return key;
}

xllm::proto::ProviderEngineKey intent_engine_key(
    const PlacementOperationIntent& intent) {
  xllm::proto::ProviderEngineKey key;
  key.set_provider_id(intent.pool.provider_id);
  key.set_profile_digest(intent.pool.profile_digest);
  key.set_engine_uid(intent.engine_uid);
  key.set_incarnation_id(intent.engine_incarnation);
  return key;
}

uint64_t steady_monotonic_ms() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

}  // namespace

HttpPlacementDeploymentActuator::HttpPlacementDeploymentActuator(
    HttpPlacementDeploymentActuatorConfig config)
    : config_(std::move(config)) {
  if (config_.address.empty() || config_.timeout_ms <= 0 ||
      config_.max_response_bytes == 0 ||
      !valid_token(config_.internal_api_token)) {
    return;
  }
  auto channel = std::make_shared<brpc::Channel>();
  brpc::ChannelOptions options;
  options.protocol = "http";
  options.max_retry = 0;
  options.timeout_ms = config_.timeout_ms;
  if (channel->Init(config_.address.c_str(), "", &options) == 0) {
    channel_ = std::move(channel);
  }
}

PlacementActuatorResponse HttpPlacementDeploymentActuator::execute(
    const PlacementOperationIntent& intent) {
  return call(intent, false);
}

PlacementActuatorResponse HttpPlacementDeploymentActuator::query(
    const PlacementOperationIntent& intent) {
  return call(intent, true);
}

PlacementActuatorResponse HttpPlacementDeploymentActuator::call(
    const PlacementOperationIntent& intent,
    bool query) {
  if (channel_ == nullptr || !valid_placement_operation_intent(intent) ||
      (intent.action != PlacementOperationAction::CREATE &&
       intent.action != PlacementOperationAction::TERMINATE)) {
    return error_response(
        intent, PlacementActuatorCode::INVALID, "invalid deployment intent");
  }
  const nlohmann::json body{
      {"schema_version", kDeploymentSchemaVersion},
      {"operation_id", intent.operation_id},
      {"action", placement_operation_action_name(intent.action)},
      {"leader_incarnation", intent.leader_incarnation},
      {"leader_epoch", intent.leader_epoch},
      {"desired_generation", intent.desired_generation},
      {"ordinal", intent.ordinal},
      {"pool", pool_json(intent.pool)},
      {"engine_uid", intent.engine_uid},
      {"engine_incarnation", intent.engine_incarnation},
  };
  brpc::Controller controller;
  controller.set_timeout_ms(config_.timeout_ms);
  controller.http_request().uri() = "http://" + config_.address +
                                    (query ? "/v1/internal/placement/query"
                                           : "/v1/internal/placement/execute");
  controller.http_request().set_method(brpc::HTTP_METHOD_POST);
  controller.http_request().SetHeader("Content-Type", "application/json");
  controller.http_request().SetHeader("X-Internal-Token",
                                      config_.internal_api_token);
  controller.request_attachment().append(body.dump());
  channel_->CallMethod(nullptr, &controller, nullptr, nullptr, nullptr);
  if (controller.Failed()) {
    return error_response(intent,
                          query ? PlacementActuatorCode::RETRYABLE_ERROR
                                : PlacementActuatorCode::UNKNOWN,
                          query ? "deployment query unavailable"
                                : "deployment execute outcome unknown");
  }
  if (controller.http_response().status_code() != 200 ||
      controller.response_attachment().size() > config_.max_response_bytes) {
    return error_response(intent,
                          query ? PlacementActuatorCode::RETRYABLE_ERROR
                                : PlacementActuatorCode::UNKNOWN,
                          "invalid deployment HTTP response");
  }
  PlacementActuatorResponse response;
  if (!parse_placement_deployment_response(
          controller.response_attachment().to_string(), intent, &response)) {
    return error_response(intent,
                          PlacementActuatorCode::INVALID,
                          "invalid deployment response contract");
  }
  return response;
}

RegistryVerifiedPlacementDeploymentActuator::
    RegistryVerifiedPlacementDeploymentActuator(
        provider::EngineRegistry* registry,
        PlacementDeploymentActuator* backend,
        std::function<uint64_t()> monotonic_clock)
    : registry_(registry),
      backend_(backend),
      monotonic_clock_(std::move(monotonic_clock)) {
  if (!monotonic_clock_) {
    monotonic_clock_ = steady_monotonic_ms;
  }
}

PlacementActuatorResponse RegistryVerifiedPlacementDeploymentActuator::execute(
    const PlacementOperationIntent& intent) {
  if (backend_ == nullptr) {
    return error_response(intent,
                          PlacementActuatorCode::INVALID,
                          "deployment backend is unavailable");
  }
  return verify(intent, backend_->execute(intent));
}

PlacementActuatorResponse RegistryVerifiedPlacementDeploymentActuator::query(
    const PlacementOperationIntent& intent) {
  if (backend_ == nullptr) {
    return error_response(intent,
                          PlacementActuatorCode::INVALID,
                          "deployment backend is unavailable");
  }
  return verify(intent, backend_->query(intent));
}

PlacementActuatorResponse RegistryVerifiedPlacementDeploymentActuator::verify(
    const PlacementOperationIntent& intent,
    PlacementActuatorResponse response) {
  if (registry_ == nullptr || !valid_placement_operation_intent(intent) ||
      (intent.action != PlacementOperationAction::CREATE &&
       intent.action != PlacementOperationAction::TERMINATE)) {
    return error_response(
        intent, PlacementActuatorCode::INVALID, "invalid deployment proof");
  }
  if (intent.action == PlacementOperationAction::CREATE) {
    if (response.code != PlacementActuatorCode::SUCCEEDED) {
      return response;
    }
    const xllm::proto::ProviderEngineKey key = engine_key(intent, response);
    const std::optional<xllm::proto::ProviderDescriptor> descriptor =
        registry_->find_member(key);
    const std::optional<xllm::proto::EngineState> state =
        registry_->find_state(key);
    const uint64_t now_ms = monotonic_clock_();
    if (descriptor.has_value() && state.has_value() &&
        descriptor_matches(*descriptor, intent, response) &&
        state->lifecycle() == xllm::proto::ENGINE_LIFECYCLE_READY &&
        registry_->state_freshness(key, now_ms) ==
            provider::EngineStateFreshness::FRESH &&
        registry_->is_schedulable(key, now_ms)) {
      response.ready_proven = true;
      response.lifecycle_state = PlacementLifecycleState::READY;
      return response;
    }
    response.code = PlacementActuatorCode::IN_PROGRESS;
    response.ready_proven = false;
    response.lifecycle_state = PlacementLifecycleState::LOADING;
    response.message = "waiting for fresh Registry READY proof";
    return response;
  }

  const xllm::proto::ProviderEngineKey key = intent_engine_key(intent);
  const bool absent = !registry_->find_member(key).has_value();
  if ((response.code == PlacementActuatorCode::SUCCEEDED &&
       response.termination_proven) ||
      ((response.code == PlacementActuatorCode::SUCCEEDED ||
        response.code == PlacementActuatorCode::NOT_FOUND) &&
       absent)) {
    response.code = PlacementActuatorCode::SUCCEEDED;
    response.engine_uid = intent.engine_uid;
    response.engine_incarnation = intent.engine_incarnation;
    response.lifecycle_state = PlacementLifecycleState::ABSENT;
    response.termination_proven = true;
    return response;
  }
  if (response.code == PlacementActuatorCode::SUCCEEDED) {
    response.code = PlacementActuatorCode::IN_PROGRESS;
    response.lifecycle_state = PlacementLifecycleState::UNLOADING;
    response.termination_proven = false;
    response.message = "waiting for Registry absence or termination proof";
  }
  return response;
}

bool parse_placement_deployment_response(const std::string& value,
                                         const PlacementOperationIntent& intent,
                                         PlacementActuatorResponse* response) {
  if (response == nullptr) {
    return false;
  }
  const nlohmann::json json =
      nlohmann::json::parse(value, nullptr, /*allow_exceptions=*/false);
  if (json.is_discarded() ||
      !exact_fields(json,
                    {"schema_version",
                     "code",
                     "operation_id",
                     "action",
                     "leader_incarnation",
                     "leader_epoch",
                     "desired_generation",
                     "pool",
                     "engine_uid",
                     "engine_incarnation",
                     "lifecycle",
                     "termination_proven",
                     "replayed",
                     "message"}) ||
      !unsigned_field(json, "schema_version") ||
      json.at("schema_version").get<uint64_t>() != kDeploymentSchemaVersion ||
      !string_field(json, "code") || !string_field(json, "operation_id") ||
      !string_field(json, "action") ||
      !string_field(json, "leader_incarnation") ||
      !unsigned_field(json, "leader_epoch") ||
      !unsigned_field(json, "desired_generation") ||
      !pool_matches(json.at("pool"), intent.pool) ||
      !string_field(json, "engine_uid") ||
      !string_field(json, "engine_incarnation") ||
      !string_field(json, "lifecycle") ||
      !json.at("termination_proven").is_boolean() ||
      !json.at("replayed").is_boolean() || !string_field(json, "message") ||
      json.at("operation_id").get<std::string>() != intent.operation_id ||
      json.at("action").get<std::string>() !=
          placement_operation_action_name(intent.action) ||
      json.at("leader_incarnation").get<std::string>() !=
          intent.leader_incarnation ||
      json.at("leader_epoch").get<uint64_t>() != intent.leader_epoch ||
      json.at("desired_generation").get<uint64_t>() !=
          intent.desired_generation) {
    return false;
  }
  PlacementActuatorCode code;
  PlacementLifecycleState lifecycle;
  const std::string engine_uid = json.at("engine_uid").get<std::string>();
  const std::string engine_incarnation =
      json.at("engine_incarnation").get<std::string>();
  const std::string message = json.at("message").get<std::string>();
  if (!parse_code(json.at("code").get<std::string>(), &code) ||
      !parse_lifecycle(json.at("lifecycle").get<std::string>(), &lifecycle) ||
      message.size() > kMaxPlacementActuatorMessageBytes ||
      !std::all_of(message.begin(),
                   message.end(),
                   [](unsigned char value) {
                     return value != 0 && value != '\r' && value != '\n';
                   }) ||
      (!engine_uid.empty() && !valid_placement_identity(engine_uid)) ||
      (!engine_incarnation.empty() &&
       !valid_placement_identity(engine_incarnation)) ||
      engine_uid.empty() != engine_incarnation.empty()) {
    return false;
  }
  *response = PlacementActuatorResponse{
      .code = code,
      .engine_uid = engine_uid,
      .engine_incarnation = engine_incarnation,
      .lifecycle_state = lifecycle,
      .termination_proven = json.at("termination_proven").get<bool>(),
      .message = message,
  };
  return true;
}

}  // namespace xllm_service::placement
