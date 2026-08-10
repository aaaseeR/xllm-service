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

#include "placement/provider_lifecycle_actuator.h"

#include <brpc/controller.h>

#include <algorithm>
#include <cstddef>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <utility>

#include "xllm_service.pb.h"

namespace xllm_service::placement {
namespace {

constexpr uint32_t kProviderLifecycleSchemaVersion = 1;
constexpr size_t kMaxInternalTokenBytes = 4096;

bool valid_internal_token(const std::string& token) {
  return !token.empty() && token.size() <= kMaxInternalTokenBytes &&
         std::all_of(token.begin(), token.end(), [](unsigned char character) {
           return character >= '!' && character <= '~';
         });
}

bool unsigned_field(const nlohmann::json& object, const char* name) {
  return object.contains(name) && object.at(name).is_number_unsigned();
}

bool string_field(const nlohmann::json& object, const char* name) {
  return object.contains(name) && object.at(name).is_string();
}

bool has_capability(const xllm::proto::ProviderDescriptor& descriptor,
                    xllm::proto::ProviderCapability capability) {
  return std::find(descriptor.capabilities().begin(),
                   descriptor.capabilities().end(),
                   capability) != descriptor.capabilities().end();
}

std::optional<xllm::proto::ProviderLifecycleAction> proto_action(
    PlacementOperationAction action) {
  if (action == PlacementOperationAction::BEGIN_DRAIN) {
    return xllm::proto::PROVIDER_LIFECYCLE_ACTION_BEGIN_DRAIN;
  }
  if (action == PlacementOperationAction::CANCEL_DRAIN) {
    return xllm::proto::PROVIDER_LIFECYCLE_ACTION_CANCEL_DRAIN;
  }
  return std::nullopt;
}

PlacementActuatorCode actuator_code(xllm::proto::ProviderLifecycleCode code) {
  switch (code) {
    case xllm::proto::PROVIDER_LIFECYCLE_CODE_NOT_FOUND:
      return PlacementActuatorCode::NOT_FOUND;
    case xllm::proto::PROVIDER_LIFECYCLE_CODE_ACCEPTED:
      return PlacementActuatorCode::ACCEPTED;
    case xllm::proto::PROVIDER_LIFECYCLE_CODE_IN_PROGRESS:
      return PlacementActuatorCode::IN_PROGRESS;
    case xllm::proto::PROVIDER_LIFECYCLE_CODE_SUCCEEDED:
      return PlacementActuatorCode::SUCCEEDED;
    case xllm::proto::PROVIDER_LIFECYCLE_CODE_RETRYABLE_ERROR:
      return PlacementActuatorCode::RETRYABLE_ERROR;
    case xllm::proto::PROVIDER_LIFECYCLE_CODE_TERMINAL_ERROR:
      return PlacementActuatorCode::TERMINAL_ERROR;
    case xllm::proto::PROVIDER_LIFECYCLE_CODE_FENCED:
      return PlacementActuatorCode::FENCED;
    case xllm::proto::PROVIDER_LIFECYCLE_CODE_CONFLICT:
      return PlacementActuatorCode::CONFLICT;
    case xllm::proto::PROVIDER_LIFECYCLE_CODE_INVALID:
    case xllm::proto::PROVIDER_LIFECYCLE_CODE_UNSPECIFIED:
    default:
      return PlacementActuatorCode::INVALID;
  }
}

PlacementLifecycleState lifecycle_state(xllm::proto::EngineLifecycle state) {
  switch (state) {
    case xllm::proto::ENGINE_LIFECYCLE_STARTING:
      return PlacementLifecycleState::LOADING;
    case xllm::proto::ENGINE_LIFECYCLE_READY:
      return PlacementLifecycleState::READY;
    case xllm::proto::ENGINE_LIFECYCLE_DRAINING:
      return PlacementLifecycleState::DRAINING;
    case xllm::proto::ENGINE_LIFECYCLE_FENCED:
    case xllm::proto::ENGINE_LIFECYCLE_MEMBERSHIP_LOST:
    case xllm::proto::ENGINE_LIFECYCLE_UNSPECIFIED:
    default:
      return PlacementLifecycleState::FAILED;
  }
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
                        const PlacementOperationIntent& intent) {
  return descriptor.identity().provider_id() == intent.pool.provider_id &&
         descriptor.identity().engine_uid() == intent.engine_uid &&
         descriptor.identity().incarnation_id() == intent.engine_incarnation &&
         descriptor.profile_digest() == intent.pool.profile_digest &&
         descriptor.model().model_revision() == intent.pool.model_revision &&
         descriptor.serving().role() == intent.pool.role &&
         has_capability(descriptor, xllm::proto::PROVIDER_CAPABILITY_DRAIN);
}

bool response_matches(const xllm::proto::ProviderLifecycleResponse& response,
                      const PlacementOperationIntent& intent,
                      xllm::proto::ProviderLifecycleAction action) {
  return response.operation_id() == intent.operation_id &&
         response.action() == action &&
         response.leader_incarnation() == intent.leader_incarnation &&
         response.leader_epoch() == intent.leader_epoch &&
         response.desired_generation() == intent.desired_generation &&
         response.engine_uid() == intent.engine_uid &&
         response.engine_incarnation() == intent.engine_incarnation;
}

}  // namespace

BrpcProviderLifecycleTransport::BrpcProviderLifecycleTransport(
    BrpcProviderLifecycleTransportConfig config)
    : config_(config) {}

std::shared_ptr<brpc::Channel> BrpcProviderLifecycleTransport::channel(
    const std::string& address) {
  if (address.empty() || config_.max_channels == 0) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  const auto existing = channels_.find(address);
  if (existing != channels_.end()) {
    return existing->second;
  }
  if (channels_.size() >= config_.max_channels) {
    return nullptr;
  }
  auto channel = std::make_shared<brpc::Channel>();
  brpc::ChannelOptions options;
  options.max_retry = 0;
  options.timeout_ms = config_.timeout_ms;
  if (channel->Init(address.c_str(), "", &options) != 0) {
    return nullptr;
  }
  channels_.emplace(address, channel);
  return channel;
}

ProviderLifecycleTransportStatus BrpcProviderLifecycleTransport::call(
    const xllm::proto::ProviderDescriptor& descriptor,
    const xllm::proto::ProviderLifecycleCommand& command,
    bool query,
    xllm::proto::ProviderLifecycleResponse* response) {
  if (response == nullptr || config_.timeout_ms <= 0 ||
      descriptor.identity().provider_id() !=
          xllm::proto::PROVIDER_ID_XLLM_NATIVE ||
      descriptor.endpoint().control_transport() != "brpc") {
    return ProviderLifecycleTransportStatus::INVALID;
  }
  const std::shared_ptr<brpc::Channel> lifecycle_channel =
      channel(descriptor.endpoint().address());
  if (lifecycle_channel == nullptr) {
    return ProviderLifecycleTransportStatus::UNAVAILABLE;
  }
  response->Clear();
  brpc::Controller controller;
  controller.set_timeout_ms(config_.timeout_ms);
  xllm::proto::XllmAPIService_Stub stub(lifecycle_channel.get());
  if (query) {
    stub.QueryProviderLifecycle(&controller, &command, response, nullptr);
  } else {
    stub.ExecuteProviderLifecycle(&controller, &command, response, nullptr);
  }
  return controller.Failed() ? ProviderLifecycleTransportStatus::OUTCOME_UNKNOWN
                             : ProviderLifecycleTransportStatus::OK;
}

HttpProviderLifecycleTransport::HttpProviderLifecycleTransport(
    HttpProviderLifecycleTransportConfig config)
    : config_(std::move(config)) {}

std::shared_ptr<brpc::Channel> HttpProviderLifecycleTransport::channel(
    const std::string& address) {
  if (address.empty() || config_.max_channels == 0) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  const auto existing = channels_.find(address);
  if (existing != channels_.end()) {
    return existing->second;
  }
  if (channels_.size() >= config_.max_channels) {
    return nullptr;
  }
  auto channel = std::make_shared<brpc::Channel>();
  brpc::ChannelOptions options;
  options.protocol = "http";
  options.max_retry = 0;
  options.timeout_ms = config_.timeout_ms;
  if (channel->Init(address.c_str(), "", &options) != 0) {
    return nullptr;
  }
  channels_.emplace(address, channel);
  return channel;
}

ProviderLifecycleTransportStatus HttpProviderLifecycleTransport::call(
    const xllm::proto::ProviderDescriptor& descriptor,
    const xllm::proto::ProviderLifecycleCommand& command,
    bool query,
    xllm::proto::ProviderLifecycleResponse* response) {
  if (response == nullptr || config_.timeout_ms <= 0 ||
      config_.max_response_bytes == 0 ||
      !valid_internal_token(config_.internal_api_token) ||
      descriptor.identity().provider_id() !=
          xllm::proto::PROVIDER_ID_VLLM_ASCEND ||
      descriptor.endpoint().control_transport() != "http") {
    return ProviderLifecycleTransportStatus::INVALID;
  }
  const std::shared_ptr<brpc::Channel> lifecycle_channel =
      channel(descriptor.endpoint().address());
  if (lifecycle_channel == nullptr) {
    return ProviderLifecycleTransportStatus::UNAVAILABLE;
  }
  const nlohmann::json body = {
      {"schema_version", command.schema_version()},
      {"operation_id", command.operation_id()},
      {"leader_incarnation", command.leader_incarnation()},
      {"leader_epoch", command.leader_epoch()},
      {"desired_generation", command.desired_generation()},
      {"engine_uid", command.engine_uid()},
      {"engine_incarnation", command.engine_incarnation()},
      {"action", xllm::proto::ProviderLifecycleAction_Name(command.action())},
  };
  brpc::Controller controller;
  controller.set_timeout_ms(config_.timeout_ms);
  controller.http_request().uri() = "http://" +
                                    descriptor.endpoint().address() +
                                    (query ? "/v1/internal/lifecycle/query"
                                           : "/v1/internal/lifecycle/execute");
  controller.http_request().set_method(brpc::HTTP_METHOD_POST);
  controller.http_request().SetHeader("Content-Type", "application/json");
  controller.http_request().SetHeader("X-Internal-Token",
                                      config_.internal_api_token);
  controller.request_attachment().append(body.dump());
  lifecycle_channel->CallMethod(
      nullptr, &controller, nullptr, nullptr, nullptr);
  if (controller.Failed()) {
    return ProviderLifecycleTransportStatus::OUTCOME_UNKNOWN;
  }
  if (controller.response_attachment().size() > config_.max_response_bytes) {
    return ProviderLifecycleTransportStatus::INVALID;
  }
  return parse_vllm_lifecycle_response(
             controller.http_response().status_code(),
             controller.response_attachment().to_string(),
             response)
             ? ProviderLifecycleTransportStatus::OK
             : ProviderLifecycleTransportStatus::INVALID;
}

ProviderLifecycleTransportRouter::ProviderLifecycleTransportRouter(
    ProviderLifecycleTransport* xllm_native,
    ProviderLifecycleTransport* vllm_ascend)
    : xllm_native_(xllm_native), vllm_ascend_(vllm_ascend) {}

ProviderLifecycleTransportStatus ProviderLifecycleTransportRouter::call(
    const xllm::proto::ProviderDescriptor& descriptor,
    const xllm::proto::ProviderLifecycleCommand& command,
    bool query,
    xllm::proto::ProviderLifecycleResponse* response) {
  ProviderLifecycleTransport* transport = nullptr;
  switch (descriptor.identity().provider_id()) {
    case xllm::proto::PROVIDER_ID_XLLM_NATIVE:
      transport = xllm_native_;
      break;
    case xllm::proto::PROVIDER_ID_VLLM_ASCEND:
      transport = vllm_ascend_;
      break;
    case xllm::proto::PROVIDER_ID_UNSPECIFIED:
    default:
      break;
  }
  return transport == nullptr
             ? ProviderLifecycleTransportStatus::INVALID
             : transport->call(descriptor, command, query, response);
}

bool parse_vllm_lifecycle_response(
    int32_t http_status_code,
    const std::string& response_body,
    xllm::proto::ProviderLifecycleResponse* response) {
  if (http_status_code != 200 || response == nullptr) {
    return false;
  }
  const nlohmann::json body =
      nlohmann::json::parse(response_body, nullptr, /*allow_exceptions=*/false);
  if (body.is_discarded() || !body.is_object() || !string_field(body, "code") ||
      !string_field(body, "operation_id") || !string_field(body, "action") ||
      !string_field(body, "leader_incarnation") ||
      !unsigned_field(body, "leader_epoch") ||
      !unsigned_field(body, "desired_generation") ||
      !string_field(body, "engine_uid") ||
      !string_field(body, "engine_incarnation") ||
      !string_field(body, "lifecycle") || !body.contains("drain") ||
      !body.at("drain").is_object() || !body.contains("replayed") ||
      !body.at("replayed").is_boolean() || !string_field(body, "message")) {
    return false;
  }
  const nlohmann::json& drain = body.at("drain");
  if (!drain.contains("admission_closed") ||
      !drain.at("admission_closed").is_boolean() ||
      !unsigned_field(drain, "prefill_queue") ||
      !unsigned_field(drain, "active_transfers") ||
      !unsigned_field(drain, "active_reservations") ||
      !unsigned_field(drain, "decode_sequences") ||
      !unsigned_field(drain, "pending_output") ||
      !unsigned_field(drain, "pending_cleanup")) {
    return false;
  }
  xllm::proto::ProviderLifecycleCode code;
  xllm::proto::ProviderLifecycleAction action;
  xllm::proto::EngineLifecycle lifecycle;
  if (!xllm::proto::ProviderLifecycleCode_Parse(
          body.at("code").get<std::string>(), &code) ||
      !xllm::proto::ProviderLifecycleAction_Parse(
          body.at("action").get<std::string>(), &action) ||
      !xllm::proto::EngineLifecycle_Parse(
          body.at("lifecycle").get<std::string>(), &lifecycle)) {
    return false;
  }
  response->Clear();
  response->set_code(code);
  response->set_operation_id(body.at("operation_id").get<std::string>());
  response->set_action(action);
  response->set_leader_incarnation(
      body.at("leader_incarnation").get<std::string>());
  response->set_leader_epoch(body.at("leader_epoch").get<uint64_t>());
  response->set_desired_generation(
      body.at("desired_generation").get<uint64_t>());
  response->set_engine_uid(body.at("engine_uid").get<std::string>());
  response->set_engine_incarnation(
      body.at("engine_incarnation").get<std::string>());
  response->set_lifecycle(lifecycle);
  response->set_replayed(body.at("replayed").get<bool>());
  response->set_message(body.at("message").get<std::string>());
  xllm::proto::ProviderDrainProof* proof = response->mutable_drain();
  proof->set_admission_closed(drain.at("admission_closed").get<bool>());
  proof->set_prefill_queue(drain.at("prefill_queue").get<uint64_t>());
  proof->set_active_transfers(drain.at("active_transfers").get<uint64_t>());
  proof->set_active_reservations(
      drain.at("active_reservations").get<uint64_t>());
  proof->set_decode_sequences(drain.at("decode_sequences").get<uint64_t>());
  proof->set_pending_output(drain.at("pending_output").get<uint64_t>());
  proof->set_pending_cleanup(drain.at("pending_cleanup").get<uint64_t>());
  return true;
}

ProviderPlacementActuator::ProviderPlacementActuator(
    provider::EngineRegistry* registry,
    ProviderLifecycleTransport* transport,
    PlacementDeploymentActuator* deployment)
    : registry_(registry), transport_(transport), deployment_(deployment) {}

PlacementActuatorResponse ProviderPlacementActuator::execute(
    const PlacementOperationIntent& intent) {
  if (intent.action == PlacementOperationAction::CREATE ||
      intent.action == PlacementOperationAction::TERMINATE) {
    return deployment_ == nullptr
               ? error_response(intent,
                                PlacementActuatorCode::INVALID,
                                "deployment actuator is unavailable")
               : deployment_->execute(intent);
  }
  return lifecycle_call(intent, false);
}

PlacementActuatorResponse ProviderPlacementActuator::query(
    const PlacementOperationIntent& intent) {
  if (intent.action == PlacementOperationAction::CREATE ||
      intent.action == PlacementOperationAction::TERMINATE) {
    return deployment_ == nullptr
               ? error_response(intent,
                                PlacementActuatorCode::INVALID,
                                "deployment actuator is unavailable")
               : deployment_->query(intent);
  }
  return lifecycle_call(intent, true);
}

PlacementActuatorResponse ProviderPlacementActuator::lifecycle_call(
    const PlacementOperationIntent& intent,
    bool query) {
  const std::optional<xllm::proto::ProviderLifecycleAction> action =
      proto_action(intent.action);
  if (registry_ == nullptr || transport_ == nullptr || !action.has_value() ||
      !valid_placement_operation_intent(intent)) {
    return error_response(
        intent, PlacementActuatorCode::INVALID, "invalid lifecycle intent");
  }
  xllm::proto::ProviderEngineKey key;
  key.set_provider_id(intent.pool.provider_id);
  key.set_profile_digest(intent.pool.profile_digest);
  key.set_engine_uid(intent.engine_uid);
  key.set_incarnation_id(intent.engine_incarnation);
  const std::optional<xllm::proto::ProviderDescriptor> descriptor =
      registry_->find_member(key);
  if (!descriptor.has_value() || !descriptor_matches(*descriptor, intent)) {
    return error_response(intent,
                          PlacementActuatorCode::FENCED,
                          "lifecycle target is not the current member");
  }

  xllm::proto::ProviderLifecycleCommand command;
  command.set_schema_version(kProviderLifecycleSchemaVersion);
  command.set_operation_id(intent.operation_id);
  command.set_leader_incarnation(intent.leader_incarnation);
  command.set_leader_epoch(intent.leader_epoch);
  command.set_desired_generation(intent.desired_generation);
  command.set_engine_uid(intent.engine_uid);
  command.set_engine_incarnation(intent.engine_incarnation);
  command.set_action(*action);

  xllm::proto::ProviderLifecycleResponse response;
  const ProviderLifecycleTransportStatus transport_status =
      transport_->call(*descriptor, command, query, &response);
  if (transport_status == ProviderLifecycleTransportStatus::UNAVAILABLE) {
    return error_response(intent,
                          PlacementActuatorCode::RETRYABLE_ERROR,
                          "lifecycle transport unavailable");
  }
  if (transport_status == ProviderLifecycleTransportStatus::OUTCOME_UNKNOWN) {
    return error_response(intent,
                          PlacementActuatorCode::UNKNOWN,
                          "lifecycle RPC outcome unknown");
  }
  if (transport_status != ProviderLifecycleTransportStatus::OK ||
      !response_matches(response, intent, *action)) {
    return error_response(intent,
                          PlacementActuatorCode::INVALID,
                          "invalid lifecycle RPC response");
  }
  PlacementDrainProof proof{
      .admission_closed = response.drain().admission_closed(),
      .prefill_queue = response.drain().prefill_queue(),
      .active_transfers = response.drain().active_transfers(),
      .active_reservations = response.drain().active_reservations(),
      .decode_sequences = response.drain().decode_sequences(),
      .pending_output = response.drain().pending_output(),
      .pending_cleanup = response.drain().pending_cleanup(),
  };
  return PlacementActuatorResponse{
      .code = actuator_code(response.code()),
      .engine_uid = response.engine_uid(),
      .engine_incarnation = response.engine_incarnation(),
      .lifecycle_state = lifecycle_state(response.lifecycle()),
      .drain = proof,
      .message = response.message(),
  };
}

}  // namespace xllm_service::placement
