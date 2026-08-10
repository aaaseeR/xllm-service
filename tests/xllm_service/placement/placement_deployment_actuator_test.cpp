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

#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <brpc/server.h>
#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <string>

#include "xllm_http_service.pb.h"

namespace xllm_service::placement {
namespace {

PlacementOperationIntent intent(
    PlacementOperationAction action = PlacementOperationAction::CREATE) {
  PlacementOperationIntent value{
      .action = action,
      .pool =
          PlacementPoolKey{
              .provider_id = xllm::proto::PROVIDER_ID_XLLM_NATIVE,
              .model_revision = "model-r1",
              .role = xllm::proto::ENGINE_ROLE_PREFILL,
              .profile_digest = "profile-a",
          },
      .engine_uid =
          action == PlacementOperationAction::CREATE ? "" : "engine-1",
      .engine_incarnation =
          action == PlacementOperationAction::CREATE ? "" : "inc-1",
      .leader_incarnation = "leader-1",
      .leader_epoch = 7,
      .desired_generation = 9,
      .ordinal = 2,
  };
  value.operation_id = make_placement_operation_id(
      PlacementLeaderIdentity{
          .address = "service:2888",
          .incarnation = value.leader_incarnation,
          .epoch = value.leader_epoch,
      },
      value.desired_generation,
      value.pool,
      value.action,
      value.ordinal,
      value.engine_uid,
      value.engine_incarnation);
  return value;
}

nlohmann::json response_json(const PlacementOperationIntent& operation) {
  return nlohmann::json{
      {"schema_version", 1},
      {"code", "SUCCEEDED"},
      {"operation_id", operation.operation_id},
      {"action", placement_operation_action_name(operation.action)},
      {"leader_incarnation", operation.leader_incarnation},
      {"leader_epoch", operation.leader_epoch},
      {"desired_generation", operation.desired_generation},
      {"pool",
       {{"provider_id", static_cast<int32_t>(operation.pool.provider_id)},
        {"model_revision", operation.pool.model_revision},
        {"role", static_cast<int32_t>(operation.pool.role)},
        {"profile_digest", operation.pool.profile_digest}}},
      {"engine_uid", "engine-1"},
      {"engine_incarnation", "inc-1"},
      {"lifecycle",
       operation.action == PlacementOperationAction::CREATE ? "READY"
                                                            : "ABSENT"},
      {"termination_proven",
       operation.action == PlacementOperationAction::TERMINATE},
      {"replayed", false},
      {"message", "completed"},
  };
}

class FakeDeploymentGateway final : public proto::XllmHttpService {
 public:
  void Models(google::protobuf::RpcController* controller,
              const proto::HttpRequest*,
              proto::HttpResponse*,
              google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    auto* brpc_controller = static_cast<brpc::Controller*>(controller);
    path = brpc_controller->http_request().uri().path();
    const std::string* header =
        brpc_controller->http_request().GetHeader("X-Internal-Token");
    token = header == nullptr ? "" : *header;
    request = nlohmann::json::parse(
        brpc_controller->request_attachment().to_string());
    brpc_controller->http_response().set_status_code(http_status);
    PlacementOperationIntent operation = intent(
        request.at("action") == "CREATE" ? PlacementOperationAction::CREATE
                                         : PlacementOperationAction::TERMINATE);
    operation.operation_id = request.at("operation_id").get<std::string>();
    operation.leader_incarnation =
        request.at("leader_incarnation").get<std::string>();
    operation.leader_epoch = request.at("leader_epoch").get<uint64_t>();
    operation.desired_generation =
        request.at("desired_generation").get<uint64_t>();
    nlohmann::json response = response_json(operation);
    if (corrupt_echo) {
      response["leader_epoch"] = operation.leader_epoch + 1;
    }
    brpc_controller->response_attachment().append(response.dump());
  }

  int32_t http_status = 200;
  bool corrupt_echo = false;
  std::string path;
  std::string token;
  nlohmann::json request;
};

class FakeDeploymentBackend final : public PlacementDeploymentActuator {
 public:
  PlacementActuatorResponse execute(const PlacementOperationIntent&) override {
    ++execute_calls;
    return response;
  }

  PlacementActuatorResponse query(const PlacementOperationIntent&) override {
    ++query_calls;
    return response;
  }

  PlacementActuatorResponse response{
      .code = PlacementActuatorCode::SUCCEEDED,
      .engine_uid = "engine-1",
      .engine_incarnation = "inc-1",
      .lifecycle_state = PlacementLifecycleState::READY,
  };
  uint32_t execute_calls = 0;
  uint32_t query_calls = 0;
};

xllm::proto::ProviderDescriptor descriptor() {
  xllm::proto::ProviderDescriptor value;
  value.set_contract_version(provider::kProviderContractVersion);
  auto* identity = value.mutable_identity();
  identity->set_engine_uid("engine-1");
  identity->set_incarnation_id("inc-1");
  identity->set_provider_id(xllm::proto::PROVIDER_ID_XLLM_NATIVE);
  identity->set_runtime_family("xllm");
  identity->set_runtime_version("runtime-v1");
  identity->set_plugin_version("plugin-v1");
  identity->set_hardware_runtime_version("cpu-simulated-hbm-v1");
  identity->set_protocol_version(provider::kProviderContractVersion);
  value.mutable_endpoint()->set_control_transport("brpc");
  value.mutable_endpoint()->set_data_transport("mooncake");
  value.mutable_endpoint()->set_address("127.0.0.1:8000");
  value.mutable_serving()->set_role(xllm::proto::ENGINE_ROLE_PREFILL);
  auto* mode = value.mutable_serving()->add_execution_modes();
  mode->set_mode(xllm::proto::EXECUTION_MODE_REMOTE_PD);
  mode->set_transfer_mode(xllm::proto::TRANSFER_MODE_LAYERWISE_PUSH);
  mode->set_selection_order(xllm::proto::SELECTION_ORDER_P_FIRST);
  mode->set_binding_stage(xllm::proto::BINDING_STAGE_BEFORE_PREFILL);
  value.mutable_serving()->add_api_features("chat");
  value.mutable_model()->set_model_revision("model-r1");
  value.mutable_model()->set_tokenizer_revision("tokenizer-r1");
  value.mutable_model()->set_chat_template_digest("template-sha256");
  value.mutable_model()->set_quantization("bf16");
  value.mutable_model()->set_renderer_digest("renderer-sha256");
  auto* topology = value.mutable_topology();
  topology->set_soc("cpu-test");
  topology->set_device_count(1);
  topology->set_tp(1);
  topology->set_dp(1);
  topology->set_pp(1);
  topology->set_ep(1);
  topology->set_cp(1);
  auto* kv = value.mutable_kv();
  kv->set_kv_layout_digest("kv-sha256");
  kv->set_cache_dtype("bf16");
  kv->set_block_size(16);
  kv->add_cache_groups("full-attention");
  kv->set_head_shard_mapping_digest("head-shard-sha256");
  kv->set_connector("mooncake");
  kv->set_connector_version("1");
  kv->add_transfer_modes(xllm::proto::TRANSFER_MODE_LAYERWISE_PUSH);
  auto* scheduler = value.mutable_scheduler();
  scheduler->set_scheduler_class("disagg-pd");
  scheduler->set_max_num_seqs(64);
  scheduler->set_max_num_batched_tokens(8192);
  scheduler->set_scheduler_policy_digest("scheduler-sha256");
  value.add_capabilities(
      xllm::proto::PROVIDER_CAPABILITY_REMOTE_PD_LAYERWISE_PUSH);
  value.add_capabilities(xllm::proto::PROVIDER_CAPABILITY_NATIVE_RESERVATION);
  value.add_capabilities(xllm::proto::PROVIDER_CAPABILITY_ATTEMPT_QUERY);
  value.add_capabilities(xllm::proto::PROVIDER_CAPABILITY_CANCEL_FENCE);
  value.add_capabilities(
      xllm::proto::PROVIDER_CAPABILITY_ENGINE_LOCAL_DEADLINE);
  value.add_capabilities(xllm::proto::PROVIDER_CAPABILITY_SELF_FENCING);
  value.add_capabilities(xllm::proto::PROVIDER_CAPABILITY_DRAIN);
  value.set_profile_digest("profile-a");
  return value;
}

provider::EngineRegistryConfig registry_config() {
  return provider::EngineRegistryConfig{
      .max_members = 8,
      .max_links = 8,
      .state_soft_ttl_ms = 10,
      .state_hard_ttl_ms = 20,
      .heartbeat_hard_ttl_ms = 20,
      .link_hard_ttl_ms = 20,
      .direct_evidence_ttl_ms = 10,
      .observation =
          provider::ObservationControllerConfig{
              .state_blind_enter_ratio = 0.5,
              .state_blind_exit_ratio = 0.25,
              .state_blind_enter_hold_ms = 5,
              .state_blind_exit_hold_ms = 5,
              .state_blind_grace_ms = 10,
              .registry_blind_grace_ms = 5,
          },
  };
}

void publish_ready(provider::EngineRegistry* registry) {
  const xllm::proto::ProviderDescriptor member = descriptor();
  ASSERT_TRUE(registry->upsert_member(member).ok());
  ASSERT_TRUE(registry->set_registry_visibility(true).ok());
  ASSERT_TRUE(registry->set_state_stream_master("master-1").ok());
  xllm::proto::EngineState state;
  state.set_engine_uid("engine-1");
  state.set_incarnation_id("inc-1");
  state.set_state_seq(1);
  state.set_observed_at_unix_ms(1000);
  state.set_lifecycle(xllm::proto::ENGINE_LIFECYCLE_READY);
  state.set_ownership(xllm::proto::ENGINE_OWNERSHIP_OWNED);
  state.set_shallow_health(xllm::proto::HEALTH_STATUS_HEALTHY);
  state.set_deep_health(xllm::proto::HEALTH_STATUS_UNKNOWN);
  state.set_state_quality(xllm::proto::STATE_QUALITY_PARTIAL);
  state.set_provider_id(xllm::proto::PROVIDER_ID_XLLM_NATIVE);
  state.set_profile_digest("profile-a");
  state.set_model_revision("model-r1");
  state.set_heartbeat_age_ms_at_publish(0);
  state.set_state_age_ms_at_publish(0);
  xllm::proto::StateBatch batch;
  batch.set_contract_version(provider::kProviderContractVersion);
  batch.set_master_incarnation("master-1");
  batch.set_snapshot_seq(1);
  batch.set_kind(xllm::proto::STATE_BATCH_KIND_FULL);
  *batch.add_engine_states() = state;
  bool applied = false;
  ASSERT_TRUE(registry->apply_state_batch(batch, 100, &applied).ok());
  ASSERT_TRUE(applied);
}

TEST(HttpPlacementDeploymentActuatorTest, SendsFencedAuthenticatedContract) {
  FakeDeploymentGateway service;
  brpc::Server server;
  ASSERT_EQ(server.AddService(&service,
                              brpc::SERVER_DOESNT_OWN_SERVICE,
                              "/v1/internal/placement/execute => Models,"
                              "/v1/internal/placement/query => Models"),
            0);
  ASSERT_EQ(server.Start("127.0.0.1:0", nullptr), 0);
  const std::string address =
      "127.0.0.1:" + std::to_string(server.listen_address().port);
  HttpPlacementDeploymentActuator actuator(
      HttpPlacementDeploymentActuatorConfig{
          .address = address,
          .timeout_ms = 2000,
          .max_response_bytes = 65536,
          .internal_api_token = "secret-token",
      });
  const PlacementOperationIntent operation = intent();
  const PlacementActuatorResponse response = actuator.execute(operation);
  EXPECT_EQ(response.code, PlacementActuatorCode::SUCCEEDED);
  EXPECT_FALSE(response.ready_proven);
  EXPECT_EQ(service.path, "/v1/internal/placement/execute");
  EXPECT_EQ(service.token, "secret-token");
  EXPECT_EQ(service.request.at("operation_id"), operation.operation_id);
  EXPECT_EQ(service.request.at("leader_epoch"), operation.leader_epoch);
  EXPECT_EQ(service.request.at("ordinal"), operation.ordinal);
  server.Stop(0);
  server.Join();
}

TEST(HttpPlacementDeploymentActuatorTest, RejectsCorruptEchoAndMapsAmbiguity) {
  FakeDeploymentGateway service;
  service.corrupt_echo = true;
  brpc::Server server;
  ASSERT_EQ(server.AddService(&service,
                              brpc::SERVER_DOESNT_OWN_SERVICE,
                              "/v1/internal/placement/execute => Models,"
                              "/v1/internal/placement/query => Models"),
            0);
  ASSERT_EQ(server.Start("127.0.0.1:0", nullptr), 0);
  HttpPlacementDeploymentActuator corrupt(HttpPlacementDeploymentActuatorConfig{
      .address = "127.0.0.1:" + std::to_string(server.listen_address().port),
      .timeout_ms = 2000,
      .max_response_bytes = 65536,
      .internal_api_token = "secret-token",
  });
  EXPECT_EQ(corrupt.execute(intent()).code, PlacementActuatorCode::INVALID);
  server.Stop(0);
  server.Join();

  HttpPlacementDeploymentActuator unavailable(
      HttpPlacementDeploymentActuatorConfig{
          .address = "127.0.0.1:1",
          .timeout_ms = 10,
          .max_response_bytes = 65536,
          .internal_api_token = "secret-token",
      });
  EXPECT_EQ(unavailable.execute(intent()).code, PlacementActuatorCode::UNKNOWN);
  EXPECT_EQ(unavailable.query(intent()).code,
            PlacementActuatorCode::RETRYABLE_ERROR);
}

TEST(RegistryVerifiedDeploymentActuatorTest, CreateRequiresFreshReadyProof) {
  provider::EngineRegistry registry(registry_config());
  FakeDeploymentBackend backend;
  RegistryVerifiedPlacementDeploymentActuator actuator(
      &registry, &backend, [] { return 105; });

  PlacementActuatorResponse response = actuator.execute(intent());
  EXPECT_EQ(response.code, PlacementActuatorCode::IN_PROGRESS);
  EXPECT_FALSE(response.ready_proven);

  publish_ready(&registry);
  response = actuator.query(intent());
  EXPECT_EQ(response.code, PlacementActuatorCode::SUCCEEDED);
  EXPECT_TRUE(response.ready_proven);
  EXPECT_EQ(response.lifecycle_state, PlacementLifecycleState::READY);
}

TEST(RegistryVerifiedDeploymentActuatorTest, TerminateNeedsAbsenceOrProof) {
  provider::EngineRegistry registry(registry_config());
  publish_ready(&registry);
  FakeDeploymentBackend backend;
  backend.response.lifecycle_state = PlacementLifecycleState::ABSENT;
  RegistryVerifiedPlacementDeploymentActuator actuator(
      &registry, &backend, [] { return 105; });
  const PlacementOperationIntent operation =
      intent(PlacementOperationAction::TERMINATE);

  PlacementActuatorResponse response = actuator.query(operation);
  EXPECT_EQ(response.code, PlacementActuatorCode::IN_PROGRESS);
  EXPECT_FALSE(response.termination_proven);

  ASSERT_TRUE(
      registry.remove_member(provider::make_provider_engine_key(descriptor())));
  response = actuator.query(operation);
  EXPECT_EQ(response.code, PlacementActuatorCode::SUCCEEDED);
  EXPECT_TRUE(response.termination_proven);
  EXPECT_EQ(response.lifecycle_state, PlacementLifecycleState::ABSENT);
}

TEST(RegistryVerifiedDeploymentActuatorTest, AcceptsTerminationProof) {
  provider::EngineRegistry registry(registry_config());
  publish_ready(&registry);
  FakeDeploymentBackend backend;
  backend.response.lifecycle_state = PlacementLifecycleState::ABSENT;
  backend.response.termination_proven = true;
  RegistryVerifiedPlacementDeploymentActuator actuator(
      &registry, &backend, [] { return 105; });
  const PlacementActuatorResponse response =
      actuator.query(intent(PlacementOperationAction::TERMINATE));
  EXPECT_EQ(response.code, PlacementActuatorCode::SUCCEEDED);
  EXPECT_TRUE(response.termination_proven);
}

TEST(PlacementDeploymentResponseTest, StrictlyRejectsUnknownFields) {
  const PlacementOperationIntent operation = intent();
  PlacementActuatorResponse response;
  nlohmann::json json = response_json(operation);
  ASSERT_TRUE(
      parse_placement_deployment_response(json.dump(), operation, &response));
  json["unknown"] = true;
  EXPECT_FALSE(
      parse_placement_deployment_response(json.dump(), operation, &response));
  json = response_json(operation);
  json["message"] = "bad\nmessage";
  EXPECT_FALSE(
      parse_placement_deployment_response(json.dump(), operation, &response));
}

}  // namespace
}  // namespace xllm_service::placement
