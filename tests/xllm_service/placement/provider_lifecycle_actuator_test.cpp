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

#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <brpc/server.h>
#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <string>
#include <utility>

#include "xllm_http_service.pb.h"

namespace xllm_service::placement {
namespace {

xllm::proto::ProviderDescriptor make_descriptor() {
  xllm::proto::ProviderDescriptor descriptor;
  descriptor.set_contract_version(provider::kProviderContractVersion);
  xllm::proto::ProviderIdentity* identity = descriptor.mutable_identity();
  identity->set_engine_uid("engine-1");
  identity->set_incarnation_id("inc-1");
  identity->set_provider_id(xllm::proto::PROVIDER_ID_XLLM_NATIVE);
  identity->set_runtime_family("xllm");
  identity->set_runtime_version("runtime-v1");
  identity->set_plugin_version("plugin-v1");
  identity->set_hardware_runtime_version("cpu-simulated-hbm-v1");
  identity->set_protocol_version(provider::kProviderContractVersion);

  descriptor.mutable_endpoint()->set_control_transport("brpc");
  descriptor.mutable_endpoint()->set_data_transport("mooncake");
  descriptor.mutable_endpoint()->set_address("127.0.0.1:8000");
  descriptor.mutable_serving()->set_role(xllm::proto::ENGINE_ROLE_PREFILL);
  xllm::proto::ExecutionModeSpec* mode =
      descriptor.mutable_serving()->add_execution_modes();
  mode->set_mode(xllm::proto::EXECUTION_MODE_REMOTE_PD);
  mode->set_transfer_mode(xllm::proto::TRANSFER_MODE_LAYERWISE_PUSH);
  mode->set_selection_order(xllm::proto::SELECTION_ORDER_P_FIRST);
  mode->set_binding_stage(xllm::proto::BINDING_STAGE_BEFORE_PREFILL);
  descriptor.mutable_serving()->add_api_features("chat");

  descriptor.mutable_model()->set_model_revision("model-r1");
  descriptor.mutable_model()->set_tokenizer_revision("tokenizer-r1");
  descriptor.mutable_model()->set_chat_template_digest("template-sha256");
  descriptor.mutable_model()->set_quantization("bf16");
  descriptor.mutable_model()->set_renderer_digest("renderer-sha256");

  xllm::proto::TopologyDescriptor* topology = descriptor.mutable_topology();
  topology->set_soc("cpu-test");
  topology->set_device_count(1);
  topology->set_tp(1);
  topology->set_dp(1);
  topology->set_pp(1);
  topology->set_ep(1);
  topology->set_cp(1);

  xllm::proto::KVDescriptor* kv = descriptor.mutable_kv();
  kv->set_kv_layout_digest("kv-sha256");
  kv->set_cache_dtype("bf16");
  kv->set_block_size(16);
  kv->add_cache_groups("full-attention");
  kv->set_head_shard_mapping_digest("head-shard-sha256");
  kv->set_connector("mooncake");
  kv->set_connector_version("1");
  kv->add_transfer_modes(xllm::proto::TRANSFER_MODE_LAYERWISE_PUSH);

  xllm::proto::SchedulerDescriptor* scheduler = descriptor.mutable_scheduler();
  scheduler->set_scheduler_class("disagg-pd");
  scheduler->set_max_num_seqs(64);
  scheduler->set_max_num_batched_tokens(8192);
  scheduler->set_scheduler_policy_digest("scheduler-sha256");

  descriptor.add_capabilities(
      xllm::proto::PROVIDER_CAPABILITY_REMOTE_PD_LAYERWISE_PUSH);
  descriptor.add_capabilities(
      xllm::proto::PROVIDER_CAPABILITY_NATIVE_RESERVATION);
  descriptor.add_capabilities(xllm::proto::PROVIDER_CAPABILITY_ATTEMPT_QUERY);
  descriptor.add_capabilities(xllm::proto::PROVIDER_CAPABILITY_CANCEL_FENCE);
  descriptor.add_capabilities(
      xllm::proto::PROVIDER_CAPABILITY_ENGINE_LOCAL_DEADLINE);
  descriptor.add_capabilities(xllm::proto::PROVIDER_CAPABILITY_SELF_FENCING);
  descriptor.add_capabilities(xllm::proto::PROVIDER_CAPABILITY_DRAIN);
  descriptor.set_profile_digest("profile-a");
  return descriptor;
}

PlacementOperationIntent make_intent(
    PlacementOperationAction action = PlacementOperationAction::BEGIN_DRAIN) {
  PlacementOperationIntent intent{
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
      .leader_epoch = 10,
      .desired_generation = 20,
      .ordinal = 0,
  };
  intent.operation_id = make_placement_operation_id(
      PlacementLeaderIdentity{
          .address = "service-1:2888",
          .incarnation = intent.leader_incarnation,
          .epoch = intent.leader_epoch,
      },
      intent.desired_generation,
      intent.pool,
      intent.action,
      intent.ordinal,
      intent.engine_uid,
      intent.engine_incarnation);
  return intent;
}

class FakeLifecycleTransport final : public ProviderLifecycleTransport {
 public:
  ProviderLifecycleTransportStatus call(
      const xllm::proto::ProviderDescriptor& descriptor,
      const xllm::proto::ProviderLifecycleCommand& command,
      bool query,
      xllm::proto::ProviderLifecycleResponse* response) override {
    ++calls;
    last_descriptor = descriptor;
    last_command = command;
    last_query = query;
    if (status != ProviderLifecycleTransportStatus::OK || response == nullptr) {
      return status;
    }
    response->set_code(code);
    response->set_operation_id(corrupt_response ? "wrong-operation"
                                                : command.operation_id());
    response->set_action(command.action());
    response->set_leader_incarnation(command.leader_incarnation());
    response->set_leader_epoch(command.leader_epoch());
    response->set_desired_generation(command.desired_generation());
    response->set_engine_uid(command.engine_uid());
    response->set_engine_incarnation(command.engine_incarnation());
    response->set_lifecycle(xllm::proto::ENGINE_LIFECYCLE_DRAINING);
    response->mutable_drain()->set_admission_closed(true);
    response->mutable_drain()->set_active_transfers(active_transfers);
    return status;
  }

  ProviderLifecycleTransportStatus status =
      ProviderLifecycleTransportStatus::OK;
  xllm::proto::ProviderLifecycleCode code =
      xllm::proto::PROVIDER_LIFECYCLE_CODE_SUCCEEDED;
  uint64_t active_transfers = 0;
  bool corrupt_response = false;
  uint32_t calls = 0;
  bool last_query = false;
  xllm::proto::ProviderDescriptor last_descriptor;
  xllm::proto::ProviderLifecycleCommand last_command;
};

class FakeDeploymentActuator final : public PlacementDeploymentActuator {
 public:
  PlacementActuatorResponse execute(
      const PlacementOperationIntent& intent) override {
    ++execute_calls;
    last_intent = intent;
    return response;
  }

  PlacementActuatorResponse query(
      const PlacementOperationIntent& intent) override {
    ++query_calls;
    last_intent = intent;
    return response;
  }

  PlacementActuatorResponse response{
      .code = PlacementActuatorCode::ACCEPTED,
  };
  uint32_t execute_calls = 0;
  uint32_t query_calls = 0;
  PlacementOperationIntent last_intent;
};

class FakeLifecycleAgentService final : public proto::XllmHttpService {
 public:
  void Models(google::protobuf::RpcController* controller,
              const proto::HttpRequest*,
              proto::HttpResponse*,
              google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    auto* brpc_controller = static_cast<brpc::Controller*>(controller);
    last_path = brpc_controller->http_request().uri().path();
    const std::string* token =
        brpc_controller->http_request().GetHeader("X-Internal-Token");
    last_token = token == nullptr ? "" : *token;
    last_body = nlohmann::json::parse(
        brpc_controller->request_attachment().to_string());
    brpc_controller->http_response().set_status_code(200);
    brpc_controller->response_attachment().append(nlohmann::json{
        {"code", "PROVIDER_LIFECYCLE_CODE_SUCCEEDED"},
        {"operation_id", last_body.at("operation_id")},
        {"action", last_body.at("action")},
        {"leader_incarnation", last_body.at("leader_incarnation")},
        {"leader_epoch", last_body.at("leader_epoch")},
        {"desired_generation", last_body.at("desired_generation")},
        {"engine_uid", last_body.at("engine_uid")},
        {"engine_incarnation", last_body.at("engine_incarnation")},
        {"lifecycle", "ENGINE_LIFECYCLE_DRAINING"},
        {"drain",
         {
             {"admission_closed", true},
             {"prefill_queue", 0},
             {"active_transfers", 0},
             {"active_reservations", 0},
             {"decode_sequences", 0},
             {"pending_output", 0},
             {"pending_cleanup", 0},
         }},
        {"replayed", false},
        {"message", "drain complete"},
    }
                                                      .dump());
  }

  std::string last_path;
  std::string last_token;
  nlohmann::json last_body;
};

TEST(ProviderPlacementActuatorTest, SendsFencedLifecycleCommandAndMapsProof) {
  provider::EngineRegistry registry(provider::EngineRegistryConfig{});
  ASSERT_TRUE(registry.upsert_member(make_descriptor()).ok());
  FakeLifecycleTransport transport;
  FakeDeploymentActuator deployment;
  ProviderPlacementActuator actuator(&registry, &transport, &deployment);
  const PlacementOperationIntent intent = make_intent();

  const PlacementActuatorResponse response = actuator.execute(intent);

  EXPECT_EQ(response.code, PlacementActuatorCode::SUCCEEDED);
  EXPECT_EQ(response.lifecycle_state, PlacementLifecycleState::DRAINING);
  EXPECT_TRUE(placement_drain_proof_complete(response.drain));
  EXPECT_EQ(transport.calls, 1u);
  EXPECT_FALSE(transport.last_query);
  EXPECT_EQ(transport.last_command.schema_version(), 1u);
  EXPECT_EQ(transport.last_command.operation_id(), intent.operation_id);
  EXPECT_EQ(transport.last_command.leader_incarnation(),
            intent.leader_incarnation);
  EXPECT_EQ(transport.last_command.leader_epoch(), intent.leader_epoch);
  EXPECT_EQ(transport.last_command.desired_generation(),
            intent.desired_generation);
  EXPECT_EQ(transport.last_command.engine_incarnation(),
            intent.engine_incarnation);
}

TEST(ProviderPlacementActuatorTest, FencesStaleTargetBeforeTransport) {
  provider::EngineRegistry registry(provider::EngineRegistryConfig{});
  ASSERT_TRUE(registry.upsert_member(make_descriptor()).ok());
  FakeLifecycleTransport transport;
  FakeDeploymentActuator deployment;
  ProviderPlacementActuator actuator(&registry, &transport, &deployment);
  PlacementOperationIntent intent = make_intent();
  intent.engine_incarnation = "stale-incarnation";
  intent.operation_id = make_placement_operation_id(
      PlacementLeaderIdentity{
          .address = "service-1:2888",
          .incarnation = intent.leader_incarnation,
          .epoch = intent.leader_epoch,
      },
      intent.desired_generation,
      intent.pool,
      intent.action,
      intent.ordinal,
      intent.engine_uid,
      intent.engine_incarnation);

  const PlacementActuatorResponse response = actuator.execute(intent);

  EXPECT_EQ(response.code, PlacementActuatorCode::FENCED);
  EXPECT_EQ(transport.calls, 0u);
}

TEST(ProviderPlacementActuatorTest, PreservesUnknownAndRejectsMalformedEcho) {
  provider::EngineRegistry registry(provider::EngineRegistryConfig{});
  ASSERT_TRUE(registry.upsert_member(make_descriptor()).ok());
  FakeLifecycleTransport transport;
  FakeDeploymentActuator deployment;
  ProviderPlacementActuator actuator(&registry, &transport, &deployment);
  const PlacementOperationIntent intent = make_intent();

  transport.status = ProviderLifecycleTransportStatus::OUTCOME_UNKNOWN;
  EXPECT_EQ(actuator.execute(intent).code, PlacementActuatorCode::UNKNOWN);

  transport.status = ProviderLifecycleTransportStatus::OK;
  transport.corrupt_response = true;
  EXPECT_EQ(actuator.query(intent).code, PlacementActuatorCode::INVALID);
  EXPECT_TRUE(transport.last_query);
}

TEST(ProviderPlacementActuatorTest, DelegatesDeploymentOperations) {
  provider::EngineRegistry registry(provider::EngineRegistryConfig{});
  FakeLifecycleTransport transport;
  FakeDeploymentActuator deployment;
  ProviderPlacementActuator actuator(&registry, &transport, &deployment);
  const PlacementOperationIntent create =
      make_intent(PlacementOperationAction::CREATE);

  EXPECT_EQ(actuator.execute(create).code, PlacementActuatorCode::ACCEPTED);
  EXPECT_EQ(actuator.query(create).code, PlacementActuatorCode::ACCEPTED);
  EXPECT_EQ(deployment.execute_calls, 1u);
  EXPECT_EQ(deployment.query_calls, 1u);
  EXPECT_EQ(transport.calls, 0u);
}

TEST(HttpProviderLifecycleTransportTest, CallsAgentExecuteAndQueryEndpoints) {
  FakeLifecycleAgentService service;
  brpc::Server server;
  ASSERT_EQ(server.AddService(&service,
                              brpc::SERVER_DOESNT_OWN_SERVICE,
                              "/v1/internal/lifecycle/execute => Models,"
                              "/v1/internal/lifecycle/query => Models"),
            0);
  ASSERT_EQ(server.Start("127.0.0.1:0", nullptr), 0);

  HttpProviderLifecycleTransport transport(HttpProviderLifecycleTransportConfig{
      .timeout_ms = 500,
      .max_channels = 2,
      .max_response_bytes = 4096,
      .internal_api_token = "test-token",
  });
  xllm::proto::ProviderDescriptor descriptor;
  descriptor.mutable_identity()->set_provider_id(
      xllm::proto::PROVIDER_ID_VLLM_ASCEND);
  descriptor.mutable_endpoint()->set_control_transport("http");
  descriptor.mutable_endpoint()->set_address(
      "127.0.0.1:" + std::to_string(server.listen_address().port));
  xllm::proto::ProviderLifecycleCommand command;
  command.set_schema_version(1);
  command.set_operation_id("operation-1");
  command.set_leader_incarnation("leader-1");
  command.set_leader_epoch(10);
  command.set_desired_generation(20);
  command.set_engine_uid("engine-1");
  command.set_engine_incarnation("inc-1");
  command.set_action(xllm::proto::PROVIDER_LIFECYCLE_ACTION_BEGIN_DRAIN);

  xllm::proto::ProviderLifecycleResponse response;
  EXPECT_EQ(transport.call(descriptor, command, false, &response),
            ProviderLifecycleTransportStatus::OK);
  EXPECT_EQ(service.last_path, "/v1/internal/lifecycle/execute");
  EXPECT_EQ(service.last_token, "test-token");
  EXPECT_EQ(service.last_body.at("leader_epoch").get<uint64_t>(), 10u);
  EXPECT_EQ(response.code(), xllm::proto::PROVIDER_LIFECYCLE_CODE_SUCCEEDED);
  EXPECT_TRUE(response.drain().admission_closed());

  EXPECT_EQ(transport.call(descriptor, command, true, &response),
            ProviderLifecycleTransportStatus::OK);
  EXPECT_EQ(service.last_path, "/v1/internal/lifecycle/query");
  EXPECT_EQ(server.Stop(0), 0);
  EXPECT_EQ(server.Join(), 0);
}

TEST(HttpProviderLifecycleTransportTest, RejectsMalformedResponseAndToken) {
  xllm::proto::ProviderLifecycleResponse response;
  EXPECT_FALSE(parse_vllm_lifecycle_response(500, "{}", &response));
  EXPECT_FALSE(parse_vllm_lifecycle_response(200, "{}", &response));
  EXPECT_FALSE(parse_vllm_lifecycle_response(200, "not-json", &response));

  HttpProviderLifecycleTransport transport(HttpProviderLifecycleTransportConfig{
      .timeout_ms = 100,
      .max_channels = 1,
      .max_response_bytes = 1024,
      .internal_api_token = "bad\ntoken",
  });
  xllm::proto::ProviderDescriptor descriptor;
  descriptor.mutable_identity()->set_provider_id(
      xllm::proto::PROVIDER_ID_VLLM_ASCEND);
  descriptor.mutable_endpoint()->set_control_transport("http");
  descriptor.mutable_endpoint()->set_address("127.0.0.1:1");
  xllm::proto::ProviderLifecycleCommand command;
  EXPECT_EQ(transport.call(descriptor, command, false, &response),
            ProviderLifecycleTransportStatus::INVALID);
}

}  // namespace
}  // namespace xllm_service::placement
