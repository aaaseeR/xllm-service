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

#include "provider/link_reconciler.h"

#include <gtest/gtest.h>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

namespace xllm_service::provider {
namespace {

xllm::proto::ProviderDescriptor make_descriptor(xllm::proto::EngineRole role,
                                                std::string engine_uid,
                                                std::string incarnation_id,
                                                std::string profile_digest) {
  xllm::proto::ProviderDescriptor descriptor;
  descriptor.set_contract_version(kProviderContractVersion);
  xllm::proto::ProviderIdentity* identity = descriptor.mutable_identity();
  identity->set_engine_uid(std::move(engine_uid));
  identity->set_incarnation_id(std::move(incarnation_id));
  identity->set_provider_id(xllm::proto::PROVIDER_ID_XLLM_NATIVE);
  identity->set_runtime_family("xllm");
  identity->set_runtime_version("runtime-v1");
  identity->set_plugin_version("builtin");
  identity->set_hardware_runtime_version("hardware-v1");
  identity->set_protocol_version(kProviderContractVersion);
  descriptor.mutable_endpoint()->set_control_transport("brpc");
  descriptor.mutable_endpoint()->set_data_transport("mooncake");
  descriptor.mutable_endpoint()->set_address("127.0.0.1:8000");
  descriptor.mutable_serving()->set_role(role);
  xllm::proto::ExecutionModeSpec* mode =
      descriptor.mutable_serving()->add_execution_modes();
  mode->set_mode(xllm::proto::EXECUTION_MODE_REMOTE_PD);
  mode->set_transfer_mode(xllm::proto::TRANSFER_MODE_LAYERWISE_PUSH);
  mode->set_selection_order(xllm::proto::SELECTION_ORDER_P_FIRST);
  mode->set_binding_stage(xllm::proto::BINDING_STAGE_BEFORE_PREFILL);
  descriptor.mutable_serving()->add_api_features("chat");
  descriptor.mutable_model()->set_model_revision("model-r1");
  descriptor.mutable_model()->set_tokenizer_revision("tokenizer-r1");
  descriptor.mutable_model()->set_chat_template_digest("template-v1");
  descriptor.mutable_model()->set_quantization("bf16");
  descriptor.mutable_model()->set_renderer_digest("renderer-v1");
  xllm::proto::TopologyDescriptor* topology = descriptor.mutable_topology();
  topology->set_soc("cpu-test");
  topology->set_device_count(1);
  topology->set_tp(1);
  topology->set_dp(1);
  topology->set_pp(1);
  topology->set_ep(1);
  topology->set_cp(1);
  xllm::proto::KVDescriptor* kv = descriptor.mutable_kv();
  kv->set_kv_layout_digest("kv-v1");
  kv->set_cache_dtype("bf16");
  kv->set_block_size(16);
  kv->add_cache_groups("full-attention");
  kv->set_head_shard_mapping_digest("head-map-v1");
  kv->set_connector("mooncake");
  kv->set_connector_version("1");
  kv->add_transfer_modes(xllm::proto::TRANSFER_MODE_LAYERWISE_PUSH);
  xllm::proto::SchedulerDescriptor* scheduler = descriptor.mutable_scheduler();
  scheduler->set_scheduler_class("disagg-pd");
  scheduler->set_max_num_seqs(64);
  scheduler->set_max_num_batched_tokens(8192);
  scheduler->set_scheduler_policy_digest("scheduler-v1");
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
  descriptor.set_profile_digest(std::move(profile_digest));
  return descriptor;
}

DesiredProviderLink make_link(std::string p_inc = "p-inc",
                              std::string d_inc = "d-inc") {
  return DesiredProviderLink{
      .prefill = make_descriptor(
          xllm::proto::ENGINE_ROLE_PREFILL, "p", std::move(p_inc), "p-profile"),
      .decode = make_descriptor(
          xllm::proto::ENGINE_ROLE_DECODE, "d", std::move(d_inc), "d-profile"),
  };
}

LinkReconcilerConfig test_config() {
  return LinkReconcilerConfig{
      .max_links = 8,
      .retry_initial_ms = 10,
      .retry_max_ms = 40,
      .ready_recheck_ms = 100,
  };
}

TEST(LinkReconcilerTest, PublishesPendingReadyAndPeriodicDegraded) {
  LinkReconciler reconciler(test_config());
  std::vector<xllm::proto::LinkState> changes;
  ASSERT_TRUE(reconciler.replace_desired({make_link()}, 5, &changes).ok());
  ASSERT_EQ(changes.size(), 1u);
  EXPECT_EQ(changes[0].lifecycle(), xllm::proto::LINK_LIFECYCLE_PENDING);
  EXPECT_EQ(changes[0].state_seq(), 1u);

  auto attempts = reconciler.begin_due_attempts(5, 4);
  ASSERT_EQ(attempts.size(), 1u);
  xllm::proto::LinkState ready;
  ASSERT_TRUE(
      reconciler.complete_attempt(attempts[0], true, "ok", 5, &ready).ok());
  EXPECT_EQ(ready.lifecycle(), xllm::proto::LINK_LIFECYCLE_READY);
  EXPECT_EQ(ready.state_seq(), 2u);
  EXPECT_TRUE(ready.has_age_ms_at_publish());
  EXPECT_TRUE(reconciler.begin_due_attempts(104, 4).empty());

  attempts = reconciler.begin_due_attempts(105, 4);
  ASSERT_EQ(attempts.size(), 1u);
  xllm::proto::LinkState degraded;
  ASSERT_TRUE(reconciler
                  .complete_attempt(
                      attempts[0], false, "remote rejected", 105, &degraded)
                  .ok());
  EXPECT_EQ(degraded.lifecycle(), xllm::proto::LINK_LIFECYCLE_DEGRADED);
  EXPECT_EQ(degraded.state_seq(), 3u);
  EXPECT_TRUE(reconciler.begin_due_attempts(114, 4).empty());
  EXPECT_EQ(reconciler.begin_due_attempts(115, 4).size(), 1u);
}

TEST(LinkReconcilerTest, FailureBackoffIsBounded) {
  LinkReconciler reconciler(test_config());
  std::vector<xllm::proto::LinkState> changes;
  ASSERT_TRUE(reconciler.replace_desired({make_link()}, 0, &changes).ok());
  uint64_t now = 0;
  for (const uint64_t expected_delay : {10u, 20u, 40u, 40u}) {
    auto attempts = reconciler.begin_due_attempts(now, 1);
    ASSERT_EQ(attempts.size(), 1u);
    xllm::proto::LinkState state;
    ASSERT_TRUE(
        reconciler.complete_attempt(attempts[0], false, "failed", now, &state)
            .ok());
    EXPECT_TRUE(
        reconciler.begin_due_attempts(now + expected_delay - 1, 1).empty());
    now += expected_delay;
  }
}

TEST(LinkReconcilerTest, IncarnationReplacementRejectsStaleCompletion) {
  LinkReconciler reconciler(test_config());
  std::vector<xllm::proto::LinkState> changes;
  ASSERT_TRUE(reconciler.replace_desired({make_link()}, 0, &changes).ok());
  const auto stale_attempt = reconciler.begin_due_attempts(0, 1);
  ASSERT_EQ(stale_attempt.size(), 1u);

  ASSERT_TRUE(
      reconciler.replace_desired({make_link("p-inc-2", "d-inc")}, 1, &changes)
          .ok());
  ASSERT_EQ(changes.size(), 1u);
  EXPECT_EQ(changes[0].prefill().incarnation_id(), "p-inc-2");
  xllm::proto::LinkState ignored;
  EXPECT_FALSE(
      reconciler.complete_attempt(stale_attempt[0], true, "late", 2, &ignored)
          .ok());
  EXPECT_EQ(reconciler.size(), 1u);
}

TEST(LinkReconcilerTest, CapacityDuplicatesAndInvalidContractsFailClosed) {
  LinkReconcilerConfig config = test_config();
  config.max_links = 1;
  LinkReconciler reconciler(config);
  std::vector<xllm::proto::LinkState> changes;
  const DesiredProviderLink link = make_link();
  EXPECT_FALSE(reconciler.replace_desired({link, link}, 0, &changes).ok());

  DesiredProviderLink incompatible = make_link();
  incompatible.decode.mutable_kv()->set_connector("other");
  EXPECT_FALSE(reconciler.replace_desired({incompatible}, 0, &changes).ok());

  LinkReconcilerConfig invalid_config = test_config();
  invalid_config.max_links = 0;
  LinkReconciler invalid(invalid_config);
  EXPECT_FALSE(invalid.replace_desired({}, 0, &changes).ok());
}

TEST(LinkReconcilerTest, ConcurrentDueSelectionKeepsOneInflightPerPair) {
  LinkReconciler reconciler(test_config());
  std::vector<xllm::proto::LinkState> changes;
  ASSERT_TRUE(reconciler.replace_desired({make_link()}, 0, &changes).ok());
  std::atomic<size_t> selected = 0;
  std::vector<std::thread> workers;
  for (size_t index = 0; index < 32; ++index) {
    workers.emplace_back([&]() {
      selected.fetch_add(reconciler.begin_due_attempts(0, 1).size());
    });
  }
  for (std::thread& worker : workers) {
    worker.join();
  }
  EXPECT_EQ(selected.load(), 1u);
}

}  // namespace
}  // namespace xllm_service::provider
