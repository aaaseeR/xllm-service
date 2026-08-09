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

#include "provider/engine_registry.h"

#include <gtest/gtest.h>

#include <limits>
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
  identity->set_plugin_version("plugin-v1");
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
  descriptor.set_profile_digest(std::move(profile_digest));
  return descriptor;
}

xllm::proto::EngineState make_state(
    const xllm::proto::ProviderDescriptor& descriptor,
    uint64_t state_seq,
    xllm::proto::EngineLifecycle lifecycle =
        xllm::proto::ENGINE_LIFECYCLE_READY,
    uint64_t state_age_ms = 0,
    uint64_t heartbeat_age_ms = 0) {
  xllm::proto::EngineState state;
  state.set_engine_uid(descriptor.identity().engine_uid());
  state.set_incarnation_id(descriptor.identity().incarnation_id());
  state.set_state_seq(state_seq);
  state.set_observed_at_unix_ms(1000);
  state.set_lifecycle(lifecycle);
  state.set_ownership(xllm::proto::ENGINE_OWNERSHIP_OWNED);
  state.set_shallow_health(xllm::proto::HEALTH_STATUS_HEALTHY);
  state.set_deep_health(xllm::proto::HEALTH_STATUS_UNKNOWN);
  state.set_state_quality(xllm::proto::STATE_QUALITY_PARTIAL);
  state.set_provider_id(descriptor.identity().provider_id());
  state.set_profile_digest(descriptor.profile_digest());
  state.set_model_revision(descriptor.model().model_revision());
  state.set_state_age_ms_at_publish(state_age_ms);
  state.set_heartbeat_age_ms_at_publish(heartbeat_age_ms);
  return state;
}

xllm::proto::LinkState make_link(
    const xllm::proto::ProviderDescriptor& prefill,
    const xllm::proto::ProviderDescriptor& decode,
    uint64_t state_seq,
    xllm::proto::LinkLifecycle lifecycle = xllm::proto::LINK_LIFECYCLE_READY,
    uint64_t age_ms = 0) {
  xllm::proto::LinkState state;
  *state.mutable_prefill() = make_provider_engine_key(prefill);
  *state.mutable_decode() = make_provider_engine_key(decode);
  state.set_lifecycle(lifecycle);
  state.set_connector(prefill.kv().connector());
  state.set_connector_version(prefill.kv().connector_version());
  state.set_transfer_mode(xllm::proto::TRANSFER_MODE_LAYERWISE_PUSH);
  std::string proof;
  EXPECT_TRUE(validate_remote_pd_compatibility(prefill, decode, &proof).ok());
  state.set_compatibility_proof(std::move(proof));
  state.set_last_handshake_result("ok");
  state.set_state_seq(state_seq);
  state.set_age_ms_at_publish(age_ms);
  return state;
}

xllm::proto::StateBatch make_batch(std::string master,
                                   uint64_t snapshot_seq,
                                   xllm::proto::StateBatchKind kind) {
  xllm::proto::StateBatch batch;
  batch.set_contract_version(kProviderContractVersion);
  batch.set_master_incarnation(std::move(master));
  batch.set_snapshot_seq(snapshot_seq);
  batch.set_kind(kind);
  return batch;
}

EngineRegistryConfig test_config() {
  return EngineRegistryConfig{
      .max_members = 32,
      .max_links = 64,
      .state_soft_ttl_ms = 10,
      .state_hard_ttl_ms = 20,
      .heartbeat_hard_ttl_ms = 20,
      .link_hard_ttl_ms = 10,
      .direct_evidence_ttl_ms = 8,
      .observation =
          ObservationControllerConfig{
              .state_blind_enter_ratio = 0.5,
              .state_blind_exit_ratio = 0.25,
              .state_blind_enter_hold_ms = 5,
              .state_blind_exit_hold_ms = 5,
              .state_blind_grace_ms = 10,
              .registry_blind_grace_ms = 4,
          },
  };
}

void set_registry_view(EngineRegistry* registry, std::string master) {
  ASSERT_TRUE(registry->set_registry_visibility(true).ok());
  ASSERT_TRUE(registry->set_state_stream_master(std::move(master)).ok());
}

TEST(EngineRegistryTest, RequiresCurrentMasterFullBeforeScheduling) {
  EngineRegistry registry(test_config());
  const xllm::proto::ProviderDescriptor prefill = make_descriptor(
      xllm::proto::ENGINE_ROLE_PREFILL, "p", "p-inc", "p-profile");
  const xllm::proto::ProviderDescriptor decode = make_descriptor(
      xllm::proto::ENGINE_ROLE_DECODE, "d", "d-inc", "d-profile");
  ASSERT_TRUE(registry.upsert_member(prefill).ok());
  ASSERT_TRUE(registry.upsert_member(decode).ok());
  set_registry_view(&registry, "master-1");

  bool applied = false;
  xllm::proto::StateBatch delta =
      make_batch("master-1", 1, xllm::proto::STATE_BATCH_KIND_DELTA);
  *delta.add_engine_states() = make_state(prefill, 1);
  EXPECT_EQ(registry.apply_state_batch(delta, 100, &applied).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE);
  EXPECT_FALSE(applied);

  xllm::proto::StateBatch full =
      make_batch("master-1", 1, xllm::proto::STATE_BATCH_KIND_FULL);
  *full.add_engine_states() = make_state(prefill, 1);
  *full.add_engine_states() = make_state(decode, 1);
  *full.add_link_states() = make_link(prefill, decode, 1);
  ASSERT_TRUE(registry.apply_state_batch(full, 100, &applied).ok());
  EXPECT_TRUE(applied);
  EXPECT_TRUE(registry.has_current_full_snapshot());
  EXPECT_TRUE(registry.is_schedulable(make_provider_engine_key(prefill), 100));
  EXPECT_TRUE(registry.is_schedulable(make_provider_engine_key(decode), 100));
  EXPECT_TRUE(registry.is_link_ready(make_provider_engine_key(prefill),
                                     make_provider_engine_key(decode),
                                     100));

  applied = true;
  ASSERT_TRUE(registry.apply_state_batch(full, 101, &applied).ok());
  EXPECT_FALSE(applied);

  set_registry_view(&registry, "master-2");
  EXPECT_FALSE(registry.has_current_full_snapshot());
  EXPECT_TRUE(registry.is_schedulable(make_provider_engine_key(prefill), 101));
  EXPECT_EQ(registry.apply_state_batch(full, 101, &applied).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);

  delta.set_master_incarnation("master-2");
  EXPECT_EQ(registry.apply_state_batch(delta, 101, &applied).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE);
}

TEST(EngineRegistryTest, UsesReceiverMonotonicAgeAndNeverRegressesState) {
  EngineRegistry registry(test_config());
  const xllm::proto::ProviderDescriptor descriptor = make_descriptor(
      xllm::proto::ENGINE_ROLE_PREFILL, "p", "p-inc", "p-profile");
  const xllm::proto::ProviderEngineKey key =
      make_provider_engine_key(descriptor);
  ASSERT_TRUE(registry.upsert_member(descriptor).ok());
  set_registry_view(&registry, "master");
  xllm::proto::StateBatch full =
      make_batch("master", 1, xllm::proto::STATE_BATCH_KIND_FULL);
  *full.add_engine_states() =
      make_state(descriptor, 10, xllm::proto::ENGINE_LIFECYCLE_READY, 2, 3);
  bool applied = false;
  ASSERT_TRUE(registry.apply_state_batch(full, 100, &applied).ok());

  EXPECT_EQ(registry.state_freshness(key, 108), EngineStateFreshness::FRESH);
  EXPECT_EQ(registry.state_freshness(key, 109),
            EngineStateFreshness::SOFT_STALE);
  EXPECT_EQ(registry.state_freshness(key, 119),
            EngineStateFreshness::HARD_STALE);
  EXPECT_EQ(registry.state_freshness(key, 99),
            EngineStateFreshness::HARD_STALE);

  xllm::proto::StateBatch delta =
      make_batch("master", 2, xllm::proto::STATE_BATCH_KIND_DELTA);
  *delta.add_engine_states() =
      make_state(descriptor, 9, xllm::proto::ENGINE_LIFECYCLE_DRAINING);
  ASSERT_TRUE(registry.apply_state_batch(delta, 101, &applied).ok());
  ASSERT_TRUE(registry.find_state(key).has_value());
  EXPECT_EQ(registry.find_state(key)->lifecycle(),
            xllm::proto::ENGINE_LIFECYCLE_READY);

  delta.set_snapshot_seq(3);
  delta.mutable_engine_states(0)->set_state_seq(11);
  ASSERT_TRUE(registry.apply_state_batch(delta, 102, &applied).ok());
  EXPECT_EQ(registry.find_state(key)->lifecycle(),
            xllm::proto::ENGINE_LIFECYCLE_DRAINING);
  EXPECT_FALSE(registry.is_schedulable(key, 102));
}

TEST(EngineRegistryTest, IncarnationReplacementCannotBeResurrectedByState) {
  EngineRegistry registry(test_config());
  const xllm::proto::ProviderDescriptor old_descriptor = make_descriptor(
      xllm::proto::ENGINE_ROLE_PREFILL, "p", "old-inc", "profile");
  xllm::proto::ProviderDescriptor replacement = old_descriptor;
  replacement.mutable_identity()->set_incarnation_id("new-inc");
  ASSERT_TRUE(registry.upsert_member(old_descriptor).ok());
  set_registry_view(&registry, "master");
  xllm::proto::StateBatch full =
      make_batch("master", 1, xllm::proto::STATE_BATCH_KIND_FULL);
  *full.add_engine_states() = make_state(old_descriptor, 1);
  bool applied = false;
  ASSERT_TRUE(registry.apply_state_batch(full, 100, &applied).ok());

  ASSERT_TRUE(registry.upsert_member(replacement).ok());
  EXPECT_EQ(registry.member_count(), 1u);
  EXPECT_FALSE(registry.find_member(make_provider_engine_key(old_descriptor))
                   .has_value());
  EXPECT_FALSE(registry.has_current_full_snapshot());

  full.set_snapshot_seq(2);
  EXPECT_EQ(registry.apply_state_batch(full, 101, &applied).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);
  EXPECT_FALSE(
      registry.remove_member(make_provider_engine_key(old_descriptor)));
  EXPECT_TRUE(registry.remove_member(make_provider_engine_key(replacement)));
  EXPECT_EQ(registry.member_count(), 0u);
  EXPECT_EQ(registry.state_count(), 0u);
}

TEST(EngineRegistryTest, RejectsIncompleteFullAndDuplicateState) {
  EngineRegistry registry(test_config());
  const xllm::proto::ProviderDescriptor prefill = make_descriptor(
      xllm::proto::ENGINE_ROLE_PREFILL, "p", "p-inc", "p-profile");
  const xllm::proto::ProviderDescriptor decode = make_descriptor(
      xllm::proto::ENGINE_ROLE_DECODE, "d", "d-inc", "d-profile");
  ASSERT_TRUE(registry.upsert_member(prefill).ok());
  ASSERT_TRUE(registry.upsert_member(decode).ok());
  set_registry_view(&registry, "master");

  xllm::proto::StateBatch full =
      make_batch("master", 1, xllm::proto::STATE_BATCH_KIND_FULL);
  *full.add_engine_states() = make_state(prefill, 1);
  bool applied = false;
  EXPECT_EQ(registry.apply_state_batch(full, 100, &applied).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE);

  *full.add_engine_states() = make_state(prefill, 2);
  EXPECT_EQ(registry.apply_state_batch(full, 100, &applied).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DUPLICATE_ENTRY);

  full.mutable_engine_states()->RemoveLast();
  *full.add_engine_states() = make_state(decode, 1);
  *full.add_removed_engines() = make_provider_engine_key(prefill);
  EXPECT_EQ(registry.apply_state_batch(full, 100, &applied).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE);
}

TEST(EngineRegistryTest, LinkProofLifecycleAndTtlFailClosed) {
  EngineRegistry registry(test_config());
  const xllm::proto::ProviderDescriptor prefill = make_descriptor(
      xllm::proto::ENGINE_ROLE_PREFILL, "p", "p-inc", "p-profile");
  const xllm::proto::ProviderDescriptor decode = make_descriptor(
      xllm::proto::ENGINE_ROLE_DECODE, "d", "d-inc", "d-profile");
  ASSERT_TRUE(registry.upsert_member(prefill).ok());
  ASSERT_TRUE(registry.upsert_member(decode).ok());
  set_registry_view(&registry, "master");
  xllm::proto::StateBatch full =
      make_batch("master", 1, xllm::proto::STATE_BATCH_KIND_FULL);
  *full.add_engine_states() = make_state(prefill, 1);
  *full.add_engine_states() = make_state(decode, 1);
  *full.add_link_states() =
      make_link(prefill, decode, 1, xllm::proto::LINK_LIFECYCLE_DEGRADED);
  bool applied = false;
  ASSERT_TRUE(registry.apply_state_batch(full, 100, &applied).ok());
  EXPECT_FALSE(registry.is_link_ready(make_provider_engine_key(prefill),
                                      make_provider_engine_key(decode),
                                      100));

  xllm::proto::StateBatch delta =
      make_batch("master", 2, xllm::proto::STATE_BATCH_KIND_DELTA);
  *delta.add_link_states() = make_link(prefill, decode, 2);
  delta.mutable_link_states(0)->set_compatibility_proof("wrong");
  EXPECT_EQ(registry.apply_state_batch(delta, 101, &applied).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);

  *delta.mutable_link_states(0) = make_link(prefill, decode, 2);
  ASSERT_TRUE(registry.apply_state_batch(delta, 101, &applied).ok());
  EXPECT_TRUE(registry.is_link_ready(make_provider_engine_key(prefill),
                                     make_provider_engine_key(decode),
                                     111));
  EXPECT_FALSE(registry.is_link_ready(make_provider_engine_key(prefill),
                                      make_provider_engine_key(decode),
                                      112));
}

TEST(EngineRegistryTest, MasterIngestionBuildsRebasedAuthoritativeFull) {
  EngineRegistry registry(test_config());
  const xllm::proto::ProviderDescriptor prefill = make_descriptor(
      xllm::proto::ENGINE_ROLE_PREFILL, "p", "p-inc", "p-profile");
  const xllm::proto::ProviderDescriptor decode = make_descriptor(
      xllm::proto::ENGINE_ROLE_DECODE, "d", "d-inc", "d-profile");
  ASSERT_TRUE(registry.upsert_member(prefill).ok());
  ASSERT_TRUE(registry.upsert_member(decode).ok());
  set_registry_view(&registry, "master");

  bool applied = false;
  xllm::proto::EngineState prefill_state =
      make_state(prefill, 2, xllm::proto::ENGINE_LIFECYCLE_READY, 2, 3);
  ASSERT_TRUE(registry.record_engine_state(prefill_state, 100, &applied).ok());
  EXPECT_TRUE(applied);
  prefill_state.set_state_seq(1);
  ASSERT_TRUE(registry.record_engine_state(prefill_state, 101, &applied).ok());
  EXPECT_FALSE(applied);
  ASSERT_TRUE(
      registry.record_engine_state(make_state(decode, 1), 100, &applied).ok());
  EXPECT_TRUE(applied);
  ASSERT_TRUE(
      registry
          .record_link_state(
              make_link(
                  prefill, decode, 1, xllm::proto::LINK_LIFECYCLE_READY, 4),
              100,
              &applied)
          .ok());
  EXPECT_TRUE(applied);

  xllm::proto::StateBatch full;
  ASSERT_TRUE(registry.build_full_state_batch("master", 1, 110, &full).ok());
  ASSERT_EQ(full.engine_states_size(), 2);
  ASSERT_EQ(full.link_states_size(), 1);
  const xllm::proto::EngineState* rebased_prefill = nullptr;
  for (const xllm::proto::EngineState& state : full.engine_states()) {
    if (state.engine_uid() == "p") {
      rebased_prefill = &state;
    }
  }
  ASSERT_NE(rebased_prefill, nullptr);
  EXPECT_EQ(rebased_prefill->state_seq(), 2u);
  EXPECT_EQ(rebased_prefill->state_age_ms_at_publish(), 12u);
  EXPECT_EQ(rebased_prefill->heartbeat_age_ms_at_publish(), 13u);
  EXPECT_EQ(full.link_states(0).age_ms_at_publish(), 14u);

  ASSERT_TRUE(registry.apply_state_batch(full, 110, &applied).ok());
  EXPECT_TRUE(applied);
  EXPECT_TRUE(registry.has_current_full_snapshot());
}

TEST(EngineRegistryTest, MasterFullWaitsForEveryMemberObservation) {
  EngineRegistry registry(test_config());
  const xllm::proto::ProviderDescriptor prefill = make_descriptor(
      xllm::proto::ENGINE_ROLE_PREFILL, "p", "p-inc", "p-profile");
  const xllm::proto::ProviderDescriptor decode = make_descriptor(
      xllm::proto::ENGINE_ROLE_DECODE, "d", "d-inc", "d-profile");
  ASSERT_TRUE(registry.upsert_member(prefill).ok());
  ASSERT_TRUE(registry.upsert_member(decode).ok());
  set_registry_view(&registry, "master");
  bool applied = false;
  ASSERT_TRUE(
      registry.record_engine_state(make_state(prefill, 1), 100, &applied).ok());

  xllm::proto::StateBatch full;
  EXPECT_EQ(registry.build_full_state_batch("master", 1, 100, &full).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE);
  EXPECT_TRUE(full.engine_states().empty());
}

TEST(EngineRegistryTest, CapacityAndInvalidConfigurationFailClosed) {
  EngineRegistryConfig config = test_config();
  config.max_members = 1;
  EngineRegistry registry(config);
  const xllm::proto::ProviderDescriptor first = make_descriptor(
      xllm::proto::ENGINE_ROLE_PREFILL, "p1", "inc1", "profile1");
  const xllm::proto::ProviderDescriptor second = make_descriptor(
      xllm::proto::ENGINE_ROLE_PREFILL, "p2", "inc2", "profile2");
  ASSERT_TRUE(registry.upsert_member(first).ok());
  EXPECT_EQ(registry.upsert_member(second).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE);

  xllm::proto::ProviderDescriptor collision = first;
  collision.mutable_identity()->set_engine_uid("different-name");
  EXPECT_EQ(registry.upsert_member(collision).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH);

  config.state_soft_ttl_ms = 21;
  EngineRegistry invalid(config);
  EXPECT_EQ(invalid.upsert_member(first).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE);
}

TEST(EngineRegistryTest, ConcurrentMembershipUpdatesRemainBounded) {
  EngineRegistry registry(test_config());
  constexpr size_t kThreadCount = 16;
  std::vector<int> inserted(kThreadCount, 0);
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (size_t i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([i, &inserted, &registry]() {
      const std::string suffix = std::to_string(i);
      inserted[i] =
          registry
              .upsert_member(make_descriptor(xllm::proto::ENGINE_ROLE_PREFILL,
                                             "p-" + suffix,
                                             "inc-" + suffix,
                                             "profile-" + suffix))
              .ok();
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(registry.member_count(), kThreadCount);
  for (int result : inserted) {
    EXPECT_TRUE(result);
  }
}

TEST(EngineRegistryTest, ColdReplicaCannotUseStateBeforeFirstFull) {
  EngineRegistry registry(test_config());
  const xllm::proto::ProviderDescriptor descriptor = make_descriptor(
      xllm::proto::ENGINE_ROLE_PREFILL, "p", "p-inc", "p-profile");
  const xllm::proto::ProviderEngineKey key =
      make_provider_engine_key(descriptor);
  ASSERT_TRUE(registry.upsert_member(descriptor).ok());
  set_registry_view(&registry, "master");
  bool applied = false;
  ASSERT_TRUE(
      registry.record_engine_state(make_state(descriptor, 1), 100, &applied)
          .ok());
  ASSERT_TRUE(applied);

  EXPECT_FALSE(registry.is_schedulable(key, 100));
  const std::optional<ObservationSnapshot> observation =
      registry.observation_snapshot(100);
  ASSERT_TRUE(observation.has_value());
  EXPECT_EQ(observation->mode, ObservationMode::STATE_BLIND);
}

TEST(EngineRegistryTest, StateBlindUsesGraceThenFreshDirectEvidence) {
  EngineRegistry registry(test_config());
  const xllm::proto::ProviderDescriptor prefill = make_descriptor(
      xllm::proto::ENGINE_ROLE_PREFILL, "p", "p-inc", "p-profile");
  const xllm::proto::ProviderDescriptor decode = make_descriptor(
      xllm::proto::ENGINE_ROLE_DECODE, "d", "d-inc", "d-profile");
  const xllm::proto::ProviderEngineKey prefill_key =
      make_provider_engine_key(prefill);
  const xllm::proto::ProviderEngineKey decode_key =
      make_provider_engine_key(decode);
  ASSERT_TRUE(registry.upsert_member(prefill).ok());
  ASSERT_TRUE(registry.upsert_member(decode).ok());
  set_registry_view(&registry, "master");
  xllm::proto::StateBatch full =
      make_batch("master", 1, xllm::proto::STATE_BATCH_KIND_FULL);
  *full.add_engine_states() = make_state(prefill, 1);
  *full.add_engine_states() = make_state(decode, 1);
  *full.add_link_states() = make_link(prefill, decode, 1);
  bool applied = false;
  ASSERT_TRUE(registry.apply_state_batch(full, 100, &applied).ok());
  ASSERT_TRUE(applied);
  ASSERT_TRUE(registry.is_link_ready(prefill_key, decode_key, 100));

  EXPECT_FALSE(registry.is_schedulable(prefill_key, 121));
  EXPECT_TRUE(registry.is_schedulable(prefill_key, 126));
  EXPECT_TRUE(registry.is_link_ready(prefill_key, decode_key, 126));
  EXPECT_FALSE(registry.is_link_ready(prefill_key, decode_key, 136));

  ASSERT_TRUE(registry.record_direct_evidence(prefill_key, true, 136).ok());
  ASSERT_TRUE(registry.record_direct_evidence(decode_key, true, 136).ok());
  EXPECT_TRUE(registry.is_link_ready(prefill_key, decode_key, 136));
  ASSERT_TRUE(registry.record_direct_evidence(decode_key, false, 137).ok());
  EXPECT_FALSE(registry.is_link_ready(prefill_key, decode_key, 137));
  ASSERT_TRUE(registry.record_direct_evidence(decode_key, true, 138).ok());
  EXPECT_TRUE(registry.is_link_ready(prefill_key, decode_key, 138));
  EXPECT_FALSE(registry.is_link_ready(prefill_key, decode_key, 147));
}

TEST(EngineRegistryTest, RegistryBlindExpiresAndFailureIsImmediate) {
  EngineRegistry registry(test_config());
  const xllm::proto::ProviderDescriptor descriptor = make_descriptor(
      xllm::proto::ENGINE_ROLE_PREFILL, "p", "p-inc", "p-profile");
  const xllm::proto::ProviderEngineKey key =
      make_provider_engine_key(descriptor);
  ASSERT_TRUE(registry.upsert_member(descriptor).ok());
  set_registry_view(&registry, "master");
  xllm::proto::StateBatch full =
      make_batch("master", 1, xllm::proto::STATE_BATCH_KIND_FULL);
  *full.add_engine_states() = make_state(descriptor, 1);
  bool applied = false;
  ASSERT_TRUE(registry.apply_state_batch(full, 100, &applied).ok());
  ASSERT_TRUE(registry.is_schedulable(key, 100));

  ASSERT_TRUE(registry.set_registry_visibility(false).ok());
  EXPECT_TRUE(registry.is_schedulable(key, 101));
  ASSERT_TRUE(registry.record_direct_evidence(key, false, 102).ok());
  EXPECT_FALSE(registry.is_schedulable(key, 102));
  ASSERT_TRUE(registry.record_direct_evidence(key, true, 103).ok());
  EXPECT_TRUE(registry.is_schedulable(key, 103));
  EXPECT_FALSE(registry.is_schedulable(key, 105));
}

TEST(EngineRegistryTest, CurrentMasterFullAndExitHoldRecoverStateBlind) {
  EngineRegistry registry(test_config());
  const xllm::proto::ProviderDescriptor descriptor = make_descriptor(
      xllm::proto::ENGINE_ROLE_PREFILL, "p", "p-inc", "p-profile");
  const xllm::proto::ProviderEngineKey key =
      make_provider_engine_key(descriptor);
  ASSERT_TRUE(registry.upsert_member(descriptor).ok());
  set_registry_view(&registry, "master");
  xllm::proto::StateBatch full =
      make_batch("master", 1, xllm::proto::STATE_BATCH_KIND_FULL);
  *full.add_engine_states() = make_state(descriptor, 1);
  bool applied = false;
  ASSERT_TRUE(registry.apply_state_batch(full, 100, &applied).ok());
  ASSERT_TRUE(registry.is_schedulable(key, 100));
  EXPECT_FALSE(registry.is_schedulable(key, 121));
  EXPECT_TRUE(registry.is_schedulable(key, 126));

  xllm::proto::StateBatch recovered =
      make_batch("master", 2, xllm::proto::STATE_BATCH_KIND_FULL);
  *recovered.add_engine_states() = make_state(descriptor, 2);
  ASSERT_TRUE(registry.apply_state_batch(recovered, 130, &applied).ok());
  ASSERT_TRUE(applied);
  const std::optional<ObservationSnapshot> recovering =
      registry.observation_snapshot(130);
  ASSERT_TRUE(recovering.has_value());
  EXPECT_EQ(recovering->mode, ObservationMode::STATE_BLIND);
  const std::optional<ObservationSnapshot> recovered_observation =
      registry.observation_snapshot(135);
  ASSERT_TRUE(recovered_observation.has_value());
  EXPECT_EQ(recovered_observation->mode, ObservationMode::NORMAL);
  EXPECT_TRUE(registry.is_schedulable(key, 135));
}

TEST(EngineRegistryTest, PublishAgeOverflowAndMissingAgeAreRejected) {
  EngineRegistry registry(test_config());
  const xllm::proto::ProviderDescriptor descriptor = make_descriptor(
      xllm::proto::ENGINE_ROLE_PREFILL, "p", "p-inc", "p-profile");
  ASSERT_TRUE(registry.upsert_member(descriptor).ok());
  set_registry_view(&registry, "master");
  xllm::proto::StateBatch full =
      make_batch("master", 1, xllm::proto::STATE_BATCH_KIND_FULL);
  *full.add_engine_states() = make_state(descriptor, 1);
  full.mutable_engine_states(0)->clear_state_age_ms_at_publish();
  bool applied = false;
  EXPECT_EQ(registry.apply_state_batch(full, 100, &applied).error(),
            xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD);

  full.mutable_engine_states(0)->set_state_age_ms_at_publish(
      std::numeric_limits<uint64_t>::max());
  ASSERT_TRUE(registry.apply_state_batch(full, 100, &applied).ok());
  EXPECT_EQ(registry.state_freshness(make_provider_engine_key(descriptor), 101),
            EngineStateFreshness::HARD_STALE);
  EXPECT_FALSE(
      registry.is_schedulable(make_provider_engine_key(descriptor), 101));
}

}  // namespace
}  // namespace xllm_service::provider
