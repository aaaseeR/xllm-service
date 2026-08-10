/* Copyright 2025-2026 The xLLM Authors.

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

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/macros.h"
#include "common/options.h"
#include "common/threadpool.h"
#include "common/time_predictor.h"
#include "common/types.h"
#include "provider/engine_registry.h"
#include "provider/kv_route_planner.h"
#include "provider/link_reconciler.h"
#include "request/request.h"
#include "scheduler/etcd_client/etcd_client.h"
#include "xllm_rpc_service.pb.h"

namespace xllm_service {
class Scheduler;

class InstanceMgr final {
 public:
  explicit InstanceMgr(const Options& options,
                       const std::shared_ptr<EtcdClient>& etcd_client,
                       const bool is_master_service,
                       Scheduler* scheduler);

  ~InstanceMgr();

  InstanceMetaInfo get_instance_info(const std::string& instance_name);

  bool get_next_provider(const std::string& model_revision,
                         xllm::proto::ProviderId* provider_id);

  bool get_next_instance_pair(Routing* routing,
                              xllm::proto::ProviderId provider_id,
                              const std::string& model_revision);

  std::vector<std::string> get_static_decode_list(
      const std::string& instance_name);

  std::vector<std::string> get_static_prefill_list(
      const std::string& instance_name);

  void get_load_metrics(LoadBalanceInfos* infos,
                        xllm::proto::ProviderId provider_id);

  bool get_kv_route_candidates(
      xllm::proto::ProviderId provider_id,
      const std::string& model_revision,
      size_t max_candidate_plans,
      std::vector<provider::KVRoutePlanCandidate>* candidates,
      bool* truncated);

  std::shared_ptr<brpc::Channel> get_channel(const std::string& instance_name);

  bool bind_request_instance_incarnations(
      const std::shared_ptr<Request>& request);
  bool validate_request_instance_incarnations(
      const std::shared_ptr<Request>& request) const;
  bool record_instance_heartbeat(const std::string& instance_name,
                                 const std::string& incarnation_id);
  bool record_direct_engine_evidence(const std::string& instance_name,
                                     const std::string& incarnation_id,
                                     bool success);
  void record_load_metrics_update(const std::string& instance_name,
                                  const proto::LoadMetrics& load_metrics);
  bool upload_load_metrics();

  provider::ContractResult set_engine_state_registry_visibility(
      bool registry_known);
  provider::ContractResult set_engine_state_master(
      std::string master_incarnation);
  provider::ContractResult apply_engine_state_batch(
      const xllm::proto::StateBatch& batch,
      uint64_t receiver_monotonic_ms,
      bool* applied);
  provider::ContractResult record_engine_state(
      const xllm::proto::EngineState& state,
      uint64_t receiver_monotonic_ms,
      bool* applied);
  provider::ContractResult record_link_state(
      const xllm::proto::LinkState& state,
      uint64_t receiver_monotonic_ms,
      bool* applied);
  provider::ContractResult build_full_state_batch(
      const std::string& master_incarnation,
      uint64_t snapshot_seq,
      uint64_t publish_monotonic_ms,
      xllm::proto::StateBatch* batch) const;
  bool has_current_engine_state_full_snapshot() const;
  bool has_accepted_engine_state_full_snapshot() const;
  std::optional<provider::ObservationSnapshot> engine_observation_snapshot(
      uint64_t receiver_monotonic_ms) const;
  provider::EngineKVCapacitySnapshot engine_kv_capacity_snapshot(
      uint64_t receiver_monotonic_ms) const;
  provider::ContractResult snapshot_engine_members(
      uint64_t receiver_monotonic_ms,
      size_t max_members,
      std::vector<provider::EngineRegistryMemberSnapshot>* snapshot) const;
  provider::EngineRegistry* mutable_engine_registry();
  size_t engine_member_count() const;
  size_t engine_state_count() const;
  size_t engine_link_count() const;

  void require_provider_link_recheck();

  // update the recent token latency metrics for the corresponding instance
  void update_latency_metrics(const std::string& instance_name,
                              const proto::LatencyMetrics& latency_metrics);

  // update request metrics under different actions
  void update_request_metrics(std::shared_ptr<Request> request,
                              RequestAction action);

  // select instances based on the SLO
  bool select_instance_pair_on_slo(std::shared_ptr<Request> request);

  void set_as_master();
  void set_as_follower();

  // Returns true if at least one valid instance group is available:
  // - a single DEFAULT instance, or
  // - a PREFILL + DECODE pair, or
  // - two MIX instances with complementary current_type (one PREFILL, one
  // DECODE)
  bool has_available_instances() const;
  bool has_available_instances_at(uint64_t now_monotonic_ms) const;

 private:
  XLLM_SERVICE_DISALLOW_COPY_AND_ASSIGN(InstanceMgr);

  void init();

  // brpc::Channel::Init only; must NOT be called while holding cluster_mutex_.
  bool init_brpc_channel(const std::string& target_uri,
                         const InstanceMetaInfo& info,
                         std::shared_ptr<brpc::Channel>* out_channel);
  bool probe_direct_engine_health(const std::string& instance_name) const;
  void probe_state_blind_engines(uint64_t now_monotonic_ms);
  void reconcile_direct_engine_evidence();
  void reconcile_instance_states();
  void reconcile_provider_links();
  void publish_link_state(const xllm::proto::LinkState& state,
                          uint64_t now_monotonic_ms);
  void refresh_instance_registration(const std::string& name,
                                     const InstanceMetaInfo& info);
  // use etcd as ServiceDiscovery
  void update_instance_metainfo(const etcd::Response& response,
                                const uint64_t& prefix_len);

  void update_load_metrics(const etcd::Response& response,
                           const uint64_t& prefix_len);

  TimePredictor& get_time_predictor(const std::string& instance_name);

  void flip_prefill_to_decode(std::string& instance_name);
  void flip_decode_to_prefill(std::string& instance_name);

  // Register a new instance with all necessary resources and connections
  bool register_instance(const std::string& name, InstanceMetaInfo& info);
  // Remove an instance and clean up its resources and connections
  void deregister_instance(const std::string& name,
                           const std::string& expected_incarnation_id = "");
  // Initialize internal resources for an instance (predictors, metrics)
  void add_instance_resources(const std::string& name,
                              const InstanceMetaInfo& info);
  // Release internal resources for an instance
  void remove_instance_resources(const std::string& name);
  // Build LinkInstance RPC list; caller must hold cluster_mutex_.
  bool gather_link_operations(
      const InstanceMetaInfo& info,
      std::vector<std::pair<std::string, InstanceMetaInfo>>* out_ops);
  // Run LinkInstance calls without holding cluster_mutex_.
  bool run_link_operations(
      const std::vector<std::pair<std::string, InstanceMetaInfo>>& ops);
  // Build UnlinkInstance RPC list; caller must hold cluster_mutex_.
  void gather_unlink_operations(
      const std::string& name,
      const InstanceMetaInfo& info,
      std::vector<std::pair<std::string, InstanceMetaInfo>>* out_ops);
  // Add instance to prefill or decode index according to its type
  void add_instance_to_index(const std::string& name, InstanceMetaInfo& info);
  // Remove instance from prefill or decode index
  void remove_instance_from_index(const std::string& name,
                                  const InstanceMetaInfo& info);
  bool call_link_instance(const std::string& target_rpc_addr,
                          const InstanceMetaInfo& peer_info);
  bool call_unlink_instance(const std::string& target_rpc_addr,
                            const InstanceMetaInfo& peer_info);

  // Locking (scheme B): only two mutexes participate in ordering.
  // L1 cluster_mutex_: instances_, indices, cached_channels_.
  // L2 metrics_mutex_: load_metrics_, request_metrics_, latency_metrics_,
  // time_predictors_, updated_metrics_, removed_instance_.
  // Order when both needed: always lock L1 before L2 (use std::scoped_lock).
  // get_time_predictor() requires metrics_mutex_ held by caller.
  // remove_instance_resources() requires cluster_mutex_ held by caller.

  Options options_;

  std::atomic_bool exited_ = false;
  bool use_etcd_ = false;
  std::atomic_bool is_master_service_ = false;

  std::shared_ptr<EtcdClient> etcd_client_;

  struct RegistryEventRecord {
    int64_t revision = 0;
    std::optional<std::string> deleted_incarnation_id;
  };

  // Serializes membership watch effects. The watch callback can be delivered
  // by several prefix workers, so revision acceptance and its side effects
  // must be one ordered operation.
  std::mutex registry_event_mutex_;
  std::unordered_map<std::string, RegistryEventRecord> registry_event_history_;
  std::atomic_bool registry_event_history_exhausted_ = false;

  // L1 — cluster topology & channels
  mutable std::shared_mutex cluster_mutex_;
  std::unordered_map<std::string, InstanceMetaInfo> instances_;
  std::vector<std::string> prefill_index_;
  std::vector<std::string> decode_index_;
  // Routing is a read-only topology operation. Keep the mutable round-robin
  // cursors behind their own lock so request traffic does not take the
  // topology writer lock and starve Link/Registry reconciliation.
  std::mutex route_cursor_mutex_;
  uint64_t next_prefill_index_ = 0;
  uint64_t next_decode_index_ = 0;
  uint64_t next_provider_index_ = 0;
  size_t next_direct_probe_index_ = 0;
  std::unordered_map<std::string, std::shared_ptr<brpc::Channel>>
      cached_channels_;
  provider::EngineRegistry engine_registry_;
  provider::LinkReconciler link_reconciler_;

  // L2 — metrics & predictors (single lock to avoid order ambiguity)
  std::shared_mutex metrics_mutex_;
  std::unordered_map<std::string, LoadMetrics> load_metrics_;
  std::unordered_map<std::string, LoadMetrics> updated_metrics_;
  std::unordered_set<std::string> removed_instance_;
  std::unordered_map<std::string, TimePredictor> time_predictors_;
  std::unordered_map<std::string, LatencyMetrics> latency_metrics_;
  std::unordered_map<std::string, RequestMetrics> request_metrics_;

  // not own
  // NOTE: need to refactor with scheduler in future
  Scheduler* scheduler_;

  ThreadPool threadpool_;
  std::unique_ptr<std::thread> state_reconcile_thread_;
  std::unique_ptr<std::thread> direct_probe_thread_;
};

}  // namespace xllm_service
