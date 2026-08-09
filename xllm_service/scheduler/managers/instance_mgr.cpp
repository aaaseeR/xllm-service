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

#include "instance_mgr.h"

#include <absl/strings/str_join.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <brpc/controller.h>
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/global_gflags.h"
#include "common/types.h"
#include "common/utils.h"
#include "common/xllm/output.h"
#include "common/xllm/status.h"
#include "disagg_pd.pb.h"
#include "provider/provider_contract.h"
#include "provider/provider_route_selector.h"
#include "scheduler/scheduler.h"

namespace {
using xllm_service::InstanceRuntimeState;
using xllm_service::InstanceType;
std::unordered_map<InstanceType, std::string> ETCD_KEYS_PREFIX_MAP = {
    {InstanceType::DEFAULT, "XLLM:DEFAULT:"},
    {InstanceType::PREFILL, "XLLM:PREFILL:"},
    {InstanceType::DECODE, "XLLM:DECODE:"},
    {InstanceType::MIX, "XLLM:MIX:"},
};

std::string ETCD_ALL_KEYS_PREFIX = "XLLM:";
std::string ETCD_LOADMETRICS_PREFIX = "XLLM:LOADMETRICS:";

constexpr char kHealthPath[] = "/health";
constexpr int64_t kDeleteProbeRetryBackoffMs = 100;

uint64_t current_time_ms() {
  return static_cast<uint64_t>(
      absl::ToInt64Milliseconds(absl::Now() - absl::UnixEpoch()));
}

uint64_t monotonic_time_ms() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

xllm_service::provider::EngineRegistryConfig engine_registry_config(
    const xllm_service::Options& options) {
  return xllm_service::provider::EngineRegistryConfig{
      .max_members = options.engine_registry_max_members(),
      .max_links = options.engine_registry_max_links(),
      .state_soft_ttl_ms = options.engine_state_soft_ttl_ms(),
      .state_hard_ttl_ms = options.engine_state_hard_ttl_ms(),
      .heartbeat_hard_ttl_ms = options.engine_heartbeat_hard_ttl_ms(),
      .link_hard_ttl_ms = options.engine_link_hard_ttl_ms(),
  };
}

xllm_service::provider::LinkReconcilerConfig link_reconciler_config(
    const xllm_service::Options& options) {
  return xllm_service::provider::LinkReconcilerConfig{
      .max_links = options.engine_registry_max_links(),
      .retry_initial_ms = options.engine_link_retry_initial_ms(),
      .retry_max_ms = options.engine_link_retry_max_ms(),
      .ready_recheck_ms = options.engine_link_ready_recheck_ms(),
  };
}

bool same_provider_engine_key(const xllm::proto::ProviderEngineKey& left,
                              const xllm::proto::ProviderEngineKey& right) {
  return left.provider_id() == right.provider_id() &&
         left.profile_digest() == right.profile_digest() &&
         left.engine_uid() == right.engine_uid() &&
         left.incarnation_id() == right.incarnation_id();
}

bool is_instance_schedulable(
    const xllm_service::InstanceMetaInfo& info,
    const xllm_service::provider::EngineRegistry& registry,
    uint64_t now_monotonic_ms) {
  if (info.runtime_state == InstanceRuntimeState::SUSPECT) {
    return false;
  }
  if (!info.provider_descriptor.has_value() ||
      !registry.has_current_full_snapshot()) {
    // Rolling migration fallback until the first valid State Stream FULL.
    return true;
  }
  return registry.is_schedulable(
      xllm_service::provider::make_provider_engine_key(
          *info.provider_descriptor),
      now_monotonic_ms);
}

size_t count_schedulable_instances(
    const std::unordered_map<std::string, xllm_service::InstanceMetaInfo>&
        instances,
    const std::vector<std::string>& index,
    const xllm_service::provider::EngineRegistry& registry,
    uint64_t now_monotonic_ms) {
  size_t count = 0;
  for (const auto& name : index) {
    auto it = instances.find(name);
    if (it == instances.end() ||
        !is_instance_schedulable(it->second, registry, now_monotonic_ms)) {
      continue;
    }
    ++count;
  }
  return count;
}

InstanceType get_cleanup_type(const xllm_service::InstanceMetaInfo& info) {
  if (info.type == InstanceType::DEFAULT) {
    return InstanceType::PREFILL;
  }
  if (info.type == InstanceType::MIX) {
    return info.current_type;
  }
  return info.type;
}

xllm::proto::EngineRole get_engine_role(
    const xllm_service::InstanceMetaInfo& info) {
  switch (info.type) {
    case InstanceType::DEFAULT:
      return xllm::proto::ENGINE_ROLE_AGGREGATED;
    case InstanceType::PREFILL:
      return xllm::proto::ENGINE_ROLE_PREFILL;
    case InstanceType::DECODE:
      return xllm::proto::ENGINE_ROLE_DECODE;
    case InstanceType::MIX:
      return info.current_type == InstanceType::PREFILL
                 ? xllm::proto::ENGINE_ROLE_PREFILL
                 : xllm::proto::ENGINE_ROLE_DECODE;
    default:
      return xllm::proto::ENGINE_ROLE_UNSPECIFIED;
  }
}

std::vector<xllm_service::provider::ProviderRouteCandidate>
make_route_candidates(
    const std::unordered_map<std::string, xllm_service::InstanceMetaInfo>&
        instances,
    const std::vector<std::string>& index,
    const xllm_service::provider::EngineRegistry& registry,
    uint64_t now_monotonic_ms) {
  std::vector<xllm_service::provider::ProviderRouteCandidate> candidates;
  candidates.reserve(index.size());
  for (const std::string& engine_uid : index) {
    xllm_service::provider::ProviderRouteCandidate candidate;
    candidate.engine_uid = engine_uid;
    const auto instance_it = instances.find(engine_uid);
    if (instance_it != instances.end()) {
      candidate.provider_id = instance_it->second.provider_id;
      candidate.role = get_engine_role(instance_it->second);
      candidate.schedulable = is_instance_schedulable(
          instance_it->second, registry, now_monotonic_ms);
      if (instance_it->second.provider_descriptor.has_value()) {
        candidate.descriptor = &instance_it->second.provider_descriptor.value();
      }
    }
    candidates.emplace_back(std::move(candidate));
  }
  return candidates;
}

void apply_link_readiness(
    std::vector<xllm_service::provider::ProviderRouteCandidate>* prefills,
    const std::vector<xllm_service::provider::ProviderRouteCandidate>& decodes,
    const xllm_service::provider::EngineRegistry& registry,
    uint64_t now_monotonic_ms) {
  if (prefills == nullptr || !registry.has_current_full_snapshot()) {
    return;
  }
  for (xllm_service::provider::ProviderRouteCandidate& prefill : *prefills) {
    if (prefill.descriptor == nullptr ||
        prefill.role != xllm::proto::ENGINE_ROLE_PREFILL) {
      continue;
    }
    prefill.link_state_required = true;
    for (const xllm_service::provider::ProviderRouteCandidate& decode :
         decodes) {
      if (decode.descriptor == nullptr ||
          decode.role != xllm::proto::ENGINE_ROLE_DECODE) {
        continue;
      }
      if (registry.is_link_ready(
              xllm_service::provider::make_provider_engine_key(
                  *prefill.descriptor),
              xllm_service::provider::make_provider_engine_key(
                  *decode.descriptor),
              now_monotonic_ms)) {
        prefill.ready_peer_engine_uids.emplace_back(decode.engine_uid);
      }
    }
  }
}

xllm_service::provider::ProviderRouteCandidate make_route_candidate(
    const std::string& engine_uid,
    const xllm_service::InstanceMetaInfo& info,
    const xllm_service::provider::EngineRegistry& registry,
    uint64_t now_monotonic_ms) {
  return xllm_service::provider::ProviderRouteCandidate{
      .engine_uid = engine_uid,
      .provider_id = info.provider_id,
      .role = get_engine_role(info),
      .schedulable = is_instance_schedulable(info, registry, now_monotonic_ms),
      .descriptor = info.provider_descriptor.has_value()
                        ? &info.provider_descriptor.value()
                        : nullptr,
  };
}

bool are_remote_pd_peers_compatible(
    const xllm_service::InstanceMetaInfo& prefill,
    const xllm_service::InstanceMetaInfo& decode,
    const xllm_service::provider::EngineRegistry& registry,
    uint64_t now_monotonic_ms) {
  const std::vector<xllm_service::provider::ProviderRouteCandidate> prefills = {
      make_route_candidate(prefill.name, prefill, registry, now_monotonic_ms)};
  const std::vector<xllm_service::provider::ProviderRouteCandidate> decodes = {
      make_route_candidate(decode.name, decode, registry, now_monotonic_ms)};
  xllm_service::provider::ProviderRouteSelection selection;
  if (!xllm_service::provider::ProviderRouteSelector::select(
          prefills, decodes, prefill.provider_id, 0, 0, &selection)) {
    return false;
  }
  if (!registry.has_current_full_snapshot() ||
      !prefill.provider_descriptor.has_value() ||
      !decode.provider_descriptor.has_value()) {
    return true;
  }
  return registry.is_link_ready(
      xllm_service::provider::make_provider_engine_key(
          *prefill.provider_descriptor),
      xllm_service::provider::make_provider_engine_key(
          *decode.provider_descriptor),
      now_monotonic_ms);
}

bool are_remote_pd_descriptors_compatible(
    const xllm_service::InstanceMetaInfo& prefill,
    const xllm_service::InstanceMetaInfo& decode) {
  const std::vector<xllm_service::provider::ProviderRouteCandidate> prefills = {
      xllm_service::provider::ProviderRouteCandidate{
          .engine_uid = prefill.name,
          .provider_id = prefill.provider_id,
          .role = get_engine_role(prefill),
          .schedulable = true,
          .descriptor = prefill.provider_descriptor.has_value()
                            ? &prefill.provider_descriptor.value()
                            : nullptr}};
  const std::vector<xllm_service::provider::ProviderRouteCandidate> decodes = {
      xllm_service::provider::ProviderRouteCandidate{
          .engine_uid = decode.name,
          .provider_id = decode.provider_id,
          .role = get_engine_role(decode),
          .schedulable = true,
          .descriptor = decode.provider_descriptor.has_value()
                            ? &decode.provider_descriptor.value()
                            : nullptr}};
  xllm_service::provider::ProviderRouteSelection selection;
  return xllm_service::provider::ProviderRouteSelector::select(
      prefills, decodes, prefill.provider_id, 0, 0, &selection);
}
}  // namespace

namespace xllm_service {

InstanceMgr::InstanceMgr(const Options& options,
                         const std::shared_ptr<EtcdClient>& etcd_client,
                         const bool is_master_service,
                         Scheduler* scheduler)
    : options_(options),
      is_master_service_(is_master_service),
      etcd_client_(etcd_client),
      engine_registry_(engine_registry_config(options)),
      link_reconciler_(link_reconciler_config(options)),
      scheduler_(scheduler) {
  if (options_.engine_link_retry_initial_ms() == 0 ||
      options_.engine_link_retry_max_ms() <
          options_.engine_link_retry_initial_ms() ||
      options_.engine_link_ready_recheck_ms() == 0 ||
      options_.engine_link_ready_recheck_ms() >=
          options_.engine_link_hard_ttl_ms() ||
      options_.engine_link_reconcile_batch_size() == 0) {
    LOG(FATAL) << "Provider Link reconciler configuration is invalid.";
  }
  auto handle_instance_metainfo =
      std::bind(&InstanceMgr::update_instance_metainfo,
                this,
                std::placeholders::_1,
                std::placeholders::_2);
  for (auto& it : ETCD_KEYS_PREFIX_MAP) {
    etcd_client_->add_watch(it.second, handle_instance_metainfo);
  }
  if (!is_master_service_) {
    auto handle_load_metrics = std::bind(&InstanceMgr::update_load_metrics,
                                         this,
                                         std::placeholders::_1,
                                         std::placeholders::_2);
    etcd_client_->add_watch(ETCD_LOADMETRICS_PREFIX, handle_load_metrics);
  }

  init();

  state_reconcile_thread_ = std::make_unique<std::thread>(
      &InstanceMgr::reconcile_instance_states, this);
}

void InstanceMgr::init() {
  std::unordered_map<std::string, InstanceMetaInfo> loaded_instances;
  for (auto& it : ETCD_KEYS_PREFIX_MAP) {
    etcd_client_->get_prefix(it.second, &loaded_instances);
  }
  LOG(INFO) << "Load instance info from etcd:" << loaded_instances.size();

  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    prefill_index_.reserve(loaded_instances.size());
    decode_index_.reserve(loaded_instances.size());
  }

  for (auto& pair : loaded_instances) {
    if (!register_instance(pair.first, pair.second)) {
      LOG(ERROR) << "Fail to register instance: " << pair.first;
    }
  }

  std::unordered_map<std::string, LoadMetrics> loaded_metrics;
  etcd_client_->get_prefix(ETCD_LOADMETRICS_PREFIX, &loaded_metrics);
  {
    std::unique_lock<std::shared_mutex> lock(metrics_mutex_);
    load_metrics_ = std::move(loaded_metrics);
  }

  {
    std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
    for (int i = 0; i < prefill_index_.size(); i++) {
      LOG(INFO) << i << " : " << prefill_index_[i];
    }
  }
}

InstanceMgr::~InstanceMgr() {
  exited_.store(true, std::memory_order_release);
  if (state_reconcile_thread_ && state_reconcile_thread_->joinable()) {
    state_reconcile_thread_->join();
  }
}

InstanceMetaInfo InstanceMgr::get_instance_info(
    const std::string& instance_name) {
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Get instance info failed, instance is not registered, "
                  "instance_name: "
               << instance_name;
    return InstanceMetaInfo();
  }
  return instances_[instance_name];
}

bool InstanceMgr::get_next_provider(xllm::proto::ProviderId* provider_id) {
  std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
  if (provider_id == nullptr) {
    return false;
  }

  const uint64_t now_monotonic_ms = monotonic_time_ms();
  std::vector<provider::ProviderRouteCandidate> prefill_candidates =
      make_route_candidates(
          instances_, prefill_index_, engine_registry_, now_monotonic_ms);
  const std::vector<provider::ProviderRouteCandidate> decode_candidates =
      make_route_candidates(
          instances_, decode_index_, engine_registry_, now_monotonic_ms);
  apply_link_readiness(&prefill_candidates,
                       decode_candidates,
                       engine_registry_,
                       now_monotonic_ms);
  provider::ProviderRouteSelection selection;
  if (!provider::ProviderRouteSelector::select(
          prefill_candidates,
          decode_candidates,
          xllm::proto::PROVIDER_ID_UNSPECIFIED,
          next_provider_index_,
          next_decode_index_,
          &selection)) {
    LOG(ERROR) << "No provider has a schedulable execution route.";
    return false;
  }
  *provider_id = selection.provider_id;
  next_provider_index_ = selection.next_prefill_index;
  return true;
}

bool InstanceMgr::get_next_instance_pair(Routing* routing,
                                         xllm::proto::ProviderId provider_id) {
  std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
  if (routing == nullptr ||
      provider_id == xllm::proto::PROVIDER_ID_UNSPECIFIED) {
    return false;
  }

  const uint64_t now_monotonic_ms = monotonic_time_ms();
  std::vector<provider::ProviderRouteCandidate> prefill_candidates =
      make_route_candidates(
          instances_, prefill_index_, engine_registry_, now_monotonic_ms);
  const std::vector<provider::ProviderRouteCandidate> decode_candidates =
      make_route_candidates(
          instances_, decode_index_, engine_registry_, now_monotonic_ms);
  apply_link_readiness(&prefill_candidates,
                       decode_candidates,
                       engine_registry_,
                       now_monotonic_ms);
  provider::ProviderRouteSelection selection;
  if (!provider::ProviderRouteSelector::select(prefill_candidates,
                                               decode_candidates,
                                               provider_id,
                                               next_prefill_index_,
                                               next_decode_index_,
                                               &selection)) {
    LOG(ERROR) << "No schedulable route for provider_id="
               << static_cast<int32_t>(provider_id);
    return false;
  }

  routing->prefill_name = selection.prefill_engine_uid;
  routing->decode_name = selection.decode_engine_uid;
  next_prefill_index_ = selection.next_prefill_index;
  next_decode_index_ = selection.next_decode_index;
  return true;
}

// TODO: refactor later, currently return all decode instances
std::vector<std::string> InstanceMgr::get_static_decode_list(
    const std::string& instance_name) {
  std::vector<std::string> decode_list;
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
  const uint64_t now_monotonic_ms = monotonic_time_ms();
  const auto source_it = instances_.find(instance_name);
  if (source_it == instances_.end()) {
    return decode_list;
  }
  for (const auto& inst : instances_) {
    if (get_engine_role(inst.second) == xllm::proto::ENGINE_ROLE_DECODE &&
        is_instance_schedulable(
            inst.second, engine_registry_, now_monotonic_ms) &&
        are_remote_pd_peers_compatible(source_it->second,
                                       inst.second,
                                       engine_registry_,
                                       now_monotonic_ms)) {
      decode_list.emplace_back(inst.second.name);
    }
  }

  return decode_list;
}

// TODO: refactor later, currently return all prefill instances
std::vector<std::string> InstanceMgr::get_static_prefill_list(
    const std::string& instance_name) {
  std::vector<std::string> prefill_list;
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
  const uint64_t now_monotonic_ms = monotonic_time_ms();
  const auto source_it = instances_.find(instance_name);
  if (source_it == instances_.end()) {
    return prefill_list;
  }
  for (const auto& inst : instances_) {
    if (get_engine_role(inst.second) == xllm::proto::ENGINE_ROLE_PREFILL &&
        is_instance_schedulable(
            inst.second, engine_registry_, now_monotonic_ms) &&
        are_remote_pd_peers_compatible(inst.second,
                                       source_it->second,
                                       engine_registry_,
                                       now_monotonic_ms)) {
      prefill_list.emplace_back(inst.second.name);
    }
  }

  return prefill_list;
}

void InstanceMgr::get_load_metrics(LoadBalanceInfos* infos,
                                   xllm::proto::ProviderId provider_id) {
  std::shared_lock<std::shared_mutex> inst_lock(cluster_mutex_);
  std::shared_lock<std::shared_mutex> metric_lock(metrics_mutex_);
  const uint64_t now_monotonic_ms = monotonic_time_ms();

  for (auto name : infos->overlap_scores.instances) {
    auto it = load_metrics_.find(name);
    if (it == load_metrics_.end()) {
      continue;
    }
    auto instance_it = instances_.find(name);
    if (instance_it == instances_.end() ||
        instance_it->second.provider_id != provider_id ||
        !is_instance_schedulable(
            instance_it->second, engine_registry_, now_monotonic_ms)) {
      continue;
    }

    if (instance_it->second.type == InstanceType::DECODE) {
      infos->decode_load_metrics.insert(std::make_pair(name, it->second));
      infos->decode_max_waiting_requests_num =
          std::max(infos->decode_max_waiting_requests_num,
                   it->second.waiting_requests_num);
    } else {
      infos->prefill_load_metrics.insert(std::make_pair(name, it->second));
      infos->prefill_max_waiting_requests_num =
          std::max(infos->prefill_max_waiting_requests_num,
                   it->second.waiting_requests_num);
    }
  }

  std::string least_loaded_prefill_instance;
  float least_loaded_prefill_gpu_cache_usage_perc = 1;
  std::string least_loaded_decode_instance;
  float least_loaded_decode_gpu_cache_usage_perc = 1;

  if (infos->prefill_load_metrics.size() == 0 ||
      infos->decode_load_metrics.size() == 0) {
    for (const auto& metric : load_metrics_) {
      auto instance_it = instances_.find(metric.first);
      if (instance_it == instances_.end() ||
          instance_it->second.provider_id != provider_id ||
          !is_instance_schedulable(
              instance_it->second, engine_registry_, now_monotonic_ms)) {
        continue;
      }
      if (instance_it->second.type != InstanceType::DECODE) {
        if (metric.second.gpu_cache_usage_perc <
            least_loaded_prefill_gpu_cache_usage_perc) {
          least_loaded_prefill_gpu_cache_usage_perc =
              metric.second.gpu_cache_usage_perc;
          least_loaded_prefill_instance = metric.first;
        }
      } else {
        if (metric.second.gpu_cache_usage_perc <
            least_loaded_decode_gpu_cache_usage_perc) {
          least_loaded_decode_gpu_cache_usage_perc =
              metric.second.gpu_cache_usage_perc;
          least_loaded_decode_instance = metric.first;
        }
      }
    }
  }

  if (infos->prefill_load_metrics.size() == 0 &&
      !least_loaded_prefill_instance.empty()) {
    infos->prefill_load_metrics.insert(
        std::make_pair(least_loaded_prefill_instance,
                       load_metrics_[least_loaded_prefill_instance]));
  }

  if (infos->decode_load_metrics.size() == 0 &&
      !least_loaded_decode_instance.empty()) {
    infos->decode_load_metrics.insert(
        std::make_pair(least_loaded_decode_instance,
                       load_metrics_[least_loaded_decode_instance]));
  }
}

void InstanceMgr::record_load_metrics_update(
    const std::string& instance_name,
    const proto::LoadMetrics& load_metrics) {
  std::unique_lock<std::shared_mutex> lock(metrics_mutex_);

  updated_metrics_.insert_or_assign(
      instance_name,
      LoadMetrics(load_metrics.waiting_requests_num(),
                  load_metrics.gpu_cache_usage_perc()));
}

bool InstanceMgr::upload_load_metrics() {
  std::unordered_map<std::string, LoadMetrics> upload_snapshot;
  std::unordered_set<std::string> remove_snapshot;
  {
    std::unique_lock<std::shared_mutex> lk(metrics_mutex_);
    for (auto& iter : updated_metrics_) {
      load_metrics_.insert_or_assign(iter.first, iter.second);
    }
    for (auto& iter : removed_instance_) {
      load_metrics_.erase(iter);
    }
    upload_snapshot = updated_metrics_;
    remove_snapshot = removed_instance_;
    updated_metrics_.clear();
    removed_instance_.clear();
  }
  bool status = etcd_client_->set(ETCD_LOADMETRICS_PREFIX, upload_snapshot);
  status = status && etcd_client_->rm(ETCD_LOADMETRICS_PREFIX, remove_snapshot);
  return status;
}

provider::ContractResult InstanceMgr::set_engine_state_registry_view(
    bool registry_known,
    std::string master_incarnation) {
  return engine_registry_.set_registry_view(registry_known,
                                            std::move(master_incarnation));
}

provider::ContractResult InstanceMgr::apply_engine_state_batch(
    const xllm::proto::StateBatch& batch,
    uint64_t receiver_monotonic_ms,
    bool* applied) {
  return engine_registry_.apply_state_batch(
      batch, receiver_monotonic_ms, applied);
}

provider::ContractResult InstanceMgr::record_engine_state(
    const xllm::proto::EngineState& state,
    uint64_t receiver_monotonic_ms,
    bool* applied) {
  return engine_registry_.record_engine_state(
      state, receiver_monotonic_ms, applied);
}

provider::ContractResult InstanceMgr::record_link_state(
    const xllm::proto::LinkState& state,
    uint64_t receiver_monotonic_ms,
    bool* applied) {
  return engine_registry_.record_link_state(
      state, receiver_monotonic_ms, applied);
}

provider::ContractResult InstanceMgr::build_full_state_batch(
    const std::string& master_incarnation,
    uint64_t snapshot_seq,
    uint64_t publish_monotonic_ms,
    xllm::proto::StateBatch* batch) const {
  return engine_registry_.build_full_state_batch(
      master_incarnation, snapshot_seq, publish_monotonic_ms, batch);
}

bool InstanceMgr::has_current_engine_state_full_snapshot() const {
  return engine_registry_.has_current_full_snapshot();
}

void InstanceMgr::set_as_master() {
  is_master_service_.store(true, std::memory_order_release);
  etcd_client_->remove_watch(ETCD_LOADMETRICS_PREFIX);
}

void InstanceMgr::set_as_follower() {
  const bool was_master =
      is_master_service_.exchange(false, std::memory_order_acq_rel);
  if (!was_master) {
    return;
  }
  auto handle_load_metrics = std::bind(&InstanceMgr::update_load_metrics,
                                       this,
                                       std::placeholders::_1,
                                       std::placeholders::_2);
  etcd_client_->add_watch(ETCD_LOADMETRICS_PREFIX, handle_load_metrics);
}

void InstanceMgr::require_provider_link_recheck() {
  link_reconciler_.require_recheck();
}

std::shared_ptr<brpc::Channel> InstanceMgr::get_channel(
    const std::string& instance_name) {
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
  auto iter = cached_channels_.find(instance_name);
  if (iter == cached_channels_.end()) {
    return nullptr;
  }
  return iter->second;
}

bool InstanceMgr::bind_request_instance_incarnations(
    const std::shared_ptr<Request>& request) {
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
  const uint64_t now_monotonic_ms = monotonic_time_ms();

  if (request->provider_id == xllm::proto::PROVIDER_ID_UNSPECIFIED) {
    LOG(ERROR) << "Request provider is not selected before route binding.";
    return false;
  }

  // Bind the selected routing to a concrete incarnation before dispatch.
  request->prefill_incarnation_id.clear();
  request->decode_incarnation_id.clear();
  request->prefill_provider_descriptor.reset();
  request->decode_provider_descriptor.reset();

  if (request->routing.prefill_name.empty()) {
    LOG(ERROR) << "Selected route has no prefill or aggregated engine.";
    return false;
  }

  if (!request->routing.prefill_name.empty()) {
    auto prefill_it = instances_.find(request->routing.prefill_name);
    if (prefill_it == instances_.end()) {
      LOG(ERROR) << "Prefill instance is not registered when binding request: "
                 << request->routing.prefill_name;
      return false;
    }
    if (!is_instance_schedulable(
            prefill_it->second, engine_registry_, now_monotonic_ms)) {
      LOG(ERROR) << "Prefill instance is not schedulable when binding request: "
                 << request->routing.prefill_name << ", state: "
                 << runtime_state_name(prefill_it->second.runtime_state);
      return false;
    }
    if (prefill_it->second.provider_id != request->provider_id) {
      LOG(ERROR) << "Prefill provider changed before route binding: "
                 << request->routing.prefill_name;
      return false;
    }
    request->prefill_incarnation_id = prefill_it->second.incarnation_id;
    request->prefill_provider_descriptor =
        prefill_it->second.provider_descriptor;
  }

  if (!request->routing.decode_name.empty()) {
    auto decode_it = instances_.find(request->routing.decode_name);
    if (decode_it == instances_.end()) {
      LOG(ERROR) << "Decode instance is not registered when binding request: "
                 << request->routing.decode_name;
      return false;
    }
    if (!is_instance_schedulable(
            decode_it->second, engine_registry_, now_monotonic_ms)) {
      LOG(ERROR) << "Decode instance is not schedulable when binding request: "
                 << request->routing.decode_name << ", state: "
                 << runtime_state_name(decode_it->second.runtime_state);
      return false;
    }
    if (decode_it->second.provider_id != request->provider_id) {
      LOG(ERROR) << "Decode provider does not match selected request provider: "
                 << request->routing.decode_name;
      return false;
    }
    request->decode_incarnation_id = decode_it->second.incarnation_id;
    request->decode_provider_descriptor = decode_it->second.provider_descriptor;
  }

  const auto selected_prefill_it =
      instances_.find(request->routing.prefill_name);
  std::vector<provider::ProviderRouteCandidate> selected_prefill = {
      make_route_candidate(selected_prefill_it->first,
                           selected_prefill_it->second,
                           engine_registry_,
                           now_monotonic_ms)};
  std::vector<provider::ProviderRouteCandidate> selected_decode;
  if (!request->routing.decode_name.empty()) {
    const auto selected_decode_it =
        instances_.find(request->routing.decode_name);
    selected_decode.emplace_back(
        make_route_candidate(selected_decode_it->first,
                             selected_decode_it->second,
                             engine_registry_,
                             now_monotonic_ms));
  }
  apply_link_readiness(
      &selected_prefill, selected_decode, engine_registry_, now_monotonic_ms);
  provider::ProviderRouteSelection validated_selection;
  if (!provider::ProviderRouteSelector::select(selected_prefill,
                                               selected_decode,
                                               request->provider_id,
                                               0,
                                               0,
                                               &validated_selection) ||
      validated_selection.prefill_engine_uid != request->routing.prefill_name ||
      validated_selection.decode_engine_uid != request->routing.decode_name) {
    LOG(ERROR) << "Load balancer produced an invalid Provider route shape: "
               << request->routing.debug_string();
    return false;
  }

  return true;
}

bool InstanceMgr::validate_request_instance_incarnations(
    const std::shared_ptr<Request>& request) const {
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
  const uint64_t now_monotonic_ms = monotonic_time_ms();
  const auto matches_bound_incarnation =
      [this, now_monotonic_ms](const std::string& instance_name,
                               const std::string& expected_incarnation_id) {
        if (instance_name.empty()) {
          return expected_incarnation_id.empty();
        }
        const auto it = instances_.find(instance_name);
        return it != instances_.end() &&
               is_instance_schedulable(
                   it->second, engine_registry_, now_monotonic_ms) &&
               !expected_incarnation_id.empty() &&
               it->second.incarnation_id == expected_incarnation_id;
      };
  if (!matches_bound_incarnation(request->routing.prefill_name,
                                 request->prefill_incarnation_id) ||
      !matches_bound_incarnation(request->routing.decode_name,
                                 request->decode_incarnation_id)) {
    return false;
  }
  if (request->routing.decode_name.empty() ||
      !engine_registry_.has_current_full_snapshot() ||
      !request->prefill_provider_descriptor.has_value() ||
      !request->decode_provider_descriptor.has_value()) {
    return true;
  }
  return engine_registry_.is_link_ready(
      provider::make_provider_engine_key(*request->prefill_provider_descriptor),
      provider::make_provider_engine_key(*request->decode_provider_descriptor),
      now_monotonic_ms);
}

bool InstanceMgr::record_instance_heartbeat(const std::string& instance_name,
                                            const std::string& incarnation_id) {
  std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
  auto it = instances_.find(instance_name);
  if (it == instances_.end()) {
    LOG(WARNING) << "Ignore heartbeat from unknown instance: " << instance_name;
    return false;
  }

  if (it->second.incarnation_id != incarnation_id) {
    LOG(WARNING) << "Ignore stale heartbeat from instance: " << instance_name
                 << ", current incarnation_id: " << it->second.incarnation_id
                 << ", heartbeat incarnation_id: " << incarnation_id;
    return false;
  }

  it->second.latest_timestamp = current_time_ms();
  if (it->second.runtime_state == InstanceRuntimeState::SUSPECT) {
    // A recovered suspect instance first goes back to LEASE_LOST and must
    // keep heartbeating before becoming fully active again via registration.
    clear_suspect_instance(instance_name, incarnation_id);
    it->second.runtime_state = InstanceRuntimeState::LEASE_LOST;
    LOG(WARNING) << "Heartbeat recovered for suspect instance, move to "
                 << "lease lost state: " << instance_name
                 << ", incarnation_id: " << incarnation_id;
  }
  return true;
}

bool InstanceMgr::init_brpc_channel(
    const std::string& instance_name,
    const InstanceMetaInfo& info,
    std::shared_ptr<brpc::Channel>* out_channel) {
  auto channel = std::make_shared<brpc::Channel>();
  brpc::ChannelOptions options;
  if (info.provider_id == xllm::proto::PROVIDER_ID_VLLM_ASCEND) {
    // Pass the bare host:port to Init: brpc treats any "scheme://" prefix as a
    // naming service, and "http" is not one, so a "http://" target fails with
    // "Unknown naming service". options.protocol selects HTTP instead.
    options.protocol = "http";
    options.timeout_ms = options_.vllm_http_timeout_ms();
    options.max_retry = 0;
    options.connect_timeout_ms = options_.connect_timeout_ms();
  } else {
    options.timeout_ms = options_.timeout_ms();
    options.max_retry = 3;
    options.connect_timeout_ms = options_.connect_timeout_ms();
  }
  std::string load_balancer = "";
  if (channel->Init(instance_name.c_str(), load_balancer.c_str(), &options) !=
      0) {
    LOG(ERROR) << "Fail to initialize channel for " << instance_name
               << " (provider="
               << xllm::proto::ProviderId_Name(info.provider_id) << ")";
    return false;
  }
  *out_channel = std::move(channel);
  return true;
}

bool InstanceMgr::probe_instance_health(const std::string& instance_name) {
  const int attempts = std::max(1, options_.instance_delete_probe_attempts());
  const int timeout_ms =
      std::max(1, options_.instance_delete_probe_timeout_ms());
  const std::string url = "http://" + instance_name + kHealthPath;

  for (int attempt = 1; attempt <= attempts; ++attempt) {
    brpc::Channel channel;
    brpc::ChannelOptions options;
    options.protocol = "http";
    options.timeout_ms = timeout_ms;
    options.connect_timeout_ms = timeout_ms;
    options.max_retry = 0;
    if (channel.Init(url.c_str(), "", &options) != 0) {
      LOG(WARNING) << "Failed to initialize health probe channel, instance: "
                   << instance_name << ", attempt: " << attempt << "/"
                   << attempts;
    } else {
      brpc::Controller cntl;
      cntl.http_request().uri() = url;
      cntl.http_request().set_method(brpc::HTTP_METHOD_GET);
      channel.CallMethod(nullptr, &cntl, nullptr, nullptr, nullptr);
      if (!cntl.Failed() && cntl.http_response().status_code() == 200) {
        return true;
      }
      LOG(WARNING) << "Health probe failed, instance: " << instance_name
                   << ", attempt: " << attempt << "/" << attempts << ", error: "
                   << (cntl.Failed() ? cntl.ErrorText()
                                     : std::to_string(
                                           cntl.http_response().status_code()));
    }

    if (attempt < attempts) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kDeleteProbeRetryBackoffMs));
    }
  }

  return false;
}

void InstanceMgr::update_instance_metainfo(const etcd::Response& response,
                                           const uint64_t& prefix_len) {
  if (response.events().empty() || exited_) {
    return;
  }

  threadpool_.schedule([this,
                        response = std::move(response),
                        prefix_len = std::move(prefix_len)] {
    if (exited_) return;
    for (const auto& event : response.events()) {
      const std::string instance_name = get_event_key_suffix(event, prefix_len);
      if (instance_name.empty()) {
        continue;
      }

      if (event.event_type() == etcd::Event::EventType::PUT) {
        InstanceMetaInfo metainfo;
        auto json_str = get_event_value(event);
        if (!metainfo.parse_from_json(json_str)) {
          LOG(ERROR) << "Parse instance json failed: " << json_str;
          continue;
        }

        std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
        auto existing_it = instances_.find(instance_name);
        if (existing_it == instances_.end()) {
          lock.unlock();
          if (!register_instance(instance_name, metainfo)) {
            LOG(ERROR) << "Fail to register instance: " << instance_name;
          }
          continue;
        }

        if (existing_it->second.incarnation_id == metainfo.incarnation_id) {
          const auto previous_state = existing_it->second.runtime_state;
          refresh_instance_registration(instance_name, metainfo);
          clear_suspect_instance(instance_name, metainfo.incarnation_id);
          if (previous_state != InstanceRuntimeState::ACTIVE) {
            LOG(INFO) << "Instance registration restored, back to active: "
                      << instance_name
                      << ", incarnation_id: " << metainfo.incarnation_id
                      << ", previous_state: "
                      << runtime_state_name(previous_state);
          }
          continue;
        }

        const std::string old_incarnation_id =
            existing_it->second.incarnation_id;
        LOG(WARNING) << "Detected instance replacement, instance_name: "
                     << instance_name
                     << ", old incarnation_id: " << old_incarnation_id
                     << ", new incarnation_id: " << metainfo.incarnation_id;
        lock.unlock();
        deregister_instance(instance_name, old_incarnation_id);
        if (!register_instance(instance_name, metainfo)) {
          LOG(ERROR) << "Fail to register replacement instance: "
                     << instance_name;
        }
        continue;
      }

      if (event.event_type() != etcd::Event::EventType::DELETE_) {
        continue;
      }

      InstanceMetaInfo deleted_info;
      std::string deleted_incarnation_id;
      const auto deleted_value = get_event_value(event);
      if (!deleted_value.empty() &&
          deleted_info.parse_from_json(deleted_value)) {
        deleted_incarnation_id = deleted_info.incarnation_id;
      }
      std::string tracked_incarnation_id;
      {
        std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
        auto existing_it = instances_.find(instance_name);
        if (existing_it == instances_.end()) {
          continue;
        }

        if (!deleted_incarnation_id.empty() &&
            existing_it->second.incarnation_id != deleted_incarnation_id) {
          LOG(INFO) << "Ignore stale delete for replaced instance: "
                    << instance_name
                    << ", deleted incarnation_id: " << deleted_incarnation_id
                    << ", current incarnation_id: "
                    << existing_it->second.incarnation_id;
          continue;
        }
        tracked_incarnation_id = existing_it->second.incarnation_id;
      }

      // Keep delete handling event-driven: use this event, one health probe,
      // and later heartbeats or PUT events to drive recovery.
      const bool probe_success = probe_instance_health(instance_name);

      std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
      auto existing_it = instances_.find(instance_name);
      if (existing_it == instances_.end() ||
          existing_it->second.incarnation_id != tracked_incarnation_id) {
        continue;
      }

      if (probe_success) {
        // Keep the instance in service temporarily and wait for heartbeats.
        clear_suspect_instance(instance_name, tracked_incarnation_id);
        existing_it->second.runtime_state = InstanceRuntimeState::LEASE_LOST;
        existing_it->second.latest_timestamp = current_time_ms();
        LOG(WARNING) << "Instance lease deleted, probe succeeded, enter "
                     << "lease lost state: " << instance_name
                     << ", incarnation_id: " << tracked_incarnation_id;
        continue;
      }

      mark_instance_suspect(instance_name, tracked_incarnation_id);
      LOG(WARNING) << "Instance lease deleted, probe failed, enter suspect "
                   << "state: " << instance_name
                   << ", incarnation_id: " << tracked_incarnation_id;
    }
  });
}

void InstanceMgr::update_load_metrics(const etcd::Response& response,
                                      const uint64_t& prefix_len) {
  if (response.events().empty() || exited_) {
    return;
  }
  threadpool_.schedule([this,
                        response = std::move(response),
                        prefix_len = std::move(prefix_len)] {
    if (exited_) return;
    std::unordered_map<std::string, LoadMetrics> put_map;
    std::vector<std::string> delete_list;

    for (const auto& event : response.events()) {
      std::string instance_name = event.kv().key().substr(prefix_len);

      if (event.event_type() == etcd::Event::EventType::PUT) {
        LoadMetrics load_metrics;
        auto json_str = event.kv().as_string();
        if (!load_metrics.parse_from_json(json_str)) {
          LOG(ERROR) << "pase json:" << json_str << " error!";
          continue;
        }

        put_map.insert(std::make_pair(instance_name, std::move(load_metrics)));

      } else if (event.event_type() == etcd::Event::EventType::DELETE_) {
        delete_list.push_back(instance_name);
      }
    }

    {
      std::unique_lock<std::shared_mutex> lock(metrics_mutex_);
      for (auto& iter : put_map) {
        load_metrics_.insert_or_assign(iter.first, std::move(iter.second));
      }

      for (auto& iter : delete_list) {
        load_metrics_.erase(iter);
      }
    }
  });
}

void InstanceMgr::update_latency_metrics(
    const std::string& instance_name,
    const proto::LatencyMetrics& latency_metrics) {
  std::unique_lock<std::shared_mutex> lock(metrics_mutex_);

  latency_metrics_.insert_or_assign(
      instance_name,
      LatencyMetrics(latency_metrics.recent_max_ttft(),
                     latency_metrics.recent_max_tbt()));
}

void InstanceMgr::reconcile_instance_states() {
  const auto suspect_interval_ms =
      std::max<int64_t>(1, options_.detect_disconnected_instance_interval()) *
      1000;
  const auto heartbeat_timeout_ms =
      std::max<int64_t>(1, options_.lease_lost_heartbeat_timeout_ms());
  while (!exited_) {
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::vector<std::pair<std::string, std::string>> to_deregister;

    {
      std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
      if (exited_) {
        return;
      }

      const uint64_t now_ms = current_time_ms();
      for (auto& [instance_name, info] : instances_) {
        // LEASE_LOST is a grace period after etcd delete but before hard
        // eviction.
        if (info.runtime_state != InstanceRuntimeState::LEASE_LOST) {
          continue;
        }
        if (now_ms - info.latest_timestamp < heartbeat_timeout_ms) {
          continue;
        }
        mark_instance_suspect(instance_name, info.incarnation_id);
        LOG(WARNING)
            << "Lease lost instance heartbeat timed out, enter suspect "
            << "state: " << instance_name
            << ", incarnation_id: " << info.incarnation_id;
      }

      for (auto it = suspect_instances_.begin();
           it != suspect_instances_.end();) {
        const std::string instance_name = it->first;
        const std::string incarnation_id = it->second.incarnation_id;
        const uint64_t enter_ts_ms = it->second.enter_ts_ms;
        ++it;

        if (now_ms - enter_ts_ms < suspect_interval_ms) {
          continue;
        }

        auto inst_it = instances_.find(instance_name);
        if (inst_it == instances_.end() ||
            inst_it->second.incarnation_id != incarnation_id) {
          suspect_instances_.erase(instance_name);
          continue;
        }

        LOG(WARNING) << "Suspect window expired, deregister instance: "
                     << instance_name << ", incarnation_id: " << incarnation_id;
        to_deregister.emplace_back(instance_name, incarnation_id);
      }
    }

    for (const auto& p : to_deregister) {
      deregister_instance(p.first, p.second);
    }
    reconcile_provider_links();
  }
}

void InstanceMgr::publish_link_state(const xllm::proto::LinkState& state,
                                     uint64_t now_monotonic_ms) {
  bool applied = false;
  const provider::ContractResult recorded =
      engine_registry_.record_link_state(state, now_monotonic_ms, &applied);
  if (!recorded.ok()) {
    LOG(WARNING) << "Discard Provider Link state: " << recorded.message();
    return;
  }
  if (applied) {
    scheduler_->notify_engine_link_state_changed(state, now_monotonic_ms);
  }
}

void InstanceMgr::reconcile_provider_links() {
  if (!is_master_service_.load(std::memory_order_acquire)) {
    return;
  }

  std::vector<provider::DesiredProviderLink> desired;
  {
    std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
    desired.reserve(std::min(options_.engine_registry_max_links(),
                             prefill_index_.size() * decode_index_.size()));
    for (const std::string& prefill_name : prefill_index_) {
      const auto prefill = instances_.find(prefill_name);
      if (prefill == instances_.end() ||
          !prefill->second.provider_descriptor.has_value() ||
          get_engine_role(prefill->second) !=
              xllm::proto::ENGINE_ROLE_PREFILL) {
        continue;
      }
      for (const std::string& decode_name : decode_index_) {
        const auto decode = instances_.find(decode_name);
        if (decode == instances_.end() ||
            !decode->second.provider_descriptor.has_value() ||
            get_engine_role(decode->second) !=
                xllm::proto::ENGINE_ROLE_DECODE) {
          continue;
        }
        std::string compatibility_proof;
        if (!provider::validate_remote_pd_compatibility(
                 *prefill->second.provider_descriptor,
                 *decode->second.provider_descriptor,
                 &compatibility_proof)
                 .ok()) {
          continue;
        }
        desired.emplace_back(provider::DesiredProviderLink{
            .prefill = *prefill->second.provider_descriptor,
            .decode = *decode->second.provider_descriptor,
        });
        if (desired.size() > options_.engine_registry_max_links()) {
          LOG(ERROR) << "Desired Provider Link set exceeds configured "
                        "capacity.";
          return;
        }
      }
    }
  }

  const uint64_t now_monotonic_ms = monotonic_time_ms();
  std::vector<xllm::proto::LinkState> state_changes;
  const provider::ContractResult replaced = link_reconciler_.replace_desired(
      desired, now_monotonic_ms, &state_changes);
  if (!replaced.ok()) {
    LOG(ERROR) << "Failed to reconcile desired Provider Links: "
               << replaced.message();
    return;
  }
  for (const xllm::proto::LinkState& state : state_changes) {
    publish_link_state(state, now_monotonic_ms);
  }

  const std::vector<provider::ProviderLinkAttempt> attempts =
      link_reconciler_.begin_due_attempts(
          now_monotonic_ms, options_.engine_link_reconcile_batch_size());
  for (const provider::ProviderLinkAttempt& attempt : attempts) {
    std::string target_rpc_address;
    InstanceMetaInfo prefill_info;
    bool pair_is_current = false;
    {
      std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
      const auto prefill = instances_.find(attempt.prefill.engine_uid());
      const auto decode = instances_.find(attempt.decode.engine_uid());
      if (prefill != instances_.end() && decode != instances_.end() &&
          prefill->second.provider_descriptor.has_value() &&
          decode->second.provider_descriptor.has_value() &&
          same_provider_engine_key(provider::make_provider_engine_key(
                                       *prefill->second.provider_descriptor),
                                   attempt.prefill) &&
          same_provider_engine_key(provider::make_provider_engine_key(
                                       *decode->second.provider_descriptor),
                                   attempt.decode)) {
        target_rpc_address = decode->second.rpc_address;
        prefill_info = prefill->second;
        pair_is_current = true;
      }
    }

    const bool linked =
        pair_is_current && call_link_instance(target_rpc_address, prefill_info);
    xllm::proto::LinkState state;
    const provider::ContractResult completed =
        link_reconciler_.complete_attempt(attempt,
                                          linked,
                                          linked ? "ready" : "link-rpc-failed",
                                          monotonic_time_ms(),
                                          &state);
    if (!completed.ok()) {
      LOG(WARNING) << "Discard stale Provider Link attempt: "
                   << completed.message();
      continue;
    }
    if (is_master_service_.load(std::memory_order_acquire)) {
      publish_link_state(state, monotonic_time_ms());
    }
  }
}

void InstanceMgr::refresh_instance_registration(const std::string& name,
                                                const InstanceMetaInfo& info) {
  auto it = instances_.find(name);
  if (it == instances_.end()) {
    return;
  }

  // Preserve local scheduling/index state across etcd refreshes.
  const auto instance_index = it->second.instance_index;
  const auto current_type = it->second.current_type;

  it->second = info;
  it->second.instance_index = instance_index;
  it->second.current_type = current_type;
  it->second.latest_timestamp = current_time_ms();
  it->second.runtime_state = InstanceRuntimeState::ACTIVE;
}

void InstanceMgr::mark_instance_suspect(const std::string& name,
                                        const std::string& incarnation_id) {
  SuspectInstanceInfo info;
  info.incarnation_id = incarnation_id;
  info.enter_ts_ms = current_time_ms();
  suspect_instances_[name] = std::move(info);
  auto it = instances_.find(name);
  if (it != instances_.end() && it->second.incarnation_id == incarnation_id) {
    it->second.runtime_state = InstanceRuntimeState::SUSPECT;
  }
}

void InstanceMgr::clear_suspect_instance(const std::string& name,
                                         const std::string& incarnation_id) {
  auto it = suspect_instances_.find(name);
  if (it == suspect_instances_.end()) {
    return;
  }
  if (!incarnation_id.empty() && it->second.incarnation_id != incarnation_id) {
    return;
  }
  suspect_instances_.erase(it);
}

void InstanceMgr::update_request_metrics(std::shared_ptr<Request> request,
                                         RequestAction action) {
  // skip request metrics update if policy is not SLO_AWARE
  if (options_.load_balance_policy() != "SLO_AWARE") {
    return;
  }

  std::scoped_lock<std::shared_mutex, std::shared_mutex> lock(cluster_mutex_,
                                                              metrics_mutex_);

  auto prefill_it = request_metrics_.find(request->routing.prefill_name);
  if (prefill_it == request_metrics_.end()) {
    LOG(ERROR) << "Failed to find instance request metrics, instance name : "
               << request->routing.prefill_name;
    return;
  }

  auto decode_it = request_metrics_.find(request->routing.decode_name);
  if (decode_it == request_metrics_.end()) {
    LOG(ERROR) << "Failed to find instance request metrics, instance name : "
               << request->routing.decode_name;
    return;
  }

  int64_t num_prompt_tokens = request->token_ids.size();
  int64_t num_generated_tokens = request->num_generated_tokens;
  switch (action) {
    case RequestAction::SCHEDULE:
      // update the request metrics for prefill and decode instances when
      // request is scheduled
      prefill_it->second.prefill_request_num += 1;
      prefill_it->second.prefill_token_num += num_prompt_tokens;

      decode_it->second.decode_request_num += 1;
      decode_it->second.decode_token_num += num_prompt_tokens;
      break;
    case RequestAction::FINISH_PREFILL:
      // update the request metrics for prefill and decode instance when request
      // finishes the prefill phase
      prefill_it->second.prefill_request_num -= 1;
      prefill_it->second.prefill_token_num -= num_prompt_tokens;
      prefill_it->second.estimated_prefill_time -= request->estimated_ttft;

      decode_it->second.decode_token_num += 1;
      break;
    case RequestAction::GENERATE:
      // update the request metrics for decode instance when request generate a
      // token
      decode_it->second.decode_token_num += 1;
      break;
    case RequestAction::FINISH_DECODE:
      // update the request metrics for decode instance when request finishes
      // the decode phase
      decode_it->second.decode_request_num -= 1;
      decode_it->second.decode_token_num -=
          (num_prompt_tokens + num_generated_tokens);

      break;
    case RequestAction::CANCEL:
      // update the request metrics for prefill and decode instances when
      // request is cancelled
      prefill_it->second.prefill_request_num -= 1;
      prefill_it->second.prefill_token_num -= num_prompt_tokens;
      prefill_it->second.estimated_prefill_time -= request->estimated_ttft;

      decode_it->second.decode_request_num -= 1;
      decode_it->second.decode_token_num -=
          (num_prompt_tokens + num_generated_tokens);

      break;
    default:
      LOG(ERROR) << "Unknown RequestAction: " << static_cast<int32_t>(action);
      break;
  }

  if (decode_it->second.decode_request_num == 0) {
    flip_decode_to_prefill(request->routing.decode_name);
  }
}

bool InstanceMgr::select_instance_pair_on_slo(
    std::shared_ptr<Request> request) {
  std::scoped_lock<std::shared_mutex, std::shared_mutex> lock(cluster_mutex_,
                                                              metrics_mutex_);
  const uint64_t now_monotonic_ms = monotonic_time_ms();

  std::string min_prefill_instance;
  int64_t min_prefill_time = std::numeric_limits<int64_t>::max();
  int64_t total_prefill_time = 0;
  size_t schedulable_prefill_count = 0;
  for (const auto& prefill_instance : prefill_index_) {
    const auto instance_it = instances_.find(prefill_instance);
    if (instance_it == instances_.end() ||
        instance_it->second.provider_id != request->provider_id ||
        !is_instance_schedulable(
            instance_it->second, engine_registry_, now_monotonic_ms)) {
      continue;
    }

    int64_t prefill_time =
        request_metrics_[prefill_instance].estimated_prefill_time;
    total_prefill_time += prefill_time;
    if (prefill_time < min_prefill_time) {
      min_prefill_instance = prefill_instance;
      min_prefill_time = prefill_time;
    }
    ++schedulable_prefill_count;
  }

  if (schedulable_prefill_count == 0) {
    LOG(ERROR) << "No prefill or default instance found!";
    return false;
  }
  int64_t avg_prefill_time = total_prefill_time / schedulable_prefill_count;

  std::string min_decode_instance;
  int64_t min_estimated_tpot = std::numeric_limits<int64_t>::max();
  std::string target_decode_instance;
  size_t schedulable_decode_count = 0;
  for (const auto& decode_instance : decode_index_) {
    const auto instance_it = instances_.find(decode_instance);
    if (instance_it == instances_.end() ||
        instance_it->second.provider_id != request->provider_id ||
        !is_instance_schedulable(
            instance_it->second, engine_registry_, now_monotonic_ms)) {
      continue;
    }

    int64_t token_num = request_metrics_[decode_instance].decode_token_num;
    int64_t request_num = request_metrics_[decode_instance].decode_request_num;
    auto& time_predictor = get_time_predictor(decode_instance);
    int64_t estimated_tpot = time_predictor.predict_tpot(
        token_num + request->token_ids.size(), request_num + 1);
    if (estimated_tpot <= FLAGS_target_tpot && target_decode_instance.empty()) {
      target_decode_instance = decode_instance;
    }

    if (estimated_tpot < min_estimated_tpot) {
      min_decode_instance = decode_instance;
      min_estimated_tpot = estimated_tpot;
    }
    ++schedulable_decode_count;
  }

  if (schedulable_decode_count == 0) {
    LOG(ERROR) << "No decode instance found!";
    return false;
  }

  if (!target_decode_instance.empty()) {
    request->routing.decode_name = target_decode_instance;
  } else {
    request->routing.decode_name = min_decode_instance;
  }

  // select prefill instance
  float tpot_threshold =
      (schedulable_decode_count - 1.0f) / schedulable_decode_count;
  // When the prefill instances are already overloaded and there are other
  // instances with lower loads in the decode group, we will dispatch the
  // prefill requests to those instances to alleviate the pressure on the
  // prefill instances.
  if (min_prefill_time > FLAGS_target_ttft &&
      target_decode_instance != min_decode_instance &&
      min_estimated_tpot < FLAGS_target_tpot * tpot_threshold &&
      request_metrics_[min_decode_instance].estimated_prefill_time <
          min_prefill_time) {
    request->routing.prefill_name = min_decode_instance;
    // update estimated ttft
    auto& time_predictor = get_time_predictor(min_decode_instance);
    request->estimated_ttft =
        time_predictor.predict_ttft(request->token_ids.size());
    request_metrics_[min_decode_instance].estimated_prefill_time +=
        request->estimated_ttft;
  } else {
    request->routing.prefill_name = min_prefill_instance;
    // update estimated ttft
    auto& time_predictor = get_time_predictor(min_prefill_instance);
    request->estimated_ttft =
        time_predictor.predict_ttft(request->token_ids.size());
    request_metrics_[min_prefill_instance].estimated_prefill_time +=
        request->estimated_ttft;
  }

  // If there are no decode instances that meet the requirements, switch a
  // prefill instance to decode if the number of instances allows. Since the
  // current disaggregated PD mode does not support prefill and decode using the
  // same instance, we only switch the instance here, without dispatching the
  // decode request to this instance.
  float ttft_threshold =
      (schedulable_prefill_count - 1.0f) / schedulable_prefill_count;
  if (target_decode_instance.empty() &&
      (avg_prefill_time < FLAGS_target_ttft * ttft_threshold ||
       schedulable_decode_count < schedulable_prefill_count)) {
    flip_prefill_to_decode(request->routing.prefill_name);
  }

  return true;
}

void InstanceMgr::flip_prefill_to_decode(std::string& instance_name) {
  if (count_schedulable_instances(
          instances_, prefill_index_, engine_registry_, monotonic_time_ms()) <=
      1) {
    // Ensure there is at least one prefill instance.
    return;
  }

  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Can't find instance, instance_name: " << instance_name;
    return;
  }

  // delete instance name from prefill_index_
  remove_instance_from_index(instance_name, instances_[instance_name]);

  // insert instance name to decode_index_
  instances_[instance_name].current_type = InstanceType::DECODE;
  add_instance_to_index(instance_name, instances_[instance_name]);

  LOG(INFO) << "Flip prefill to decode, instance name : " << instance_name;
}

void InstanceMgr::flip_decode_to_prefill(std::string& instance_name) {
  if (count_schedulable_instances(
          instances_, decode_index_, engine_registry_, monotonic_time_ms()) <=
      1) {
    // Ensure there is at least one decode instance.
    return;
  }

  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Can't find instance, instance_name: " << instance_name;
    return;
  }

  // delete instance name from decode_index_
  remove_instance_from_index(instance_name, instances_[instance_name]);

  // insert instance name to prefill_index
  instances_[instance_name].current_type = InstanceType::PREFILL;
  add_instance_to_index(instance_name, instances_[instance_name]);

  LOG(INFO) << "Flip decode to prefill, instance name : " << instance_name;
}

TimePredictor& InstanceMgr::get_time_predictor(
    const std::string& instance_name) {
  auto it = time_predictors_.find(instance_name);
  if (it == time_predictors_.end()) {
    LOG(FATAL) << "Find TimePredictor failed, instance name : "
               << instance_name;
  }
  return it->second;
}

bool InstanceMgr::call_link_instance(const std::string& target_rpc_addr,
                                     const InstanceMetaInfo& peer_info) {
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.protocol = "http";
  options.timeout_ms = options_.timeout_ms();
  options.max_retry = 3;
  if (channel.Init(target_rpc_addr.c_str(), "", &options) != 0) {
    LOG(ERROR) << "Fail to initialize channel for LinkInstance to "
               << target_rpc_addr;
    return false;
  }
  xllm::proto::DisaggPDService_Stub stub(&channel);
  brpc::Controller cntl;
  xllm::proto::InstanceClusterInfo req;
  req.set_instance_name(peer_info.name);
  for (auto& cluster_id : peer_info.cluster_ids) {
    req.add_cluster_ids(cluster_id);
  }
  for (auto& addr : peer_info.addrs) {
    req.add_addrs(addr);
  }
  for (auto& port : peer_info.ports) {
    req.add_ports(port);
  }
  req.set_dp_size(peer_info.dp_size);
  req.set_kv_split_size(peer_info.kv_split_size);
  xllm::proto::Status res;
  stub.LinkInstance(&cntl, &req, &res, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "LinkInstance failed, target: " << target_rpc_addr
               << ", peer: " << peer_info.name
               << ", error: " << cntl.ErrorText();
    return false;
  }
  return res.ok();
}

bool InstanceMgr::call_unlink_instance(const std::string& target_rpc_addr,
                                       const InstanceMetaInfo& peer_info) {
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.protocol = "http";
  options.timeout_ms = options_.timeout_ms();
  options.max_retry = 3;
  if (channel.Init(target_rpc_addr.c_str(), "", &options) != 0) {
    LOG(ERROR) << "Fail to initialize channel for UnlinkInstance to "
               << target_rpc_addr;
    return false;
  }
  xllm::proto::DisaggPDService_Stub stub(&channel);
  brpc::Controller cntl;
  xllm::proto::InstanceClusterInfo req;
  req.set_instance_name(peer_info.name);
  for (auto& cluster_id : peer_info.cluster_ids) {
    req.add_cluster_ids(cluster_id);
  }
  for (auto& addr : peer_info.addrs) {
    req.add_addrs(addr);
  }
  for (auto& port : peer_info.ports) {
    req.add_ports(port);
  }
  req.set_dp_size(peer_info.dp_size);
  req.set_kv_split_size(peer_info.kv_split_size);
  xllm::proto::Status res;
  stub.UnlinkInstance(&cntl, &req, &res, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "UnlinkInstance failed, target: " << target_rpc_addr
               << ", peer: " << peer_info.name
               << ", error: " << cntl.ErrorText();
    return false;
  }
  return res.ok();
}

bool InstanceMgr::register_instance(const std::string& name,
                                    InstanceMetaInfo& info) {
  info.runtime_state = InstanceRuntimeState::ACTIVE;
  info.latest_timestamp = current_time_ms();
  info.name = name;

  if (info.type == InstanceType::MIX) {
    std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
    const bool has_provider_decode =
        std::any_of(decode_index_.begin(),
                    decode_index_.end(),
                    [this, &info](const std::string& decode_name) {
                      const auto decode_it = instances_.find(decode_name);
                      return decode_it != instances_.end() &&
                             decode_it->second.provider_id == info.provider_id;
                    });
    info.current_type =
        has_provider_decode ? InstanceType::PREFILL : InstanceType::DECODE;
  }

  if (info.provider_id == xllm::proto::PROVIDER_ID_VLLM_ASCEND &&
      info.type != InstanceType::DEFAULT) {
    LOG(ERROR) << "Reject non-aggregated vLLM-Ascend instance: " << name;
    return false;
  }

  if (info.provider_descriptor.has_value()) {
    if (info.type == InstanceType::MIX) {
      LOG(ERROR) << "Reject mutable MIX role with immutable Provider "
                    "Descriptor: "
                 << name;
      return false;
    }
    const provider::ContractResult validation =
        provider::validate_provider_descriptor(*info.provider_descriptor);
    if (!validation.ok()) {
      LOG(ERROR) << "Reject invalid Provider Descriptor for " << name << ": "
                 << validation.message();
      return false;
    }
    if (info.provider_descriptor->identity().engine_uid() != name ||
        info.provider_descriptor->identity().incarnation_id() !=
            info.incarnation_id) {
      LOG(ERROR) << "Reject Provider Descriptor identity mismatch for " << name;
      return false;
    }
    if (info.provider_descriptor->serving().role() != get_engine_role(info)) {
      LOG(ERROR) << "Reject Provider Descriptor role mismatch for " << name;
      return false;
    }
  }

  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    if (instances_.find(name) != instances_.end() ||
        cached_channels_.find(name) != cached_channels_.end()) {
      LOG(ERROR) << "Instance is already registered, instance_name: " << name;
      return false;
    }
  }

  std::shared_ptr<brpc::Channel> channel;
  if (!init_brpc_channel(name, info, &channel)) {
    LOG(ERROR) << "create channel fail: " << name;
    return false;
  }

  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    if (instances_.find(name) != instances_.end() ||
        cached_channels_.find(name) != cached_channels_.end()) {
      LOG(WARNING) << "Instance registered concurrently during channel init: "
                   << name;
      return false;
    }
    cached_channels_[name] = std::move(channel);
  }

  bool registry_member_installed = false;
  if (info.provider_descriptor.has_value()) {
    const provider::ContractResult registry_result =
        engine_registry_.upsert_member(*info.provider_descriptor);
    if (!registry_result.ok()) {
      LOG(ERROR) << "Fail to add strict instance to Engine Registry: " << name
                 << ", error: " << registry_result.message();
      std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
      cached_channels_.erase(name);
      return false;
    }
    registry_member_installed = true;
  }

  add_instance_resources(name, info);

  // Strict V2 links are independent, incarnation-scoped state machines. Make
  // the member visible first and let the master reconciler publish PENDING,
  // then READY or DEGRADED per pair. A bad peer must not roll back this member
  // or any healthy pair.
  if (info.provider_descriptor.has_value()) {
    {
      std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
      add_instance_to_index(name, info);
      instances_.insert(std::make_pair(name, info));
    }
    scheduler_->notify_engine_registry_membership_changed();
    return true;
  }

  std::vector<std::pair<std::string, InstanceMetaInfo>> link_ops;
  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    if (!gather_link_operations(info, &link_ops)) {
      remove_instance_resources(name);
      if (registry_member_installed) {
        engine_registry_.remove_member(
            provider::make_provider_engine_key(*info.provider_descriptor));
      }
      return false;
    }
  }

  if (!run_link_operations(link_ops)) {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    remove_instance_resources(name);
    if (registry_member_installed) {
      engine_registry_.remove_member(
          provider::make_provider_engine_key(*info.provider_descriptor));
    }
    return false;
  }

  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    add_instance_to_index(name, info);
    instances_.insert(std::make_pair(name, info));
  }
  return true;
}

void InstanceMgr::deregister_instance(
    const std::string& name,
    const std::string& expected_incarnation_id) {
  InstanceMetaInfo info;
  std::vector<std::pair<std::string, InstanceMetaInfo>> unlink_ops;
  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    auto it = instances_.find(name);
    if (it == instances_.end()) {
      LOG(ERROR) << "Instance is not registered, instance_name: " << name;
      return;
    }

    if (!expected_incarnation_id.empty() &&
        it->second.incarnation_id != expected_incarnation_id) {
      LOG(INFO) << "Skip deregistering stale incarnation, instance_name: "
                << name
                << ", current incarnation_id: " << it->second.incarnation_id
                << ", expected incarnation_id: " << expected_incarnation_id;
      return;
    }

    info = it->second;
    clear_suspect_instance(name, info.incarnation_id);
    gather_unlink_operations(name, info, &unlink_ops);
  }

  for (const auto& op : unlink_ops) {
    call_unlink_instance(op.first, op.second);
  }

  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    auto it = instances_.find(name);
    if (it == instances_.end()) {
      return;
    }
    // Close the dispatch gate before notifying Scheduler. The instance and
    // channel remain available until active attempts have converged, but no
    // request bound earlier may enter dispatch while deregistration is in
    // progress.
    it->second.runtime_state = InstanceRuntimeState::SUSPECT;
    remove_instance_from_index(name, it->second);
  }

  if (info.provider_descriptor.has_value()) {
    engine_registry_.remove_member(
        provider::make_provider_engine_key(*info.provider_descriptor));
    scheduler_->notify_engine_registry_membership_changed();
  }

  scheduler_->clear_requests_on_failed_instance(
      name, info.incarnation_id, get_cleanup_type(info));

  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    auto it = instances_.find(name);
    if (it == instances_.end()) {
      return;
    }
    remove_instance_resources(name);
    instances_.erase(it);
  }
  LOG(INFO) << "delete instance: " << name;
}

void InstanceMgr::add_instance_resources(const std::string& name,
                                         const InstanceMetaInfo& info) {
  std::unique_lock<std::shared_mutex> lock(metrics_mutex_);

  time_predictors_.insert_or_assign(
      name, TimePredictor(info.ttft_profiling_data, info.tpot_profiling_data));

  request_metrics_.insert_or_assign(name, RequestMetrics());
}

void InstanceMgr::remove_instance_resources(const std::string& name) {
  // Caller must hold cluster_mutex_ (cached_channels_ is L1).
  cached_channels_.erase(name);
  std::unique_lock<std::shared_mutex> lock(metrics_mutex_);
  time_predictors_.erase(name);
  request_metrics_.erase(name);
  latency_metrics_.erase(name);
  updated_metrics_.erase(name);
  removed_instance_.insert(name);
  load_metrics_.erase(name);
}

bool InstanceMgr::gather_link_operations(
    const InstanceMetaInfo& info,
    std::vector<std::pair<std::string, InstanceMetaInfo>>* out_ops) {
  out_ops->clear();
  if (info.provider_id == xllm::proto::PROVIDER_ID_VLLM_ASCEND) {
    return true;
  }
  switch (info.type) {
    case InstanceType::DEFAULT:
      break;
    case InstanceType::PREFILL: {
      for (const std::string& decode_name : decode_index_) {
        const InstanceMetaInfo& peer_info = instances_[decode_name];
        if (get_engine_role(peer_info) == xllm::proto::ENGINE_ROLE_DECODE &&
            are_remote_pd_descriptors_compatible(info, peer_info)) {
          out_ops->emplace_back(peer_info.rpc_address, info);
        }
      }
      break;
    }
    case InstanceType::DECODE: {
      for (const std::string& prefill_name : prefill_index_) {
        const InstanceMetaInfo& peer_info = instances_[prefill_name];
        if (get_engine_role(peer_info) == xllm::proto::ENGINE_ROLE_PREFILL &&
            are_remote_pd_descriptors_compatible(peer_info, info)) {
          out_ops->emplace_back(info.rpc_address, peer_info);
        }
      }
      break;
    }
    case InstanceType::MIX: {
      for (const auto& [peer_name, peer_info] : instances_) {
        if (peer_name == info.name ||
            get_engine_role(peer_info) == get_engine_role(info)) {
          continue;
        }
        const bool compatible =
            get_engine_role(info) == xllm::proto::ENGINE_ROLE_PREFILL
                ? are_remote_pd_descriptors_compatible(info, peer_info)
                : are_remote_pd_descriptors_compatible(peer_info, info);
        if (!compatible) {
          continue;
        }
        out_ops->emplace_back(info.rpc_address, peer_info);
      }
      break;
    }
    default:
      LOG(WARNING) << "Unknown InstanceType: " << int(info.type);
      return false;
  }
  return true;
}

bool InstanceMgr::run_link_operations(
    const std::vector<std::pair<std::string, InstanceMetaInfo>>& ops) {
  for (size_t i = 0; i < ops.size(); ++i) {
    if (!call_link_instance(ops[i].first, ops[i].second)) {
      LOG(ERROR) << "Fail to link instance during registration, op index " << i;
      for (size_t j = 0; j < i; ++j) {
        call_unlink_instance(ops[j].first, ops[j].second);
      }
      return false;
    }
  }
  return true;
}

void InstanceMgr::gather_unlink_operations(
    const std::string& name,
    const InstanceMetaInfo& info,
    std::vector<std::pair<std::string, InstanceMetaInfo>>* out_ops) {
  out_ops->clear();
  if (info.type == InstanceType::PREFILL) {
    for (const std::string& decode_name : decode_index_) {
      const InstanceMetaInfo& peer_info = instances_[decode_name];
      if (get_engine_role(peer_info) == xllm::proto::ENGINE_ROLE_DECODE &&
          are_remote_pd_descriptors_compatible(info, peer_info)) {
        out_ops->emplace_back(peer_info.rpc_address, info);
      }
    }
  } else if (info.type == InstanceType::DECODE) {
    for (const std::string& prefill_name : prefill_index_) {
      const InstanceMetaInfo& peer_info = instances_[prefill_name];
      if (get_engine_role(peer_info) == xllm::proto::ENGINE_ROLE_PREFILL &&
          are_remote_pd_descriptors_compatible(peer_info, info)) {
        out_ops->emplace_back(peer_info.rpc_address, info);
      }
    }
  } else if (info.type == InstanceType::MIX) {
    for (const auto& [peer_name, peer_info] : instances_) {
      if (peer_name == name ||
          get_engine_role(peer_info) == get_engine_role(info)) {
        continue;
      }
      const bool compatible =
          get_engine_role(info) == xllm::proto::ENGINE_ROLE_PREFILL
              ? are_remote_pd_descriptors_compatible(info, peer_info)
              : are_remote_pd_descriptors_compatible(peer_info, info);
      if (!compatible) {
        continue;
      }
      out_ops->emplace_back(peer_info.rpc_address, info);
    }
  }
}

void InstanceMgr::add_instance_to_index(const std::string& name,
                                        InstanceMetaInfo& info) {
  switch (info.type) {
    case InstanceType::DEFAULT:
      info.instance_index = prefill_index_.size();
      prefill_index_.emplace_back(name);
      LOG(INFO) << "Register a new default instance, instance name : " << name;
      break;
    case InstanceType::PREFILL:
      info.instance_index = prefill_index_.size();
      prefill_index_.emplace_back(name);
      LOG(INFO) << "Register a new prefill instance, instance name : " << name;
      break;
    case InstanceType::DECODE:
      info.instance_index = decode_index_.size();
      decode_index_.emplace_back(name);
      LOG(INFO) << "Register a new decode instance, instance name : " << name;
      break;
    case InstanceType::MIX:
      if (info.current_type == InstanceType::PREFILL) {
        info.instance_index = prefill_index_.size();
        prefill_index_.emplace_back(name);
        LOG(INFO) << "Register a new prefill instance, instance name : "
                  << name;
      } else {
        info.instance_index = decode_index_.size();
        decode_index_.emplace_back(name);
        LOG(INFO) << "Register a new decode instance, instance name : " << name;
      }
      break;
    default:
      break;
  }
}

void InstanceMgr::remove_instance_from_index(const std::string& name,
                                             const InstanceMetaInfo& info) {
  uint64_t index = info.instance_index;
  if (index == -1) return;

  auto remove_from_vec = [&](std::vector<std::string>& vec) {
    if (index >= vec.size()) return;
    std::swap(vec[index], vec.back());
    instances_[vec[index]].instance_index = index;
    vec.pop_back();
  };

  switch (info.type) {
    case InstanceType::DEFAULT:
    case InstanceType::PREFILL:
      remove_from_vec(prefill_index_);
      break;
    case InstanceType::DECODE:
      remove_from_vec(decode_index_);
      break;
    case InstanceType::MIX:
      if (info.current_type == InstanceType::PREFILL) {
        remove_from_vec(prefill_index_);
      } else {
        remove_from_vec(decode_index_);
      }
      break;
    default:
      break;
  }
}

bool InstanceMgr::has_available_instances() const {
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
  const uint64_t now_monotonic_ms = monotonic_time_ms();
  std::vector<provider::ProviderRouteCandidate> prefill_candidates =
      make_route_candidates(
          instances_, prefill_index_, engine_registry_, now_monotonic_ms);
  const std::vector<provider::ProviderRouteCandidate> decode_candidates =
      make_route_candidates(
          instances_, decode_index_, engine_registry_, now_monotonic_ms);
  apply_link_readiness(&prefill_candidates,
                       decode_candidates,
                       engine_registry_,
                       now_monotonic_ms);
  provider::ProviderRouteSelection selection;
  return provider::ProviderRouteSelector::select(
      prefill_candidates,
      decode_candidates,
      xllm::proto::PROVIDER_ID_UNSPECIFIED,
      0,
      0,
      &selection);
}

}  // namespace xllm_service
