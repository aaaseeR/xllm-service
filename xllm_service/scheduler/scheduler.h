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

#include <atomic>
#include <condition_variable>
#include <thread>

#include "chat_template/chat_template.h"
#include "chat_template/jinja_chat_template.h"
#include "common/call_data.h"
#include "common/generation_delivery_status.h"
#include "common/options.h"
#include "common/threadpool.h"
#include "common/xllm/output.h"
#include "etcd_client/etcd_client.h"
#include "loadbalance_policy/loadbalance_policy.h"
#include "managers/global_kvcache_mgr.h"
#include "managers/instance_mgr.h"
#include "provider/provider_registry.h"
#include "provider/state_stream_outbox.h"
#include "request/client_disconnect_monitor.h"
#include "request/request.h"
#include "request/request_deadline_queue.h"
#include "response_handler.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/tokenizer_args.h"

namespace xllm_service {

// A scheduler for scheduling requests and instances
class Scheduler final {
 public:
  Scheduler(const Options& options);
  ~Scheduler();

  bool schedule(std::shared_ptr<Request> request);

  std::shared_ptr<brpc::Channel> get_channel(const std::string& target_name);

  InstanceMetaInfo get_instance_info(const std::string& instance_name);

  std::vector<std::string> get_static_decode_list(
      const std::string& instance_name);

  std::vector<std::string> get_static_prefill_list(
      const std::string& instance_name);

  bool handle_instance_heartbeat(const proto::HeartbeatRequest* req);

  provider::ContractResult handle_engine_state_batch(
      const xllm::proto::StateBatch& batch,
      bool* applied);

  void exited() { exited_.store(true, std::memory_order_release); }

  // Called by InstanceMgr only after an authoritative strict membership
  // change. It invalidates subscriber baselines but does not publish soft
  // state as Registry truth.
  void notify_engine_registry_membership_changed();

  // Returns true if at least one valid instance group is available.
  bool has_available_instances() const;

  // register new requests from http service
  // keep http callback util request finished.
  // `handle_generation` will handle response with these callbacks.
  bool record_new_request(std::shared_ptr<ChatCallData> call_data,
                          std::shared_ptr<Request> request);
  bool record_new_request(std::shared_ptr<AnthropicCallData> call_data,
                          std::shared_ptr<Request> request);
  bool record_new_request(std::shared_ptr<CompletionCallData> call_data,
                          std::shared_ptr<Request> request);
  void finish_request(const std::string& service_request_id,
                      bool error = false);

  void clear_requests_on_failed_instance(const std::string& instance_name,
                                         const std::string& incarnation_id,
                                         InstanceType type);

  // handle generations from prefill/decode instance
  bool handle_generation(const llm::RequestOutput& request_output);
  GenerationDeliveryResult handle_generation_detailed(
      const llm::RequestOutput& request_output);

  // Async native submission completion is attempt-scoped: a late failure
  // from an old P must never terminate a replacement attempt.
  void handle_attempt_dispatch_failure(const std::string& request_uid,
                                       uint64_t attempt_seq,
                                       std::string message);

  // update request metrics for prefill finished request
  void update_request_metrics(std::shared_ptr<Request> request,
                              bool finished_on_prefill_instance);

  // update token latency metrics
  void update_token_latency_metrics(std::shared_ptr<Request> request,
                                    bool finished_on_prefill_instance);

 private:
  DISALLOW_COPY_AND_ASSIGN(Scheduler);

  void update_master_service_heartbeat();

  void activate_as_master();
  void run_state_stream_publisher();
  bool refresh_state_stream_subscribers();
  void try_apply_local_full_state(uint64_t now_monotonic_ms);

  bool register_current_service();

  void handle_master_service_watch(const etcd::Response& response,
                                   const uint64_t& prefix_len);

  void handle_master_identity_watch(const etcd::Response& response,
                                    const uint64_t& prefix_len);

  void handle_xservice_watch(const etcd::Response& response,
                             const uint64_t& prefix_len);

  Tokenizer* get_tls_tokenizer();

  bool install_execution_hold_locked(const std::shared_ptr<Request>& request);
  bool install_request_safety_guards_locked(
      const std::shared_ptr<Request>& request);
  void rollback_request_safety_guards_locked(
      const std::shared_ptr<Request>& request);
  bool confirm_generation_commit(const std::shared_ptr<Request>& request);
  bool resolve_terminal_execution_hold(const std::shared_ptr<Request>& request);
  bool converge_execution_hold_for_retry_locked(
      const std::shared_ptr<Request>& request);
  bool retry_first_output_attempt_locked(
      const std::shared_ptr<Request>& request,
      std::string* failure_message);
  bool select_retry_instances(const std::shared_ptr<Request>& request);
  bool prepare_v2_execution_plan(const std::shared_ptr<Request>& request);
  void cancel_or_detach_execution_hold_locked(
      const std::shared_ptr<Request>& request);
  void fail_output_dispatch_locked(const std::shared_ptr<Request>& request,
                                   llm::StatusCode status_code,
                                   std::string message);
  bool call_attempt_control(const xllm::proto::ExecutionResourceHold& hold,
                            const xllm::proto::ExecutionHolder& holder,
                            bool query,
                            int32_t timeout_ms);
  void recover_first_output_events(
      const std::vector<std::shared_ptr<Request>>& requests,
      bool fail_if_unavailable);
  void run_execution_hold_cleanup();
  void run_request_watchdog();
  void arm_client_disconnect_notification(
      const std::shared_ptr<Request>& request);

 private:
  Options options_;

  std::string service_incarnation_id_;

  std::unique_ptr<provider::ExecutionHoldCleanupTable>
      execution_hold_cleanup_table_;
  std::unique_ptr<RequestDeadlineQueue> request_deadline_queue_;
  std::shared_ptr<ClientDisconnectMonitor> client_disconnect_monitor_;

  // Serializes the transition between request-owned and detached holds with
  // exact process-termination evidence. Lock order is this mutex, then
  // request_mutex_, then InstanceMgr's internal cluster mutex.
  std::mutex execution_hold_cleanup_mutex_;

  std::mutex execution_hold_cleanup_wait_mutex_;
  std::condition_variable execution_hold_cleanup_cv_;
  bool execution_hold_cleanup_stopped_ = false;
  std::unique_ptr<std::thread> execution_hold_cleanup_thread_;
  std::unique_ptr<std::thread> request_watchdog_thread_;

  std::mutex output_gap_watch_mutex_;
  std::unordered_map<std::string, std::weak_ptr<Request>> output_gap_watchlist_;
  std::unordered_map<std::string, std::weak_ptr<Request>>
      failed_prefill_recovery_watchlist_;

  std::atomic_bool exited_ = false;

  std::atomic_bool is_master_service_ = false;

  TokenizerArgs tokenizer_args_;

  // chat template instance
  std::unique_ptr<ChatTemplate> chat_template_;

  std::shared_ptr<EtcdClient> etcd_client_;

  std::unique_ptr<Tokenizer> tokenizer_;

  std::shared_ptr<InstanceMgr> instance_mgr_;

  std::shared_ptr<GlobalKVCacheMgr> global_kvcache_mgr_;

  std::unique_ptr<LoadBalancePolicy> lb_policy_;

  // Append-only cache of immutable Provider/profile Adapters. Request-scoped
  // data is supplied separately to RequestCodec::encode().
  provider::ProviderAdapterRegistry provider_adapter_registry_;

  std::unique_ptr<provider::StateStreamOutbox> state_stream_outbox_;
  std::atomic<uint64_t> next_state_stream_snapshot_seq_ = 1;
  std::mutex state_stream_wait_mutex_;
  std::condition_variable state_stream_cv_;
  std::unique_ptr<std::thread> state_stream_thread_;

  std::unique_ptr<std::thread> heartbeat_thread_;

  // `service request id` -> `request` map
  std::unordered_map<std::string, std::shared_ptr<Request>> requests_;
  std::mutex request_mutex_;

  // use threadpool to handle all RequestOuputs queue
  static constexpr size_t kOutputTheadNum_ = 128;  // magic num
  ThreadPool output_threadpools_[kOutputTheadNum_];
  // A request will be handled in the same thread to guarantee the token's
  // order.
  std::unordered_map<std::string, size_t> remote_requests_output_thread_map_;
  size_t next_thread_idx = 0;
  std::mutex thread_map_mutex_;

  // used when receive token from decode instance.
  ResponseHandler response_handler_;
};

}  // namespace xllm_service
