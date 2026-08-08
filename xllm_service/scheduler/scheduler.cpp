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

#include "scheduler/scheduler.h"

#include <brpc/callback.h>
#include <brpc/controller.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <unordered_set>

#include "chat_template/deepseek_v4_cpp_chat_template.h"
#include "chat_template/model_type.h"
#include "common/metrics.h"
#include "common/utils.h"
#include "common/xllm/status.h"
#include "common/xllm/uuid.h"
#include "disagg_pd.pb.h"
#include "http_service/anthropic_adapter.h"
#include "http_service/anthropic_stream_encoder.h"
#include "loadbalance_policy/cache_aware_routing.h"
#include "loadbalance_policy/round_robin.h"
#include "loadbalance_policy/slo_aware_policy.h"
#include "rpc_service/first_event_recovery_client.h"
#include "scheduler/xllm_chat_parse_bridge.h"
#include "tokenizer/tokenizer_factory.h"

namespace {
constexpr int32_t kHeartbeatInterval = 3;  // in seconds
constexpr int32_t kRegistrationMaxRetries = 5;

constexpr const char* kEtcdUsernameEnvVar = "ETCD_USERNAME";
constexpr const char* kEtcdPasswordEnvVar = "ETCD_PASSWORD";

xllm_service::provider::ExecutionHoldCleanupTable::Config execution_hold_config(
    const xllm_service::Options& options) {
  return xllm_service::provider::ExecutionHoldCleanupTable::Config{
      .record_capacity = options.execution_hold_cleanup_record_capacity(),
      .byte_capacity = options.execution_hold_cleanup_byte_capacity(),
      .max_cleanup_record_bytes =
          options.execution_hold_max_cleanup_record_bytes(),
      .max_potential_holders = options.execution_hold_max_potential_holders(),
      .max_identifier_bytes = options.execution_hold_max_identifier_bytes(),
      .allow_hard_time_bound_proof = false,
  };
}

xllm::proto::ExecutionAttemptId execution_attempt(
    const xllm_service::Request& request) {
  xllm::proto::ExecutionAttemptId attempt;
  attempt.set_request_uid(request.correlation.request_uid());
  if (request.correlation.has_attempt_seq()) {
    attempt.set_attempt_seq(request.correlation.attempt_seq());
  }
  return attempt;
}

xllm::proto::ExecutionHolder decode_holder(
    const xllm_service::Request& request) {
  xllm::proto::ExecutionHolder holder;
  holder.set_engine_uid(request.routing.decode_name);
  holder.set_incarnation_id(request.decode_incarnation_id);
  return holder;
}

xllm::proto::ExecutionHolder prefill_holder(
    const xllm_service::Request& request) {
  xllm::proto::ExecutionHolder holder;
  holder.set_engine_uid(request.routing.prefill_name);
  holder.set_incarnation_id(request.prefill_incarnation_id);
  return holder;
}

bool same_attempt_key(const xllm::proto::RequestAttemptKey& key,
                      const xllm::proto::ExecutionAttemptId& attempt,
                      const xllm::proto::ExecutionHolder& holder) {
  return key.request_uid() == attempt.request_uid() && key.has_attempt_seq() &&
         attempt.has_attempt_seq() &&
         key.attempt_seq() == attempt.attempt_seq() &&
         key.incarnation_id() == holder.incarnation_id();
}

bool terminal_attempt_state(xllm::proto::AttemptLifecycleState state) {
  return state == xllm::proto::ATTEMPT_LIFECYCLE_STATE_DONE ||
         state == xllm::proto::ATTEMPT_LIFECYCLE_STATE_CANCELLED ||
         state == xllm::proto::ATTEMPT_LIFECYCLE_STATE_EXPIRED ||
         state == xllm::proto::ATTEMPT_LIFECYCLE_STATE_FAILED ||
         state == xllm::proto::ATTEMPT_LIFECYCLE_STATE_CANCELLED_BEFORE_CREATE;
}

const char* retry_decision_name(
    xllm_service::FirstOutputRetryDecision decision) {
  switch (decision) {
    case xllm_service::FirstOutputRetryDecision::ALLOWED:
      return "ALLOWED";
    case xllm_service::FirstOutputRetryDecision::FIRST_OUTPUT_EMITTED:
      return "FIRST_OUTPUT_EMITTED";
    case xllm_service::FirstOutputRetryDecision::ATTEMPT_BUDGET_EXHAUSTED:
      return "ATTEMPT_BUDGET_EXHAUSTED";
    case xllm_service::FirstOutputRetryDecision::DEADLINE_BUDGET_INSUFFICIENT:
      return "DEADLINE_BUDGET_INSUFFICIENT";
    case xllm_service::FirstOutputRetryDecision::DEVICE_TIME_BUDGET_EXHAUSTED:
      return "DEVICE_TIME_BUDGET_EXHAUSTED";
    case xllm_service::FirstOutputRetryDecision::CLOCK_REGRESSION:
      return "CLOCK_REGRESSION";
  }
  return "UNKNOWN";
}

void signal_client_disconnect(
    std::shared_ptr<xllm_service::ClientDisconnectMonitor> monitor,
    std::string request_uid) {
  monitor->notify_disconnected(request_uid);
}

}  // namespace

namespace xllm_service {

Scheduler::Scheduler(const Options& options)
    : options_(options),
      service_incarnation_id_(llm::new_uuid_v7()),
      execution_hold_cleanup_table_(
          std::make_unique<provider::ExecutionHoldCleanupTable>(
              execution_hold_config(options))),
      request_deadline_queue_(std::make_unique<RequestDeadlineQueue>(
          options.request_deadline_capacity())),
      client_disconnect_monitor_(std::make_shared<ClientDisconnectMonitor>(
          options.request_deadline_capacity())) {
  const std::optional<xllm::FirstEventRetryPolicy> first_event_retry_policy =
      xllm::FirstEventRetryPolicy::from_durations_ms(
          static_cast<uint64_t>(options_.p_first_event_retry_ub_ms()),
          static_cast<uint64_t>(options_.first_event_dispatch_margin_ms()));
  if (options_.execution_hold_cleanup_retry_interval_ms() <= 0 ||
      options_.execution_hold_cleanup_retry_batch_size() == 0 ||
      options_.execution_hold_cleanup_rpc_timeout_ms() <= 0) {
    LOG(FATAL) << "Invalid execution hold cleanup retry configuration.";
  }
  if (options_.output_reorder_max_events() == 0 ||
      options_.output_reorder_max_bytes() == 0) {
    LOG(FATAL) << "Invalid output reorder capacity configuration.";
  }
  if (options_.request_watchdog_interval_ms() <= 0 ||
      options_.output_gap_timeout_ms() <= 0 ||
      options_.output_gap_query_timeout_ms() <= 0 ||
      options_.output_gap_query_timeout_ms() >
          options_.output_gap_timeout_ms() ||
      options_.output_gap_query_batch_size() == 0 ||
      !first_event_retry_policy.has_value() ||
      !first_event_retry_policy->fits_within_gap_timeout_ms(
          static_cast<uint64_t>(options_.output_gap_timeout_ms())) ||
      options_.request_deadline_capacity() == 0 ||
      options_.request_watchdog_batch_size() == 0 ||
      options_.default_request_deadline_ms() <= 0 ||
      options_.request_watchdog_interval_ms() >
          options_.output_gap_timeout_ms()) {
    LOG(FATAL) << "Invalid request watchdog configuration.";
  }
  if (options_.max_first_output_attempt_retries() > 0 &&
      (options_.max_nonstream_retry_wasted_device_ms() == 0 ||
       options_.min_first_output_retry_remaining_ms() == 0)) {
    LOG(FATAL) << "Invalid first-output retry budget configuration.";
  }
  // vLLM-backend clusters forward the raw client JSON to vLLM, which does its
  // own tokenization / chat templating. Skip building the local tokenizer and
  // chat template so the master does not require a tokenizer.model / template.
  if (options_.default_backend_type() != "vllm") {
    tokenizer_ = TokenizerFactory::create_tokenizer(options_.tokenizer_path(),
                                                    &tokenizer_args_);

    // Select the chat template by config.json model_type (jinja by default).
    const auto model_type = load_model_type(options_.tokenizer_path());
    switch (select_chat_template_kind(model_type)) {
      case ChatTemplateKind::kDeepseekV4Cpp:
        chat_template_ =
            std::make_unique<DeepseekV4CppChatTemplate>(tokenizer_args_);
        LOG(INFO) << "Selected DeepseekV4CppChatTemplate (model_type="
                  << model_type.value_or("") << ").";
        break;
      case ChatTemplateKind::kJinja:
        chat_template_ = std::make_unique<JinjaChatTemplate>(tokenizer_args_);
        LOG(INFO) << "Selected JinjaChatTemplate (model_type="
                  << model_type.value_or("") << ").";
        break;
    }
  }

  const std::string etcd_username =
      utils::get_optional_string_env(kEtcdUsernameEnvVar).value_or("");
  const std::string etcd_password =
      utils::get_optional_string_env(kEtcdPasswordEnvVar).value_or("");
  const bool has_etcd_auth_user = !etcd_username.empty();
  const bool has_etcd_auth_password = !etcd_password.empty();
  if (has_etcd_auth_user != has_etcd_auth_password) {
    LOG(FATAL) << "Both " << kEtcdUsernameEnvVar << " and "
               << kEtcdPasswordEnvVar << " must be set together.";
  }
  if (has_etcd_auth_user) {
    etcd_client_ = std::make_shared<EtcdClient>(options_.etcd_addr(),
                                                etcd_username,
                                                etcd_password,
                                                options_.etcd_namespace());
  } else {
    etcd_client_ = std::make_shared<EtcdClient>(options_.etcd_addr(),
                                                options_.etcd_namespace());
  }

  if (!register_current_service()) {
    LOG(FATAL)
        << "Failed to register current xllm_service in etcd, service_name: "
        << options_.service_name();
  }

  auto handle_xservice = std::bind(&Scheduler::handle_xservice_watch,
                                   this,
                                   std::placeholders::_1,
                                   std::placeholders::_2);
  etcd_client_->add_watch(ETCD_XSERVICE_KEY_PREFIX, handle_xservice);

  if (!etcd_client_->get(ETCD_MASTER_SERVICE_KEY, nullptr)) {
    is_master_service_ = etcd_client_->set(
        ETCD_MASTER_SERVICE_KEY, options_.service_name(), kHeartbeatInterval);
    LOG(INFO) << "Set current service as master!";
  }

  instance_mgr_ = std::make_shared<InstanceMgr>(
      options, etcd_client_, is_master_service_, this);

  global_kvcache_mgr_ = std::make_shared<GlobalKVCacheMgr>(
      options, etcd_client_, is_master_service_);

  if (options.load_balance_policy() == "CAR") {
    lb_policy_ =
        std::make_unique<CacheAwareRouting>(instance_mgr_, global_kvcache_mgr_);
  } else if (options.load_balance_policy() == "SLO_AWARE") {
    lb_policy_ = std::make_unique<SloAwarePolicy>(options, instance_mgr_);
  } else {
    lb_policy_ = std::make_unique<RoundRobin>(instance_mgr_);
  }

  if (is_master_service_) {
    heartbeat_thread_ = std::make_unique<std::thread>(
        &Scheduler::update_master_service_heartbeat, this);
  } else {
    auto handle_master = std::bind(&Scheduler::handle_master_service_watch,
                                   this,
                                   std::placeholders::_1,
                                   std::placeholders::_2);
    etcd_client_->add_watch(ETCD_MASTER_SERVICE_KEY, handle_master);
  }

  execution_hold_cleanup_thread_ = std::make_unique<std::thread>(
      &Scheduler::run_execution_hold_cleanup, this);
  request_watchdog_thread_ =
      std::make_unique<std::thread>(&Scheduler::run_request_watchdog, this);
}

Scheduler::~Scheduler() {
  client_disconnect_monitor_->close();
  {
    std::lock_guard<std::mutex> lock(execution_hold_cleanup_wait_mutex_);
    execution_hold_cleanup_stopped_ = true;
  }
  execution_hold_cleanup_cv_.notify_all();
  if (execution_hold_cleanup_thread_ != nullptr &&
      execution_hold_cleanup_thread_->joinable()) {
    execution_hold_cleanup_thread_->join();
  }
  if (request_watchdog_thread_ != nullptr &&
      request_watchdog_thread_->joinable()) {
    request_watchdog_thread_->join();
  }
  etcd_client_->stop_watch();
}

bool Scheduler::schedule(std::shared_ptr<Request> request) {
  if (request->request_deadline_present &&
      (!request->request_deadline.has_value() ||
       request->request_deadline->expired())) {
    LOG(ERROR) << "Request deadline is invalid or already expired.";
    return false;
  }
  // For vLLM backend clusters we forward the client's raw OpenAI JSON straight
  // through to vLLM, which applies its own chat template and tokenization. So
  // skip the local chat-template + tokenize steps entirely (leaving token_ids
  // empty) and only run instance selection. See default_backend_type option.
  const bool is_vllm = (options_.default_backend_type() == "vllm");

  // apply chat template
  if (!is_vllm && request->messages.size() > 0) {
    if (chat_template_ == nullptr) {
      LOG(ERROR) << "Chat template has not configured.";
      return false;
    }

    const std::vector<JsonTool> empty_tools;
    const std::vector<JsonTool>& tools_for_template =
        request->tool_choice == "none" ? empty_tools : request->tools;
    auto prompt = chat_template_->apply(
        request->messages, tools_for_template, request->chat_template_kwargs);
    if (!prompt.has_value()) {
      LOG(ERROR) << "Failed to construct prompt from messages";
      return false;
    }
    request->prompt = prompt.value();
  }

  // encode prompt
  if (!is_vllm && request->prompt.size() != 0) {
    if (chat_template_ == nullptr) {
      LOG(ERROR) << "Chat template has not configured.";
      return false;
    }
    if (!get_tls_tokenizer()->encode(
            request->prompt,
            &request->token_ids,
            chat_template_->encode_add_special_tokens())) {
      LOG(ERROR) << "Encode prompt failed: " << request->prompt;
      return false;
    }
  }

  auto ret = lb_policy_->select_instances_pair(request);
  if (!ret) {
    return false;
  }

  if (!instance_mgr_->bind_request_instance_incarnations(request)) {
    LOG(ERROR) << "Failed to bind request to instance incarnation ids. "
               << request->routing.debug_string();
    return false;
  }
  DLOG(INFO) << request->routing.debug_string();

  // update request metrics
  if (!is_vllm && request->prompt.size() != 0) {
    instance_mgr_->update_request_metrics(request, RequestAction::SCHEDULE);
  }

  return true;
}

std::shared_ptr<brpc::Channel> Scheduler::get_channel(
    const std::string& target_name) {
  return instance_mgr_->get_channel(target_name);
}

void Scheduler::update_master_service_heartbeat() {
  while (!exited_) {
    std::this_thread::sleep_for(std::chrono::seconds(kHeartbeatInterval));

    global_kvcache_mgr_->upload_kvcache();

    instance_mgr_->upload_load_metrics();
  }
}

bool Scheduler::register_current_service() {
  const std::string service_key =
      ETCD_XSERVICE_KEY_PREFIX + options_.service_name();

  if (etcd_client_->set(
          service_key, options_.service_name(), kHeartbeatInterval)) {
    return true;
  }

  // Key already exists — likely a stale lease from a previous process.
  // Since the lease TTL is short (3 seconds), we can safely wait for it
  // to expire naturally and retry registration without deleting the key.
  LOG(WARNING) << "Service key already exists: " << service_key
               << ". Waiting for the stale lease to expire and retrying.";

  for (int attempt = 1; attempt <= kRegistrationMaxRetries; ++attempt) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    if (etcd_client_->set(
            service_key, options_.service_name(), kHeartbeatInterval)) {
      LOG(INFO) << "Service registered successfully on retry " << attempt;
      return true;
    }
    LOG(WARNING) << "Registration retry " << attempt << " failed, "
                 << "waiting for old lease to expire...";
  }

  LOG(ERROR) << "Registration failed after " << kRegistrationMaxRetries
             << " retries: " << service_key;
  return false;
}

bool Scheduler::handle_instance_heartbeat(const proto::HeartbeatRequest* req) {
  if (exited_) {
    return false;
  }
  if (!instance_mgr_->record_instance_heartbeat(req->name(),
                                                req->incarnation_id())) {
    return false;
  }
  global_kvcache_mgr_->record_updated_kvcaches(req->name(), req->cache_event());
  instance_mgr_->record_load_metrics_update(req->name(), req->load_metrics());
  instance_mgr_->update_latency_metrics(req->name(), req->latency_metrics());
  return true;
}

void Scheduler::handle_master_service_watch(const etcd::Response& response,
                                            const uint64_t& prefix_len) {
  if (exited_ || response.events().empty()) {
    return;
  }

  if (etcd_client_->set(ETCD_MASTER_SERVICE_KEY,
                        options_.service_name(),
                        kHeartbeatInterval)) {
    is_master_service_ = true;

    heartbeat_thread_ = std::make_unique<std::thread>(
        &Scheduler::update_master_service_heartbeat, this);

    global_kvcache_mgr_->set_as_master();
    instance_mgr_->set_as_master();
  }
}

void Scheduler::handle_xservice_watch(const etcd::Response& response,
                                      const uint64_t& prefix_len) {
  if (exited_ || response.events().empty()) {
    return;
  }

  for (const auto& event : response.events()) {
    if (event.event_type() != etcd::Event::EventType::DELETE_) {
      continue;
    }

    std::string deleted_service;
    if (event.has_prev_kv()) {
      deleted_service = event.prev_kv().key().substr(prefix_len);
    } else if (event.has_kv()) {
      deleted_service = event.kv().key().substr(prefix_len);
    }

    if (deleted_service.empty()) {
      continue;
    }

    if (deleted_service == options_.service_name()) {
      LOG(INFO) << "Current xllm_service registration expired, re-registering";
      register_current_service();
      continue;
    }

    if (deleted_service == ETCD_MASTER_SERVICE_NAME) {
      continue;
    }

    if (!is_master_service_) {
      continue;
    }

    LOG(INFO) << "Detected xllm_service offline: " << deleted_service;
  }
}

InstanceMetaInfo Scheduler::get_instance_info(
    const std::string& instance_name) {
  return instance_mgr_->get_instance_info(instance_name);
}

std::vector<std::string> Scheduler::get_static_decode_list(
    const std::string& instance_name) {
  return instance_mgr_->get_static_decode_list(instance_name);
}

std::vector<std::string> Scheduler::get_static_prefill_list(
    const std::string& instance_name) {
  return instance_mgr_->get_static_prefill_list(instance_name);
}

Tokenizer* Scheduler::get_tls_tokenizer() {
  thread_local std::unique_ptr<Tokenizer> tls_tokenizer(tokenizer_->clone());
  return tls_tokenizer.get();
}

bool Scheduler::install_execution_hold_locked(
    const std::shared_ptr<Request>& request) {
  if (!instance_mgr_->validate_request_instance_incarnations(request)) {
    LOG(ERROR) << "Refuse dispatch to a stale instance incarnation. "
               << request->routing.debug_string();
    return false;
  }
  if (request->routing.decode_name.empty()) {
    return true;
  }
  const InstanceMetaInfo prefill_info =
      instance_mgr_->get_instance_info(request->routing.prefill_name);
  if (prefill_info.backend_type == "vllm") {
    // vLLM AGGREGATED holds require the Provider Agent submit identity and are
    // installed by that path when it is opened.
    return true;
  }

  const xllm::proto::ExecutionHolder holder = decode_holder(*request);
  const provider::ExecutionHoldStatus status =
      execution_hold_cleanup_table_->install_request_hold(
          &request->execution_hold,
          xllm::proto::EXECUTION_HOLD_KIND_REMOTE_D_RESERVATION,
          execution_attempt(*request),
          service_incarnation_id_,
          {holder});
  if (status == provider::ExecutionHoldStatus::kOk) {
    request->output_event_sequencer =
        std::make_unique<OutputEventSequencer>(OutputEventSequencer::Config{
            .max_buffered_events = options_.output_reorder_max_events(),
            .max_buffered_bytes = options_.output_reorder_max_bytes(),
            .max_sequence_gap = options_.output_reorder_max_events()});
    return true;
  }
  LOG(ERROR) << "Failed to install execution hold before dispatch, request_uid="
             << request->correlation.request_uid()
             << ", status=" << static_cast<int>(status);
  return false;
}

bool Scheduler::install_request_safety_guards_locked(
    const std::shared_ptr<Request>& request) {
  const std::string& request_uid = request->correlation.request_uid();
  if (client_disconnect_monitor_->register_request(request_uid, request) !=
      ClientDisconnectMonitorStatus::OK) {
    LOG(ERROR) << "Failed to reserve client disconnect monitor capacity, "
                  "request_uid="
               << request_uid;
    return false;
  }
  if (!install_execution_hold_locked(request)) {
    client_disconnect_monitor_->erase(request_uid);
    return false;
  }
  return true;
}

void Scheduler::rollback_request_safety_guards_locked(
    const std::shared_ptr<Request>& request) {
  request->execution_hold.abandon_before_dispatch();
  client_disconnect_monitor_->erase(request->correlation.request_uid());
}

bool Scheduler::confirm_generation_commit(
    const std::shared_ptr<Request>& request) {
  if (request->routing.decode_name.empty()) {
    return true;
  }
  const provider::ExecutionHoldStatus status =
      request->execution_hold.confirm_holder(
          decode_holder(*request),
          xllm::proto::EXECUTION_HOLD_PROOF_GENERATION_COMMITTED);
  if (status == provider::ExecutionHoldStatus::kOk) {
    return true;
  }
  LOG(ERROR) << "GenerationCommit output did not match the execution hold, "
                "request_uid="
             << request->correlation.request_uid()
             << ", status=" << static_cast<int>(status);
  return false;
}

bool Scheduler::resolve_terminal_execution_hold(
    const std::shared_ptr<Request>& request) {
  if (request->routing.decode_name.empty()) {
    return true;
  }
  const provider::ExecutionHoldStatus status =
      request->execution_hold.apply_convergence_proof(
          execution_attempt(*request),
          decode_holder(*request),
          xllm::proto::HOLDER_CONVERGENCE_PROOF_TERMINAL_OUTCOME);
  if (status == provider::ExecutionHoldStatus::kResolved) {
    return true;
  }
  LOG(ERROR) << "Terminal Decode output did not resolve the execution hold, "
                "request_uid="
             << request->correlation.request_uid()
             << ", status=" << static_cast<int>(status);
  return false;
}

bool Scheduler::converge_execution_hold_for_retry_locked(
    const std::shared_ptr<Request>& request) {
  const std::optional<xllm::proto::ExecutionResourceHold> hold =
      request->execution_hold.snapshot();
  if (!hold.has_value()) {
    return true;
  }

  // Ordinary P submission is not an execution hold, but it still receives a
  // best-effort attempt-scoped Cancel before replacement.
  const xllm::proto::ExecutionHolder prefill = prefill_holder(*request);
  if (!prefill.engine_uid().empty() && !prefill.incarnation_id().empty()) {
    call_attempt_control(*hold,
                         prefill,
                         /*query=*/false,
                         options_.instance_delete_probe_timeout_ms());
  }

  for (const xllm::proto::ExecutionHolder& holder : hold->potential_holders()) {
    if (!call_attempt_control(*hold,
                              holder,
                              /*query=*/false,
                              options_.instance_delete_probe_timeout_ms())) {
      return false;
    }
    request->execution_hold.apply_convergence_proof(
        hold->attempt(),
        holder,
        xllm::proto::HOLDER_CONVERGENCE_PROOF_CANCEL_FENCE_ACK);
  }
  return !request->execution_hold.has_hold();
}

bool Scheduler::select_retry_instances(
    const std::shared_ptr<Request>& request) {
  request->routing = Routing();
  request->prefill_incarnation_id.clear();
  request->decode_incarnation_id.clear();
  if (!lb_policy_->select_instances_pair(request) ||
      !instance_mgr_->bind_request_instance_incarnations(request)) {
    return false;
  }
  const InstanceMetaInfo prefill_info =
      instance_mgr_->get_instance_info(request->routing.prefill_name);
  if (prefill_info.backend_type == "vllm" ||
      request->routing.decode_name.empty()) {
    LOG(ERROR) << "First-output retry selected an incompatible execution "
                  "mode, routing="
               << request->routing.debug_string();
    return false;
  }
  return true;
}

bool Scheduler::retry_first_output_attempt_locked(
    const std::shared_ptr<Request>& request,
    std::string* failure_message) {
  if (request->first_output_retry_budget == nullptr ||
      request->retry_dispatch_callback == nullptr ||
      !request->correlation.has_attempt_seq() ||
      !request->request_deadline.has_value()) {
    *failure_message = "First-output retry context is unavailable";
    return false;
  }

  const FirstOutputRetryBudget::TimePoint now =
      FirstOutputRetryBudget::Clock::now();
  const uint64_t remaining_deadline_ms =
      request->request_deadline->remaining_ms();
  const FirstOutputRetryDecision decision =
      request->first_output_retry_budget->evaluate(
          request->first_token_emitted.load(std::memory_order_acquire),
          remaining_deadline_ms,
          now);
  if (decision != FirstOutputRetryDecision::ALLOWED) {
    *failure_message = std::string("First-output retry denied: ") +
                       retry_decision_name(decision);
    return false;
  }
  if (request->correlation.attempt_seq() ==
      std::numeric_limits<uint64_t>::max()) {
    *failure_message = "First-output retry denied: ATTEMPT_SEQUENCE_EXHAUSTED";
    return false;
  }
  std::lock_guard<std::mutex> cleanup_guard(execution_hold_cleanup_mutex_);
  if (!converge_execution_hold_for_retry_locked(request)) {
    *failure_message = "First-output retry denied: OLD_EXECUTION_NOT_CONVERGED";
    return false;
  }
  if (!request->first_output_retry_budget->commit_retry(now)) {
    *failure_message = "First-output retry budget changed before commit";
    return false;
  }

  instance_mgr_->update_request_metrics(request, RequestAction::CANCEL);
  if (!select_retry_instances(request)) {
    *failure_message = "First-output retry could not select a native P/D pair";
    return false;
  }
  request->correlation.set_attempt_seq(request->correlation.attempt_seq() + 1);
  request->prefill_stage_finished.store(false, std::memory_order_release);
  request->latest_generate_time = absl::Now();
  request->output_event_sequencer.reset();
  if (!install_execution_hold_locked(request)) {
    *failure_message = "First-output retry could not install execution hold";
    return false;
  }
  instance_mgr_->update_request_metrics(request, RequestAction::SCHEDULE);

  {
    std::lock_guard<std::mutex> watch_guard(output_gap_watch_mutex_);
    output_gap_watchlist_.erase(request->correlation.request_uid());
    failed_prefill_recovery_watchlist_.erase(
        request->correlation.request_uid());
  }
  LOG(INFO) << "Prepared bounded first-output retry, request_uid="
            << request->correlation.request_uid()
            << ", attempt_seq=" << request->correlation.attempt_seq()
            << ", retries=" << request->first_output_retry_budget->retries()
            << ", retry_wasted_device_ms="
            << request->first_output_retry_budget->wasted_device_ms();
  return true;
}

void Scheduler::fail_output_dispatch_locked(
    const std::shared_ptr<Request>& request,
    llm::StatusCode status_code,
    std::string message) {
  if (request->output_dispatch_closed) {
    return;
  }
  request->output_dispatch_closed = true;
  const std::string request_uid = request->correlation.request_uid();
  if (request->output_event_sequencer != nullptr) {
    request->output_event_sequencer->close();
  }
  {
    std::lock_guard<std::mutex> watch_guard(output_gap_watch_mutex_);
    output_gap_watchlist_.erase(request_uid);
    failed_prefill_recovery_watchlist_.erase(request_uid);
  }

  size_t output_thread_index = 0;
  bool output_thread_found = false;
  {
    std::lock_guard<std::mutex> thread_guard(thread_map_mutex_);
    const auto it = remote_requests_output_thread_map_.find(request_uid);
    if (it != remote_requests_output_thread_map_.end()) {
      output_thread_index = it->second;
      output_thread_found = true;
    }
  }

  OutputCallback callback = request->output_callback;
  auto fail = [this,
               request,
               request_uid,
               callback = std::move(callback),
               status_code,
               message = std::move(message)]() mutable {
    if (!request->client_disconnected.load(std::memory_order_acquire) &&
        !request->call_data->is_disconnected()) {
      llm::RequestOutput error_output;
      error_output.service_request_id = request_uid;
      error_output.status = llm::Status(status_code, std::move(message));
      callback(std::move(error_output));
    }
    finish_request(request_uid, true);
  };
  if (output_thread_found) {
    output_threadpools_[output_thread_index].schedule(std::move(fail));
  } else {
    fail();
  }
}

void Scheduler::cancel_or_detach_execution_hold_locked(
    const std::shared_ptr<Request>& request) {
  std::optional<xllm::proto::ExecutionResourceHold> hold =
      request->execution_hold.snapshot();
  if (!hold.has_value()) {
    return;
  }

  for (const xllm::proto::ExecutionHolder& holder : hold->potential_holders()) {
    if (!call_attempt_control(*hold,
                              holder,
                              /*query=*/false,
                              options_.instance_delete_probe_timeout_ms())) {
      continue;
    }
    request->execution_hold.apply_convergence_proof(
        hold->attempt(),
        holder,
        xllm::proto::HOLDER_CONVERGENCE_PROOF_CANCEL_FENCE_ACK);
  }

  if (!request->execution_hold.has_hold()) {
    return;
  }
  const provider::ExecutionHoldStatus status =
      execution_hold_cleanup_table_->adopt(&request->execution_hold);
  if (status != provider::ExecutionHoldStatus::kOk) {
    LOG(ERROR) << "Failed to detach unresolved execution hold, request_uid="
               << request->correlation.request_uid()
               << ", status=" << static_cast<int>(status);
  }
}

bool Scheduler::call_attempt_control(
    const xllm::proto::ExecutionResourceHold& hold,
    const xllm::proto::ExecutionHolder& holder,
    bool query,
    int32_t timeout_ms) {
  const std::shared_ptr<brpc::Channel> channel =
      instance_mgr_->get_channel(holder.engine_uid());
  if (channel == nullptr) {
    return false;
  }

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
  if (query) {
    stub.QueryRequest(&controller, &request, &response, nullptr);
  } else {
    stub.CancelRequest(&controller, &request, &response, nullptr);
  }
  return !controller.Failed() && response.ok() &&
         same_attempt_key(response.status().key(), hold.attempt(), holder) &&
         terminal_attempt_state(response.status().state());
}

void Scheduler::recover_first_output_events(
    const std::vector<std::shared_ptr<Request>>& requests,
    bool fail_if_unavailable) {
  std::vector<std::shared_ptr<Request>> active_requests;
  active_requests.reserve(requests.size());
  std::vector<FirstEventRecoveryQuery> queries;
  queries.reserve(requests.size());
  for (const std::shared_ptr<Request>& service_request : requests) {
    {
      std::lock_guard<std::mutex> output_guard(
          service_request->output_dispatch_mutex);
      if (service_request->output_dispatch_closed ||
          service_request->output_event_sequencer == nullptr ||
          service_request->output_event_sequencer->next_expected_seq() != 0) {
        continue;
      }
      std::lock_guard<std::mutex> request_guard(request_mutex_);
      const auto it =
          requests_.find(service_request->correlation.request_uid());
      if (it == requests_.end() || it->second != service_request) {
        continue;
      }
    }
    uint64_t timeout_ms =
        static_cast<uint64_t>(options_.output_gap_query_timeout_ms());
    if (service_request->request_deadline.has_value()) {
      timeout_ms = std::min(timeout_ms,
                            service_request->request_deadline->remaining_ms());
    }
    active_requests.emplace_back(service_request);
    queries.emplace_back(FirstEventRecoveryQuery{
        .channel =
            instance_mgr_->get_channel(service_request->routing.decode_name),
        .attempt = execution_attempt(*service_request),
        .decode = decode_holder(*service_request),
        .prefill = prefill_holder(*service_request),
        .max_payload_bytes = options_.output_reorder_max_bytes(),
        .timeout_ms = timeout_ms,
    });
  }

  std::vector<FirstEventRecoveryResult> results =
      query_first_output_events(queries);
  CHECK_EQ(results.size(), active_requests.size());
  for (size_t index = 0; index < active_requests.size(); ++index) {
    const FirstEventRecoveryQuery& query = queries[index];
    const FirstEventRecoveryResult& result = results[index];
    bool recovered = false;
    if (result.status.ok() && result.output.has_value()) {
      recovered = handle_generation(*result.output);
    } else {
      LOG(ERROR) << "Failed to recover first event, request_uid="
                 << query.attempt.request_uid()
                 << ", error=" << result.status.message();
    }
    if (recovered) {
      continue;
    }

    const std::shared_ptr<Request>& service_request = active_requests[index];
    const std::string request_uid = service_request->correlation.request_uid();
    bool retry_prepared = false;
    uint64_t retry_attempt_seq = 0;
    {
      std::lock_guard<std::mutex> output_guard(
          service_request->output_dispatch_mutex);
      {
        std::lock_guard<std::mutex> request_guard(request_mutex_);
        const auto it = requests_.find(request_uid);
        if (it == requests_.end() || it->second != service_request) {
          continue;
        }
      }
      if (service_request->output_dispatch_closed ||
          service_request->output_event_sequencer == nullptr ||
          service_request->output_event_sequencer->next_expected_seq() != 0) {
        continue;
      }
      if (!fail_if_unavailable &&
          !service_request->output_event_sequencer->gap_expired(
              OutputEventSequencer::Clock::now(),
              std::chrono::milliseconds(options_.output_gap_timeout_ms()))) {
        continue;
      }
      std::string retry_failure;
      retry_prepared =
          retry_first_output_attempt_locked(service_request, &retry_failure);
      if (retry_prepared) {
        retry_attempt_seq = service_request->correlation.attempt_seq();
      } else {
        fail_output_dispatch_locked(
            service_request,
            fail_if_unavailable ? llm::StatusCode::CANCELLED
                                : llm::StatusCode::DEADLINE_EXCEEDED,
            (fail_if_unavailable
                 ? "Prefill process failed before seq=0 could be recovered: "
                 : "Output seq=0 recovery failed after the local gap "
                   "timeout: ") +
                retry_failure);
      }
    }
    if (!retry_prepared) {
      continue;
    }
    if (service_request->retry_dispatch_callback(*service_request)) {
      continue;
    }

    std::lock_guard<std::mutex> output_guard(
        service_request->output_dispatch_mutex);
    {
      std::lock_guard<std::mutex> request_guard(request_mutex_);
      const auto it = requests_.find(request_uid);
      if (it == requests_.end() || it->second != service_request ||
          !service_request->correlation.has_attempt_seq() ||
          service_request->correlation.attempt_seq() != retry_attempt_seq) {
        continue;
      }
    }
    fail_output_dispatch_locked(
        service_request,
        llm::StatusCode::CANCELLED,
        "First-output retry dispatch could not be started");
  }
}

void Scheduler::run_execution_hold_cleanup() {
  std::unique_lock<std::mutex> wait_lock(execution_hold_cleanup_wait_mutex_);
  while (!execution_hold_cleanup_stopped_) {
    const bool stopped = execution_hold_cleanup_cv_.wait_for(
        wait_lock,
        std::chrono::milliseconds(
            options_.execution_hold_cleanup_retry_interval_ms()),
        [this] { return execution_hold_cleanup_stopped_; });
    if (stopped) {
      break;
    }
    wait_lock.unlock();

    const std::vector<xllm::proto::ExecutionResourceHold> batch =
        execution_hold_cleanup_table_->next_retry_batch(
            options_.execution_hold_cleanup_retry_batch_size());
    for (const xllm::proto::ExecutionResourceHold& hold : batch) {
      for (const xllm::proto::ExecutionHolder& holder :
           hold.potential_holders()) {
        xllm::proto::HolderConvergenceProof proof =
            xllm::proto::HOLDER_CONVERGENCE_PROOF_UNSPECIFIED;
        if (call_attempt_control(
                hold,
                holder,
                /*query=*/true,
                options_.execution_hold_cleanup_rpc_timeout_ms())) {
          proof = xllm::proto::HOLDER_CONVERGENCE_PROOF_TERMINAL_OUTCOME;
        } else if (call_attempt_control(
                       hold,
                       holder,
                       /*query=*/false,
                       options_.execution_hold_cleanup_rpc_timeout_ms())) {
          proof = xllm::proto::HOLDER_CONVERGENCE_PROOF_CANCEL_FENCE_ACK;
        }
        if (proof != xllm::proto::HOLDER_CONVERGENCE_PROOF_UNSPECIFIED) {
          execution_hold_cleanup_table_->apply_convergence_proof(
              hold.attempt(), holder, proof);
        }
      }
    }

    wait_lock.lock();
  }
}

void Scheduler::run_request_watchdog() {
  std::unique_lock<std::mutex> wait_lock(execution_hold_cleanup_wait_mutex_);
  bool deadline_backlog = false;
  while (!execution_hold_cleanup_stopped_) {
    bool stopped = execution_hold_cleanup_stopped_;
    if (!deadline_backlog) {
      stopped = execution_hold_cleanup_cv_.wait_for(
          wait_lock,
          std::chrono::milliseconds(options_.request_watchdog_interval_ms()),
          [this] { return execution_hold_cleanup_stopped_; });
    }
    if (stopped) {
      break;
    }
    wait_lock.unlock();

    const std::vector<std::shared_ptr<Request>> disconnected_requests =
        client_disconnect_monitor_->take_disconnected(
            options_.request_watchdog_batch_size());
    for (const std::shared_ptr<Request>& request : disconnected_requests) {
      std::lock_guard<std::mutex> output_guard(request->output_dispatch_mutex);
      const std::string request_uid = request->correlation.request_uid();
      {
        std::lock_guard<std::mutex> request_guard(request_mutex_);
        const auto it = requests_.find(request_uid);
        if (it == requests_.end() || it->second != request) {
          continue;
        }
      }
      fail_output_dispatch_locked(
          request, llm::StatusCode::CANCELLED, "Client disconnected");
    }

    const RequestDeadlineQueue::TimePoint now =
        xllm::RequestDeadline::Clock::now();
    const std::vector<std::shared_ptr<Request>> deadline_requests =
        request_deadline_queue_->take_expired(
            now, options_.request_watchdog_batch_size());
    for (const std::shared_ptr<Request>& request : deadline_requests) {
      std::lock_guard<std::mutex> output_guard(request->output_dispatch_mutex);
      const std::string request_uid = request->correlation.request_uid();
      {
        std::lock_guard<std::mutex> request_guard(request_mutex_);
        const auto it = requests_.find(request_uid);
        if (it == requests_.end() || it->second != request) {
          continue;
        }
      }
      fail_output_dispatch_locked(
          request,
          llm::StatusCode::DEADLINE_EXCEEDED,
          "Request exceeded the Service monotonic deadline");
    }
    deadline_backlog = request_deadline_queue_->has_expired(now);

    std::vector<std::shared_ptr<Request>> failed_prefill_requests;
    std::unordered_set<std::string> failed_prefill_request_uids;
    {
      std::lock_guard<std::mutex> watch_guard(output_gap_watch_mutex_);
      for (auto it = failed_prefill_recovery_watchlist_.begin();
           it != failed_prefill_recovery_watchlist_.end() &&
           failed_prefill_requests.size() <
               options_.output_gap_query_batch_size();) {
        std::shared_ptr<Request> request = it->second.lock();
        if (request != nullptr) {
          failed_prefill_request_uids.insert(it->first);
          failed_prefill_requests.emplace_back(std::move(request));
        }
        it = failed_prefill_recovery_watchlist_.erase(it);
      }
    }
    recover_first_output_events(failed_prefill_requests,
                                /*fail_if_unavailable=*/true);

    std::vector<std::shared_ptr<Request>> gap_requests;
    {
      std::lock_guard<std::mutex> watch_guard(output_gap_watch_mutex_);
      gap_requests.reserve(output_gap_watchlist_.size());
      for (auto it = output_gap_watchlist_.begin();
           it != output_gap_watchlist_.end();) {
        std::shared_ptr<Request> request = it->second.lock();
        if (request == nullptr) {
          it = output_gap_watchlist_.erase(it);
          continue;
        }
        gap_requests.emplace_back(std::move(request));
        ++it;
      }
    }

    std::vector<std::shared_ptr<Request>> first_event_recovery_requests;
    first_event_recovery_requests.reserve(
        options_.output_gap_query_batch_size());
    for (const std::shared_ptr<Request>& request : gap_requests) {
      std::lock_guard<std::mutex> output_guard(request->output_dispatch_mutex);
      if (request->output_event_sequencer == nullptr ||
          !request->output_event_sequencer->gap_expired(
              OutputEventSequencer::Clock::now(),
              std::chrono::milliseconds(options_.output_gap_timeout_ms()))) {
        continue;
      }

      const std::string request_uid = request->correlation.request_uid();
      if (failed_prefill_request_uids.find(request_uid) !=
          failed_prefill_request_uids.end()) {
        continue;
      }
      {
        std::lock_guard<std::mutex> request_guard(request_mutex_);
        const auto it = requests_.find(request_uid);
        if (it == requests_.end() || it->second != request) {
          continue;
        }
      }
      if (request->output_event_sequencer->next_expected_seq() == 0) {
        if (first_event_recovery_requests.size() <
            options_.output_gap_query_batch_size() -
                failed_prefill_requests.size()) {
          first_event_recovery_requests.emplace_back(request);
        }
        continue;
      }
      fail_output_dispatch_locked(
          request,
          llm::StatusCode::DEADLINE_EXCEEDED,
          "Output sequence gap exceeded the local timeout");
    }
    recover_first_output_events(first_event_recovery_requests,
                                /*fail_if_unavailable=*/false);

    wait_lock.lock();
  }
}

void Scheduler::arm_client_disconnect_notification(
    const std::shared_ptr<Request>& request) {
  request->call_data->notify_on_disconnect(
      brpc::NewCallback(&signal_client_disconnect,
                        client_disconnect_monitor_,
                        request->correlation.request_uid()));
}

void Scheduler::handle_attempt_dispatch_failure(const std::string& request_uid,
                                                uint64_t attempt_seq,
                                                std::string message) {
  std::shared_ptr<Request> request;
  {
    std::lock_guard<std::mutex> request_guard(request_mutex_);
    const auto it = requests_.find(request_uid);
    if (it == requests_.end()) {
      return;
    }
    request = it->second;
  }

  std::lock_guard<std::mutex> output_guard(request->output_dispatch_mutex);
  if (request->output_dispatch_closed ||
      !request->correlation.has_attempt_seq() ||
      request->correlation.attempt_seq() != attempt_seq) {
    LOG(INFO) << "Ignore stale native dispatch failure, request_uid="
              << request_uid << ", attempt_seq=" << attempt_seq;
    return;
  }
  if (request->first_token_emitted.load(std::memory_order_acquire) ||
      (request->output_event_sequencer != nullptr &&
       request->output_event_sequencer->next_expected_seq() != 0)) {
    LOG(WARNING) << "Ignore native submission response failure after output "
                    "commit, request_uid="
                 << request_uid << ", attempt_seq=" << attempt_seq
                 << ", error=" << message;
    return;
  }

  LOG(ERROR) << "Native attempt dispatch failed before first output, "
                "request_uid="
             << request_uid << ", attempt_seq=" << attempt_seq
             << ", error=" << message;
  {
    std::lock_guard<std::mutex> watch_guard(output_gap_watch_mutex_);
    failed_prefill_recovery_watchlist_.insert_or_assign(request_uid, request);
  }
  execution_hold_cleanup_cv_.notify_all();
}

bool Scheduler::record_new_request(std::shared_ptr<ChatCallData> call_data,
                                   std::shared_ptr<Request> request) {
  {
    std::lock_guard<std::mutex> cleanup_guard(execution_hold_cleanup_mutex_);
    std::lock_guard<std::mutex> guard(request_mutex_);
    if (requests_.find(request->correlation.request_uid()) != requests_.end()) {
      LOG(ERROR) << "The request ID already exists. Requests with the same ID "
                    "are not allowed. "
                 << request->correlation.request_uid();
      return false;
    }
    if (!install_request_safety_guards_locked(request)) {
      return false;
    }

    request->latest_generate_time = absl::Now();
    auto tools_for_parse =
        (request->tool_choice == "none" ? std::vector<JsonTool>{}
                                        : request->tools);
    auto tool_call_parser_pref = options_.tool_call_parser();
    auto reasoning_parser_pref = options_.reasoning_parser();
    const auto parser_formats = resolve_chat_parser_formats_with_xllm(
        request->model, tool_call_parser_pref, reasoning_parser_pref);
    const bool force_reasoning = get_enable_thinking_from_request(
        request->chat_template_kwargs, parser_formats.reasoning_parser);
    std::shared_ptr<ChatStreamParseState> stream_state;
    if (request->stream) {
      stream_state = response_handler_.create_chat_stream_parse_state(
          tools_for_parse,
          request->model,
          tool_call_parser_pref,
          reasoning_parser_pref,
          force_reasoning);
    }

    request->call_data = call_data;
    request->output_callback =
        [this,
         call_data,
         model = request->model,
         stream = request->stream,
         include_usage = request->include_usage,
         tools = std::move(tools_for_parse),
         tool_call_parser = std::move(tool_call_parser_pref),
         reasoning_parser = std::move(reasoning_parser_pref),
         force_reasoning,
         stream_state = std::move(stream_state),
         created_time = absl::ToUnixSeconds(request->latest_generate_time)](
            const llm::RequestOutput& req_output) mutable -> bool {
      if (req_output.status.has_value()) {
        const auto& status = req_output.status.value();
        if (!status.ok()) {
          return call_data->finish_with_error(status.message());
        }
      }

      if (stream) {
        return response_handler_.send_delta_to_client(call_data,
                                                      include_usage,
                                                      created_time,
                                                      model,
                                                      req_output,
                                                      stream_state);
      } else if (!req_output.finished_on_prefill_instance) {
        // for non-stream request, only send final result from decode instance
        return response_handler_.send_result_to_client(call_data,
                                                       created_time,
                                                       model,
                                                       req_output,
                                                       tools,
                                                       tool_call_parser,
                                                       reasoning_parser,
                                                       force_reasoning);
      }
      return true;
    };
    if (request->request_deadline.has_value() &&
        request_deadline_queue_->insert(
            request->correlation.request_uid(),
            request,
            request->request_deadline->time_point()) !=
            RequestDeadlineQueueStatus::kOk) {
      LOG(ERROR) << "Failed to index request deadline, request_uid="
                 << request->correlation.request_uid();
      rollback_request_safety_guards_locked(request);
      return false;
    }
    requests_.emplace(request->correlation.request_uid(), request);
    COUNTER_INC(server_request_in_total);
  }

  {
    // allocate thread for the request
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    remote_requests_output_thread_map_[request->correlation.request_uid()] =
        next_thread_idx;
    next_thread_idx = (++next_thread_idx) % kOutputTheadNum_;
  }

  arm_client_disconnect_notification(request);

  return true;
}

bool Scheduler::record_new_request(std::shared_ptr<AnthropicCallData> call_data,
                                   std::shared_ptr<Request> request) {
  {
    std::lock_guard<std::mutex> cleanup_guard(execution_hold_cleanup_mutex_);
    std::lock_guard<std::mutex> guard(request_mutex_);
    if (requests_.find(request->correlation.request_uid()) != requests_.end()) {
      LOG(ERROR) << "The request ID already exists. Requests with the same ID "
                    "are not allowed. "
                 << request->correlation.request_uid();
      return false;
    }
    if (!install_request_safety_guards_locked(request)) {
      return false;
    }

    request->latest_generate_time = absl::Now();
    auto tools_for_parse =
        (request->tool_choice == "none" ? std::vector<JsonTool>{}
                                        : request->tools);
    auto tool_call_parser_pref = options_.tool_call_parser();
    auto reasoning_parser_pref = options_.reasoning_parser();
    const auto parser_formats = resolve_chat_parser_formats_with_xllm(
        request->model, tool_call_parser_pref, reasoning_parser_pref);
    const bool force_reasoning = get_enable_thinking_from_request(
        request->chat_template_kwargs, parser_formats.reasoning_parser);
    auto stream_encoder =
        request->stream
            ? std::make_shared<AnthropicStreamEncoder>(request->model)
            : nullptr;
    auto stream_parser =
        request->stream
            ? create_stream_output_parser_with_xllm(tools_for_parse,
                                                    request->model,
                                                    tool_call_parser_pref,
                                                    reasoning_parser_pref,
                                                    force_reasoning)
            : nullptr;
    request->call_data = call_data;
    request->output_callback =
        [this,
         call_data,
         model = request->model,
         stream = request->stream,
         stream_encoder,
         stream_parser,
         tools = std::move(tools_for_parse),
         tool_call_parser = std::move(tool_call_parser_pref),
         reasoning_parser = std::move(reasoning_parser_pref),
         force_reasoning](
            const llm::RequestOutput& req_output) mutable -> bool {
      if (req_output.status.has_value()) {
        const auto& status = req_output.status.value();
        if (!status.ok()) {
          return call_data->finish_with_error(status.message());
        }
      }

      if (stream) {
        return response_handler_.send_delta_to_client(
            call_data, model, req_output, *stream_encoder, stream_parser);
      } else if (!req_output.finished_on_prefill_instance) {
        return response_handler_.send_result_to_client(call_data,
                                                       model,
                                                       req_output,
                                                       tools,
                                                       tool_call_parser,
                                                       reasoning_parser,
                                                       force_reasoning);
      }
      return true;
    };
    if (request->request_deadline.has_value() &&
        request_deadline_queue_->insert(
            request->correlation.request_uid(),
            request,
            request->request_deadline->time_point()) !=
            RequestDeadlineQueueStatus::kOk) {
      LOG(ERROR) << "Failed to index request deadline, request_uid="
                 << request->correlation.request_uid();
      rollback_request_safety_guards_locked(request);
      return false;
    }
    requests_.emplace(request->correlation.request_uid(), request);
    COUNTER_INC(server_request_in_total);
  }

  {
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    remote_requests_output_thread_map_[request->correlation.request_uid()] =
        next_thread_idx;
    next_thread_idx = (++next_thread_idx) % kOutputTheadNum_;
  }

  arm_client_disconnect_notification(request);

  return true;
}

bool Scheduler::record_new_request(
    std::shared_ptr<CompletionCallData> call_data,
    std::shared_ptr<Request> request) {
  {
    std::lock_guard<std::mutex> cleanup_guard(execution_hold_cleanup_mutex_);
    std::lock_guard<std::mutex> guard(request_mutex_);
    if (requests_.find(request->correlation.request_uid()) != requests_.end()) {
      LOG(ERROR) << "The request ID already exists. Requests with the same ID "
                    "are not allowed. "
                 << request->correlation.request_uid();
      return false;
    }
    if (!install_request_safety_guards_locked(request)) {
      return false;
    }

    request->latest_generate_time = absl::Now();

    request->call_data = call_data;
    request->output_callback =
        [this,
         call_data,
         model = request->model,
         stream = request->stream,
         include_usage = request->include_usage,
         created_time = absl::ToUnixSeconds(request->latest_generate_time)](
            const llm::RequestOutput& req_output) mutable -> bool {
      if (req_output.status.has_value()) {
        const auto& status = req_output.status.value();
        if (!status.ok()) {
          return call_data->finish_with_error(status.message());
        }
      }

      if (stream) {
        return response_handler_.send_delta_to_client(
            call_data, include_usage, created_time, model, req_output);
      } else if (!req_output.finished_on_prefill_instance) {
        // for non-stream request, only send final result from decode instance
        return response_handler_.send_result_to_client(
            call_data, created_time, model, req_output);
      }
      return true;
    };
    if (request->request_deadline.has_value() &&
        request_deadline_queue_->insert(
            request->correlation.request_uid(),
            request,
            request->request_deadline->time_point()) !=
            RequestDeadlineQueueStatus::kOk) {
      LOG(ERROR) << "Failed to index request deadline, request_uid="
                 << request->correlation.request_uid();
      rollback_request_safety_guards_locked(request);
      return false;
    }
    requests_.emplace(request->correlation.request_uid(), request);
    COUNTER_INC(server_request_in_total);
  }

  {
    // allocate thread for the request
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    remote_requests_output_thread_map_[request->correlation.request_uid()] =
        next_thread_idx;
    next_thread_idx = (++next_thread_idx) % kOutputTheadNum_;
  }

  arm_client_disconnect_notification(request);

  return true;
}

void Scheduler::finish_request(const std::string& service_request_id,
                               bool error) {
  std::shared_ptr<Request> request;
  {
    std::lock_guard<std::mutex> cleanup_guard(execution_hold_cleanup_mutex_);
    {
      std::lock_guard<std::mutex> guard(request_mutex_);
      auto it = requests_.find(service_request_id);
      if (it != requests_.end()) {
        request = it->second;
        requests_.erase(it);
      }
    }
    if (request != nullptr) {
      cancel_or_detach_execution_hold_locked(request);
    }
  }

  if (request != nullptr) {
    if (error) {
      instance_mgr_->update_request_metrics(request, RequestAction::CANCEL);
    } else {
      instance_mgr_->update_request_metrics(request,
                                            RequestAction::FINISH_DECODE);
    }
  }

  {
    std::lock_guard<std::mutex> watch_guard(output_gap_watch_mutex_);
    output_gap_watchlist_.erase(service_request_id);
    failed_prefill_recovery_watchlist_.erase(service_request_id);
  }
  request_deadline_queue_->erase(service_request_id);
  client_disconnect_monitor_->erase(service_request_id);

  {
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    remote_requests_output_thread_map_.erase(service_request_id);
  }
}

void Scheduler::clear_requests_on_failed_instance(
    const std::string& instance_name,
    const std::string& incarnation_id,
    InstanceType type) {
  std::vector<std::shared_ptr<Request>> cleared_requests;
  std::vector<std::shared_ptr<Request>> first_event_recovery_requests;
  {
    std::lock_guard<std::mutex> cleanup_guard(execution_hold_cleanup_mutex_);
    xllm::proto::ExecutionHolder terminated_holder;
    terminated_holder.set_engine_uid(instance_name);
    terminated_holder.set_incarnation_id(incarnation_id);
    execution_hold_cleanup_table_->mark_holder_process_terminated(
        terminated_holder);

    {
      std::lock_guard<std::mutex> lock(request_mutex_);
      for (auto it = requests_.begin(); it != requests_.end();) {
        const bool clear_prefill =
            ((type == InstanceType::DEFAULT || type == InstanceType::PREFILL) &&
             it->second->routing.prefill_name == instance_name &&
             it->second->prefill_incarnation_id == incarnation_id &&
             !it->second->prefill_stage_finished.load(
                 std::memory_order_acquire));
        const bool clear_decode =
            (type == InstanceType::DECODE &&
             it->second->routing.decode_name == instance_name &&
             it->second->decode_incarnation_id == incarnation_id);
        const bool recover_first_event =
            clear_prefill && !clear_decode &&
            !it->second->routing.decode_name.empty() &&
            it->second->output_event_sequencer != nullptr;
        if (recover_first_event) {
          first_event_recovery_requests.emplace_back(it->second);
          ++it;
        } else if (clear_prefill || clear_decode) {
          cleared_requests.emplace_back(it->second);
          it = requests_.erase(it);
        } else {
          ++it;
        }
      }
    }

    for (const std::shared_ptr<Request>& request : cleared_requests) {
      if (request->routing.decode_name == instance_name &&
          request->decode_incarnation_id == incarnation_id) {
        request->execution_hold.apply_convergence_proof(
            execution_attempt(*request),
            terminated_holder,
            xllm::proto::HOLDER_CONVERGENCE_PROOF_PROCESS_TERMINATED);
      }
      cancel_or_detach_execution_hold_locked(request);
    }
  }

  if (!first_event_recovery_requests.empty()) {
    std::lock_guard<std::mutex> watch_guard(output_gap_watch_mutex_);
    for (const std::shared_ptr<Request>& request :
         first_event_recovery_requests) {
      failed_prefill_recovery_watchlist_.insert_or_assign(
          request->correlation.request_uid(), request);
    }
  }

  for (const std::shared_ptr<Request>& request : cleared_requests) {
    std::lock_guard<std::mutex> output_guard(request->output_dispatch_mutex);
    fail_output_dispatch_locked(
        request, llm::StatusCode::CANCELLED, "Instance is failed and deleted");
    LOG(INFO) << "Clear request on failed instance: " << instance_name
              << ", incarnation_id: " << incarnation_id
              << ", service_request_id: " << request->correlation.request_uid();
  }
}

bool Scheduler::handle_generation(const llm::RequestOutput& request_output) {
  return handle_generation_detailed(request_output).accepted();
}

GenerationDeliveryResult Scheduler::handle_generation_detailed(
    const llm::RequestOutput& request_output) {
  const std::string& service_request_id = request_output.service_request_id;

  OutputCallback cb;
  std::shared_ptr<Request> request;
  bool client_disconnected = false;
  {
    std::lock_guard<std::mutex> guard(request_mutex_);
    auto it = requests_.find(service_request_id);
    if (it == requests_.end()) {
      LOG(ERROR) << "Can not found the callback for the received request "
                    "output, request id is: "
                 << service_request_id;
      return GenerationDeliveryResult(
          /*code=*/proto::GENERATION_DELIVERY_CODE_UNKNOWN_REQUEST,
          /*message=*/"Unknown service request");
    }
    request = it->second;
    cb = request->output_callback;

    // check client connection
    if (request->client_disconnected.load(std::memory_order_acquire) ||
        request->call_data->is_disconnected()) {
      LOG(INFO) << "Client has disconnected and the request will be cancelled, "
                   "request id: "
                << service_request_id;
      client_disconnected = true;
    }
  }

  std::lock_guard<std::mutex> output_guard(request->output_dispatch_mutex);
  if (client_disconnected) {
    request->output_dispatch_closed = true;
    if (request->output_event_sequencer != nullptr) {
      request->output_event_sequencer->close();
    }
    finish_request(service_request_id, /*error=*/true);
    return GenerationDeliveryResult(
        /*code=*/proto::GENERATION_DELIVERY_CODE_CLIENT_DISCONNECTED,
        /*message=*/"Client disconnected");
  }

  OutputEventSequenceResult sequence_result;
  if (request->output_event_sequencer != nullptr) {
    if (request_output.attempt_seq.has_value() &&
        request->correlation.has_attempt_seq() &&
        *request_output.attempt_seq != request->correlation.attempt_seq()) {
      LOG(INFO) << "Reject output from replaced attempt without closing the "
                   "current request, request_uid="
                << service_request_id
                << ", output_attempt_seq=" << *request_output.attempt_seq
                << ", current_attempt_seq="
                << request->correlation.attempt_seq();
      return GenerationDeliveryResult(
          /*code=*/proto::GENERATION_DELIVERY_CODE_INVALID_IDENTITY,
          /*message=*/"Output belongs to a replaced attempt");
    }
    RemotePdOutputBinding binding;
    if (request->correlation.has_attempt_seq()) {
      binding.attempt_seq = request->correlation.attempt_seq();
    }
    binding.prefill_engine_uid = request->routing.prefill_name;
    binding.prefill_incarnation_id = request->prefill_incarnation_id;
    binding.decode_engine_uid = request->routing.decode_name;
    binding.decode_incarnation_id = request->decode_incarnation_id;
    if (!matches_remote_pd_output_identity(request_output, binding)) {
      LOG(ERROR) << "Reject stale or mismatched V2 output, request_uid="
                 << service_request_id;
      fail_output_dispatch_locked(request,
                                  llm::StatusCode::INVALID_ARGUMENT,
                                  "Stale or mismatched V2 output identity");
      return GenerationDeliveryResult(
          /*code=*/proto::GENERATION_DELIVERY_CODE_INVALID_IDENTITY,
          /*message=*/"Stale or mismatched output identity");
    }
    sequence_result = request->output_event_sequencer->push(
        request_output, /*require_sequence=*/true);
  } else {
    sequence_result.ready_outputs.emplace_back(request_output);
  }

  if (sequence_result.status == OutputEventSequenceStatus::kBuffered ||
      sequence_result.status == OutputEventSequenceStatus::kDuplicate) {
    if (sequence_result.status == OutputEventSequenceStatus::kBuffered) {
      std::lock_guard<std::mutex> watch_guard(output_gap_watch_mutex_);
      output_gap_watchlist_.insert_or_assign(service_request_id, request);
    }
    return GenerationDeliveryResult(
        /*code=*/proto::GENERATION_DELIVERY_CODE_ACCEPTED,
        /*message=*/"");
  }
  if (sequence_result.status == OutputEventSequenceStatus::kClosed) {
    return GenerationDeliveryResult(
        /*code=*/proto::GENERATION_DELIVERY_CODE_REQUEST_CLOSED,
        /*message=*/"Request output stream is closed");
  }
  if (sequence_result.status != OutputEventSequenceStatus::kReady) {
    LOG(ERROR) << "Reject invalid V2 output sequence, request_uid="
               << service_request_id
               << ", status=" << static_cast<int>(sequence_result.status);
    fail_output_dispatch_locked(request,
                                llm::StatusCode::INVALID_ARGUMENT,
                                "Invalid V2 output sequence");
    return GenerationDeliveryResult(
        /*code=*/proto::GENERATION_DELIVERY_CODE_INVALID_SEQUENCE,
        /*message=*/"Invalid output event sequence");
  }
  if (request->output_event_sequencer != nullptr &&
      request->output_event_sequencer->buffered_events() == 0) {
    std::lock_guard<std::mutex> watch_guard(output_gap_watch_mutex_);
    output_gap_watchlist_.erase(service_request_id);
    failed_prefill_recovery_watchlist_.erase(service_request_id);
  }

  for (const llm::RequestOutput& ready_output : sequence_result.ready_outputs) {
    const bool status_error =
        ready_output.status.has_value() && !ready_output.status->ok();
    const bool finished_on_prefill_instance =
        ready_output.finished_on_prefill_instance;
    if (!status_error && finished_on_prefill_instance &&
        !confirm_generation_commit(request)) {
      fail_output_dispatch_locked(
          request, llm::StatusCode::UNKNOWN, "Invalid generation commit proof");
      return GenerationDeliveryResult(
          /*code=*/proto::GENERATION_DELIVERY_CODE_COMMIT_UNPROVEN,
          /*message=*/"Generation commit is not proven");
    }
    if (!status_error && !finished_on_prefill_instance &&
        ready_output.finished && !resolve_terminal_execution_hold(request)) {
      fail_output_dispatch_locked(request,
                                  llm::StatusCode::UNKNOWN,
                                  "Invalid terminal execution proof");
      return GenerationDeliveryResult(
          /*code=*/proto::GENERATION_DELIVERY_CODE_TERMINAL_UNPROVEN,
          /*message=*/"Terminal execution is not proven");
    }
    if (!status_error) {
      update_request_metrics(request, finished_on_prefill_instance);
      update_token_latency_metrics(request, finished_on_prefill_instance);
    }
  }

  size_t req_thread_idx = -1;
  bool output_thread_found = false;
  {
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    auto it = remote_requests_output_thread_map_.find(service_request_id);
    if (it != remote_requests_output_thread_map_.end()) {
      req_thread_idx = it->second;
      output_thread_found = true;
    }
  }
  if (!output_thread_found) {
    LOG(ERROR) << "Can not found the thread for the received request output, "
                  "request id is: "
               << service_request_id;
    fail_output_dispatch_locked(
        request, llm::StatusCode::UNKNOWN, "Output affinity thread is missing");
    return GenerationDeliveryResult(
        /*code=*/proto::GENERATION_DELIVERY_CODE_AFFINITY_MISSING,
        /*message=*/"Output affinity thread is missing");
  }

  const bool terminal_output =
      std::any_of(sequence_result.ready_outputs.begin(),
                  sequence_result.ready_outputs.end(),
                  [](const llm::RequestOutput& output) {
                    return output.finished ||
                           (output.status.has_value() && !output.status->ok());
                  });
  if (terminal_output) {
    request->output_dispatch_closed = true;
    if (request->output_event_sequencer != nullptr) {
      request->output_event_sequencer->close();
    }
  }

  output_threadpools_[req_thread_idx].schedule(
      [this,
       request,
       service_request_id,
       cb,
       ready_outputs = std::move(sequence_result.ready_outputs)]() mutable {
        for (llm::RequestOutput& ready_output : ready_outputs) {
          const bool status_error =
              ready_output.status.has_value() && !ready_output.status->ok();
          const bool finished = ready_output.finished;
          const bool crosses_response_boundary =
              request->stream || !ready_output.finished_on_prefill_instance;
          if (!cb(std::move(ready_output)) || status_error) {
            finish_request(service_request_id, true);
            return;
          }
          if (crosses_response_boundary) {
            request->first_token_emitted.store(true, std::memory_order_release);
          }
          if (finished) {
            finish_request(service_request_id);
            return;
          }
        }
      });

  return GenerationDeliveryResult(
      /*code=*/proto::GENERATION_DELIVERY_CODE_ACCEPTED,
      /*message=*/"");
}

void Scheduler::update_request_metrics(std::shared_ptr<Request> request,
                                       bool finished_on_prefill_instance) {
  request->num_generated_tokens += 1;
  if (finished_on_prefill_instance) {
    request->prefill_stage_finished.store(true, std::memory_order_release);
    // update instance request metrics for prefill finished request
    instance_mgr_->update_request_metrics(request,
                                          RequestAction::FINISH_PREFILL);
  } else {
    // update instance request metrics
    instance_mgr_->update_request_metrics(request, RequestAction::GENERATE);
  }
}

void Scheduler::update_token_latency_metrics(
    std::shared_ptr<Request> request,
    bool finished_on_prefill_instance) {
  int64_t tbt_milliseconds =
      absl::ToInt64Milliseconds(absl::Now() - request->latest_generate_time);
  request->latest_generate_time = absl::Now();
  if (finished_on_prefill_instance) {
    HISTOGRAM_OBSERVE(time_to_first_token_latency_milliseconds,
                      tbt_milliseconds);
  } else {
    HISTOGRAM_OBSERVE(inter_token_latency_milliseconds, tbt_milliseconds);
  }
}

bool Scheduler::has_available_instances() const {
  return instance_mgr_->has_available_instances();
}

}  // namespace xllm_service
