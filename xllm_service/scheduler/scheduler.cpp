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
#include <exception>
#include <functional>
#include <future>
#include <limits>
#include <unordered_set>

#include "chat_template/deepseek_v4_cpp_chat_template.h"
#include "chat_template/model_type.h"
#include "common/global_gflags.h"
#include "common/metrics.h"
#include "common/utils.h"
#include "common/xllm/status.h"
#include "common/xllm/uuid.h"
#include "http_service/anthropic_adapter.h"
#include "http_service/anthropic_stream_encoder.h"
#include "loadbalance_policy/cache_aware_routing.h"
#include "loadbalance_policy/round_robin.h"
#include "loadbalance_policy/slo_aware_policy.h"
#include "provider/attempt_control_client.h"
#include "provider/execution_plan_builder.h"
#include "provider/native_execution_mode_selector.h"
#include "provider/provider_adapter.h"
#include "rpc_service/first_event_recovery_client.h"
#include "rpc_service/kv_snapshot_client.h"
#include "rpc_service/kv_state_stream_client.h"
#include "rpc_service/state_stream_client.h"
#include "scheduler/xllm_chat_parse_bridge.h"
#include "tokenizer/tokenizer_factory.h"

namespace {
constexpr int32_t kHeartbeatInterval = 3;  // in seconds
constexpr int32_t kRegistrationMaxRetries = 5;
constexpr size_t kMaxObservabilityEventCapacity = 1 << 20;

constexpr const char* kEtcdUsernameEnvVar = "ETCD_USERNAME";
constexpr const char* kEtcdPasswordEnvVar = "ETCD_PASSWORD";

bool valid_log_token(const std::string& value) {
  return !value.empty() && value.size() <= 256 &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return (character >= 'a' && character <= 'z') ||
                  (character >= 'A' && character <= 'Z') ||
                  (character >= '0' && character <= '9') || character == '-' ||
                  character == '_' || character == '.' || character == ':' ||
                  character == '/' || character == '@' || character == '+';
         });
}

uint64_t monotonic_time_ms() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

uint64_t unix_time_ms() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

const char* placement_action_name(
    xllm_service::placement::PlacementAction action) {
  switch (action) {
    case xllm_service::placement::PlacementAction::NONE:
      return "NONE";
    case xllm_service::placement::PlacementAction::SCALE_UP:
      return "SCALE_UP";
    case xllm_service::placement::PlacementAction::SCALE_DOWN:
      return "SCALE_DOWN";
  }
  return "UNKNOWN";
}

int64_t monotonic_time_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void saturated_atomic_add(std::atomic<uint64_t>* target, uint64_t value) {
  uint64_t current = target->load(std::memory_order_relaxed);
  while (true) {
    const uint64_t next = value > std::numeric_limits<uint64_t>::max() - current
                              ? std::numeric_limits<uint64_t>::max()
                              : current + value;
    if (target->compare_exchange_weak(current,
                                      next,
                                      std::memory_order_relaxed,
                                      std::memory_order_relaxed)) {
      return;
    }
  }
}

uint64_t counter_delta(uint64_t current, uint64_t previous) {
  return current >= previous ? current - previous : 0;
}

std::optional<uint64_t> elapsed_ns_since(
    std::chrono::steady_clock::time_point started) {
  const std::chrono::steady_clock::time_point now =
      std::chrono::steady_clock::now();
  if (now < started) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now - started)
          .count());
}

std::string execution_mode_label(xllm::proto::ExecutionMode mode) {
  if (!xllm::proto::ExecutionMode_IsValid(mode)) {
    return "EXECUTION_MODE_INVALID";
  }
  return xllm::proto::ExecutionMode_Name(mode);
}

xllm::proto::EventReason decode_admission_event_reason(
    xllm::proto::AdmissionReason reason) {
  switch (reason) {
    case xllm::proto::ADMISSION_REASON_NONE:
      return xllm::proto::EVENT_REASON_NONE;
    case xllm::proto::ADMISSION_REASON_CONTEXT_TOO_LARGE:
    case xllm::proto::ADMISSION_REASON_KV_PERMANENTLY_INFEASIBLE:
    case xllm::proto::ADMISSION_REASON_UNSUPPORTED_KV_LAYOUT:
      return xllm::proto::EVENT_REASON_PERMANENTLY_INFEASIBLE;
    case xllm::proto::ADMISSION_REASON_ATTEMPT_CONFLICT:
    case xllm::proto::ADMISSION_REASON_ATTEMPT_TERMINAL:
      return xllm::proto::EVENT_REASON_ATTEMPT_CONFLICT;
    case xllm::proto::ADMISSION_REASON_DEADLINE_EXCEEDED:
    case xllm::proto::ADMISSION_REASON_RESERVATION_EXPIRED:
      return xllm::proto::EVENT_REASON_DEADLINE_EXCEEDED;
    case xllm::proto::ADMISSION_REASON_CANCELLED_BEFORE_CREATE:
    case xllm::proto::ADMISSION_REASON_CANCELLED:
      return xllm::proto::EVENT_REASON_CANCELLED;
    case xllm::proto::ADMISSION_REASON_STALE_INCARNATION:
      return xllm::proto::EVENT_REASON_STALE_STATE;
    case xllm::proto::ADMISSION_REASON_KV_TEMPORARILY_INSUFFICIENT:
    case xllm::proto::ADMISSION_REASON_DECODE_CREDIT_EXHAUSTED:
    case xllm::proto::ADMISSION_REASON_SLOT_EXHAUSTED:
    case xllm::proto::ADMISSION_REASON_CAPACITY_CHANGED:
    case xllm::proto::ADMISSION_REASON_TOMBSTONE_CAPACITY:
    case xllm::proto::ADMISSION_REASON_CANCEL_FENCE_CAPACITY:
    case xllm::proto::ADMISSION_REASON_RECOVERY_FENCE_PRESSURE:
      return xllm::proto::EVENT_REASON_CAPACITY_EXHAUSTED;
    case xllm::proto::ADMISSION_REASON_INVALID_REQUEST:
    case xllm::proto::ADMISSION_REASON_ENGINE_DRAINING:
    case xllm::proto::ADMISSION_REASON_TRANSFER_NOT_STARTED:
    case xllm::proto::ADMISSION_REASON_HANDOFF_REJECTED:
    case xllm::proto::ADMISSION_REASON_INTERNAL_ERROR:
    case xllm::proto::ADMISSION_REASON_ATTEMPT_NOT_FOUND:
    case xllm::proto::ADMISSION_REASON_UNSPECIFIED:
      return xllm::proto::EVENT_REASON_PROVIDER_ERROR;
    default:
      return xllm::proto::EVENT_REASON_PROVIDER_ERROR;
  }
}

const char* saturation_state_name(xllm_service::SaturationState state) {
  switch (state) {
    case xllm_service::SaturationState::AVAILABLE:
      return "AVAILABLE";
    case xllm_service::SaturationState::SATURATED:
      return "SATURATED";
    case xllm_service::SaturationState::UNKNOWN:
      return "UNKNOWN";
  }
  return "UNKNOWN";
}

xllm::proto::EventResult terminal_result_for_status(
    xllm_service::llm::StatusCode status) {
  if (status == xllm_service::llm::StatusCode::CANCELLED) {
    return xllm::proto::EVENT_RESULT_CANCELLED;
  }
  if (status == xllm_service::llm::StatusCode::DEADLINE_EXCEEDED) {
    return xllm::proto::EVENT_RESULT_DEADLINE_EXCEEDED;
  }
  return xllm::proto::EVENT_RESULT_FAILED;
}

xllm::proto::EventReason event_reason_for_status(
    xllm_service::llm::StatusCode status) {
  switch (status) {
    case xllm_service::llm::StatusCode::CANCELLED:
      return xllm::proto::EVENT_REASON_CANCELLED;
    case xllm_service::llm::StatusCode::DEADLINE_EXCEEDED:
      return xllm::proto::EVENT_REASON_DEADLINE_EXCEEDED;
    case xllm_service::llm::StatusCode::RESOURCE_EXHAUSTED:
      return xllm::proto::EVENT_REASON_CAPACITY_EXHAUSTED;
    case xllm_service::llm::StatusCode::UNAVAILABLE:
      return xllm::proto::EVENT_REASON_STALE_STATE;
    case xllm_service::llm::StatusCode::UNIMPLEMENTED:
      return xllm::proto::EVENT_REASON_MODE_UNSUPPORTED;
    case xllm_service::llm::StatusCode::OK:
    case xllm_service::llm::StatusCode::UNKNOWN:
    case xllm_service::llm::StatusCode::INVALID_ARGUMENT:
    case xllm_service::llm::StatusCode::UNAUTHENTICATED:
      return xllm::proto::EVENT_REASON_PROVIDER_ERROR;
  }
  return xllm::proto::EVENT_REASON_INTERNAL_ERROR;
}

xllm::proto::EventReason event_reason_for_flow_status(
    xllm_service::FlowControlStatus status) {
  switch (status) {
    case xllm_service::FlowControlStatus::QUEUE_CAPACITY_EXHAUSTED:
      return xllm::proto::EVENT_REASON_CAPACITY_EXHAUSTED;
    case xllm_service::FlowControlStatus::QUEUE_DEADLINE_UNSATISFIABLE:
      return xllm::proto::EVENT_REASON_DEADLINE_EXCEEDED;
    case xllm_service::FlowControlStatus::DUPLICATE_REQUEST:
      return xllm::proto::EVENT_REASON_ATTEMPT_CONFLICT;
    case xllm_service::FlowControlStatus::UNKNOWN_REQUEST:
    case xllm_service::FlowControlStatus::INVALID_ARGUMENT:
      return xllm::proto::EVENT_REASON_INTERNAL_ERROR;
    case xllm_service::FlowControlStatus::OK:
      return xllm::proto::EVENT_REASON_NONE;
  }
  return xllm::proto::EVENT_REASON_INTERNAL_ERROR;
}

const char* output_sequence_status_name(
    xllm_service::OutputEventSequenceStatus status) {
  switch (status) {
    case xllm_service::OutputEventSequenceStatus::kReady:
      return "ready";
    case xllm_service::OutputEventSequenceStatus::kBuffered:
      return "buffered_gap";
    case xllm_service::OutputEventSequenceStatus::kDuplicate:
      return "duplicate";
    case xllm_service::OutputEventSequenceStatus::kMissingSequence:
      return "missing_sequence";
    case xllm_service::OutputEventSequenceStatus::kGapTooLarge:
      return "gap_too_large";
    case xllm_service::OutputEventSequenceStatus::kCapacityExceeded:
      return "capacity_exceeded";
    case xllm_service::OutputEventSequenceStatus::kTerminalConflict:
      return "terminal_conflict";
    case xllm_service::OutputEventSequenceStatus::kInvalidSequence:
      return "invalid_sequence";
    case xllm_service::OutputEventSequenceStatus::kClosed:
      return "closed";
  }
  return "unknown";
}

const char* observation_mode_name(
    xllm_service::provider::ObservationMode mode) {
  switch (mode) {
    case xllm_service::provider::ObservationMode::NORMAL:
      return "NORMAL";
    case xllm_service::provider::ObservationMode::STATE_BLIND:
      return "STATE_BLIND";
    case xllm_service::provider::ObservationMode::REGISTRY_BLIND:
      return "REGISTRY_BLIND";
  }
  return "UNKNOWN";
}

bool uint64_add_overflows(uint64_t left, uint64_t right) {
  return left > std::numeric_limits<uint64_t>::max() - right;
}

uint64_t stable_identity_hash(const std::string& identity) {
  constexpr uint64_t kOffset = 1469598103934665603ULL;
  constexpr uint64_t kPrime = 1099511628211ULL;
  uint64_t hash = kOffset;
  for (unsigned char byte : identity) {
    hash ^= static_cast<uint64_t>(byte);
    hash *= kPrime;
  }
  return hash;
}

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

xllm_service::provider::ReadinessControllerConfig readiness_config(
    const xllm_service::Options& options) {
  return xllm_service::provider::ReadinessControllerConfig{
      .recovery_hold_ms = options.readiness_recovery_hold_ms(),
  };
}

xllm_service::FlowControlConfig flow_control_config(
    const xllm_service::Options& options) {
  const xllm_service::FlowOrder order = options.flow_order() == "EDF"
                                            ? xllm_service::FlowOrder::EDF
                                            : xllm_service::FlowOrder::FCFS;
  return xllm_service::FlowControlConfig{
      .max_queued_requests = options.flow_max_queued_requests(),
      .max_dispatched_request_contexts = options.flow_max_dispatched_contexts(),
      .max_queued_prompt_tokens = options.flow_max_queued_prompt_tokens(),
      .max_queued_bytes = options.flow_max_queued_bytes(),
      .max_queue_wait_ms = options.flow_max_queue_wait_ms(),
      .max_queued_requests_per_tenant =
          options.flow_max_queued_requests_per_tenant(),
      .max_queued_tokens_per_tenant =
          options.flow_max_queued_tokens_per_tenant(),
      .max_model_queued_requests = options.flow_max_model_queued_requests(),
      .max_model_dispatched_request_contexts =
          options.flow_max_model_dispatched_contexts(),
      .max_model_queued_prompt_tokens =
          options.flow_max_model_queued_prompt_tokens(),
      .max_model_queued_bytes = options.flow_max_model_queued_bytes(),
      .service_crash_request_budget =
          options.flow_service_crash_request_budget(),
      .service_memory_budget_bytes = options.flow_service_memory_budget_bytes(),
      .dispatched_context_bytes = options.flow_dispatched_context_bytes(),
      .dispatch_rate_lb_per_second = options.flow_dispatch_rate_lb_per_second(),
      .probe_round_ub_ms = options.flow_probe_round_ub_ms(),
      .blind_dispatch_probe_concurrency =
          options.flow_blind_dispatch_probe_concurrency(),
      .starvation_dispatch_bound = options.flow_starvation_dispatch_bound(),
      .flow_order = order,
  };
}

xllm_service::provider::KVShadowIndexConfig kv_shadow_config(
    const xllm_service::Options& options) {
  return xllm_service::provider::KVShadowIndexConfig{
      .max_engine_streams = options.kv_shadow_max_engine_streams(),
      .max_index_entries = options.kv_shadow_max_entries(),
      .max_index_bytes = options.kv_shadow_max_bytes(),
      .max_recovery_events_per_engine = options.kv_shadow_max_recovery_events(),
      .max_recovery_bytes_per_engine = options.kv_shadow_max_recovery_bytes(),
      .max_snapshot_entries_per_engine =
          options.kv_shadow_max_snapshot_entries(),
      .max_snapshot_bytes_per_engine = options.kv_shadow_max_snapshot_bytes(),
      .event_ttl_ms = options.kv_shadow_event_ttl_ms(),
      .recovery_timeout_ms = options.kv_shadow_recovery_timeout_ms(),
  };
}

bool matches_current_kv_engine(const xllm_service::InstanceMetaInfo& info,
                               const xllm::proto::KVStreamIdentity& identity) {
  const xllm::proto::ProviderEngineKey& engine = identity.engine();
  if (info.name.empty() || info.name != engine.engine_uid() ||
      info.incarnation_id != engine.incarnation_id() ||
      info.provider_id != engine.provider_id() ||
      info.provider_profile_digest != engine.profile_digest() ||
      !info.provider_descriptor.has_value() ||
      info.provider_descriptor->model().model_revision() !=
          identity.model_revision()) {
    return false;
  }
  return !info.provider_descriptor->kv().kv_namespace().empty() &&
         identity.kv_namespace() ==
             info.provider_descriptor->kv().kv_namespace();
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

xllm::proto::ExecutionHolder execution_holder(
    const xllm_service::Request& request) {
  if (request.provider_id == xllm::proto::PROVIDER_ID_VLLM_ASCEND) {
    return prefill_holder(request);
  }
  if (request.execution_mode ==
      xllm::proto::EXECUTION_MODE_LOCAL_PREFILL_DECODE) {
    return prefill_holder(request);
  }
  if (request.execution_mode == xllm::proto::EXECUTION_MODE_PREFILL_ONLY) {
    return xllm::proto::ExecutionHolder();
  }
  return decode_holder(request);
}

bool has_execution_holder(const xllm_service::Request& request) {
  const xllm::proto::ExecutionHolder holder = execution_holder(request);
  return !holder.engine_uid().empty() && !holder.incarnation_id().empty();
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
      readiness_controller_(readiness_config(options)),
      service_incarnation_id_(llm::new_uuid_v7()),
      execution_hold_cleanup_table_(
          std::make_unique<provider::ExecutionHoldCleanupTable>(
              execution_hold_config(options))),
      request_deadline_queue_(std::make_unique<RequestDeadlineQueue>(
          options.request_deadline_capacity())),
      flow_control_queue_(
          std::make_unique<FlowControlQueue>(flow_control_config(options))),
      client_disconnect_monitor_(std::make_shared<ClientDisconnectMonitor>(
          options.request_deadline_capacity())) {
  if (!readiness_controller_.valid() ||
      options_.readiness_check_interval_ms() == 0) {
    LOG(FATAL) << "Invalid readiness configuration.";
  }
  if ((options_.flow_order() != "FCFS" && options_.flow_order() != "EDF") ||
      (options_.flow_drain_policy() != "COMPLETE_QUEUED" &&
       options_.flow_drain_policy() != "RETRY_UNDISPATCHED") ||
      options_.flow_dispatch_interval_ms() <= 0 ||
      options_.native_local_prefill_bucket_permyriad() > 10000 ||
      options_.native_local_prefill_token_cap() == 0 ||
      options_.native_prefill_only_output_token_cap() == 0 ||
      !flow_control_queue_->valid()) {
    LOG(FATAL) << "Invalid V2 flow-control configuration.";
  }
  if (options_.observability_event_capacity() == 0 ||
      options_.observability_event_capacity() >
          kMaxObservabilityEventCapacity ||
      options_.observability_export_batch_size() == 0 ||
      options_.observability_export_batch_size() >
          options_.observability_event_capacity() ||
      options_.observability_export_interval_ms() <= 0 ||
      options_.observability_snapshot_interval_ms() <= 0 ||
      !valid_log_token(options_.observability_build_id())) {
    LOG(FATAL) << "Invalid V2 observability configuration.";
  }
  request_event_recorder_ =
      std::make_unique<observability::RequestEventRecorder>(
          options_.observability_event_capacity());
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
  // Provider selection happens per request. A Service without native model
  // assets can still relay HTTP providers; native requests fail closed below.
  if (!options_.tokenizer_path().empty()) {
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

  std::string observed_master_incarnation;
  if (!etcd_client_->get(ETCD_MASTER_SERVICE_KEY, nullptr)) {
    is_master_service_ = etcd_client_->elect_master(
        options_.service_name(), service_incarnation_id_, kHeartbeatInterval);
    LOG(INFO) << "Set current service as master!";
  } else {
    etcd_client_->get(ETCD_MASTER_SERVICE_INCARNATION_KEY,
                      &observed_master_incarnation);
  }

  instance_mgr_ = std::make_shared<InstanceMgr>(
      options, etcd_client_, is_master_service_, this);
  if (options_.state_stream_max_subscribers() == 0 ||
      options_.state_stream_full_interval_ms() <= 0 ||
      options_.state_stream_publish_interval_ms() <= 0 ||
      options_.state_stream_rpc_timeout_ms() <= 0) {
    LOG(FATAL) << "State Stream publisher configuration is invalid.";
  }
  state_stream_outbox_ = std::make_unique<provider::StateStreamOutbox>(
      provider::StateStreamOutboxConfig{
          .max_subscribers = options_.state_stream_max_subscribers(),
          .max_pending_engine_states = options_.engine_registry_max_members(),
          .max_pending_link_states = options_.engine_registry_max_links(),
      },
      service_incarnation_id_);
  if (options_.kv_state_max_subscribers() == 0 ||
      options_.kv_state_max_pending_batches() == 0 ||
      options_.kv_state_max_pending_events() == 0 ||
      options_.kv_state_max_pending_bytes() == 0 ||
      options_.kv_state_max_delivery_batches() == 0 ||
      options_.kv_state_max_delivery_bytes() == 0 ||
      options_.kv_state_publish_interval_ms() <= 0 ||
      options_.kv_state_rpc_timeout_ms() <= 0 ||
      options_.kv_shadow_max_engine_streams() == 0 ||
      options_.kv_shadow_max_entries() == 0 ||
      options_.kv_shadow_max_bytes() == 0 ||
      options_.kv_shadow_max_recovery_events() == 0 ||
      options_.kv_shadow_max_recovery_bytes() == 0 ||
      options_.kv_shadow_max_snapshot_entries() == 0 ||
      options_.kv_shadow_max_snapshot_bytes() == 0 ||
      options_.kv_shadow_event_ttl_ms() == 0 ||
      options_.kv_shadow_recovery_timeout_ms() == 0 ||
      options_.kv_snapshot_recovery_interval_ms() <= 0 ||
      options_.kv_snapshot_recovery_batch_size() == 0 ||
      options_.kv_snapshot_recovery_max_concurrency() == 0 ||
      options_.kv_snapshot_recovery_max_concurrency() >
          options_.kv_snapshot_recovery_batch_size() ||
      options_.kv_snapshot_max_pages_per_recovery() == 0 ||
      options_.kv_snapshot_page_entries() == 0 ||
      options_.kv_snapshot_page_bytes() == 0 ||
      options_.kv_snapshot_page_bytes() >
          std::numeric_limits<size_t>::max() - 4096 ||
      options_.kv_snapshot_page_generation_ms() == 0 ||
      options_.kv_snapshot_rpc_timeout_ms() <= 0) {
    LOG(FATAL) << "KV State publisher configuration is invalid.";
  }
  kv_shadow_index_ =
      std::make_unique<provider::KVShadowIndex>(kv_shadow_config(options_));
  kv_state_outbox_ = std::make_unique<provider::KVStateOutbox>(
      provider::KVStateOutboxConfig{
          .max_subscribers = options_.kv_state_max_subscribers(),
          .max_pending_batches_per_subscriber =
              options_.kv_state_max_pending_batches(),
          .max_pending_events_per_subscriber =
              options_.kv_state_max_pending_events(),
          .max_pending_bytes_per_subscriber =
              options_.kv_state_max_pending_bytes(),
          .max_delivery_batches = options_.kv_state_max_delivery_batches(),
          .max_delivery_bytes = options_.kv_state_max_delivery_bytes(),
      },
      service_incarnation_id_);
  kv_state_replica_ = std::make_unique<provider::KVStateReplica>(
      provider::KVStateReplicaConfig{
          .max_engine_batches = options_.kv_state_max_delivery_batches(),
          .max_serialized_bytes = options_.kv_state_max_delivery_bytes(),
      },
      kv_shadow_index_.get());
  if (!kv_state_replica_->set_master(is_master_service_
                                         ? service_incarnation_id_
                                         : observed_master_incarnation)) {
    LOG(FATAL) << "Failed to initialize KV State replica view.";
  }
  const provider::ContractResult state_registry_result =
      instance_mgr_->set_engine_state_master(is_master_service_
                                                 ? service_incarnation_id_
                                                 : observed_master_incarnation);
  if (!state_registry_result.ok()) {
    LOG(FATAL) << "Failed to initialize Engine State Registry view: "
               << state_registry_result.message();
  }

  global_kvcache_mgr_ = std::make_shared<GlobalKVCacheMgr>(
      options, etcd_client_, is_master_service_);

  if (options.load_balance_policy() == "CAR") {
    lb_policy_ = std::make_unique<CacheAwareRouting>(
        options_, instance_mgr_, kv_shadow_index_.get());
  } else if (options.load_balance_policy() == "SLO_AWARE") {
    lb_policy_ = std::make_unique<SloAwarePolicy>(options, instance_mgr_);
  } else {
    lb_policy_ = std::make_unique<RoundRobin>(instance_mgr_);
  }

  initialize_placement();

  auto handle_master = std::bind(&Scheduler::handle_master_service_watch,
                                 this,
                                 std::placeholders::_1,
                                 std::placeholders::_2);
  etcd_client_->add_watch(ETCD_MASTER_SERVICE_KEY, handle_master);
  auto handle_master_identity =
      std::bind(&Scheduler::handle_master_identity_watch,
                this,
                std::placeholders::_1,
                std::placeholders::_2);
  etcd_client_->add_watch(
      ETCD_MASTER_SERVICE_INCARNATION_KEY, handle_master_identity, false);
  if (is_master_service_) {
    activate_as_master();
  }

  state_stream_thread_ = std::make_unique<std::thread>(
      &Scheduler::run_state_stream_publisher, this);
  kv_state_thread_ =
      std::make_unique<std::thread>(&Scheduler::run_kv_state_publisher, this);
  kv_snapshot_thread_ =
      std::make_unique<std::thread>(&Scheduler::run_kv_snapshot_recovery, this);

  execution_hold_cleanup_thread_ = std::make_unique<std::thread>(
      &Scheduler::run_execution_hold_cleanup, this);
  request_watchdog_thread_ =
      std::make_unique<std::thread>(&Scheduler::run_request_watchdog, this);
  flow_dispatch_thread_ =
      std::make_unique<std::thread>(&Scheduler::run_flow_dispatch, this);
  observability_thread_ = std::make_unique<std::thread>(
      &Scheduler::run_observability_exporter, this);
  if (placement_config_.has_value()) {
    placement_thread_ = std::make_unique<std::thread>(
        &Scheduler::run_placement_controller, this);
  }
}

Scheduler::~Scheduler() {
  set_draining(true);
  refresh_readiness();
  exited_.store(true, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(placement_wait_mutex_);
    placement_stopped_ = true;
  }
  placement_cv_.notify_all();
  {
    std::lock_guard<std::mutex> lock(flow_dispatch_wait_mutex_);
    flow_dispatch_stopped_ = true;
  }
  flow_dispatch_cv_.notify_all();
  state_stream_cv_.notify_all();
  kv_state_cv_.notify_all();
  kv_snapshot_cv_.notify_all();
  if (placement_thread_ != nullptr && placement_thread_->joinable()) {
    placement_thread_->join();
  }
  etcd_client_->stop_watch();
  if (state_stream_thread_ != nullptr && state_stream_thread_->joinable()) {
    state_stream_thread_->join();
  }
  if (kv_state_thread_ != nullptr && kv_state_thread_->joinable()) {
    kv_state_thread_->join();
  }
  if (kv_snapshot_thread_ != nullptr && kv_snapshot_thread_->joinable()) {
    kv_snapshot_thread_->join();
  }
  if (heartbeat_thread_ != nullptr && heartbeat_thread_->joinable()) {
    heartbeat_thread_->join();
  }
  if (flow_dispatch_thread_ != nullptr && flow_dispatch_thread_->joinable()) {
    flow_dispatch_thread_->join();
  }
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
  // No retry worker remains after this point. Close unresolved release traces
  // explicitly instead of exporting a dangling STARTED event at shutdown.
  finish_resource_release_traces(/*fail_pending=*/true);
  {
    std::lock_guard<std::mutex> lock(observability_wait_mutex_);
    observability_stopped_ = true;
  }
  observability_cv_.notify_all();
  if (observability_thread_ != nullptr && observability_thread_->joinable()) {
    observability_thread_->join();
  }
}

bool Scheduler::schedule(std::shared_ptr<Request> request) {
  if (request == nullptr) {
    return false;
  }
  const bool traced_request = request->canonical_request.has_value();
  if (traced_request) {
    record_request_event(request,
                         xllm::proto::REQUEST_EVENT_TYPE_INGRESS,
                         xllm::proto::EVENT_RESULT_STARTED,
                         xllm::proto::ERROR_STAGE_NONE,
                         xllm::proto::EVENT_REASON_NONE);
  }
  const auto reject = [this, &request, traced_request](
                          xllm::proto::ErrorStage stage,
                          xllm::proto::EventReason reason) {
    if (traced_request) {
      record_request_terminal(
          request, xllm::proto::EVENT_RESULT_REJECTED, stage, reason);
    }
    return false;
  };
  if (!accepting_new_requests_.load(std::memory_order_acquire)) {
    LOG(WARNING) << "Reject request while Service is not ready, reason="
                 << provider::readiness_reason_name(readiness_status().reason);
    return reject(xllm::proto::ERROR_STAGE_INGRESS,
                  xllm::proto::EVENT_REASON_STALE_STATE);
  }
  if (request->request_deadline_present &&
      (!request->request_deadline.has_value() ||
       request->request_deadline->expired())) {
    LOG(ERROR) << "Request deadline is invalid or already expired.";
    return reject(xllm::proto::ERROR_STAGE_INGRESS,
                  xllm::proto::EVENT_REASON_DEADLINE_EXCEEDED);
  }
  if (!instance_mgr_->get_next_provider(request->model,
                                        &request->provider_id)) {
    return reject(xllm::proto::ERROR_STAGE_ROUTING,
                  xllm::proto::EVENT_REASON_STALE_STATE);
  }
  const std::optional<provider::ProviderDispatchKind> dispatch_kind =
      provider::resolve_provider_dispatch_kind(request->provider_id);
  if (!dispatch_kind.has_value()) {
    LOG(ERROR) << "Selected provider has no dispatch adapter: "
               << static_cast<int32_t>(request->provider_id);
    return reject(xllm::proto::ERROR_STAGE_ROUTING,
                  xllm::proto::EVENT_REASON_MODE_UNSUPPORTED);
  }
  const bool is_native =
      *dispatch_kind == provider::ProviderDispatchKind::XLLM_NATIVE_RPC;

  // apply chat template
  if (is_native && !request->messages.empty()) {
    if (chat_template_ == nullptr) {
      LOG(ERROR) << "Chat template has not configured.";
      return reject(xllm::proto::ERROR_STAGE_INGRESS,
                    xllm::proto::EVENT_REASON_PROVIDER_ERROR);
    }

    const std::vector<JsonTool> empty_tools;
    const std::vector<JsonTool>& tools_for_template =
        request->tool_choice == "none" ? empty_tools : request->tools;
    auto prompt = chat_template_->apply(
        request->messages, tools_for_template, request->chat_template_kwargs);
    if (!prompt.has_value()) {
      LOG(ERROR) << "Failed to construct prompt from messages, request_uid="
                 << request->correlation.request_uid()
                 << ", model=" << request->model;
      return reject(xllm::proto::ERROR_STAGE_INGRESS,
                    xllm::proto::EVENT_REASON_PROVIDER_ERROR);
    }
    request->prompt = prompt.value();
  }

  // encode prompt
  if (is_native && !request->prompt.empty()) {
    if (chat_template_ == nullptr || tokenizer_ == nullptr) {
      LOG(ERROR) << "Native provider tokenizer assets are not configured.";
      return reject(xllm::proto::ERROR_STAGE_INGRESS,
                    xllm::proto::EVENT_REASON_PROVIDER_ERROR);
    }
    if (!get_tls_tokenizer()->encode(
            request->prompt,
            &request->token_ids,
            chat_template_->encode_add_special_tokens())) {
      LOG(ERROR) << "Prompt encoding failed, request_uid="
                 << request->correlation.request_uid()
                 << ", model=" << request->model
                 << ", prompt_bytes=" << request->prompt.size();
      return reject(xllm::proto::ERROR_STAGE_INGRESS,
                    xllm::proto::EVENT_REASON_PROVIDER_ERROR);
    }
  }

  // Control-plane model discovery has no CanonicalRequest and remains a
  // direct bounded RPC. Inference requests defer route binding to dequeue.
  if (!request->canonical_request.has_value()) {
    return select_and_prepare_dispatch(request);
  }
  return true;
}

bool Scheduler::select_and_prepare_dispatch(
    const std::shared_ptr<Request>& request) {
  const auto reject_route = [&](xllm::proto::EventReason reason) {
    if (!request->trace_route_failure_recorded.exchange(
            true, std::memory_order_acq_rel)) {
      record_request_event(request,
                           xllm::proto::REQUEST_EVENT_TYPE_ROUTE,
                           xllm::proto::EVENT_RESULT_REJECTED,
                           xllm::proto::ERROR_STAGE_ROUTING,
                           reason,
                           "",
                           "");
    }
    return false;
  };
  request->routing = Routing();
  request->prefill_incarnation_id.clear();
  request->decode_incarnation_id.clear();
  request->prefill_provider_descriptor.reset();
  request->decode_provider_descriptor.reset();
  request->encoded_request.reset();
  request->execution_plan.reset();
  request->execution_mode = xllm::proto::EXECUTION_MODE_UNSPECIFIED;
  if (!lb_policy_->select_instances_pair(request)) {
    return reject_route(xllm::proto::EVENT_REASON_CAPACITY_EXHAUSTED);
  }

  if (!instance_mgr_->bind_request_instance_incarnations(request)) {
    LOG(ERROR) << "Failed to bind request to instance incarnation ids. "
               << request->routing.debug_string();
    return reject_route(xllm::proto::EVENT_REASON_STALE_STATE);
  }

  if (!apply_native_execution_mode(request)) {
    return reject_route(xllm::proto::EVENT_REASON_MODE_UNSUPPORTED);
  }
  if (!prepare_v2_execution_plan(request)) {
    return reject_route(xllm::proto::EVENT_REASON_PERMANENTLY_INFEASIBLE);
  }
  request->trace_route_failure_recorded.store(false, std::memory_order_release);
  record_kv_route_decision(request);
  DLOG(INFO) << request->routing.debug_string();

  const std::optional<provider::ProviderDispatchKind> dispatch_kind =
      provider::resolve_provider_dispatch_kind(request->provider_id);
  if (dispatch_kind.has_value() &&
      *dispatch_kind == provider::ProviderDispatchKind::XLLM_NATIVE_RPC &&
      !request->prompt.empty()) {
    instance_mgr_->update_request_metrics(request, RequestAction::SCHEDULE);
  }

  return true;
}

bool Scheduler::apply_native_execution_mode(
    const std::shared_ptr<Request>& request) {
  request->execution_mode = xllm::proto::EXECUTION_MODE_UNSPECIFIED;
  if (request->provider_id != xllm::proto::PROVIDER_ID_XLLM_NATIVE ||
      !request->canonical_request.has_value()) {
    return true;
  }
  const bool has_prefill = request->prefill_provider_descriptor.has_value();
  const bool has_decode = request->decode_provider_descriptor.has_value();
  if (!has_prefill && !has_decode) {
    return true;
  }
  if (!has_prefill || !has_decode) {
    LOG(ERROR) << "Native strict route must bind both P and D before mode "
                  "selection, routing="
               << request->routing.debug_string();
    return false;
  }

  provider::NativeExecutionModeDecision decision;
  const provider::ContractResult result =
      provider::select_native_execution_mode(
          provider::NativeExecutionModeConfig{
              .local_prefill_decode_enabled =
                  options_.native_local_prefill_enabled(),
              .local_prefill_decode_bucket_permyriad =
                  options_.native_local_prefill_bucket_permyriad(),
              .local_prefill_decode_prompt_token_cap =
                  options_.native_local_prefill_token_cap(),
              .prefill_only_enabled = options_.native_prefill_only_enabled(),
              .prefill_only_output_token_cap =
                  options_.native_prefill_only_output_token_cap(),
          },
          provider::NativeExecutionModeInput{
              .stable_request_hash =
                  stable_identity_hash(request->correlation.request_uid()),
              .prompt_tokens = request->token_ids.size(),
              .output_tokens =
                  request->canonical_request->effective_max_new_tokens(),
              .prefill = &*request->prefill_provider_descriptor,
              .decode = &*request->decode_provider_descriptor,
          },
          &decision);
  if (!result.ok()) {
    LOG(ERROR) << "Native execution mode selection failed: "
               << result.message();
    return false;
  }

  request->execution_mode = decision.mode;
  if (decision.mode == xllm::proto::EXECUTION_MODE_PREFILL_ONLY) {
    request->routing.decode_name.clear();
    request->decode_incarnation_id.clear();
    request->decode_provider_descriptor.reset();
  } else if (decision.mode ==
             xllm::proto::EXECUTION_MODE_LOCAL_PREFILL_DECODE) {
    request->routing.prefill_name = request->routing.decode_name;
    request->prefill_incarnation_id = request->decode_incarnation_id;
    request->prefill_provider_descriptor = request->decode_provider_descriptor;
    request->routing.decode_name.clear();
    request->decode_incarnation_id.clear();
    request->decode_provider_descriptor.reset();
  }
  return true;
}

SaturationState Scheduler::flow_saturation_state() const {
  const provider::ReadinessSnapshot readiness = readiness_status();
  const bool has_capacity = has_available_instances();
  if (readiness.reason == provider::ReadinessReason::READY) {
    return has_capacity ? SaturationState::AVAILABLE
                        : SaturationState::SATURATED;
  }
  if (readiness.reason == provider::ReadinessReason::DRAINING && has_capacity) {
    return SaturationState::AVAILABLE;
  }
  return SaturationState::UNKNOWN;
}

bool Scheduler::admit_flow_control_locked(
    const std::shared_ptr<Request>& request) {
  if (request->dispatch_callback == nullptr ||
      !request->request_deadline.has_value() ||
      !request->canonical_request.has_value() ||
      request->queue_state.load(std::memory_order_acquire) !=
          RequestQueueState::RECEIVED) {
    request->admission_status = FlowControlStatus::INVALID_ARGUMENT;
    return false;
  }

  const xllm::proto::CanonicalRequest& canonical = *request->canonical_request;
  const uint64_t token_count = request->token_ids.empty()
                                   ? canonical.canonical_payload().size()
                                   : request->token_ids.size();
  const uint64_t token_bytes =
      token_count > std::numeric_limits<uint64_t>::max() / sizeof(int32_t)
          ? std::numeric_limits<uint64_t>::max()
          : token_count * sizeof(int32_t);
  const uint64_t base_bytes =
      static_cast<uint64_t>(sizeof(Request)) + canonical.ByteSizeLong();
  const uint64_t request_bytes = uint64_add_overflows(base_bytes, token_bytes)
                                     ? std::numeric_limits<uint64_t>::max()
                                     : base_bytes + token_bytes;
  int32_t priority = canonical.priority();
  if (priority == static_cast<int32_t>(xllm::proto::DEFAULT)) {
    priority = static_cast<int32_t>(xllm::proto::NORMAL);
  }
  FlowControlWork work{
      .request_uid = request->correlation.request_uid(),
      .model_pool = request->model,
      .tenant_id = request->tenant_id,
      .flow_id = request->flow_id,
      .priority_band = priority,
      .prompt_tokens = token_count,
      .request_bytes = request_bytes,
      .deadline = request->request_deadline->time_point(),
      .strict = canonical.strict(),
  };
  const FlowControlAdmission admission = flow_control_queue_->admit(
      work, FlowControlQueue::Clock::now(), flow_saturation_state());
  request->admission_status = admission.status;
  if (admission.status != FlowControlStatus::OK) {
    if (placement_observation_collector_ != nullptr) {
      const placement::PlacementObservationStatus observation_status =
          placement_observation_collector_->record_admission_reject(
              request->model, monotonic_time_ms());
      MULTI_COUNTER_INC(
          xllm_service_v3_placement_observations_total,
          placement::placement_observation_status_name(observation_status));
    }
    const xllm::proto::EventReason reason =
        event_reason_for_flow_status(admission.status);
    record_request_event(
        request,
        xllm::proto::REQUEST_EVENT_TYPE_PREFILL_QUEUE,
        admission.status == FlowControlStatus::QUEUE_DEADLINE_UNSATISFIABLE
            ? xllm::proto::EVENT_RESULT_DEADLINE_EXCEEDED
            : xllm::proto::EVENT_RESULT_REJECTED,
        xllm::proto::ERROR_STAGE_INGRESS,
        reason);
    record_request_terminal(
        request,
        admission.status == FlowControlStatus::QUEUE_DEADLINE_UNSATISFIABLE
            ? xllm::proto::EVENT_RESULT_DEADLINE_EXCEEDED
            : xllm::proto::EVENT_RESULT_REJECTED,
        xllm::proto::ERROR_STAGE_INGRESS,
        reason);
    return false;
  }
  RequestQueueState expected = RequestQueueState::RECEIVED;
  if (!request->queue_state.compare_exchange_strong(
          expected,
          RequestQueueState::QUEUED,
          std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    flow_control_queue_->cancel(request->correlation.request_uid());
    request->admission_status = FlowControlStatus::INVALID_ARGUMENT;
    record_request_terminal(request,
                            xllm::proto::EVENT_RESULT_REJECTED,
                            xllm::proto::ERROR_STAGE_INGRESS,
                            xllm::proto::EVENT_REASON_INTERNAL_ERROR);
    return false;
  }
  request->enqueue_time = FlowControlQueue::Clock::now();
  if (placement_observation_collector_ != nullptr) {
    const placement::PlacementObservationStatus observation_status =
        placement_observation_collector_->record_ingress(
            request->model,
            token_count,
            canonical.effective_max_new_tokens(),
            monotonic_time_ms());
    MULTI_COUNTER_INC(
        xllm_service_v3_placement_observations_total,
        placement::placement_observation_status_name(observation_status));
  }
  record_request_event(request,
                       xllm::proto::REQUEST_EVENT_TYPE_PREFILL_QUEUE,
                       xllm::proto::EVENT_RESULT_ACCEPTED,
                       xllm::proto::ERROR_STAGE_NONE,
                       xllm::proto::EVENT_REASON_NONE,
                       "",
                       "",
                       elapsed_ns_since(request->trace_ingress_time));
  return true;
}

FlowControlSnapshot Scheduler::flow_control_snapshot() const {
  return flow_control_queue_->snapshot();
}

bool Scheduler::prepare_v2_execution_plan(
    const std::shared_ptr<Request>& request) {
  request->encoded_request.reset();
  request->execution_plan.reset();
  if (!request->canonical_request.has_value()) {
    return true;
  }

  // A route is either wholly STRICT (immutable descriptors on every selected
  // role) or wholly BEST_EFFORT legacy. The route selector already rejects a
  // one-sided STRICT P/D pair; preserve that invariant at the codec boundary.
  if (!request->prefill_provider_descriptor.has_value()) {
    if (request->decode_provider_descriptor.has_value()) {
      LOG(ERROR) << "Selected route mixes strict and legacy descriptors.";
      return false;
    }
    return true;
  }
  if (!request->request_deadline.has_value()) {
    LOG(ERROR) << "V2 execution plan has no request deadline.";
    return false;
  }
  const uint64_t remaining_deadline_ms =
      request->request_deadline->remaining_ms();
  if (remaining_deadline_ms == 0 || !request->correlation.has_attempt_seq()) {
    LOG(ERROR) << "V2 execution plan deadline or attempt is invalid.";
    return false;
  }

  xllm::proto::CanonicalRequest canonical = *request->canonical_request;
  canonical.set_attempt_seq(request->correlation.attempt_seq());
  canonical.set_remaining_deadline_ms(remaining_deadline_ms);
  const xllm::proto::ProviderDescriptor& primary =
      *request->prefill_provider_descriptor;
  if (primary.identity().provider_id() != request->provider_id) {
    LOG(ERROR) << "Bound Provider Descriptor changed before encoding.";
    return false;
  }

  const provider::ProviderAdapter* adapter = nullptr;
  const provider::ContractResult find_result =
      provider_adapter_registry_.find_compatible_adapter(primary, &adapter);
  if (!find_result.ok()) {
    LOG(ERROR) << "V2 Provider Adapter lookup failed: "
               << find_result.message();
    return false;
  }
  if (adapter == nullptr) {
    std::unique_ptr<provider::ProviderAdapter> candidate;
    const provider::ContractResult create_result =
        provider::create_provider_adapter(primary, &candidate);
    if (!create_result.ok()) {
      LOG(ERROR) << "V2 Provider Adapter creation failed: "
                 << create_result.message();
      return false;
    }
    const provider::ContractResult register_result =
        provider_adapter_registry_.find_or_register_adapter(
            std::move(candidate), &adapter);
    if (!register_result.ok()) {
      LOG(ERROR) << "V2 Provider Adapter registration failed: "
                 << register_result.message();
      return false;
    }
  }

  provider::RequestEncodingContext encoding_context;
  if (request->provider_id == xllm::proto::PROVIDER_ID_XLLM_NATIVE) {
    encoding_context.native_token_ids = &request->token_ids;
    encoding_context.request_uid = request->correlation.request_uid();
    encoding_context.attempt_seq = request->correlation.attempt_seq();
    encoding_context.native_renderer_digest = options_.native_renderer_digest();
  }

  xllm::proto::EncodedRequest encoded;
  const provider::ContractResult encoded_result =
      adapter->request_codec().encode(canonical, encoding_context, &encoded);
  if (!encoded_result.ok()) {
    LOG(ERROR) << "V2 Provider request encoding failed: "
               << encoded_result.message();
    return false;
  }
  const xllm::proto::ProviderDescriptor* decode =
      request->decode_provider_descriptor.has_value()
          ? &request->decode_provider_descriptor.value()
          : nullptr;
  xllm::proto::ExecutionPlan plan;
  const provider::ContractResult plan_result = provider::build_execution_plan(
      canonical, encoded, primary, decode, &plan, request->execution_mode);
  if (!plan_result.ok()) {
    LOG(ERROR) << "V2 ExecutionPlan construction failed: "
               << plan_result.message();
    return false;
  }

  request->canonical_request = std::move(canonical);
  request->encoded_request = std::move(encoded);
  request->execution_plan = std::move(plan);
  if (request->execution_mode ==
      xllm::proto::EXECUTION_MODE_LOCAL_PREFILL_DECODE) {
    request->execution_plan->add_reason_codes("native-local-prefill-allowlist");
  } else if (request->execution_mode ==
             xllm::proto::EXECUTION_MODE_PREFILL_ONLY) {
    request->execution_plan->add_reason_codes("native-prefill-only");
  }
  return true;
}

std::shared_ptr<brpc::Channel> Scheduler::get_channel(
    const std::string& target_name) {
  return instance_mgr_->get_channel(target_name);
}

void Scheduler::activate_as_master() {
  const bool was_master =
      is_master_service_.exchange(true, std::memory_order_acq_rel);
  if (!was_master) {
    global_kvcache_mgr_->set_as_master();
    instance_mgr_->set_as_master();
  }
  instance_mgr_->require_provider_link_recheck();
  const provider::ContractResult state_registry_result =
      instance_mgr_->set_engine_state_master(service_incarnation_id_);
  if (!state_registry_result.ok()) {
    LOG(ERROR) << "Failed to activate Engine State Registry master: "
               << state_registry_result.message();
  }
  if (heartbeat_thread_ == nullptr) {
    heartbeat_thread_ = std::make_unique<std::thread>(
        &Scheduler::update_master_service_heartbeat, this);
  }
  state_stream_outbox_->require_full_for_all();
  state_stream_cv_.notify_all();
  if (!kv_state_replica_->set_master(service_incarnation_id_)) {
    LOG(ERROR) << "Failed to fence KV State on master activation.";
  }
  kv_state_cv_.notify_all();
  placement_cv_.notify_all();
}

void Scheduler::deactivate_as_master() {
  const bool was_master =
      is_master_service_.exchange(false, std::memory_order_acq_rel);
  if (!was_master) {
    return;
  }
  global_kvcache_mgr_->set_as_follower();
  instance_mgr_->set_as_follower();
  if (!kv_state_replica_->set_master("")) {
    LOG(ERROR) << "Failed to fence KV State on master deactivation.";
  }
  placement_cv_.notify_all();
}

bool Scheduler::refresh_state_stream_subscribers() {
  std::unordered_map<std::string, std::string> service_members;
  if (!etcd_client_->get_prefix(ETCD_XSERVICE_KEY_PREFIX, &service_members)) {
    return false;
  }
  std::vector<std::string> subscribers;
  const provider::ContractResult resolved =
      provider::resolve_state_stream_subscribers(
          service_members,
          options_.service_name(),
          options_.state_stream_max_subscribers(),
          &subscribers);
  if (!resolved.ok()) {
    LOG(ERROR) << "Failed to resolve State Stream subscribers: "
               << resolved.message();
    return false;
  }
  const provider::ContractResult replaced =
      state_stream_outbox_->replace_subscribers(subscribers);
  if (!replaced.ok()) {
    LOG(ERROR) << "Failed to update State Stream subscribers: "
               << replaced.message();
    return false;
  }
  return true;
}

bool Scheduler::refresh_kv_state_subscribers() {
  std::unordered_map<std::string, std::string> service_members;
  if (!etcd_client_->get_prefix(ETCD_XSERVICE_KEY_PREFIX, &service_members)) {
    return false;
  }
  std::vector<std::string> subscribers;
  const provider::ContractResult resolved =
      provider::resolve_state_stream_subscribers(
          service_members,
          options_.service_name(),
          options_.kv_state_max_subscribers(),
          &subscribers);
  if (!resolved.ok()) {
    LOG(ERROR) << "Failed to resolve KV State subscribers: "
               << resolved.message();
    return false;
  }
  const provider::ContractResult replaced =
      kv_state_outbox_->replace_subscribers(subscribers);
  if (!replaced.ok()) {
    LOG(ERROR) << "Failed to update KV State subscribers: "
               << replaced.message();
    return false;
  }
  return true;
}

void Scheduler::try_apply_local_full_state(uint64_t now_monotonic_ms) {
  if (!is_master_service_.load(std::memory_order_acquire)) {
    return;
  }
  if (state_stream_outbox_ == nullptr) {
    return;
  }
  if (instance_mgr_->has_current_engine_state_full_snapshot()) {
    return;
  }
  const uint64_t snapshot_seq =
      next_state_stream_snapshot_seq_.fetch_add(1, std::memory_order_relaxed);
  if (snapshot_seq == 0) {
    LOG(FATAL) << "State Stream snapshot sequence exhausted.";
  }
  xllm::proto::StateBatch full;
  const provider::ContractResult built = instance_mgr_->build_full_state_batch(
      service_incarnation_id_, snapshot_seq, now_monotonic_ms, &full);
  if (!built.ok()) {
    return;
  }
  bool applied = false;
  const provider::ContractResult result =
      instance_mgr_->apply_engine_state_batch(full, now_monotonic_ms, &applied);
  if (!result.ok()) {
    LOG(ERROR) << "Failed to activate local State Stream FULL: "
               << result.message();
  }
}

void Scheduler::notify_engine_registry_membership_changed() {
  if (state_stream_outbox_ == nullptr) {
    return;
  }
  state_stream_outbox_->require_full_for_all();
  try_apply_local_full_state(monotonic_time_ms());
  state_stream_cv_.notify_all();
}

void Scheduler::notify_engine_link_state_changed(
    const xllm::proto::LinkState& state,
    uint64_t received_monotonic_ms) {
  if (!is_master_service_.load(std::memory_order_acquire)) {
    return;
  }
  if (state_stream_outbox_ == nullptr) {
    return;
  }
  xllm::proto::StateBatch delta;
  delta.set_contract_version(provider::kProviderContractVersion);
  delta.set_master_incarnation(service_incarnation_id_);
  delta.set_snapshot_seq(1);
  delta.set_kind(xllm::proto::STATE_BATCH_KIND_DELTA);
  *delta.add_link_states() = state;
  const provider::ContractResult queued =
      state_stream_outbox_->enqueue_delta(delta, received_monotonic_ms);
  if (!queued.ok()) {
    LOG(ERROR) << "Failed to enqueue Provider Link state: " << queued.message();
    return;
  }
  try_apply_local_full_state(received_monotonic_ms);
  state_stream_cv_.notify_all();
}

void Scheduler::run_state_stream_publisher() {
  uint64_t last_subscriber_refresh_ms = 0;
  uint64_t last_periodic_full_ms = 0;
  while (!exited_.load(std::memory_order_acquire)) {
    {
      std::unique_lock lock(state_stream_wait_mutex_);
      state_stream_cv_.wait_for(
          lock,
          std::chrono::milliseconds(
              options_.state_stream_publish_interval_ms()),
          [this]() { return exited_.load(std::memory_order_acquire); });
    }
    if (exited_.load(std::memory_order_acquire) ||
        !is_master_service_.load(std::memory_order_acquire)) {
      continue;
    }

    const uint64_t now_monotonic_ms = monotonic_time_ms();
    if (last_subscriber_refresh_ms == 0 ||
        now_monotonic_ms - last_subscriber_refresh_ms >= 1000) {
      std::string master_address;
      std::string master_incarnation;
      const bool still_master =
          etcd_client_->get(ETCD_MASTER_SERVICE_KEY, &master_address) &&
          etcd_client_->get(ETCD_MASTER_SERVICE_INCARNATION_KEY,
                            &master_incarnation) &&
          master_address == options_.service_name() &&
          master_incarnation == service_incarnation_id_;
      if (!still_master) {
        deactivate_as_master();
        instance_mgr_->set_engine_state_master(master_incarnation);
        continue;
      }
      refresh_state_stream_subscribers();
      last_subscriber_refresh_ms = now_monotonic_ms;
    }
    if (last_periodic_full_ms == 0 ||
        now_monotonic_ms - last_periodic_full_ms >=
            static_cast<uint64_t>(options_.state_stream_full_interval_ms())) {
      state_stream_outbox_->require_full_for_all();
      last_periodic_full_ms = now_monotonic_ms;
    }

    const std::vector<std::string> ready =
        state_stream_outbox_->ready_subscribers();
    if (ready.empty()) {
      continue;
    }
    xllm::proto::StateBatch authoritative_full;
    const provider::ContractResult full_result =
        instance_mgr_->build_full_state_batch(
            service_incarnation_id_, 1, now_monotonic_ms, &authoritative_full);

    std::vector<StateStreamPush> pushes;
    pushes.reserve(ready.size());
    for (const std::string& subscriber : ready) {
      StateStreamPush push;
      push.subscriber = subscriber;
      push.timeout_ms =
          static_cast<uint64_t>(options_.state_stream_rpc_timeout_ms());
      const uint64_t snapshot_seq = next_state_stream_snapshot_seq_.fetch_add(
          1, std::memory_order_relaxed);
      if (snapshot_seq == 0) {
        LOG(FATAL) << "State Stream snapshot sequence exhausted.";
      }
      if (!state_stream_outbox_->begin_delivery(subscriber,
                                                authoritative_full,
                                                snapshot_seq,
                                                now_monotonic_ms,
                                                &push.batch)) {
        continue;
      }
      pushes.emplace_back(std::move(push));
    }

    const std::vector<StateStreamPushResult> results =
        push_state_stream_batches(pushes);
    for (size_t index = 0; index < results.size(); ++index) {
      state_stream_outbox_->complete_delivery(pushes[index].subscriber,
                                              results[index].ok);
      if (!results[index].ok) {
        LOG(WARNING) << "State Stream push failed for "
                     << pushes[index].subscriber << ": "
                     << results[index].message;
      }
    }
    static_cast<void>(full_result);
  }
}

void Scheduler::run_kv_state_publisher() {
  uint64_t last_subscriber_refresh_ms = 0;
  while (!exited_.load(std::memory_order_acquire)) {
    {
      std::unique_lock<std::mutex> lock(kv_state_wait_mutex_);
      kv_state_cv_.wait_for(
          lock,
          std::chrono::milliseconds(options_.kv_state_publish_interval_ms()),
          [this]() { return exited_.load(std::memory_order_acquire); });
    }
    if (exited_.load(std::memory_order_acquire) ||
        !is_master_service_.load(std::memory_order_acquire)) {
      continue;
    }

    const uint64_t now_monotonic_ms = monotonic_time_ms();
    if (last_subscriber_refresh_ms == 0 ||
        now_monotonic_ms < last_subscriber_refresh_ms ||
        now_monotonic_ms - last_subscriber_refresh_ms >= 1000) {
      std::string master_address;
      std::string master_incarnation;
      const bool still_master =
          etcd_client_->get(ETCD_MASTER_SERVICE_KEY, &master_address) &&
          etcd_client_->get(ETCD_MASTER_SERVICE_INCARNATION_KEY,
                            &master_incarnation) &&
          master_address == options_.service_name() &&
          master_incarnation == service_incarnation_id_;
      if (!still_master) {
        deactivate_as_master();
        instance_mgr_->set_engine_state_master(master_incarnation);
        kv_state_replica_->set_master(master_incarnation);
        continue;
      }
      refresh_kv_state_subscribers();
      last_subscriber_refresh_ms = now_monotonic_ms;
    }

    const std::vector<std::string> ready =
        kv_state_outbox_->ready_subscribers();
    std::vector<KVStateStreamPush> pushes;
    pushes.reserve(ready.size());
    for (const std::string& subscriber : ready) {
      KVStateStreamPush push;
      push.subscriber = subscriber;
      push.timeout_ms =
          static_cast<uint64_t>(options_.kv_state_rpc_timeout_ms());
      if (kv_state_outbox_->begin_delivery(
              subscriber, now_monotonic_ms, &push.batch)) {
        pushes.emplace_back(std::move(push));
      }
    }
    const std::vector<KVStateStreamPushResult> results =
        push_kv_state_stream_batches(pushes);
    for (size_t index = 0; index < results.size(); ++index) {
      kv_state_outbox_->complete_delivery(pushes[index].subscriber,
                                          results[index].ok);
      if (!results[index].ok) {
        LOG(WARNING) << "KV State Stream push failed for "
                     << pushes[index].subscriber << ": "
                     << results[index].message;
      }
    }
  }
}

bool Scheduler::recover_kv_snapshot(
    const xllm::proto::KVStreamIdentity& identity) {
  const InstanceMetaInfo info =
      instance_mgr_->get_instance_info(identity.engine().engine_uid());
  if (!matches_current_kv_engine(info, identity)) {
    return false;
  }
  const std::shared_ptr<brpc::Channel> channel =
      instance_mgr_->get_channel(identity.engine().engine_uid());
  if (channel == nullptr) {
    return false;
  }
  const uint64_t started_ms = monotonic_time_ms();
  if (kv_shadow_index_->begin_recovery(identity, started_ms).code !=
      provider::KVApplyCode::RECOVERING) {
    return false;
  }

  xllm::proto::KVCacheSnapshotRequest request;
  request.set_contract_version(provider::kProviderContractVersion);
  *request.mutable_identity() = identity;
  request.set_max_entries(options_.kv_snapshot_page_entries());
  request.set_max_bytes(options_.kv_snapshot_page_bytes());
  request.set_max_generation_time_ms(options_.kv_snapshot_page_generation_ms());

  size_t total_entries = 0;
  size_t total_bytes = 0;
  const size_t max_response_bytes =
      static_cast<size_t>(options_.kv_snapshot_page_bytes()) + 4096;
  for (size_t page_index = 0;
       page_index < options_.kv_snapshot_max_pages_per_recovery();
       ++page_index) {
    if (exited_.load(std::memory_order_acquire)) {
      kv_shadow_index_->abort_recovery(identity);
      return false;
    }
    KVSnapshotPageQuery query;
    query.channel = channel;
    query.request = request;
    query.max_response_bytes = max_response_bytes;
    query.timeout_ms =
        static_cast<uint64_t>(options_.kv_snapshot_rpc_timeout_ms());
    const std::vector<KVSnapshotPageResult> queried =
        query_kv_snapshot_pages({query});
    if (queried.size() != 1 || !queried[0].ok || !queried[0].page.has_value()) {
      kv_shadow_index_->abort_recovery(identity);
      return false;
    }
    const xllm::proto::KVCacheSnapshotPage& page = *queried[0].page;
    if (page.status() != xllm::proto::KV_SNAPSHOT_STATUS_OK ||
        static_cast<size_t>(page.entries_size()) >
            options_.kv_shadow_max_snapshot_entries() - total_entries ||
        page.serialized_bytes() >
            options_.kv_shadow_max_snapshot_bytes() - total_bytes) {
      kv_shadow_index_->abort_recovery(identity);
      return false;
    }
    total_entries += static_cast<size_t>(page.entries_size());
    total_bytes += static_cast<size_t>(page.serialized_bytes());
    const provider::KVApplyResult applied =
        kv_shadow_index_->apply_snapshot_page(page, monotonic_time_ms());
    if (applied.code == provider::KVApplyCode::REJECTED ||
        applied.code == provider::KVApplyCode::SNAPSHOT_REQUIRED) {
      kv_shadow_index_->abort_recovery(identity);
      return false;
    }
    if (page.done()) {
      return applied.code == provider::KVApplyCode::APPLIED;
    }
    if (page.next_cursor() <= request.cursor()) {
      kv_shadow_index_->abort_recovery(identity);
      return false;
    }
    request.set_snapshot_id(page.snapshot_id());
    request.set_cursor(page.next_cursor());
  }
  kv_shadow_index_->abort_recovery(identity);
  return false;
}

void Scheduler::run_kv_snapshot_recovery() {
  std::unordered_map<std::string, uint64_t> retry_after_ms;
  size_t round_robin_cursor = 0;
  while (!exited_.load(std::memory_order_acquire)) {
    {
      std::unique_lock<std::mutex> lock(kv_snapshot_wait_mutex_);
      kv_snapshot_cv_.wait_for(
          lock,
          std::chrono::milliseconds(
              options_.kv_snapshot_recovery_interval_ms()),
          [this]() { return exited_.load(std::memory_order_acquire); });
    }
    if (exited_.load(std::memory_order_acquire)) {
      continue;
    }
    const uint64_t now_monotonic_ms = monotonic_time_ms();
    kv_shadow_index_->expire(now_monotonic_ms);
    const std::vector<xllm::proto::KVStreamIdentity> identities =
        kv_shadow_index_->snapshot_required();
    if (identities.empty()) {
      retry_after_ms.clear();
      round_robin_cursor = 0;
      continue;
    }
    std::unordered_set<std::string> current_keys;
    current_keys.reserve(identities.size());
    for (const xllm::proto::KVStreamIdentity& identity : identities) {
      current_keys.emplace(identity.SerializeAsString());
    }
    for (auto retry = retry_after_ms.begin(); retry != retry_after_ms.end();) {
      if (current_keys.find(retry->first) == current_keys.end()) {
        retry = retry_after_ms.erase(retry);
      } else {
        ++retry;
      }
    }

    std::vector<xllm::proto::KVStreamIdentity> selected;
    selected.reserve(options_.kv_snapshot_recovery_batch_size());
    const size_t start = round_robin_cursor % identities.size();
    for (size_t offset = 0;
         offset < identities.size() &&
         selected.size() < options_.kv_snapshot_recovery_batch_size();
         ++offset) {
      const xllm::proto::KVStreamIdentity& identity =
          identities[(start + offset) % identities.size()];
      const std::string key = identity.SerializeAsString();
      const auto retry = retry_after_ms.find(key);
      if (retry != retry_after_ms.end() && now_monotonic_ms < retry->second) {
        continue;
      }
      selected.emplace_back(identity);
    }
    round_robin_cursor =
        (start + std::max<size_t>(selected.size(), 1)) % identities.size();

    const size_t concurrency = options_.kv_snapshot_recovery_max_concurrency();
    for (size_t batch_begin = 0; batch_begin < selected.size();
         batch_begin += concurrency) {
      const size_t batch_end =
          std::min(selected.size(), batch_begin + concurrency);
      std::vector<std::future<bool>> recoveries;
      recoveries.reserve(batch_end - batch_begin);
      for (size_t index = batch_begin; index < batch_end; ++index) {
        recoveries.emplace_back(std::async(
            std::launch::async, [this, identity = selected[index]]() {
              return recover_kv_snapshot(identity);
            }));
      }
      for (size_t index = batch_begin; index < batch_end; ++index) {
        const std::string key = selected[index].SerializeAsString();
        if (recoveries[index - batch_begin].get()) {
          retry_after_ms.erase(key);
          continue;
        }
        const size_t jitter_bucket = std::hash<std::string>{}(key) % 3;
        retry_after_ms.insert_or_assign(
            key,
            now_monotonic_ms +
                static_cast<uint64_t>(
                    options_.kv_snapshot_recovery_interval_ms()) *
                    (3 + jitter_bucket));
      }
    }
  }
}

void Scheduler::initialize_placement() {
  GAUGE_SET(xllm_service_v3_placement_leader, 0);
  GAUGE_SET(xllm_service_v3_placement_mode,
            static_cast<int32_t>(placement::PlacementMode::DISABLED));
  GAUGE_SET(xllm_service_v3_placement_pools, 0);
  GAUGE_SET(xllm_service_v3_placement_operations, 0);
  if (options_.placement_config_path().empty()) {
    LOG(INFO) << "V3 Placement is disabled because no config path was set.";
    return;
  }

  placement::PlacementRuntimeConfig config;
  std::string error;
  const placement::PlacementConfigStatus status =
      placement::load_placement_runtime_config(
          options_.placement_config_path(), &config, &error);
  if (status != placement::PlacementConfigStatus::OK) {
    LOG(FATAL) << "Failed to load V3 Placement config, status="
               << placement::placement_config_status_name(status)
               << ", error=" << error;
  }
  placement_config_ = std::move(config);
  placement_observation_collector_ =
      std::make_unique<placement::PlacementObservationCollector>(
          placement_config_->observation);
  placement_input_builder_ = std::make_unique<placement::PlacementInputBuilder>(
      placement_config_->input_builder, kv_shadow_index_.get());
  const uint64_t now_ms = monotonic_time_ms();
  std::unordered_set<std::string> registered_models;
  for (const placement::PlacementPoolRuntimeSpec& pool :
       placement_config_->pools) {
    if (!registered_models.insert(pool.profile.pool.model_revision).second) {
      continue;
    }
    const placement::PlacementObservationStatus register_status =
        placement_observation_collector_->register_model(
            pool.profile.pool.model_revision, now_ms);
    if (register_status != placement::PlacementObservationStatus::OK) {
      LOG(FATAL) << "Failed to register Placement model observation, status="
                 << placement::placement_observation_status_name(
                        register_status);
    }
  }

  placement_fenced_kv_ =
      std::make_unique<placement::EtcdPlacementFencedKv>(etcd_client_.get());
  placement_desired_store_ = std::make_unique<placement::PlacementDesiredStore>(
      placement_fenced_kv_.get());
  placement_operation_store_ =
      std::make_unique<placement::PlacementOperationStore>(
          placement_fenced_kv_.get());
  placement_native_transport_ =
      std::make_unique<placement::BrpcProviderLifecycleTransport>(
          placement_config_->transports.native);
  placement_vllm_transport_ =
      std::make_unique<placement::HttpProviderLifecycleTransport>(
          placement_config_->transports.vllm_ascend);
  placement_transport_router_ =
      std::make_unique<placement::ProviderLifecycleTransportRouter>(
          placement_native_transport_.get(), placement_vllm_transport_.get());
  placement_deployment_http_ =
      std::make_unique<placement::HttpPlacementDeploymentActuator>(
          placement_config_->deployment);
  placement_deployment_verified_ =
      std::make_unique<placement::RegistryVerifiedPlacementDeploymentActuator>(
          instance_mgr_->mutable_engine_registry(),
          placement_deployment_http_.get(),
          monotonic_time_ms);
  placement_actuator_ = std::make_unique<placement::ProviderPlacementActuator>(
      instance_mgr_->mutable_engine_registry(),
      placement_transport_router_.get(),
      placement_deployment_verified_.get());
  placement_operation_executor_ =
      std::make_unique<placement::PlacementOperationExecutor>(
          placement_config_->executor,
          placement_actuator_.get(),
          placement_operation_store_.get());
  placement_controller_ = std::make_unique<placement::PlacementController>(
      placement_config_->controller,
      placement_desired_store_.get(),
      placement_operation_executor_.get());
  if (FLAGS_placement_mode_override >= 0 &&
      !placement_controller_->set_mode(static_cast<placement::PlacementMode>(
          FLAGS_placement_mode_override))) {
    LOG(FATAL) << "Invalid V3 Placement mode override: "
               << FLAGS_placement_mode_override;
  }

  GAUGE_SET(xllm_service_v3_placement_mode,
            static_cast<int32_t>(placement_controller_->mode()));
  GAUGE_SET(xllm_service_v3_placement_pools, placement_config_->pools.size());
  LOG(INFO) << "Initialized V3 Placement, mode="
            << placement::placement_mode_name(placement_controller_->mode())
            << ", pools=" << placement_config_->pools.size()
            << ", loop_interval_ms=" << placement_config_->loop_interval_ms;
}

void Scheduler::run_placement_controller() {
  std::optional<placement::PlacementLeaderIdentity> active_leader;
  while (!exited_.load(std::memory_order_acquire)) {
    const auto cycle_started = std::chrono::steady_clock::now();
    bool cycle_attempted = false;
    if (!is_master_service_.load(std::memory_order_acquire)) {
      active_leader.reset();
      GAUGE_SET(xllm_service_v3_placement_leader, 0);
      MULTI_COUNTER_INC(xllm_service_v3_placement_cycles_total, "FOLLOWER");
    } else {
      std::string address;
      std::string incarnation;
      uint64_t epoch = 0;
      const EtcdReadStatus identity_status =
          etcd_client_->get_master_identity(&address, &incarnation, &epoch);
      if (identity_status != EtcdReadStatus::OK ||
          address != options_.service_name() ||
          incarnation != service_incarnation_id_ || epoch == 0) {
        active_leader.reset();
        GAUGE_SET(xllm_service_v3_placement_leader, 0);
        MULTI_COUNTER_INC(xllm_service_v3_placement_cycles_total,
                          "LEADER_FENCE_UNAVAILABLE");
        LOG_EVERY_N(ERROR, 10)
            << "Placement leader identity is unavailable or mismatched.";
      } else {
        cycle_attempted = true;
        const placement::PlacementMode requested_mode =
            FLAGS_placement_mode_override < 0
                ? placement_config_->controller.mode
                : static_cast<placement::PlacementMode>(
                      FLAGS_placement_mode_override);
        if (placement_controller_->mode() != requested_mode) {
          const placement::PlacementMode previous_mode =
              placement_controller_->mode();
          if (!placement_controller_->set_mode(requested_mode)) {
            LOG(ERROR) << "Rejected invalid V3 Placement mode override: "
                       << FLAGS_placement_mode_override;
          } else {
            GAUGE_SET(xllm_service_v3_placement_mode,
                      static_cast<int32_t>(requested_mode));
            LOG(WARNING) << "V3 Placement runtime mode changed, previous="
                         << placement::placement_mode_name(previous_mode)
                         << ", current="
                         << placement::placement_mode_name(requested_mode);
          }
        }
        const placement::PlacementLeaderIdentity leader{
            .address = std::move(address),
            .incarnation = std::move(incarnation),
            .epoch = epoch,
        };
        const bool leader_changed =
            !active_leader.has_value() ||
            active_leader->address != leader.address ||
            active_leader->incarnation != leader.incarnation ||
            active_leader->epoch != leader.epoch;
        const uint64_t now_monotonic_ms = monotonic_time_ms();
        if (leader_changed || !placement_controller_->recovered()) {
          const placement::PlacementControllerStatus recover_status =
              placement_controller_->recover(leader, now_monotonic_ms);
          if (recover_status != placement::PlacementControllerStatus::OK) {
            active_leader.reset();
            GAUGE_SET(xllm_service_v3_placement_leader, 0);
            MULTI_COUNTER_INC(xllm_service_v3_placement_cycles_total,
                              "RECOVERY_ERROR");
            LOG_EVERY_N(ERROR, 10)
                << "Placement recovery failed, status="
                << placement::placement_controller_status_name(recover_status);
          } else {
            active_leader = leader;
            GAUGE_SET(xllm_service_v3_placement_leader, 1);
            LOG(INFO) << "Placement leadership recovered, epoch="
                      << leader.epoch << ", incarnation=" << leader.incarnation;
          }
        }

        if (active_leader.has_value()) {
          std::vector<provider::EngineRegistryMemberSnapshot> members;
          const provider::ContractResult snapshot_status =
              instance_mgr_->snapshot_engine_members(
                  now_monotonic_ms,
                  placement_config_->input_builder.max_members,
                  &members);
          std::vector<placement::PlacementPoolRuntimeSpec> runtime_pools =
              placement_config_->pools;
          for (placement::PlacementPoolRuntimeSpec& pool : runtime_pools) {
            const FlowControlModelSnapshot flow =
                flow_control_queue_->model_snapshot(
                    pool.profile.pool.model_revision);
            pool.external.queue_depth =
                static_cast<double>(flow.queued_requests);
          }
          std::vector<placement::PlacementPoolCycleInput> inputs;
          placement::PlacementInputBuildStatus input_status =
              placement::PlacementInputBuildStatus::OBSERVATION_UNAVAILABLE;
          if (snapshot_status.ok()) {
            input_status = placement_input_builder_->build(
                runtime_pools,
                members,
                placement_observation_collector_.get(),
                now_monotonic_ms,
                &inputs);
          }
          MULTI_COUNTER_INC(
              xllm_service_v3_placement_observations_total,
              snapshot_status.ok()
                  ? placement::placement_input_build_status_name(input_status)
                  : "REGISTRY_ERROR");
          if (!snapshot_status.ok() ||
              input_status != placement::PlacementInputBuildStatus::OK) {
            inputs.clear();
            LOG_EVERY_N(ERROR, 10)
                << "Placement input unavailable, registry_ok="
                << snapshot_status.ok() << ", input_status="
                << placement::placement_input_build_status_name(input_status);
          }

          const placement::PlacementControllerResult result =
              placement_controller_->run_cycle(
                  *active_leader, inputs, now_monotonic_ms, unix_time_ms());
          MULTI_COUNTER_INC(
              xllm_service_v3_placement_cycles_total,
              placement::placement_controller_status_name(result.status));
          for (const placement::PlacementPoolCycleReport& pool : result.pools) {
            MULTI_COUNTER_INC(
                xllm_service_v3_placement_recommendations_total,
                placement_action_name(pool.recommendation.action));
            VLOG(1) << "placement_pool_cycle provider="
                    << static_cast<int32_t>(pool.pool.provider_id)
                    << " model=" << pool.pool.model_revision
                    << " role=" << static_cast<int32_t>(pool.pool.role)
                    << " profile=" << pool.pool.profile_digest << " action="
                    << placement_action_name(pool.recommendation.action)
                    << " reason="
                    << placement::placement_reason_name(
                           pool.recommendation.reason)
                    << " previous_desired="
                    << pool.recommendation.previous_desired_replicas
                    << " desired=" << pool.recommendation.desired_replicas
                    << " persisted=" << pool.desired_persisted
                    << " intents_added=" << pool.intents_added;
          }
          GAUGE_SET(xllm_service_v3_placement_operations,
                    placement_operation_executor_->size());
          if (result.intents_added > 0) {
            MULTI_COUNTER_ADD(xllm_service_v3_placement_operations_total,
                              "ADDED",
                              result.intents_added);
          }
          if (result.actuator.driven > 0) {
            MULTI_COUNTER_ADD(xllm_service_v3_placement_operations_total,
                              "DRIVEN",
                              result.actuator.driven);
          }
          if (result.actuator.compacted > 0) {
            MULTI_COUNTER_ADD(xllm_service_v3_placement_operations_total,
                              "COMPACTED",
                              result.actuator.compacted);
          }
          VLOG(1) << "placement_cycle status="
                  << placement::placement_controller_status_name(result.status)
                  << " mode=" << placement::placement_mode_name(result.mode)
                  << " pools=" << result.pools.size()
                  << " desired_writes=" << result.desired_writes
                  << " intents_added=" << result.intents_added
                  << " actuator_driven=" << result.actuator.driven
                  << " actuator_terminal=" << result.actuator.terminal
                  << " actuator_unknown=" << result.actuator.unknown
                  << " actuator_compacted=" << result.actuator.compacted;
        }
      }
    }

    if (cycle_attempted) {
      const int64_t elapsed_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - cycle_started)
              .count();
      HISTOGRAM_OBSERVE(xllm_service_v3_placement_cycle_milliseconds,
                        elapsed_ms);
    }
    std::unique_lock<std::mutex> lock(placement_wait_mutex_);
    placement_cv_.wait_for(
        lock,
        std::chrono::milliseconds(placement_config_->loop_interval_ms),
        [this] {
          return placement_stopped_ || exited_.load(std::memory_order_acquire);
        });
    if (placement_stopped_) {
      break;
    }
  }
  GAUGE_SET(xllm_service_v3_placement_leader, 0);
}

void Scheduler::update_master_service_heartbeat() {
  while (!exited_) {
    std::this_thread::sleep_for(std::chrono::seconds(kHeartbeatInterval));

    if (exited_.load(std::memory_order_acquire)) {
      break;
    }
    if (!is_master_service_.load(std::memory_order_acquire)) {
      continue;
    }

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
  if (req->has_engine_state()) {
    if (!is_master_service_.load(std::memory_order_acquire) ||
        req->engine_state().engine_uid() != req->name() ||
        req->engine_state().incarnation_id() != req->incarnation_id()) {
      return false;
    }
    const uint64_t now_monotonic_ms = monotonic_time_ms();
    bool applied = false;
    const provider::ContractResult recorded =
        instance_mgr_->record_engine_state(
            req->engine_state(), now_monotonic_ms, &applied);
    if (!recorded.ok()) {
      LOG(ERROR) << "Rejected EngineState from " << req->name() << ": "
                 << recorded.message();
      return false;
    }
    if (applied) {
      xllm::proto::StateBatch delta;
      delta.set_contract_version(provider::kProviderContractVersion);
      delta.set_master_incarnation(service_incarnation_id_);
      delta.set_snapshot_seq(1);
      delta.set_kind(xllm::proto::STATE_BATCH_KIND_DELTA);
      *delta.add_engine_states() = req->engine_state();
      const provider::ContractResult queued =
          state_stream_outbox_->enqueue_delta(delta, now_monotonic_ms);
      if (!queued.ok()) {
        LOG(ERROR) << "Failed to enqueue EngineState from " << req->name()
                   << ": " << queued.message();
        return false;
      }
      try_apply_local_full_state(now_monotonic_ms);
      state_stream_cv_.notify_all();
    }
  }
  global_kvcache_mgr_->record_updated_kvcaches(req->name(), req->cache_event());
  instance_mgr_->record_load_metrics_update(req->name(), req->load_metrics());
  instance_mgr_->update_latency_metrics(req->name(), req->latency_metrics());
  instance_mgr_->record_direct_engine_evidence(
      req->name(), req->incarnation_id(), true);
  return true;
}

bool Scheduler::record_direct_engine_evidence(const std::string& instance_name,
                                              const std::string& incarnation_id,
                                              bool success) {
  return instance_mgr_->record_direct_engine_evidence(
      instance_name, incarnation_id, success);
}

provider::ContractResult Scheduler::handle_engine_state_batch(
    const xllm::proto::StateBatch& batch,
    bool* applied) {
  if (exited_) {
    return provider::ContractResult::failure(
        xllm::proto::PROVIDER_CONTRACT_ERROR_INVALID_STATE,
        "Scheduler is exiting");
  }
  return instance_mgr_->apply_engine_state_batch(
      batch, monotonic_time_ms(), applied);
}

provider::KVApplyResult Scheduler::handle_kv_event_batch(
    const xllm::proto::KVEventBatch& batch) {
  if (exited_.load(std::memory_order_acquire) ||
      !is_master_service_.load(std::memory_order_acquire)) {
    return provider::KVApplyResult{
        .code = provider::KVApplyCode::REJECTED,
        .reason = "KV events are accepted only by the current master",
    };
  }
  const InstanceMetaInfo info =
      instance_mgr_->get_instance_info(batch.identity().engine().engine_uid());
  if (!matches_current_kv_engine(info, batch.identity())) {
    return provider::KVApplyResult{
        .code = provider::KVApplyCode::REJECTED,
        .reason = "KV event identity is not a current registered Engine",
    };
  }

  const uint64_t now_monotonic_ms = monotonic_time_ms();
  const provider::KVApplyResult applied =
      kv_shadow_index_->apply_event_batch(batch, now_monotonic_ms);
  if (applied.code == provider::KVApplyCode::REJECTED) {
    return applied;
  }
  const provider::ContractResult queued =
      kv_state_outbox_->enqueue(batch, now_monotonic_ms);
  if (!queued.ok()) {
    // The master must not retain KV credit that it could not replicate. A
    // later Engine retry or keepalive will drive snapshot recovery without
    // affecting the independent load plane.
    kv_shadow_index_->reset_all();
    return provider::KVApplyResult{
        .code = provider::KVApplyCode::REJECTED,
        .reason = queued.message(),
        .accepted_through_event_seq = applied.accepted_through_event_seq,
    };
  }
  kv_state_cv_.notify_all();
  if (applied.code == provider::KVApplyCode::SNAPSHOT_REQUIRED) {
    kv_snapshot_cv_.notify_all();
  }
  return applied;
}

provider::KVApplyResult Scheduler::handle_kv_state_batch(
    const xllm::proto::KVStateBatch& batch) {
  if (exited_.load(std::memory_order_acquire) ||
      is_master_service_.load(std::memory_order_acquire)) {
    return provider::KVApplyResult{
        .code = provider::KVApplyCode::REJECTED,
        .reason = "KV State batches are accepted only by Service replicas",
    };
  }
  const provider::KVApplyResult applied =
      kv_state_replica_->apply(batch, monotonic_time_ms());
  if (applied.code == provider::KVApplyCode::SNAPSHOT_REQUIRED) {
    kv_snapshot_cv_.notify_all();
  }
  return applied;
}

void Scheduler::handle_master_service_watch(const etcd::Response& response,
                                            const uint64_t& prefix_len) {
  static_cast<void>(prefix_len);
  if (exited_ || response.events().empty()) {
    return;
  }
  const bool master_deleted = std::any_of(
      response.events().begin(),
      response.events().end(),
      [](const etcd::Event& event) {
        return event.event_type() == etcd::Event::EventType::DELETE_;
      });
  if (master_deleted && etcd_client_->elect_master(options_.service_name(),
                                                   service_incarnation_id_,
                                                   kHeartbeatInterval)) {
    activate_as_master();
  }
}

void Scheduler::handle_master_identity_watch(const etcd::Response& response,
                                             const uint64_t& prefix_len) {
  static_cast<void>(prefix_len);
  if (exited_ || response.events().empty()) {
    return;
  }

  std::string master_address;
  std::string master_incarnation;
  if (!etcd_client_->get(ETCD_MASTER_SERVICE_KEY, &master_address) ||
      !etcd_client_->get(ETCD_MASTER_SERVICE_INCARNATION_KEY,
                         &master_incarnation) ||
      master_address.empty() || master_incarnation.empty()) {
    deactivate_as_master();
    instance_mgr_->set_engine_state_master("");
    kv_state_replica_->set_master("");
    return;
  }
  if (master_address == options_.service_name() &&
      master_incarnation == service_incarnation_id_) {
    activate_as_master();
    return;
  }
  deactivate_as_master();
  const provider::ContractResult result =
      instance_mgr_->set_engine_state_master(master_incarnation);
  if (!result.ok()) {
    LOG(ERROR) << "Failed to update Engine State Registry master view: "
               << result.message();
  }
  if (!kv_state_replica_->set_master(master_incarnation)) {
    LOG(ERROR) << "Failed to update KV State master view.";
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

void Scheduler::reset_d_admission_trace(
    const std::shared_ptr<Request>& request) {
  if (request == nullptr) {
    return;
  }
  request->trace_d_admission_started_ns.store(0, std::memory_order_release);
  request->trace_d_admission_terminal_recorded.store(false,
                                                     std::memory_order_release);
}

void Scheduler::begin_d_admission(const std::shared_ptr<Request>& request) {
  if (request == nullptr ||
      request->execution_mode != xllm::proto::EXECUTION_MODE_REMOTE_PD) {
    return;
  }
  int64_t expected = 0;
  const int64_t started_ns = monotonic_time_ns();
  if (!request->trace_d_admission_started_ns.compare_exchange_strong(
          expected,
          started_ns,
          std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    return;
  }
  record_request_event(request,
                       xllm::proto::REQUEST_EVENT_TYPE_D_ADMISSION,
                       xllm::proto::EVENT_RESULT_STARTED,
                       xllm::proto::ERROR_STAGE_NONE,
                       xllm::proto::EVENT_REASON_NONE,
                       request->routing.decode_name,
                       request->decode_incarnation_id);
}

void Scheduler::finish_d_admission(
    const std::shared_ptr<Request>& request,
    xllm::proto::EventResult result,
    xllm::proto::EventReason reason,
    std::optional<uint32_t> engine_attempts,
    std::optional<uint64_t> engine_rpc_duration_ns) {
  if (request == nullptr ||
      request->execution_mode != xllm::proto::EXECUTION_MODE_REMOTE_PD ||
      request->trace_d_admission_started_ns.load(std::memory_order_acquire) ==
          0 ||
      request->trace_d_admission_terminal_recorded.exchange(
          true, std::memory_order_acq_rel)) {
    return;
  }
  record_request_event(request,
                       xllm::proto::REQUEST_EVENT_TYPE_D_ADMISSION,
                       result,
                       result == xllm::proto::EVENT_RESULT_ACCEPTED
                           ? xllm::proto::ERROR_STAGE_NONE
                           : xllm::proto::ERROR_STAGE_D_ADMISSION,
                       result == xllm::proto::EVENT_RESULT_ACCEPTED
                           ? xllm::proto::EVENT_REASON_NONE
                           : reason,
                       request->routing.decode_name,
                       request->decode_incarnation_id,
                       engine_rpc_duration_ns);
  VLOG(1) << "Decode admission terminal request_uid="
          << request->correlation.request_uid() << ", attempt_seq="
          << (request->correlation.has_attempt_seq()
                  ? std::to_string(request->correlation.attempt_seq())
                  : "absent")
          << ", result=" << xllm::proto::EventResult_Name(result)
          << ", reason=" << xllm::proto::EventReason_Name(reason)
          << ", engine_attempts="
          << (engine_attempts.has_value() ? std::to_string(*engine_attempts)
                                          : "absent")
          << ", engine_rpc_duration_ns="
          << (engine_rpc_duration_ns.has_value()
                  ? std::to_string(*engine_rpc_duration_ns)
                  : "absent");
}

void Scheduler::finish_d_admission_from_output(
    const std::shared_ptr<Request>& request,
    const llm::RequestOutput& output) {
  if (request == nullptr ||
      request->execution_mode != xllm::proto::EXECUTION_MODE_REMOTE_PD ||
      request->trace_d_admission_terminal_recorded.load(
          std::memory_order_acquire)) {
    return;
  }
  if (output.decode_admission_disposition.has_value() &&
      output.decode_admission_reason.has_value() &&
      output.decode_admission_attempts.has_value() &&
      output.decode_admission_rpc_duration_ns.has_value()) {
    const xllm::proto::AdmissionDisposition disposition =
        *output.decode_admission_disposition;
    const xllm::proto::AdmissionReason admission_reason =
        *output.decode_admission_reason;
    finish_d_admission(
        request,
        disposition == xllm::proto::ADMISSION_DISPOSITION_ACCEPTED
            ? xllm::proto::EVENT_RESULT_ACCEPTED
            : xllm::proto::EVENT_RESULT_REJECTED,
        decode_admission_event_reason(admission_reason),
        output.decode_admission_attempts,
        output.decode_admission_rpc_duration_ns);
    return;
  }
  const bool terminal_without_observation =
      output.finished_on_prefill_instance || output.finished ||
      (output.status.has_value() && !output.status->ok());
  if (terminal_without_observation) {
    finish_d_admission(request,
                       xllm::proto::EVENT_RESULT_FAILED,
                       xllm::proto::EVENT_REASON_MISSING_TERMINAL);
  }
}

void Scheduler::record_resource_release(
    const std::shared_ptr<Request>& request) {
  if (request == nullptr) {
    return;
  }
  const bool unresolved = request->execution_hold.has_hold();
  const bool deferred =
      !unresolved && request->correlation.has_attempt_seq() &&
      execution_hold_cleanup_table_->contains(execution_attempt(*request));
  if (deferred) {
    request->trace_resource_release_started_ns.store(monotonic_time_ns(),
                                                     std::memory_order_release);
    std::lock_guard<std::mutex> trace_guard(resource_release_trace_mutex_);
    if (pending_resource_release_traces_.size() >=
        options_.execution_hold_cleanup_record_capacity()) {
      record_request_event(request,
                           xllm::proto::REQUEST_EVENT_TYPE_RESOURCE_RELEASE,
                           xllm::proto::EVENT_RESULT_FAILED,
                           xllm::proto::ERROR_STAGE_RESOURCE_RELEASE,
                           xllm::proto::EVENT_REASON_INTERNAL_ERROR,
                           "",
                           "");
      return;
    }
    pending_resource_release_traces_.insert_or_assign(
        request->correlation.request_uid(), request);
  }
  record_request_event(request,
                       xllm::proto::REQUEST_EVENT_TYPE_RESOURCE_RELEASE,
                       unresolved
                           ? xllm::proto::EVENT_RESULT_FAILED
                           : (deferred ? xllm::proto::EVENT_RESULT_STARTED
                                       : xllm::proto::EVENT_RESULT_SUCCEEDED),
                       unresolved ? xllm::proto::ERROR_STAGE_RESOURCE_RELEASE
                                  : xllm::proto::ERROR_STAGE_NONE,
                       unresolved ? xllm::proto::EVENT_REASON_INTERNAL_ERROR
                                  : xllm::proto::EVENT_REASON_NONE,
                       "",
                       "");
}

void Scheduler::finish_resource_release_traces(bool fail_pending) {
  std::vector<std::shared_ptr<Request>> completed;
  std::vector<std::shared_ptr<Request>> failed;
  {
    std::lock_guard<std::mutex> trace_guard(resource_release_trace_mutex_);
    for (auto trace = pending_resource_release_traces_.begin();
         trace != pending_resource_release_traces_.end();) {
      const std::shared_ptr<Request>& request = trace->second;
      if (execution_hold_cleanup_table_->contains(
              execution_attempt(*request))) {
        if (fail_pending) {
          failed.emplace_back(request);
          trace = pending_resource_release_traces_.erase(trace);
          continue;
        }
        ++trace;
        continue;
      }
      completed.emplace_back(request);
      trace = pending_resource_release_traces_.erase(trace);
    }
  }
  const int64_t finished_ns = monotonic_time_ns();
  for (const std::shared_ptr<Request>& request : completed) {
    const int64_t started_ns = request->trace_resource_release_started_ns.load(
        std::memory_order_acquire);
    record_request_event(request,
                         xllm::proto::REQUEST_EVENT_TYPE_RESOURCE_RELEASE,
                         xllm::proto::EVENT_RESULT_SUCCEEDED,
                         xllm::proto::ERROR_STAGE_NONE,
                         xllm::proto::EVENT_REASON_NONE,
                         "",
                         "",
                         finished_ns >= started_ns && started_ns != 0
                             ? std::optional<uint64_t>(static_cast<uint64_t>(
                                   finished_ns - started_ns))
                             : std::nullopt);
  }
  for (const std::shared_ptr<Request>& request : failed) {
    record_request_event(request,
                         xllm::proto::REQUEST_EVENT_TYPE_RESOURCE_RELEASE,
                         xllm::proto::EVENT_RESULT_FAILED,
                         xllm::proto::ERROR_STAGE_RESOURCE_RELEASE,
                         xllm::proto::EVENT_REASON_MISSING_TERMINAL,
                         "",
                         "");
  }
}

bool Scheduler::install_execution_hold_locked(
    const std::shared_ptr<Request>& request) {
  if (!instance_mgr_->validate_request_instance_incarnations(request)) {
    LOG(ERROR) << "Refuse dispatch to a stale instance incarnation. "
               << request->routing.debug_string();
    return false;
  }
  if (!has_execution_holder(*request)) {
    if (request->provider_id == xllm::proto::PROVIDER_ID_XLLM_NATIVE &&
        request->execution_mode == xllm::proto::EXECUTION_MODE_PREFILL_ONLY) {
      request->output_event_sequencer =
          std::make_unique<OutputEventSequencer>(OutputEventSequencer::Config{
              .max_buffered_events = options_.output_reorder_max_events(),
              .max_buffered_bytes = options_.output_reorder_max_bytes(),
              .max_sequence_gap = options_.output_reorder_max_events()});
    }
    return true;
  }
  if (request->provider_id == xllm::proto::PROVIDER_ID_VLLM_ASCEND) {
    const provider::ExecutionHoldStatus status =
        execution_hold_cleanup_table_->install_request_hold(
            &request->execution_hold,
            xllm::proto::EXECUTION_HOLD_KIND_AGGREGATED_EXECUTION,
            execution_attempt(*request),
            service_incarnation_id_,
            {execution_holder(*request)});
    if (status == provider::ExecutionHoldStatus::kOk) {
      return true;
    }
    LOG(ERROR) << "Failed to install aggregated execution hold before "
                  "dispatch, request_uid="
               << request->correlation.request_uid()
               << ", status=" << static_cast<int>(status);
    return false;
  }

  const bool local = request->execution_mode ==
                     xllm::proto::EXECUTION_MODE_LOCAL_PREFILL_DECODE;
  const xllm::proto::ExecutionHolder holder = execution_holder(*request);
  const provider::ExecutionHoldStatus status =
      execution_hold_cleanup_table_->install_request_hold(
          &request->execution_hold,
          local ? xllm::proto::EXECUTION_HOLD_KIND_LOCAL_DECODE_SUBMISSION
                : xllm::proto::EXECUTION_HOLD_KIND_REMOTE_D_RESERVATION,
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
  return true;
}

void Scheduler::rollback_request_safety_guards_locked(
    const std::shared_ptr<Request>& request) {
  client_disconnect_monitor_->erase(request->correlation.request_uid());
}

bool Scheduler::confirm_generation_commit(
    const std::shared_ptr<Request>& request) {
  if (!has_execution_holder(*request)) {
    return true;
  }
  const provider::ExecutionHoldStatus status =
      request->execution_hold.confirm_holder(
          execution_holder(*request),
          xllm::proto::EXECUTION_HOLD_PROOF_GENERATION_COMMITTED);
  if (provider::execution_holder_confirmation_succeeded(status)) {
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
  if (!has_execution_holder(*request)) {
    return true;
  }
  const provider::ExecutionHoldStatus status =
      request->execution_hold.apply_convergence_proof(
          execution_attempt(*request),
          execution_holder(*request),
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

bool Scheduler::request_cancel_fences_for_retry(
    const std::shared_ptr<Request>& request,
    std::optional<xllm::proto::ExecutionResourceHold>* fenced_hold) {
  CHECK(fenced_hold != nullptr);
  fenced_hold->reset();
  const std::optional<xllm::proto::ExecutionResourceHold> hold =
      request->execution_hold.snapshot();
  if (!hold.has_value()) {
    if (request->provider_id == xllm::proto::PROVIDER_ID_XLLM_NATIVE &&
        request->execution_mode == xllm::proto::EXECUTION_MODE_PREFILL_ONLY) {
      xllm::proto::ExecutionResourceHold attempt_identity;
      *attempt_identity.mutable_attempt() = execution_attempt(*request);
      if (!call_attempt_control(attempt_identity,
                                prefill_holder(*request),
                                /*query=*/false,
                                options_.instance_delete_probe_timeout_ms())) {
        return false;
      }
    }
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
  }
  *fenced_hold = *hold;
  return true;
}

bool Scheduler::select_retry_instances(
    const std::shared_ptr<Request>& request) {
  const bool requires_strict_route = request->execution_plan.has_value();
  record_kv_route_actual(request, nullptr);
  request->routing = Routing();
  request->prefill_incarnation_id.clear();
  request->decode_incarnation_id.clear();
  request->prefill_provider_descriptor.reset();
  request->decode_provider_descriptor.reset();
  request->execution_mode = xllm::proto::EXECUTION_MODE_UNSPECIFIED;
  if (!lb_policy_->select_instances_pair(request) ||
      !instance_mgr_->bind_request_instance_incarnations(request)) {
    return false;
  }
  if (!apply_native_execution_mode(request)) {
    return false;
  }
  const bool remote =
      request->execution_mode == xllm::proto::EXECUTION_MODE_REMOTE_PD;
  if (request->provider_id == xllm::proto::PROVIDER_ID_VLLM_ASCEND ||
      request->execution_mode == xllm::proto::EXECUTION_MODE_UNSPECIFIED ||
      (remote && request->routing.decode_name.empty()) ||
      (requires_strict_route &&
       (!request->prefill_provider_descriptor.has_value() ||
        (remote && !request->decode_provider_descriptor.has_value())))) {
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
  std::optional<xllm::proto::ExecutionResourceHold> fenced_hold;
  if (!request_cancel_fences_for_retry(request, &fenced_hold)) {
    *failure_message = "First-output retry denied: OLD_EXECUTION_NOT_CONVERGED";
    return false;
  }

  std::lock_guard<std::mutex> cleanup_guard(execution_hold_cleanup_mutex_);
  {
    std::lock_guard<std::mutex> request_guard(request_mutex_);
    const auto request_it = requests_.find(request->correlation.request_uid());
    if (request_it == requests_.end() || request_it->second != request) {
      *failure_message = "First-output retry request ended during fencing";
      return false;
    }
  }
  if (fenced_hold.has_value()) {
    const std::optional<xllm::proto::ExecutionResourceHold> current_hold =
        request->execution_hold.snapshot();
    if (!current_hold.has_value() ||
        current_hold->attempt().request_uid() !=
            fenced_hold->attempt().request_uid() ||
        !current_hold->attempt().has_attempt_seq() ||
        !fenced_hold->attempt().has_attempt_seq() ||
        current_hold->attempt().attempt_seq() !=
            fenced_hold->attempt().attempt_seq()) {
      *failure_message =
          "First-output retry execution hold changed during fencing";
      return false;
    }
    for (const xllm::proto::ExecutionHolder& holder :
         fenced_hold->potential_holders()) {
      request->execution_hold.apply_convergence_proof(
          fenced_hold->attempt(),
          holder,
          xllm::proto::HOLDER_CONVERGENCE_PROOF_CANCEL_FENCE_ACK);
    }
    if (request->execution_hold.has_hold()) {
      *failure_message =
          "First-output retry cancel proofs did not resolve the old hold";
      return false;
    }
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
  if (!prepare_v2_execution_plan(request)) {
    *failure_message =
        "First-output retry could not rebuild the V2 ExecutionPlan";
    return false;
  }
  record_kv_route_decision(request);
  request->prefill_stage_finished.store(false, std::memory_order_release);
  request->latest_generate_time = absl::Now();
  request->output_event_sequencer.reset();
  reset_d_admission_trace(request);
  if (!install_execution_hold_locked(request)) {
    *failure_message = "First-output retry could not install execution hold";
    return false;
  }
  instance_mgr_->update_request_metrics(request, RequestAction::SCHEDULE);
  record_request_event(request,
                       xllm::proto::REQUEST_EVENT_TYPE_ROUTE,
                       xllm::proto::EVENT_RESULT_SUCCEEDED,
                       xllm::proto::ERROR_STAGE_NONE,
                       xllm::proto::EVENT_REASON_NONE,
                       request->routing.prefill_name,
                       request->prefill_incarnation_id);
  if (!request->routing.decode_name.empty()) {
    record_request_event(request,
                         xllm::proto::REQUEST_EVENT_TYPE_ROUTE,
                         xllm::proto::EVENT_RESULT_SUCCEEDED,
                         xllm::proto::ERROR_STAGE_NONE,
                         xllm::proto::EVENT_REASON_NONE,
                         request->routing.decode_name,
                         request->decode_incarnation_id);
  }

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
  record_request_terminal(request,
                          terminal_result_for_status(status_code),
                          xllm::proto::ERROR_STAGE_OUTPUT,
                          event_reason_for_status(status_code));
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

void Scheduler::detach_execution_hold_locked(
    const std::shared_ptr<Request>& request) {
  if (!request->execution_hold.has_hold()) {
    return;
  }
  const provider::ExecutionHoldStatus status =
      execution_hold_cleanup_table_->adopt(&request->execution_hold);
  if (status != provider::ExecutionHoldStatus::kOk) {
    LOG(ERROR) << "Failed to detach unresolved execution hold, request_uid="
               << request->correlation.request_uid()
               << ", status=" << static_cast<int>(status);
    return;
  }
  execution_hold_cleanup_cv_.notify_one();
}

bool Scheduler::call_attempt_control(
    const xllm::proto::ExecutionResourceHold& hold,
    const xllm::proto::ExecutionHolder& holder,
    bool query,
    int32_t timeout_ms) {
  const std::shared_ptr<brpc::Channel> channel =
      instance_mgr_->get_channel(holder.engine_uid());
  if (channel == nullptr) {
    COUNTER_INC(attempt_control_no_channel_total);
    return false;
  }
  const InstanceMetaInfo instance = get_instance_info(holder.engine_uid());
  if (instance.provider_id == xllm::proto::PROVIDER_ID_VLLM_ASCEND &&
      !provider::valid_vllm_agent_internal_token(
          options_.internal_api_token())) {
    COUNTER_INC(attempt_control_token_invalid_total);
    LOG_EVERY_N(ERROR, 100)
        << "Cannot converge vLLM attempt: internal_api_token is invalid";
    return false;
  }
  const provider::AttemptControlResult result =
      provider::call_provider_attempt_control(
          instance.provider_id,
          channel,
          hold,
          holder,
          query ? provider::AttemptControlOperation::QUERY
                : provider::AttemptControlOperation::CANCEL,
          options_.internal_api_token(),
          timeout_ms);
  record_direct_engine_evidence(
      holder.engine_uid(), holder.incarnation_id(), result.direct_success);
  if (!result.direct_success) {
    COUNTER_INC(attempt_control_rpc_failed_total);
  } else if (!result.terminal_proof) {
    COUNTER_INC(attempt_control_non_terminal_total);
  }
  return result.direct_success && result.terminal_proof;
}

void Scheduler::recover_first_output_events(
    const std::vector<std::shared_ptr<Request>>& requests,
    bool fail_if_unavailable) {
  std::vector<std::shared_ptr<Request>> active_requests;
  active_requests.reserve(requests.size());
  std::vector<FirstEventRecoveryQuery> queries;
  queries.reserve(requests.size());
  std::vector<bool> requires_remote_query;
  requires_remote_query.reserve(requests.size());
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
    const bool remote = service_request->execution_mode ==
                        xllm::proto::EXECUTION_MODE_REMOTE_PD;
    requires_remote_query.emplace_back(remote);
    if (remote) {
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
  }

  std::vector<FirstEventRecoveryResult> results =
      query_first_output_events(queries);
  CHECK_EQ(results.size(), queries.size());
  size_t result_index = 0;
  for (size_t index = 0; index < active_requests.size(); ++index) {
    bool recovered = false;
    if (requires_remote_query[index]) {
      const FirstEventRecoveryQuery& query = queries[result_index];
      const FirstEventRecoveryResult& result = results[result_index++];
      if (result.status.ok() && result.output.has_value()) {
        recovered = handle_generation(*result.output);
      } else {
        LOG(ERROR) << "Failed to recover first event, request_uid="
                   << query.attempt.request_uid()
                   << ", error=" << result.status.message();
      }
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
    begin_d_admission(service_request);
    if (service_request->retry_dispatch_callback(*service_request)) {
      MULTI_COUNTER_INC(xllm_service_v2_execution_mode_total,
                        execution_mode_label(service_request->execution_mode));
      xllm::proto::ExecutionHolder holder = execution_holder(*service_request);
      if (holder.engine_uid().empty()) {
        holder = prefill_holder(*service_request);
      }
      record_request_event(service_request,
                           xllm::proto::REQUEST_EVENT_TYPE_P_DISPATCH,
                           xllm::proto::EVENT_RESULT_ACCEPTED,
                           xllm::proto::ERROR_STAGE_NONE,
                           xllm::proto::EVENT_REASON_NONE,
                           holder.engine_uid(),
                           holder.incarnation_id());
      continue;
    }

    finish_d_admission(service_request,
                       xllm::proto::EVENT_RESULT_FAILED,
                       xllm::proto::EVENT_REASON_TRANSPORT_ERROR);

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
    finish_resource_release_traces();

    wait_lock.lock();
  }
}

void Scheduler::run_flow_dispatch() {
  std::unique_lock<std::mutex> wait_lock(flow_dispatch_wait_mutex_);
  while (!flow_dispatch_stopped_) {
    flow_dispatch_cv_.wait_for(
        wait_lock,
        std::chrono::milliseconds(options_.flow_dispatch_interval_ms()),
        [this] { return flow_dispatch_stopped_; });
    if (flow_dispatch_stopped_) {
      break;
    }
    wait_lock.unlock();

    const std::vector<FlowControlWork> expired_work =
        flow_control_queue_->take_expired(
            FlowControlQueue::Clock::now(),
            options_.request_watchdog_batch_size());
    const auto fail_queued_work = [this](
                                      const std::vector<FlowControlWork>& work,
                                      llm::StatusCode status,
                                      const char* message) {
      for (const FlowControlWork& item : work) {
        std::shared_ptr<Request> request;
        {
          std::lock_guard<std::mutex> request_guard(request_mutex_);
          const auto it = requests_.find(item.request_uid);
          if (it != requests_.end()) {
            request = it->second;
          }
        }
        if (request == nullptr) {
          continue;
        }
        std::lock_guard<std::mutex> output_guard(
            request->output_dispatch_mutex);
        {
          std::lock_guard<std::mutex> request_guard(request_mutex_);
          const auto it = requests_.find(item.request_uid);
          if (it == requests_.end() || it->second != request ||
              request->queue_state.load(std::memory_order_acquire) !=
                  RequestQueueState::QUEUED) {
            continue;
          }
        }
        fail_output_dispatch_locked(request, status, message);
      }
    };
    fail_queued_work(expired_work,
                     llm::StatusCode::DEADLINE_EXCEEDED,
                     "QUEUE_WAIT_EXCEEDED");

    if (draining_.load(std::memory_order_acquire) &&
        options_.flow_drain_policy() == "RETRY_UNDISPATCHED") {
      const std::vector<FlowControlWork> retry_work =
          flow_control_queue_->retry_undispatched(
              options_.request_watchdog_batch_size());
      fail_queued_work(retry_work,
                       llm::StatusCode::UNAVAILABLE,
                       "SERVICE_DRAIN_RETRY_UNDISPATCHED");
    }

    // COMPLETE_QUEUED intentionally reaches the work-conserving loop below.
    while (!exited_.load(std::memory_order_acquire)) {
      const FlowControlDispatch dispatch = flow_control_queue_->take_next(
          FlowControlQueue::Clock::now(), flow_saturation_state());
      if (dispatch.status != FlowControlStatus::OK ||
          !dispatch.work.has_value()) {
        break;
      }

      const std::string request_uid = dispatch.work->request_uid;
      std::shared_ptr<Request> request;
      {
        std::lock_guard<std::mutex> request_guard(request_mutex_);
        const auto it = requests_.find(request_uid);
        if (it != requests_.end()) {
          request = it->second;
        }
      }
      if (request == nullptr ||
          !request->dispatch_ready.load(std::memory_order_acquire) ||
          request->queue_state.load(std::memory_order_acquire) !=
              RequestQueueState::QUEUED) {
        if (request != nullptr &&
            request->queue_state.load(std::memory_order_acquire) ==
                RequestQueueState::QUEUED) {
          flow_control_queue_->return_to_queue(request_uid);
          break;
        }
        flow_control_queue_->cancel(request_uid);
        continue;
      }

      if (!select_and_prepare_dispatch(request)) {
        flow_control_queue_->return_to_queue(request_uid);
        break;
      }

      std::optional<uint64_t> queue_wait_ns;
      if (request->enqueue_time.has_value()) {
        queue_wait_ns = elapsed_ns_since(*request->enqueue_time);
      }
      const std::string mode = execution_mode_label(request->execution_mode);
      if (queue_wait_ns.has_value()) {
        MULTI_HISTOGRAM_OBSERVE(xllm_service_v2_queue_wait_milliseconds,
                                mode,
                                static_cast<int64_t>(*queue_wait_ns / 1000000));
      }
      record_request_event(request,
                           xllm::proto::REQUEST_EVENT_TYPE_ROUTE,
                           xllm::proto::EVENT_RESULT_SUCCEEDED,
                           xllm::proto::ERROR_STAGE_NONE,
                           xllm::proto::EVENT_REASON_NONE,
                           request->routing.prefill_name,
                           request->prefill_incarnation_id,
                           queue_wait_ns);
      if (!request->routing.decode_name.empty()) {
        record_request_event(request,
                             xllm::proto::REQUEST_EVENT_TYPE_ROUTE,
                             xllm::proto::EVENT_RESULT_SUCCEEDED,
                             xllm::proto::ERROR_STAGE_NONE,
                             xllm::proto::EVENT_REASON_NONE,
                             request->routing.decode_name,
                             request->decode_incarnation_id);
      }

      bool dispatch_started = false;
      bool request_active = false;
      bool hold_installed = false;
      {
        std::lock_guard<std::mutex> output_guard(
            request->output_dispatch_mutex);
        if (!request->output_dispatch_closed) {
          std::lock_guard<std::mutex> cleanup_guard(
              execution_hold_cleanup_mutex_);
          std::lock_guard<std::mutex> request_guard(request_mutex_);
          const auto it = requests_.find(request_uid);
          request_active =
              it != requests_.end() && it->second == request &&
              request->queue_state.load(std::memory_order_acquire) ==
                  RequestQueueState::QUEUED;
          if (request_active) {
            hold_installed = install_execution_hold_locked(request);
            if (hold_installed) {
              request->queue_state.store(RequestQueueState::DISPATCHED,
                                         std::memory_order_release);
            }
          }
        }
        if (request_active && hold_installed &&
            request->dispatch_callback != nullptr) {
          begin_d_admission(request);
          dispatch_started = request->dispatch_callback(request);
        }
      }

      if (!request_active) {
        flow_control_queue_->cancel(request_uid);
        continue;
      }
      if (!hold_installed) {
        request->execution_hold.abandon_before_dispatch();
        flow_control_queue_->return_to_queue(request_uid);
        request->queue_state.store(RequestQueueState::QUEUED,
                                   std::memory_order_release);
        break;
      }
      if (!dispatch_started) {
        finish_d_admission(request,
                           xllm::proto::EVENT_RESULT_FAILED,
                           xllm::proto::EVENT_REASON_TRANSPORT_ERROR);
        handle_attempt_dispatch_failure(
            request_uid,
            request->correlation.has_attempt_seq()
                ? request->correlation.attempt_seq()
                : 0,
            "Provider dispatch could not be started");
      } else {
        MULTI_COUNTER_INC(xllm_service_v2_execution_mode_total, mode);
        xllm::proto::ExecutionHolder holder = execution_holder(*request);
        if (holder.engine_uid().empty()) {
          holder = prefill_holder(*request);
        }
        record_request_event(request,
                             xllm::proto::REQUEST_EVENT_TYPE_P_DISPATCH,
                             xllm::proto::EVENT_RESULT_ACCEPTED,
                             xllm::proto::ERROR_STAGE_NONE,
                             xllm::proto::EVENT_REASON_NONE,
                             holder.engine_uid(),
                             holder.incarnation_id());
      }
    }

    wait_lock.lock();
  }
}

void Scheduler::run_observability_exporter() {
  uint64_t last_snapshot_ms = 0;
  uint64_t previous_successful_terminals = 0;
  uint64_t previous_failed_terminals = 0;
  uint64_t previous_delivered_tokens = 0;
  provider::KVRouteMetricsSnapshot previous_kv;
  observability::RecorderStats previous_stats;
  const auto export_one_batch = [this]() {
    std::vector<xllm::proto::RequestEvent> events =
        request_event_recorder_->drain(
            options_.observability_export_batch_size());
    for (const xllm::proto::RequestEvent& event : events) {
      try {
        VLOG(1) << "xllm_service_request_event "
                << observability::format_request_event_log(event);
      } catch (const std::exception& error) {
        LOG(ERROR) << "Failed to format request event: " << error.what();
      }
    }
    return events.size();
  };

  std::unique_lock<std::mutex> wait_lock(observability_wait_mutex_);
  while (!observability_stopped_) {
    observability_cv_.wait_for(
        wait_lock,
        std::chrono::milliseconds(options_.observability_export_interval_ms()),
        [this] { return observability_stopped_; });
    if (observability_stopped_) {
      break;
    }
    wait_lock.unlock();
    export_one_batch();

    const FlowControlSnapshot flow = flow_control_queue_->snapshot();
    size_t active_requests = 0;
    {
      std::lock_guard<std::mutex> request_guard(request_mutex_);
      active_requests = requests_.size();
    }
    GAUGE_SET(xllm_service_v2_queued_requests, flow.queued_requests);
    GAUGE_SET(xllm_service_v2_dispatched_requests, flow.dispatched_requests);
    GAUGE_SET(xllm_service_v2_queued_prompt_tokens, flow.queued_prompt_tokens);
    GAUGE_SET(xllm_service_v2_queued_bytes, flow.queued_bytes);
    GAUGE_SET(xllm_service_v2_active_requests, active_requests);
    GAUGE_SET(xllm_service_v2_observability_ring_events,
              request_event_recorder_->size());

    const observability::RecorderStats stats = request_event_recorder_->stats();
    if (stats.dropped_capacity > previous_stats.dropped_capacity ||
        stats.dropped_contention > previous_stats.dropped_contention ||
        stats.invalid_events > previous_stats.invalid_events) {
      LOG(WARNING) << "xllm_service_observability_loss "
                   << "service_incarnation_id=" << service_incarnation_id_
                   << " dropped_capacity=" << stats.dropped_capacity
                   << " dropped_contention=" << stats.dropped_contention
                   << " invalid_events=" << stats.invalid_events
                   << " buffered=" << request_event_recorder_->size();
    }
    previous_stats = stats;

    const uint64_t now_ms = monotonic_time_ms();
    if (last_snapshot_ms == 0 ||
        now_ms - last_snapshot_ms >=
            static_cast<uint64_t>(
                options_.observability_snapshot_interval_ms())) {
      const provider::ReadinessSnapshot readiness = readiness_status();
      const SaturationState saturation = flow_saturation_state();
      const provider::KVRouteMetricsSnapshot kv = kv_route_metrics_.snapshot();
      const uint64_t successful_terminals =
          observability_successful_terminals_.load(std::memory_order_relaxed);
      const uint64_t failed_terminals =
          observability_failed_terminals_.load(std::memory_order_relaxed);
      const uint64_t delivered_tokens =
          observability_delivered_tokens_.load(std::memory_order_relaxed);
      const uint64_t snapshot_window_ms =
          last_snapshot_ms == 0 ? 0 : now_ms - last_snapshot_ms;
      const std::optional<provider::ObservationSnapshot> observation =
          instance_mgr_->engine_observation_snapshot(now_ms);
      const provider::EngineKVCapacitySnapshot engine_kv =
          instance_mgr_->engine_kv_capacity_snapshot(now_ms);
      GAUGE_SET(xllm_service_v2_engine_kv_reporting_engines,
                engine_kv.reporting_engines);
      GAUGE_SET(xllm_service_v2_engine_kv_reporting_dp_ranks,
                engine_kv.reporting_dp_ranks);
      GAUGE_SET(xllm_service_v2_engine_kv_max_used_ratio,
                engine_kv.has_used_ratio ? engine_kv.max_used_ratio : 0.0);
      GAUGE_SET(xllm_service_v2_engine_kv_min_free_blocks,
                engine_kv.has_free_blocks ? engine_kv.min_free_blocks : 0);
      GAUGE_SET(xllm_service_v2_engine_kv_total_free_blocks,
                engine_kv.has_free_blocks ? engine_kv.total_free_blocks : 0);
      LOG(INFO)
          << "xllm_service_cluster_snapshot "
          << "service_incarnation_id=" << service_incarnation_id_
          << " build_id=" << options_.observability_build_id()
          << " readiness=" << provider::readiness_reason_name(readiness.reason)
          << " accepting_new_requests=" << readiness.accepting_new_requests
          << " saturation=" << saturation_state_name(saturation)
          << " observation_mode="
          << (observation.has_value() ? observation_mode_name(observation->mode)
                                      : "UNAVAILABLE")
          << " state_hard_stale_ratio="
          << (observation.has_value() ? observation->hard_stale_ratio : 1.0)
          << " engine_members=" << instance_mgr_->engine_member_count()
          << " engine_states=" << instance_mgr_->engine_state_count()
          << " engine_links=" << instance_mgr_->engine_link_count()
          << " engine_kv_reporting_engines=" << engine_kv.reporting_engines
          << " engine_kv_reporting_dp_ranks=" << engine_kv.reporting_dp_ranks
          << " engine_kv_capacity_known="
          << (engine_kv.has_used_ratio || engine_kv.has_free_blocks)
          << " engine_kv_max_used_ratio=" << engine_kv.max_used_ratio
          << " engine_kv_min_free_blocks=" << engine_kv.min_free_blocks
          << " engine_kv_total_free_blocks=" << engine_kv.total_free_blocks
          << " snapshot_window_ms=" << snapshot_window_ms
          << " queued_requests=" << flow.queued_requests
          << " dispatched_requests=" << flow.dispatched_requests
          << " active_requests=" << active_requests
          << " queued_prompt_tokens=" << flow.queued_prompt_tokens
          << " queued_bytes=" << flow.queued_bytes
          << " blind_probes_inflight=" << flow.blind_probes_inflight
          << " successful_terminals=" << successful_terminals
          << " successful_terminals_delta="
          << (last_snapshot_ms == 0
                  ? 0
                  : counter_delta(successful_terminals,
                                  previous_successful_terminals))
          << " failed_terminals=" << failed_terminals
          << " failed_terminals_delta="
          << (last_snapshot_ms == 0
                  ? 0
                  : counter_delta(failed_terminals, previous_failed_terminals))
          << " delivered_tokens=" << delivered_tokens
          << " delivered_tokens_delta="
          << (last_snapshot_ms == 0
                  ? 0
                  : counter_delta(delivered_tokens, previous_delivered_tokens))
          << " kv_decisions=" << kv.decisions << " kv_decisions_delta="
          << (last_snapshot_ms == 0
                  ? 0
                  : counter_delta(kv.decisions, previous_kv.decisions))
          << " kv_shadow_decisions=" << kv.shadow_decisions
          << " kv_enforced_decisions=" << kv.enforced_decisions
          << " kv_predicted_prefill_hit_tokens="
          << kv.predicted_prefill_hit_tokens
          << " kv_predicted_decode_hit_tokens="
          << kv.predicted_decode_hit_tokens
          << " kv_predicted_transfer_bytes=" << kv.predicted_transfer_bytes
          << " kv_shadow_prefill_host_hit_tokens_ub="
          << kv.shadow_prefill_host_hit_tokens_ub
          << " kv_shadow_decode_host_hit_tokens_ub="
          << kv.shadow_decode_host_hit_tokens_ub
          << " kv_actual_hit_tokens=" << kv.actual_hit_tokens
          << " kv_actual_decode_hit_tokens=" << kv.actual_decode_hit_tokens
          << " kv_skipped_transfer_bytes=" << kv.skipped_transfer_bytes
          << " kv_overpredicted_requests=" << kv.overpredicted_requests
          << " kv_decode_overpredicted_requests="
          << kv.decode_overpredicted_requests
          << " kv_admission_conflicts=" << kv.admission_conflicts
          << " events_recorded=" << stats.recorded
          << " events_dropped_capacity=" << stats.dropped_capacity
          << " events_dropped_contention=" << stats.dropped_contention
          << " events_invalid=" << stats.invalid_events
          << " invalid_metric_samples=" << stats.invalid_metric_samples
          << " duplicate_terminal_calls=" << stats.duplicate_terminal_calls;
      previous_successful_terminals = successful_terminals;
      previous_failed_terminals = failed_terminals;
      previous_delivered_tokens = delivered_tokens;
      previous_kv = kv;
      last_snapshot_ms = now_ms;
    }
    wait_lock.lock();
  }
  wait_lock.unlock();

  while (export_one_batch() > 0) {
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
         request_context = request.get(),
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
      } else if (!req_output.finished_on_prefill_instance ||
                 request_context->execution_mode ==
                     xllm::proto::EXECUTION_MODE_PREFILL_ONLY ||
                 request_context->execution_mode ==
                     xllm::proto::EXECUTION_MODE_LOCAL_PREFILL_DECODE) {
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
    if (!admit_flow_control_locked(request)) {
      request_deadline_queue_->erase(request->correlation.request_uid());
      rollback_request_safety_guards_locked(request);
      return false;
    }
    requests_.emplace(request->correlation.request_uid(), request);
    COUNTER_INC(server_request_in_total);
    record_request_event(request,
                         xllm::proto::REQUEST_EVENT_TYPE_INGRESS,
                         xllm::proto::EVENT_RESULT_ACCEPTED,
                         xllm::proto::ERROR_STAGE_NONE,
                         xllm::proto::EVENT_REASON_NONE,
                         "",
                         "",
                         elapsed_ns_since(request->trace_ingress_time));
  }

  {
    // allocate thread for the request
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    remote_requests_output_thread_map_[request->correlation.request_uid()] =
        next_thread_idx;
    next_thread_idx = (++next_thread_idx) % kOutputTheadNum_;
  }

  arm_client_disconnect_notification(request);
  request->dispatch_ready.store(true, std::memory_order_release);
  flow_dispatch_cv_.notify_one();

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
         request_context = request.get(),
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
      } else if (!req_output.finished_on_prefill_instance ||
                 request_context->execution_mode ==
                     xllm::proto::EXECUTION_MODE_PREFILL_ONLY ||
                 request_context->execution_mode ==
                     xllm::proto::EXECUTION_MODE_LOCAL_PREFILL_DECODE) {
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
    if (!admit_flow_control_locked(request)) {
      request_deadline_queue_->erase(request->correlation.request_uid());
      rollback_request_safety_guards_locked(request);
      return false;
    }
    requests_.emplace(request->correlation.request_uid(), request);
    COUNTER_INC(server_request_in_total);
    record_request_event(request,
                         xllm::proto::REQUEST_EVENT_TYPE_INGRESS,
                         xllm::proto::EVENT_RESULT_ACCEPTED,
                         xllm::proto::ERROR_STAGE_NONE,
                         xllm::proto::EVENT_REASON_NONE,
                         "",
                         "",
                         elapsed_ns_since(request->trace_ingress_time));
  }

  {
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    remote_requests_output_thread_map_[request->correlation.request_uid()] =
        next_thread_idx;
    next_thread_idx = (++next_thread_idx) % kOutputTheadNum_;
  }

  arm_client_disconnect_notification(request);
  request->dispatch_ready.store(true, std::memory_order_release);
  flow_dispatch_cv_.notify_one();

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
         request_context = request.get(),
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
      } else if (!req_output.finished_on_prefill_instance ||
                 request_context->execution_mode ==
                     xllm::proto::EXECUTION_MODE_PREFILL_ONLY ||
                 request_context->execution_mode ==
                     xllm::proto::EXECUTION_MODE_LOCAL_PREFILL_DECODE) {
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
    if (!admit_flow_control_locked(request)) {
      request_deadline_queue_->erase(request->correlation.request_uid());
      rollback_request_safety_guards_locked(request);
      return false;
    }
    requests_.emplace(request->correlation.request_uid(), request);
    COUNTER_INC(server_request_in_total);
    record_request_event(request,
                         xllm::proto::REQUEST_EVENT_TYPE_INGRESS,
                         xllm::proto::EVENT_RESULT_ACCEPTED,
                         xllm::proto::ERROR_STAGE_NONE,
                         xllm::proto::EVENT_REASON_NONE,
                         "",
                         "",
                         elapsed_ns_since(request->trace_ingress_time));
  }

  {
    // allocate thread for the request
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    remote_requests_output_thread_map_[request->correlation.request_uid()] =
        next_thread_idx;
    next_thread_idx = (++next_thread_idx) % kOutputTheadNum_;
  }

  arm_client_disconnect_notification(request);
  request->dispatch_ready.store(true, std::memory_order_release);
  flow_dispatch_cv_.notify_one();

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
        requests_drained_cv_.notify_all();
      }
    }
    if (request != nullptr) {
      finish_d_admission(request,
                         xllm::proto::EVENT_RESULT_FAILED,
                         xllm::proto::EVENT_REASON_MISSING_TERMINAL);
      detach_execution_hold_locked(request);
      record_resource_release(request);
      request->queue_state.store(RequestQueueState::TERMINAL,
                                 std::memory_order_release);
    }
  }

  flow_control_queue_->cancel(service_request_id);
  flow_dispatch_cv_.notify_one();

  if (request != nullptr) {
    if (!request->trace_terminal_recorded.load(std::memory_order_acquire)) {
      record_request_terminal(request,
                              error ? xllm::proto::EVENT_RESULT_FAILED
                                    : xllm::proto::EVENT_RESULT_SUCCEEDED,
                              error ? xllm::proto::ERROR_STAGE_OUTPUT
                                    : xllm::proto::ERROR_STAGE_NONE,
                              error ? xllm::proto::EVENT_REASON_PROVIDER_ERROR
                                    : xllm::proto::EVENT_REASON_NONE);
    }
    record_kv_route_actual(request, nullptr);
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
             ((it->second->routing.decode_name == instance_name &&
               it->second->decode_incarnation_id == incarnation_id) ||
              (it->second->execution_mode ==
                   xllm::proto::EXECUTION_MODE_LOCAL_PREFILL_DECODE &&
               it->second->routing.prefill_name == instance_name &&
               it->second->prefill_incarnation_id == incarnation_id)));
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
      if (!cleared_requests.empty()) {
        requests_drained_cv_.notify_all();
      }
    }

    for (const std::shared_ptr<Request>& request : cleared_requests) {
      finish_d_admission(request,
                         xllm::proto::EVENT_RESULT_FAILED,
                         xllm::proto::EVENT_REASON_STALE_STATE);
      const xllm::proto::ExecutionHolder holder = execution_holder(*request);
      if (holder.engine_uid() == instance_name &&
          holder.incarnation_id() == incarnation_id) {
        request->execution_hold.apply_convergence_proof(
            execution_attempt(*request),
            terminated_holder,
            xllm::proto::HOLDER_CONVERGENCE_PROOF_PROCESS_TERMINATED);
      }
      detach_execution_hold_locked(request);
      record_resource_release(request);
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
    record_request_terminal(request,
                            xllm::proto::EVENT_RESULT_CANCELLED,
                            xllm::proto::ERROR_STAGE_OUTPUT,
                            xllm::proto::EVENT_REASON_CANCELLED);
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
    binding.execution_mode = request->execution_mode;
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
  MULTI_COUNTER_INC(xllm_service_v2_output_sequence_total,
                    output_sequence_status_name(sequence_result.status));

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
    finish_d_admission_from_output(request, ready_output);
    const bool status_error =
        ready_output.status.has_value() && !ready_output.status->ok();
    const bool finished_on_prefill_instance =
        ready_output.finished_on_prefill_instance;
    const bool local_first_output =
        request->execution_mode ==
            xllm::proto::EXECUTION_MODE_LOCAL_PREFILL_DECODE &&
        ready_output.output_event_seq.has_value() &&
        *ready_output.output_event_seq == 0;
    const bool generation_commit =
        local_first_output || finished_on_prefill_instance;
    if (!status_error && generation_commit &&
        !confirm_generation_commit(request)) {
      fail_output_dispatch_locked(
          request, llm::StatusCode::UNKNOWN, "Invalid generation commit proof");
      return GenerationDeliveryResult(
          /*code=*/proto::GENERATION_DELIVERY_CODE_COMMIT_UNPROVEN,
          /*message=*/"Generation commit is not proven");
    }
    if (!status_error && ready_output.finished &&
        !resolve_terminal_execution_hold(request)) {
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
    if (status_error || ready_output.finished) {
      record_kv_route_actual(request, &ready_output);
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
          const std::optional<uint64_t> cumulative_output_tokens =
              ready_output.usage.has_value()
                  ? std::optional<uint64_t>(
                        ready_output.usage->num_generated_tokens)
                  : std::nullopt;
          const bool crosses_response_boundary =
              request->stream || !ready_output.finished_on_prefill_instance ||
              request->execution_mode ==
                  xllm::proto::EXECUTION_MODE_PREFILL_ONLY ||
              request->execution_mode ==
                  xllm::proto::EXECUTION_MODE_LOCAL_PREFILL_DECODE;
          if (!cb(std::move(ready_output)) || status_error) {
            finish_request(service_request_id, true);
            return;
          }
          if (crosses_response_boundary) {
            record_response_boundary(request, cumulative_output_tokens);
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

void Scheduler::record_kv_route_decision(
    const std::shared_ptr<Request>& request) {
  if (request == nullptr || !request->kv_route_observation.has_value()) {
    return;
  }
  request->kv_route_actual_recorded.store(false, std::memory_order_release);
  kv_route_metrics_.record_decision(*request->kv_route_observation);
}

void Scheduler::record_kv_route_actual(const std::shared_ptr<Request>& request,
                                       const llm::RequestOutput* output) {
  if (request == nullptr || !request->kv_route_observation.has_value() ||
      request->kv_route_actual_recorded.exchange(true,
                                                 std::memory_order_acq_rel)) {
    return;
  }
  provider::KVRouteActual actual;
  if (request->kv_route_observation->mode == provider::KVRouteMode::DISABLED) {
    actual.prefix_state = provider::PrefixMetricState::DISABLED;
    actual.decode_prefix_state = provider::PrefixMetricState::DISABLED;
  } else if (output != nullptr && output->usage.has_value() &&
             output->usage->num_cached_tokens <=
                 output->usage->num_prompt_tokens) {
    const uint64_t hit_tokens = output->usage->num_cached_tokens;
    actual.prefix_state = hit_tokens == 0
                              ? provider::PrefixMetricState::VALID_ZERO
                              : provider::PrefixMetricState::VALID_NONZERO;
    actual.actual_hit_tokens = hit_tokens;
    actual.actual_prefill_tokens =
        output->usage->num_prompt_tokens - hit_tokens;
    if (output->usage->num_decode_cached_tokens.has_value() &&
        *output->usage->num_decode_cached_tokens <=
            output->usage->num_prompt_tokens) {
      const uint64_t decode_hit_tokens =
          *output->usage->num_decode_cached_tokens;
      actual.decode_prefix_state =
          decode_hit_tokens == 0 ? provider::PrefixMetricState::VALID_ZERO
                                 : provider::PrefixMetricState::VALID_NONZERO;
      actual.actual_decode_hit_tokens = decode_hit_tokens;
      actual.skipped_transfer_bytes = provider::logical_skipped_transfer_bytes(
          decode_hit_tokens, request->kv_route_observation->kv_bytes_per_token);
    }
  }
  kv_route_metrics_.record_actual(*request->kv_route_observation, actual);
  if (!VLOG_IS_ON(1)) {
    return;
  }
  xllm::proto::RequestEvent event = make_request_event_base(request);
  event.set_event_type(xllm::proto::REQUEST_EVENT_TYPE_KV_TRANSFER);
  event.set_result(xllm::proto::EVENT_RESULT_SUCCEEDED);
  event.set_error_stage(xllm::proto::ERROR_STAGE_NONE);
  event.set_reason(xllm::proto::EVENT_REASON_NONE);
  event.set_stage_duration_validity(
      xllm::proto::METRIC_VALIDITY_NOT_APPLICABLE);
  event.set_stage_duration_invalid_reason(
      xllm::proto::INVALID_METRIC_REASON_NONE);
  if (actual.actual_hit_tokens.has_value()) {
    event.mutable_workload()->set_actual_prefill_hit_tokens(
        *actual.actual_hit_tokens);
  }
  if (actual.actual_decode_hit_tokens.has_value()) {
    event.mutable_workload()->set_actual_decode_hit_tokens(
        *actual.actual_decode_hit_tokens);
  }
  if (actual.skipped_transfer_bytes.has_value()) {
    event.mutable_workload()->set_skipped_transfer_bytes(
        *actual.skipped_transfer_bytes);
  }
  MULTI_COUNTER_INC(xllm_service_v2_request_lifecycle_total,
                    xllm::proto::RequestEventType_Name(event.event_type()));
  submit_request_event(std::move(event));
}

provider::KVRouteMetricsSnapshot Scheduler::kv_route_metrics_snapshot() const {
  return kv_route_metrics_.snapshot();
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

xllm::proto::RequestEvent Scheduler::make_request_event_base(
    const std::shared_ptr<Request>& request) {
  xllm::proto::RequestEvent event;
  event.set_schema_version(observability::kRequestEventSchemaVersion);
  *event.mutable_correlation() = request->correlation;
  event.set_event_seq(
      request->trace_next_event_seq.fetch_add(1, std::memory_order_relaxed));
  event.set_owner_role(xllm::proto::EVENT_OWNER_ROLE_SERVICE);
  event.set_owner_incarnation_id(service_incarnation_id_);
  event.set_build_id(options_.observability_build_id());
  if (!request->token_ids.empty()) {
    event.mutable_workload()->set_prompt_tokens(request->token_ids.size());
  }
  if (request->canonical_request.has_value()) {
    event.mutable_workload()->set_effective_max_new_tokens(
        request->canonical_request->effective_max_new_tokens());
  }
  if (request->kv_route_observation.has_value()) {
    event.mutable_workload()->set_predicted_prefill_hit_tokens(
        request->kv_route_observation->predicted_prefill_hit_tokens);
    event.mutable_workload()->set_predicted_decode_hit_tokens(
        request->kv_route_observation->predicted_decode_hit_tokens);
    event.mutable_workload()->set_predicted_transfer_bytes(
        request->kv_route_observation->predicted_transfer_bytes);
    event.mutable_workload()->set_shadow_prefill_host_hit_tokens_ub(
        request->kv_route_observation->shadow_prefill_host_hit_tokens_ub);
    event.mutable_workload()->set_shadow_decode_host_hit_tokens_ub(
        request->kv_route_observation->shadow_decode_host_hit_tokens_ub);
  }
  const uint64_t output_tokens =
      request->trace_output_tokens.load(std::memory_order_relaxed);
  if (output_tokens > 0) {
    event.mutable_workload()->set_output_tokens(output_tokens);
  }

  const xllm::proto::ProviderDescriptor* descriptor = nullptr;
  if (request->prefill_provider_descriptor.has_value()) {
    descriptor = &*request->prefill_provider_descriptor;
  } else if (request->decode_provider_descriptor.has_value()) {
    descriptor = &*request->decode_provider_descriptor;
  }
  if (descriptor != nullptr &&
      request->execution_mode != xllm::proto::EXECUTION_MODE_UNSPECIFIED) {
    xllm::proto::RuntimeProfileIdentity* profile =
        event.mutable_runtime_profile();
    profile->set_provider_id(descriptor->identity().provider_id());
    profile->set_runtime_version(descriptor->identity().runtime_version());
    profile->set_plugin_version(descriptor->identity().plugin_version());
    profile->set_hardware_runtime_version(
        descriptor->identity().hardware_runtime_version());
    profile->set_profile_digest(descriptor->profile_digest());
    profile->set_model_revision(descriptor->model().model_revision());
    profile->set_execution_mode(request->execution_mode);
  }
  return event;
}

void Scheduler::submit_request_event(xllm::proto::RequestEvent event) {
  const observability::RecordStatus status =
      request_event_recorder_->record(std::move(event));
  MULTI_COUNTER_INC(xllm_service_v2_observability_events_total,
                    observability::record_status_name(status));
}

void Scheduler::record_request_event(
    const std::shared_ptr<Request>& request,
    xllm::proto::RequestEventType event_type,
    xllm::proto::EventResult result,
    xllm::proto::ErrorStage error_stage,
    xllm::proto::EventReason reason,
    const std::string& target_engine_uid,
    const std::string& target_incarnation_id,
    std::optional<uint64_t> stage_duration_ns) {
  if (request == nullptr) {
    return;
  }
  MULTI_COUNTER_INC(xllm_service_v2_request_lifecycle_total,
                    xllm::proto::RequestEventType_Name(event_type));
  if (result == xllm::proto::EVENT_RESULT_REJECTED ||
      result == xllm::proto::EVENT_RESULT_FAILED ||
      result == xllm::proto::EVENT_RESULT_CANCELLED ||
      result == xllm::proto::EVENT_RESULT_DEADLINE_EXCEEDED) {
    MULTI_COUNTER_INC(xllm_service_v2_request_failure_total,
                      xllm::proto::EventReason_Name(reason));
  }
  // Match the repository's VLOG convention: low-cardinality counters remain
  // always on, while per-request protobuf construction and ring traffic are
  // paid only when detailed tracing is explicitly enabled with --v=1.
  if (!VLOG_IS_ON(1)) {
    return;
  }
  xllm::proto::RequestEvent event = make_request_event_base(request);
  event.set_event_type(event_type);
  event.set_result(result);
  event.set_error_stage(error_stage);
  event.set_reason(reason);
  event.set_target_engine_uid(target_engine_uid);
  event.set_target_incarnation_id(target_incarnation_id);
  if (stage_duration_ns.has_value()) {
    event.set_stage_duration_ns(*stage_duration_ns);
    event.set_stage_duration_validity(xllm::proto::METRIC_VALIDITY_VALID);
  } else {
    event.set_stage_duration_validity(
        xllm::proto::METRIC_VALIDITY_NOT_APPLICABLE);
  }
  event.set_stage_duration_invalid_reason(
      xllm::proto::INVALID_METRIC_REASON_NONE);
  submit_request_event(std::move(event));
}

void Scheduler::record_request_metric(const std::shared_ptr<Request>& request,
                                      xllm::proto::RequestMetric metric) {
  if (request == nullptr) {
    return;
  }
  if (!VLOG_IS_ON(1)) {
    return;
  }
  xllm::proto::RequestEvent event = make_request_event_base(request);
  event.set_event_type(xllm::proto::REQUEST_EVENT_TYPE_METRIC_SAMPLE);
  event.set_result(xllm::proto::EVENT_RESULT_SUCCEEDED);
  event.set_error_stage(xllm::proto::ERROR_STAGE_NONE);
  event.set_reason(xllm::proto::EVENT_REASON_NONE);
  event.set_stage_duration_validity(
      xllm::proto::METRIC_VALIDITY_NOT_APPLICABLE);
  event.set_stage_duration_invalid_reason(
      xllm::proto::INVALID_METRIC_REASON_NONE);
  *event.mutable_metric() = std::move(metric);
  submit_request_event(std::move(event));
}

void Scheduler::record_response_boundary(
    const std::shared_ptr<Request>& request,
    std::optional<uint64_t> cumulative_output_tokens) {
  const int64_t now_ns = monotonic_time_ns();
  if (!cumulative_output_tokens.has_value() || *cumulative_output_tokens == 0) {
    // A response chunk is not a token. Without the Provider's cumulative
    // generated-token count, TTFT/ITL/TPOT cannot be reported truthfully.
    return;
  }
  uint64_t observed =
      request->trace_output_tokens.load(std::memory_order_acquire);
  while (observed < *cumulative_output_tokens &&
         !request->trace_output_tokens.compare_exchange_weak(
             observed,
             *cumulative_output_tokens,
             std::memory_order_acq_rel,
             std::memory_order_acquire)) {
  }
  if (observed >= *cumulative_output_tokens) {
    // Terminal/control chunks can repeat the last cumulative usage. They do
    // not define a new inter-token boundary.
    return;
  }
  const uint64_t output_tokens = *cumulative_output_tokens;
  int64_t expected = 0;
  const bool first = request->trace_first_response_ns.compare_exchange_strong(
      expected, now_ns, std::memory_order_acq_rel, std::memory_order_acquire);
  const int64_t previous_ns = request->trace_last_response_ns.exchange(
      now_ns, std::memory_order_acq_rel);
  if (first) {
    request->trace_first_response_output_tokens.store(
        output_tokens, std::memory_order_release);
  }

  xllm::proto::RequestMetric metric;
  metric.set_measurement_boundary("service_response_write");
  metric.set_output_tokens(output_tokens);
  metric.set_invalid_reason(xllm::proto::INVALID_METRIC_REASON_NONE);
  const std::string mode = execution_mode_label(request->execution_mode);
  if (first) {
    const std::optional<uint64_t> ttft =
        elapsed_ns_since(request->trace_ingress_time);
    metric.set_kind(xllm::proto::REQUEST_METRIC_KIND_SERVER_TTFT);
    if (ttft.has_value()) {
      metric.set_validity(xllm::proto::METRIC_VALIDITY_VALID);
      metric.set_duration_ns(*ttft);
      MULTI_HISTOGRAM_OBSERVE(xllm_service_v2_ttft_milliseconds,
                              mode,
                              static_cast<int64_t>(*ttft / 1000000));
    } else {
      metric.set_validity(xllm::proto::METRIC_VALIDITY_INVALID);
      metric.set_invalid_reason(
          xllm::proto::INVALID_METRIC_REASON_CLOCK_REGRESSION);
    }
    record_request_metric(request, metric);
    record_request_event(request,
                         xllm::proto::REQUEST_EVENT_TYPE_FIRST_TOKEN_FLUSH,
                         xllm::proto::EVENT_RESULT_SUCCEEDED,
                         xllm::proto::ERROR_STAGE_NONE,
                         xllm::proto::EVENT_REASON_NONE,
                         "",
                         "",
                         ttft);
    return;
  }

  metric.set_kind(xllm::proto::REQUEST_METRIC_KIND_SERVER_ITL);
  if (previous_ns > 0 && now_ns >= previous_ns) {
    metric.set_validity(xllm::proto::METRIC_VALIDITY_VALID);
    metric.set_duration_ns(static_cast<uint64_t>(now_ns - previous_ns));
  } else {
    metric.set_validity(xllm::proto::METRIC_VALIDITY_INVALID);
    metric.set_invalid_reason(
        xllm::proto::INVALID_METRIC_REASON_CLOCK_REGRESSION);
  }
  record_request_metric(request, std::move(metric));
}

void Scheduler::record_request_terminal(const std::shared_ptr<Request>& request,
                                        xllm::proto::EventResult result,
                                        xllm::proto::ErrorStage error_stage,
                                        xllm::proto::EventReason reason) {
  bool expected = false;
  if (request == nullptr ||
      !request->trace_terminal_recorded.compare_exchange_strong(
          expected,
          true,
          std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    if (request != nullptr) {
      request_event_recorder_->note_duplicate_terminal();
      MULTI_COUNTER_INC(xllm_service_v2_observability_events_total,
                        "duplicate_terminal");
    }
    return;
  }
  MULTI_COUNTER_INC(xllm_service_v2_request_terminal_total,
                    xllm::proto::EventResult_Name(result));
  if (result == xllm::proto::EVENT_RESULT_SUCCEEDED) {
    saturated_atomic_add(&observability_successful_terminals_, 1);
  } else {
    saturated_atomic_add(&observability_failed_terminals_, 1);
  }
  saturated_atomic_add(
      &observability_delivered_tokens_,
      request->trace_output_tokens.load(std::memory_order_acquire));
  const std::optional<uint64_t> e2e =
      elapsed_ns_since(request->trace_ingress_time);
  record_request_event(request,
                       xllm::proto::REQUEST_EVENT_TYPE_REQUEST_TERMINAL,
                       result,
                       error_stage,
                       reason,
                       "",
                       "",
                       e2e);

  xllm::proto::RequestMetric metric;
  metric.set_kind(xllm::proto::REQUEST_METRIC_KIND_SERVER_E2E);
  metric.set_measurement_boundary("service_request_terminal");
  metric.set_output_tokens(
      request->trace_output_tokens.load(std::memory_order_acquire));
  if (e2e.has_value()) {
    metric.set_validity(xllm::proto::METRIC_VALIDITY_VALID);
    metric.set_invalid_reason(xllm::proto::INVALID_METRIC_REASON_NONE);
    metric.set_duration_ns(*e2e);
    MULTI_HISTOGRAM_OBSERVE(xllm_service_v2_e2e_milliseconds,
                            execution_mode_label(request->execution_mode),
                            static_cast<int64_t>(*e2e / 1000000));
  } else {
    metric.set_validity(xllm::proto::METRIC_VALIDITY_INVALID);
    metric.set_invalid_reason(
        xllm::proto::INVALID_METRIC_REASON_CLOCK_REGRESSION);
  }
  record_request_metric(request, std::move(metric));

  const uint64_t first_response_output_tokens =
      request->trace_first_response_output_tokens.load(
          std::memory_order_acquire);
  const uint64_t output_tokens =
      request->trace_output_tokens.load(std::memory_order_acquire);
  const int64_t first_ns =
      request->trace_first_response_ns.load(std::memory_order_acquire);
  const int64_t last_ns =
      request->trace_last_response_ns.load(std::memory_order_acquire);
  metric.Clear();
  metric.set_kind(xllm::proto::REQUEST_METRIC_KIND_SERVER_TPOT);
  metric.set_measurement_boundary("service_response_write");
  metric.set_output_tokens(output_tokens);
  if (output_tokens <= first_response_output_tokens) {
    metric.set_validity(xllm::proto::METRIC_VALIDITY_NOT_APPLICABLE);
    metric.set_invalid_reason(
        xllm::proto::INVALID_METRIC_REASON_INSUFFICIENT_OUTPUT_TOKENS);
  } else if (last_ns < first_ns || first_ns <= 0) {
    metric.set_validity(xllm::proto::METRIC_VALIDITY_INVALID);
    metric.set_invalid_reason(
        xllm::proto::INVALID_METRIC_REASON_CLOCK_REGRESSION);
  } else {
    const uint64_t tpot_ns = static_cast<uint64_t>(last_ns - first_ns) /
                             (output_tokens - first_response_output_tokens);
    if (tpot_ns == 0) {
      metric.set_validity(xllm::proto::METRIC_VALIDITY_INVALID);
      metric.set_invalid_reason(xllm::proto::INVALID_METRIC_REASON_ZERO_TPOT);
    } else {
      metric.set_validity(xllm::proto::METRIC_VALIDITY_VALID);
      metric.set_invalid_reason(xllm::proto::INVALID_METRIC_REASON_NONE);
      metric.set_duration_ns(tpot_ns);
      MULTI_HISTOGRAM_OBSERVE(xllm_service_v2_tpot_milliseconds,
                              execution_mode_label(request->execution_mode),
                              static_cast<int64_t>(tpot_ns / 1000000));
    }
  }
  record_request_metric(request, std::move(metric));

  if (result == xllm::proto::EVENT_RESULT_SUCCEEDED &&
      placement_observation_collector_ != nullptr) {
    std::optional<double> ttft_ms;
    std::optional<double> tpot_ms;
    const int64_t ingress_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            request->trace_ingress_time.time_since_epoch())
            .count();
    if (first_ns >= ingress_ns && ingress_ns > 0) {
      ttft_ms = static_cast<double>(first_ns - ingress_ns) / 1000000.0;
    }
    if (output_tokens > first_response_output_tokens && first_ns > 0 &&
        last_ns >= first_ns) {
      tpot_ms =
          static_cast<double>(last_ns - first_ns) /
          static_cast<double>(output_tokens - first_response_output_tokens) /
          1000000.0;
    }
    const placement::PlacementObservationStatus observation_status =
        placement_observation_collector_->record_terminal(request->model,
                                                          output_tokens,
                                                          ttft_ms,
                                                          tpot_ms,
                                                          monotonic_time_ms());
    MULTI_COUNTER_INC(
        xllm_service_v3_placement_observations_total,
        placement::placement_observation_status_name(observation_status));
  }
}

bool Scheduler::has_available_instances() const {
  return instance_mgr_->has_available_instances();
}

void Scheduler::refresh_readiness() {
  const uint64_t now_monotonic_ms = monotonic_time_ms();
  provider::ReadinessInput input;
  input.has_accepted_full_snapshot =
      instance_mgr_->has_accepted_engine_state_full_snapshot();
  input.has_compatible_capacity =
      instance_mgr_->has_available_instances_at(now_monotonic_ms);
  input.draining = draining_.load(std::memory_order_acquire);
  input.observation =
      instance_mgr_->engine_observation_snapshot(now_monotonic_ms);

  std::string error;
  std::lock_guard<std::mutex> lock(readiness_mutex_);
  const std::optional<provider::ReadinessSnapshot> snapshot =
      readiness_controller_.update(input, now_monotonic_ms, &error);
  if (!snapshot.has_value()) {
    readiness_snapshot_ = provider::ReadinessSnapshot{
        .accepting_new_requests = false,
        .reason = provider::ReadinessReason::OBSERVATION_UNAVAILABLE,
        .changed_monotonic_ms = now_monotonic_ms,
    };
    accepting_new_requests_.store(false, std::memory_order_release);
    LOG(ERROR) << "Readiness update failed closed: " << error;
    return;
  }
  const bool changed = snapshot->accepting_new_requests !=
                           readiness_snapshot_.accepting_new_requests ||
                       snapshot->reason != readiness_snapshot_.reason;
  readiness_snapshot_ = *snapshot;
  accepting_new_requests_.store(snapshot->accepting_new_requests,
                                std::memory_order_release);
  if (changed) {
    LOG(INFO) << "Service readiness changed, accepting_new_requests="
              << snapshot->accepting_new_requests << ", reason="
              << provider::readiness_reason_name(snapshot->reason);
  }
}

void Scheduler::set_draining(bool draining) {
  draining_.store(draining, std::memory_order_release);
  flow_dispatch_cv_.notify_all();
}

bool Scheduler::wait_for_requests_drained(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(request_mutex_);
  return requests_drained_cv_.wait_for(
      lock, timeout, [this] { return requests_.empty(); });
}

bool Scheduler::accepting_new_requests() const {
  return accepting_new_requests_.load(std::memory_order_acquire);
}

provider::ReadinessSnapshot Scheduler::readiness_status() const {
  std::lock_guard<std::mutex> lock(readiness_mutex_);
  return readiness_snapshot_;
}

void Scheduler::exited() {
  set_draining(true);
  exited_.store(true, std::memory_order_release);
}

}  // namespace xllm_service
