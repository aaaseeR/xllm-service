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

#include <brpc/controller.h>

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
}  // namespace

namespace xllm_service {

Scheduler::Scheduler(const Options& options)
    : options_(options),
      service_incarnation_id_(llm::new_uuid_v7()),
      execution_hold_cleanup_table_(
          std::make_unique<provider::ExecutionHoldCleanupTable>(
              execution_hold_config(options))) {
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
}

Scheduler::~Scheduler() { etcd_client_->stop_watch(); }

bool Scheduler::schedule(std::shared_ptr<Request> request) {
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
    return true;
  }
  LOG(ERROR) << "Failed to install execution hold before dispatch, request_uid="
             << request->correlation.request_uid()
             << ", status=" << static_cast<int>(status);
  return false;
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

void Scheduler::cancel_or_detach_execution_hold_locked(
    const std::shared_ptr<Request>& request) {
  std::optional<xllm::proto::ExecutionResourceHold> hold =
      request->execution_hold.snapshot();
  if (!hold.has_value()) {
    return;
  }

  for (const xllm::proto::ExecutionHolder& holder : hold->potential_holders()) {
    const std::shared_ptr<brpc::Channel> channel =
        instance_mgr_->get_channel(holder.engine_uid());
    if (channel == nullptr) {
      continue;
    }

    xllm::proto::AttemptControlRequest cancel_request;
    xllm::proto::RequestAttemptKey* key = cancel_request.mutable_key();
    key->set_request_uid(hold->attempt().request_uid());
    if (hold->attempt().has_attempt_seq()) {
      key->set_attempt_seq(hold->attempt().attempt_seq());
    }
    key->set_incarnation_id(holder.incarnation_id());
    xllm::proto::AttemptControlResponse cancel_response;
    brpc::Controller controller;
    if (options_.instance_delete_probe_timeout_ms() > 0) {
      controller.set_timeout_ms(options_.instance_delete_probe_timeout_ms());
    }
    xllm::proto::DisaggPDService_Stub stub(channel.get());
    stub.CancelRequest(&controller, &cancel_request, &cancel_response, nullptr);
    if (controller.Failed() || !cancel_response.ok() ||
        !same_attempt_key(
            cancel_response.status().key(), hold->attempt(), holder) ||
        !terminal_attempt_state(cancel_response.status().state())) {
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
    if (!install_execution_hold_locked(request)) {
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
    if (!install_execution_hold_locked(request)) {
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
    requests_.emplace(request->correlation.request_uid(), request);
    COUNTER_INC(server_request_in_total);
  }

  {
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    remote_requests_output_thread_map_[request->correlation.request_uid()] =
        next_thread_idx;
    next_thread_idx = (++next_thread_idx) % kOutputTheadNum_;
  }

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
    if (!install_execution_hold_locked(request)) {
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
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    remote_requests_output_thread_map_.erase(service_request_id);
  }
}

void Scheduler::clear_requests_on_failed_instance(
    const std::string& instance_name,
    const std::string& incarnation_id,
    InstanceType type) {
  std::vector<std::shared_ptr<Request>> cleared_requests;
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
             !it->second->prefill_stage_finished);
        const bool clear_decode =
            (type == InstanceType::DECODE &&
             it->second->routing.decode_name == instance_name &&
             it->second->decode_incarnation_id == incarnation_id);
        if (clear_prefill || clear_decode) {
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

  for (const std::shared_ptr<Request>& request : cleared_requests) {
    llm::RequestOutput req_output;
    req_output.status = llm::Status(llm::StatusCode::CANCELLED,
                                    "Instance is failed and deleted");
    request->output_callback(req_output);
    LOG(INFO) << "Clear request on failed instance: " << instance_name
              << ", incarnation_id: " << incarnation_id
              << ", service_request_id: " << request->correlation.request_uid();
  }

  if (!cleared_requests.empty()) {
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    for (const std::shared_ptr<Request>& request : cleared_requests) {
      remote_requests_output_thread_map_.erase(
          request->correlation.request_uid());
    }
  }
}

bool Scheduler::handle_generation(const llm::RequestOutput& request_output) {
  bool finished_on_prefill_instance =
      request_output.finished_on_prefill_instance;
  const std::string& service_request_id = request_output.service_request_id;
  bool status_error =
      request_output.status.has_value() && !request_output.status.value().ok();

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
      return false;
    }
    request = it->second;
    cb = request->output_callback;

    // check client connection
    if (request->call_data->is_disconnected()) {
      LOG(INFO) << "Client has disconnected and the request will be cancelled, "
                   "request id: "
                << service_request_id;
      client_disconnected = true;
    }
  }

  if (client_disconnected) {
    finish_request(service_request_id, /*error=*/true);
    return false;
  }

  if (!status_error && finished_on_prefill_instance &&
      !confirm_generation_commit(request)) {
    finish_request(service_request_id, true);
    return false;
  }
  if (!status_error && !finished_on_prefill_instance &&
      request_output.finished && !resolve_terminal_execution_hold(request)) {
    finish_request(service_request_id, true);
    return false;
  }

  if (!status_error) {
    // no error, update instance request metrics
    update_request_metrics(request, finished_on_prefill_instance);
    update_token_latency_metrics(request, finished_on_prefill_instance);
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
    finish_request(service_request_id, true);
    return false;
  }

  output_threadpools_[req_thread_idx].schedule(
      [this,
       service_request_id,
       cb,
       status_error,
       request_output = std::move(request_output)]() mutable {
        if (!cb(request_output) || status_error) {
          finish_request(service_request_id, true);
          return;
        }
        if (request_output.finished) {
          finish_request(service_request_id);
          return;
        }
      });

  return true;
}

void Scheduler::update_request_metrics(std::shared_ptr<Request> request,
                                       bool finished_on_prefill_instance) {
  request->num_generated_tokens += 1;
  if (finished_on_prefill_instance) {
    request->prefill_stage_finished = true;
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
