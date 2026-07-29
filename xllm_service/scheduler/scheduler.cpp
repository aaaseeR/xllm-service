/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "common/metrics.h"
#include "common/utils.h"
#include "common/xllm/status.h"
#include "loadbalance_policy/cache_aware_routing.h"
#include "loadbalance_policy/round_robin.h"
#include "loadbalance_policy/slo_aware_policy.h"
#include "tokenizer/tokenizer_factory.h"

#include <utility>

namespace {
constexpr int32_t kHeartbeatInterval = 3;  // in seconds

constexpr const char* kEtcdUsernameEnvVar = "ETCD_USERNAME";
constexpr const char* kEtcdPasswordEnvVar = "ETCD_PASSWORD";
}  // namespace

namespace xllm_service {

Scheduler::Scheduler(const Options& options)
    : options_(options),
      lifecycle_events_([this](const InstanceLifecycleEvent& event) {
        handle_instance_lifecycle_event(event);
      }) {
  GAUGE_SET(peer_service_enabled, options_.enable_peer_service() ? 1.0 : 0.0);

  tokenizer_ = TokenizerFactory::create_tokenizer(options_.tokenizer_path(),
                                                  &tokenizer_args_);
  chat_template_ = std::make_unique<JinjaChatTemplate>(tokenizer_args_);

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

  if (!options_.enable_peer_service() &&
      !etcd_client_->get(ETCD_MASTER_SERVICE_KEY, nullptr)) {
    is_master_service_ = etcd_client_->set(
        ETCD_MASTER_SERVICE_KEY, options_.service_name(), kHeartbeatInterval);
    if (is_master_service_) {
      LOG(INFO) << "Set current service as master!";
    }
  }

  global_kvcache_mgr_ = std::make_shared<GlobalKVCacheMgr>(
      options, etcd_client_, is_master_service_);

  if (options_.enable_peer_service() && options_.kv_event_zmq_enable()) {
    KvEventSubscriber::Options subscriber_options;
    subscriber_options.enabled(true)
        .poll_interval_ms(options_.kv_event_zmq_poll_interval_ms())
        .reconnect_interval_ms(options_.kv_event_zmq_reconnect_interval_ms())
        .reconnect_interval_max_ms(
            options_.kv_event_zmq_reconnect_interval_max_ms())
        .record_callback([this](const std::string& instance_name,
                                const proto::KvCacheEvent& cache_event) {
          record_instance_cache_event(instance_name, cache_event);
        })
        .snapshot_callback([this](const std::string& instance_name,
                                  const proto::KvCacheEvent& cache_event) {
          replace_instance_cache_snapshot(instance_name, cache_event);
        })
        .clear_callback([this](const std::string& instance_name) {
          clear_instance_cache(instance_name);
        });
    kv_event_subscriber_ =
        std::make_unique<KvEventSubscriber>(std::move(subscriber_options));
    kv_event_subscriber_->start();
  }

  instance_mgr_ = std::make_shared<InstanceMgr>(
      options, etcd_client_, is_master_service_, lifecycle_events_);

  request_registry_ = std::make_unique<RequestSessionRegistry>(
      [this](const std::shared_ptr<Request>& request,
             const llm::RequestOutput& output) {
        observe_session_generation(request, output);
      },
      [this](const std::shared_ptr<Request>& request,
             RequestTerminalReason reason) {
        handle_session_terminal(request, reason);
      });

  if (options.load_balance_policy() == "CAR") {
    lb_policy_ =
        std::make_unique<CacheAwareRouting>(instance_mgr_, global_kvcache_mgr_);
  } else if (options.load_balance_policy() == "SLO_AWARE") {
    lb_policy_ = std::make_unique<SloAwarePolicy>(options, instance_mgr_);
  } else {
    lb_policy_ = std::make_unique<RoundRobin>(instance_mgr_);
  }

  if (options_.enable_peer_service()) {
    LOG(INFO) << "Peer service mode enabled; skip master election and "
                 "metrics/cache etcd upload paths.";
  } else if (is_master_service_) {
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

Scheduler::~Scheduler() {
  exited_ = true;
  if (etcd_client_ != nullptr) {
    etcd_client_->stop_watch();
  }
  if (heartbeat_thread_ && heartbeat_thread_->joinable()) {
    heartbeat_thread_->join();
  }
  lifecycle_events_.close();
  if (request_registry_ != nullptr) {
    request_registry_->close();
  }
  lb_policy_.reset();
  instance_mgr_.reset();
  if (kv_event_subscriber_ != nullptr) {
    kv_event_subscriber_->stop();
  }
}

bool Scheduler::schedule(std::shared_ptr<Request> request) {
  // apply chat template
  if (request->messages.size() > 0) {
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
  if (request->prompt.size() != 0) {
    if (!get_tls_tokenizer()->encode(request->prompt, &request->token_ids)) {
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
  if (request->prompt.size() != 0) {
    instance_mgr_->update_request_metrics(request, RequestAction::SCHEDULE);
    instance_mgr_->record_dispatch(request);
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

  LOG(ERROR) << "Service key already exists, registration failed: "
             << service_key
             << ". Please ensure service_name is unique across xllm_service "
                "instances.";
  return false;
}

bool Scheduler::handle_instance_heartbeat(const proto::HeartbeatRequest* req) {
  if (exited_) {
    return false;
  }
  COUNTER_INC(xservice_heartbeat_total);
  if (req->has_xtensor_info()) {
    COUNTER_INC(xservice_heartbeat_xtensor_total);
  }
  if (!instance_mgr_->record_instance_heartbeat(req->name(),
                                                req->incarnation_id())) {
    return false;
  }
  const auto& cache_event = req->cache_event();
  const bool has_heartbeat_cache_event = cache_event.stored_cache_size() > 0 ||
                                         cache_event.removed_cache_size() > 0 ||
                                         cache_event.offload_cache_size() > 0;
  if (!(options_.enable_peer_service() && options_.kv_event_zmq_enable()) ||
      has_heartbeat_cache_event) {
    global_kvcache_mgr_->record_updated_kvcaches(req->name(),
                                                 cache_event);
  }
  instance_mgr_->record_load_metrics_update(
      req->name(), req->incarnation_id(), req->load_metrics());
  instance_mgr_->update_latency_metrics(
      req->name(), req->incarnation_id(), req->latency_metrics());
  return true;
}

void Scheduler::handle_master_service_watch(const etcd::Response& response,
                                            const uint64_t& prefix_len) {
  if (options_.enable_peer_service() || exited_ || response.events().empty()) {
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

    if (deleted_service == ETCD_XSERVICE_KEY_PREFIX + options_.service_name()) {
      LOG(INFO) << "Current xllm_service registration expired, re-registering";
      register_current_service();
      continue;
    }

    if (!options_.enable_peer_service()) {
      if (deleted_service == ETCD_MASTER_SERVICE_KEY) {
        continue;
      }

      if (!is_master_service_) {
        continue;
      }
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

bool Scheduler::record_new_request(std::shared_ptr<ChatCallData> call_data,
                                   std::shared_ptr<Request> request) {
  request->latest_generate_time = absl::Now();
  auto tools_for_parse =
      (request->tool_choice == "none" ? std::vector<JsonTool>{}
                                      : request->tools);
  auto tool_call_parser_pref = options_.tool_call_parser();
  auto reasoning_parser_pref = options_.reasoning_parser();
  std::shared_ptr<ChatStreamParseState> stream_state;
  if (request->stream) {
    stream_state = response_handler_.create_chat_stream_parse_state(
        tools_for_parse,
        request->model,
        tool_call_parser_pref,
        reasoning_parser_pref);
  }

  OutputCallback output_callback =
      [this,
       call_data,
       model = request->model,
       stream = request->stream,
       include_usage = request->include_usage,
       tools = std::move(tools_for_parse),
       tool_call_parser = std::move(tool_call_parser_pref),
       reasoning_parser = std::move(reasoning_parser_pref),
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
                                                       reasoning_parser);
      }
      return true;
  };

  if (!request_registry_->register_request(
          request,
          std::move(output_callback),
          [call_data]() { return call_data->is_disconnected(); })) {
    LOG(ERROR) << "The request ID already exists or the registry is closed: "
               << request->service_request_id;
    instance_mgr_->record_request_finished(request);
    instance_mgr_->update_request_metrics(request, RequestAction::CANCEL);
    return false;
  }
  COUNTER_INC(server_request_in_total);

  return true;
}

bool Scheduler::record_new_request(
    std::shared_ptr<CompletionCallData> call_data,
    std::shared_ptr<Request> request) {
  request->latest_generate_time = absl::Now();
  OutputCallback output_callback =
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

  if (!request_registry_->register_request(
          request,
          std::move(output_callback),
          [call_data]() { return call_data->is_disconnected(); })) {
    LOG(ERROR) << "The request ID already exists or the registry is closed: "
               << request->service_request_id;
    instance_mgr_->record_request_finished(request);
    instance_mgr_->update_request_metrics(request, RequestAction::CANCEL);
    return false;
  }
  COUNTER_INC(server_request_in_total);

  return true;
}

bool Scheduler::handle_transport_failure(
    const std::string& service_request_id,
    const std::string& message) {
  return request_registry_->on_transport_failure(service_request_id, message);
}

void Scheduler::handle_instance_lifecycle_event(
    const InstanceLifecycleEvent& event) {
  const InstanceMetaInfo& instance = event.instance;
  switch (event.type) {
    case InstanceLifecycleEventType::REGISTERED:
      add_kv_event_source(instance);
      return;
    case InstanceLifecycleEventType::REGISTRATION_UPDATED:
      clear_instance_cache(instance.name);
      remove_kv_event_source(instance.name, instance.incarnation_id);
      add_kv_event_source(instance);
      return;
    case InstanceLifecycleEventType::DEREGISTERING:
      remove_kv_event_source(instance.name, instance.incarnation_id);
      return;
    case InstanceLifecycleEventType::DEREGISTERED:
      clear_requests_on_failed_instance(
          instance.name,
          instance.incarnation_id,
          instance.type == InstanceType::MIX ? instance.current_type
                                             : instance.type);
      clear_instance_cache(instance.name);
      return;
  }
}

void Scheduler::clear_requests_on_failed_instance(
    const std::string& instance_name,
    const std::string& incarnation_id,
    InstanceType type) {
  request_registry_->on_instance_failure(
      {instance_name, incarnation_id, type});
}

void Scheduler::clear_instance_cache(const std::string& instance_name) {
  if (global_kvcache_mgr_ != nullptr) {
    global_kvcache_mgr_->clear_instance_cache(instance_name);
  }
}

void Scheduler::add_kv_event_source(const InstanceMetaInfo& info) {
  if (kv_event_subscriber_ != nullptr) {
    kv_event_subscriber_->add_or_update_source(info);
  }
}

void Scheduler::remove_kv_event_source(
    const std::string& instance_name,
    const std::string& incarnation_id) {
  if (kv_event_subscriber_ != nullptr) {
    kv_event_subscriber_->remove_source(instance_name, incarnation_id);
  }
}

void Scheduler::record_instance_cache_event(
    const std::string& instance_name,
    const proto::KvCacheEvent& cache_event) {
  if (global_kvcache_mgr_ != nullptr) {
    global_kvcache_mgr_->record_updated_kvcaches(instance_name, cache_event);
  }
}

void Scheduler::replace_instance_cache_snapshot(
    const std::string& instance_name,
    const proto::KvCacheEvent& cache_event) {
  if (global_kvcache_mgr_ != nullptr) {
    global_kvcache_mgr_->replace_instance_kvcaches(instance_name, cache_event);
  }
}

bool Scheduler::handle_generation(const llm::RequestOutput& request_output) {
  const GenerationDispatchResult result =
      request_registry_->on_generation(request_output);
  if (result == GenerationDispatchResult::ACCEPTED) {
    return true;
  }
  if (result == GenerationDispatchResult::CLIENT_DISCONNECTED) {
    LOG(INFO) << "Client disconnected; request session was cancelled, "
              << "request id: " << request_output.service_request_id;
    return false;
  }
  if (result == GenerationDispatchResult::REGISTRY_CLOSED) {
    LOG(WARNING) << "Generation ignored because request registry is closed, "
                 << "request id: " << request_output.service_request_id;
    return false;
  }
  LOG(ERROR) << "Cannot find an active session for generation, request id: "
             << request_output.service_request_id;
  return false;
}

void Scheduler::observe_session_generation(
    const std::shared_ptr<Request>& request,
    const llm::RequestOutput& output) {
  update_request_metrics(request, output.finished_on_prefill_instance);
  update_token_latency_metrics(request, output.finished_on_prefill_instance);
}

void Scheduler::handle_session_terminal(
    const std::shared_ptr<Request>& request,
    RequestTerminalReason reason) {
  instance_mgr_->record_request_finished(request);
  instance_mgr_->update_request_metrics(
      request,
      reason == RequestTerminalReason::COMPLETED
          ? RequestAction::FINISH_DECODE
          : RequestAction::CANCEL);
  if (reason != RequestTerminalReason::COMPLETED) {
    LOG(INFO) << "Request session terminated, request id: "
              << request->service_request_id
              << ", reason: " << request_terminal_reason_name(reason);
  }
}

void Scheduler::update_request_metrics(std::shared_ptr<Request> request,
                                       bool finished_on_prefill_instance) {
  request->num_generated_tokens += 1;
  if (finished_on_prefill_instance) {
    instance_mgr_->record_prefill_finished(request);
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

nlohmann::json Scheduler::debug_summary() const {
  nlohmann::json summary;
  summary["service_name"] = options_.service_name();
  summary["enable_peer_service"] = options_.enable_peer_service();
  summary["is_master_service"] = is_master_service_;
  summary["instance_view"] =
      instance_mgr_ ? instance_mgr_->debug_summary() : nlohmann::json::object();
  summary["cache_index"] = global_kvcache_mgr_
                               ? global_kvcache_mgr_->debug_summary()
                               : nlohmann::json::object();
  summary["kv_event_subscriber"] =
      kv_event_subscriber_ ? kv_event_subscriber_->debug_summary()
                           : nlohmann::json::object();
  summary["active_request_sessions"] =
      request_registry_ ? request_registry_->size() : 0;
  return summary;
}

}  // namespace xllm_service
