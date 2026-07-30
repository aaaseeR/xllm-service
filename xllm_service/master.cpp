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

#include "master.h"

#include <chrono>
#include <csignal>

#include "common/global_gflags.h"
#include "common/types.h"
#include "common/utils.h"

namespace xllm_service {

Master::Master(const Options& options) : options_(options) {
  channel_pool_ = std::make_shared<ChannelPool>(options);
  scheduler_ = std::make_unique<Scheduler>(
      options,
      [channel_pool = channel_pool_](const InstanceLifecycleEvent& event) {
        const InstanceMetaInfo& instance = event.instance;
        switch (event.type) {
          case InstanceLifecycleEventType::REGISTERED:
          case InstanceLifecycleEventType::REGISTRATION_UPDATED:
            channel_pool->activate(instance.name, instance.incarnation_id);
            return;
          case InstanceLifecycleEventType::DEREGISTERING:
            return;
          case InstanceLifecycleEventType::DEREGISTERED:
            channel_pool->remove(instance.name, instance.incarnation_id);
            return;
        }
      });

  dispatcher_ = std::make_unique<Dispatcher>(
      channel_pool_,
      [scheduler = scheduler_.get()](const TransportResult& result) {
        if (result.code == TransportResultCode::SUCCESS) {
          return;
        }
        scheduler->handle_transport_failure(result.request_id,
                                            result.failure_stage,
                                            result.message);
      });

  rpc_service_ =
      std::make_unique<xllm_service::XllmRpcService>(options, scheduler_.get());

  http_service_ = std::make_unique<xllm_service::XllmHttpServiceImpl>(
      options, scheduler_.get(), runtime_state_, dispatcher_.get());
}

Master::~Master() { stop(); }

bool Master::start() {
  if (stopped_.load()) {
    LOG(ERROR) << "Cannot restart a stopped master.";
    return false;
  }
  if (!setup_http_server()) {
    return false;
  }

  if (http_server_.Start(http_endpoint_, &http_options_) != 0) {
    LOG(ERROR) << "Failed to start HTTP server on: " << http_endpoint_;
    return false;
  }
  http_server_started_ = true;
  LOG(INFO) << "HTTP server started on: " << http_endpoint_;

  if (!start_rpc_server()) {
    stop();
    return false;
  }

  runtime_state_.set_backend_ready(
      scheduler_->has_available_instances(),
      "no schedulable xLLM endpoint is available");
  runtime_state_.mark_running();
  readiness_thread_ = std::make_unique<std::thread>(
      [this]() { reconcile_runtime_readiness(); });

  return true;
}

void Master::stop() {
  if (stopped_.exchange(true)) {
    return;
  }

  runtime_state_.begin_draining();

  if (http_server_started_) {
    http_server_.Stop(0);
  }
  if (rpc_server_started_) {
    rpc_server_.Stop(0);
  }

  if (readiness_thread_ && readiness_thread_->joinable()) {
    readiness_thread_->join();
  }

  // Backend callbacks must release inbound RPCs before Server::Join waits.
  dispatcher_->close();
  scheduler_->cancel_active_requests();

  if (http_server_started_) {
    http_server_.Join();
    http_server_started_ = false;
  }
  if (rpc_server_started_) {
    rpc_server_.Join();
    rpc_server_started_ = false;
  }

  runtime_state_.mark_stopped();
}

bool Master::setup_http_server() {
  if (http_server_.AddService(http_service_.get(),
                              brpc::SERVER_DOESNT_OWN_SERVICE,
                              // for testing
                              "/hello => Hello,"
                              "/health => Health,"
                              "/livez => Liveness,"
                              "/readyz => Readiness,"
                              "/v1/completions => Completions,"
                              "/v1/chat/completions => ChatCompletions,"
                              "/v1/embeddings => Embeddings,"
                              "/v1/models => Models,"
                              "/metrics => Metrics,"
                              "/debug/summary => DebugSummary,") != 0) {
    LOG(ERROR) << "Failed to add HTTP service.";
    return false;
  }

  http_options_.idle_timeout_sec = options_.http_idle_timeout_s();
  http_options_.num_threads = options_.http_num_threads();
  http_options_.max_concurrency = options_.http_max_concurrency();
  http_options_.has_builtin_services = false;

  if (!options_.server_host().empty()) {
    http_server_address_ =
        options_.server_host() + ":" + std::to_string(options_.http_port());
    if (butil::str2endpoint(http_server_address_.c_str(), &http_endpoint_) <
        0) {
      LOG(ERROR) << "Convert server address to endpoint failed: "
                 << http_server_address_;
      return false;
    }
  } else {
    http_endpoint_ = butil::EndPoint(butil::IP_ANY, options_.http_port());
  }

  return true;
}

void Master::reconcile_runtime_readiness() {
  while (!stopped_.load()) {
    runtime_state_.set_backend_ready(
        scheduler_->has_available_instances(),
        "no schedulable xLLM endpoint is available");

    const auto end_time =
        std::chrono::steady_clock::now() +
        std::chrono::seconds(FLAGS_readiness_check_interval_s);
    while (!stopped_.load() && std::chrono::steady_clock::now() < end_time) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
}

bool Master::start_rpc_server() {
  if (rpc_server_.AddService(rpc_service_.get(),
                             brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    LOG(ERROR) << "Failed to add RPC service.";
    return false;
  }

  brpc::ServerOptions options;
  options.idle_timeout_sec = options_.rpc_idle_timeout_s();
  options.num_threads = options_.rpc_num_threads();
  options.max_concurrency = options_.rpc_max_concurrency();

  butil::EndPoint endpoint;
  if (!options_.server_host().empty()) {
    rpc_server_address_ =
        options_.server_host() + ":" + std::to_string(options_.rpc_port());
    if (butil::str2endpoint(rpc_server_address_.c_str(), &endpoint) < 0) {
      LOG(ERROR) << "Convert server address to endpoint failed: "
                 << rpc_server_address_;
      return false;
    }
  } else {
    endpoint = butil::EndPoint(butil::IP_ANY, options_.rpc_port());
  }

  if (rpc_server_.Start(endpoint, &options) != 0) {
    LOG(ERROR) << "Failed to start RPC server on: " << endpoint;
    return false;
  }

  LOG(INFO) << "Xllm rpc server started on: " << endpoint;
  rpc_server_started_ = true;
  return true;
}

}  // namespace xllm_service

static volatile std::sig_atomic_t g_shutdown_signal = 0;
void shutdown_handler(int signal) {
  g_shutdown_signal = signal;
}

int main(int argc, char* argv[]) {
  // Initialize gflags
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Initialize glog
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;

  LOG(INFO) << "Starting xllm master service.";

  // check port available or not
  if (!xllm_service::utils::is_port_available(FLAGS_http_server_port)) {
    LOG(ERROR)
        << "Http server port " << FLAGS_http_server_port
        << " is already in use. "
        << "Please specify a different port using --http_server_port flag.";
    return -1;
  }
  if (!xllm_service::utils::is_port_available(FLAGS_rpc_server_port)) {
    LOG(ERROR)
        << "Rpc server port " << FLAGS_rpc_server_port << " is already in use. "
        << "Please specify a different port using --rpc_server_port flag.";
    return -1;
  }

  xllm_service::Options options;
  options.server_host(FLAGS_server_host)
      .http_port(FLAGS_http_server_port)
      .http_idle_timeout_s(FLAGS_http_server_idle_timeout_s)
      .http_num_threads(FLAGS_http_server_num_threads)
      .http_max_concurrency(FLAGS_http_server_max_concurrency)
      .rpc_port(FLAGS_rpc_server_port)
      .rpc_idle_timeout_s(FLAGS_rpc_server_idle_timeout_s)
      .rpc_num_threads(FLAGS_rpc_server_num_threads)
      .rpc_max_concurrency(FLAGS_rpc_server_max_concurrency)
      .num_threads(FLAGS_num_threads)
      .max_concurrency(FLAGS_max_concurrency)
      .timeout_ms(FLAGS_timeout_ms)
      .connect_timeout_ms(FLAGS_connect_timeout_ms)
      .etcd_addr(FLAGS_etcd_addr)
      .etcd_namespace(FLAGS_etcd_namespace)
      .load_balance_policy(FLAGS_load_balance_policy)
      .xxh3_128bits_seed(FLAGS_xxh3_128bits_seed)
      .service_name(xllm_service::utils::get_local_ip() + ":" +
                    std::to_string(FLAGS_rpc_server_port))
      .detect_disconnected_instance_interval(
          FLAGS_detect_disconnected_instance_interval)
      .instance_delete_probe_timeout_ms(FLAGS_instance_delete_probe_timeout_ms)
      .instance_delete_probe_attempts(FLAGS_instance_delete_probe_attempts)
      .lease_lost_heartbeat_timeout_ms(FLAGS_lease_lost_heartbeat_timeout_ms)
      .enable_request_trace(FLAGS_enable_request_trace)
      .block_size(FLAGS_block_size)
      .enable_peer_service(FLAGS_enable_peer_service)
      .kv_event_zmq_enable(FLAGS_kv_event_zmq_enable)
      .kv_event_zmq_poll_interval_ms(FLAGS_kv_event_zmq_poll_interval_ms)
      .kv_event_zmq_reconnect_interval_ms(
          FLAGS_kv_event_zmq_reconnect_interval_ms)
      .kv_event_zmq_reconnect_interval_max_ms(
          FLAGS_kv_event_zmq_reconnect_interval_max_ms)
      .tokenizer_path(FLAGS_tokenizer_path)
      .tool_call_parser(FLAGS_tool_call_parser)
      .reasoning_parser(FLAGS_reasoning_parser);

  xllm_service::Master master(options);

  if (!master.start()) {
    LOG(ERROR) << "Failed to start master service.";
    return -1;
  }

  // install graceful shutdown handler
  (void)std::signal(SIGINT, shutdown_handler);
  (void)std::signal(SIGTERM, shutdown_handler);

  while (g_shutdown_signal == 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  LOG(WARNING) << "Received signal " << g_shutdown_signal
               << ", stopping master...";
  master.stop();

  return 0;
}
