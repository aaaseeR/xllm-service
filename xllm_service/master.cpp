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

#include "master.h"

#include <chrono>
#include <cmath>
#include <csignal>

#include "common/global_gflags.h"
#include "common/types.h"
#include "common/utils.h"
#include "http_service/request_trust_policy.h"

namespace xllm_service {

Master::Master(const Options& options) : options_(options) {
  scheduler_ = std::make_unique<Scheduler>(options);

  rpc_service_ =
      std::make_unique<xllm_service::XllmRpcService>(options, scheduler_.get());

  http_service_ = std::make_unique<xllm_service::XllmHttpServiceImpl>(
      options, scheduler_.get());
}

Master::~Master() { stop(); }

bool Master::start() {
  if (!setup_http_server() || !start_http_server()) {
    return false;
  }
  if (!start_rpc_server()) {
    http_server_.Stop(0);
    http_server_.Join();
    http_started_ = false;
    return false;
  }
  scheduler_->refresh_readiness();
  readiness_thread_ = std::make_unique<std::thread>(
      [this]() { manage_http_server_lifecycle(); });

  return true;
}

void Master::stop() {
  if (stopped_.exchange(true)) {
    return;
  }
  scheduler_->set_draining(true);
  scheduler_->refresh_readiness();
  if (!scheduler_->wait_for_requests_drained(
          std::chrono::milliseconds(options_.shutdown_drain_timeout_ms()))) {
    LOG(WARNING) << "Shutdown drain timed out with active requests; "
                    "remaining holds will be transferred to cleanup";
  }

  if (readiness_thread_ && readiness_thread_->joinable()) {
    readiness_thread_->join();
  }
  if (http_started_) {
    http_server_.Stop(0);
    http_server_.Join();
    http_started_ = false;
  }
  if (rpc_started_) {
    rpc_server_.Stop(0);
    rpc_server_.Join();
    rpc_started_ = false;
  }
}

bool Master::setup_http_server() {
  if (http_server_.AddService(http_service_.get(),
                              brpc::SERVER_DOESNT_OWN_SERVICE,
                              // for testing
                              "/hello => Hello,"
                              "/v1/completions => Completions,"
                              "/v1/chat/completions => ChatCompletions,"
                              "/v1/messages => AnthropicMessages,"
                              "/v1/embeddings => Embeddings,"
                              "/v1/models => Models,"
                              "/metrics => Metrics,"
                              "/livez => Livez,"
                              "/readyz => Readyz,"
                              "/v1/internal/heartbeat => Heartbeat,") != 0) {
    LOG(FATAL) << "Fail to add http service";
    return false;
  }

  http_options_.idle_timeout_sec = options_.http_idle_timeout_s();
  http_options_.num_threads = options_.http_num_threads();
  http_options_.max_concurrency = options_.http_max_concurrency();

  if (!options_.server_host().empty()) {
    http_server_address_ =
        options_.server_host() + ":" + std::to_string(options_.http_port());
    if (butil::str2endpoint(http_server_address_.c_str(), &http_endpoint_) <
        0) {
      LOG(FATAL) << "Convert server_addr to endpoint failed: "
                 << http_server_address_;
      return false;
    }
  } else {
    http_endpoint_ = butil::EndPoint(butil::IP_ANY, options_.http_port());
  }

  return true;
}

bool Master::start_http_server() {
  if (http_server_.Start(http_endpoint_, &http_options_) != 0) {
    LOG(ERROR) << "Failed to start HTTP server on: " << http_endpoint_;
    return false;
  }
  http_started_ = true;
  LOG(INFO) << "HTTP server started on: " << http_endpoint_;
  return true;
}

void Master::manage_http_server_lifecycle() {
  while (!stopped_.load()) {
    scheduler_->refresh_readiness();

    const auto end_time =
        std::chrono::steady_clock::now() +
        std::chrono::milliseconds(options_.readiness_check_interval_ms());
    while (!stopped_.load() && std::chrono::steady_clock::now() < end_time) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }
}

bool Master::start_rpc_server() {
  if (rpc_server_.AddService(rpc_service_.get(),
                             brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    LOG(FATAL) << "Failed to add rpc service.";
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
      LOG(FATAL) << "Convert server_addr to endpoint failed: "
                 << rpc_server_address_;
      return false;
    }
  } else {
    endpoint = butil::EndPoint(butil::IP_ANY, options_.rpc_port());
  }

  if (rpc_server_.Start(endpoint, &options) != 0) {
    LOG(FATAL) << "Failed to start rpc server on: " << endpoint;
    return false;
  }

  LOG(INFO) << "Xllm rpc server started on: " << endpoint;
  rpc_started_ = true;
  return true;
}

}  // namespace xllm_service

static std::atomic<uint32_t> g_signal_received{0};
void shutdown_handler(int signal) {
  g_signal_received.store(static_cast<uint32_t>(signal),
                          std::memory_order_relaxed);
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
  if (FLAGS_kv_route_mode != "DISABLED" && FLAGS_kv_route_mode != "SHADOW" &&
      FLAGS_kv_route_mode != "ENFORCED") {
    LOG(ERROR) << "Invalid --kv_route_mode: " << FLAGS_kv_route_mode;
    return -1;
  }
  if (FLAGS_kv_route_enforced_bucket_permyriad > 10000) {
    LOG(ERROR) << "--kv_route_enforced_bucket_permyriad must be <= 10000";
    return -1;
  }
  if (FLAGS_kv_route_enforced_gate_open &&
      (FLAGS_load_balance_policy != "CAR" ||
       FLAGS_kv_route_mode != "ENFORCED" ||
       FLAGS_kv_route_enforced_bucket_permyriad == 0 ||
       FLAGS_kv_route_bytes_per_token == 0 ||
       FLAGS_kv_route_transfer_byte_cost_us <= 0.0 ||
       FLAGS_kv_route_prefill_token_cost_us == 0)) {
    LOG(ERROR) << "KV route enforced gate requires --load_balance_policy=CAR, "
                  "--kv_route_mode=ENFORCED, a non-zero bucket, and a "
                  "non-zero byte/token and transfer/prefill cost calibration";
    return -1;
  }
  if (FLAGS_kv_route_max_candidate_plans == 0 ||
      FLAGS_kv_route_least_load_shortlist == 0 ||
      FLAGS_kv_route_top_prefix_shortlist == 0 ||
      FLAGS_kv_route_prefill_queue_cost_us == 0 ||
      FLAGS_kv_route_decode_request_cost_us == 0 ||
      FLAGS_kv_route_prefill_token_cost_us == 0 ||
      !std::isfinite(FLAGS_kv_route_transfer_byte_cost_us) ||
      FLAGS_kv_route_transfer_byte_cost_us < 0.0 ||
      FLAGS_kv_route_decode_headroom_cost_us == 0) {
    LOG(ERROR) << "KV route planner capacity and shortlist bounds must be "
                  "greater than zero and calibrated costs must be finite";
    return -1;
  }
  if (!FLAGS_kv_session_hmac_secret.empty() &&
      (FLAGS_kv_session_hmac_secret.size() < 32 ||
       FLAGS_kv_session_hmac_secret.size() > 256)) {
    LOG(ERROR) << "--kv_session_hmac_secret must contain 32-256 bytes when "
                  "configured";
    return -1;
  }
  if (!FLAGS_kv_session_hmac_previous_secret.empty() &&
      (FLAGS_kv_session_hmac_previous_secret.size() < 32 ||
       FLAGS_kv_session_hmac_previous_secret.size() > 256)) {
    LOG(ERROR) << "--kv_session_hmac_previous_secret must contain 32-256 "
                  "bytes when configured";
    return -1;
  }
  if (FLAGS_kv_session_token_ttl_seconds == 0 ||
      FLAGS_kv_session_token_ttl_seconds >
          xllm_service::KVSessionTokenCodec::kMaxTokenTtlSeconds) {
    LOG(ERROR) << "--kv_session_token_ttl_seconds must be in [1, "
               << xllm_service::KVSessionTokenCodec::kMaxTokenTtlSeconds << "]";
    return -1;
  }
  if (FLAGS_kv_route_enforced_gate_open &&
      (!FLAGS_trusted_tenant_headers_enabled ||
       FLAGS_trusted_client_identity_headers_enabled) &&
      FLAGS_kv_session_hmac_secret.empty()) {
    LOG(ERROR) << "KV route enforcement with an opaque or derived session "
                  "domain requires a shared --kv_session_hmac_secret";
    return -1;
  }
  if (FLAGS_kv_route_enforced_gate_open &&
      (FLAGS_observability_build_id.empty() ||
       FLAGS_observability_build_id == "development" ||
       FLAGS_observability_build_id == "REPLACE_WITH_IMMUTABLE_ARTIFACT_ID")) {
    LOG(ERROR) << "KV route enforcement requires an immutable "
                  "--observability_build_id";
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
      .kv_route_mode(FLAGS_kv_route_mode)
      .kv_route_enforced_gate_open(FLAGS_kv_route_enforced_gate_open)
      .kv_route_enforced_bucket_permyriad(
          FLAGS_kv_route_enforced_bucket_permyriad)
      .kv_route_max_candidate_plans(FLAGS_kv_route_max_candidate_plans)
      .kv_route_least_load_shortlist(FLAGS_kv_route_least_load_shortlist)
      .kv_route_top_prefix_shortlist(FLAGS_kv_route_top_prefix_shortlist)
      .kv_route_prefill_queue_cost_us(FLAGS_kv_route_prefill_queue_cost_us)
      .kv_route_decode_request_cost_us(FLAGS_kv_route_decode_request_cost_us)
      .kv_route_prefill_token_cost_us(FLAGS_kv_route_prefill_token_cost_us)
      .kv_route_transfer_byte_cost_us(FLAGS_kv_route_transfer_byte_cost_us)
      .kv_route_decode_headroom_cost_us(FLAGS_kv_route_decode_headroom_cost_us)
      .kv_route_prefill_reserve_blocks(FLAGS_kv_route_prefill_reserve_blocks)
      .kv_route_margin_us(FLAGS_kv_route_margin_us)
      .kv_route_near_equal_cost_us(FLAGS_kv_route_near_equal_cost_us)
      .kv_route_bytes_per_token(FLAGS_kv_route_bytes_per_token)
      .flow_max_queued_requests(FLAGS_flow_max_queued_requests)
      .flow_max_dispatched_contexts(FLAGS_flow_max_dispatched_contexts)
      .flow_max_queued_prompt_tokens(FLAGS_flow_max_queued_prompt_tokens)
      .flow_max_queued_bytes(FLAGS_flow_max_queued_bytes)
      .flow_max_queue_wait_ms(FLAGS_flow_max_queue_wait_ms)
      .flow_max_queued_requests_per_tenant(
          FLAGS_flow_max_queued_requests_per_tenant)
      .flow_max_queued_tokens_per_tenant(
          FLAGS_flow_max_queued_tokens_per_tenant)
      .flow_max_model_queued_requests(FLAGS_flow_max_model_queued_requests)
      .flow_max_model_dispatched_contexts(
          FLAGS_flow_max_model_dispatched_contexts)
      .flow_max_model_queued_prompt_tokens(
          FLAGS_flow_max_model_queued_prompt_tokens)
      .flow_max_model_queued_bytes(FLAGS_flow_max_model_queued_bytes)
      .flow_service_crash_request_budget(
          FLAGS_flow_service_crash_request_budget)
      .flow_service_memory_budget_bytes(FLAGS_flow_service_memory_budget_bytes)
      .flow_dispatched_context_bytes(FLAGS_flow_dispatched_context_bytes)
      .flow_dispatch_rate_lb_per_second(FLAGS_flow_dispatch_rate_lb_per_second)
      .flow_probe_round_ub_ms(FLAGS_flow_probe_round_ub_ms)
      .flow_blind_dispatch_probe_concurrency(
          FLAGS_flow_blind_dispatch_probe_concurrency)
      .flow_starvation_dispatch_bound(FLAGS_flow_starvation_dispatch_bound)
      .flow_order(FLAGS_flow_order)
      .flow_dispatch_interval_ms(FLAGS_flow_dispatch_interval_ms)
      .flow_drain_policy(FLAGS_flow_drain_policy)
      .native_local_prefill_enabled(FLAGS_native_local_prefill_enabled)
      .native_local_prefill_bucket_permyriad(
          FLAGS_native_local_prefill_bucket_permyriad)
      .native_local_prefill_token_cap(FLAGS_native_local_prefill_token_cap)
      .native_prefill_only_enabled(FLAGS_native_prefill_only_enabled)
      .native_prefill_only_output_token_cap(
          FLAGS_native_prefill_only_output_token_cap)
      .trusted_tenant_headers_enabled(FLAGS_trusted_tenant_headers_enabled)
      .trusted_client_identity_headers_enabled(
          FLAGS_trusted_client_identity_headers_enabled)
      .kv_session_hmac_secret(FLAGS_kv_session_hmac_secret)
      .kv_session_hmac_previous_secret(FLAGS_kv_session_hmac_previous_secret)
      .kv_session_token_ttl_seconds(FLAGS_kv_session_token_ttl_seconds)
      .observability_event_capacity(FLAGS_observability_event_capacity)
      .observability_export_batch_size(FLAGS_observability_export_batch_size)
      .observability_export_interval_ms(FLAGS_observability_export_interval_ms)
      .observability_snapshot_interval_ms(
          FLAGS_observability_snapshot_interval_ms)
      .observability_build_id(FLAGS_observability_build_id)
      .placement_config_path(FLAGS_placement_config_path)
      .xxh3_128bits_seed(FLAGS_xxh3_128bits_seed)
      .service_name(xllm_service::utils::get_local_ip() + ":" +
                    std::to_string(FLAGS_rpc_server_port))
      .instance_delete_probe_timeout_ms(FLAGS_instance_delete_probe_timeout_ms)
      .engine_state_soft_ttl_ms(FLAGS_engine_state_soft_ttl_ms)
      .engine_state_hard_ttl_ms(FLAGS_engine_state_hard_ttl_ms)
      .engine_heartbeat_hard_ttl_ms(FLAGS_engine_heartbeat_hard_ttl_ms)
      .output_reorder_max_events(FLAGS_output_reorder_max_events)
      .output_reorder_max_bytes(FLAGS_output_reorder_max_bytes)
      .request_watchdog_interval_ms(FLAGS_request_watchdog_interval_ms)
      .output_gap_timeout_ms(FLAGS_output_gap_timeout_ms)
      .output_gap_query_timeout_ms(FLAGS_output_gap_query_timeout_ms)
      .output_gap_query_batch_size(FLAGS_output_gap_query_batch_size)
      .p_first_event_retry_ub_ms(FLAGS_p_first_event_retry_ub_ms)
      .first_event_dispatch_margin_ms(FLAGS_first_event_dispatch_margin_ms)
      .max_first_output_attempt_retries(FLAGS_max_first_output_attempt_retries)
      .max_nonstream_retry_wasted_device_ms(
          FLAGS_max_nonstream_retry_wasted_device_ms)
      .min_first_output_retry_remaining_ms(
          FLAGS_min_first_output_retry_remaining_ms)
      .request_deadline_capacity(FLAGS_request_deadline_capacity)
      .request_watchdog_batch_size(FLAGS_request_watchdog_batch_size)
      .default_request_deadline_ms(FLAGS_default_request_deadline_ms)
      .enable_request_trace(FLAGS_enable_request_trace)
      .block_size(FLAGS_block_size)
      .tokenizer_path(FLAGS_tokenizer_path)
      .native_renderer_digest(FLAGS_native_renderer_digest)
      .tool_call_parser(FLAGS_tool_call_parser)
      .reasoning_parser(FLAGS_reasoning_parser)
      .vllm_http_timeout_ms(FLAGS_vllm_http_timeout_ms)
      .internal_api_token(FLAGS_internal_api_token);

  xllm_service::Master master(options);

  if (!master.start()) {
    LOG(ERROR) << "Failed to start master service.";
    return -1;
  }

  // install graceful shutdown handler
  (void)signal(SIGINT, shutdown_handler);
  (void)signal(SIGTERM, shutdown_handler);

  while (g_signal_received.load(std::memory_order_relaxed) == 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  // wait here
  master.stop();

  return 0;
}
