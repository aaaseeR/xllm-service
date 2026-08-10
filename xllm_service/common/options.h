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

#include <cstddef>
#include <cstdint>
#include <string>

#include "common/macros.h"

namespace xllm_service {

class Options {
 public:
  Options() = default;
  ~Options() = default;

  // http server options
  XLLM_SERVICE_PROPERTY(std::string, server_host);

  XLLM_SERVICE_PROPERTY(int32_t, http_port) = 9998;

  XLLM_SERVICE_PROPERTY(int32_t, http_idle_timeout_s) = -1;

  XLLM_SERVICE_PROPERTY(int32_t, http_num_threads) = 32;

  XLLM_SERVICE_PROPERTY(int32_t, http_max_concurrency) = 0;

  // rpc server options
  XLLM_SERVICE_PROPERTY(int32_t, rpc_port) = 9999;

  XLLM_SERVICE_PROPERTY(int32_t, rpc_idle_timeout_s) = -1;

  XLLM_SERVICE_PROPERTY(int32_t, rpc_num_threads) = 32;

  XLLM_SERVICE_PROPERTY(int32_t, rpc_max_concurrency) = 0;

  XLLM_SERVICE_PROPERTY(int32_t, num_threads) = 32;

  XLLM_SERVICE_PROPERTY(int32_t, max_concurrency) = 32;

  XLLM_SERVICE_PROPERTY(int32_t, timeout_ms) = -1;

  XLLM_SERVICE_PROPERTY(int32_t, connect_timeout_ms) = -1;

  // instance manager options
  XLLM_SERVICE_PROPERTY(std::string, etcd_addr);

  XLLM_SERVICE_PROPERTY(std::string, etcd_namespace);

  XLLM_SERVICE_PROPERTY(int32_t, instance_delete_probe_timeout_ms) = 1000;

  // V2 immutable Engine Registry and State Stream cache bounds. These are
  // injected into InstanceMgr and do not create new global FLAGS dependencies.
  XLLM_SERVICE_PROPERTY(size_t, engine_registry_max_members) = 4096;

  XLLM_SERVICE_PROPERTY(size_t, engine_registry_max_links) = 16384;

  XLLM_SERVICE_PROPERTY(size_t, engine_registry_event_history_capacity) = 8192;

  XLLM_SERVICE_PROPERTY(uint64_t, engine_state_soft_ttl_ms) = 3000;

  XLLM_SERVICE_PROPERTY(uint64_t, engine_state_hard_ttl_ms) = 10000;

  XLLM_SERVICE_PROPERTY(uint64_t, engine_heartbeat_hard_ttl_ms) = 10000;

  XLLM_SERVICE_PROPERTY(uint64_t, engine_link_hard_ttl_ms) = 10000;

  XLLM_SERVICE_PROPERTY(uint64_t, engine_direct_evidence_ttl_ms) = 3000;

  XLLM_SERVICE_PROPERTY(uint64_t, engine_direct_probe_timeout_ms) = 200;

  XLLM_SERVICE_PROPERTY(size_t, engine_direct_probe_batch_size) = 16;

  XLLM_SERVICE_PROPERTY(double, state_blind_enter_ratio) = 0.5;

  XLLM_SERVICE_PROPERTY(double, state_blind_exit_ratio) = 0.2;

  XLLM_SERVICE_PROPERTY(uint64_t, state_blind_enter_hold_ms) = 1000;

  XLLM_SERVICE_PROPERTY(uint64_t, state_blind_exit_hold_ms) = 3000;

  XLLM_SERVICE_PROPERTY(uint64_t, state_blind_grace_ms) = 10000;

  XLLM_SERVICE_PROPERTY(uint64_t, registry_blind_grace_ms) = 3000;

  XLLM_SERVICE_PROPERTY(uint64_t, readiness_recovery_hold_ms) = 3000;

  XLLM_SERVICE_PROPERTY(uint64_t, readiness_check_interval_ms) = 200;

  // Stop accepting new requests, then allow this bounded interval for
  // already-admitted requests to reach a terminal outcome during shutdown.
  XLLM_SERVICE_PROPERTY(uint64_t, shutdown_drain_timeout_ms) = 5000;

  XLLM_SERVICE_PROPERTY(uint64_t, engine_link_retry_initial_ms) = 1000;

  XLLM_SERVICE_PROPERTY(uint64_t, engine_link_retry_max_ms) = 10000;

  XLLM_SERVICE_PROPERTY(uint64_t, engine_link_ready_recheck_ms) = 3000;

  XLLM_SERVICE_PROPERTY(size_t, engine_link_reconcile_batch_size) = 64;

  XLLM_SERVICE_PROPERTY(size_t, state_stream_max_subscribers) = 256;

  XLLM_SERVICE_PROPERTY(int32_t, state_stream_full_interval_ms) = 1000;

  XLLM_SERVICE_PROPERTY(int32_t, state_stream_publish_interval_ms) = 50;

  XLLM_SERVICE_PROPERTY(int32_t, state_stream_rpc_timeout_ms) = 200;

  // The KV observation plane has independent bounds and worker timing so
  // cache churn or a slow replica cannot delay Engine health/load state.
  XLLM_SERVICE_PROPERTY(size_t, kv_state_max_subscribers) = 256;

  XLLM_SERVICE_PROPERTY(size_t, kv_state_max_pending_batches) = 4096;

  XLLM_SERVICE_PROPERTY(size_t, kv_state_max_pending_events) = 16384;

  XLLM_SERVICE_PROPERTY(size_t, kv_state_max_pending_bytes) = 16 * 1024 * 1024;

  XLLM_SERVICE_PROPERTY(size_t, kv_state_max_delivery_batches) = 64;

  XLLM_SERVICE_PROPERTY(size_t, kv_state_max_delivery_bytes) = 1024 * 1024;

  XLLM_SERVICE_PROPERTY(int32_t, kv_state_publish_interval_ms) = 20;

  XLLM_SERVICE_PROPERTY(int32_t, kv_state_rpc_timeout_ms) = 200;

  XLLM_SERVICE_PROPERTY(size_t, kv_shadow_max_engine_streams) = 4096;

  XLLM_SERVICE_PROPERTY(size_t, kv_shadow_max_entries) = 1048576;

  XLLM_SERVICE_PROPERTY(size_t, kv_shadow_max_bytes) = 512 * 1024 * 1024;

  XLLM_SERVICE_PROPERTY(size_t, kv_shadow_max_recovery_events) = 8192;

  XLLM_SERVICE_PROPERTY(size_t, kv_shadow_max_recovery_bytes) = 8 * 1024 * 1024;

  XLLM_SERVICE_PROPERTY(size_t, kv_shadow_max_snapshot_entries) = 262144;

  XLLM_SERVICE_PROPERTY(size_t, kv_shadow_max_snapshot_bytes) = 128 * 1024 *
                                                                1024;

  XLLM_SERVICE_PROPERTY(uint64_t, kv_shadow_event_ttl_ms) = 30000;

  XLLM_SERVICE_PROPERTY(uint64_t, kv_shadow_recovery_timeout_ms) = 30000;

  XLLM_SERVICE_PROPERTY(int32_t, kv_snapshot_recovery_interval_ms) = 50;

  XLLM_SERVICE_PROPERTY(size_t, kv_snapshot_recovery_batch_size) = 8;

  XLLM_SERVICE_PROPERTY(size_t, kv_snapshot_recovery_max_concurrency) = 4;

  XLLM_SERVICE_PROPERTY(size_t, kv_snapshot_max_pages_per_recovery) = 256;

  XLLM_SERVICE_PROPERTY(uint32_t, kv_snapshot_page_entries) = 1024;

  XLLM_SERVICE_PROPERTY(uint64_t, kv_snapshot_page_bytes) = 1024 * 1024;

  XLLM_SERVICE_PROPERTY(uint64_t, kv_snapshot_page_generation_ms) = 50;

  XLLM_SERVICE_PROPERTY(int32_t, kv_snapshot_rpc_timeout_ms) = 200;

  // scheduler options
  XLLM_SERVICE_PROPERTY(std::string, load_balance_policy);

  XLLM_SERVICE_PROPERTY(int32_t, block_size) = 128;

  XLLM_SERVICE_PROPERTY(uint32_t, xxh3_128bits_seed) = 1024;

  // V2-K1 stays in SHADOW by default. ENFORCED is an explicit workload
  // bucket gate and still falls back to load-only for UNKNOWN/recovery/OOD.
  XLLM_SERVICE_PROPERTY(std::string, kv_route_mode) = "SHADOW";

  // Both controls must be set before ENFORCED affects traffic. Operators open
  // the gate only after the shadow reconciliation thresholds are satisfied.
  XLLM_SERVICE_PROPERTY(bool, kv_route_enforced_gate_open) = false;

  // Stable workload bucket in [0, 10000]. Zero keeps every request in SHADOW.
  XLLM_SERVICE_PROPERTY(uint32_t, kv_route_enforced_bucket_permyriad) = 0;

  XLLM_SERVICE_PROPERTY(size_t, kv_route_max_candidate_plans) = 16384;

  XLLM_SERVICE_PROPERTY(size_t, kv_route_least_load_shortlist) = 8;

  XLLM_SERVICE_PROPERTY(size_t, kv_route_top_prefix_shortlist) = 8;

  XLLM_SERVICE_PROPERTY(uint64_t, kv_route_prefill_queue_cost_us) = 1000;

  XLLM_SERVICE_PROPERTY(uint64_t, kv_route_decode_request_cost_us) = 1000;

  XLLM_SERVICE_PROPERTY(uint64_t, kv_route_prefill_token_cost_us) = 10;

  XLLM_SERVICE_PROPERTY(double, kv_route_transfer_byte_cost_us) = 0.001;

  XLLM_SERVICE_PROPERTY(uint64_t, kv_route_decode_headroom_cost_us) = 1000;

  XLLM_SERVICE_PROPERTY(uint64_t, kv_route_prefill_reserve_blocks) = 1;

  XLLM_SERVICE_PROPERTY(uint64_t, kv_route_margin_us) = 100;

  XLLM_SERVICE_PROPERTY(uint64_t, kv_route_near_equal_cost_us) = 10;

  // Must come from the target model/profile capacity evidence. Zero keeps D
  // Prefix bytes in observation-only mode while P Prefix scoring remains
  // available.
  XLLM_SERVICE_PROPERTY(uint64_t, kv_route_bytes_per_token) = 0;

  // V2 Service/ModelPool flow-control hard bounds. Queue accounting includes
  // normalized payload/token storage and an explicit dispatched context
  // estimate so process crash and memory exposure are configuration-bounded.
  XLLM_SERVICE_PROPERTY(size_t, flow_max_queued_requests) = 4096;
  XLLM_SERVICE_PROPERTY(size_t, flow_max_dispatched_contexts) = 4096;
  XLLM_SERVICE_PROPERTY(uint64_t, flow_max_queued_prompt_tokens) = 16 * 1024 *
                                                                   1024;
  XLLM_SERVICE_PROPERTY(uint64_t, flow_max_queued_bytes) = 512 * 1024 * 1024;
  XLLM_SERVICE_PROPERTY(uint64_t, flow_max_queue_wait_ms) = 2000;
  XLLM_SERVICE_PROPERTY(size_t, flow_max_queued_requests_per_tenant) = 512;
  XLLM_SERVICE_PROPERTY(uint64_t,
                        flow_max_queued_tokens_per_tenant) = 2 * 1024 * 1024;
  XLLM_SERVICE_PROPERTY(size_t, flow_max_model_queued_requests) = 4096;
  XLLM_SERVICE_PROPERTY(size_t, flow_max_model_dispatched_contexts) = 4096;
  XLLM_SERVICE_PROPERTY(uint64_t,
                        flow_max_model_queued_prompt_tokens) = 16 * 1024 * 1024;
  XLLM_SERVICE_PROPERTY(uint64_t, flow_max_model_queued_bytes) = 512 * 1024 *
                                                                 1024;
  XLLM_SERVICE_PROPERTY(size_t, flow_service_crash_request_budget) = 8192;
  XLLM_SERVICE_PROPERTY(uint64_t,
                        flow_service_memory_budget_bytes) = 528 * 1024 * 1024;
  XLLM_SERVICE_PROPERTY(uint64_t, flow_dispatched_context_bytes) = 4096;
  XLLM_SERVICE_PROPERTY(double, flow_dispatch_rate_lb_per_second) = 1.0;
  XLLM_SERVICE_PROPERTY(uint64_t, flow_probe_round_ub_ms) = 200;
  XLLM_SERVICE_PROPERTY(size_t, flow_blind_dispatch_probe_concurrency) = 1;
  XLLM_SERVICE_PROPERTY(size_t, flow_starvation_dispatch_bound) = 32;
  XLLM_SERVICE_PROPERTY(std::string, flow_order) = "FCFS";
  XLLM_SERVICE_PROPERTY(int32_t, flow_dispatch_interval_ms) = 10;
  // COMPLETE_QUEUED preserves accepted work. RETRY_UNDISPATCHED returns only
  // work that has not crossed the dispatch boundary.
  XLLM_SERVICE_PROPERTY(std::string, flow_drain_policy) = "COMPLETE_QUEUED";

  // Native mode gates. LOCAL remains an explicit allowlist bucket because the
  // D-side mixed-accounting capability is necessary but not sufficient proof
  // that a workload profile has passed the co-resident TPOT gate.
  XLLM_SERVICE_PROPERTY(bool, native_local_prefill_enabled) = false;
  XLLM_SERVICE_PROPERTY(uint32_t, native_local_prefill_bucket_permyriad) = 0;
  XLLM_SERVICE_PROPERTY(uint64_t, native_local_prefill_token_cap) = 512;
  XLLM_SERVICE_PROPERTY(bool, native_prefill_only_enabled) = false;
  XLLM_SERVICE_PROPERTY(uint64_t, native_prefill_only_output_token_cap) = 1;

  // The deployment, not an end client, asserts that tenant/flow headers are
  // authenticated and stripped/replaced at the Gateway trust boundary.
  XLLM_SERVICE_PROPERTY(bool, trusted_tenant_headers_enabled) = false;
  XLLM_SERVICE_PROPERTY(std::string, kv_session_hmac_secret);
  XLLM_SERVICE_PROPERTY(std::string, kv_session_hmac_previous_secret);
  XLLM_SERVICE_PROPERTY(uint64_t, kv_session_token_ttl_seconds) = 86400;

  // Request events are written by producers into a fixed-capacity ring and
  // exported by a dedicated thread. Observability loss is visible but never
  // backpressures inference.
  XLLM_SERVICE_PROPERTY(size_t, observability_event_capacity) = 65536;
  XLLM_SERVICE_PROPERTY(size_t, observability_export_batch_size) = 1024;
  XLLM_SERVICE_PROPERTY(int32_t, observability_export_interval_ms) = 20;
  XLLM_SERVICE_PROPERTY(int32_t, observability_snapshot_interval_ms) = 5000;
  XLLM_SERVICE_PROPERTY(std::string, observability_build_id) = "development";

  XLLM_SERVICE_PROPERTY(std::string, service_name);

  // V2 execution-hold cleanup capacity. One fixed-size token is reserved
  // before dispatch and follows an unresolved hold beyond request lifetime.
  XLLM_SERVICE_PROPERTY(size_t, execution_hold_cleanup_record_capacity) = 65536;

  XLLM_SERVICE_PROPERTY(size_t, execution_hold_cleanup_byte_capacity) = 64 *
                                                                        1024 *
                                                                        1024;

  XLLM_SERVICE_PROPERTY(size_t, execution_hold_max_cleanup_record_bytes) = 1024;

  XLLM_SERVICE_PROPERTY(size_t, execution_hold_max_potential_holders) = 16;

  XLLM_SERVICE_PROPERTY(size_t, execution_hold_max_identifier_bytes) = 256;

  XLLM_SERVICE_PROPERTY(int32_t,
                        execution_hold_cleanup_retry_interval_ms) = 1000;

  XLLM_SERVICE_PROPERTY(size_t, execution_hold_cleanup_retry_batch_size) = 8;

  XLLM_SERVICE_PROPERTY(int32_t, execution_hold_cleanup_rpc_timeout_ms) = 100;

  XLLM_SERVICE_PROPERTY(size_t, output_reorder_max_events) = 64;

  XLLM_SERVICE_PROPERTY(size_t, output_reorder_max_bytes) = 4 * 1024 * 1024;

  XLLM_SERVICE_PROPERTY(int32_t, request_watchdog_interval_ms) = 100;

  XLLM_SERVICE_PROPERTY(int32_t, output_gap_timeout_ms) = 1000;

  XLLM_SERVICE_PROPERTY(int32_t, output_gap_query_timeout_ms) = 100;

  XLLM_SERVICE_PROPERTY(size_t, output_gap_query_batch_size) = 8;

  XLLM_SERVICE_PROPERTY(int32_t, p_first_event_retry_ub_ms) = 700;

  XLLM_SERVICE_PROPERTY(int32_t, first_event_dispatch_margin_ms) = 200;

  XLLM_SERVICE_PROPERTY(size_t, max_first_output_attempt_retries) = 1;

  XLLM_SERVICE_PROPERTY(uint64_t, max_nonstream_retry_wasted_device_ms) = 5000;

  XLLM_SERVICE_PROPERTY(uint64_t, min_first_output_retry_remaining_ms) = 1000;

  XLLM_SERVICE_PROPERTY(size_t, request_deadline_capacity) = 65536;

  XLLM_SERVICE_PROPERTY(size_t, request_watchdog_batch_size) = 1024;

  // Requests without an explicit client/Gateway duration receive this local
  // business deadline. Zero is invalid for the V2 service path.
  XLLM_SERVICE_PROPERTY(int32_t, default_request_deadline_ms) = 300000;

  // tokenizer options
  XLLM_SERVICE_PROPERTY(std::string, tokenizer_path);

  // Digest of the complete local tokenizer + template rendering contract.
  // STRICT Native requests compare this independently configured value with
  // the immutable Provider Descriptor; an empty value fails closed.
  XLLM_SERVICE_PROPERTY(std::string, native_renderer_digest);

  // trace options
  XLLM_SERVICE_PROPERTY(bool, enable_request_trace) = false;

  // parser options
  XLLM_SERVICE_PROPERTY(std::string, tool_call_parser);

  XLLM_SERVICE_PROPERTY(std::string, reasoning_parser);

  XLLM_SERVICE_PROPERTY(int32_t, vllm_http_timeout_ms) = 60000;

  XLLM_SERVICE_PROPERTY(std::string, internal_api_token);
};

}  // namespace xllm_service
