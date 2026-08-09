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
  PROPERTY(std::string, server_host);

  PROPERTY(int32_t, http_port) = 9998;

  PROPERTY(int32_t, http_idle_timeout_s) = -1;

  PROPERTY(int32_t, http_num_threads) = 32;

  PROPERTY(int32_t, http_max_concurrency) = 0;

  // rpc server options
  PROPERTY(int32_t, rpc_port) = 9999;

  PROPERTY(int32_t, rpc_idle_timeout_s) = -1;

  PROPERTY(int32_t, rpc_num_threads) = 32;

  PROPERTY(int32_t, rpc_max_concurrency) = 0;

  PROPERTY(int32_t, num_threads) = 32;

  PROPERTY(int32_t, max_concurrency) = 32;

  PROPERTY(int32_t, timeout_ms) = -1;

  PROPERTY(int32_t, connect_timeout_ms) = -1;

  // instance manager options
  PROPERTY(std::string, etcd_addr);

  PROPERTY(std::string, etcd_namespace);

  PROPERTY(int32_t, instance_delete_probe_timeout_ms) = 1000;

  // V2 immutable Engine Registry and State Stream cache bounds. These are
  // injected into InstanceMgr and do not create new global FLAGS dependencies.
  PROPERTY(size_t, engine_registry_max_members) = 4096;

  PROPERTY(size_t, engine_registry_max_links) = 16384;

  PROPERTY(size_t, engine_registry_event_history_capacity) = 8192;

  PROPERTY(uint64_t, engine_state_soft_ttl_ms) = 3000;

  PROPERTY(uint64_t, engine_state_hard_ttl_ms) = 10000;

  PROPERTY(uint64_t, engine_heartbeat_hard_ttl_ms) = 10000;

  PROPERTY(uint64_t, engine_link_hard_ttl_ms) = 10000;

  PROPERTY(uint64_t, engine_direct_evidence_ttl_ms) = 3000;

  PROPERTY(uint64_t, engine_direct_probe_timeout_ms) = 200;

  PROPERTY(size_t, engine_direct_probe_batch_size) = 16;

  PROPERTY(double, state_blind_enter_ratio) = 0.5;

  PROPERTY(double, state_blind_exit_ratio) = 0.2;

  PROPERTY(uint64_t, state_blind_enter_hold_ms) = 1000;

  PROPERTY(uint64_t, state_blind_exit_hold_ms) = 3000;

  PROPERTY(uint64_t, state_blind_grace_ms) = 10000;

  PROPERTY(uint64_t, registry_blind_grace_ms) = 3000;

  PROPERTY(uint64_t, readiness_recovery_hold_ms) = 3000;

  PROPERTY(uint64_t, readiness_check_interval_ms) = 200;

  // Stop accepting new requests, then allow this bounded interval for
  // already-admitted requests to reach a terminal outcome during shutdown.
  PROPERTY(uint64_t, shutdown_drain_timeout_ms) = 5000;

  PROPERTY(uint64_t, engine_link_retry_initial_ms) = 1000;

  PROPERTY(uint64_t, engine_link_retry_max_ms) = 10000;

  PROPERTY(uint64_t, engine_link_ready_recheck_ms) = 3000;

  PROPERTY(size_t, engine_link_reconcile_batch_size) = 64;

  PROPERTY(size_t, state_stream_max_subscribers) = 256;

  PROPERTY(int32_t, state_stream_full_interval_ms) = 1000;

  PROPERTY(int32_t, state_stream_publish_interval_ms) = 50;

  PROPERTY(int32_t, state_stream_rpc_timeout_ms) = 200;

  // The KV observation plane has independent bounds and worker timing so
  // cache churn or a slow replica cannot delay Engine health/load state.
  PROPERTY(size_t, kv_state_max_subscribers) = 256;

  PROPERTY(size_t, kv_state_max_pending_batches) = 4096;

  PROPERTY(size_t, kv_state_max_pending_events) = 16384;

  PROPERTY(size_t, kv_state_max_pending_bytes) = 16 * 1024 * 1024;

  PROPERTY(size_t, kv_state_max_delivery_batches) = 64;

  PROPERTY(size_t, kv_state_max_delivery_bytes) = 1024 * 1024;

  PROPERTY(int32_t, kv_state_publish_interval_ms) = 20;

  PROPERTY(int32_t, kv_state_rpc_timeout_ms) = 200;

  PROPERTY(size_t, kv_shadow_max_engine_streams) = 4096;

  PROPERTY(size_t, kv_shadow_max_entries) = 1048576;

  PROPERTY(size_t, kv_shadow_max_bytes) = 512 * 1024 * 1024;

  PROPERTY(size_t, kv_shadow_max_recovery_events) = 8192;

  PROPERTY(size_t, kv_shadow_max_recovery_bytes) = 8 * 1024 * 1024;

  PROPERTY(size_t, kv_shadow_max_snapshot_entries) = 262144;

  PROPERTY(size_t, kv_shadow_max_snapshot_bytes) = 128 * 1024 * 1024;

  PROPERTY(uint64_t, kv_shadow_event_ttl_ms) = 30000;

  PROPERTY(uint64_t, kv_shadow_recovery_timeout_ms) = 30000;

  PROPERTY(int32_t, kv_snapshot_recovery_interval_ms) = 50;

  PROPERTY(size_t, kv_snapshot_recovery_batch_size) = 8;

  PROPERTY(size_t, kv_snapshot_recovery_max_concurrency) = 4;

  PROPERTY(size_t, kv_snapshot_max_pages_per_recovery) = 256;

  PROPERTY(uint32_t, kv_snapshot_page_entries) = 1024;

  PROPERTY(uint64_t, kv_snapshot_page_bytes) = 1024 * 1024;

  PROPERTY(uint64_t, kv_snapshot_page_generation_ms) = 50;

  PROPERTY(int32_t, kv_snapshot_rpc_timeout_ms) = 200;

  // scheduler options
  PROPERTY(std::string, load_balance_policy);

  PROPERTY(int32_t, block_size) = 128;

  PROPERTY(uint32_t, xxh3_128bits_seed) = 1024;

  // V2-K1 stays in SHADOW by default. ENFORCED is an explicit workload
  // bucket gate and still falls back to load-only for UNKNOWN/recovery/OOD.
  PROPERTY(std::string, kv_route_mode) = "SHADOW";

  // Both controls must be set before ENFORCED affects traffic. Operators open
  // the gate only after the shadow reconciliation thresholds are satisfied.
  PROPERTY(bool, kv_route_enforced_gate_open) = false;

  // Stable workload bucket in [0, 10000]. Zero keeps every request in SHADOW.
  PROPERTY(uint32_t, kv_route_enforced_bucket_permyriad) = 0;

  PROPERTY(size_t, kv_route_max_candidate_plans) = 16384;

  PROPERTY(size_t, kv_route_least_load_shortlist) = 8;

  PROPERTY(size_t, kv_route_top_prefix_shortlist) = 8;

  PROPERTY(uint64_t, kv_route_prefill_queue_cost_us) = 1000;

  PROPERTY(uint64_t, kv_route_decode_request_cost_us) = 1000;

  PROPERTY(uint64_t, kv_route_prefill_token_cost_us) = 10;

  PROPERTY(double, kv_route_transfer_byte_cost_us) = 0.001;

  PROPERTY(uint64_t, kv_route_decode_headroom_cost_us) = 1000;

  PROPERTY(uint64_t, kv_route_prefill_reserve_blocks) = 1;

  PROPERTY(uint64_t, kv_route_margin_us) = 100;

  PROPERTY(uint64_t, kv_route_near_equal_cost_us) = 10;

  // Must come from the target model/profile capacity evidence. Zero keeps D
  // Prefix bytes in observation-only mode while P Prefix scoring remains
  // available.
  PROPERTY(uint64_t, kv_route_bytes_per_token) = 0;

  PROPERTY(std::string, service_name);

  // V2 execution-hold cleanup capacity. One fixed-size token is reserved
  // before dispatch and follows an unresolved hold beyond request lifetime.
  PROPERTY(size_t, execution_hold_cleanup_record_capacity) = 65536;

  PROPERTY(size_t, execution_hold_cleanup_byte_capacity) = 64 * 1024 * 1024;

  PROPERTY(size_t, execution_hold_max_cleanup_record_bytes) = 1024;

  PROPERTY(size_t, execution_hold_max_potential_holders) = 16;

  PROPERTY(size_t, execution_hold_max_identifier_bytes) = 256;

  PROPERTY(int32_t, execution_hold_cleanup_retry_interval_ms) = 1000;

  PROPERTY(size_t, execution_hold_cleanup_retry_batch_size) = 8;

  PROPERTY(int32_t, execution_hold_cleanup_rpc_timeout_ms) = 100;

  PROPERTY(size_t, output_reorder_max_events) = 64;

  PROPERTY(size_t, output_reorder_max_bytes) = 4 * 1024 * 1024;

  PROPERTY(int32_t, request_watchdog_interval_ms) = 100;

  PROPERTY(int32_t, output_gap_timeout_ms) = 1000;

  PROPERTY(int32_t, output_gap_query_timeout_ms) = 100;

  PROPERTY(size_t, output_gap_query_batch_size) = 8;

  PROPERTY(int32_t, p_first_event_retry_ub_ms) = 700;

  PROPERTY(int32_t, first_event_dispatch_margin_ms) = 200;

  PROPERTY(size_t, max_first_output_attempt_retries) = 1;

  PROPERTY(uint64_t, max_nonstream_retry_wasted_device_ms) = 5000;

  PROPERTY(uint64_t, min_first_output_retry_remaining_ms) = 1000;

  PROPERTY(size_t, request_deadline_capacity) = 65536;

  PROPERTY(size_t, request_watchdog_batch_size) = 1024;

  // Requests without an explicit client/Gateway duration receive this local
  // business deadline. Zero is invalid for the V2 service path.
  PROPERTY(int32_t, default_request_deadline_ms) = 300000;

  // tokenizer options
  PROPERTY(std::string, tokenizer_path);

  // Digest of the complete local tokenizer + template rendering contract.
  // STRICT Native requests compare this independently configured value with
  // the immutable Provider Descriptor; an empty value fails closed.
  PROPERTY(std::string, native_renderer_digest);

  // trace options
  PROPERTY(bool, enable_request_trace) = false;

  // parser options
  PROPERTY(std::string, tool_call_parser);

  PROPERTY(std::string, reasoning_parser);

  PROPERTY(int32_t, vllm_http_timeout_ms) = 60000;

  PROPERTY(std::string, internal_api_token);
};

}  // namespace xllm_service
