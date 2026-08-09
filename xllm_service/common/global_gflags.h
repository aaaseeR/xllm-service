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

#include <gflags/gflags.h>

DECLARE_string(server_host);

DECLARE_int32(http_server_port);

DECLARE_int32(http_server_idle_timeout_s);

DECLARE_int32(http_server_num_threads);

DECLARE_int32(http_server_max_concurrency);

DECLARE_int32(rpc_server_port);

DECLARE_int32(rpc_server_idle_timeout_s);

DECLARE_int32(rpc_server_num_threads);

DECLARE_int32(rpc_server_max_concurrency);

DECLARE_uint32(xxh3_128bits_seed);

DECLARE_int32(timeout_ms);

DECLARE_int32(connect_timeout_ms);

DECLARE_string(listen_addr);

DECLARE_int32(port);

DECLARE_int32(idle_timeout_s);

DECLARE_int32(num_threads);

DECLARE_int32(max_concurrency);

DECLARE_string(etcd_addr);

DECLARE_string(etcd_namespace);

DECLARE_string(load_balance_policy);

DECLARE_string(kv_route_mode);
DECLARE_bool(kv_route_enforced_gate_open);
DECLARE_uint32(kv_route_enforced_bucket_permyriad);
DECLARE_uint64(kv_route_bytes_per_token);
DECLARE_uint64(flow_max_queued_requests);
DECLARE_uint64(flow_max_dispatched_contexts);
DECLARE_uint64(flow_max_queued_prompt_tokens);
DECLARE_uint64(flow_max_queued_bytes);
DECLARE_uint64(flow_max_queue_wait_ms);
DECLARE_uint64(flow_max_queued_requests_per_tenant);
DECLARE_uint64(flow_max_queued_tokens_per_tenant);
DECLARE_uint64(flow_max_model_queued_requests);
DECLARE_uint64(flow_max_model_dispatched_contexts);
DECLARE_uint64(flow_max_model_queued_prompt_tokens);
DECLARE_uint64(flow_max_model_queued_bytes);
DECLARE_uint64(flow_service_crash_request_budget);
DECLARE_uint64(flow_service_memory_budget_bytes);
DECLARE_uint64(flow_dispatched_context_bytes);
DECLARE_double(flow_dispatch_rate_lb_per_second);
DECLARE_uint64(flow_probe_round_ub_ms);
DECLARE_uint64(flow_blind_dispatch_probe_concurrency);
DECLARE_uint64(flow_starvation_dispatch_bound);
DECLARE_string(flow_order);
DECLARE_int32(flow_dispatch_interval_ms);
DECLARE_string(flow_drain_policy);
DECLARE_bool(native_local_prefill_enabled);
DECLARE_uint32(native_local_prefill_bucket_permyriad);
DECLARE_uint64(native_local_prefill_token_cap);
DECLARE_bool(native_prefill_only_enabled);
DECLARE_uint64(native_prefill_only_output_token_cap);

DECLARE_int32(detect_disconnected_instance_interval);

DECLARE_int32(instance_delete_probe_timeout_ms);

DECLARE_int32(instance_delete_probe_attempts);

DECLARE_int32(lease_lost_heartbeat_timeout_ms);

DECLARE_uint64(output_reorder_max_events);
DECLARE_uint64(output_reorder_max_bytes);
DECLARE_int32(request_watchdog_interval_ms);
DECLARE_int32(output_gap_timeout_ms);
DECLARE_int32(output_gap_query_timeout_ms);
DECLARE_uint64(output_gap_query_batch_size);
DECLARE_int32(p_first_event_retry_ub_ms);
DECLARE_int32(first_event_dispatch_margin_ms);
DECLARE_uint64(max_first_output_attempt_retries);
DECLARE_uint64(max_nonstream_retry_wasted_device_ms);
DECLARE_uint64(min_first_output_retry_remaining_ms);
DECLARE_uint64(request_deadline_capacity);
DECLARE_uint64(request_watchdog_batch_size);
DECLARE_int32(default_request_deadline_ms);

DECLARE_int32(block_size);

DECLARE_string(tokenizer_path);
DECLARE_string(native_renderer_digest);

DECLARE_bool(enable_request_trace);

DECLARE_int32(target_ttft);

DECLARE_int32(target_tpot);

DECLARE_string(tool_call_parser);

DECLARE_string(reasoning_parser);

DECLARE_int32(readiness_check_interval_s);

DECLARE_string(default_backend_type);

DECLARE_int32(vllm_http_timeout_ms);

DECLARE_string(internal_api_token);
