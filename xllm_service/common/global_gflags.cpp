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

#include "common/global_gflags.h"

#include "brpc/reloadable_flags.h"

DEFINE_string(server_host,
              "",
              "Server listen address, may be IPV4/IPV6/UDS."
              " If this is set, the flag port will be ignored");

DEFINE_int32(http_server_port, 8888, "Port for xllm http service to listen on");

DEFINE_int32(http_server_idle_timeout_s,
             -1,
             "Connection will be closed if there is no "
             "read/write operations during the last `idle_timeout_s'");

DEFINE_int32(http_server_num_threads, 32, "Maximum number of threads to use");

DEFINE_int32(http_server_max_concurrency,
             128,
             "Limit number of requests processed in parallel");

DEFINE_int32(rpc_server_port, 8889, "Port for xllm rpc service to listen on");

DEFINE_int32(rpc_server_idle_timeout_s,
             -1,
             "Connection will be closed if there is no "
             "read/write operations during the last `idle_timeout_s'");

DEFINE_int32(rpc_server_num_threads, 32, "Maximum number of threads to use");

DEFINE_int32(rpc_server_max_concurrency,
             128,
             "Limit number of requests processed in parallel");

DEFINE_string(etcd_addr,
              "0.0.0.0:2379",
              "etcd adderss for save instance meta info");

DEFINE_string(
    etcd_namespace,
    "",
    "Optional etcd namespace prefix for all xllm-service keys, e.g. prod-a.");

DEFINE_uint32(xxh3_128bits_seed, 1024, "default XXH3 128bits Hash seed");

DEFINE_int32(port, 8888, "Port for xllm service to listen on");

DEFINE_int32(num_threads, 32, "Number of threads to process requests");

DEFINE_int32(max_concurrency,
             128,
             "Limit number of requests processed in parallel");

DEFINE_int32(
    timeout_ms,
    -1,
    "Max duration (millisecond) of bRPC Channel. -1 means wait indefinitely.");

DEFINE_int32(connect_timeout_ms,
             -1,
             "Max duration (millisecond) of bRPC to establish connections. -1 "
             "means wait "
             "indefinitely.");

DEFINE_string(listen_addr,
              "",
              "Server listen address, may be IPV4/IPV6/UDS."
              " If this is set, the flag port will be ignored");

DEFINE_int32(idle_timeout_s,
             -1,
             "Connection will be closed if there is no "
             "read/write operations during the last `idle_timeout_s'");

DEFINE_string(load_balance_policy,
              "RR",
              "Disaggregated prefill-decode policy.");

DEFINE_string(kv_route_mode,
              "SHADOW",
              "KV-aware route mode: DISABLED, SHADOW, or ENFORCED.");

DEFINE_bool(kv_route_enforced_gate_open,
            false,
            "Operator approval gate for ENFORCED KV-aware routing.");

DEFINE_uint32(kv_route_enforced_bucket_permyriad,
              0,
              "Stable ENFORCED traffic bucket in [0, 10000].");

DEFINE_uint64(kv_route_max_candidate_plans,
              16384,
              "Bound on retained KV route candidate plans.");
DEFINE_uint64(kv_route_least_load_shortlist,
              8,
              "Least-load candidates evaluated by the KV route planner.");
DEFINE_uint64(kv_route_top_prefix_shortlist,
              8,
              "Top-prefix candidates evaluated by the KV route planner.");
DEFINE_uint64(kv_route_prefill_queue_cost_us,
              1000,
              "Calibrated cost per queued Prefill request.");
DEFINE_uint64(kv_route_decode_request_cost_us,
              1000,
              "Calibrated cost per running Decode request.");
DEFINE_uint64(kv_route_prefill_token_cost_us,
              10,
              "Calibrated Prefill cost per uncached prompt token.");
DEFINE_double(kv_route_transfer_byte_cost_us,
              0.001,
              "Calibrated transfer cost in microseconds per KV byte.");
DEFINE_uint64(kv_route_decode_headroom_cost_us,
              1000,
              "Penalty for each Decode request beyond advertised headroom.");
DEFINE_uint64(kv_route_prefill_reserve_blocks,
              1,
              "Free KV blocks reserved before a Prefill route is feasible.");
DEFINE_uint64(kv_route_margin_us,
              100,
              "Safety margin added to predicted KV route completion cost.");
DEFINE_uint64(kv_route_near_equal_cost_us,
              10,
              "Cost delta treated as near-equal for deterministic tie-breaks.");

DEFINE_uint64(kv_route_bytes_per_token,
              0,
              "Logical KV bytes per prompt token for routing calibration.");

DEFINE_uint64(flow_max_queued_requests, 4096, "Service queued request bound.");
DEFINE_uint64(flow_max_dispatched_contexts,
              4096,
              "Service dispatched request context bound.");
DEFINE_uint64(flow_max_queued_prompt_tokens,
              16 * 1024 * 1024,
              "Service queued prompt token bound.");
DEFINE_uint64(flow_max_queued_bytes,
              512 * 1024 * 1024,
              "Service queued normalized request byte bound.");
DEFINE_uint64(flow_max_queue_wait_ms, 2000, "Maximum Service queue wait.");
DEFINE_uint64(flow_max_queued_requests_per_tenant,
              512,
              "Per-tenant queued request bound on one Service.");
DEFINE_uint64(flow_max_queued_tokens_per_tenant,
              2 * 1024 * 1024,
              "Per-tenant queued prompt token bound on one Service.");
DEFINE_uint64(flow_max_model_queued_requests,
              4096,
              "Per-ModelPool queued request bound.");
DEFINE_uint64(flow_max_model_dispatched_contexts,
              4096,
              "Per-ModelPool dispatched request context bound.");
DEFINE_uint64(flow_max_model_queued_prompt_tokens,
              16 * 1024 * 1024,
              "Per-ModelPool queued prompt token bound.");
DEFINE_uint64(flow_max_model_queued_bytes,
              512 * 1024 * 1024,
              "Per-ModelPool queued normalized request byte bound.");
DEFINE_uint64(flow_service_crash_request_budget,
              8192,
              "Maximum queued plus dispatched requests exposed by kill -9.");
DEFINE_uint64(flow_service_memory_budget_bytes,
              528 * 1024 * 1024,
              "Queue plus dispatched context memory exposure bound.");
DEFINE_uint64(flow_dispatched_context_bytes,
              4096,
              "Conservative estimated bytes charged to one dispatched "
              "request context for Service memory budgeting.");
DEFINE_double(flow_dispatch_rate_lb_per_second,
              1.0,
              "Fresh normal-observation dispatch-rate lower bound.");
DEFINE_uint64(flow_probe_round_ub_ms,
              200,
              "Upper bound for one capacity probe round.");
DEFINE_uint64(flow_blind_dispatch_probe_concurrency,
              1,
              "Maximum blind BEST_EFFORT probes in flight.");
DEFINE_uint64(flow_starvation_dispatch_bound,
              32,
              "Maximum consecutive higher-band dispatches.");
DEFINE_string(flow_order, "FCFS", "Within-flow order: FCFS or EDF.");
DEFINE_int32(flow_dispatch_interval_ms,
             10,
             "Maximum wait between Service flow dispatch scans.");
DEFINE_string(flow_drain_policy,
              "COMPLETE_QUEUED",
              "Drain policy: COMPLETE_QUEUED or RETRY_UNDISPATCHED.");
DEFINE_bool(native_local_prefill_enabled,
            false,
            "Enable allowlisted Native local Prefill/Decode selection.");
DEFINE_uint32(native_local_prefill_bucket_permyriad,
              0,
              "Stable Native local Prefill workload bucket in [0, 10000].");
DEFINE_uint64(native_local_prefill_token_cap,
              512,
              "Maximum prompt tokens for Native local Prefill.");
DEFINE_bool(native_prefill_only_enabled,
            false,
            "Enable capability-gated Native Prefill-only selection.");
DEFINE_uint64(native_prefill_only_output_token_cap,
              1,
              "Maximum output tokens for Native Prefill-only execution.");
DEFINE_bool(trusted_tenant_headers_enabled,
            false,
            "Trust authenticated x-tenant-id/x-flow-id Gateway headers. When "
            "false, tenant headers and request priority are ignored; an "
            "opaque Service-signed session scopes cross-request KV reuse.");
DEFINE_bool(trusted_client_identity_headers_enabled,
            false,
            "Trust x-authenticated-client-id injected by an authenticating "
            "Gateway for standard-SDK user session derivation. Enable only "
            "when the Gateway strips/replaces this header and Service cannot "
            "be reached directly.");
DEFINE_string(kv_session_hmac_secret,
              "",
              "Shared 32-256 byte secret for signing opaque KV session "
              "tokens. Empty generates an instance-local secret and requires "
              "sticky routing for session reuse across requests.");
DEFINE_string(kv_session_hmac_previous_secret,
              "",
              "Previous shared KV session signing secret accepted only for "
              "verification during bounded key rotation.");
DEFINE_uint64(kv_session_token_ttl_seconds,
              86400,
              "Maximum lifetime of an opaque signed KV session token.");
DEFINE_uint64(observability_event_capacity,
              65536,
              "Fixed-capacity Service request-event ring size.");
DEFINE_uint64(observability_export_batch_size,
              1024,
              "Maximum request events exported per observability tick.");
DEFINE_int32(observability_export_interval_ms,
             20,
             "Request-event export interval in milliseconds.");
DEFINE_int32(observability_snapshot_interval_ms,
             5000,
             "Cluster performance snapshot interval in milliseconds.");
DEFINE_string(observability_build_id,
              "development",
              "Immutable build or artifact identifier in request events.");

DEFINE_string(placement_config_path,
              "",
              "Strict V3 Placement JSON config. Empty disables Placement.");

namespace {

bool valid_placement_mode_override(const char*, int32_t value) {
  return value >= -1 && value <= 3;
}

}  // namespace

DEFINE_int32(placement_mode_override,
             -1,
             "Reloadable V3 Placement mode override: -1 uses config, "
             "0 DISABLED, 1 SHADOW, 2 ENFORCED_CREATE_ONLY, 3 ENFORCED.");

BRPC_VALIDATE_GFLAG(placement_mode_override, valid_placement_mode_override);

DEFINE_int32(detect_disconnected_instance_interval,
             15,
             "Deprecated V1 compatibility flag; ignored by V2 membership "
             "fencing.");

DEFINE_int32(instance_delete_probe_timeout_ms,
             1000,
             "Timeout in milliseconds for the initial health probe after an "
             "instance lease delete event.");

DEFINE_int32(instance_delete_probe_attempts,
             2,
             "Deprecated V1 compatibility flag; V2 never probes to override "
             "an authoritative membership delete.");

DEFINE_int32(lease_lost_heartbeat_timeout_ms,
             3000,
             "Deprecated V1 compatibility flag; V2 fences an authoritative "
             "membership delete immediately.");

DEFINE_uint64(output_reorder_max_events,
              64,
              "Maximum buffered output events for one Service request.");

DEFINE_uint64(output_reorder_max_bytes,
              4 * 1024 * 1024,
              "Maximum dynamic bytes in one Service output reorder buffer.");

DEFINE_int32(request_watchdog_interval_ms,
             100,
             "Service request watchdog scan interval in milliseconds.");

DEFINE_int32(output_gap_timeout_ms,
             1000,
             "Maximum local duration for an output sequence gap.");

DEFINE_int32(output_gap_query_timeout_ms,
             100,
             "Hard timeout for one Decode first-event QueryRequest.");

DEFINE_uint64(output_gap_query_batch_size,
              8,
              "Maximum concurrent first-event recovery queries per scan.");

DEFINE_int32(p_first_event_retry_ub_ms,
             700,
             "Prefill first-event Commit/Query retry budget.");

DEFINE_int32(first_event_dispatch_margin_ms,
             200,
             "Reserved first-event delivery and Cancel margin.");

DEFINE_uint64(max_first_output_attempt_retries,
              1,
              "Maximum Service attempt replacements before first output.");

DEFINE_uint64(max_nonstream_retry_wasted_device_ms,
              5000,
              "Maximum cumulative device time discarded by request retry.");

DEFINE_uint64(min_first_output_retry_remaining_ms,
              1000,
              "Minimum remaining request deadline required for retry.");

DEFINE_uint64(request_deadline_capacity,
              65536,
              "Maximum indexed request deadlines and disconnect monitors.");

DEFINE_uint64(request_watchdog_batch_size,
              1024,
              "Maximum deadline or disconnect records handled per scan.");

DEFINE_int32(default_request_deadline_ms,
             300000,
             "Default business deadline when the client omits one.");

DEFINE_int32(block_size,
             128,
             "Number of slots per kv cache block. Default is 128.");

DEFINE_string(tokenizer_path, "", "tokenizer config path.");

DEFINE_string(native_renderer_digest,
              "",
              "Digest of the local tokenizer and chat rendering contract.");

DEFINE_bool(enable_request_trace, false, "Whether to enable request trace");

DEFINE_int32(target_ttft,
             1000,
             "Target Time to First Token (TTFT), in milliseconds.");

BRPC_VALIDATE_GFLAG(target_ttft, brpc::NonNegativeInteger);

DEFINE_int32(target_tpot,
             50,
             "Target Time Per Output Token (TPOT), in milliseconds.");

BRPC_VALIDATE_GFLAG(target_tpot, brpc::NonNegativeInteger);

DEFINE_string(reasoning_parser,
              "",
              "Specify the reasoning parser for handling reasoning "
              "interactions(e.g. auto, glm45, glm47, qwen3, deepseek-r1).");

DEFINE_string(tool_call_parser,
              "",
              "Specify the parser for handling tool-call interactions(e.g. "
              "auto, qwen25, qwen3, kimi_k2, deepseekv3, glm45, glm47).");

DEFINE_int32(readiness_check_interval_s,
             3,
             "Interval in seconds to check for available instance groups "
             "before starting and during runtime of the HTTP service.");

BRPC_VALIDATE_GFLAG(readiness_check_interval_s, brpc::PositiveInteger);

DEFINE_string(default_backend_type,
              "xllm",
              "Deprecated compatibility flag. Runtime provider selection is "
              "derived from registered Provider identity.");

DEFINE_int32(vllm_http_timeout_ms,
             60000,
             "HTTP timeout (ms) when forwarding to vLLM backend.");

DEFINE_string(internal_api_token,
              "",
              "Shared token guarding the internal Heartbeat endpoint via the "
              "X-Internal-Token header. Empty value disables the check "
              "(backwards compatible).");
