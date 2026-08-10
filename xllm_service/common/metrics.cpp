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

#include "metrics.h"

DEFINE_COUNTER(server_request_in_total,
               "Total number of request that server received");
DEFINE_COUNTER(attempt_control_no_channel_total,
               "Attempt-control calls without an Engine channel");
DEFINE_COUNTER(attempt_control_token_invalid_total,
               "vLLM attempt-control calls blocked by invalid auth config");
DEFINE_COUNTER(attempt_control_rpc_failed_total,
               "Attempt-control RPC or response validation failures");
DEFINE_COUNTER(attempt_control_non_terminal_total,
               "Attempt-control replies that did not prove convergence");

DEFINE_GAUGE(xllm_service_v2_queued_requests, "Current queued V2 requests");
DEFINE_GAUGE(xllm_service_v2_dispatched_requests,
             "Current dispatched V2 requests");
DEFINE_GAUGE(xllm_service_v2_queued_prompt_tokens,
             "Current queued V2 prompt tokens");
DEFINE_GAUGE(xllm_service_v2_queued_bytes,
             "Current queued normalized V2 request bytes");
DEFINE_GAUGE(xllm_service_v2_active_requests,
             "Current Service-owned request contexts");
DEFINE_GAUGE(xllm_service_v2_observability_ring_events,
             "Current buffered request events");
DEFINE_GAUGE(xllm_service_v2_engine_kv_reporting_engines,
             "Fresh Engine states reporting KV capacity");
DEFINE_GAUGE(xllm_service_v2_engine_kv_reporting_dp_ranks,
             "Fresh Engine DP ranks reporting KV capacity");
DEFINE_GAUGE(xllm_service_v2_engine_kv_max_used_ratio,
             "Maximum fresh Engine KV used ratio");
DEFINE_GAUGE(xllm_service_v2_engine_kv_min_free_blocks,
             "Minimum fresh Engine KV free blocks per DP rank");
DEFINE_GAUGE(xllm_service_v2_engine_kv_total_free_blocks,
             "Sum of fresh Engine KV free blocks across DP ranks");

DEFINE_MULTI_COUNTER(xllm_service_v2_request_lifecycle_total,
                     "phase",
                     "Request lifecycle transitions by bounded phase");
DEFINE_MULTI_COUNTER(xllm_service_v2_request_failure_total,
                     "reason",
                     "Request failures by bounded protocol reason");
DEFINE_MULTI_COUNTER(xllm_service_v2_request_terminal_total,
                     "result",
                     "Request terminal transitions by bounded result");
DEFINE_MULTI_COUNTER(xllm_service_v2_execution_mode_total,
                     "mode",
                     "Dispatched requests by execution mode");
DEFINE_MULTI_COUNTER(xllm_service_v2_observability_events_total,
                     "outcome",
                     "Request-event recorder outcomes");
DEFINE_MULTI_COUNTER(xllm_service_v2_output_sequence_total,
                     "outcome",
                     "Output sequencing outcomes");

// ttft latency histogram
DEFINE_HISTOGRAM(time_to_first_token_latency_milliseconds,
                 "Histogram of time to first token latency in milliseconds");
// inter token latency histogram
DEFINE_HISTOGRAM(inter_token_latency_milliseconds,
                 "Histogram of inter token latency in milliseconds");
DEFINE_MULTI_HISTOGRAM(xllm_service_v2_queue_wait_milliseconds,
                       "mode",
                       "Service queue wait by execution mode");
DEFINE_MULTI_HISTOGRAM(xllm_service_v2_ttft_milliseconds,
                       "mode",
                       "Server TTFT by execution mode");
DEFINE_MULTI_HISTOGRAM(xllm_service_v2_tpot_milliseconds,
                       "mode",
                       "Server TPOT by execution mode");
DEFINE_MULTI_HISTOGRAM(xllm_service_v2_e2e_milliseconds,
                       "mode",
                       "Server E2E by execution mode");
