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

  PROPERTY(int32_t, detect_disconnected_instance_interval) = 15;

  PROPERTY(int32_t, instance_delete_probe_timeout_ms) = 1000;

  PROPERTY(int32_t, instance_delete_probe_attempts) = 2;

  PROPERTY(int32_t, lease_lost_heartbeat_timeout_ms) = 3000;

  // scheduler options
  PROPERTY(std::string, load_balance_policy);

  PROPERTY(int32_t, block_size) = 128;

  PROPERTY(uint32_t, xxh3_128bits_seed) = 1024;

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
