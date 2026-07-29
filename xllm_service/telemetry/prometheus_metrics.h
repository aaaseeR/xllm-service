/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

namespace xllm_service {

struct PrometheusMetricsSnapshot {
  std::string service_name;
  bool ready = false;
  std::string routing_mode = "legacy";
  int32_t block_size = 0;
  uint64_t instance_count = 0;
  uint64_t suspect_instance_count = 0;
  uint64_t load_metrics_count = 0;
  uint64_t latency_metrics_count = 0;
  uint64_t total_waiting_requests = 0;
  uint64_t total_running_requests = 0;
  double max_gpu_cache_usage_perc = 0.0;
  uint64_t cache_index_size = 0;
  uint64_t hbm_entry_count = 0;
  uint64_t dram_entry_count = 0;
  uint64_t ssd_entry_count = 0;
  uint64_t hbm_instance_count = 0;
  uint64_t dram_instance_count = 0;
  uint64_t ssd_instance_count = 0;
};

PrometheusMetricsSnapshot build_prometheus_metrics_snapshot(
    const nlohmann::json& scheduler_summary,
    const std::string& service_name,
    int32_t block_size,
    bool ready);

std::string render_prometheus_metrics(
    const PrometheusMetricsSnapshot& snapshot);

}  // namespace xllm_service
