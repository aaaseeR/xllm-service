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

#include "telemetry/prometheus_metrics.h"

#include <gtest/gtest.h>

namespace xllm_service {
namespace {

TEST(PrometheusMetricsTest, BuildsSnapshotFromLegacyDebugSummary) {
  nlohmann::json summary = {
      {"service_name", "127.0.0.1:8889"},
      {"instance_view",
       {{"instance_count", 2},
        {"suspect_instance_count", 1},
        {"load_metrics_count", 2},
        {"latency_metrics_count", 1},
        {"load_metrics",
         {{"prefill-a",
           {{"waiting_requests_num", 3}, {"gpu_cache_usage_perc", 0.25}}},
          {"decode-a",
           {{"waiting_requests_num", 5}, {"gpu_cache_usage_perc", 0.75}}}}},
        {"inflight_request_counts", {{"prefill-a", 2}, {"decode-a", 4}}}}},
      {"cache_index",
       {{"cache_index_size", 10},
        {"hbm_entry_count", 8},
        {"dram_entry_count", 2},
        {"ssd_entry_count", 1},
        {"hbm_instance_count", 2},
        {"dram_instance_count", 1},
        {"ssd_instance_count", 1}}},
      {"dispatcher",
       {{"inflight", 3},
        {"transport_failure_total", 7},
        {"channel_count", 2},
        {"endpoint_count", 4}}}};

  PrometheusMetricsSnapshot snapshot = build_prometheus_metrics_snapshot(
      summary,
      "fallback",
      /*block_size=*/128,
      /*ready=*/true,
      /*runtime_phase=*/"running");

  EXPECT_EQ(snapshot.service_name, "127.0.0.1:8889");
  EXPECT_TRUE(snapshot.ready);
  EXPECT_EQ(snapshot.runtime_phase, "running");
  EXPECT_EQ(snapshot.instance_count, 2);
  EXPECT_EQ(snapshot.suspect_instance_count, 1);
  EXPECT_EQ(snapshot.total_waiting_requests, 8);
  EXPECT_EQ(snapshot.total_running_requests, 6);
  EXPECT_DOUBLE_EQ(snapshot.max_gpu_cache_usage_perc, 0.75);
  EXPECT_EQ(snapshot.transport_inflight, 3);
  EXPECT_EQ(snapshot.transport_failure_total, 7);
  EXPECT_EQ(snapshot.channel_count, 2);
  EXPECT_EQ(snapshot.endpoint_count, 4);
  EXPECT_EQ(snapshot.cache_index_size, 10);
  EXPECT_EQ(snapshot.block_size, 128);
}

TEST(PrometheusMetricsTest, RendersLlmDCompatibleMetrics) {
  PrometheusMetricsSnapshot snapshot;
  snapshot.service_name = "127.0.0.1:8889";
  snapshot.ready = true;
  snapshot.runtime_phase = "running";
  snapshot.block_size = 128;
  snapshot.total_waiting_requests = 8;
  snapshot.total_running_requests = 6;
  snapshot.max_gpu_cache_usage_perc = 0.75;
  snapshot.transport_inflight = 3;
  snapshot.transport_failure_total = 7;
  snapshot.channel_count = 2;
  snapshot.endpoint_count = 4;

  std::string output = render_prometheus_metrics(snapshot);

  EXPECT_NE(output.find("xllm_service_routing_mode{mode=\"legacy\"} 1"),
            std::string::npos);
  EXPECT_NE(output.find("xllm_service_runtime_phase{phase=\"running\"} 1"),
            std::string::npos);
  EXPECT_NE(output.find("xllm_service_ready 1"), std::string::npos);
  EXPECT_NE(output.find("vllm:num_requests_waiting 8"), std::string::npos);
  EXPECT_NE(output.find("vllm:num_requests_running 6"), std::string::npos);
  EXPECT_NE(output.find("vllm:kv_cache_usage_perc 0.75"), std::string::npos);
  EXPECT_NE(output.find("xllm_service_block_size 128"),
            std::string::npos);
  EXPECT_NE(output.find("xllm_service_transport_inflight 3"),
            std::string::npos);
  EXPECT_NE(output.find("# TYPE xllm_service_transport_failures_total counter"),
            std::string::npos);
  EXPECT_NE(output.find("xllm_service_transport_failures_total 7"),
            std::string::npos);
  EXPECT_NE(output.find("xllm_service_channel_count 2"),
            std::string::npos);
  EXPECT_NE(output.find("xllm_service_endpoint_count 4"),
            std::string::npos);
}

}  // namespace
}  // namespace xllm_service
