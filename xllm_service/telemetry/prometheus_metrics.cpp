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

#include <algorithm>
#include <sstream>

namespace xllm_service {
namespace {

uint64_t read_uint64(const nlohmann::json& value, uint64_t default_value = 0) {
  if (value.is_number_unsigned()) {
    return value.get<uint64_t>();
  }
  if (value.is_number_integer()) {
    int64_t signed_value = value.get<int64_t>();
    if (signed_value >= 0) {
      return static_cast<uint64_t>(signed_value);
    }
  }
  if (value.is_number_float()) {
    double double_value = value.get<double>();
    if (double_value >= 0.0) {
      return static_cast<uint64_t>(double_value);
    }
  }
  return default_value;
}

uint64_t read_uint64_field(const nlohmann::json& object,
                           const char* key,
                           uint64_t default_value = 0) {
  if (!object.is_object()) {
    return default_value;
  }
  auto it = object.find(key);
  if (it == object.end()) {
    return default_value;
  }
  return read_uint64(*it, default_value);
}

double read_double_field(const nlohmann::json& object,
                         const char* key,
                         double default_value = 0.0) {
  if (!object.is_object()) {
    return default_value;
  }
  auto it = object.find(key);
  if (it == object.end() || !it->is_number()) {
    return default_value;
  }
  return it->get<double>();
}

std::string read_string_field(const nlohmann::json& object,
                              const char* key,
                              const std::string& default_value) {
  if (!object.is_object()) {
    return default_value;
  }
  auto it = object.find(key);
  if (it == object.end() || !it->is_string()) {
    return default_value;
  }
  return it->get<std::string>();
}

const nlohmann::json& object_field(const nlohmann::json& object,
                                   const char* key) {
  static const nlohmann::json kEmptyObject = nlohmann::json::object();
  if (!object.is_object()) {
    return kEmptyObject;
  }
  auto it = object.find(key);
  if (it == object.end() || !it->is_object()) {
    return kEmptyObject;
  }
  return *it;
}

std::string escape_label_value(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (char c : value) {
    if (c == '\\' || c == '"') {
      escaped.push_back('\\');
      escaped.push_back(c);
    } else if (c == '\n') {
      escaped.append("\\n");
    } else {
      escaped.push_back(c);
    }
  }
  return escaped;
}

void append_gauge(std::ostringstream* out,
                  const std::string& name,
                  const std::string& help,
                  double value) {
  *out << "# HELP " << name << " " << help << "\n";
  *out << "# TYPE " << name << " gauge\n";
  *out << name << " " << value << "\n";
}

void append_counter(std::ostringstream* out,
                    const std::string& name,
                    const std::string& help,
                    uint64_t value) {
  *out << "# HELP " << name << " " << help << "\n";
  *out << "# TYPE " << name << " counter\n";
  *out << name << " " << value << "\n";
}

void append_labeled_gauge(std::ostringstream* out,
                          const std::string& name,
                          const std::string& help,
                          const std::string& label_name,
                          const std::string& label_value,
                          double value) {
  *out << "# HELP " << name << " " << help << "\n";
  *out << "# TYPE " << name << " gauge\n";
  *out << name << "{" << label_name << "=\""
       << escape_label_value(label_value) << "\"} " << value << "\n";
}

}  // namespace

PrometheusMetricsSnapshot build_prometheus_metrics_snapshot(
    const nlohmann::json& scheduler_summary,
    const std::string& service_name,
    int32_t block_size,
    bool ready,
    const std::string& runtime_phase) {
  PrometheusMetricsSnapshot snapshot;
  snapshot.service_name = service_name;
  snapshot.block_size = block_size;
  snapshot.ready = ready;
  snapshot.runtime_phase = runtime_phase;
  snapshot.routing_mode =
      read_string_field(scheduler_summary, "routing_mode", "legacy");

  if (scheduler_summary.is_object() &&
      scheduler_summary.contains("service_name") &&
      scheduler_summary["service_name"].is_string()) {
    snapshot.service_name = scheduler_summary["service_name"].get<std::string>();
  }

  const nlohmann::json& instance_view =
      object_field(scheduler_summary, "instance_view");
  snapshot.instance_count = read_uint64_field(instance_view, "instance_count");
  snapshot.suspect_instance_count =
      read_uint64_field(instance_view, "suspect_instance_count");
  snapshot.load_metrics_count =
      read_uint64_field(instance_view, "load_metrics_count");
  snapshot.latency_metrics_count =
      read_uint64_field(instance_view, "latency_metrics_count");

  const nlohmann::json& load_metrics =
      object_field(instance_view, "load_metrics");
  if (snapshot.routing_mode == "external") {
    const std::string endpoint = read_string_field(
        scheduler_summary, "external_backend_endpoint", "");
    auto it = load_metrics.find(endpoint);
    if (it != load_metrics.end() && it->is_object()) {
      snapshot.load_metrics_count = 1;
      snapshot.total_waiting_requests =
          read_uint64_field(*it, "waiting_requests_num");
      snapshot.max_gpu_cache_usage_perc =
          read_double_field(*it, "gpu_cache_usage_perc");
    } else {
      snapshot.load_metrics_count = 0;
    }
    const nlohmann::json& latency_metrics =
        object_field(instance_view, "latency_metrics");
    auto latency_it = latency_metrics.find(endpoint);
    snapshot.latency_metrics_count =
        latency_it != latency_metrics.end() && latency_it->is_object() ? 1 : 0;
    snapshot.instance_count = ready ? 1 : 0;
    snapshot.suspect_instance_count = 0;
  } else {
    for (auto it = load_metrics.begin(); it != load_metrics.end(); ++it) {
      const nlohmann::json& metrics = it.value();
      snapshot.total_waiting_requests +=
          read_uint64_field(metrics, "waiting_requests_num");
      snapshot.max_gpu_cache_usage_perc =
          std::max(snapshot.max_gpu_cache_usage_perc,
                   read_double_field(metrics, "gpu_cache_usage_perc"));
    }
  }

  const nlohmann::json& inflight_request_counts =
      object_field(instance_view, "inflight_request_counts");
  for (auto it = inflight_request_counts.begin();
       it != inflight_request_counts.end();
       ++it) {
    snapshot.total_running_requests += read_uint64(it.value());
  }
  if (snapshot.routing_mode == "external") {
    snapshot.total_running_requests =
        read_uint64_field(scheduler_summary, "active_request_sessions");
  }

  const nlohmann::json& dispatcher =
      object_field(scheduler_summary, "dispatcher");
  snapshot.transport_inflight = read_uint64_field(dispatcher, "inflight");
  snapshot.transport_failure_total =
      read_uint64_field(dispatcher, "transport_failure_total");
  snapshot.channel_count = read_uint64_field(dispatcher, "channel_count");
  snapshot.endpoint_count = read_uint64_field(dispatcher, "endpoint_count");

  const nlohmann::json& cache_index =
      object_field(scheduler_summary, "cache_index");
  snapshot.cache_index_size = read_uint64_field(cache_index, "cache_index_size");
  snapshot.hbm_entry_count = read_uint64_field(cache_index, "hbm_entry_count");
  snapshot.dram_entry_count = read_uint64_field(cache_index, "dram_entry_count");
  snapshot.ssd_entry_count = read_uint64_field(cache_index, "ssd_entry_count");
  snapshot.hbm_instance_count =
      read_uint64_field(cache_index, "hbm_instance_count");
  snapshot.dram_instance_count =
      read_uint64_field(cache_index, "dram_instance_count");
  snapshot.ssd_instance_count =
      read_uint64_field(cache_index, "ssd_instance_count");
  if (snapshot.routing_mode == "external") {
    snapshot.cache_index_size = 0;
    snapshot.hbm_entry_count = 0;
    snapshot.dram_entry_count = 0;
    snapshot.ssd_entry_count = 0;
    snapshot.hbm_instance_count = 0;
    snapshot.dram_instance_count = 0;
    snapshot.ssd_instance_count = 0;
  }
  return snapshot;
}

std::string render_prometheus_metrics(
    const PrometheusMetricsSnapshot& snapshot) {
  std::ostringstream out;

  append_labeled_gauge(&out,
                       "xllm_service_routing_mode",
                       "Routing authority mode for llm-d compatibility.",
                       "mode",
                       snapshot.routing_mode,
                       1.0);
  append_labeled_gauge(&out,
                       "xllm_service_runtime_phase",
                       "Current lifecycle phase of this backend adapter.",
                       "phase",
                       snapshot.runtime_phase,
                       1.0);
  append_labeled_gauge(&out,
                       "xllm_service_info",
                       "Static xllm-service endpoint information.",
                       "service_name",
                       snapshot.service_name,
                       1.0);
  append_gauge(&out,
               "xllm_service_ready",
               "Whether xllm-service has routable backend instances.",
               snapshot.ready ? 1.0 : 0.0);
  append_gauge(&out,
               "xllm_service_instance_count",
               "Number of instances in the local xllm-service view.",
               static_cast<double>(snapshot.instance_count));
  append_gauge(&out,
               "xllm_service_suspect_instance_count",
               "Number of suspect instances in the local xllm-service view.",
               static_cast<double>(snapshot.suspect_instance_count));
  append_gauge(&out,
               "xllm_service_cache_index_size",
               "Number of prefix cache entries in the local legacy cache index.",
               static_cast<double>(snapshot.cache_index_size));
  append_gauge(&out,
               "xllm_service_hbm_entry_count",
               "Number of prefix cache entries resident in HBM.",
               static_cast<double>(snapshot.hbm_entry_count));
  append_gauge(&out,
               "xllm_service_dram_entry_count",
               "Number of prefix cache entries resident in DRAM.",
               static_cast<double>(snapshot.dram_entry_count));
  append_gauge(&out,
               "xllm_service_ssd_entry_count",
               "Number of prefix cache entries resident in SSD.",
               static_cast<double>(snapshot.ssd_entry_count));
  append_gauge(&out,
               "xllm_service_hbm_instance_count",
               "Number of instances with HBM prefix cache entries.",
               static_cast<double>(snapshot.hbm_instance_count));
  append_gauge(&out,
               "xllm_service_dram_instance_count",
               "Number of instances with DRAM prefix cache entries.",
               static_cast<double>(snapshot.dram_instance_count));
  append_gauge(&out,
               "xllm_service_ssd_instance_count",
               "Number of instances with SSD prefix cache entries.",
               static_cast<double>(snapshot.ssd_instance_count));
  append_gauge(&out,
               "xllm_service_block_size",
               "Configured number of tokens per KV cache block.",
               static_cast<double>(snapshot.block_size));
  append_gauge(&out,
               "xllm_service_transport_inflight",
               "Number of backend transport RPCs awaiting completion.",
               static_cast<double>(snapshot.transport_inflight));
  append_counter(&out,
                 "xllm_service_transport_failures_total",
                 "Number of failed backend transport dispatches.",
                 snapshot.transport_failure_total);
  append_gauge(&out,
               "xllm_service_channel_count",
               "Number of initialized backend channels.",
               static_cast<double>(snapshot.channel_count));
  append_gauge(&out,
               "xllm_service_endpoint_count",
               "Number of active backend endpoint incarnations.",
               static_cast<double>(snapshot.endpoint_count));

  append_gauge(&out,
               "vllm:num_requests_waiting",
               "llm-d compatible total queued requests gauge.",
               static_cast<double>(snapshot.total_waiting_requests));
  append_gauge(&out,
               "vllm:num_requests_running",
               "llm-d compatible total running requests gauge.",
               static_cast<double>(snapshot.total_running_requests));
  append_gauge(&out,
               "vllm:kv_cache_usage_perc",
               "llm-d compatible KV cache usage fraction gauge.",
               snapshot.max_gpu_cache_usage_perc);

  return out.str();
}

}  // namespace xllm_service
