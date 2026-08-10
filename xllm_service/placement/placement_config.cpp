/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "placement/placement_config.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <string_view>
#include <utility>

namespace xllm_service::placement {
namespace {

using Json = nlohmann::json;
inline constexpr size_t kMaxPlacementInternalTokenBytes = 4096;

PlacementConfigStatus fail(PlacementConfigStatus status,
                           std::string message,
                           std::string* error) {
  if (error != nullptr) {
    *error = std::move(message);
  }
  return status;
}

bool exact_fields(const Json& object,
                  std::initializer_list<std::string_view> fields) {
  if (!object.is_object() || object.size() != fields.size()) {
    return false;
  }
  for (std::string_view field : fields) {
    if (!object.contains(field)) {
      return false;
    }
  }
  return true;
}

bool uint64_field(const Json& object, const char* name, uint64_t* value) {
  if (!object.at(name).is_number_unsigned()) {
    return false;
  }
  *value = object.at(name).get<uint64_t>();
  return true;
}

bool uint32_field(const Json& object, const char* name, uint32_t* value) {
  uint64_t parsed = 0;
  if (!uint64_field(object, name, &parsed) ||
      parsed > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  *value = static_cast<uint32_t>(parsed);
  return true;
}

bool size_field(const Json& object, const char* name, size_t* value) {
  uint64_t parsed = 0;
  if (!uint64_field(object, name, &parsed) ||
      parsed > std::numeric_limits<size_t>::max()) {
    return false;
  }
  *value = static_cast<size_t>(parsed);
  return true;
}

bool int32_field(const Json& object, const char* name, int32_t* value) {
  if (!object.at(name).is_number_integer()) {
    return false;
  }
  const int64_t parsed = object.at(name).get<int64_t>();
  if (parsed < std::numeric_limits<int32_t>::min() ||
      parsed > std::numeric_limits<int32_t>::max()) {
    return false;
  }
  *value = static_cast<int32_t>(parsed);
  return true;
}

bool double_field(const Json& object, const char* name, double* value) {
  if (!object.at(name).is_number()) {
    return false;
  }
  *value = object.at(name).get<double>();
  return std::isfinite(*value);
}

bool bool_field(const Json& object, const char* name, bool* value) {
  if (!object.at(name).is_boolean()) {
    return false;
  }
  *value = object.at(name).get<bool>();
  return true;
}

bool string_field(const Json& object, const char* name, std::string* value) {
  if (!object.at(name).is_string()) {
    return false;
  }
  *value = object.at(name).get<std::string>();
  return true;
}

bool valid_internal_token(const std::string& value) {
  return !value.empty() && value.size() <= kMaxPlacementInternalTokenBytes &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return character >= '!' && character <= '~';
         });
}

bool parse_mode(const Json& json, PlacementMode* mode) {
  if (!json.is_string()) {
    return false;
  }
  const std::string value = json.get<std::string>();
  if (value == "DISABLED") {
    *mode = PlacementMode::DISABLED;
  } else if (value == "SHADOW") {
    *mode = PlacementMode::SHADOW;
  } else if (value == "ENFORCED_CREATE_ONLY") {
    *mode = PlacementMode::ENFORCED_CREATE_ONLY;
  } else if (value == "ENFORCED") {
    *mode = PlacementMode::ENFORCED;
  } else {
    return false;
  }
  return true;
}

bool parse_provider(const Json& json, xllm::proto::ProviderId* provider) {
  if (!json.is_string()) {
    return false;
  }
  const std::string value = json.get<std::string>();
  if (value == "XLLM_NATIVE") {
    *provider = xllm::proto::PROVIDER_ID_XLLM_NATIVE;
  } else if (value == "VLLM_ASCEND") {
    *provider = xllm::proto::PROVIDER_ID_VLLM_ASCEND;
  } else {
    return false;
  }
  return true;
}

bool parse_role(const Json& json, xllm::proto::EngineRole* role) {
  if (!json.is_string()) {
    return false;
  }
  const std::string value = json.get<std::string>();
  if (value == "AGGREGATED") {
    *role = xllm::proto::ENGINE_ROLE_AGGREGATED;
  } else if (value == "PREFILL") {
    *role = xllm::proto::ENGINE_ROLE_PREFILL;
  } else if (value == "DECODE") {
    *role = xllm::proto::ENGINE_ROLE_DECODE;
  } else {
    return false;
  }
  return true;
}

bool parse_planner(const Json& json, PlacementPlannerConfig* config) {
  if (!exact_fields(json,
                    {"scale_up_hold_ms",
                     "scale_down_stabilization_ms",
                     "cooldown_ms",
                     "economic_horizon_ms",
                     "min_scale_down_samples",
                     "max_scale_up_step",
                     "max_scale_down_step",
                     "queue_high_watermark",
                     "queue_low_watermark",
                     "admission_reject_high_watermark",
                     "admission_reject_low_watermark",
                     "kv_high_watermark",
                     "kv_low_watermark"})) {
    return false;
  }
  return uint64_field(json, "scale_up_hold_ms", &config->scale_up_hold_ms) &&
         uint64_field(json,
                      "scale_down_stabilization_ms",
                      &config->scale_down_stabilization_ms) &&
         uint64_field(json, "cooldown_ms", &config->cooldown_ms) &&
         uint64_field(
             json, "economic_horizon_ms", &config->economic_horizon_ms) &&
         uint64_field(
             json, "min_scale_down_samples", &config->min_scale_down_samples) &&
         uint32_field(json, "max_scale_up_step", &config->max_scale_up_step) &&
         uint32_field(
             json, "max_scale_down_step", &config->max_scale_down_step) &&
         double_field(
             json, "queue_high_watermark", &config->queue_high_watermark) &&
         double_field(
             json, "queue_low_watermark", &config->queue_low_watermark) &&
         double_field(json,
                      "admission_reject_high_watermark",
                      &config->admission_reject_high_watermark) &&
         double_field(json,
                      "admission_reject_low_watermark",
                      &config->admission_reject_low_watermark) &&
         double_field(json, "kv_high_watermark", &config->kv_high_watermark) &&
         double_field(json, "kv_low_watermark", &config->kv_low_watermark);
}

bool parse_reconcile(const Json& json, PlacementReconcileConfig* config) {
  return exact_fields(json,
                      {"max_operations_per_cycle",
                       "max_operations_per_pool",
                       "max_create_per_cycle",
                       "max_drain_per_cycle",
                       "terminal_visibility_grace_ms"}) &&
         uint32_field(json,
                      "max_operations_per_cycle",
                      &config->max_operations_per_cycle) &&
         uint32_field(json,
                      "max_operations_per_pool",
                      &config->max_operations_per_pool) &&
         uint32_field(
             json, "max_create_per_cycle", &config->max_create_per_cycle) &&
         uint32_field(
             json, "max_drain_per_cycle", &config->max_drain_per_cycle) &&
         uint64_field(json,
                      "terminal_visibility_grace_ms",
                      &config->terminal_visibility_grace_ms);
}

bool parse_controller(const Json& json, PlacementControllerConfig* config) {
  if (!exact_fields(json,
                    {"mode",
                     "max_pools",
                     "max_desired_snapshot_bytes",
                     "max_operation_snapshot_bytes",
                     "max_devices",
                     "max_new_operations_per_cycle",
                     "max_actuator_actions_per_cycle",
                     "planner",
                     "reconcile"})) {
    return false;
  }
  return parse_mode(json.at("mode"), &config->mode) &&
         size_field(json, "max_pools", &config->max_pools) &&
         size_field(json,
                    "max_desired_snapshot_bytes",
                    &config->max_desired_snapshot_bytes) &&
         size_field(json,
                    "max_operation_snapshot_bytes",
                    &config->max_operation_snapshot_bytes) &&
         uint64_field(json, "max_devices", &config->max_devices) &&
         uint32_field(json,
                      "max_new_operations_per_cycle",
                      &config->max_new_operations_per_cycle) &&
         uint32_field(json,
                      "max_actuator_actions_per_cycle",
                      &config->max_actuator_actions_per_cycle) &&
         parse_planner(json.at("planner"), &config->planner) &&
         parse_reconcile(json.at("reconcile"), &config->reconcile);
}

bool parse_executor(const Json& json,
                    PlacementOperationExecutorConfig* config) {
  return exact_fields(json,
                      {"max_records",
                       "max_message_bytes",
                       "operation_timeout_ms",
                       "terminal_retention_ms",
                       "max_terminal_compactions_per_cycle"}) &&
         size_field(json, "max_records", &config->max_records) &&
         size_field(json, "max_message_bytes", &config->max_message_bytes) &&
         uint64_field(
             json, "operation_timeout_ms", &config->operation_timeout_ms) &&
         uint64_field(
             json, "terminal_retention_ms", &config->terminal_retention_ms) &&
         uint32_field(json,
                      "max_terminal_compactions_per_cycle",
                      &config->max_terminal_compactions_per_cycle);
}

bool parse_observation(const Json& json,
                       PlacementObservationCollectorConfig* config) {
  return exact_fields(json,
                      {"max_models",
                       "bucket_count",
                       "bucket_width_ms",
                       "max_latency_samples_per_bucket",
                       "forecast_horizon_ms",
                       "forecast_headroom"}) &&
         size_field(json, "max_models", &config->max_models) &&
         size_field(json, "bucket_count", &config->bucket_count) &&
         uint64_field(json, "bucket_width_ms", &config->bucket_width_ms) &&
         size_field(json,
                    "max_latency_samples_per_bucket",
                    &config->max_latency_samples_per_bucket) &&
         uint64_field(
             json, "forecast_horizon_ms", &config->forecast_horizon_ms) &&
         double_field(json, "forecast_headroom", &config->forecast_headroom);
}

bool parse_input_builder(const Json& json,
                         PlacementInputBuilderConfig* config) {
  return exact_fields(json, {"max_pools", "max_members"}) &&
         size_field(json, "max_pools", &config->max_pools) &&
         size_field(json, "max_members", &config->max_members);
}

bool parse_transports(const Json& json, PlacementTransportConfig* config) {
  if (!exact_fields(json,
                    {"native_timeout_ms",
                     "native_max_channels",
                     "vllm_timeout_ms",
                     "vllm_max_channels",
                     "vllm_max_response_bytes",
                     "vllm_internal_api_token"})) {
    return false;
  }
  return int32_field(json, "native_timeout_ms", &config->native.timeout_ms) &&
         size_field(
             json, "native_max_channels", &config->native.max_channels) &&
         int32_field(
             json, "vllm_timeout_ms", &config->vllm_ascend.timeout_ms) &&
         size_field(
             json, "vllm_max_channels", &config->vllm_ascend.max_channels) &&
         size_field(json,
                    "vllm_max_response_bytes",
                    &config->vllm_ascend.max_response_bytes) &&
         string_field(json,
                      "vllm_internal_api_token",
                      &config->vllm_ascend.internal_api_token);
}

bool parse_deployment(const Json& json,
                      HttpPlacementDeploymentActuatorConfig* config) {
  return exact_fields(json,
                      {"address",
                       "timeout_ms",
                       "max_response_bytes",
                       "internal_api_token"}) &&
         string_field(json, "address", &config->address) &&
         int32_field(json, "timeout_ms", &config->timeout_ms) &&
         size_field(json, "max_response_bytes", &config->max_response_bytes) &&
         string_field(json, "internal_api_token", &config->internal_api_token);
}

bool parse_external(const Json& json,
                    PlacementObservationExternalInputs* external) {
  return exact_fields(json,
                      {"queue_depth",
                       "kv_used_ratio",
                       "full_cache_loss_cost",
                       "confirmed_store_coverage",
                       "out_of_distribution"}) &&
         double_field(json, "queue_depth", &external->queue_depth) &&
         double_field(json, "kv_used_ratio", &external->kv_used_ratio) &&
         double_field(
             json, "full_cache_loss_cost", &external->full_cache_loss_cost) &&
         double_field(json,
                      "confirmed_store_coverage",
                      &external->confirmed_store_coverage) &&
         bool_field(
             json, "out_of_distribution", &external->out_of_distribution);
}

bool parse_pool(const Json& json, PlacementPoolRuntimeSpec* pool) {
  if (!exact_fields(json,
                    {"provider",
                     "model_revision",
                     "role",
                     "profile_digest",
                     "devices_per_replica",
                     "instance_cost_per_hour",
                     "load_warmup_p99_ms",
                     "prefill_tokens_per_second_under_slo",
                     "decode_tokens_per_second_under_slo",
                     "requests_per_second_under_slo",
                     "target_utilization",
                     "min_replicas",
                     "max_replicas",
                     "failure_headroom_replicas",
                     "ttft_slo_ms",
                     "tpot_slo_ms",
                     "priority",
                     "slo_risk_score",
                     "config_digest",
                     "external"})) {
    return false;
  }
  PlacementCapacityProfile& profile = pool->profile;
  return parse_provider(json.at("provider"), &profile.pool.provider_id) &&
         string_field(json, "model_revision", &profile.pool.model_revision) &&
         parse_role(json.at("role"), &profile.pool.role) &&
         string_field(json, "profile_digest", &profile.pool.profile_digest) &&
         uint32_field(
             json, "devices_per_replica", &profile.devices_per_replica) &&
         double_field(
             json, "instance_cost_per_hour", &profile.instance_cost_per_hour) &&
         uint64_field(
             json, "load_warmup_p99_ms", &profile.load_warmup_p99_ms) &&
         double_field(json,
                      "prefill_tokens_per_second_under_slo",
                      &profile.prefill_tokens_per_second_under_slo) &&
         double_field(json,
                      "decode_tokens_per_second_under_slo",
                      &profile.decode_tokens_per_second_under_slo) &&
         double_field(json,
                      "requests_per_second_under_slo",
                      &profile.requests_per_second_under_slo) &&
         double_field(
             json, "target_utilization", &profile.target_utilization) &&
         uint32_field(json, "min_replicas", &profile.min_replicas) &&
         uint32_field(json, "max_replicas", &profile.max_replicas) &&
         uint32_field(json,
                      "failure_headroom_replicas",
                      &profile.failure_headroom_replicas) &&
         double_field(json, "ttft_slo_ms", &profile.ttft_slo_ms) &&
         double_field(json, "tpot_slo_ms", &profile.tpot_slo_ms) &&
         uint32_field(json, "priority", &pool->priority) &&
         double_field(json, "slo_risk_score", &pool->slo_risk_score) &&
         string_field(json, "config_digest", &pool->config_digest) &&
         parse_external(json.at("external"), &pool->external);
}

std::string pool_identity(const PlacementPoolKey& pool) {
  return std::to_string(static_cast<int32_t>(pool.provider_id)) + "\n" +
         pool.model_revision + "\n" +
         std::to_string(static_cast<int32_t>(pool.role)) + "\n" +
         pool.profile_digest;
}

}  // namespace

PlacementConfigStatus parse_placement_runtime_config(
    const std::string& value,
    PlacementRuntimeConfig* config,
    std::string* error) {
  if (config == nullptr || value.empty()) {
    return fail(PlacementConfigStatus::INVALID_INPUT,
                "configuration and output must not be empty",
                error);
  }
  if (value.size() > kMaxPlacementConfigBytes) {
    return fail(PlacementConfigStatus::CAPACITY_EXCEEDED,
                "configuration exceeds byte limit",
                error);
  }
  const Json json = Json::parse(value, nullptr, /*allow_exceptions=*/false);
  if (json.is_discarded()) {
    return fail(PlacementConfigStatus::INVALID_JSON, "invalid JSON", error);
  }
  if (!exact_fields(json,
                    {"schema_version",
                     "loop_interval_ms",
                     "controller",
                     "executor",
                     "observation",
                     "input_builder",
                     "transports",
                     "deployment",
                     "pools"})) {
    return fail(PlacementConfigStatus::INVALID_SCHEMA,
                "top-level fields do not match V3 schema",
                error);
  }
  PlacementRuntimeConfig parsed;
  uint32_t schema_version = 0;
  try {
    if (!uint32_field(json, "schema_version", &schema_version) ||
        schema_version != kPlacementConfigSchemaVersion ||
        !uint64_field(json, "loop_interval_ms", &parsed.loop_interval_ms) ||
        !parse_controller(json.at("controller"), &parsed.controller) ||
        !parse_executor(json.at("executor"), &parsed.executor) ||
        !parse_observation(json.at("observation"), &parsed.observation) ||
        !parse_input_builder(json.at("input_builder"), &parsed.input_builder) ||
        !parse_transports(json.at("transports"), &parsed.transports) ||
        !parse_deployment(json.at("deployment"), &parsed.deployment) ||
        !json.at("pools").is_array() ||
        json.at("pools").size() > parsed.controller.max_pools) {
      return fail(PlacementConfigStatus::INVALID_SCHEMA,
                  "configuration field has invalid type or value",
                  error);
    }
    parsed.pools.reserve(json.at("pools").size());
    for (const Json& pool_json : json.at("pools")) {
      PlacementPoolRuntimeSpec pool;
      if (!parse_pool(pool_json, &pool)) {
        return fail(PlacementConfigStatus::INVALID_SCHEMA,
                    "pool field has invalid type or unknown field",
                    error);
      }
      parsed.pools.push_back(std::move(pool));
    }
  } catch (const Json::exception&) {
    return fail(PlacementConfigStatus::INVALID_SCHEMA,
                "configuration value is outside its bounded type",
                error);
  }
  if (!valid_placement_runtime_config(parsed)) {
    return fail(PlacementConfigStatus::INVALID_SCHEMA,
                "configuration violates V3 safety constraints",
                error);
  }
  *config = std::move(parsed);
  if (error != nullptr) {
    error->clear();
  }
  return PlacementConfigStatus::OK;
}

PlacementConfigStatus load_placement_runtime_config(
    const std::string& path,
    PlacementRuntimeConfig* config,
    std::string* error) {
  if (!valid_placement_identity(path) || config == nullptr) {
    return fail(PlacementConfigStatus::INVALID_INPUT,
                "configuration path is invalid",
                error);
  }
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    return fail(PlacementConfigStatus::IO_ERROR,
                "configuration file cannot be opened",
                error);
  }
  std::ostringstream buffer;
  char chunk[4096];
  size_t total = 0;
  while (input.good()) {
    input.read(chunk, sizeof(chunk));
    const std::streamsize count = input.gcount();
    if (count <= 0) {
      break;
    }
    total += static_cast<size_t>(count);
    if (total > kMaxPlacementConfigBytes) {
      return fail(PlacementConfigStatus::CAPACITY_EXCEEDED,
                  "configuration file exceeds byte limit",
                  error);
    }
    buffer.write(chunk, count);
  }
  if (input.bad()) {
    return fail(PlacementConfigStatus::IO_ERROR,
                "configuration file cannot be read",
                error);
  }
  return parse_placement_runtime_config(buffer.str(), config, error);
}

bool valid_placement_runtime_config(const PlacementRuntimeConfig& config) {
  if (config.loop_interval_ms == 0 ||
      !valid_placement_controller_config(config.controller) ||
      config.executor.max_records == 0 ||
      config.executor.max_message_bytes == 0 ||
      config.executor.max_message_bytes > kMaxPlacementActuatorMessageBytes ||
      config.executor.operation_timeout_ms == 0 ||
      config.executor.terminal_retention_ms == 0 ||
      config.executor.terminal_retention_ms <
          config.controller.reconcile.terminal_visibility_grace_ms ||
      config.executor.max_terminal_compactions_per_cycle == 0 ||
      !valid_placement_observation_collector_config(config.observation) ||
      !valid_placement_input_builder_config(config.input_builder) ||
      config.controller.max_pools != config.input_builder.max_pools ||
      config.pools.empty() ||
      config.pools.size() > config.controller.max_pools ||
      config.transports.native.timeout_ms <= 0 ||
      config.transports.native.max_channels == 0 ||
      config.transports.vllm_ascend.timeout_ms <= 0 ||
      config.transports.vllm_ascend.max_channels == 0 ||
      config.transports.vllm_ascend.max_response_bytes == 0 ||
      config.transports.vllm_ascend.max_response_bytes >
          kMaxPlacementConfigBytes ||
      !valid_internal_token(config.transports.vllm_ascend.internal_api_token) ||
      !valid_placement_identity(config.deployment.address) ||
      config.deployment.timeout_ms <= 0 ||
      config.deployment.max_response_bytes == 0 ||
      config.deployment.max_response_bytes > kMaxPlacementConfigBytes ||
      !valid_internal_token(config.deployment.internal_api_token)) {
    return false;
  }
  std::set<std::string> identities;
  uint64_t protected_devices = 0;
  for (const PlacementPoolRuntimeSpec& pool : config.pools) {
    if (!valid_placement_capacity_profile(pool.profile) ||
        !valid_placement_identity(pool.config_digest) ||
        !std::isfinite(pool.slo_risk_score) || pool.slo_risk_score < 0.0 ||
        !std::isfinite(pool.external.queue_depth) ||
        pool.external.queue_depth < 0.0 ||
        !std::isfinite(pool.external.kv_used_ratio) ||
        pool.external.kv_used_ratio < 0.0 ||
        pool.external.kv_used_ratio > 1.0 ||
        !std::isfinite(pool.external.full_cache_loss_cost) ||
        pool.external.full_cache_loss_cost < 0.0 ||
        !std::isfinite(pool.external.confirmed_store_coverage) ||
        pool.external.confirmed_store_coverage < 0.0 ||
        pool.external.confirmed_store_coverage > 1.0 ||
        pool.profile.load_warmup_p99_ms >
            config.observation.forecast_horizon_ms ||
        !identities.insert(pool_identity(pool.profile.pool)).second) {
      return false;
    }
    const uint64_t protected_replicas =
        static_cast<uint64_t>(pool.profile.min_replicas) +
        pool.profile.failure_headroom_replicas;
    if (protected_replicas > std::numeric_limits<uint64_t>::max() /
                                 pool.profile.devices_per_replica ||
        protected_replicas * pool.profile.devices_per_replica >
            std::numeric_limits<uint64_t>::max() - protected_devices) {
      return false;
    }
    protected_devices += protected_replicas * pool.profile.devices_per_replica;
  }
  return protected_devices <= config.controller.max_devices &&
         config.executor.max_records >=
             config.controller.reconcile.max_operations_per_pool;
}

const char* placement_config_status_name(PlacementConfigStatus status) {
  switch (status) {
    case PlacementConfigStatus::OK:
      return "OK";
    case PlacementConfigStatus::INVALID_INPUT:
      return "INVALID_INPUT";
    case PlacementConfigStatus::INVALID_JSON:
      return "INVALID_JSON";
    case PlacementConfigStatus::INVALID_SCHEMA:
      return "INVALID_SCHEMA";
    case PlacementConfigStatus::CAPACITY_EXCEEDED:
      return "CAPACITY_EXCEEDED";
    case PlacementConfigStatus::IO_ERROR:
      return "IO_ERROR";
  }
  return "UNKNOWN";
}

}  // namespace xllm_service::placement
