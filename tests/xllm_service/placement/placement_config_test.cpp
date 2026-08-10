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

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <string>

namespace xllm_service::placement {
namespace {

nlohmann::json valid_json() {
  return nlohmann::json{
      {"schema_version", 3},
      {"loop_interval_ms", 1000},
      {"controller",
       {{"mode", "SHADOW"},
        {"max_pools", 4},
        {"max_desired_snapshot_bytes", 65536},
        {"max_operation_snapshot_bytes", 65536},
        {"max_devices", 16},
        {"max_new_operations_per_cycle", 4},
        {"max_actuator_actions_per_cycle", 4},
        {"planner",
         {{"scale_up_hold_ms", 1000},
          {"scale_down_stabilization_ms", 60000},
          {"cooldown_ms", 30000},
          {"economic_horizon_ms", 3600000},
          {"min_scale_down_samples", 100},
          {"max_scale_up_step", 2},
          {"max_scale_down_step", 1},
          {"queue_high_watermark", 10.0},
          {"queue_low_watermark", 1.0},
          {"admission_reject_high_watermark", 0.1},
          {"admission_reject_low_watermark", 0.01},
          {"kv_high_watermark", 0.9},
          {"kv_low_watermark", 0.5}}},
        {"reconcile",
         {{"max_operations_per_cycle", 4},
          {"max_operations_per_pool", 8},
          {"max_create_per_cycle", 2},
          {"max_drain_per_cycle", 1}}}}},
      {"executor",
       {{"max_records", 64},
        {"max_message_bytes", 512},
        {"operation_timeout_ms", 300000}}},
      {"observation",
       {{"max_models", 4},
        {"bucket_count", 60},
        {"bucket_width_ms", 1000},
        {"max_latency_samples_per_bucket", 256},
        {"forecast_horizon_ms", 120000},
        {"forecast_headroom", 1.2}}},
      {"input_builder", {{"max_pools", 4}, {"max_members", 4096}}},
      {"transports",
       {{"native_timeout_ms", 2000},
        {"native_max_channels", 1024},
        {"vllm_timeout_ms", 2000},
        {"vllm_max_channels", 1024},
        {"vllm_max_response_bytes", 65536},
        {"vllm_internal_api_token", "test-token"}}},
      {"pools",
       {{{"provider", "XLLM_NATIVE"},
         {"model_revision", "model-r1"},
         {"role", "PREFILL"},
         {"profile_digest", "profile-a"},
         {"devices_per_replica", 1},
         {"instance_cost_per_hour", 1.0},
         {"load_warmup_p99_ms", 60000},
         {"prefill_tokens_per_second_under_slo", 1000.0},
         {"decode_tokens_per_second_under_slo", 0.0},
         {"requests_per_second_under_slo", 0.0},
         {"target_utilization", 0.8},
         {"min_replicas", 1},
         {"max_replicas", 8},
         {"failure_headroom_replicas", 1},
         {"ttft_slo_ms", 500.0},
         {"tpot_slo_ms", 50.0},
         {"priority", 10},
         {"slo_risk_score", 0.5},
         {"config_digest", "config-a"},
         {"external",
          {{"queue_depth", 0.0},
           {"kv_used_ratio", 0.0},
           {"full_cache_loss_cost", 0.0},
           {"confirmed_store_coverage", 0.0},
           {"out_of_distribution", false}}}}}}};
}

TEST(PlacementConfigTest, ParsesCompleteStrictV3Configuration) {
  PlacementRuntimeConfig config;
  std::string error;
  ASSERT_EQ(
      parse_placement_runtime_config(valid_json().dump(), &config, &error),
      PlacementConfigStatus::OK)
      << error;
  EXPECT_EQ(config.controller.mode, PlacementMode::SHADOW);
  ASSERT_EQ(config.pools.size(), 1u);
  EXPECT_EQ(config.pools[0].profile.pool.provider_id,
            xllm::proto::PROVIDER_ID_XLLM_NATIVE);
  EXPECT_EQ(config.pools[0].profile.pool.role,
            xllm::proto::ENGINE_ROLE_PREFILL);
  EXPECT_EQ(config.observation.forecast_horizon_ms, 120000u);
}

TEST(PlacementConfigTest, RejectsUnknownAndMissingFields) {
  PlacementRuntimeConfig config;
  nlohmann::json json = valid_json();
  json["surprise"] = true;
  EXPECT_EQ(parse_placement_runtime_config(json.dump(), &config),
            PlacementConfigStatus::INVALID_SCHEMA);
  json = valid_json();
  json["controller"].erase("planner");
  EXPECT_EQ(parse_placement_runtime_config(json.dump(), &config),
            PlacementConfigStatus::INVALID_SCHEMA);
}

TEST(PlacementConfigTest, RejectsUnknownEnumsAndDuplicatePools) {
  PlacementRuntimeConfig config;
  nlohmann::json json = valid_json();
  json["controller"]["mode"] = "AUTOMATIC";
  EXPECT_EQ(parse_placement_runtime_config(json.dump(), &config),
            PlacementConfigStatus::INVALID_SCHEMA);
  json = valid_json();
  json["pools"].push_back(json["pools"][0]);
  EXPECT_EQ(parse_placement_runtime_config(json.dump(), &config),
            PlacementConfigStatus::INVALID_SCHEMA);
}

TEST(PlacementConfigTest, RejectsUnsafeForecastAndBudget) {
  PlacementRuntimeConfig config;
  nlohmann::json json = valid_json();
  json["observation"]["forecast_horizon_ms"] = 1000;
  EXPECT_EQ(parse_placement_runtime_config(json.dump(), &config),
            PlacementConfigStatus::INVALID_SCHEMA);
  json = valid_json();
  json["controller"]["max_devices"] = 1;
  EXPECT_EQ(parse_placement_runtime_config(json.dump(), &config),
            PlacementConfigStatus::INVALID_SCHEMA);
}

TEST(PlacementConfigTest, RejectsNumericCoercionAndInvalidRanges) {
  PlacementRuntimeConfig config;
  nlohmann::json json = valid_json();
  json["controller"]["max_pools"] = 4.0;
  EXPECT_EQ(parse_placement_runtime_config(json.dump(), &config),
            PlacementConfigStatus::INVALID_SCHEMA);
  json = valid_json();
  json["pools"][0]["external"]["kv_used_ratio"] = 1.1;
  EXPECT_EQ(parse_placement_runtime_config(json.dump(), &config),
            PlacementConfigStatus::INVALID_SCHEMA);
}

TEST(PlacementConfigTest, RejectsMalformedAndOversizedInput) {
  PlacementRuntimeConfig config;
  EXPECT_EQ(parse_placement_runtime_config("{", &config),
            PlacementConfigStatus::INVALID_JSON);
  EXPECT_EQ(parse_placement_runtime_config(
                std::string(kMaxPlacementConfigBytes + 1, 'x'), &config),
            PlacementConfigStatus::CAPACITY_EXCEEDED);
}

}  // namespace
}  // namespace xllm_service::placement
