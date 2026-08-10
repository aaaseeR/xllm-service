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

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "placement/placement_actuator.h"
#include "placement/placement_controller.h"
#include "placement/placement_deployment_actuator.h"
#include "placement/placement_input_builder.h"
#include "placement/placement_observation_collector.h"
#include "placement/provider_lifecycle_actuator.h"

namespace xllm_service::placement {

inline constexpr uint32_t kPlacementConfigSchemaVersion = 3;
inline constexpr size_t kMaxPlacementConfigBytes = 1024 * 1024;

enum class PlacementConfigStatus : int8_t {
  OK = 0,
  INVALID_INPUT = 1,
  INVALID_JSON = 2,
  INVALID_SCHEMA = 3,
  CAPACITY_EXCEEDED = 4,
  IO_ERROR = 5,
};

struct PlacementTransportConfig {
  BrpcProviderLifecycleTransportConfig native;
  HttpProviderLifecycleTransportConfig vllm_ascend;
};

struct PlacementRuntimeConfig {
  uint64_t loop_interval_ms = 0;
  PlacementControllerConfig controller;
  PlacementOperationExecutorConfig executor;
  PlacementObservationCollectorConfig observation;
  PlacementInputBuilderConfig input_builder;
  PlacementTransportConfig transports;
  HttpPlacementDeploymentActuatorConfig deployment;
  std::vector<PlacementPoolRuntimeSpec> pools;
};

PlacementConfigStatus parse_placement_runtime_config(
    const std::string& value,
    PlacementRuntimeConfig* config,
    std::string* error = nullptr);

PlacementConfigStatus load_placement_runtime_config(
    const std::string& path,
    PlacementRuntimeConfig* config,
    std::string* error = nullptr);

bool valid_placement_runtime_config(const PlacementRuntimeConfig& config);

const char* placement_config_status_name(PlacementConfigStatus status);

}  // namespace xllm_service::placement
