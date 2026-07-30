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
#include <functional>
#include <string>

#include "routing/routing_decision.h"

namespace xllm_service {

enum class RoutingMode : uint8_t {
  LEGACY = 0,
  EXTERNAL = 1,
};

const char* routing_mode_name(RoutingMode mode);

struct RoutingConfiguration {
  RoutingMode mode = RoutingMode::LEGACY;
  std::string external_backend_endpoint;
};

enum class RoutingConfigurationError : uint8_t {
  NONE = 0,
  UNKNOWN_MODE = 1,
  MISSING_EXTERNAL_BACKEND_ENDPOINT = 2,
  EXTERNAL_BACKEND_ENDPOINT_IN_LEGACY_MODE = 3,
  INVALID_EXTERNAL_BACKEND_ENDPOINT = 4,
  NULL_OUTPUT = 5,
};

const char* routing_configuration_error_name(
    RoutingConfigurationError error);

struct RoutingConfigurationResult {
  RoutingConfigurationError error = RoutingConfigurationError::NONE;
  std::string message;
};

bool routing_configuration_result_ok(
    const RoutingConfigurationResult& result);

RoutingConfigurationResult parse_routing_configuration(
    const std::string& mode,
    const std::string& external_backend_endpoint,
    RoutingConfiguration* configuration);

RoutingDecision make_external_aggregated_routing_decision(
    const RoutingConfiguration& configuration);

enum class RoutingSelectionError : uint8_t {
  NONE = 0,
  LEGACY_SELECTOR_UNAVAILABLE = 1,
  LEGACY_SELECTION_FAILED = 2,
  UNKNOWN_MODE = 3,
  NULL_OUTPUT = 4,
};

struct RoutingSelectionResult {
  RoutingSelectionError error = RoutingSelectionError::NONE;
  std::string message;
};

using LegacyRoutingSelector = std::function<bool()>;

bool routing_selection_result_ok(const RoutingSelectionResult& result);
RoutingSelectionResult resolve_routing_decision(
    const RoutingConfiguration& configuration,
    const LegacyRoutingSelector& legacy_selector,
    RoutingDecision* decision);

}  // namespace xllm_service
