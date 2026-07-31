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

enum class ExternalRoutingTopology : uint8_t {
  AGGREGATED = 0,
  DISAGGREGATED = 1,
};

const char* external_routing_topology_name(ExternalRoutingTopology topology);

struct ExternalRoutingDirective {
  bool present = false;
  bool valid = true;
  uint32_t version = 0;
  std::string prefill_endpoint;
  std::string decode_endpoint;
  uint32_t attempt = 0;
  std::string error;
};

struct RoutingConfiguration {
  RoutingMode mode = RoutingMode::LEGACY;
  ExternalRoutingTopology external_topology =
      ExternalRoutingTopology::AGGREGATED;
  std::string external_backend_endpoint;
};

enum class RoutingConfigurationError : uint8_t {
  NONE = 0,
  UNKNOWN_MODE = 1,
  MISSING_EXTERNAL_BACKEND_ENDPOINT = 2,
  EXTERNAL_BACKEND_ENDPOINT_IN_LEGACY_MODE = 3,
  INVALID_EXTERNAL_BACKEND_ENDPOINT = 4,
  UNKNOWN_EXTERNAL_ROUTING_TOPOLOGY = 5,
  EXTERNAL_ROUTING_TOPOLOGY_IN_LEGACY_MODE = 6,
  NULL_OUTPUT = 7,
};

const char* routing_configuration_error_name(RoutingConfigurationError error);

struct RoutingConfigurationResult {
  RoutingConfigurationError error = RoutingConfigurationError::NONE;
  std::string message;
};

bool routing_configuration_result_ok(const RoutingConfigurationResult& result);

RoutingConfigurationResult parse_routing_configuration(
    const std::string& mode,
    const std::string& external_routing_topology,
    const std::string& external_backend_endpoint,
    RoutingConfiguration* configuration);

RoutingDecision make_external_aggregated_routing_decision(
    const RoutingConfiguration& configuration);
RoutingDecision make_external_pd_routing_decision(
    const ExternalRoutingDirective& directive);

enum class RoutingSelectionError : uint8_t {
  NONE = 0,
  LEGACY_SELECTOR_UNAVAILABLE = 1,
  LEGACY_SELECTION_FAILED = 2,
  UNKNOWN_MODE = 3,
  EXTERNAL_DIRECTIVE_IN_LEGACY_MODE = 4,
  EXTERNAL_DIRECTIVE_IN_AGGREGATED_MODE = 5,
  MISSING_EXTERNAL_DIRECTIVE = 6,
  INVALID_EXTERNAL_DIRECTIVE = 7,
  UNSUPPORTED_EXTERNAL_DIRECTIVE_VERSION = 8,
  EXTERNAL_OWNER_MISMATCH = 9,
  NULL_OUTPUT = 10,
};

struct RoutingSelectionResult {
  RoutingSelectionError error = RoutingSelectionError::NONE;
  std::string message;
};

using LegacyRoutingSelector = std::function<bool()>;

bool routing_selection_result_ok(const RoutingSelectionResult& result);
RoutingSelectionResult resolve_routing_decision(
    const RoutingConfiguration& configuration,
    const ExternalRoutingDirective& external_directive,
    const LegacyRoutingSelector& legacy_selector,
    RoutingDecision* decision);

}  // namespace xllm_service
