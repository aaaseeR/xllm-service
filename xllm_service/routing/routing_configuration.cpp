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

#include "routing/routing_configuration.h"

#include <charconv>
#include <limits>
#include <system_error>
#include <utility>

namespace xllm_service {
namespace {

RoutingConfigurationResult make_configuration_error(
    RoutingConfigurationError error,
    const std::string& message) {
  return {error, message};
}

bool valid_backend_endpoint(const std::string& endpoint) {
  const size_t separator = endpoint.find(':');
  if (separator == std::string::npos || separator == 0 ||
      separator + 1 == endpoint.size() ||
      endpoint.find(':', separator + 1) != std::string::npos) {
    return false;
  }
  if (endpoint.find_first_of(" \t\r\n/") != std::string::npos) {
    return false;
  }

  const char* port_begin = endpoint.data() + separator + 1;
  const char* port_end = endpoint.data() + endpoint.size();
  uint32_t port = 0;
  const std::from_chars_result result =
      std::from_chars(port_begin, port_end, port);
  return result.ec == std::errc() && result.ptr == port_end && port > 0 &&
         port <= std::numeric_limits<uint16_t>::max();
}

RoutingSelectionResult make_selection_error(RoutingSelectionError error,
                                            const std::string& message) {
  return {error, message};
}

}  // namespace

const char* routing_mode_name(RoutingMode mode) {
  switch (mode) {
    case RoutingMode::LEGACY:
      return "legacy";
    case RoutingMode::EXTERNAL:
      return "external";
  }
  return "unknown";
}

const char* external_routing_topology_name(ExternalRoutingTopology topology) {
  switch (topology) {
    case ExternalRoutingTopology::AGGREGATED:
      return "aggregated";
    case ExternalRoutingTopology::DISAGGREGATED:
      return "pd";
  }
  return "unknown";
}

const char* routing_configuration_error_name(RoutingConfigurationError error) {
  switch (error) {
    case RoutingConfigurationError::NONE:
      return "none";
    case RoutingConfigurationError::UNKNOWN_MODE:
      return "unknown_mode";
    case RoutingConfigurationError::MISSING_EXTERNAL_BACKEND_ENDPOINT:
      return "missing_external_backend_endpoint";
    case RoutingConfigurationError::EXTERNAL_BACKEND_ENDPOINT_IN_LEGACY_MODE:
      return "external_backend_endpoint_in_legacy_mode";
    case RoutingConfigurationError::INVALID_EXTERNAL_BACKEND_ENDPOINT:
      return "invalid_external_backend_endpoint";
    case RoutingConfigurationError::UNKNOWN_EXTERNAL_ROUTING_TOPOLOGY:
      return "unknown_external_routing_topology";
    case RoutingConfigurationError::EXTERNAL_ROUTING_TOPOLOGY_IN_LEGACY_MODE:
      return "external_routing_topology_in_legacy_mode";
    case RoutingConfigurationError::NULL_OUTPUT:
      return "null_output";
  }
  return "unknown";
}

bool routing_configuration_result_ok(const RoutingConfigurationResult& result) {
  return result.error == RoutingConfigurationError::NONE;
}

RoutingConfigurationResult parse_routing_configuration(
    const std::string& mode,
    const std::string& external_routing_topology,
    const std::string& external_backend_endpoint,
    RoutingConfiguration* configuration) {
  if (configuration == nullptr) {
    return make_configuration_error(RoutingConfigurationError::NULL_OUTPUT,
                                    "Routing configuration output is null");
  }

  RoutingConfiguration parsed;
  if (external_routing_topology == "aggregated") {
    parsed.external_topology = ExternalRoutingTopology::AGGREGATED;
  } else if (external_routing_topology == "pd") {
    parsed.external_topology = ExternalRoutingTopology::DISAGGREGATED;
  } else {
    return make_configuration_error(
        RoutingConfigurationError::UNKNOWN_EXTERNAL_ROUTING_TOPOLOGY,
        "Unsupported external routing topology: " + external_routing_topology);
  }

  if (mode == "legacy") {
    if (!external_backend_endpoint.empty()) {
      return make_configuration_error(
          RoutingConfigurationError::EXTERNAL_BACKEND_ENDPOINT_IN_LEGACY_MODE,
          "external_backend_endpoint is only valid in external routing mode");
    }
    if (parsed.external_topology != ExternalRoutingTopology::AGGREGATED) {
      return make_configuration_error(
          RoutingConfigurationError::EXTERNAL_ROUTING_TOPOLOGY_IN_LEGACY_MODE,
          "external routing topology pd is only valid in external mode");
    }
  } else if (mode == "external") {
    if (external_backend_endpoint.empty()) {
      return make_configuration_error(
          RoutingConfigurationError::MISSING_EXTERNAL_BACKEND_ENDPOINT,
          "external routing mode requires external_backend_endpoint");
    }
    if (!valid_backend_endpoint(external_backend_endpoint)) {
      return make_configuration_error(
          RoutingConfigurationError::INVALID_EXTERNAL_BACKEND_ENDPOINT,
          "external_backend_endpoint must use host:port format");
    }
    parsed.mode = RoutingMode::EXTERNAL;
    parsed.external_backend_endpoint = external_backend_endpoint;
  } else {
    return make_configuration_error(RoutingConfigurationError::UNKNOWN_MODE,
                                    "Unsupported routing mode: " + mode);
  }

  *configuration = std::move(parsed);
  return {};
}

RoutingDecision make_external_aggregated_routing_decision(
    const RoutingConfiguration& configuration) {
  RoutingDecision decision;
  decision.source = RoutingDecisionSource::EXTERNAL;
  decision.prefill_endpoint = configuration.external_backend_endpoint;
  return decision;
}

RoutingDecision make_external_pd_routing_decision(
    const ExternalRoutingDirective& directive) {
  RoutingDecision decision;
  decision.source = RoutingDecisionSource::EXTERNAL;
  decision.prefill_endpoint = directive.prefill_endpoint;
  decision.decode_endpoint = directive.decode_endpoint;
  decision.attempt = directive.attempt;
  return decision;
}

bool routing_selection_result_ok(const RoutingSelectionResult& result) {
  return result.error == RoutingSelectionError::NONE;
}

RoutingSelectionResult resolve_routing_decision(
    const RoutingConfiguration& configuration,
    const ExternalRoutingDirective& external_directive,
    const LegacyRoutingSelector& legacy_selector,
    RoutingDecision* decision) {
  if (decision == nullptr) {
    return make_selection_error(RoutingSelectionError::NULL_OUTPUT,
                                "Routing decision output is null");
  }

  switch (configuration.mode) {
    case RoutingMode::LEGACY:
      if (external_directive.present) {
        return make_selection_error(
            RoutingSelectionError::EXTERNAL_DIRECTIVE_IN_LEGACY_MODE,
            "External routing directive is not accepted in legacy mode");
      }
      if (!legacy_selector) {
        return make_selection_error(
            RoutingSelectionError::LEGACY_SELECTOR_UNAVAILABLE,
            "Legacy routing selector is unavailable");
      }
      *decision = RoutingDecision{};
      if (!legacy_selector()) {
        return make_selection_error(
            RoutingSelectionError::LEGACY_SELECTION_FAILED,
            "Legacy routing selector failed");
      }
      return {};
    case RoutingMode::EXTERNAL:
      if (configuration.external_topology ==
          ExternalRoutingTopology::AGGREGATED) {
        if (external_directive.present) {
          return make_selection_error(
              RoutingSelectionError::EXTERNAL_DIRECTIVE_IN_AGGREGATED_MODE,
              "External P/D directive is not accepted in aggregated mode");
        }
        *decision = make_external_aggregated_routing_decision(configuration);
        return {};
      }
      if (!external_directive.present) {
        return make_selection_error(
            RoutingSelectionError::MISSING_EXTERNAL_DIRECTIVE,
            "External P/D routing requires a routing directive");
      }
      if (!external_directive.valid) {
        return make_selection_error(
            RoutingSelectionError::INVALID_EXTERNAL_DIRECTIVE,
            external_directive.error.empty()
                ? "External P/D routing directive is invalid"
                : external_directive.error);
      }
      if (external_directive.version != kRoutingDecisionVersion) {
        return make_selection_error(
            RoutingSelectionError::UNSUPPORTED_EXTERNAL_DIRECTIVE_VERSION,
            "Unsupported external routing directive version: " +
                std::to_string(external_directive.version));
      }
      if (!valid_backend_endpoint(external_directive.prefill_endpoint) ||
          !valid_backend_endpoint(external_directive.decode_endpoint) ||
          external_directive.prefill_endpoint ==
              external_directive.decode_endpoint) {
        return make_selection_error(
            RoutingSelectionError::INVALID_EXTERNAL_DIRECTIVE,
            "External P/D routing requires distinct host:port prefill and "
            "decode endpoints");
      }
      if (external_directive.prefill_endpoint !=
          configuration.external_backend_endpoint) {
        return make_selection_error(
            RoutingSelectionError::EXTERNAL_OWNER_MISMATCH,
            "External P/D prefill endpoint does not match this adapter "
            "owner");
      }
      *decision = make_external_pd_routing_decision(external_directive);
      return {};
  }
  return make_selection_error(RoutingSelectionError::UNKNOWN_MODE,
                              "Unknown routing mode");
}

}  // namespace xllm_service
