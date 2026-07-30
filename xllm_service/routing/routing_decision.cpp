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

#include "routing/routing_decision.h"

#include <nlohmann/json.hpp>

namespace xllm_service {
namespace {

RoutingDecisionValidationResult make_validation_error(
    RoutingDecisionError error,
    const std::string& message) {
  return {error, message};
}

}  // namespace

const char* routing_decision_source_name(RoutingDecisionSource source) {
  switch (source) {
    case RoutingDecisionSource::LEGACY:
      return "legacy";
    case RoutingDecisionSource::EXTERNAL:
      return "external";
  }
  return "unknown";
}

bool RoutingDecision::is_disaggregated() const {
  return !decode_endpoint.empty();
}

std::string RoutingDecision::debug_string() const {
  return nlohmann::json{{"version", version},
                        {"source", routing_decision_source_name(source)},
                        {"prefill_endpoint", prefill_endpoint},
                        {"decode_endpoint", decode_endpoint},
                        {"prefill_incarnation", prefill_incarnation},
                        {"decode_incarnation", decode_incarnation},
                        {"attempt", attempt}}
      .dump(2);
}

const char* routing_decision_error_name(RoutingDecisionError error) {
  switch (error) {
    case RoutingDecisionError::NONE:
      return "none";
    case RoutingDecisionError::UNSUPPORTED_VERSION:
      return "unsupported_version";
    case RoutingDecisionError::UNKNOWN_SOURCE:
      return "unknown_source";
    case RoutingDecisionError::MISSING_PREFILL_ENDPOINT:
      return "missing_prefill_endpoint";
    case RoutingDecisionError::MISSING_PREFILL_INCARNATION:
      return "missing_prefill_incarnation";
    case RoutingDecisionError::INCOMPLETE_DECODE_IDENTITY:
      return "incomplete_decode_identity";
  }
  return "unknown";
}

RoutingDecisionValidationResult validate_routing_decision(
    const RoutingDecision& decision) {
  if (decision.version != kRoutingDecisionVersion) {
    return make_validation_error(
        RoutingDecisionError::UNSUPPORTED_VERSION,
        "Unsupported routing decision version: " +
            std::to_string(decision.version));
  }
  switch (decision.source) {
    case RoutingDecisionSource::LEGACY:
    case RoutingDecisionSource::EXTERNAL:
      break;
    default:
      return make_validation_error(RoutingDecisionError::UNKNOWN_SOURCE,
                                   "Unknown routing decision source");
  }
  if (decision.prefill_endpoint.empty()) {
    return make_validation_error(
        RoutingDecisionError::MISSING_PREFILL_ENDPOINT,
        "Routing decision has no prefill endpoint");
  }
  if (decision.prefill_incarnation.empty()) {
    return make_validation_error(
        RoutingDecisionError::MISSING_PREFILL_INCARNATION,
        "Routing decision has no prefill incarnation");
  }
  if (decision.decode_endpoint.empty() !=
      decision.decode_incarnation.empty()) {
    return make_validation_error(
        RoutingDecisionError::INCOMPLETE_DECODE_IDENTITY,
        "Decode endpoint and incarnation must be set together");
  }
  return {};
}

}  // namespace xllm_service
