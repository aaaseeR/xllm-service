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

namespace xllm_service {

inline constexpr uint32_t kRoutingDecisionVersion = 1;

enum class RoutingDecisionSource : uint8_t {
  LEGACY = 0,
  EXTERNAL = 1,
};

const char* routing_decision_source_name(RoutingDecisionSource source);

struct RoutingDecision {
  uint32_t version = kRoutingDecisionVersion;
  RoutingDecisionSource source = RoutingDecisionSource::LEGACY;
  std::string prefill_endpoint;
  std::string decode_endpoint;
  std::string prefill_incarnation;
  std::string decode_incarnation;
  uint32_t attempt = 0;
};

bool routing_decision_is_disaggregated(const RoutingDecision& decision);
std::string routing_decision_debug_string(const RoutingDecision& decision);

enum class RoutingDecisionError : uint8_t {
  NONE = 0,
  UNSUPPORTED_VERSION = 1,
  UNKNOWN_SOURCE = 2,
  MISSING_PREFILL_ENDPOINT = 3,
  MISSING_PREFILL_INCARNATION = 4,
  INCOMPLETE_DECODE_IDENTITY = 5,
};

const char* routing_decision_error_name(RoutingDecisionError error);

struct RoutingDecisionValidationResult {
  RoutingDecisionError error = RoutingDecisionError::NONE;
  std::string message;
};

bool routing_decision_validation_ok(
    const RoutingDecisionValidationResult& result);

RoutingDecisionValidationResult validate_routing_decision(
    const RoutingDecision& decision);

}  // namespace xllm_service
