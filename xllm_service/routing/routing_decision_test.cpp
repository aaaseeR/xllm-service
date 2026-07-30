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

#include <gtest/gtest.h>

namespace xllm_service {
namespace {

RoutingDecision make_valid_decision() {
  RoutingDecision decision;
  decision.prefill_endpoint = "prefill:8000";
  decision.prefill_incarnation = "prefill-v1";
  return decision;
}

TEST(RoutingDecisionTest, AcceptsAggregatedAndDisaggregatedDecisions) {
  RoutingDecision aggregated = make_valid_decision();
  EXPECT_TRUE(routing_decision_validation_ok(
      validate_routing_decision(aggregated)));
  EXPECT_FALSE(routing_decision_is_disaggregated(aggregated));

  RoutingDecision disaggregated = aggregated;
  disaggregated.decode_endpoint = "decode:8000";
  disaggregated.decode_incarnation = "decode-v1";
  EXPECT_TRUE(routing_decision_validation_ok(
      validate_routing_decision(disaggregated)));
  EXPECT_TRUE(routing_decision_is_disaggregated(disaggregated));
}

TEST(RoutingDecisionTest, RejectsUnsupportedVersionAndUnknownSource) {
  RoutingDecision decision = make_valid_decision();
  decision.version = kRoutingDecisionVersion + 1;
  EXPECT_EQ(validate_routing_decision(decision).error,
            RoutingDecisionError::UNSUPPORTED_VERSION);

  decision = make_valid_decision();
  decision.source = static_cast<RoutingDecisionSource>(255);
  EXPECT_EQ(validate_routing_decision(decision).error,
            RoutingDecisionError::UNKNOWN_SOURCE);
}

TEST(RoutingDecisionTest, RequiresPrefillIdentity) {
  RoutingDecision decision;
  EXPECT_EQ(validate_routing_decision(decision).error,
            RoutingDecisionError::MISSING_PREFILL_ENDPOINT);

  decision.prefill_endpoint = "prefill:8000";
  EXPECT_EQ(validate_routing_decision(decision).error,
            RoutingDecisionError::MISSING_PREFILL_INCARNATION);
}

TEST(RoutingDecisionTest, RequiresCompleteDecodeIdentity) {
  RoutingDecision decision = make_valid_decision();
  decision.decode_endpoint = "decode:8000";
  EXPECT_EQ(validate_routing_decision(decision).error,
            RoutingDecisionError::INCOMPLETE_DECODE_IDENTITY);

  decision = make_valid_decision();
  decision.decode_incarnation = "decode-v1";
  EXPECT_EQ(validate_routing_decision(decision).error,
            RoutingDecisionError::INCOMPLETE_DECODE_IDENTITY);
}

TEST(RoutingDecisionTest, DebugStringIncludesContractMetadata) {
  RoutingDecision decision = make_valid_decision();
  decision.source = RoutingDecisionSource::EXTERNAL;
  decision.attempt = 2;
  const std::string output = routing_decision_debug_string(decision);

  EXPECT_NE(output.find("\"version\": 1"), std::string::npos);
  EXPECT_NE(output.find("\"source\": \"external\""),
            std::string::npos);
  EXPECT_NE(output.find("\"attempt\": 2"), std::string::npos);
}

}  // namespace
}  // namespace xllm_service
