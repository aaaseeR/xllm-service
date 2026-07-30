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

#include <gtest/gtest.h>

namespace xllm_service {
namespace {

TEST(RoutingConfigurationTest, ParsesLegacyModeWithoutExternalEndpoint) {
  RoutingConfiguration configuration;
  const RoutingConfigurationResult result =
      parse_routing_configuration("legacy", "", &configuration);

  EXPECT_TRUE(routing_configuration_result_ok(result));
  EXPECT_EQ(configuration.mode, RoutingMode::LEGACY);
  EXPECT_TRUE(configuration.external_backend_endpoint.empty());
}

TEST(RoutingConfigurationTest, ParsesExternalAggregatedEndpoint) {
  RoutingConfiguration configuration;
  const RoutingConfigurationResult result = parse_routing_configuration(
      "external", "127.0.0.1:8000", &configuration);

  ASSERT_TRUE(routing_configuration_result_ok(result));
  EXPECT_EQ(configuration.mode, RoutingMode::EXTERNAL);
  EXPECT_EQ(configuration.external_backend_endpoint, "127.0.0.1:8000");

  const RoutingDecision decision =
      make_external_aggregated_routing_decision(configuration);
  EXPECT_EQ(decision.source, RoutingDecisionSource::EXTERNAL);
  EXPECT_EQ(decision.prefill_endpoint, "127.0.0.1:8000");
  EXPECT_TRUE(decision.decode_endpoint.empty());
  EXPECT_TRUE(decision.prefill_incarnation.empty());
}

TEST(RoutingConfigurationTest, RejectsAmbiguousOrIncompleteConfiguration) {
  RoutingConfiguration configuration;
  EXPECT_EQ(parse_routing_configuration(
                "legacy", "127.0.0.1:8000", &configuration)
                .error,
            RoutingConfigurationError::
                EXTERNAL_BACKEND_ENDPOINT_IN_LEGACY_MODE);
  EXPECT_EQ(parse_routing_configuration("external", "", &configuration).error,
            RoutingConfigurationError::MISSING_EXTERNAL_BACKEND_ENDPOINT);
  EXPECT_EQ(parse_routing_configuration(
                "external", "https://backend:8000", &configuration)
                .error,
            RoutingConfigurationError::INVALID_EXTERNAL_BACKEND_ENDPOINT);
  EXPECT_EQ(parse_routing_configuration(
                "external", "backend:not-a-port", &configuration)
                .error,
            RoutingConfigurationError::INVALID_EXTERNAL_BACKEND_ENDPOINT);
  EXPECT_EQ(parse_routing_configuration(
                "external", "backend:70000", &configuration)
                .error,
            RoutingConfigurationError::INVALID_EXTERNAL_BACKEND_ENDPOINT);
  EXPECT_EQ(parse_routing_configuration("shadow", "", &configuration).error,
            RoutingConfigurationError::UNKNOWN_MODE);
  EXPECT_EQ(parse_routing_configuration("legacy", "", nullptr).error,
            RoutingConfigurationError::NULL_OUTPUT);
}

TEST(RoutingConfigurationTest, ExternalAuthorityDoesNotInvokeLegacySelector) {
  RoutingConfiguration configuration;
  configuration.mode = RoutingMode::EXTERNAL;
  configuration.external_backend_endpoint = "backend:8000";
  bool legacy_selector_called = false;
  RoutingDecision decision;

  const RoutingSelectionResult result = resolve_routing_decision(
      configuration,
      [&legacy_selector_called]() {
        legacy_selector_called = true;
        return true;
      },
      &decision);

  EXPECT_TRUE(routing_selection_result_ok(result));
  EXPECT_FALSE(legacy_selector_called);
  EXPECT_EQ(decision.source, RoutingDecisionSource::EXTERNAL);
  EXPECT_EQ(decision.prefill_endpoint, "backend:8000");
}

TEST(RoutingConfigurationTest, LegacyAuthorityUsesOnlyLegacySelector) {
  RoutingConfiguration configuration;
  RoutingDecision decision;
  const RoutingSelectionResult result = resolve_routing_decision(
      configuration,
      [&decision]() {
        decision.prefill_endpoint = "legacy:8000";
        return true;
      },
      &decision);

  EXPECT_TRUE(routing_selection_result_ok(result));
  EXPECT_EQ(decision.source, RoutingDecisionSource::LEGACY);
  EXPECT_EQ(decision.prefill_endpoint, "legacy:8000");
}

}  // namespace
}  // namespace xllm_service
