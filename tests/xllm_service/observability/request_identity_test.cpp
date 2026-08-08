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

#include "observability/request_identity.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <vector>

#include "common/xllm/uuid.h"

namespace xllm_service::observability {
namespace {

CorrelationIdGenerator sequence_generator(std::vector<std::string> values) {
  return [values = std::move(values), index = size_t{0}]() mutable {
    return values.at(index++);
  };
}

TEST(RequestIdentityTest, PreservesValidUpstreamIdsAndCreatesExecutionUid) {
  RequestCorrelationInput input;
  input.global_request_id = "gateway-request-1";
  input.trace_id = "trace-1";
  auto generator = sequence_generator({"01234567-89ab-7cde-bf01-23456789abcd"});

  auto correlation = make_request_correlation(input, generator);
  EXPECT_EQ(correlation.global_request_id(), "gateway-request-1");
  EXPECT_EQ(correlation.trace_id(), "trace-1");
  EXPECT_EQ(correlation.request_uid(), "01234567-89ab-7cde-bf01-23456789abcd");
  EXPECT_TRUE(correlation.has_attempt_seq());
  EXPECT_EQ(correlation.attempt_seq(), 0);
  EXPECT_EQ(correlation.global_request_id_source(),
            xllm::proto::CORRELATION_ID_SOURCE_UPSTREAM);
  EXPECT_EQ(correlation.trace_id_source(),
            xllm::proto::CORRELATION_ID_SOURCE_UPSTREAM);
}

TEST(RequestIdentityTest, GeneratesEachMissingIdAndMarksItsSource) {
  auto generator = sequence_generator({"generated-global",
                                       "generated-trace",
                                       "01234567-89ab-7cde-bf01-23456789abcd"});
  auto correlation =
      make_request_correlation(RequestCorrelationInput{}, generator);

  EXPECT_EQ(correlation.global_request_id(), "generated-global");
  EXPECT_EQ(correlation.trace_id(), "generated-trace");
  EXPECT_EQ(correlation.request_uid(), "01234567-89ab-7cde-bf01-23456789abcd");
  EXPECT_EQ(correlation.global_request_id_source(),
            xllm::proto::CORRELATION_ID_SOURCE_SERVICE_GENERATED);
  EXPECT_EQ(correlation.trace_id_source(),
            xllm::proto::CORRELATION_ID_SOURCE_SERVICE_GENERATED);
}

TEST(RequestIdentityTest, InvalidUpstreamIdsAreNeverPropagated) {
  RequestCorrelationInput input;
  input.global_request_id = std::string(kMaxCorrelationIdLength + 1, 'g');
  input.trace_id = "contains space";
  auto generator = sequence_generator({"generated-global",
                                       "generated-trace",
                                       "01234567-89ab-7cde-bf01-23456789abcd"});

  auto correlation = make_request_correlation(input, generator);
  EXPECT_EQ(correlation.global_request_id(), "generated-global");
  EXPECT_EQ(correlation.trace_id(), "generated-trace");
}

TEST(RequestIdentityTest, ExtractsValidatedW3cTraceId) {
  RequestCorrelationInput input;
  input.traceparent = "00-4BF92F3577B34DA6A3CE929D0E0E4736-00F067AA0BA902B7-01";
  auto generator = sequence_generator(
      {"generated-global", "01234567-89ab-7cde-bf01-23456789abcd"});

  auto correlation = make_request_correlation(input, generator);
  EXPECT_EQ(correlation.trace_id(), "4bf92f3577b34da6a3ce929d0e0e4736");
  EXPECT_EQ(correlation.trace_id_source(),
            xllm::proto::CORRELATION_ID_SOURCE_UPSTREAM);

  EXPECT_FALSE(trace_id_from_traceparent(
                   "00-00000000000000000000000000000000-00f067aa0ba902b7-01")
                   .has_value());
  EXPECT_FALSE(trace_id_from_traceparent(
                   "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01")
                   .has_value());
  EXPECT_FALSE(trace_id_from_traceparent(
                   "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
                   .has_value());
  EXPECT_FALSE(
      trace_id_from_traceparent(
          "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra")
          .has_value());
  EXPECT_FALSE(
      trace_id_from_traceparent(std::string(kMaxCorrelationIdLength + 1, '0'))
          .has_value());
}

TEST(RequestIdentityTest, InvalidInjectedExecutionUidFallsBackToUuidV7) {
  RequestCorrelationInput input;
  input.global_request_id = "gateway-request-1";
  input.trace_id = "trace-1";
  auto generator = sequence_generator({"not-a-uuid-v7"});

  auto correlation = make_request_correlation(input, generator);

  EXPECT_TRUE(llm::is_uuid_v7(correlation.request_uid()));
}

}  // namespace
}  // namespace xllm_service::observability
