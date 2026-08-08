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

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>

#include "common/options.h"
#include "core/framework/request/first_event_retry_policy.h"

namespace xllm_service {
namespace {

std::optional<xllm::FirstEventRetryPolicy> policy_from_options(
    const Options& options) {
  return xllm::FirstEventRetryPolicy::from_durations_ms(
      static_cast<uint64_t>(options.p_first_event_retry_ub_ms()),
      static_cast<uint64_t>(options.first_event_dispatch_margin_ms()));
}

TEST(RequestTimingPolicyTest, DefaultsFitWithinOutputGapTimeout) {
  const Options options;
  const auto policy = policy_from_options(options);

  ASSERT_TRUE(policy.has_value());
  EXPECT_TRUE(policy->fits_within_gap_timeout_ms(
      static_cast<uint64_t>(options.output_gap_timeout_ms())));
}

TEST(RequestTimingPolicyTest, InvalidAndMisorderedOptionsFailClosed) {
  Options invalid_duration;
  invalid_duration.p_first_event_retry_ub_ms(-1);
  EXPECT_FALSE(policy_from_options(invalid_duration).has_value());

  Options invalid_order;
  invalid_order.p_first_event_retry_ub_ms(900);
  invalid_order.first_event_dispatch_margin_ms(200);
  const auto policy = policy_from_options(invalid_order);
  ASSERT_TRUE(policy.has_value());
  EXPECT_FALSE(policy->fits_within_gap_timeout_ms(
      static_cast<uint64_t>(invalid_order.output_gap_timeout_ms())));
}

}  // namespace
}  // namespace xllm_service
