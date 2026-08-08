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

#include "request/first_output_retry_budget.h"

#include <gtest/gtest.h>

#include <chrono>

namespace xllm_service {
namespace {

using Decision = FirstOutputRetryDecision;
using TimePoint = FirstOutputRetryBudget::TimePoint;

TEST(FirstOutputRetryBudgetTest, AllowsAndAccountsBoundedRetry) {
  const TimePoint start(std::chrono::milliseconds(100));
  FirstOutputRetryBudget budget({.max_attempt_retries = 2,
                                 .max_wasted_device_ms = 500,
                                 .min_remaining_deadline_ms = 100},
                                start);

  EXPECT_EQ(budget.evaluate(false, 100, start + std::chrono::milliseconds(200)),
            Decision::ALLOWED);
  EXPECT_TRUE(budget.commit_retry(start + std::chrono::milliseconds(200)));
  EXPECT_EQ(budget.retries(), 1);
  EXPECT_EQ(budget.wasted_device_ms(), 200);
}

TEST(FirstOutputRetryBudgetTest, FirstOutputPermanentlyClosesRetryWindow) {
  FirstOutputRetryBudget budget({});
  EXPECT_EQ(budget.evaluate(true, 10000), Decision::FIRST_OUTPUT_EMITTED);
}

TEST(FirstOutputRetryBudgetTest, AttemptBudgetIsHardBound) {
  const TimePoint start(std::chrono::milliseconds(100));
  FirstOutputRetryBudget budget({.max_attempt_retries = 1,
                                 .max_wasted_device_ms = 500,
                                 .min_remaining_deadline_ms = 100},
                                start);
  ASSERT_TRUE(budget.commit_retry(start + std::chrono::milliseconds(100)));
  EXPECT_EQ(
      budget.evaluate(false, 10000, start + std::chrono::milliseconds(101)),
      Decision::ATTEMPT_BUDGET_EXHAUSTED);
}

TEST(FirstOutputRetryBudgetTest, DeadlineThresholdIsInclusive) {
  FirstOutputRetryBudget budget({.max_attempt_retries = 1,
                                 .max_wasted_device_ms = 500,
                                 .min_remaining_deadline_ms = 100});
  EXPECT_EQ(budget.evaluate(false, 99), Decision::DEADLINE_BUDGET_INSUFFICIENT);
  EXPECT_EQ(budget.evaluate(false, 100), Decision::ALLOWED);
}

TEST(FirstOutputRetryBudgetTest, CumulativeDeviceTimeIsHardBound) {
  const TimePoint start(std::chrono::milliseconds(100));
  FirstOutputRetryBudget budget({.max_attempt_retries = 2,
                                 .max_wasted_device_ms = 300,
                                 .min_remaining_deadline_ms = 100},
                                start);
  ASSERT_TRUE(budget.commit_retry(start + std::chrono::milliseconds(200)));
  EXPECT_EQ(
      budget.evaluate(false, 10000, start + std::chrono::milliseconds(301)),
      Decision::DEVICE_TIME_BUDGET_EXHAUSTED);
  EXPECT_FALSE(budget.commit_retry(start + std::chrono::milliseconds(301)));
  EXPECT_EQ(budget.retries(), 1);
  EXPECT_EQ(budget.wasted_device_ms(), 200);
}

TEST(FirstOutputRetryBudgetTest, RejectsClockRegressionWithoutAccounting) {
  const TimePoint start(std::chrono::milliseconds(100));
  FirstOutputRetryBudget budget({.max_attempt_retries = 1,
                                 .max_wasted_device_ms = 500,
                                 .min_remaining_deadline_ms = 100},
                                start);
  EXPECT_EQ(budget.evaluate(false, 10000, start - std::chrono::milliseconds(1)),
            Decision::CLOCK_REGRESSION);
  EXPECT_FALSE(budget.commit_retry(start - std::chrono::milliseconds(1)));
  EXPECT_EQ(budget.retries(), 0);
  EXPECT_EQ(budget.wasted_device_ms(), 0);
}

TEST(FirstOutputRetryBudgetTest, ZeroAttemptBudgetDisablesRetry) {
  FirstOutputRetryBudget budget({.max_attempt_retries = 0,
                                 .max_wasted_device_ms = 0,
                                 .min_remaining_deadline_ms = 0});
  EXPECT_EQ(budget.evaluate(false, 10000), Decision::ATTEMPT_BUDGET_EXHAUSTED);
}

TEST(FirstOutputRetryBudgetTest, RejectsIncompleteEnabledConfiguration) {
  EXPECT_DEATH(FirstOutputRetryBudget({.max_attempt_retries = 1,
                                       .max_wasted_device_ms = 0,
                                       .min_remaining_deadline_ms = 100}),
               "Invalid first-output retry budget");
  EXPECT_DEATH(FirstOutputRetryBudget({.max_attempt_retries = 1,
                                       .max_wasted_device_ms = 100,
                                       .min_remaining_deadline_ms = 0}),
               "Invalid first-output retry budget");
}

}  // namespace
}  // namespace xllm_service
