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

#include <glog/logging.h>

#include <limits>

namespace xllm_service {

FirstOutputRetryBudget::FirstOutputRetryBudget(Config config, TimePoint now)
    : config_(config), attempt_started_at_(now) {
  if (config_.max_attempt_retries > 0 &&
      (config_.max_wasted_device_ms == 0 ||
       config_.min_remaining_deadline_ms == 0)) {
    LOG(FATAL) << "Invalid first-output retry budget.";
  }
}

uint64_t FirstOutputRetryBudget::current_attempt_ms(TimePoint now) const {
  if (now < attempt_started_at_) {
    return std::numeric_limits<uint64_t>::max();
  }
  const int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                              now - attempt_started_at_)
                              .count();
  return static_cast<uint64_t>(elapsed);
}

FirstOutputRetryDecision FirstOutputRetryBudget::evaluate(
    bool first_output_emitted,
    uint64_t remaining_deadline_ms,
    TimePoint now) const {
  if (first_output_emitted) {
    return FirstOutputRetryDecision::FIRST_OUTPUT_EMITTED;
  }
  if (retries_ >= config_.max_attempt_retries) {
    return FirstOutputRetryDecision::ATTEMPT_BUDGET_EXHAUSTED;
  }
  if (remaining_deadline_ms < config_.min_remaining_deadline_ms) {
    return FirstOutputRetryDecision::DEADLINE_BUDGET_INSUFFICIENT;
  }
  const uint64_t attempt_ms = current_attempt_ms(now);
  if (attempt_ms == std::numeric_limits<uint64_t>::max()) {
    return FirstOutputRetryDecision::CLOCK_REGRESSION;
  }
  if (wasted_device_ms_ > config_.max_wasted_device_ms ||
      attempt_ms > config_.max_wasted_device_ms - wasted_device_ms_) {
    return FirstOutputRetryDecision::DEVICE_TIME_BUDGET_EXHAUSTED;
  }
  return FirstOutputRetryDecision::ALLOWED;
}

bool FirstOutputRetryBudget::commit_retry(TimePoint now) {
  const uint64_t attempt_ms = current_attempt_ms(now);
  if (attempt_ms == std::numeric_limits<uint64_t>::max() ||
      retries_ >= config_.max_attempt_retries ||
      wasted_device_ms_ > config_.max_wasted_device_ms ||
      attempt_ms > config_.max_wasted_device_ms - wasted_device_ms_) {
    return false;
  }
  wasted_device_ms_ += attempt_ms;
  ++retries_;
  attempt_started_at_ = now;
  return true;
}

}  // namespace xllm_service
