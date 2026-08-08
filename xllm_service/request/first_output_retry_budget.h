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

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>

namespace xllm_service {

enum class FirstOutputRetryDecision {
  ALLOWED = 0,
  FIRST_OUTPUT_EMITTED,
  ATTEMPT_BUDGET_EXHAUSTED,
  DEADLINE_BUDGET_INSUFFICIENT,
  DEVICE_TIME_BUDGET_EXHAUSTED,
  CLOCK_REGRESSION,
};

// Service-local deterministic gate for replacing an attempt before the first
// response byte is written. The owner serializes access with the request's
// output-dispatch mutex.
class FirstOutputRetryBudget final {
 public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  struct Config {
    size_t max_attempt_retries = 1;
    uint64_t max_wasted_device_ms = 5000;
    uint64_t min_remaining_deadline_ms = 1000;
  };

  explicit FirstOutputRetryBudget(Config config, TimePoint now = Clock::now());

  FirstOutputRetryDecision evaluate(bool first_output_emitted,
                                    uint64_t remaining_deadline_ms,
                                    TimePoint now = Clock::now()) const;

  // Commits the currently elapsed attempt to the wasted-device budget and
  // starts the next attempt. Call only after the old execution hold converges.
  bool commit_retry(TimePoint now = Clock::now());

  size_t retries() const { return retries_; }
  uint64_t wasted_device_ms() const { return wasted_device_ms_; }

 private:
  uint64_t current_attempt_ms(TimePoint now) const;

  Config config_;
  TimePoint attempt_started_at_;
  size_t retries_ = 0;
  uint64_t wasted_device_ms_ = 0;
};

}  // namespace xllm_service
