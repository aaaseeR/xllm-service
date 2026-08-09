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
#include <optional>
#include <string>

#include "provider/observation_controller.h"

namespace xllm_service::provider {

enum class ReadinessReason : int8_t {
  READY = 0,
  STARTING_NO_FULL = 1,
  STARTING_NO_COMPATIBLE_CAPACITY = 2,
  OBSERVATION_UNAVAILABLE = 3,
  STATE_BLIND_NO_DIRECT_CAPACITY = 4,
  REGISTRY_BLIND_NO_CACHED_CAPACITY = 5,
  REGISTRY_BLIND_GRACE_EXPIRED = 6,
  DRAINING = 7,
  RECOVERY_HOLD = 8,
};

const char* readiness_reason_name(ReadinessReason reason);

struct ReadinessControllerConfig {
  uint64_t recovery_hold_ms = 3000;
};

struct ReadinessInput {
  bool has_accepted_full_snapshot = false;
  bool has_compatible_capacity = false;
  bool draining = false;
  std::optional<ObservationSnapshot> observation;
};

struct ReadinessSnapshot {
  bool accepting_new_requests = false;
  ReadinessReason reason = ReadinessReason::STARTING_NO_FULL;
  uint64_t changed_monotonic_ms = 0;
};

// Keeps process liveness independent from admission readiness. Unsafe states
// are applied immediately, while every recovery must remain valid for the
// configured hold. A transient capacity shortage in NORMAL does not remove an
// already-ready replica from the load balancer.
class ReadinessController final {
 public:
  explicit ReadinessController(ReadinessControllerConfig config);

  bool valid() const;

  std::optional<ReadinessSnapshot> update(const ReadinessInput& input,
                                          uint64_t now_monotonic_ms,
                                          std::string* error);

 private:
  std::optional<ReadinessReason> unsafe_reason(
      const ReadinessInput& input) const;
  void set_snapshot(bool accepting_new_requests,
                    ReadinessReason reason,
                    uint64_t now_monotonic_ms);

  ReadinessControllerConfig config_;
  bool config_valid_ = false;
  bool initialized_ = false;
  bool snapshot_initialized_ = false;
  bool ever_ready_ = false;
  uint64_t last_update_monotonic_ms_ = 0;
  std::optional<uint64_t> recovery_candidate_since_ms_;
  ReadinessSnapshot snapshot_;
};

}  // namespace xllm_service::provider
