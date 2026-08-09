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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace xllm_service::provider {

enum class ObservationMode : int8_t {
  NORMAL = 0,
  STATE_BLIND = 1,
  REGISTRY_BLIND = 2,
};

struct ObservationControllerConfig {
  double state_blind_enter_ratio = 0.5;
  double state_blind_exit_ratio = 0.2;
  uint64_t state_blind_enter_hold_ms = 1000;
  uint64_t state_blind_exit_hold_ms = 3000;
  uint64_t state_blind_grace_ms = 10000;
  uint64_t registry_blind_grace_ms = 3000;
};

struct ObservationInput {
  bool registry_known = false;
  bool has_usable_state_snapshot = false;
  bool has_current_full_snapshot = false;
  size_t member_count = 0;
  size_t hard_stale_member_count = 0;
};

struct ObservationSnapshot {
  ObservationMode mode = ObservationMode::REGISTRY_BLIND;
  uint64_t mode_entered_monotonic_ms = 0;
  double hard_stale_ratio = 1.0;
  bool within_grace = false;
};

// Applies same-unit enter/exit thresholds plus time holds. Registry blindness
// is immediate because member identity cannot be confirmed; state blindness
// uses hysteresis so a transient publication delay cannot flap the pool.
class ObservationController final {
 public:
  explicit ObservationController(ObservationControllerConfig config);

  bool valid() const;

  std::optional<ObservationSnapshot> update(const ObservationInput& input,
                                            uint64_t now_monotonic_ms,
                                            std::string* error);

 private:
  ObservationSnapshot make_snapshot(uint64_t now_monotonic_ms,
                                    double hard_stale_ratio) const;
  void enter_mode(ObservationMode mode, uint64_t now_monotonic_ms);

  ObservationControllerConfig config_;
  bool config_valid_ = false;
  bool initialized_ = false;
  ObservationMode mode_ = ObservationMode::REGISTRY_BLIND;
  uint64_t mode_entered_monotonic_ms_ = 0;
  uint64_t last_update_monotonic_ms_ = 0;
  std::optional<uint64_t> state_blind_candidate_since_ms_;
  std::optional<uint64_t> state_recovery_candidate_since_ms_;
};

}  // namespace xllm_service::provider
