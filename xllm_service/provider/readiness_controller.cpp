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

#include "provider/readiness_controller.h"

#include <utility>

namespace xllm_service::provider {
namespace {

void set_error(std::string message, std::string* error) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

}  // namespace

const char* readiness_reason_name(ReadinessReason reason) {
  switch (reason) {
    case ReadinessReason::READY:
      return "READY";
    case ReadinessReason::STARTING_NO_FULL:
      return "STARTING_NO_FULL";
    case ReadinessReason::STARTING_NO_COMPATIBLE_CAPACITY:
      return "STARTING_NO_COMPATIBLE_CAPACITY";
    case ReadinessReason::OBSERVATION_UNAVAILABLE:
      return "OBSERVATION_UNAVAILABLE";
    case ReadinessReason::STATE_BLIND_NO_DIRECT_CAPACITY:
      return "STATE_BLIND_NO_DIRECT_CAPACITY";
    case ReadinessReason::REGISTRY_BLIND_NO_CACHED_CAPACITY:
      return "REGISTRY_BLIND_NO_CACHED_CAPACITY";
    case ReadinessReason::REGISTRY_BLIND_GRACE_EXPIRED:
      return "REGISTRY_BLIND_GRACE_EXPIRED";
    case ReadinessReason::DRAINING:
      return "DRAINING";
    case ReadinessReason::RECOVERY_HOLD:
      return "RECOVERY_HOLD";
    case ReadinessReason::NOT_LEADER:
      return "NOT_LEADER";
  }
  return "OBSERVATION_UNAVAILABLE";
}

ReadinessController::ReadinessController(ReadinessControllerConfig config)
    : config_(std::move(config)) {
  config_valid_ = config_.recovery_hold_ms > 0;
}

bool ReadinessController::valid() const { return config_valid_; }

std::optional<ReadinessSnapshot> ReadinessController::update(
    const ReadinessInput& input,
    uint64_t now_monotonic_ms,
    std::string* error) {
  if (error != nullptr) {
    error->clear();
  }
  if (!config_valid_) {
    set_error("Readiness Controller configuration is invalid", error);
    return std::nullopt;
  }
  if (initialized_ && now_monotonic_ms < last_update_monotonic_ms_) {
    set_error("Readiness monotonic clock regressed", error);
    return std::nullopt;
  }
  initialized_ = true;
  last_update_monotonic_ms_ = now_monotonic_ms;

  const std::optional<ReadinessReason> unsafe = unsafe_reason(input);
  if (unsafe.has_value()) {
    recovery_candidate_since_ms_.reset();
    set_snapshot(false, *unsafe, now_monotonic_ms);
    return snapshot_;
  }

  if (snapshot_.accepting_new_requests) {
    set_snapshot(true, ReadinessReason::READY, now_monotonic_ms);
    return snapshot_;
  }
  if (!recovery_candidate_since_ms_.has_value()) {
    recovery_candidate_since_ms_ = now_monotonic_ms;
    set_snapshot(false, ReadinessReason::RECOVERY_HOLD, now_monotonic_ms);
    return snapshot_;
  }
  if (now_monotonic_ms - *recovery_candidate_since_ms_ <
      config_.recovery_hold_ms) {
    set_snapshot(false, ReadinessReason::RECOVERY_HOLD, now_monotonic_ms);
    return snapshot_;
  }

  ever_ready_ = true;
  recovery_candidate_since_ms_.reset();
  set_snapshot(true, ReadinessReason::READY, now_monotonic_ms);
  return snapshot_;
}

std::optional<ReadinessReason> ReadinessController::unsafe_reason(
    const ReadinessInput& input) const {
  if (!input.is_leader) {
    return ReadinessReason::NOT_LEADER;
  }
  if (input.draining) {
    return ReadinessReason::DRAINING;
  }
  if (!input.has_accepted_full_snapshot) {
    return ReadinessReason::STARTING_NO_FULL;
  }
  if (!input.observation.has_value()) {
    return ReadinessReason::OBSERVATION_UNAVAILABLE;
  }
  if (input.observation->mode == ObservationMode::REGISTRY_BLIND) {
    if (!input.observation->within_grace) {
      return ReadinessReason::REGISTRY_BLIND_GRACE_EXPIRED;
    }
    if (!input.has_compatible_capacity) {
      return ReadinessReason::REGISTRY_BLIND_NO_CACHED_CAPACITY;
    }
  } else if (input.observation->mode == ObservationMode::STATE_BLIND &&
             !input.has_compatible_capacity) {
    return ReadinessReason::STATE_BLIND_NO_DIRECT_CAPACITY;
  }
  if (!input.has_compatible_capacity &&
      (!ever_ready_ || !snapshot_.accepting_new_requests)) {
    return ReadinessReason::STARTING_NO_COMPATIBLE_CAPACITY;
  }
  return std::nullopt;
}

void ReadinessController::set_snapshot(bool accepting_new_requests,
                                       ReadinessReason reason,
                                       uint64_t now_monotonic_ms) {
  if (!snapshot_initialized_ ||
      snapshot_.accepting_new_requests != accepting_new_requests ||
      snapshot_.reason != reason) {
    snapshot_.changed_monotonic_ms = now_monotonic_ms;
  }
  snapshot_initialized_ = true;
  snapshot_.accepting_new_requests = accepting_new_requests;
  snapshot_.reason = reason;
}

}  // namespace xllm_service::provider
