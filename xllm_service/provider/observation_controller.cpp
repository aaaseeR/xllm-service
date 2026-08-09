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

#include "provider/observation_controller.h"

#include <cmath>
#include <utility>

namespace xllm_service::provider {
namespace {

void set_error(std::string message, std::string* error) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

bool hold_elapsed(uint64_t candidate_since_ms,
                  uint64_t now_monotonic_ms,
                  uint64_t hold_ms) {
  return now_monotonic_ms - candidate_since_ms >= hold_ms;
}

}  // namespace

ObservationController::ObservationController(ObservationControllerConfig config)
    : config_(std::move(config)) {
  config_valid_ =
      std::isfinite(config_.state_blind_enter_ratio) &&
      std::isfinite(config_.state_blind_exit_ratio) &&
      config_.state_blind_exit_ratio >= 0.0 &&
      config_.state_blind_exit_ratio < config_.state_blind_enter_ratio &&
      config_.state_blind_enter_ratio <= 1.0 &&
      config_.state_blind_enter_hold_ms > 0 &&
      config_.state_blind_exit_hold_ms > 0 &&
      config_.state_blind_grace_ms > 0 && config_.registry_blind_grace_ms > 0;
}

bool ObservationController::valid() const { return config_valid_; }

std::optional<ObservationSnapshot> ObservationController::update(
    const ObservationInput& input,
    uint64_t now_monotonic_ms,
    std::string* error) {
  if (error != nullptr) {
    error->clear();
  }
  if (!config_valid_) {
    set_error("Observation Controller configuration is invalid", error);
    return std::nullopt;
  }
  if (input.hard_stale_member_count > input.member_count) {
    set_error("Observation input stale count exceeds member count", error);
    return std::nullopt;
  }
  if (initialized_ && now_monotonic_ms < last_update_monotonic_ms_) {
    set_error("Observation monotonic clock regressed", error);
    return std::nullopt;
  }

  const double hard_stale_ratio =
      input.member_count == 0
          ? 0.0
          : static_cast<double>(input.hard_stale_member_count) /
                static_cast<double>(input.member_count);
  if (!initialized_) {
    initialized_ = true;
    if (!input.registry_known) {
      enter_mode(ObservationMode::REGISTRY_BLIND, now_monotonic_ms);
    } else if (!input.has_usable_state_snapshot ||
               hard_stale_ratio >= config_.state_blind_enter_ratio) {
      enter_mode(ObservationMode::STATE_BLIND, now_monotonic_ms);
    } else {
      enter_mode(ObservationMode::NORMAL, now_monotonic_ms);
    }
    last_update_monotonic_ms_ = now_monotonic_ms;
    return make_snapshot(now_monotonic_ms, hard_stale_ratio);
  }

  last_update_monotonic_ms_ = now_monotonic_ms;
  if (!input.registry_known) {
    if (mode_ != ObservationMode::REGISTRY_BLIND) {
      enter_mode(ObservationMode::REGISTRY_BLIND, now_monotonic_ms);
    }
    return make_snapshot(now_monotonic_ms, hard_stale_ratio);
  }

  if (mode_ == ObservationMode::REGISTRY_BLIND) {
    if (!input.has_usable_state_snapshot || !input.has_current_full_snapshot ||
        hard_stale_ratio > config_.state_blind_exit_ratio) {
      enter_mode(ObservationMode::STATE_BLIND, now_monotonic_ms);
    } else {
      enter_mode(ObservationMode::NORMAL, now_monotonic_ms);
    }
    return make_snapshot(now_monotonic_ms, hard_stale_ratio);
  }

  if (mode_ == ObservationMode::NORMAL) {
    const bool should_enter_state_blind =
        !input.has_usable_state_snapshot ||
        hard_stale_ratio >= config_.state_blind_enter_ratio;
    if (!should_enter_state_blind) {
      state_blind_candidate_since_ms_.reset();
      return make_snapshot(now_monotonic_ms, hard_stale_ratio);
    }
    if (!state_blind_candidate_since_ms_.has_value()) {
      state_blind_candidate_since_ms_ = now_monotonic_ms;
    } else if (hold_elapsed(*state_blind_candidate_since_ms_,
                            now_monotonic_ms,
                            config_.state_blind_enter_hold_ms)) {
      enter_mode(ObservationMode::STATE_BLIND, now_monotonic_ms);
    }
    return make_snapshot(now_monotonic_ms, hard_stale_ratio);
  }

  const bool should_exit_state_blind =
      input.has_usable_state_snapshot && input.has_current_full_snapshot &&
      hard_stale_ratio <= config_.state_blind_exit_ratio;
  if (!should_exit_state_blind) {
    state_recovery_candidate_since_ms_.reset();
    return make_snapshot(now_monotonic_ms, hard_stale_ratio);
  }
  if (!state_recovery_candidate_since_ms_.has_value()) {
    state_recovery_candidate_since_ms_ = now_monotonic_ms;
  } else if (hold_elapsed(*state_recovery_candidate_since_ms_,
                          now_monotonic_ms,
                          config_.state_blind_exit_hold_ms)) {
    enter_mode(ObservationMode::NORMAL, now_monotonic_ms);
  }
  return make_snapshot(now_monotonic_ms, hard_stale_ratio);
}

ObservationSnapshot ObservationController::make_snapshot(
    uint64_t now_monotonic_ms,
    double hard_stale_ratio) const {
  uint64_t grace_ms = 0;
  if (mode_ == ObservationMode::STATE_BLIND) {
    grace_ms = config_.state_blind_grace_ms;
  } else if (mode_ == ObservationMode::REGISTRY_BLIND) {
    grace_ms = config_.registry_blind_grace_ms;
  }
  return ObservationSnapshot{
      .mode = mode_,
      .mode_entered_monotonic_ms = mode_entered_monotonic_ms_,
      .hard_stale_ratio = hard_stale_ratio,
      .within_grace = grace_ms > 0 &&
                      now_monotonic_ms - mode_entered_monotonic_ms_ < grace_ms,
  };
}

void ObservationController::enter_mode(ObservationMode mode,
                                       uint64_t now_monotonic_ms) {
  mode_ = mode;
  mode_entered_monotonic_ms_ = now_monotonic_ms;
  state_blind_candidate_since_ms_.reset();
  state_recovery_candidate_since_ms_.reset();
}

}  // namespace xllm_service::provider
