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

#include "runtime/runtime_state.h"

#include <utility>

namespace xllm_service {

namespace {
constexpr char kBackendNotReady[] = "backend is not ready";
constexpr char kStarting[] = "starting";
constexpr char kDraining[] = "draining";
constexpr char kStopped[] = "stopped";
}  // namespace

const char* runtime_phase_name(RuntimePhase phase) {
  switch (phase) {
    case RuntimePhase::STARTING:
      return "starting";
    case RuntimePhase::RUNNING:
      return "running";
    case RuntimePhase::DRAINING:
      return "draining";
    case RuntimePhase::STOPPED:
      return "stopped";
  }
  return "unknown";
}

void RuntimeState::mark_running() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (phase_ == RuntimePhase::STARTING) {
    phase_ = RuntimePhase::RUNNING;
  }
}

void RuntimeState::set_backend_ready(bool ready, std::string reason) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (phase_ == RuntimePhase::DRAINING || phase_ == RuntimePhase::STOPPED) {
    return;
  }

  backend_ready_ = ready;
  if (ready) {
    backend_reason_.clear();
    return;
  }
  backend_reason_ = reason.empty() ? kBackendNotReady : std::move(reason);
}

void RuntimeState::begin_draining() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (phase_ == RuntimePhase::STOPPED) {
    return;
  }
  phase_ = RuntimePhase::DRAINING;
  backend_ready_ = false;
  backend_reason_ = kDraining;
}

void RuntimeState::mark_stopped() {
  std::lock_guard<std::mutex> lock(mutex_);
  phase_ = RuntimePhase::STOPPED;
  backend_ready_ = false;
  backend_reason_ = kStopped;
}

RuntimeHealthSnapshot RuntimeState::health_snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);

  RuntimeHealthSnapshot snapshot;
  snapshot.phase = phase_;
  snapshot.live = phase_ != RuntimePhase::STOPPED;
  snapshot.ready = phase_ == RuntimePhase::RUNNING && backend_ready_;

  switch (phase_) {
    case RuntimePhase::STARTING:
      snapshot.reason = kStarting;
      break;
    case RuntimePhase::RUNNING:
      snapshot.reason = backend_ready_ ? "" : backend_reason_;
      break;
    case RuntimePhase::DRAINING:
      snapshot.reason = kDraining;
      break;
    case RuntimePhase::STOPPED:
      snapshot.reason = kStopped;
      break;
  }
  return snapshot;
}

}  // namespace xllm_service
