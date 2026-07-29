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
#include <mutex>
#include <string>

namespace xllm_service {

enum class RuntimePhase : uint8_t {
  STARTING = 0,
  RUNNING = 1,
  DRAINING = 2,
  STOPPED = 3,
};

struct RuntimeHealthSnapshot {
  RuntimePhase phase = RuntimePhase::STARTING;
  bool live = true;
  bool ready = false;
  std::string reason = "starting";
};

const char* runtime_phase_name(RuntimePhase phase);

class RuntimeState final {
 public:
  RuntimeState() = default;

  void mark_running();
  void set_backend_ready(bool ready, std::string reason = "");
  void begin_draining();
  void mark_stopped();

  RuntimeHealthSnapshot health_snapshot() const;

 private:
  mutable std::mutex mutex_;
  RuntimePhase phase_ = RuntimePhase::STARTING;
  bool backend_ready_ = false;
  std::string backend_reason_ = "backend is not ready";
};

}  // namespace xllm_service
