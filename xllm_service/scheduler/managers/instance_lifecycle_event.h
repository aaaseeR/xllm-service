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
#include <condition_variable>
#include <functional>
#include <mutex>
#include <vector>

#include "common/types.h"

namespace xllm_service {

enum class InstanceLifecycleEventType : uint8_t {
  REGISTERED = 0,
  REGISTRATION_UPDATED = 1,
  DEREGISTERING = 2,
  DEREGISTERED = 3,
};

struct InstanceLifecycleEvent {
  InstanceLifecycleEventType type = InstanceLifecycleEventType::REGISTERED;
  InstanceMetaInfo instance;
};

class InstanceLifecycleEventDispatcher final {
 public:
  using Handler = std::function<void(const InstanceLifecycleEvent&)>;

  explicit InstanceLifecycleEventDispatcher(Handler handler);
  explicit InstanceLifecycleEventDispatcher(std::vector<Handler> handlers);

  void close();
  void publish(const InstanceLifecycleEvent& event) const;

 private:
  const std::vector<Handler> handlers_;
  mutable std::mutex mutex_;
  mutable std::condition_variable condition_;
  mutable bool closed_ = false;
  mutable size_t inflight_publish_count_ = 0;
};

}  // namespace xllm_service
