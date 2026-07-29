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

#include "scheduler/managers/instance_lifecycle_event.h"

#include <utility>

namespace xllm_service {

InstanceLifecycleEventDispatcher::InstanceLifecycleEventDispatcher(
    Handler handler)
    : InstanceLifecycleEventDispatcher(
          std::vector<Handler>{std::move(handler)}) {}

InstanceLifecycleEventDispatcher::InstanceLifecycleEventDispatcher(
    std::vector<Handler> handlers)
    : handlers_(std::move(handlers)) {}

void InstanceLifecycleEventDispatcher::publish(
    const InstanceLifecycleEvent& event) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) {
    return;
  }
  for (const auto& handler : handlers_) {
    if (handler) {
      handler(event);
    }
  }
}

void InstanceLifecycleEventDispatcher::close() {
  std::lock_guard<std::mutex> lock(mutex_);
  closed_ = true;
}

}  // namespace xllm_service
