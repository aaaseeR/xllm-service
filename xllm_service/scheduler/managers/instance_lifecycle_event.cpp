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

#include <glog/logging.h>

#include <exception>
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
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return;
    }
    ++inflight_publish_count_;
  }
  for (const auto& handler : handlers_) {
    if (handler) {
      try {
        handler(event);
      } catch (const std::exception& error) {
        LOG(ERROR) << "Instance lifecycle handler failed: " << error.what();
      } catch (...) {
        LOG(ERROR) << "Instance lifecycle handler failed with unknown error";
      }
    }
  }

  std::lock_guard<std::mutex> lock(mutex_);
  --inflight_publish_count_;
  if (inflight_publish_count_ == 0) {
    condition_.notify_all();
  }
}

void InstanceLifecycleEventDispatcher::close() {
  std::unique_lock<std::mutex> lock(mutex_);
  closed_ = true;
  condition_.wait(lock, [this]() { return inflight_publish_count_ == 0; });
}

}  // namespace xllm_service
