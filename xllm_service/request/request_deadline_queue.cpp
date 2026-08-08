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

#include "request/request_deadline_queue.h"

#include <algorithm>
#include <utility>

namespace xllm_service {

RequestDeadlineQueueStatus RequestDeadlineQueue::insert(
    const std::string& request_uid,
    const std::shared_ptr<Request>& request,
    TimePoint deadline) {
  if (request_uid.empty() || request == nullptr) {
    return RequestDeadlineQueueStatus::kInvalid;
  }
  std::lock_guard<std::mutex> guard(mutex_);
  if (index_.find(request_uid) != index_.end()) {
    return RequestDeadlineQueueStatus::kDuplicate;
  }
  if (deadlines_.size() >= capacity_) {
    return RequestDeadlineQueueStatus::kCapacityExceeded;
  }
  auto it = deadlines_.emplace(
      deadline, Entry{.request_uid = request_uid, .request = request});
  index_.emplace(request_uid, it);
  return RequestDeadlineQueueStatus::kOk;
}

bool RequestDeadlineQueue::erase(const std::string& request_uid) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = index_.find(request_uid);
  if (it == index_.end()) {
    return false;
  }
  deadlines_.erase(it->second);
  index_.erase(it);
  return true;
}

std::vector<std::shared_ptr<Request>> RequestDeadlineQueue::take_expired(
    TimePoint now,
    size_t max_entries) {
  std::vector<std::shared_ptr<Request>> expired;
  std::lock_guard<std::mutex> guard(mutex_);
  expired.reserve(std::min(max_entries, deadlines_.size()));
  size_t visited = 0;
  while (visited < max_entries && !deadlines_.empty() &&
         deadlines_.begin()->first <= now) {
    auto it = deadlines_.begin();
    std::shared_ptr<Request> request = it->second.request.lock();
    index_.erase(it->second.request_uid);
    deadlines_.erase(it);
    ++visited;
    if (request != nullptr) {
      expired.emplace_back(std::move(request));
    }
  }
  return expired;
}

bool RequestDeadlineQueue::has_expired(TimePoint now) const {
  std::lock_guard<std::mutex> guard(mutex_);
  return !deadlines_.empty() && deadlines_.begin()->first <= now;
}

size_t RequestDeadlineQueue::size() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return deadlines_.size();
}

}  // namespace xllm_service
