/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "client_disconnect_monitor.h"

#include <algorithm>
#include <iterator>
#include <utility>

namespace xllm_service {

ClientDisconnectMonitor::ClientDisconnectMonitor(size_t capacity)
    : capacity_(capacity) {}

ClientDisconnectMonitorStatus ClientDisconnectMonitor::register_request(
    std::string request_uid,
    std::weak_ptr<Request> request) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) {
    return ClientDisconnectMonitorStatus::CLOSED;
  }
  if (entries_.find(request_uid) != entries_.end()) {
    return ClientDisconnectMonitorStatus::DUPLICATE;
  }
  if (capacity_ == 0 || entries_.size() >= capacity_) {
    return ClientDisconnectMonitorStatus::CAPACITY_EXCEEDED;
  }
  entries_.emplace(std::move(request_uid),
                   Entry{std::move(request), std::nullopt});
  return ClientDisconnectMonitorStatus::OK;
}

void ClientDisconnectMonitor::notify_disconnected(
    const std::string& request_uid) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) {
    return;
  }
  auto it = entries_.find(request_uid);
  if (it == entries_.end() || it->second.pending_it.has_value()) {
    return;
  }
  if (std::shared_ptr<Request> request = it->second.request.lock()) {
    request->client_disconnected.store(true, std::memory_order_release);
    pending_.emplace_back(request_uid);
    it->second.pending_it = std::prev(pending_.end());
  } else {
    entries_.erase(it);
  }
}

std::vector<std::shared_ptr<Request>>
ClientDisconnectMonitor::take_disconnected(size_t max_items) {
  std::vector<std::shared_ptr<Request>> requests;
  std::lock_guard<std::mutex> lock(mutex_);
  const size_t count = std::min(max_items, pending_.size());
  requests.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    std::string request_uid = std::move(pending_.front());
    pending_.pop_front();
    auto it = entries_.find(request_uid);
    if (it == entries_.end() || !it->second.pending_it.has_value()) {
      continue;
    }
    it->second.pending_it.reset();
    if (std::shared_ptr<Request> request = it->second.request.lock()) {
      requests.emplace_back(std::move(request));
    } else {
      entries_.erase(it);
    }
  }
  return requests;
}

bool ClientDisconnectMonitor::erase(const std::string& request_uid) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = entries_.find(request_uid);
  if (it == entries_.end()) {
    return false;
  }
  if (it->second.pending_it.has_value()) {
    pending_.erase(*it->second.pending_it);
  }
  entries_.erase(it);
  return true;
}

void ClientDisconnectMonitor::close() {
  std::lock_guard<std::mutex> lock(mutex_);
  closed_ = true;
  entries_.clear();
  pending_.clear();
}

size_t ClientDisconnectMonitor::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return entries_.size();
}

size_t ClientDisconnectMonitor::pending() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return pending_.size();
}

}  // namespace xllm_service
