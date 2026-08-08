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

#pragma once

#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "request/request.h"

namespace xllm_service {

enum class ClientDisconnectMonitorStatus {
  OK,
  DUPLICATE,
  CAPACITY_EXCEEDED,
  CLOSED,
};

// Reserves one bounded notification slot at request ingress. brpc cancellation
// callbacks only mark the existing slot ready, so a disconnect can never be
// dropped because an event queue fills during a burst.
class ClientDisconnectMonitor final {
 public:
  explicit ClientDisconnectMonitor(size_t capacity);

  ClientDisconnectMonitorStatus register_request(
      std::string request_uid,
      std::weak_ptr<Request> request);
  void notify_disconnected(const std::string& request_uid);
  std::vector<std::shared_ptr<Request>> take_disconnected(size_t max_items);
  bool erase(const std::string& request_uid);
  void close();

  size_t size() const;
  size_t pending() const;

 private:
  struct Entry {
    std::weak_ptr<Request> request;
    std::optional<std::list<std::string>::iterator> pending_it;
  };

  const size_t capacity_;
  mutable std::mutex mutex_;
  bool closed_ = false;
  std::unordered_map<std::string, Entry> entries_;
  std::list<std::string> pending_;
};

}  // namespace xllm_service
