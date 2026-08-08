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
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/request/request_deadline.h"

namespace xllm_service {

struct Request;

enum class RequestDeadlineQueueStatus {
  kOk = 0,
  kDuplicate,
  kCapacityExceeded,
  kInvalid,
};

// Bounded deadline index. It holds weak request references and removes entries
// eagerly on request completion, so historical traffic cannot accumulate.
class RequestDeadlineQueue final {
 public:
  using TimePoint = xllm::RequestDeadline::TimePoint;

  explicit RequestDeadlineQueue(size_t capacity) : capacity_(capacity) {}

  RequestDeadlineQueueStatus insert(const std::string& request_uid,
                                    const std::shared_ptr<Request>& request,
                                    TimePoint deadline);

  bool erase(const std::string& request_uid);

  std::vector<std::shared_ptr<Request>> take_expired(TimePoint now,
                                                     size_t max_entries);

  bool has_expired(TimePoint now) const;

  size_t size() const;

 private:
  struct Entry {
    std::string request_uid;
    std::weak_ptr<Request> request;
  };

  using DeadlineMap = std::multimap<TimePoint, Entry>;

  size_t capacity_ = 0;
  mutable std::mutex mutex_;
  DeadlineMap deadlines_;
  std::unordered_map<std::string, DeadlineMap::iterator> index_;
};

}  // namespace xllm_service
