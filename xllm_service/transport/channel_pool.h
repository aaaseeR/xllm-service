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

#include <brpc/channel.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "common/options.h"

namespace xllm_service {

enum class ChannelLookupStatus : uint8_t {
  AVAILABLE = 0,
  ENDPOINT_NOT_FOUND = 1,
  INCARNATION_MISMATCH = 2,
  CHANNEL_INITIALIZATION_FAILED = 3,
  MEMBERSHIP_CHANGED = 4,
};

const char* channel_lookup_status_name(ChannelLookupStatus status);

struct ChannelLookupResult {
  ChannelLookupStatus status = ChannelLookupStatus::ENDPOINT_NOT_FOUND;
  std::shared_ptr<brpc::Channel> channel;
  std::string message;

  bool available() const {
    return status == ChannelLookupStatus::AVAILABLE && channel != nullptr;
  }

  bool stale() const {
    return status == ChannelLookupStatus::ENDPOINT_NOT_FOUND ||
           status == ChannelLookupStatus::INCARNATION_MISMATCH ||
           status == ChannelLookupStatus::MEMBERSHIP_CHANGED;
  }
};

class ChannelPool final {
 public:
  using ChannelFactory = std::function<std::shared_ptr<brpc::Channel>(
      const std::string& endpoint)>;

  explicit ChannelPool(const Options& options);
  explicit ChannelPool(ChannelFactory channel_factory);

  ChannelPool(const ChannelPool&) = delete;
  ChannelPool& operator=(const ChannelPool&) = delete;

  bool activate(const std::string& endpoint, const std::string& incarnation_id);
  bool remove(const std::string& endpoint, const std::string& incarnation_id);
  ChannelLookupResult get_or_create_with_status(
      const std::string& endpoint,
      const std::string& incarnation_id);
  std::shared_ptr<brpc::Channel> get_or_create(
      const std::string& endpoint,
      const std::string& incarnation_id);

  size_t channel_count() const;
  size_t endpoint_count() const;

 private:
  struct Entry {
    std::string incarnation_id;
    uint64_t version = 0;
    std::shared_ptr<brpc::Channel> channel;
  };

  const ChannelFactory channel_factory_;
  mutable std::mutex mutex_;
  std::unordered_map<std::string, Entry> entries_;
};

}  // namespace xllm_service
