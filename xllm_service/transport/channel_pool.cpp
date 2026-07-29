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

#include "transport/channel_pool.h"

#include <glog/logging.h>

#include <utility>

namespace xllm_service {
namespace {

ChannelPool::ChannelFactory make_channel_factory(const Options& options) {
  return [options](const std::string& endpoint) {
    auto channel = std::make_shared<brpc::Channel>();
    brpc::ChannelOptions channel_options;
    channel_options.timeout_ms = options.timeout_ms();
    channel_options.connect_timeout_ms = options.connect_timeout_ms();
    channel_options.max_retry = 3;
    if (channel->Init(endpoint.c_str(), "", &channel_options) != 0) {
      LOG(ERROR) << "Failed to initialize backend channel: " << endpoint;
      return std::shared_ptr<brpc::Channel>();
    }
    return channel;
  };
}

}  // namespace

ChannelPool::ChannelPool(const Options& options)
    : ChannelPool(make_channel_factory(options)) {}

ChannelPool::ChannelPool(ChannelFactory channel_factory)
    : channel_factory_(std::move(channel_factory)) {}

bool ChannelPool::activate(const std::string& endpoint,
                           const std::string& incarnation_id) {
  if (endpoint.empty()) {
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  auto it = entries_.find(endpoint);
  if (it == entries_.end()) {
    entries_.emplace(endpoint, Entry{incarnation_id, 1, nullptr});
    return true;
  }
  if (it->second.incarnation_id == incarnation_id) {
    return true;
  }
  it->second.incarnation_id = incarnation_id;
  ++it->second.version;
  it->second.channel.reset();
  return true;
}

bool ChannelPool::remove(const std::string& endpoint,
                         const std::string& incarnation_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = entries_.find(endpoint);
  if (it == entries_.end() ||
      it->second.incarnation_id != incarnation_id) {
    return false;
  }
  entries_.erase(it);
  return true;
}

std::shared_ptr<brpc::Channel> ChannelPool::get_or_create(
    const std::string& endpoint,
    const std::string& incarnation_id) {
  uint64_t version = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(endpoint);
    if (it == entries_.end() ||
        it->second.incarnation_id != incarnation_id) {
      return nullptr;
    }
    if (it->second.channel != nullptr) {
      return it->second.channel;
    }
    version = it->second.version;
  }

  std::shared_ptr<brpc::Channel> channel = channel_factory_(endpoint);
  if (channel == nullptr) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  auto it = entries_.find(endpoint);
  if (it == entries_.end() || it->second.version != version ||
      it->second.incarnation_id != incarnation_id) {
    return nullptr;
  }
  if (it->second.channel == nullptr) {
    it->second.channel = std::move(channel);
  }
  return it->second.channel;
}

size_t ChannelPool::channel_count() const {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t count = 0;
  for (const auto& item : entries_) {
    if (item.second.channel != nullptr) {
      ++count;
    }
  }
  return count;
}

size_t ChannelPool::endpoint_count() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return entries_.size();
}

}  // namespace xllm_service
