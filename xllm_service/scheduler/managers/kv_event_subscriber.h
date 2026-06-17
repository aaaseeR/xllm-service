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

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include "common/macros.h"
#include "common/types.h"
#include "xllm_rpc_service.pb.h"

namespace xllm_service {

class KvEventSubscriber final {
 public:
  using RecordCallback =
      std::function<void(const std::string&, const proto::KvCacheEvent&)>;
  using SnapshotCallback =
      std::function<void(const std::string&, const proto::KvCacheEvent&)>;
  using ClearCallback = std::function<void(const std::string&)>;

  struct Options {
    PROPERTY(bool, enabled) = false;
    PROPERTY(int32_t, poll_interval_ms) = 20;
    PROPERTY(int32_t, reconnect_interval_ms) = 1000;
    PROPERTY(int32_t, reconnect_interval_max_ms) = 10000;
    PROPERTY(RecordCallback, record_callback);
    PROPERTY(SnapshotCallback, snapshot_callback);
    PROPERTY(ClearCallback, clear_callback);
  };

  explicit KvEventSubscriber(Options options);
  ~KvEventSubscriber();

  bool start();
  void stop();

  void add_or_update_source(const InstanceMetaInfo& info);
  void remove_source(const std::string& instance_name,
                     const std::string& incarnation_id = "");

  nlohmann::json debug_summary() const;

 private:
  DISALLOW_COPY_AND_ASSIGN(KvEventSubscriber);

  enum class CommandType {
    ADD_OR_UPDATE,
    REMOVE,
  };

  struct Command {
    CommandType type;
    InstanceMetaInfo info;
    std::string instance_name;
    std::string incarnation_id;
  };

  struct SourceState {
    std::string endpoint;
    std::string incarnation_id;
    uint64_t last_seq_no = 0;
    bool has_seq = false;
    bool suspect = false;
  };

  struct ReceivedEvent {
    std::string instance_name;
    proto::KvCacheEvent cache_event;
    bool snapshot = false;
  };

  void enqueue(Command command);
  void run_loop();

 private:
  Options options_;
  std::atomic_bool exited_{false};
  std::atomic_bool started_{false};
  std::unique_ptr<std::thread> subscriber_thread_;

  mutable std::mutex mutex_;
  std::deque<Command> commands_;
  std::unordered_map<std::string, SourceState> sources_;
  uint64_t received_events_ = 0;
  uint64_t received_snapshots_ = 0;
  uint64_t seq_gap_events_ = 0;
  uint64_t stale_events_ = 0;
};

}  // namespace xllm_service
