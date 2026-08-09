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
#include <mutex>
#include <string>

#include "provider.pb.h"
#include "provider/kv_shadow_index.h"

namespace xllm_service::provider {

struct KVStateReplicaConfig {
  size_t max_engine_batches = 64;
  size_t max_serialized_bytes = 1024 * 1024;
};

// Validates the master-to-replica ordering domain before applying individual
// Engine journals. A master change or Service-lane sequence gap drops all KV
// credit, while the ordinary health/load routing plane remains untouched.
class KVStateReplica final {
 public:
  KVStateReplica(KVStateReplicaConfig config, KVShadowIndex* shadow_index);

  bool set_master(std::string master_incarnation);
  KVApplyResult apply(const xllm::proto::KVStateBatch& batch,
                      uint64_t received_monotonic_ms);

  std::string master_incarnation() const;
  uint64_t stream_epoch() const;
  uint64_t last_stream_seq() const;

 private:
  static KVApplyResult result(KVApplyCode code,
                              std::string reason,
                              uint64_t accepted_through_event_seq);

  KVStateReplicaConfig config_;
  bool config_valid_ = false;
  KVShadowIndex* shadow_index_ = nullptr;
  mutable std::mutex mutex_;
  std::string master_incarnation_;
  uint64_t stream_epoch_ = 0;
  uint64_t last_stream_seq_ = 0;
};

}  // namespace xllm_service::provider
