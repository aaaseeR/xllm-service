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

#include <cstdint>
#include <string>
#include <vector>

#include "provider.pb.h"

namespace xllm_service {

struct KVStateStreamPush {
  std::string subscriber;
  xllm::proto::KVStateBatch batch;
  uint64_t timeout_ms = 0;
};

struct KVStateStreamPushResult {
  bool ok = false;
  bool timed_out = false;
  std::string message;
};

// Issues all KV pushes before joining any call. This worker and its channels
// are independent from the health/load State Stream worker.
std::vector<KVStateStreamPushResult> push_kv_state_stream_batches(
    const std::vector<KVStateStreamPush>& pushes);

}  // namespace xllm_service
