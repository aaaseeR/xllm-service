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
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "provider.pb.h"

namespace xllm_service {

struct KVSnapshotPageQuery {
  std::shared_ptr<brpc::Channel> channel;
  xllm::proto::KVCacheSnapshotRequest request;
  size_t max_response_bytes = 0;
  uint64_t timeout_ms = 0;
};

struct KVSnapshotPageResult {
  bool ok = false;
  bool timed_out = false;
  std::string message;
  std::optional<xllm::proto::KVCacheSnapshotPage> page;
};

// Issues all page queries before joining. Pagination for one Engine remains
// sequential, while independent Engine recoveries can progress concurrently.
std::vector<KVSnapshotPageResult> query_kv_snapshot_pages(
    const std::vector<KVSnapshotPageQuery>& queries);

}  // namespace xllm_service
