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

#include <brpc/channel.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "common/xllm/output.h"
#include "common/xllm/status.h"
#include "provider.pb.h"

namespace xllm_service {

struct FirstEventRecoveryQuery {
  std::shared_ptr<brpc::Channel> channel;
  xllm::proto::ExecutionAttemptId attempt;
  xllm::proto::ExecutionHolder decode;
  xllm::proto::ExecutionHolder prefill;
  size_t max_payload_bytes = 0;
  uint64_t timeout_ms = 0;
};

struct FirstEventRecoveryResult {
  llm::Status status;
  std::optional<llm::RequestOutput> output;
};

// Issues every valid QueryRequest before joining any call. Results preserve
// input order. Each query has an independent timeout and identity fence.
std::vector<FirstEventRecoveryResult> query_first_output_events(
    const std::vector<FirstEventRecoveryQuery>& queries);

}  // namespace xllm_service
