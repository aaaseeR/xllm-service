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
#include <functional>
#include <optional>
#include <string>

#include "observability.pb.h"

namespace xllm_service::observability {

inline constexpr size_t kMaxCorrelationIdLength = 256;

struct RequestCorrelationInput {
  std::string global_request_id;
  std::string trace_id;
  std::string traceparent;
};

using CorrelationIdGenerator = std::function<std::string()>;

bool valid_correlation_id(const std::string& value);
std::optional<std::string> trace_id_from_traceparent(
    const std::string& traceparent);

xllm::proto::RequestCorrelation make_request_correlation(
    const RequestCorrelationInput& input,
    const CorrelationIdGenerator& id_generator);
xllm::proto::RequestCorrelation make_request_correlation(
    const RequestCorrelationInput& input);

}  // namespace xllm_service::observability
