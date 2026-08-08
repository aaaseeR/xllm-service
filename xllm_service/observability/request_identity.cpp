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

#include "observability/request_identity.h"

#include <algorithm>
#include <cctype>
#include <string>

#include "common/xllm/uuid.h"

namespace xllm_service::observability {
namespace {

bool is_hex(char value) {
  return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f') ||
         (value >= 'A' && value <= 'F');
}

bool valid_hex_field(const std::string& value,
                     size_t offset,
                     size_t length,
                     bool reject_all_zero) {
  bool has_nonzero = false;
  for (size_t i = offset; i < offset + length; ++i) {
    if (!is_hex(value[i])) {
      return false;
    }
    has_nonzero = has_nonzero || value[i] != '0';
  }
  return !reject_all_zero || has_nonzero;
}

std::string generate_valid_id(const CorrelationIdGenerator& id_generator) {
  std::string value = id_generator();
  if (valid_correlation_id(value)) {
    return value;
  }
  return llm::new_uuid_v7();
}

std::string generate_request_uid(const CorrelationIdGenerator& id_generator) {
  std::string value = id_generator();
  if (llm::is_uuid_v7(value)) {
    return value;
  }
  return llm::new_uuid_v7();
}

void lowercase_ascii(std::string* value) {
  std::transform(value->begin(),
                 value->end(),
                 value->begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
}

}  // namespace

bool valid_correlation_id(const std::string& value) {
  if (value.empty() || value.size() > kMaxCorrelationIdLength) {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](unsigned char character) {
    return character >= 0x21 && character <= 0x7e;
  });
}

std::optional<std::string> trace_id_from_traceparent(
    const std::string& traceparent) {
  constexpr size_t kBaseTraceparentLength = 55;
  if (traceparent.size() < kBaseTraceparentLength ||
      traceparent.size() > kMaxCorrelationIdLength ||
      !valid_correlation_id(traceparent) || traceparent[2] != '-' ||
      traceparent[35] != '-' || traceparent[52] != '-' ||
      !valid_hex_field(traceparent, 0, 2, /*reject_all_zero=*/false) ||
      !valid_hex_field(traceparent, 3, 32, /*reject_all_zero=*/true) ||
      !valid_hex_field(traceparent, 36, 16, /*reject_all_zero=*/true) ||
      !valid_hex_field(traceparent, 53, 2, /*reject_all_zero=*/false)) {
    return std::nullopt;
  }

  std::string version = traceparent.substr(0, 2);
  lowercase_ascii(&version);
  if (version == "ff" ||
      (version == "00" && traceparent.size() != kBaseTraceparentLength) ||
      (traceparent.size() > kBaseTraceparentLength &&
       traceparent[kBaseTraceparentLength] != '-')) {
    return std::nullopt;
  }

  std::string trace_id = traceparent.substr(3, 32);
  lowercase_ascii(&trace_id);
  return trace_id;
}

xllm::proto::RequestCorrelation make_request_correlation(
    const RequestCorrelationInput& input,
    const CorrelationIdGenerator& id_generator) {
  xllm::proto::RequestCorrelation correlation;
  if (valid_correlation_id(input.global_request_id)) {
    correlation.set_global_request_id(input.global_request_id);
    correlation.set_global_request_id_source(
        xllm::proto::CORRELATION_ID_SOURCE_UPSTREAM);
  } else {
    correlation.set_global_request_id(generate_valid_id(id_generator));
    correlation.set_global_request_id_source(
        xllm::proto::CORRELATION_ID_SOURCE_SERVICE_GENERATED);
  }

  if (valid_correlation_id(input.trace_id)) {
    correlation.set_trace_id(input.trace_id);
    correlation.set_trace_id_source(
        xllm::proto::CORRELATION_ID_SOURCE_UPSTREAM);
  } else if (auto trace_id = trace_id_from_traceparent(input.traceparent)) {
    correlation.set_trace_id(std::move(*trace_id));
    correlation.set_trace_id_source(
        xllm::proto::CORRELATION_ID_SOURCE_UPSTREAM);
  } else {
    correlation.set_trace_id(generate_valid_id(id_generator));
    correlation.set_trace_id_source(
        xllm::proto::CORRELATION_ID_SOURCE_SERVICE_GENERATED);
  }

  correlation.set_request_uid(generate_request_uid(id_generator));
  correlation.set_attempt_seq(0);
  return correlation;
}

xllm::proto::RequestCorrelation make_request_correlation(
    const RequestCorrelationInput& input) {
  return make_request_correlation(input, [] { return llm::new_uuid_v7(); });
}

}  // namespace xllm_service::observability
