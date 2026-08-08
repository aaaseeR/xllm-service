/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "uuid.h"

#include <absl/random/distributions.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace xllm_service {
namespace llm {

namespace {

char hex_digit(uint8_t value) {
  constexpr char kHexDigits[] = "0123456789abcdef";
  return kHexDigits[value & 0x0f];
}

}  // namespace

std::string ShortUUID::random(size_t len) {
  if (len == 0) {
    len = 22;
  }

  std::string uuid(len, ' ');
  for (size_t i = 0; i < len; i++) {
    const size_t rand = absl::Uniform<size_t>(
        absl::IntervalClosedOpen, gen_, 0, alphabet_.size());
    uuid[i] = alphabet_[rand];
  }
  return uuid;
}

std::optional<std::string> uuid_v7_from_parts(
    uint64_t unix_timestamp_ms,
    const std::array<uint8_t, 10>& entropy) {
  if (unix_timestamp_ms > kMaxUuidV7UnixTimestampMs) {
    return std::nullopt;
  }

  std::array<uint8_t, 16> bytes{};
  for (size_t i = 0; i < 6; ++i) {
    bytes[5 - i] = static_cast<uint8_t>(unix_timestamp_ms & 0xff);
    unix_timestamp_ms >>= 8;
  }
  bytes[6] = static_cast<uint8_t>(0x70 | (entropy[0] & 0x0f));
  bytes[7] = entropy[1];
  bytes[8] = static_cast<uint8_t>(0x80 | (entropy[2] & 0x3f));
  std::copy(entropy.begin() + 3, entropy.end(), bytes.begin() + 9);

  std::string uuid(36, '-');
  size_t output_index = 0;
  for (size_t byte_index = 0; byte_index < bytes.size(); ++byte_index) {
    if (output_index == 8 || output_index == 13 || output_index == 18 ||
        output_index == 23) {
      ++output_index;
    }
    uuid[output_index++] = hex_digit(bytes[byte_index] >> 4);
    uuid[output_index++] = hex_digit(bytes[byte_index]);
  }
  return uuid;
}

std::string new_uuid_v7() {
  const int64_t timestamp_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  const uint64_t bounded_timestamp_ms = std::min(
      timestamp_ms < 0 ? uint64_t{0} : static_cast<uint64_t>(timestamp_ms),
      kMaxUuidV7UnixTimestampMs);

  thread_local absl::BitGen bit_generator;
  std::array<uint8_t, 10> entropy{};
  for (uint8_t& byte : entropy) {
    byte = static_cast<uint8_t>(
        absl::Uniform<int>(absl::IntervalClosedOpen, bit_generator, 0, 256));
  }
  return *uuid_v7_from_parts(bounded_timestamp_ms, entropy);
}

}  // namespace llm
}  // namespace xllm_service
