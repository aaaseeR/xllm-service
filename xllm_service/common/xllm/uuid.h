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

#pragma once
#include <absl/random/random.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace xllm_service {
namespace llm {

class ShortUUID {
 public:
  ShortUUID() = default;

  std::string random(size_t len = 0);

 private:
  std::string alphabet_ =
      "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
      "abcdefghijkmnopqrstuvwxyz";
  absl::BitGen gen_;
};

inline constexpr uint64_t kMaxUuidV7UnixTimestampMs = (uint64_t{1} << 48) - 1;

// Encodes RFC 9562 UUIDv7 bytes from a 48-bit Unix millisecond timestamp and
// deterministic entropy. The low 4 bits of entropy[0] plus entropy[1] supply
// rand_a; the low 6 bits of entropy[2] plus entropy[3..9] supply rand_b.
std::optional<std::string> uuid_v7_from_parts(
    uint64_t unix_timestamp_ms,
    const std::array<uint8_t, 10>& entropy);

inline bool is_uuid_v7(const std::string& value) {
  if (value.size() != 36 || value[8] != '-' || value[13] != '-' ||
      value[18] != '-' || value[23] != '-' || value[14] != '7') {
    return false;
  }
  const char variant = value[19];
  if (variant != '8' && variant != '9' && variant != 'a' && variant != 'A' &&
      variant != 'b' && variant != 'B') {
    return false;
  }
  for (size_t index = 0; index < value.size(); ++index) {
    if (index == 8 || index == 13 || index == 18 || index == 23) {
      continue;
    }
    const char character = value[index];
    if (!((character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f') ||
          (character >= 'A' && character <= 'F'))) {
      return false;
    }
  }
  return true;
}

std::string new_uuid_v7();

}  // namespace llm
}  // namespace xllm_service
