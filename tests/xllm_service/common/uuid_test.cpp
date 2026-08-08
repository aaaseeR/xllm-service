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

#include "common/xllm/uuid.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <regex>
#include <string>
#include <unordered_set>

namespace xllm_service::llm {
namespace {

TEST(UuidV7Test, DeterministicPartsMatchRfcLayout) {
  constexpr uint64_t kTimestampMs = 0x0123456789ab;
  const std::array<uint8_t, 10> entropy = {
      0x0c, 0xde, 0x3f, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd};

  const auto uuid = uuid_v7_from_parts(kTimestampMs, entropy);
  ASSERT_TRUE(uuid.has_value());
  EXPECT_EQ(*uuid, "01234567-89ab-7cde-bf01-23456789abcd");
}

TEST(UuidV7Test, RejectsTimestampOutside48BitField) {
  const std::array<uint8_t, 10> entropy{};
  EXPECT_FALSE(
      uuid_v7_from_parts(kMaxUuidV7UnixTimestampMs + 1, entropy).has_value());
}

TEST(UuidV7Test, ValidationRejectsWrongVersionVariantAndShape) {
  EXPECT_TRUE(is_uuid_v7("01234567-89ab-7cde-bf01-23456789abcd"));
  EXPECT_FALSE(is_uuid_v7("01234567-89ab-6cde-bf01-23456789abcd"));
  EXPECT_FALSE(is_uuid_v7("01234567-89ab-7cde-7f01-23456789abcd"));
  EXPECT_FALSE(is_uuid_v7("01234567-89ab-7cde-bf01-23456789abcg"));
  EXPECT_FALSE(is_uuid_v7(""));
}

TEST(UuidV7Test, LexicalOrderFollowsTimestamp) {
  std::array<uint8_t, 10> low_entropy{};
  std::array<uint8_t, 10> high_entropy{};
  high_entropy.fill(0xff);

  const auto earlier = uuid_v7_from_parts(100, high_entropy);
  const auto later = uuid_v7_from_parts(101, low_entropy);
  ASSERT_TRUE(earlier.has_value());
  ASSERT_TRUE(later.has_value());
  EXPECT_LT(*earlier, *later);
}

TEST(UuidV7Test, GeneratedValuesHaveVersionVariantAndAreUnique) {
  const std::regex uuid_v7_pattern(
      "^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-"
      "[0-9a-f]{12}$");
  std::unordered_set<std::string> values;
  constexpr size_t kSampleCount = 4096;
  for (size_t i = 0; i < kSampleCount; ++i) {
    std::string uuid = new_uuid_v7();
    EXPECT_TRUE(std::regex_match(uuid, uuid_v7_pattern));
    EXPECT_TRUE(is_uuid_v7(uuid));
    values.insert(std::move(uuid));
  }
  EXPECT_EQ(values.size(), kSampleCount);
}

}  // namespace
}  // namespace xllm_service::llm
