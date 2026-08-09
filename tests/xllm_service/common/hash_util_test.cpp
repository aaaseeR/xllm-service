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

#include "common/hash_util.h"

#include <gtest/gtest.h>

#include <array>
#include <numeric>
#include <string>
#include <vector>

namespace xllm_service {
namespace {

TEST(HashUtilTest, CanonicalNamespaceMatchesEngineGolden) {
  const std::string kv_namespace =
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  std::vector<int32_t> tokens(32);
  std::iota(tokens.begin(), tokens.end(), 0);
  const std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN> first_golden = {0x0f,
                                                                         0x25,
                                                                         0xfd,
                                                                         0x58,
                                                                         0xe8,
                                                                         0x7f,
                                                                         0xcf,
                                                                         0x41,
                                                                         0x01,
                                                                         0xe0,
                                                                         0xa7,
                                                                         0xd3,
                                                                         0x2b,
                                                                         0xd0,
                                                                         0x89,
                                                                         0xfa};
  const std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN> second_golden = {0x22,
                                                                          0x22,
                                                                          0x5f,
                                                                          0x22,
                                                                          0xbb,
                                                                          0xea,
                                                                          0x94,
                                                                          0x1b,
                                                                          0x15,
                                                                          0xe6,
                                                                          0x4c,
                                                                          0x1a,
                                                                          0xfe,
                                                                          0x9b,
                                                                          0x6f,
                                                                          0x04};
  std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN> first{};
  std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN> second{};
  const Slice<int32_t> token_slice(tokens);

  xxh3_128bits_hash(kv_namespace,
                    /*hash_seed=*/1024,
                    nullptr,
                    token_slice.slice(0, 16),
                    /*block_extra=*/{},
                    first.data());
  xxh3_128bits_hash(kv_namespace,
                    /*hash_seed=*/1024,
                    first.data(),
                    token_slice.slice(16, 32),
                    /*block_extra=*/{},
                    second.data());

  EXPECT_EQ(first, first_golden);
  EXPECT_EQ(second, second_golden);

  std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN> isolated{};
  xxh3_128bits_hash("different-isolation-domain",
                    /*hash_seed=*/1024,
                    nullptr,
                    token_slice.slice(0, 16),
                    /*block_extra=*/{},
                    isolated.data());
  EXPECT_NE(isolated, first);
}

TEST(HashUtilTest, RequestNamespaceStrictlyIsolatesTenantAndAdapter) {
  const std::string tenant_a =
      derive_request_kv_namespace("base", "tenant-a", "");
  const std::string tenant_a_repeat =
      derive_request_kv_namespace("base", "tenant-a", "");
  ASSERT_FALSE(tenant_a.empty());
  EXPECT_EQ(tenant_a, tenant_a_repeat);
  EXPECT_NE(tenant_a, derive_request_kv_namespace("base", "tenant-b", ""));
  EXPECT_NE(tenant_a,
            derive_request_kv_namespace("base", "tenant-a", "lora-a"));
  EXPECT_NE(tenant_a, derive_request_kv_namespace("other", "tenant-a", ""));
  EXPECT_TRUE(derive_request_kv_namespace("", "tenant-a", "").empty());
  EXPECT_TRUE(
      derive_request_kv_namespace("base", std::string(257, 'x'), "").empty());
}

}  // namespace
}  // namespace xllm_service
