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

#include "routing/pd_compatibility.h"

#include <gtest/gtest.h>

namespace xllm_service {
namespace {

InstanceMetaInfo make_endpoint(const std::string& name, InstanceType type) {
  InstanceMetaInfo endpoint;
  endpoint.name = name;
  endpoint.type = type;
  endpoint.block_size = 128;
  endpoint.xxh3_128bits_seed = 1024;
  endpoint.kv_split_size = 1;
  return endpoint;
}

TEST(PdCompatibilityTest, AcceptsCompatibleDedicatedAndMixRoles) {
  InstanceMetaInfo prefill =
      make_endpoint("prefill:8000", InstanceType::PREFILL);
  InstanceMetaInfo decode = make_endpoint("decode:8000", InstanceType::DECODE);
  EXPECT_TRUE(pd_compatibility_result_ok(
      validate_external_pd_compatibility(prefill, decode, 128, 1024)));

  prefill.type = InstanceType::MIX;
  prefill.current_type = InstanceType::PREFILL;
  decode.type = InstanceType::MIX;
  decode.current_type = InstanceType::DECODE;
  EXPECT_TRUE(pd_compatibility_result_ok(
      validate_external_pd_compatibility(prefill, decode, 128, 1024)));
}

TEST(PdCompatibilityTest, RejectsWrongRolesAndSameEndpoint) {
  InstanceMetaInfo prefill =
      make_endpoint("prefill:8000", InstanceType::DECODE);
  InstanceMetaInfo decode = make_endpoint("decode:8000", InstanceType::DECODE);
  EXPECT_EQ(
      validate_external_pd_compatibility(prefill, decode, 128, 1024).error,
      PdCompatibilityError::INVALID_PREFILL_ROLE);

  prefill = make_endpoint("decode:8000", InstanceType::PREFILL);
  EXPECT_EQ(
      validate_external_pd_compatibility(prefill, decode, 128, 1024).error,
      PdCompatibilityError::SAME_ENDPOINT);
}

TEST(PdCompatibilityTest, RejectsIncompleteOrMismatchedCacheContracts) {
  InstanceMetaInfo prefill =
      make_endpoint("prefill:8000", InstanceType::PREFILL);
  InstanceMetaInfo decode = make_endpoint("decode:8000", InstanceType::DECODE);

  decode.block_size = 0;
  EXPECT_EQ(
      validate_external_pd_compatibility(prefill, decode, 128, 1024).error,
      PdCompatibilityError::MISSING_CACHE_CONTRACT);

  decode = make_endpoint("decode:8000", InstanceType::DECODE);
  decode.block_size = 64;
  EXPECT_EQ(
      validate_external_pd_compatibility(prefill, decode, 128, 1024).error,
      PdCompatibilityError::BLOCK_SIZE_MISMATCH);

  decode = make_endpoint("decode:8000", InstanceType::DECODE);
  decode.xxh3_128bits_seed = 2048;
  EXPECT_EQ(
      validate_external_pd_compatibility(prefill, decode, 128, 1024).error,
      PdCompatibilityError::HASH_SEED_MISMATCH);

  decode = make_endpoint("decode:8000", InstanceType::DECODE);
  decode.kv_split_size = 2;
  EXPECT_EQ(
      validate_external_pd_compatibility(prefill, decode, 128, 1024).error,
      PdCompatibilityError::KV_SPLIT_SIZE_MISMATCH);
}

}  // namespace
}  // namespace xllm_service
