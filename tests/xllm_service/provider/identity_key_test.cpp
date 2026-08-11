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

#include "provider/identity_key.h"

#include <google/protobuf/unknown_field_set.h>
#include <gtest/gtest.h>

namespace xllm_service::provider {
namespace {

void add_unknown_varint(google::protobuf::Message* message,
                        int field_number,
                        uint64_t value) {
  message->GetReflection()->MutableUnknownFields(message)->AddVarint(
      field_number, value);
}

xllm::proto::ProviderEngineKey engine_identity() {
  xllm::proto::ProviderEngineKey identity;
  identity.set_provider_id(xllm::proto::PROVIDER_ID_XLLM_NATIVE);
  identity.set_profile_digest("profile");
  identity.set_engine_uid("engine");
  identity.set_incarnation_id("incarnation");
  return identity;
}

xllm::proto::KVStreamIdentity stream_identity() {
  xllm::proto::KVStreamIdentity identity;
  *identity.mutable_engine() = engine_identity();
  identity.set_model_revision("model");
  identity.set_kv_namespace("namespace");
  identity.set_cache_epoch(7);
  return identity;
}

TEST(IdentityKeyTest, IgnoresUnknownFieldsButNotKnownIdentityFields) {
  const xllm::proto::ProviderEngineKey local = engine_identity();
  xllm::proto::ProviderEngineKey future = local;
  add_unknown_varint(&future, 100, 42);
  ASSERT_NE(local.SerializeAsString(), future.SerializeAsString());
  EXPECT_TRUE(same_provider_engine_identity(local, future));
  EXPECT_EQ(provider_engine_identity_key(local),
            provider_engine_identity_key(future));

  future.set_incarnation_id("replacement");
  EXPECT_FALSE(same_provider_engine_identity(local, future));
  EXPECT_NE(provider_engine_identity_key(local),
            provider_engine_identity_key(future));
}

TEST(IdentityKeyTest, StreamAndBlockKeysUseOnlyExplicitContractFields) {
  const xllm::proto::KVStreamIdentity local_stream = stream_identity();
  xllm::proto::KVStreamIdentity future_stream = local_stream;
  add_unknown_varint(&future_stream, 101, 1);
  add_unknown_varint(future_stream.mutable_engine(), 102, 2);
  EXPECT_TRUE(same_kv_stream_identity(local_stream, future_stream));
  EXPECT_EQ(kv_stream_identity_key(local_stream),
            kv_stream_identity_key(future_stream));

  xllm::proto::KVBlockEntry local_block;
  local_block.set_block_hash(std::string(16, 'h'));
  local_block.set_parent_hash(std::string(16, 'p'));
  local_block.set_token_begin(1);
  local_block.set_token_end(17);
  local_block.set_cache_group("full-attention");
  local_block.set_tier(xllm::proto::KV_CACHE_TIER_HBM);
  local_block.set_dp_rank(2);
  local_block.set_device_rank(3);
  xllm::proto::KVBlockEntry future_block = local_block;
  add_unknown_varint(&future_block, 103, 3);
  EXPECT_EQ(kv_block_identity_key(local_block),
            kv_block_identity_key(future_block));
}

}  // namespace
}  // namespace xllm_service::provider
