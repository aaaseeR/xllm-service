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

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <tuple>

namespace xllm_service::provider {
namespace {

void append_u64(uint64_t value, std::string* output) {
  for (size_t index = 0; index < sizeof(value); ++index) {
    output->push_back(static_cast<char>(value >> (index * 8)));
  }
}

void append_component(std::string_view value, std::string* output) {
  append_u64(value.size(), output);
  output->append(value.data(), value.size());
}

void append_engine_identity(const xllm::proto::ProviderEngineKey& identity,
                            std::string* output) {
  append_u64(static_cast<uint64_t>(identity.provider_id()), output);
  append_component(identity.profile_digest(), output);
  append_component(identity.engine_uid(), output);
  append_component(identity.incarnation_id(), output);
}

}  // namespace

bool same_provider_engine_identity(
    const xllm::proto::ProviderEngineKey& left,
    const xllm::proto::ProviderEngineKey& right) {
  return left.provider_id() == right.provider_id() &&
         left.profile_digest() == right.profile_digest() &&
         left.engine_uid() == right.engine_uid() &&
         left.incarnation_id() == right.incarnation_id();
}

std::string provider_engine_identity_key(
    const xllm::proto::ProviderEngineKey& identity) {
  std::string key = "xllm-provider-engine-v1";
  append_engine_identity(identity, &key);
  return key;
}

std::string provider_link_identity_key(
    const xllm::proto::ProviderEngineKey& prefill,
    const xllm::proto::ProviderEngineKey& decode) {
  std::string key = "xllm-provider-link-v1";
  append_engine_identity(prefill, &key);
  append_engine_identity(decode, &key);
  return key;
}

bool same_kv_stream_identity(const xllm::proto::KVStreamIdentity& left,
                             const xllm::proto::KVStreamIdentity& right,
                             bool include_cache_epoch) {
  return same_provider_engine_identity(left.engine(), right.engine()) &&
         left.model_revision() == right.model_revision() &&
         left.kv_namespace() == right.kv_namespace() &&
         (!include_cache_epoch || left.cache_epoch() == right.cache_epoch());
}

std::string kv_stream_identity_key(
    const xllm::proto::KVStreamIdentity& identity,
    bool include_cache_epoch) {
  std::string key =
      include_cache_epoch ? "xllm-kv-stream-v1" : "xllm-kv-stream-domain-v1";
  append_engine_identity(identity.engine(), &key);
  append_component(identity.model_revision(), &key);
  append_component(identity.kv_namespace(), &key);
  if (include_cache_epoch) {
    append_u64(identity.cache_epoch(), &key);
  }
  return key;
}

bool kv_stream_identity_less(const xllm::proto::KVStreamIdentity& left,
                             const xllm::proto::KVStreamIdentity& right) {
  return std::forward_as_tuple(left.engine().provider_id(),
                               left.engine().profile_digest(),
                               left.engine().engine_uid(),
                               left.engine().incarnation_id(),
                               left.model_revision(),
                               left.kv_namespace(),
                               left.cache_epoch()) <
         std::forward_as_tuple(right.engine().provider_id(),
                               right.engine().profile_digest(),
                               right.engine().engine_uid(),
                               right.engine().incarnation_id(),
                               right.model_revision(),
                               right.kv_namespace(),
                               right.cache_epoch());
}

std::string kv_block_identity_key(const xllm::proto::KVBlockEntry& entry) {
  std::string key = "xllm-kv-block-v1";
  append_component(entry.block_hash(), &key);
  append_component(entry.parent_hash(), &key);
  append_u64(entry.token_begin(), &key);
  append_u64(entry.token_end(), &key);
  append_component(entry.cache_group(), &key);
  append_u64(static_cast<uint64_t>(entry.tier()), &key);
  append_u64(entry.dp_rank(), &key);
  append_u64(entry.device_rank(), &key);
  return key;
}

}  // namespace xllm_service::provider
