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

#include <string>

#include "provider.pb.h"

namespace xllm_service::provider {

// Identity is defined by explicit contract fields, never protobuf wire bytes.
// Unknown fields must remain forward-compatible during rolling upgrades.
bool same_provider_engine_identity(const xllm::proto::ProviderEngineKey& left,
                                   const xllm::proto::ProviderEngineKey& right);

std::string provider_engine_identity_key(
    const xllm::proto::ProviderEngineKey& identity);
std::string provider_link_identity_key(
    const xllm::proto::ProviderEngineKey& prefill,
    const xllm::proto::ProviderEngineKey& decode);

bool same_kv_stream_identity(const xllm::proto::KVStreamIdentity& left,
                             const xllm::proto::KVStreamIdentity& right,
                             bool include_cache_epoch = true);
std::string kv_stream_identity_key(
    const xllm::proto::KVStreamIdentity& identity,
    bool include_cache_epoch = true);
bool kv_stream_identity_less(const xllm::proto::KVStreamIdentity& left,
                             const xllm::proto::KVStreamIdentity& right);

std::string kv_block_identity_key(const xllm::proto::KVBlockEntry& entry);

}  // namespace xllm_service::provider
