/* Copyright 2025-2026 The xLLM Authors.

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

#include <MurmurHash3.h>
#include <assert.h>
#include <glog/logging.h>
#include <xxhash.h>

#include <array>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string_view>
#include <thread>

#include "common/global_gflags.h"

namespace xllm_service {
namespace {

constexpr std::string_view kCanonicalHashDomain = "xkvh-v1";
constexpr std::string_view kRequestNamespaceDomain = "xkvns-request-v2";
constexpr size_t kMaxNamespaceComponentLength = 256;
constexpr uint64_t kRequestNamespaceSeed = 0xd0c4b10c5e771a2bULL;

void append_u32_le(uint32_t value, std::vector<uint8_t>* output) {
  output->push_back(static_cast<uint8_t>(value));
  output->push_back(static_cast<uint8_t>(value >> 8));
  output->push_back(static_cast<uint8_t>(value >> 16));
  output->push_back(static_cast<uint8_t>(value >> 24));
}

void write_u64_le(uint64_t value, uint8_t* output) {
  for (size_t index = 0; index < sizeof(value); ++index) {
    output[index] = static_cast<uint8_t>(value >> (index * 8));
  }
}

void append_bytes(std::string_view value, std::vector<uint8_t>* output) {
  append_u32_le(static_cast<uint32_t>(value.size()), output);
  output->insert(output->end(), value.begin(), value.end());
}

}  // namespace

void xxh3_128bits_hash(const uint8_t* pre_hash_value,
                       const Slice<int32_t>& token_ids,
                       uint8_t* hash_value) {
  if (pre_hash_value == nullptr) {
    XXH128_hash_t xxh3_128bits_hash_value =
        XXH3_128bits_withSeed(reinterpret_cast<const void*>(token_ids.data()),
                              sizeof(int32_t) * token_ids.size(),
                              FLAGS_xxh3_128bits_seed);
    memcpy(
        hash_value, &xxh3_128bits_hash_value, sizeof(xxh3_128bits_hash_value));
  } else {
    const size_t data_len =
        sizeof(int32_t) * token_ids.size() + XXH3_128BITS_HASH_VALUE_LEN;
    std::vector<uint8_t> key(data_len);
    memcpy(key.data(), pre_hash_value, XXH3_128BITS_HASH_VALUE_LEN);
    memcpy(key.data() + XXH3_128BITS_HASH_VALUE_LEN,
           reinterpret_cast<const void*>(token_ids.data()),
           sizeof(int32_t) * token_ids.size());

    XXH128_hash_t xxh3_128bits_hash_value =
        XXH3_128bits_withSeed(reinterpret_cast<const void*>(key.data()),
                              data_len,
                              FLAGS_xxh3_128bits_seed);
    memcpy(
        hash_value, &xxh3_128bits_hash_value, sizeof(xxh3_128bits_hash_value));
  }
}

void xxh3_128bits_hash(std::string_view kv_namespace,
                       uint64_t hash_seed,
                       const uint8_t* pre_hash_value,
                       const Slice<int32_t>& token_ids,
                       std::string_view block_extra,
                       uint8_t* hash_value) {
  if (kv_namespace.empty()) {
    xxh3_128bits_hash(pre_hash_value, token_ids, hash_value);
    return;
  }
  CHECK_LE(kv_namespace.size(), std::numeric_limits<uint32_t>::max());
  CHECK_LE(token_ids.size(), std::numeric_limits<uint32_t>::max());
  CHECK_LE(block_extra.size(), std::numeric_limits<uint32_t>::max());

  std::vector<uint8_t> preimage;
  preimage.reserve(kCanonicalHashDomain.size() + kv_namespace.size() +
                   block_extra.size() + token_ids.size() * sizeof(int32_t) +
                   XXH3_128BITS_HASH_VALUE_LEN + 16);
  preimage.insert(
      preimage.end(), kCanonicalHashDomain.begin(), kCanonicalHashDomain.end());
  append_bytes(kv_namespace, &preimage);
  preimage.push_back(pre_hash_value == nullptr ? 0 : 1);
  if (pre_hash_value != nullptr) {
    preimage.insert(preimage.end(),
                    pre_hash_value,
                    pre_hash_value + XXH3_128BITS_HASH_VALUE_LEN);
  }
  append_u32_le(static_cast<uint32_t>(token_ids.size()), &preimage);
  for (size_t index = 0; index < token_ids.size(); ++index) {
    append_u32_le(static_cast<uint32_t>(token_ids[index]), &preimage);
  }
  append_bytes(block_extra, &preimage);

  const XXH128_hash_t digest =
      XXH3_128bits_withSeed(preimage.data(), preimage.size(), hash_seed);
  write_u64_le(digest.low64, hash_value);
  write_u64_le(digest.high64, hash_value + sizeof(digest.low64));
}

std::string derive_request_kv_namespace(std::string_view base_namespace,
                                        std::string_view isolation_domain) {
  if (base_namespace.empty() || isolation_domain.empty() ||
      base_namespace.size() > kMaxNamespaceComponentLength ||
      isolation_domain.size() > kMaxNamespaceComponentLength) {
    return "";
  }
  std::vector<uint8_t> preimage;
  preimage.reserve(kRequestNamespaceDomain.size() + base_namespace.size() +
                   isolation_domain.size() + 8);
  preimage.insert(preimage.end(),
                  kRequestNamespaceDomain.begin(),
                  kRequestNamespaceDomain.end());
  append_bytes(base_namespace, &preimage);
  append_bytes(isolation_domain, &preimage);
  const XXH128_hash_t digest = XXH3_128bits_withSeed(
      preimage.data(), preimage.size(), kRequestNamespaceSeed);
  std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN> bytes{};
  write_u64_le(digest.low64, bytes.data());
  write_u64_le(digest.high64, bytes.data() + sizeof(digest.low64));
  constexpr char kHex[] = "0123456789abcdef";
  std::string output = "xkvns-request-v2:";
  output.reserve(output.size() + bytes.size() * 2);
  for (const uint8_t byte : bytes) {
    output.push_back(kHex[byte >> 4]);
    output.push_back(kHex[byte & 0x0f]);
  }
  return output;
}

void print_hex_array(uint8_t* array) {
  for (size_t i = 0; i < XXH3_128BITS_HASH_VALUE_LEN; ++i) {
    unsigned char uc = static_cast<unsigned char>(array[i]);
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(uc);

    if (i % XXH3_128BITS_HASH_VALUE_LEN == XXH3_128BITS_HASH_VALUE_LEN - 1) {
      std::cout << std::endl;
    }

    else {
      std::cout << " ";
    }
  }
  std::cout << std::dec << std::endl;
}

}  // namespace xllm_service
