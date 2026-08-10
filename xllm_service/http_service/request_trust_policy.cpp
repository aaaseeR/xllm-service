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

#include "http_service/request_trust_policy.h"

#include <glog/logging.h>
#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/rand.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <exception>
#include <limits>
#include <string_view>

namespace xllm_service {
namespace {

constexpr std::string_view kTokenPrefix = "v2.";
constexpr size_t kMaxSessionIdBytes = 64;
constexpr size_t kMaxAuthenticatedClientBytes = 256;
constexpr size_t kMaxClientSessionHintBytes = 256;
constexpr size_t kSha256HexBytes = 64;

bool valid_session_id(const std::string& value) {
  return !value.empty() && value.size() <= kMaxSessionIdBytes &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return std::isalnum(character) != 0 || character == '-' ||
                  character == '_';
         });
}

bool valid_hex_signature(const std::string& value) {
  return value.size() == kSha256HexBytes &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return std::isdigit(character) != 0 ||
                  (character >= 'a' && character <= 'f');
         });
}

std::string hex_encode(const unsigned char* bytes, size_t size) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string result;
  result.reserve(size * 2);
  for (size_t index = 0; index < size; ++index) {
    result.push_back(kHex[bytes[index] >> 4]);
    result.push_back(kHex[bytes[index] & 0x0f]);
  }
  return result;
}

uint64_t unix_seconds() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

bool parse_uint64(const std::string& value, uint64_t* result) {
  if (value.empty() || result == nullptr ||
      !std::all_of(value.begin(), value.end(), [](unsigned char character) {
        return std::isdigit(character) != 0;
      })) {
    return false;
  }
  try {
    size_t consumed = 0;
    const uint64_t parsed = std::stoull(value, &consumed);
    if (consumed != value.size()) {
      return false;
    }
    *result = parsed;
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

bool valid_secret(const std::string& secret) {
  return secret.size() >= KVSessionTokenCodec::kMinSecretBytes &&
         secret.size() <= KVSessionTokenCodec::kMaxSecretBytes;
}

}  // namespace

std::optional<KVSessionTokenCodec> KVSessionTokenCodec::from_secrets(
    std::string active_secret,
    std::string previous_secret,
    uint64_t token_ttl_seconds) {
  if (!valid_secret(active_secret) || token_ttl_seconds == 0 ||
      token_ttl_seconds > kMaxTokenTtlSeconds ||
      (!previous_secret.empty() && !valid_secret(previous_secret))) {
    return std::nullopt;
  }
  return KVSessionTokenCodec(
      std::move(active_secret), std::move(previous_secret), token_ttl_seconds);
}

KVSessionTokenCodec KVSessionTokenCodec::random(uint64_t token_ttl_seconds) {
  CHECK_GT(token_ttl_seconds, 0u);
  CHECK_LE(token_ttl_seconds, kMaxTokenTtlSeconds);
  std::array<unsigned char, kMinSecretBytes> bytes{};
  CHECK_EQ(RAND_bytes(bytes.data(), bytes.size()), 1)
      << "Failed to initialize the KV session signer";
  return KVSessionTokenCodec(
      std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size()),
      /*previous_secret=*/"",
      token_ttl_seconds);
}

std::string KVSessionTokenCodec::signature(const std::string& secret,
                                           const std::string& payload) const {
  if (!valid_secret(secret) ||
      secret.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return "";
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int digest_size = 0;
  if (HMAC(EVP_sha256(),
           secret.data(),
           static_cast<int>(secret.size()),
           reinterpret_cast<const unsigned char*>(payload.data()),
           payload.size(),
           digest.data(),
           &digest_size) == nullptr ||
      digest_size != 32) {
    return "";
  }
  return hex_encode(digest.data(), digest_size);
}

std::string KVSessionTokenCodec::issue(const std::string& session_id) const {
  return issue_at(session_id, unix_seconds());
}

std::string KVSessionTokenCodec::issue_at(const std::string& session_id,
                                          uint64_t issued_at_seconds) const {
  if (!valid_session_id(session_id) || issued_at_seconds == 0) {
    return "";
  }
  const std::string signed_payload = std::string(kTokenPrefix) +
                                     std::to_string(issued_at_seconds) + "." +
                                     session_id;
  const std::string mac = signature(active_secret_, signed_payload);
  if (mac.empty()) {
    return "";
  }
  return signed_payload + "." + mac;
}

std::optional<std::string> KVSessionTokenCodec::verify(
    const std::string& token) const {
  return verify_at(token, unix_seconds());
}

std::optional<std::string> KVSessionTokenCodec::verify_at(
    const std::string& token,
    uint64_t now_seconds) const {
  if (token.size() < kTokenPrefix.size() ||
      token.compare(
          0, kTokenPrefix.size(), kTokenPrefix.data(), kTokenPrefix.size()) !=
          0) {
    return std::nullopt;
  }
  const size_t issued_at_separator = token.find('.', kTokenPrefix.size());
  if (issued_at_separator == std::string::npos) {
    return std::nullopt;
  }
  const size_t session_separator = token.find('.', issued_at_separator + 1);
  if (session_separator == std::string::npos) {
    return std::nullopt;
  }
  const std::string issued_at_value = token.substr(
      kTokenPrefix.size(), issued_at_separator - kTokenPrefix.size());
  const std::string session_id = token.substr(
      issued_at_separator + 1, session_separator - issued_at_separator - 1);
  const std::string provided = token.substr(session_separator + 1);
  uint64_t issued_at_seconds = 0;
  if (!parse_uint64(issued_at_value, &issued_at_seconds) ||
      !valid_session_id(session_id) || !valid_hex_signature(provided) ||
      (issued_at_seconds > now_seconds &&
       issued_at_seconds - now_seconds > kMaxClockSkewSeconds) ||
      (now_seconds > issued_at_seconds &&
       now_seconds - issued_at_seconds > token_ttl_seconds_)) {
    return std::nullopt;
  }
  const std::string signed_payload = token.substr(0, session_separator);
  const std::string expected = signature(active_secret_, signed_payload);
  bool matches =
      expected.size() == provided.size() &&
      CRYPTO_memcmp(expected.data(), provided.data(), expected.size()) == 0;
  if (!matches && !previous_secret_.empty()) {
    const std::string previous = signature(previous_secret_, signed_payload);
    matches =
        previous.size() == provided.size() &&
        CRYPTO_memcmp(previous.data(), provided.data(), previous.size()) == 0;
  }
  if (!matches) {
    return std::nullopt;
  }
  return session_id;
}

std::optional<std::string> KVSessionTokenCodec::derive_client_session(
    const std::string& authenticated_client,
    const std::string& client_session_hint) const {
  if (authenticated_client.empty() ||
      authenticated_client.size() > kMaxAuthenticatedClientBytes ||
      client_session_hint.empty() ||
      client_session_hint.size() > kMaxClientSessionHintBytes) {
    return std::nullopt;
  }
  const std::string payload =
      "sdk-session-v1\n" + std::to_string(authenticated_client.size()) + "\n" +
      authenticated_client + "\n" + std::to_string(client_session_hint.size()) +
      "\n" + client_session_hint;
  const std::string session = signature(active_secret_, payload);
  return session.empty() ? std::nullopt : std::optional<std::string>(session);
}

std::optional<RequestTrustDecision> decide_request_trust(
    const RequestTrustInput& input,
    const KVSessionTokenCodec& session_codec) {
  if (!valid_session_id(input.request_uid)) {
    return std::nullopt;
  }

  RequestTrustDecision decision;
  const bool trusted_tenant =
      input.trusted_tenant_headers_enabled && !input.tenant_header.empty();
  if (trusted_tenant) {
    decision.tenant_id = input.tenant_header;
    decision.flow_id =
        input.flow_header.empty() ? input.tenant_header : input.flow_header;
    decision.kv_isolation_domain = input.tenant_header;
    decision.kv_isolation_reusable = true;
    decision.effective_priority = input.requested_priority.value_or(0);
    return decision;
  }

  decision.tenant_id = "anonymous";
  decision.flow_id = "anonymous";
  decision.effective_priority = 0;
  if (!input.kv_session_token.empty()) {
    const std::optional<std::string> verified_session =
        session_codec.verify(input.kv_session_token);
    if (!verified_session.has_value()) {
      return std::nullopt;
    }
    decision.kv_isolation_domain = *verified_session;
    decision.kv_isolation_reusable = true;
    return decision;
  }
  const std::optional<std::string> client_session =
      input.trusted_client_identity_headers_enabled
          ? session_codec.derive_client_session(input.authenticated_client,
                                                input.client_session_hint)
          : std::nullopt;
  if (client_session.has_value()) {
    decision.kv_isolation_domain = *client_session;
    decision.kv_isolation_reusable = true;
    return decision;
  }
  decision.kv_isolation_domain = input.request_uid;
  decision.kv_isolation_reusable = true;
  decision.issued_kv_session_token =
      session_codec.issue(decision.kv_isolation_domain);
  if (decision.issued_kv_session_token.empty()) {
    return std::nullopt;
  }
  return decision;
}

}  // namespace xllm_service
