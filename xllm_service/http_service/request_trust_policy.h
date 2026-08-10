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

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace xllm_service {

// Stateless signer for the opaque KV session token returned by Service. A
// shared secret lets any Service replica verify the token; an instance-local
// random secret remains safe but only preserves reuse on a sticky replica.
class KVSessionTokenCodec final {
 public:
  static constexpr size_t kMinSecretBytes = 32;
  static constexpr size_t kMaxSecretBytes = 256;
  static constexpr uint64_t kMaxClockSkewSeconds = 60;
  static constexpr uint64_t kMaxTokenTtlSeconds = 30 * 24 * 60 * 60;

  static std::optional<KVSessionTokenCodec> from_secrets(
      std::string active_secret,
      std::string previous_secret,
      uint64_t token_ttl_seconds);
  static KVSessionTokenCodec random(uint64_t token_ttl_seconds);

  std::string issue(const std::string& session_id) const;
  std::string issue_at(const std::string& session_id,
                       uint64_t issued_at_seconds) const;
  std::optional<std::string> verify(const std::string& token) const;
  std::optional<std::string> verify_at(const std::string& token,
                                       uint64_t now_seconds) const;
  std::optional<std::string> derive_client_session(
      const std::string& authenticated_client,
      const std::string& client_session_hint) const;

 private:
  KVSessionTokenCodec(std::string active_secret,
                      std::string previous_secret,
                      uint64_t token_ttl_seconds)
      : active_secret_(std::move(active_secret)),
        previous_secret_(std::move(previous_secret)),
        token_ttl_seconds_(token_ttl_seconds) {}

  std::string signature(const std::string& secret,
                        const std::string& payload) const;

  std::string active_secret_;
  std::string previous_secret_;
  uint64_t token_ttl_seconds_ = 0;
};

struct RequestTrustInput {
  bool trusted_tenant_headers_enabled = false;
  bool trusted_client_identity_headers_enabled = false;
  std::string tenant_header;
  std::string flow_header;
  std::string kv_session_token;
  std::string authenticated_client;
  std::string client_session_hint;
  std::string request_uid;
  std::optional<int32_t> requested_priority;
};

struct RequestTrustDecision {
  std::string tenant_id;
  std::string flow_id;
  std::string kv_isolation_domain;
  bool kv_isolation_reusable = false;
  int32_t effective_priority = 0;
  std::string issued_kv_session_token;
};

// Tenant/flow/priority and the authenticated client identity are independent
// Gateway trust assertions. Without either trust, Service issues or verifies
// an opaque signed session so repeated prompts can reuse Engine prefix cache
// without accepting a forgeable identity from the client.
std::optional<RequestTrustDecision> decide_request_trust(
    const RequestTrustInput& input,
    const KVSessionTokenCodec& session_codec);

}  // namespace xllm_service
