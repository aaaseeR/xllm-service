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

#include <gtest/gtest.h>

#include <string>

namespace xllm_service {
namespace {

constexpr char kRequestUid[] = "018f47b2-c198-7cc8-98d7-503f58e2a612";

KVSessionTokenCodec codec() {
  return *KVSessionTokenCodec::from_secrets(std::string(32, 's'),
                                            /*previous_secret=*/"",
                                            /*token_ttl_seconds=*/3600);
}

TEST(RequestTrustPolicyTest, AnonymousSessionIsSignedAndReusable) {
  const KVSessionTokenCodec signer = codec();
  const auto first = decide_request_trust(
      RequestTrustInput{
          .kv_session_token = "",
          .request_uid = kRequestUid,
          .requested_priority = 1,
      },
      signer);
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->tenant_id, "anonymous");
  EXPECT_EQ(first->flow_id, "anonymous");
  EXPECT_EQ(first->kv_isolation_domain, kRequestUid);
  EXPECT_TRUE(first->kv_isolation_reusable);
  EXPECT_EQ(first->effective_priority, 0);
  ASSERT_FALSE(first->issued_kv_session_token.empty());

  const auto next = decide_request_trust(
      RequestTrustInput{
          .kv_session_token = first->issued_kv_session_token,
          .request_uid = "018f47b2-c198-7cc8-98d7-503f58e2a613",
          .requested_priority = 1,
      },
      signer);
  ASSERT_TRUE(next.has_value());
  EXPECT_EQ(next->kv_isolation_domain, first->kv_isolation_domain);
  EXPECT_TRUE(next->kv_isolation_reusable);
  EXPECT_EQ(next->effective_priority, 0);
  EXPECT_TRUE(next->issued_kv_session_token.empty());
}

TEST(RequestTrustPolicyTest, ForgedSessionCannotSelectAnIsolationDomain) {
  const KVSessionTokenCodec signer = codec();
  const std::string valid = signer.issue(kRequestUid);
  ASSERT_FALSE(valid.empty());
  std::string forged = valid;
  forged.back() = forged.back() == '0' ? '1' : '0';
  EXPECT_FALSE(signer.verify(forged).has_value());

  const auto decision = decide_request_trust(
      RequestTrustInput{
          .kv_session_token = forged,
          .request_uid = "018f47b2-c198-7cc8-98d7-503f58e2a614",
      },
      signer);
  EXPECT_FALSE(decision.has_value());
}

TEST(RequestTrustPolicyTest, PriorityRequiresTrustedTenantIdentity) {
  const auto untrusted = decide_request_trust(
      RequestTrustInput{
          .trusted_tenant_headers_enabled = false,
          .tenant_header = "forged-tenant",
          .flow_header = "forged-flow",
          .request_uid = kRequestUid,
          .requested_priority = 1,
      },
      codec());
  ASSERT_TRUE(untrusted.has_value());
  EXPECT_EQ(untrusted->tenant_id, "anonymous");
  EXPECT_EQ(untrusted->effective_priority, 0);

  const auto trusted = decide_request_trust(
      RequestTrustInput{
          .trusted_tenant_headers_enabled = true,
          .tenant_header = "tenant-a",
          .flow_header = "flow-a",
          .request_uid = kRequestUid,
          .requested_priority = 1,
      },
      codec());
  ASSERT_TRUE(trusted.has_value());
  EXPECT_EQ(trusted->tenant_id, "tenant-a");
  EXPECT_EQ(trusted->flow_id, "flow-a");
  EXPECT_EQ(trusted->kv_isolation_domain, "tenant-a");
  EXPECT_EQ(trusted->effective_priority, 1);
  EXPECT_TRUE(trusted->issued_kv_session_token.empty());
}

TEST(RequestTrustPolicyTest, SecretAndTokenBoundariesFailClosed) {
  EXPECT_FALSE(KVSessionTokenCodec::from_secrets(std::string(31, 's'),
                                                 "",
                                                 /*token_ttl_seconds=*/3600)
                   .has_value());
  EXPECT_FALSE(KVSessionTokenCodec::from_secrets(std::string(257, 's'),
                                                 "",
                                                 /*token_ttl_seconds=*/3600)
                   .has_value());
  EXPECT_FALSE(KVSessionTokenCodec::from_secrets(std::string(32, 's'),
                                                 "",
                                                 /*token_ttl_seconds=*/0)
                   .has_value());
  EXPECT_FALSE(KVSessionTokenCodec::from_secrets(
                   std::string(32, 's'),
                   "",
                   KVSessionTokenCodec::kMaxTokenTtlSeconds + 1)
                   .has_value());
  const KVSessionTokenCodec signer = codec();
  EXPECT_TRUE(signer.issue("").empty());
  EXPECT_FALSE(signer.verify("v2.bad").has_value());
}

TEST(RequestTrustPolicyTest, SessionTokenExpiresAndSupportsKeyRotation) {
  constexpr uint64_t kIssuedAt = 1000;
  const KVSessionTokenCodec old_signer =
      *KVSessionTokenCodec::from_secrets(std::string(32, 'o'),
                                         "",
                                         /*token_ttl_seconds=*/3600);
  const std::string token = old_signer.issue_at(kRequestUid, kIssuedAt);
  ASSERT_FALSE(token.empty());

  const KVSessionTokenCodec rotated =
      *KVSessionTokenCodec::from_secrets(std::string(32, 'n'),
                                         std::string(32, 'o'),
                                         /*token_ttl_seconds=*/3600);
  EXPECT_EQ(rotated.verify_at(token, kIssuedAt + 3599), kRequestUid);
  EXPECT_FALSE(rotated.verify_at(token, kIssuedAt + 3601).has_value());
  EXPECT_FALSE(
      rotated
          .verify_at(token,
                     kIssuedAt - KVSessionTokenCodec::kMaxClockSkewSeconds - 1)
          .has_value());
}

TEST(RequestTrustPolicyTest, StandardSdkUserIsBoundToAuthenticatedClient) {
  const KVSessionTokenCodec signer = codec();
  const auto first = decide_request_trust(
      RequestTrustInput{
          .trusted_client_identity_headers_enabled = true,
          .authenticated_client = "gateway-client-a",
          .client_session_hint = "sdk-user-7",
          .request_uid = kRequestUid,
      },
      signer);
  const auto repeated = decide_request_trust(
      RequestTrustInput{
          .trusted_client_identity_headers_enabled = true,
          .authenticated_client = "gateway-client-a",
          .client_session_hint = "sdk-user-7",
          .request_uid = "018f47b2-c198-7cc8-98d7-503f58e2a613",
      },
      signer);
  const auto other_client = decide_request_trust(
      RequestTrustInput{
          .trusted_client_identity_headers_enabled = true,
          .authenticated_client = "gateway-client-b",
          .client_session_hint = "sdk-user-7",
          .request_uid = "018f47b2-c198-7cc8-98d7-503f58e2a614",
      },
      signer);
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(repeated.has_value());
  ASSERT_TRUE(other_client.has_value());
  EXPECT_EQ(first->kv_isolation_domain, repeated->kv_isolation_domain);
  EXPECT_NE(first->kv_isolation_domain, other_client->kv_isolation_domain);
  EXPECT_TRUE(first->issued_kv_session_token.empty());
}

TEST(RequestTrustPolicyTest, StandardSdkUserRequiresTrustedIdentityBoundary) {
  const KVSessionTokenCodec signer = codec();
  const auto first = decide_request_trust(
      RequestTrustInput{
          .authenticated_client = "attacker-controlled-credential",
          .client_session_hint = "victim-user",
          .request_uid = kRequestUid,
      },
      signer);
  const auto second = decide_request_trust(
      RequestTrustInput{
          .authenticated_client = "attacker-controlled-credential",
          .client_session_hint = "victim-user",
          .request_uid = "018f47b2-c198-7cc8-98d7-503f58e2a613",
      },
      signer);
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  EXPECT_NE(first->kv_isolation_domain, second->kv_isolation_domain);
  EXPECT_FALSE(first->issued_kv_session_token.empty());
  EXPECT_FALSE(second->issued_kv_session_token.empty());

  const auto oversized_identity = decide_request_trust(
      RequestTrustInput{
          .trusted_client_identity_headers_enabled = true,
          .authenticated_client = std::string(257, 'c'),
          .client_session_hint = "victim-user",
          .request_uid = kRequestUid,
      },
      signer);
  ASSERT_TRUE(oversized_identity.has_value());
  EXPECT_EQ(oversized_identity->kv_isolation_domain, kRequestUid);
  EXPECT_FALSE(oversized_identity->issued_kv_session_token.empty());
}

}  // namespace
}  // namespace xllm_service
