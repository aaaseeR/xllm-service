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

#include "provider/kv_route_planner.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace xllm_service::provider {
namespace {

KVRoutePlannerConfig config() {
  return KVRoutePlannerConfig{
      .max_candidate_plans = 16,
      .least_load_shortlist = 1,
      .top_prefix_shortlist = 1,
      .prefill_queue_cost_us = 1000,
      .decode_request_cost_us = 1000,
      .prefill_token_cost_us = 10,
      .transfer_byte_cost_us = 0.001,
      .decode_headroom_cost_us = 1000,
      .prefill_reserve_blocks = 1,
      .kv_routing_margin_us = 100,
      .near_equal_cost_us = 10,
  };
}

KVRouteRequest request(uint64_t request_hash = 1) {
  return KVRouteRequest{
      .prompt_tokens = 160,
      .block_size = 16,
      .kv_bytes_per_token = 1024,
      .request_hash = request_hash,
  };
}

KVRouteEngineCandidate engine(std::string uid,
                              uint64_t waiting,
                              uint64_t prefix_blocks = 0) {
  return KVRouteEngineCandidate{
      .engine_uid = std::move(uid),
      .role = xllm::proto::ENGINE_ROLE_PREFILL,
      .load_known = true,
      .waiting_requests = waiting,
      .kv_free_blocks = 100,
      .kv_health =
          prefix_blocks == 0 ? KVShadowHealth::UNKNOWN : KVShadowHealth::READY,
      .hbm_prefix_blocks = prefix_blocks,
  };
}

KVRoutePlanCandidate plan(std::string uid,
                          uint64_t waiting,
                          uint64_t prefix_blocks = 0) {
  return KVRoutePlanCandidate{
      .prefill = engine(std::move(uid), waiting, prefix_blocks),
  };
}

TEST(KVRoutePlannerTest, EnforcedModeRequiresOpenDeterministicBucketGate) {
  EXPECT_EQ(select_kv_route_mode(KVRouteMode::ENFORCED,
                                 /*enforced_gate_open=*/false,
                                 /*enforced_bucket_permyriad=*/10000,
                                 /*request_hash=*/0),
            KVRouteMode::SHADOW);
  EXPECT_EQ(select_kv_route_mode(KVRouteMode::ENFORCED,
                                 /*enforced_gate_open=*/true,
                                 /*enforced_bucket_permyriad=*/0,
                                 /*request_hash=*/0),
            KVRouteMode::SHADOW);
  EXPECT_EQ(select_kv_route_mode(KVRouteMode::ENFORCED,
                                 /*enforced_gate_open=*/true,
                                 /*enforced_bucket_permyriad=*/100,
                                 /*request_hash=*/99),
            KVRouteMode::ENFORCED);
  EXPECT_EQ(select_kv_route_mode(KVRouteMode::ENFORCED,
                                 /*enforced_gate_open=*/true,
                                 /*enforced_bucket_permyriad=*/100,
                                 /*request_hash=*/100),
            KVRouteMode::SHADOW);
  EXPECT_EQ(select_kv_route_mode(KVRouteMode::ENFORCED,
                                 /*enforced_gate_open=*/true,
                                 /*enforced_bucket_permyriad=*/10001,
                                 /*request_hash=*/0),
            KVRouteMode::SHADOW);
  EXPECT_EQ(select_kv_route_mode(KVRouteMode::DISABLED,
                                 /*enforced_gate_open=*/true,
                                 /*enforced_bucket_permyriad=*/10000,
                                 /*request_hash=*/0),
            KVRouteMode::DISABLED);
}

TEST(KVRoutePlannerTest, RejectsInvalidAndUnboundedInputs) {
  KVRoutePlannerConfig invalid = config();
  invalid.least_load_shortlist = 0;
  EXPECT_FALSE(KVRoutePlanner(invalid).valid());

  KVRoutePlanner planner(config());
  EXPECT_FALSE(planner.select({}, {plan("p0", 0)}, KVRouteMode::SHADOW).valid);
  std::vector<KVRoutePlanCandidate> too_many(17, plan("p0", 0));
  EXPECT_FALSE(planner.select(request(), too_many, KVRouteMode::SHADOW).valid);
}

TEST(KVRoutePlannerTest, LocalPendingWorkProtectsLoadOnlyChoice) {
  KVRoutePlanner planner(config());
  std::vector<KVRoutePlanCandidate> candidates = {plan("p0", 0), plan("p1", 1)};
  candidates[0].prefill.local_pending_requests = 4;

  const KVRouteDecision decision =
      planner.select(request(), candidates, KVRouteMode::DISABLED);
  ASSERT_TRUE(decision.valid);
  EXPECT_EQ(decision.load_only_index, 1u);
  EXPECT_EQ(decision.selected_index, 1u);
  EXPECT_EQ(decision.fallback, KVRouteFallback::DISABLED);
}

TEST(KVRoutePlannerTest, EnforcedModeUsesDimensionedPrefixSavings) {
  KVRoutePlanner planner(config());
  const std::vector<KVRoutePlanCandidate> candidates = {
      plan("least-load", 0), plan("cache-rich", 1, 10)};

  const KVRouteDecision decision =
      planner.select(request(), candidates, KVRouteMode::ENFORCED);
  ASSERT_TRUE(decision.valid);
  EXPECT_EQ(decision.load_only_index, 0u);
  EXPECT_EQ(decision.kv_preferred_index, 1u);
  EXPECT_EQ(decision.selected_index, 1u);
  EXPECT_EQ(decision.fallback, KVRouteFallback::NONE);
  EXPECT_EQ(decision.evaluations[1].predicted_prefill_hit_tokens, 160u);
  EXPECT_EQ(decision.evaluations[1].effective_prefill_tokens, 0u);
}

TEST(KVRoutePlannerTest, ShadowModeObservesButKeepsLoadOnlyRoute) {
  KVRoutePlanner planner(config());
  const std::vector<KVRoutePlanCandidate> candidates = {
      plan("least-load", 0), plan("cache-rich", 1, 10)};

  const KVRouteDecision decision =
      planner.select(request(), candidates, KVRouteMode::SHADOW);
  ASSERT_TRUE(decision.valid);
  EXPECT_EQ(decision.load_only_index, 0u);
  EXPECT_EQ(decision.kv_preferred_index, 1u);
  EXPECT_EQ(decision.selected_index, 0u);
  EXPECT_EQ(decision.fallback, KVRouteFallback::NONE);
}

TEST(KVRoutePlannerTest, UnknownIndexFallsBackWithoutLosingAvailability) {
  KVRoutePlanner planner(config());
  const std::vector<KVRoutePlanCandidate> candidates = {plan("p0", 0),
                                                        plan("p1", 1)};

  const KVRouteDecision decision =
      planner.select(request(), candidates, KVRouteMode::ENFORCED);
  ASSERT_TRUE(decision.valid);
  EXPECT_EQ(decision.selected_index, decision.load_only_index);
  EXPECT_EQ(decision.fallback, KVRouteFallback::KV_UNAVAILABLE);
}

TEST(KVRoutePlannerTest, SurvivalCreditCapsUnretainablePrefix) {
  KVRoutePlanner planner(config());
  std::vector<KVRoutePlanCandidate> candidates = {plan("least-load", 0),
                                                  plan("cache-rich", 1, 10)};
  candidates[1].prefill.kv_free_blocks = 1;

  const KVRouteDecision decision =
      planner.select(request(), candidates, KVRouteMode::ENFORCED);
  ASSERT_TRUE(decision.valid);
  EXPECT_EQ(decision.evaluations[1].survival_credit, 0.0);
  EXPECT_EQ(decision.evaluations[1].predicted_prefill_hit_tokens, 0u);
  EXPECT_EQ(decision.selected_index, decision.load_only_index);
  EXPECT_EQ(decision.fallback, KVRouteFallback::BELOW_MARGIN);
}

TEST(KVRoutePlannerTest, ShortlistUnionsLeastLoadAndTopPrefix) {
  KVRoutePlanner planner(config());
  const std::vector<KVRoutePlanCandidate> candidates = {
      plan("least-load", 0),
      plan("middle", 1, 1),
      plan("top-prefix", 2, 10),
  };

  const KVRouteDecision decision =
      planner.select(request(), candidates, KVRouteMode::SHADOW);
  ASSERT_TRUE(decision.valid);
  EXPECT_TRUE(decision.evaluations[0].in_shortlist);
  EXPECT_FALSE(decision.evaluations[1].in_shortlist);
  EXPECT_TRUE(decision.evaluations[2].in_shortlist);
}

TEST(KVRoutePlannerTest, CandidateTruncationFailsClosedToLoadOnly) {
  KVRoutePlanner planner(config());
  const std::vector<KVRoutePlanCandidate> candidates = {
      plan("least-load", 0), plan("cache-rich", 1, 10)};

  const KVRouteDecision decision =
      planner.select(request(), candidates, KVRouteMode::ENFORCED, true);
  ASSERT_TRUE(decision.valid);
  EXPECT_EQ(decision.selected_index, decision.load_only_index);
  EXPECT_EQ(decision.fallback, KVRouteFallback::CANDIDATE_LIMIT);
}

TEST(KVRoutePlannerTest, NearEqualCostUsesRequestHashTieBreak) {
  KVRoutePlanner planner(config());
  const std::vector<KVRoutePlanCandidate> candidates = {plan("p0", 0),
                                                        plan("p1", 0)};
  bool observed_different_choice = false;
  const size_t first =
      planner.select(request(0), candidates, KVRouteMode::DISABLED)
          .selected_index;
  for (uint64_t hash = 1; hash < 128; ++hash) {
    const size_t selected =
        planner.select(request(hash), candidates, KVRouteMode::DISABLED)
            .selected_index;
    observed_different_choice = observed_different_choice || selected != first;
  }
  EXPECT_TRUE(observed_different_choice);
}

}  // namespace
}  // namespace xllm_service::provider
