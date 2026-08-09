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

#include "provider/provider_route_selector.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace xllm_service::provider {
namespace {

ProviderRouteCandidate candidate(const std::string& engine_uid,
                                 xllm::proto::ProviderId provider_id,
                                 xllm::proto::EngineRole role,
                                 bool schedulable = true,
                                 std::string model_revision = "") {
  return ProviderRouteCandidate{
      .engine_uid = engine_uid,
      .provider_id = provider_id,
      .role = role,
      .schedulable = schedulable,
      .model_revision = std::move(model_revision),
  };
}

TEST(ProviderRouteSelectorTest, SelectsNativeRemotePdWithinProvider) {
  const std::vector<ProviderRouteCandidate> prefills = {
      candidate("native-p",
                xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                xllm::proto::ENGINE_ROLE_PREFILL)};
  const std::vector<ProviderRouteCandidate> decodes = {
      candidate("native-d",
                xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                xllm::proto::ENGINE_ROLE_DECODE)};
  ProviderRouteSelection selection;

  ASSERT_TRUE(
      ProviderRouteSelector::select(prefills,
                                    decodes,
                                    xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                                    0,
                                    0,
                                    &selection));
  EXPECT_EQ(selection.prefill_engine_uid, "native-p");
  EXPECT_EQ(selection.decode_engine_uid, "native-d");
  EXPECT_EQ(selection.provider_id, xllm::proto::PROVIDER_ID_XLLM_NATIVE);
}

TEST(ProviderRouteSelectorTest, SelectsVllmAsSingleAggregatedEngine) {
  const std::vector<ProviderRouteCandidate> prefills = {
      candidate("vllm",
                xllm::proto::PROVIDER_ID_VLLM_ASCEND,
                xllm::proto::ENGINE_ROLE_AGGREGATED)};
  ProviderRouteSelection selection;

  ASSERT_TRUE(ProviderRouteSelector::select(
      prefills, {}, xllm::proto::PROVIDER_ID_VLLM_ASCEND, 0, 0, &selection));
  EXPECT_EQ(selection.prefill_engine_uid, "vllm");
  EXPECT_TRUE(selection.decode_engine_uid.empty());
}

TEST(ProviderRouteSelectorTest, NeverPairsAcrossProviders) {
  const std::vector<ProviderRouteCandidate> prefills = {
      candidate("native-p",
                xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                xllm::proto::ENGINE_ROLE_PREFILL),
      candidate("vllm",
                xllm::proto::PROVIDER_ID_VLLM_ASCEND,
                xllm::proto::ENGINE_ROLE_AGGREGATED)};
  const std::vector<ProviderRouteCandidate> decodes = {
      candidate("foreign-d",
                xllm::proto::PROVIDER_ID_VLLM_ASCEND,
                xllm::proto::ENGINE_ROLE_DECODE)};
  ProviderRouteSelection selection;

  ASSERT_TRUE(
      ProviderRouteSelector::select(prefills,
                                    decodes,
                                    xllm::proto::PROVIDER_ID_UNSPECIFIED,
                                    0,
                                    0,
                                    &selection));
  EXPECT_EQ(selection.provider_id, xllm::proto::PROVIDER_ID_VLLM_ASCEND);
  EXPECT_EQ(selection.prefill_engine_uid, "vllm");
  EXPECT_TRUE(selection.decode_engine_uid.empty());
}

TEST(ProviderRouteSelectorTest, RequiredProviderFailsWithoutCompletePlan) {
  const std::vector<ProviderRouteCandidate> prefills = {
      candidate("native-p",
                xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                xllm::proto::ENGINE_ROLE_PREFILL),
      candidate("vllm",
                xllm::proto::PROVIDER_ID_VLLM_ASCEND,
                xllm::proto::ENGINE_ROLE_AGGREGATED)};
  ProviderRouteSelection selection;

  EXPECT_FALSE(ProviderRouteSelector::select(
      prefills, {}, xllm::proto::PROVIDER_ID_XLLM_NATIVE, 0, 0, &selection));
}

TEST(ProviderRouteSelectorTest, RejectsSplitVllmTopology) {
  const std::vector<ProviderRouteCandidate> prefills = {
      candidate("vllm-p",
                xllm::proto::PROVIDER_ID_VLLM_ASCEND,
                xllm::proto::ENGINE_ROLE_PREFILL)};
  const std::vector<ProviderRouteCandidate> decodes = {
      candidate("vllm-d",
                xllm::proto::PROVIDER_ID_VLLM_ASCEND,
                xllm::proto::ENGINE_ROLE_DECODE)};
  ProviderRouteSelection selection;

  EXPECT_FALSE(
      ProviderRouteSelector::select(prefills,
                                    decodes,
                                    xllm::proto::PROVIDER_ID_VLLM_ASCEND,
                                    0,
                                    0,
                                    &selection));
}

TEST(ProviderRouteSelectorTest, SkipsUnschedulableEngines) {
  const std::vector<ProviderRouteCandidate> prefills = {
      candidate("suspect",
                xllm::proto::PROVIDER_ID_VLLM_ASCEND,
                xllm::proto::ENGINE_ROLE_AGGREGATED,
                false),
      candidate("healthy",
                xllm::proto::PROVIDER_ID_VLLM_ASCEND,
                xllm::proto::ENGINE_ROLE_AGGREGATED)};
  ProviderRouteSelection selection;

  ASSERT_TRUE(ProviderRouteSelector::select(
      prefills, {}, xllm::proto::PROVIDER_ID_UNSPECIFIED, 0, 0, &selection));
  EXPECT_EQ(selection.prefill_engine_uid, "healthy");
}

TEST(ProviderRouteSelectorTest, AdvancesBothRoundRobinCursors) {
  const std::vector<ProviderRouteCandidate> prefills = {
      candidate("p0",
                xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                xllm::proto::ENGINE_ROLE_PREFILL),
      candidate("p1",
                xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                xllm::proto::ENGINE_ROLE_PREFILL)};
  const std::vector<ProviderRouteCandidate> decodes = {
      candidate("d0",
                xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                xllm::proto::ENGINE_ROLE_DECODE),
      candidate("d1",
                xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                xllm::proto::ENGINE_ROLE_DECODE)};
  ProviderRouteSelection selection;

  ASSERT_TRUE(
      ProviderRouteSelector::select(prefills,
                                    decodes,
                                    xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                                    1,
                                    1,
                                    &selection));
  EXPECT_EQ(selection.prefill_engine_uid, "p1");
  EXPECT_EQ(selection.decode_engine_uid, "d1");
  EXPECT_EQ(selection.next_prefill_index, uint64_t(2));
  EXPECT_EQ(selection.next_decode_index, uint64_t(2));
}

TEST(ProviderRouteSelectorTest, RejectsUnknownProvider) {
  const std::vector<ProviderRouteCandidate> prefills = {
      candidate("unknown",
                xllm::proto::PROVIDER_ID_UNSPECIFIED,
                xllm::proto::ENGINE_ROLE_AGGREGATED)};
  ProviderRouteSelection selection;

  EXPECT_FALSE(ProviderRouteSelector::select(
      prefills, {}, xllm::proto::PROVIDER_ID_UNSPECIFIED, 0, 0, &selection));
}

TEST(ProviderRouteSelectorTest, SelectsOnlyTheRequestedModelPool) {
  const std::vector<ProviderRouteCandidate> prefills = {
      candidate("model-a-p",
                xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                xllm::proto::ENGINE_ROLE_PREFILL,
                true,
                "model-a"),
      candidate("model-b-p",
                xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                xllm::proto::ENGINE_ROLE_PREFILL,
                true,
                "model-b")};
  const std::vector<ProviderRouteCandidate> decodes = {
      candidate("model-a-d",
                xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                xllm::proto::ENGINE_ROLE_DECODE,
                true,
                "model-a"),
      candidate("model-b-d",
                xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                xllm::proto::ENGINE_ROLE_DECODE,
                true,
                "model-b")};
  ProviderRouteSelection selection;

  ASSERT_TRUE(
      ProviderRouteSelector::select(prefills,
                                    decodes,
                                    xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                                    0,
                                    0,
                                    &selection,
                                    "model-b"));
  EXPECT_EQ(selection.prefill_engine_uid, "model-b-p");
  EXPECT_EQ(selection.decode_engine_uid, "model-b-d");

  EXPECT_FALSE(
      ProviderRouteSelector::select(prefills,
                                    decodes,
                                    xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                                    0,
                                    0,
                                    &selection,
                                    "model-c"));
}

TEST(ProviderRouteSelectorTest, EnumeratesOnlyCompleteHardFilteredPlans) {
  const std::vector<ProviderRouteCandidate> prefills = {
      candidate("ready-p",
                xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                xllm::proto::ENGINE_ROLE_PREFILL),
      candidate("stale-p",
                xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                xllm::proto::ENGINE_ROLE_PREFILL,
                false),
      candidate("aggregated",
                xllm::proto::PROVIDER_ID_VLLM_ASCEND,
                xllm::proto::ENGINE_ROLE_AGGREGATED),
  };
  const std::vector<ProviderRouteCandidate> decodes = {
      candidate("d0",
                xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                xllm::proto::ENGINE_ROLE_DECODE),
      candidate("foreign-d",
                xllm::proto::PROVIDER_ID_VLLM_ASCEND,
                xllm::proto::ENGINE_ROLE_DECODE),
  };
  std::vector<ProviderRouteSelection> selections;
  bool truncated = true;

  ASSERT_TRUE(ProviderRouteSelector::select_candidates(
      prefills,
      decodes,
      xllm::proto::PROVIDER_ID_UNSPECIFIED,
      8,
      &selections,
      &truncated));
  ASSERT_EQ(selections.size(), 2u);
  EXPECT_EQ(selections[0].prefill_engine_uid, "ready-p");
  EXPECT_EQ(selections[0].decode_engine_uid, "d0");
  EXPECT_EQ(selections[1].prefill_engine_uid, "aggregated");
  EXPECT_TRUE(selections[1].decode_engine_uid.empty());
  EXPECT_FALSE(truncated);
}

TEST(ProviderRouteSelectorTest, CandidateEnumerationReportsTruncation) {
  const std::vector<ProviderRouteCandidate> prefills = {
      candidate("p0",
                xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                xllm::proto::ENGINE_ROLE_PREFILL),
      candidate("p1",
                xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                xllm::proto::ENGINE_ROLE_PREFILL),
  };
  const std::vector<ProviderRouteCandidate> decodes = {
      candidate("d0",
                xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                xllm::proto::ENGINE_ROLE_DECODE),
      candidate("d1",
                xllm::proto::PROVIDER_ID_XLLM_NATIVE,
                xllm::proto::ENGINE_ROLE_DECODE),
  };
  std::vector<ProviderRouteSelection> selections;
  bool truncated = false;

  ASSERT_TRUE(ProviderRouteSelector::select_candidates(
      prefills,
      decodes,
      xllm::proto::PROVIDER_ID_XLLM_NATIVE,
      2,
      &selections,
      &truncated));
  EXPECT_EQ(selections.size(), 2u);
  EXPECT_TRUE(truncated);
}

}  // namespace
}  // namespace xllm_service::provider
