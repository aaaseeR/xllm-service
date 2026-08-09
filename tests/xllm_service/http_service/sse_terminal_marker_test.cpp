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

#include "http_service/sse_terminal_marker.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace xllm_service {
namespace {

TEST(SseTerminalMarkerTest, RecognizesTerminalAcrossArbitraryParts) {
  SseTerminalMarker marker;
  const std::string first = "data: {\"text\":\"partial\"}\n\ndata: [DO";
  const std::string second = "NE]\n\n";
  marker.observe(first.data(), first.size());
  EXPECT_FALSE(marker.terminal_at_end());
  marker.observe(second.data(), second.size());
  EXPECT_TRUE(marker.terminal_at_end());
}

TEST(SseTerminalMarkerTest, AcceptsExactMarkerAtCleanEndOfStream) {
  SseTerminalMarker marker;
  const std::string payload = "data: [DONE]";
  marker.observe(payload.data(), payload.size());
  EXPECT_TRUE(marker.terminal_at_end());
}

TEST(SseTerminalMarkerTest, RejectsEmbeddedOrTruncatedMarker) {
  const std::vector<std::string> payloads = {
      "data: {\"text\":\"data: [DONE]\"}\n\n",
      "prefix data: [DONE]\n\n",
      "data: [DON",
      "data: [DONE]suffix\n\n",
  };
  for (const std::string& payload : payloads) {
    SseTerminalMarker marker;
    marker.observe(payload.data(), payload.size());
    EXPECT_FALSE(marker.terminal_at_end()) << payload;
  }
}

}  // namespace
}  // namespace xllm_service
