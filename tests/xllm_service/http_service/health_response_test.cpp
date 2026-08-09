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

#include "http_service/health_response.h"

#include <gtest/gtest.h>

namespace xllm_service {
namespace {

TEST(HealthResponseTest, LivenessIsIndependentFromReadiness) {
  const HealthResponse response = make_liveness_response();
  EXPECT_EQ(response.status_code, 200);
  EXPECT_EQ(response.body, "{\"status\":\"live\"}");
}

TEST(HealthResponseTest, ReadyResponseUsesStableReason) {
  const HealthResponse response =
      make_readiness_response(provider::ReadinessSnapshot{
          .accepting_new_requests = true,
          .reason = provider::ReadinessReason::READY,
          .changed_monotonic_ms = 100,
      });
  EXPECT_EQ(response.status_code, 200);
  EXPECT_EQ(response.body, "{\"status\":\"ready\",\"reason\":\"READY\"}");
}

TEST(HealthResponseTest, NotReadyAndAdmissionResponsesAreStable) {
  const provider::ReadinessSnapshot readiness{
      .accepting_new_requests = false,
      .reason = provider::ReadinessReason::REGISTRY_BLIND_GRACE_EXPIRED,
      .changed_monotonic_ms = 100,
  };
  const HealthResponse readiness_response = make_readiness_response(readiness);
  EXPECT_EQ(readiness_response.status_code, 503);
  EXPECT_EQ(readiness_response.body,
            "{\"status\":\"not_ready\",\"reason\":"
            "\"REGISTRY_BLIND_GRACE_EXPIRED\"}");

  const HealthResponse admission_response =
      make_service_not_ready_response(readiness);
  EXPECT_EQ(admission_response.status_code, 503);
  EXPECT_EQ(admission_response.body,
            "{\"code\":\"SERVICE_NOT_READY\",\"reason\":"
            "\"REGISTRY_BLIND_GRACE_EXPIRED\"}");
}

}  // namespace
}  // namespace xllm_service
