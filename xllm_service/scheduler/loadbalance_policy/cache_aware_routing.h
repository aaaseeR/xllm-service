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

#include "common/macros.h"
#include "common/options.h"
#include "loadbalance_policy.h"
#include "provider/kv_route_planner.h"
#include "provider/kv_shadow_index.h"

namespace xllm_service {

class CacheAwareRouting final : public LoadBalancePolicy {
 public:
  CacheAwareRouting(const Options& options,
                    std::shared_ptr<InstanceMgr> instance_mgr,
                    provider::KVShadowIndex* kv_shadow_index);

  virtual ~CacheAwareRouting() = default;

  bool select_instances_pair(std::shared_ptr<Request> request) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(CacheAwareRouting);

  bool fallback_load_only(const std::shared_ptr<Request>& request) const;

  Options options_;
  provider::KVRoutePlanner planner_;
  provider::KVRouteMode mode_ = provider::KVRouteMode::DISABLED;
  provider::KVShadowIndex* kv_shadow_index_ = nullptr;
};

}  // namespace xllm_service
