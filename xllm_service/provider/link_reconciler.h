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

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "provider.pb.h"
#include "provider/provider_contract.h"

namespace xllm_service::provider {

struct LinkReconcilerConfig {
  size_t max_links = 16384;
  uint64_t retry_initial_ms = 1000;
  uint64_t retry_max_ms = 30000;
  uint64_t ready_recheck_ms = 5000;
};

struct DesiredProviderLink {
  xllm::proto::ProviderDescriptor prefill;
  xllm::proto::ProviderDescriptor decode;
};

struct ProviderLinkAttempt {
  xllm::proto::ProviderEngineKey prefill;
  xllm::proto::ProviderEngineKey decode;
};

// Bounded, incarnation-scoped P/D handshake state machine. It contains no
// networking: callers start the returned attempts and report their outcomes.
class LinkReconciler final {
 public:
  explicit LinkReconciler(LinkReconcilerConfig config);

  ContractResult replace_desired(
      const std::vector<DesiredProviderLink>& desired,
      uint64_t now_monotonic_ms,
      std::vector<xllm::proto::LinkState>* state_changes);

  std::vector<ProviderLinkAttempt> begin_due_attempts(uint64_t now_monotonic_ms,
                                                      size_t max_attempts);

  ContractResult complete_attempt(const ProviderLinkAttempt& attempt,
                                  bool success,
                                  std::string handshake_result,
                                  uint64_t now_monotonic_ms,
                                  xllm::proto::LinkState* state_change);

  void require_recheck();
  size_t size() const;

 private:
  struct Entry {
    xllm::proto::LinkState state;
    uint64_t next_attempt_ms = 0;
    uint32_t consecutive_failures = 0;
    bool in_flight = false;
  };

  static std::string link_key(const xllm::proto::ProviderEngineKey& prefill,
                              const xllm::proto::ProviderEngineKey& decode);
  uint64_t retry_delay_ms(uint32_t consecutive_failures) const;

  LinkReconcilerConfig config_;
  bool config_valid_ = false;
  mutable std::mutex mutex_;
  std::map<std::string, Entry> entries_;
};

}  // namespace xllm_service::provider
