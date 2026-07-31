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
#include <functional>
#include <memory>
#include <string>

#include "chat.pb.h"
#include "completion.pb.h"
#include "models.pb.h"
#include "routing/routing_decision.h"
#include "transport/channel_pool.h"
#include "transport/types.h"

namespace xllm_service {

struct DispatcherStats {
  bool closed = false;
  uint64_t inflight = 0;
  uint64_t transport_failure_total = 0;
  uint64_t stale_routing_decision_total = 0;
  size_t channel_count = 0;
  size_t endpoint_count = 0;
};

class DispatcherState;

class Dispatcher final {
 public:
  using TransportObserver = std::function<void(const TransportResult&)>;
  using ModelResultCallback =
      std::function<void(const TransportResult&,
                         const xllm::proto::ModelListResponse&)>;

  Dispatcher(std::shared_ptr<ChannelPool> channel_pool,
             TransportObserver transport_observer);
  ~Dispatcher();

  Dispatcher(const Dispatcher&) = delete;
  Dispatcher& operator=(const Dispatcher&) = delete;

  bool dispatch_completion(const RoutingDecision& decision,
                           const std::string& request_id,
                           const xllm::proto::CompletionRequest& request);
  bool dispatch_chat(const RoutingDecision& decision,
                     const std::string& request_id,
                     const xllm::proto::ChatRequest& request);
  bool dispatch_models(const std::string& endpoint,
                       const std::string& incarnation_id,
                       const xllm::proto::ModelListRequest& request,
                       ModelResultCallback callback);

  void close();
  DispatcherStats stats() const;

 private:
  const std::shared_ptr<ChannelPool> channel_pool_;
  const std::shared_ptr<DispatcherState> state_;
};

}  // namespace xllm_service
