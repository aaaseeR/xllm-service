/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <brpc/channel.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "placement/placement_actuator.h"
#include "provider.pb.h"
#include "provider/engine_registry.h"

namespace xllm_service::placement {

enum class ProviderLifecycleTransportStatus : int8_t {
  OK = 0,
  UNAVAILABLE = 1,
  OUTCOME_UNKNOWN = 2,
  INVALID = 3,
};

class ProviderLifecycleTransport {
 public:
  virtual ~ProviderLifecycleTransport() = default;

  virtual ProviderLifecycleTransportStatus call(
      const xllm::proto::ProviderDescriptor& descriptor,
      const xllm::proto::ProviderLifecycleCommand& command,
      bool query,
      xllm::proto::ProviderLifecycleResponse* response) = 0;
};

struct BrpcProviderLifecycleTransportConfig {
  int32_t timeout_ms = 0;
  size_t max_channels = 0;
};

struct HttpProviderLifecycleTransportConfig {
  int32_t timeout_ms = 0;
  size_t max_channels = 0;
  size_t max_response_bytes = 0;
  std::string internal_api_token;
};

// Production xLLM Native binary RPC transport. Provider-specific transports
// stay behind ProviderLifecycleTransport; placement logic only sees the shared
// lifecycle command/proof contract.
class BrpcProviderLifecycleTransport final : public ProviderLifecycleTransport {
 public:
  explicit BrpcProviderLifecycleTransport(
      BrpcProviderLifecycleTransportConfig config);

  ProviderLifecycleTransportStatus call(
      const xllm::proto::ProviderDescriptor& descriptor,
      const xllm::proto::ProviderLifecycleCommand& command,
      bool query,
      xllm::proto::ProviderLifecycleResponse* response) override;

 private:
  std::shared_ptr<brpc::Channel> channel(const std::string& address);

  BrpcProviderLifecycleTransportConfig config_;
  std::mutex mutex_;
  std::map<std::string, std::shared_ptr<brpc::Channel>> channels_;
};

// Production vLLM-Ascend Provider Agent HTTP transport. Authentication and
// strict response parsing remain inside the Provider-specific adapter.
class HttpProviderLifecycleTransport final : public ProviderLifecycleTransport {
 public:
  explicit HttpProviderLifecycleTransport(
      HttpProviderLifecycleTransportConfig config);

  ProviderLifecycleTransportStatus call(
      const xllm::proto::ProviderDescriptor& descriptor,
      const xllm::proto::ProviderLifecycleCommand& command,
      bool query,
      xllm::proto::ProviderLifecycleResponse* response) override;

 private:
  std::shared_ptr<brpc::Channel> channel(const std::string& address);

  HttpProviderLifecycleTransportConfig config_;
  std::mutex mutex_;
  std::map<std::string, std::shared_ptr<brpc::Channel>> channels_;
};

// Fixed Provider router: placement remains hardware-neutral while each
// Provider owns its wire protocol and runtime-specific control surface.
class ProviderLifecycleTransportRouter final
    : public ProviderLifecycleTransport {
 public:
  ProviderLifecycleTransportRouter(ProviderLifecycleTransport* xllm_native,
                                   ProviderLifecycleTransport* vllm_ascend);

  ProviderLifecycleTransportStatus call(
      const xllm::proto::ProviderDescriptor& descriptor,
      const xllm::proto::ProviderLifecycleCommand& command,
      bool query,
      xllm::proto::ProviderLifecycleResponse* response) override;

 private:
  ProviderLifecycleTransport* xllm_native_ = nullptr;
  ProviderLifecycleTransport* vllm_ascend_ = nullptr;
};

bool parse_vllm_lifecycle_response(
    int32_t http_status_code,
    const std::string& response_body,
    xllm::proto::ProviderLifecycleResponse* response);

// CREATE and TERMINATE are deployment-system operations, never Engine RPCs.
class PlacementDeploymentActuator {
 public:
  virtual ~PlacementDeploymentActuator() = default;
  virtual PlacementActuatorResponse execute(
      const PlacementOperationIntent& intent) = 0;
  virtual PlacementActuatorResponse query(
      const PlacementOperationIntent& intent) = 0;
};

class ProviderPlacementActuator final : public PlacementActuator {
 public:
  ProviderPlacementActuator(provider::EngineRegistry* registry,
                            ProviderLifecycleTransport* transport,
                            PlacementDeploymentActuator* deployment);

  PlacementActuatorResponse execute(
      const PlacementOperationIntent& intent) override;

  PlacementActuatorResponse query(
      const PlacementOperationIntent& intent) override;

 private:
  PlacementActuatorResponse lifecycle_call(
      const PlacementOperationIntent& intent,
      bool query);

  provider::EngineRegistry* registry_ = nullptr;
  ProviderLifecycleTransport* transport_ = nullptr;
  PlacementDeploymentActuator* deployment_ = nullptr;
};

}  // namespace xllm_service::placement
