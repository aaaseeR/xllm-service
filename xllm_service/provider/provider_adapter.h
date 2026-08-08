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

#include "provider.pb.h"
#include "provider/provider_contract.h"

namespace xllm_service::provider {

class RequestCodec {
 public:
  virtual ~RequestCodec() = default;

  virtual ContractResult encode(const xllm::proto::CanonicalRequest& request,
                                xllm::proto::EncodedRequest* encoded) const = 0;
};

class ProviderAdapter {
 public:
  virtual ~ProviderAdapter() = default;

  virtual const xllm::proto::ProviderDescriptor& describe() const = 0;
  virtual const RequestCodec& request_codec() const = 0;
};

}  // namespace xllm_service::provider
