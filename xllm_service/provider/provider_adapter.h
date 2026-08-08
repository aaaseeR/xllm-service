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

#include <cstdint>
#include <memory>
#include <string>

#include "provider.pb.h"
#include "provider/provider_contract.h"

namespace xllm_service::provider {

inline constexpr char kOpenAiHttpJsonSchema[] = "openai.http.json.v1";

// The scheduler and HTTP layer branch on this stable Adapter-owned semantic,
// never on legacy registration strings such as backend_type.
enum class ProviderDispatchKind : int8_t {
  XLLM_NATIVE_RPC = 0,
  OPENAI_HTTP = 1,
};

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
  virtual ProviderDispatchKind dispatch_kind() const = 0;
};

// xLLM Native owns tokenization and template rendering in the Service. The
// injected renderer keeps those model-specific dependencies outside the
// Provider contract while making their digest and exact token count explicit.
class XllmNativeRequestRenderer {
 public:
  virtual ~XllmNativeRequestRenderer() = default;

  virtual ContractResult render(const xllm::proto::CanonicalRequest& request,
                                std::string* provider_payload,
                                uint64_t* prompt_tokens,
                                std::string* renderer_digest) const = 0;
};

class XllmNativeAdapter final : public ProviderAdapter {
 public:
  XllmNativeAdapter(
      xllm::proto::ProviderDescriptor descriptor,
      std::unique_ptr<XllmNativeRequestRenderer> request_renderer);
  ~XllmNativeAdapter() override;

  const xllm::proto::ProviderDescriptor& describe() const override;
  const RequestCodec& request_codec() const override;
  ProviderDispatchKind dispatch_kind() const override;

 private:
  xllm::proto::ProviderDescriptor descriptor_;
  std::unique_ptr<XllmNativeRequestRenderer> request_renderer_;
  std::unique_ptr<RequestCodec> request_codec_;
};

class VllmAscendAdapter final : public ProviderAdapter {
 public:
  explicit VllmAscendAdapter(xllm::proto::ProviderDescriptor descriptor);
  ~VllmAscendAdapter() override;

  const xllm::proto::ProviderDescriptor& describe() const override;
  const RequestCodec& request_codec() const override;
  ProviderDispatchKind dispatch_kind() const override;

 private:
  xllm::proto::ProviderDescriptor descriptor_;
  std::unique_ptr<RequestCodec> request_codec_;
};

}  // namespace xllm_service::provider
