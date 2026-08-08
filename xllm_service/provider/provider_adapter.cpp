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

#include "provider/provider_adapter.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace xllm_service::provider {
namespace {

class XllmNativeRequestCodec final : public RequestCodec {
 public:
  XllmNativeRequestCodec(const xllm::proto::ProviderDescriptor& descriptor,
                         const XllmNativeRequestRenderer* request_renderer)
      : descriptor_(descriptor), request_renderer_(request_renderer) {}

  ContractResult encode(const xllm::proto::CanonicalRequest& request,
                        xllm::proto::EncodedRequest* encoded) const override {
    ContractResult validation = validate_canonical_request(request);
    if (!validation.ok()) {
      return validation;
    }
    if (encoded == nullptr || request_renderer_ == nullptr) {
      return ContractResult::failure(
          xllm::proto::PROVIDER_CONTRACT_ERROR_ENCODING_FAILED,
          "xLLM Native encoder dependencies are unavailable");
    }

    std::string provider_payload;
    std::string renderer_digest;
    uint64_t prompt_tokens = 0;
    ContractResult render_result = request_renderer_->render(
        request, &provider_payload, &prompt_tokens, &renderer_digest);
    if (!render_result.ok()) {
      return render_result;
    }

    encoded->Clear();
    encoded->set_provider_id(xllm::proto::PROVIDER_ID_XLLM_NATIVE);
    encoded->set_token_count_quality(xllm::proto::TOKEN_COUNT_QUALITY_EXACT);
    encoded->set_prompt_tokens(prompt_tokens);
    encoded->set_prompt_tokens_upper_bound(prompt_tokens);
    encoded->set_renderer_digest(std::move(renderer_digest));
    encoded->set_provider_payload(std::move(provider_payload));
    return validate_encoded_request(descriptor_, request, *encoded);
  }

 private:
  const xllm::proto::ProviderDescriptor& descriptor_;
  const XllmNativeRequestRenderer* request_renderer_;
};

class VllmAscendRequestCodec final : public RequestCodec {
 public:
  explicit VllmAscendRequestCodec(
      const xllm::proto::ProviderDescriptor& descriptor)
      : descriptor_(descriptor) {}

  ContractResult encode(const xllm::proto::CanonicalRequest& request,
                        xllm::proto::EncodedRequest* encoded) const override {
    ContractResult validation = validate_canonical_request(request);
    if (!validation.ok()) {
      return validation;
    }
    if (encoded == nullptr) {
      return ContractResult::failure(
          xllm::proto::PROVIDER_CONTRACT_ERROR_ENCODING_FAILED,
          "vLLM-Ascend encoded output must not be null");
    }
    if (request.canonical_payload_schema() != kOpenAiHttpJsonSchema) {
      return ContractResult::failure(
          xllm::proto::PROVIDER_CONTRACT_ERROR_ENCODING_FAILED,
          "vLLM-Ascend requires the OpenAI HTTP JSON canonical schema");
    }

    encoded->Clear();
    encoded->set_provider_id(xllm::proto::PROVIDER_ID_VLLM_ASCEND);
    encoded->set_token_count_quality(xllm::proto::TOKEN_COUNT_QUALITY_UNKNOWN);
    encoded->set_prompt_tokens(0);
    encoded->set_prompt_tokens_upper_bound(0);
    encoded->set_renderer_digest(descriptor_.model().renderer_digest());
    encoded->set_provider_payload(request.canonical_payload());
    return validate_encoded_request(descriptor_, request, *encoded);
  }

 private:
  const xllm::proto::ProviderDescriptor& descriptor_;
};

}  // namespace

XllmNativeAdapter::XllmNativeAdapter(
    xllm::proto::ProviderDescriptor descriptor,
    std::unique_ptr<XllmNativeRequestRenderer> request_renderer)
    : descriptor_(std::move(descriptor)),
      request_renderer_(std::move(request_renderer)),
      request_codec_(
          std::make_unique<XllmNativeRequestCodec>(descriptor_,
                                                   request_renderer_.get())) {}

XllmNativeAdapter::~XllmNativeAdapter() = default;

const xllm::proto::ProviderDescriptor& XllmNativeAdapter::describe() const {
  return descriptor_;
}

const RequestCodec& XllmNativeAdapter::request_codec() const {
  return *request_codec_;
}

ProviderDispatchKind XllmNativeAdapter::dispatch_kind() const {
  return ProviderDispatchKind::XLLM_NATIVE_RPC;
}

VllmAscendAdapter::VllmAscendAdapter(xllm::proto::ProviderDescriptor descriptor)
    : descriptor_(std::move(descriptor)),
      request_codec_(std::make_unique<VllmAscendRequestCodec>(descriptor_)) {}

VllmAscendAdapter::~VllmAscendAdapter() = default;

const xllm::proto::ProviderDescriptor& VllmAscendAdapter::describe() const {
  return descriptor_;
}

const RequestCodec& VllmAscendAdapter::request_codec() const {
  return *request_codec_;
}

ProviderDispatchKind VllmAscendAdapter::dispatch_kind() const {
  return ProviderDispatchKind::OPENAI_HTTP;
}

}  // namespace xllm_service::provider
