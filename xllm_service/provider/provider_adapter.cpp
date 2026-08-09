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
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <utility>

#include "provider/canonical_request_builder.h"

namespace xllm_service::provider {
namespace {

class XllmNativeRequestCodec final : public RequestCodec {
 public:
  XllmNativeRequestCodec(const xllm::proto::ProviderDescriptor& descriptor,
                         const XllmNativeRequestRenderer* request_renderer)
      : descriptor_(descriptor), request_renderer_(request_renderer) {}

  ContractResult encode(const xllm::proto::CanonicalRequest& request,
                        const RequestEncodingContext& context,
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
        request, context, &provider_payload, &prompt_tokens, &renderer_digest);
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
                        const RequestEncodingContext& context,
                        xllm::proto::EncodedRequest* encoded) const override {
    static_cast<void>(context);
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

std::optional<ProviderDispatchKind> resolve_provider_dispatch_kind(
    xllm::proto::ProviderId provider_id) {
  switch (provider_id) {
    case xllm::proto::PROVIDER_ID_XLLM_NATIVE:
      return ProviderDispatchKind::XLLM_NATIVE_RPC;
    case xllm::proto::PROVIDER_ID_VLLM_ASCEND:
      return ProviderDispatchKind::OPENAI_HTTP;
    default:
      return std::nullopt;
  }
}

ContractResult XllmNativePreparedRequestRenderer::render(
    const xllm::proto::CanonicalRequest& request,
    const RequestEncodingContext& context,
    std::string* provider_payload,
    uint64_t* prompt_tokens,
    std::string* renderer_digest) const {
  if (context.native_token_ids == nullptr || provider_payload == nullptr ||
      prompt_tokens == nullptr || renderer_digest == nullptr ||
      context.native_renderer_digest.empty()) {
    return ContractResult::failure(
        xllm::proto::PROVIDER_CONTRACT_ERROR_ENCODING_FAILED,
        "Native prepared renderer dependencies are unavailable");
  }
  if (request.request_uid() != context.request_uid ||
      !request.has_attempt_seq() || !context.attempt_seq.has_value() ||
      request.attempt_seq() != context.attempt_seq.value()) {
    return ContractResult::failure(
        xllm::proto::PROVIDER_CONTRACT_ERROR_DESCRIPTOR_MISMATCH,
        "Native prepared renderer request identity changed");
  }

  nlohmann::json canonical_payload = nlohmann::json::parse(
      request.canonical_payload(), nullptr, /*allow_exceptions=*/false);
  if (canonical_payload.is_discarded()) {
    return ContractResult::failure(
        xllm::proto::PROVIDER_CONTRACT_ERROR_ENCODING_FAILED,
        "Native canonical JSON payload is invalid");
  }
  nlohmann::json payload = nlohmann::json::object();
  payload["canonical_payload_schema"] = request.canonical_payload_schema();
  payload["canonical_request"] = std::move(canonical_payload);
  payload["token_ids"] = *context.native_token_ids;
  *provider_payload = payload.dump();
  *prompt_tokens = context.native_token_ids->size();
  *renderer_digest = context.native_renderer_digest;
  return ContractResult::success();
}

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

ContractResult create_provider_adapter(
    const xllm::proto::ProviderDescriptor& descriptor,
    std::unique_ptr<ProviderAdapter>* adapter) {
  if (adapter == nullptr) {
    return ContractResult::failure(
        xllm::proto::PROVIDER_CONTRACT_ERROR_MISSING_REQUIRED_FIELD,
        "Provider Adapter output must not be null");
  }
  adapter->reset();
  switch (descriptor.identity().provider_id()) {
    case xllm::proto::PROVIDER_ID_XLLM_NATIVE:
      *adapter = std::make_unique<XllmNativeAdapter>(
          descriptor, std::make_unique<XllmNativePreparedRequestRenderer>());
      return ContractResult::success();
    case xllm::proto::PROVIDER_ID_VLLM_ASCEND:
      *adapter = std::make_unique<VllmAscendAdapter>(descriptor);
      return ContractResult::success();
    default:
      return ContractResult::failure(
          xllm::proto::PROVIDER_CONTRACT_ERROR_UNKNOWN_ENUM_VALUE,
          "Provider has no production Adapter");
  }
}

}  // namespace xllm_service::provider
