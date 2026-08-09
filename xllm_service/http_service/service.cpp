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

#include "http_service/service.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <brpc/controller.h>
#include <brpc/progressive_reader.h>
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>
#include <json2pb/json_to_pb.h>
#include <json2pb/pb_to_json.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <nlohmann/json.hpp>
#include <type_traits>

#include "chat.pb.h"
#include "common/anthropic_tracer.h"
#include "common/call_data.h"
#include "common/closure_guard.h"
#include "common/utils.h"
#include "common/xllm/status.h"
#include "completion.pb.h"
#include "http_service/anthropic_adapter.h"
#include "http_service/chat_json_parser.h"
#include "http_service/health_response.h"
#include "http_service/request_execution_context.h"
#include "observability/request_identity.h"
#include "provider/canonical_request_builder.h"
#include "scheduler/scheduler.h"
#include "xllm_rpc_service.pb.h"
#include "xllm_service.pb.h"

namespace xllm_service {

namespace {

std::string first_header_value(
    const brpc::Controller* controller,
    std::initializer_list<const char*> header_names) {
  for (const char* header_name : header_names) {
    if (const std::string* value =
            controller->http_request().GetHeader(header_name)) {
      return *value;
    }
  }
  return {};
}

observability::RequestCorrelationInput correlation_input(
    const brpc::Controller* controller) {
  observability::RequestCorrelationInput input;
  input.global_request_id = first_header_value(
      controller,
      {"x-global-request-id", "x-request-id", "x-ms-client-request-id"});
  input.trace_id = first_header_value(controller, {"x-trace-id", "trace-id"});
  input.traceparent = first_header_value(controller, {"traceparent"});
  return input;
}

void reply_json(brpc::Controller* controller,
                int32_t status_code,
                const std::string& body) {
  controller->http_response().set_status_code(status_code);
  controller->http_response().set_content_type("application/json");
  controller->response_attachment().append(body);
}

void reply_service_not_ready(Scheduler* scheduler,
                             brpc::Controller* controller) {
  const provider::ReadinessSnapshot readiness = scheduler->readiness_status();
  const HealthResponse response = make_service_not_ready_response(readiness);
  reply_json(controller, response.status_code, response.body);
}

bool require_accepting_new_requests(Scheduler* scheduler,
                                    brpc::Controller* controller) {
  if (scheduler->accepting_new_requests()) {
    return true;
  }
  reply_service_not_ready(scheduler, controller);
  return false;
}

bool schedule_request(Scheduler* scheduler,
                      const std::shared_ptr<Request>& request,
                      brpc::Controller* controller) {
  if (scheduler->schedule(request)) {
    return true;
  }
  if (!scheduler->accepting_new_requests()) {
    reply_service_not_ready(scheduler, controller);
  } else {
    controller->SetFailed("Schedule request failed!");
  }
  return false;
}

std::string proto_json(const google::protobuf::Message& message) {
  std::string json;
  google::protobuf::util::JsonPrintOptions options;
  options.add_whitespace = false;
  options.always_print_primitive_fields = true;
  auto status =
      google::protobuf::util::MessageToJsonString(message, &json, options);
  if (!status.ok()) {
    return message.DebugString();
  }
  return json;
}

AnthropicTracer make_anthropic_tracer(const std::shared_ptr<Request>& request) {
  AnthropicTracer::Sink sink;
  std::string service_request_id;
  if (request) {
    sink = request->trace_callback;
    service_request_id = request->correlation.request_uid();
  }
  return AnthropicTracer(
      std::move(sink), /*request_id=*/"", service_request_id);
}

nlohmann::json proto_value_to_json(const google::protobuf::Value& pb_value);

nlohmann::json proto_struct_to_json(const google::protobuf::Struct& pb_struct) {
  nlohmann::json result = nlohmann::json::object();
  for (const auto& field : pb_struct.fields()) {
    result[field.first] = proto_value_to_json(field.second);
  }
  return result;
}

nlohmann::json proto_value_to_json(const google::protobuf::Value& pb_value) {
  switch (pb_value.kind_case()) {
    case google::protobuf::Value::kNullValue:
      return nlohmann::json(nullptr);
    case google::protobuf::Value::kNumberValue:
      return nlohmann::json(pb_value.number_value());
    case google::protobuf::Value::kStringValue:
      return nlohmann::json(pb_value.string_value());
    case google::protobuf::Value::kBoolValue:
      return nlohmann::json(pb_value.bool_value());
    case google::protobuf::Value::kStructValue:
      return proto_struct_to_json(pb_value.struct_value());
    case google::protobuf::Value::kListValue: {
      nlohmann::json result = nlohmann::json::array();
      for (const auto& item : pb_value.list_value().values()) {
        result.push_back(proto_value_to_json(item));
      }
      return result;
    }
    case google::protobuf::Value::KIND_NOT_SET:
    default:
      return nlohmann::json(nullptr);
  }
}
std::vector<JsonTool> parse_tools_from_proto(
    const google::protobuf::RepeatedPtrField<::xllm::proto::Tool>&
        proto_tools) {
  std::vector<JsonTool> tools;
  tools.reserve(proto_tools.size());

  for (const auto& proto_tool : proto_tools) {
    JsonTool json_tool;
    json_tool.type = proto_tool.type();
    json_tool.function.name = proto_tool.function().name();
    json_tool.function.description = proto_tool.function().description();
    if (proto_tool.function().has_parameters()) {
      json_tool.function.parameters =
          proto_struct_to_json(proto_tool.function().parameters());
    } else {
      json_tool.function.parameters = nlohmann::json::object();
    }
    tools.emplace_back(std::move(json_tool));
  }
  return tools;
}
}  // namespace

XllmHttpServiceImpl::XllmHttpServiceImpl(const Options& options,
                                         Scheduler* scheduler)
    : options_(options), scheduler_(scheduler) {
  initialized_ = true;
  thread_pool_ = std::make_unique<ThreadPool>(options_.num_threads());
  request_tracer_ =
      std::make_unique<RequestTracer>(options_.enable_request_trace());
}

XllmHttpServiceImpl::~XllmHttpServiceImpl() {}

void XllmHttpServiceImpl::Hello(::google::protobuf::RpcController* controller,
                                const proto::HttpHelloRequest* request,
                                proto::HttpHelloResponse* response,
                                ::google::protobuf::Closure* done) {
  assert(initialized_);
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  LOG(INFO) << "Get request: " << request->ping();

  response->set_pong(request->ping());
}

namespace {
template <typename T>
void handle_non_stream_response(brpc::Controller* cntl,
                                std::shared_ptr<T> call_data) {
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  if (cntl->Failed()) {
    call_data->finish_with_error(cntl->ErrorText());
    LOG(ERROR) << "Fail to send stream generation, " << cntl->ErrorText();
    return;
  }
  call_data->write_and_finish(cntl->response_attachment().to_string());
}

// fire and forget
template <typename RequestProto>
void handle_first_send_request(brpc::Controller* cntl,
                               RequestProto* request_pb,
                               Scheduler* scheduler,
                               std::string instance_name,
                               std::string incarnation_id,
                               std::string service_request_id,
                               uint64_t attempt_seq,
                               std::shared_ptr<brpc::Channel> channel) {
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  std::unique_ptr<RequestProto> request_guard(request_pb);
  UNUSED_PARAMETER(channel);
  scheduler->record_direct_engine_evidence(
      instance_name, incarnation_id, !cntl->Failed());
  if (cntl->Failed()) {
    LOG(ERROR) << "Fail to send stream generation, " << cntl->ErrorText();
    scheduler->handle_attempt_dispatch_failure(
        service_request_id, attempt_seq, cntl->ErrorText());
  }
}

template <typename T>
bool dispatch_native_request(std::shared_ptr<T> call_data,
                             const Request& request,
                             Scheduler* scheduler) {
  using RequestProto =
      std::remove_cv_t<std::remove_reference_t<decltype(call_data->request())>>;
  auto request_pb = std::make_unique<RequestProto>(call_data->request());
  if (!set_request_execution_context(request_pb.get(), request)) {
    return false;
  }
  request_pb->mutable_routing()->set_prefill_name(request.routing.prefill_name);
  request_pb->mutable_routing()->set_decode_name(request.routing.decode_name);
  request_pb->mutable_routing()->set_prefill_incarnation_id(
      request.prefill_incarnation_id);
  request_pb->mutable_routing()->set_decode_incarnation_id(
      request.decode_incarnation_id);

  const std::shared_ptr<brpc::Channel> channel =
      scheduler->get_channel(request.routing.prefill_name);
  if (channel == nullptr || !request.request_deadline.has_value() ||
      !request.correlation.has_attempt_seq()) {
    return false;
  }
  const uint64_t remaining_ms = request.request_deadline->remaining_ms();
  if (remaining_ms == 0) {
    return false;
  }

  auto* redirect_cntl = new brpc::Controller();
  redirect_cntl->set_timeout_ms(static_cast<int>(std::min<uint64_t>(
      remaining_ms, static_cast<uint64_t>(std::numeric_limits<int>::max()))));
  google::protobuf::Closure* done =
      brpc::NewCallback(&handle_first_send_request<RequestProto>,
                        redirect_cntl,
                        request_pb.get(),
                        scheduler,
                        request.routing.prefill_name,
                        request.prefill_incarnation_id,
                        request.correlation.request_uid(),
                        request.correlation.attempt_seq(),
                        channel);

  xllm::proto::XllmAPIService_Stub stub(channel.get());
  if constexpr (std::is_same_v<T, CompletionCallData>) {
    stub.Completions(redirect_cntl, request_pb.get(), nullptr, done);
  } else if constexpr (std::is_same_v<T, ChatCallData> ||
                       std::is_same_v<T, AnthropicCallData>) {
    stub.ChatCompletions(redirect_cntl, request_pb.get(), nullptr, done);
  } else {
    delete done;
    delete redirect_cntl;
    return false;
  }
  request_pb.release();
  return true;
}

template <typename T>
class CustomProgressiveReader final : public brpc::ProgressiveReader {
 public:
  explicit CustomProgressiveReader(brpc::Controller* redirect_cntl,
                                   std::shared_ptr<T> call_data,
                                   Scheduler* scheduler,
                                   std::string instance_name,
                                   std::string incarnation_id,
                                   bool backend_success,
                                   std::shared_ptr<Request> request)
      : redirect_cntl_(redirect_cntl),
        call_data_(call_data),
        scheduler_(scheduler),
        instance_name_(std::move(instance_name)),
        incarnation_id_(std::move(incarnation_id)),
        backend_success_(backend_success),
        request_(std::move(request)) {}

  ~CustomProgressiveReader() override { delete redirect_cntl_; }

  // Called when one part was read.
  // Error returned is treated as *permanent* and the socket where the
  // data was read will be closed.
  // A temporary error may be handled by blocking this function, which
  // may block the HTTP parsing on the socket.
  butil::Status OnReadOnePart(const void* data, size_t length) override {
    call_data_->write(std::string(static_cast<const char*>(data), length));
    return butil::Status::OK();
  }

  // Called when there's nothing to read anymore. The `status' is a hint for
  // why this method is called.
  // - status.ok(): the message is complete and successfully consumed.
  // - otherwise: socket was broken or OnReadOnePart() failed.
  // This method will be called once and only once. No other methods will
  // be called after. User can release the memory of this object inside.
  void OnEndOfMessage(const butil::Status& status) override {
    scheduler_->record_direct_engine_evidence(
        instance_name_, incarnation_id_, status.ok() && backend_success_);
    bool terminal_resolved = true;
    if (status.ok() && backend_success_ && request_ != nullptr) {
      terminal_resolved = scheduler_->resolve_terminal_execution_hold(request_);
    }
    if (request_ != nullptr) {
      scheduler_->finish_request(
          request_->correlation.request_uid(),
          !status.ok() || !backend_success_ || !terminal_resolved);
    }
    delete this;
  }

 private:
  brpc::Controller* redirect_cntl_ = nullptr;
  std::shared_ptr<T> call_data_;
  Scheduler* scheduler_ = nullptr;
  std::string instance_name_;
  std::string incarnation_id_;
  bool backend_success_ = false;
  std::shared_ptr<Request> request_;
};

// Done callback for a streaming vLLM forward: once the response header has
// arrived, attach a progressive reader that relays the SSE body straight to
// the client. The reader takes ownership of redirect_cntl. `channel` is held
// only to keep the shared_ptr alive until the asynchronous call completes.
template <typename T>
void handle_vllm_stream_done(brpc::Controller* redirect_cntl,
                             std::shared_ptr<T> call_data,
                             Scheduler* scheduler,
                             std::string instance_name,
                             std::string incarnation_id,
                             std::shared_ptr<Request> request,
                             std::shared_ptr<brpc::Channel> channel) {
  UNUSED_PARAMETER(channel);
  const int32_t status_code = redirect_cntl->http_response().status_code();
  const bool success =
      !redirect_cntl->Failed() && status_code >= 200 && status_code < 300;
  scheduler->record_direct_engine_evidence(
      instance_name, incarnation_id, success);
  if (redirect_cntl->Failed()) {
    LOG(ERROR) << "Fail to forward to vLLM (stream): "
               << redirect_cntl->ErrorText();
    call_data->finish_with_error(redirect_cntl->ErrorText());
    if (request != nullptr) {
      scheduler->finish_request(request->correlation.request_uid(), true);
    }
    delete redirect_cntl;
    return;
  }
  if (success && request != nullptr &&
      !scheduler->confirm_generation_commit(request)) {
    call_data->finish_with_error("Provider GenerationCommit proof failed.");
    scheduler->finish_request(request->correlation.request_uid(), true);
    delete redirect_cntl;
    return;
  }
  redirect_cntl->ReadProgressiveAttachmentBy(
      new CustomProgressiveReader<T>(redirect_cntl,
                                     call_data,
                                     scheduler,
                                     instance_name,
                                     incarnation_id,
                                     success,
                                     std::move(request)));
}

// Done callback for a non-streaming vLLM forward. Delegates to the shared
// non-stream handler; `channel` is held only to keep the shared_ptr alive
// until the asynchronous call completes.
template <typename T>
void handle_vllm_non_stream_done(brpc::Controller* redirect_cntl,
                                 std::shared_ptr<T> call_data,
                                 Scheduler* scheduler,
                                 std::string instance_name,
                                 std::string incarnation_id,
                                 std::shared_ptr<Request> request,
                                 std::shared_ptr<brpc::Channel> channel) {
  UNUSED_PARAMETER(channel);
  const int32_t status_code = redirect_cntl->http_response().status_code();
  bool success =
      !redirect_cntl->Failed() && status_code >= 200 && status_code < 300;
  scheduler->record_direct_engine_evidence(
      instance_name, incarnation_id, success);
  if (success && request != nullptr) {
    success = scheduler->confirm_generation_commit(request) &&
              scheduler->resolve_terminal_execution_hold(request);
    if (!success) {
      redirect_cntl->SetFailed("Provider attempt proof failed.");
    }
  }
  handle_non_stream_response(redirect_cntl, call_data);
  if (request != nullptr) {
    scheduler->finish_request(request->correlation.request_uid(), !success);
  }
}

// Forward an OpenAI HTTP request to a vLLM backend, transparently relaying the
// raw client body (no proto re-serialization). Reuses
// handle_non_stream_response (non-stream) and CustomProgressiveReader (stream);
// the vLLM SSE format passes through unchanged.
template <typename T>
void handle_vllm(std::shared_ptr<T> call_data,
                 Scheduler* scheduler,
                 const std::string& target_name,
                 const std::string& target_incarnation_id,
                 const std::string& path,
                 bool is_post,
                 const std::string& body,
                 bool stream,
                 const xllm::proto::RequestCorrelation& correlation = {},
                 std::shared_ptr<Request> request = nullptr) {
  auto channel = scheduler->get_channel(target_name);
  if (channel == nullptr) {
    LOG(ERROR) << "No channel for vLLM instance: " << target_name;
    call_data->finish_with_error("vLLM backend instance is not available.");
    if (request != nullptr) {
      scheduler->finish_request(request->correlation.request_uid(), true);
    }
    return;
  }

  brpc::Controller* redirect_cntl = new brpc::Controller();
  redirect_cntl->http_request().uri() = "http://" + target_name + path;
  redirect_cntl->http_request().set_method(is_post ? brpc::HTTP_METHOD_POST
                                                   : brpc::HTTP_METHOD_GET);
  if (!correlation.request_uid().empty()) {
    redirect_cntl->http_request().SetHeader("X-Global-Request-ID",
                                            correlation.global_request_id());
    redirect_cntl->http_request().SetHeader("X-Request-ID",
                                            correlation.global_request_id());
    redirect_cntl->http_request().SetHeader("X-Trace-ID",
                                            correlation.trace_id());
    if (request == nullptr || !request->request_deadline.has_value()) {
      call_data->finish_with_error("Provider request deadline is unavailable.");
      if (request != nullptr) {
        scheduler->finish_request(request->correlation.request_uid(), true);
      }
      delete redirect_cntl;
      return;
    }
    const uint64_t remaining_deadline_ms =
        request->request_deadline->remaining_ms();
    if (remaining_deadline_ms == 0) {
      call_data->finish_with_error("Provider request deadline expired.");
      scheduler->finish_request(request->correlation.request_uid(), true);
      delete redirect_cntl;
      return;
    }
    if (!provider::set_vllm_agent_attempt_headers(redirect_cntl,
                                                  correlation.request_uid(),
                                                  correlation.attempt_seq(),
                                                  target_incarnation_id,
                                                  remaining_deadline_ms)) {
      call_data->finish_with_error("Provider attempt identity is invalid.");
      scheduler->finish_request(request->correlation.request_uid(), true);
      delete redirect_cntl;
      return;
    }
    redirect_cntl->set_timeout_ms(static_cast<int>(std::min<uint64_t>(
        remaining_deadline_ms,
        static_cast<uint64_t>(std::numeric_limits<int>::max()))));
  }
  if (is_post) {
    redirect_cntl->http_request().SetHeader("Content-Type", "application/json");
    redirect_cntl->request_attachment().append(body);
  }

  if (stream) {
    redirect_cntl->response_will_be_read_progressively();
    google::protobuf::Closure* done =
        brpc::NewCallback(&handle_vllm_stream_done<T>,
                          redirect_cntl,
                          call_data,
                          scheduler,
                          target_name,
                          target_incarnation_id,
                          request,
                          channel);
    channel->CallMethod(nullptr, redirect_cntl, nullptr, nullptr, done);
  } else {
    google::protobuf::Closure* done =
        brpc::NewCallback(&handle_vllm_non_stream_done<T>,
                          redirect_cntl,
                          call_data,
                          scheduler,
                          target_name,
                          target_incarnation_id,
                          request,
                          channel);
    channel->CallMethod(nullptr, redirect_cntl, nullptr, nullptr, done);
  }
}
}  // namespace

namespace {

constexpr char kInferContentLength[] = "Infer-Content-Length";
constexpr char kContentLength[] = "Content-Length";

size_t GetJsonContentLength(const brpc::Controller* ctrl) {
  const auto parse_length = [](const std::string& value) -> size_t {
    try {
      return std::stoul(value);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Invalid Content-Length value: " << value
                 << ", error: " << e.what();
      return (size_t)-1L;
    }
  };

  const auto infer_content_len =
      ctrl->http_request().GetHeader(kInferContentLength);
  if (infer_content_len != nullptr) {
    return parse_length(*infer_content_len);
  }

  const auto content_len = ctrl->http_request().GetHeader(kContentLength);
  if (content_len != nullptr) {
    return parse_length(*content_len);
  }

  LOG(ERROR) << "Content-Length header is missing.";
  return (size_t)-1L;
}

}  // namespace

template <typename T>
void XllmHttpServiceImpl::handle(std::shared_ptr<T> call_data,
                                 std::shared_ptr<Request> request) {
  request->first_output_retry_budget =
      std::make_unique<FirstOutputRetryBudget>(FirstOutputRetryBudget::Config{
          .max_attempt_retries = options_.max_first_output_attempt_retries(),
          .max_wasted_device_ms =
              options_.max_nonstream_retry_wasted_device_ms(),
          .min_remaining_deadline_ms =
              options_.min_first_output_retry_remaining_ms(),
      });
  request->retry_dispatch_callback =
      [call_data, scheduler = scheduler_](const Request& retry_request) {
        return dispatch_native_request(call_data, retry_request, scheduler);
      };
  // record request
  bool success = scheduler_->record_new_request(call_data, request);
  if (!success) {
    LOG(ERROR) << "rpc service add new request error: "
               << request->correlation.request_uid();
    call_data->finish_with_error("Internal runtime error.");
    return;
  }

  if (!request->retry_dispatch_callback(*request)) {
    scheduler_->handle_attempt_dispatch_failure(
        request->correlation.request_uid(),
        request->correlation.attempt_seq(),
        "Native dispatch could not be started");
  }
}

template <typename T>
std::shared_ptr<Request> XllmHttpServiceImpl::generate_request(
    T* req_pb,
    brpc::Controller* controller,
    xllm::proto::ApiKind api_kind,
    const std::string& payload_schema,
    const std::string& payload) {
  auto request = std::make_shared<Request>();
  request->model = req_pb->model();
  request->correlation =
      observability::make_request_correlation(correlation_input(controller));
  request->first_event_retry_policy =
      xllm::FirstEventRetryPolicy::from_durations_ms(
          static_cast<uint64_t>(options_.p_first_event_retry_ub_ms()),
          static_cast<uint64_t>(options_.first_event_dispatch_margin_ms()));
  request->request_deadline_present = req_pb->has_remaining_deadline_ms();
  if (request->request_deadline_present) {
    request->request_deadline = xllm::RequestDeadline::from_remaining_ms(
        req_pb->remaining_deadline_ms());
  } else {
    request->request_deadline_present = true;
    request->request_deadline = xllm::RequestDeadline::from_remaining_ms(
        static_cast<uint64_t>(options_.default_request_deadline_ms()));
  }

  if (req_pb->has_stream()) {
    request->stream = req_pb->stream();
  }

  if (req_pb->has_stream_options()) {
    request->include_usage = req_pb->stream_options().include_usage();
  }

  if (options_.enable_request_trace()) {
    request->trace_callback =
        [this, service_request_id = request->correlation.request_uid()](
            const std::string& message) {
          request_tracer_->log(service_request_id, message);
        };
  }

  if ((req_pb->has_ttft_slo_ms() && req_pb->ttft_slo_ms() <= 0) ||
      (req_pb->has_tpot_slo_ms() && req_pb->tpot_slo_ms() <= 0)) {
    LOG(ERROR) << "Request SLO durations must be positive.";
    return nullptr;
  }
  if (!request->request_deadline.has_value()) {
    LOG(ERROR) << "Canonical request has no local deadline.";
    return nullptr;
  }

  const uint32_t n = req_pb->has_n() ? req_pb->n() : 1;
  provider::CanonicalRequestInput canonical_input;
  canonical_input.correlation = request->correlation;
  canonical_input.api_kind = api_kind;
  canonical_input.model_revision = req_pb->model();
  canonical_input.payload_schema = payload_schema;
  canonical_input.payload = payload;
  canonical_input.effective_max_new_tokens =
      req_pb->has_max_tokens() ? req_pb->max_tokens() : 5120;
  canonical_input.n = n;
  canonical_input.best_of = req_pb->has_best_of() ? req_pb->best_of() : n;
  canonical_input.priority =
      req_pb->has_priority() ? static_cast<int32_t>(req_pb->priority()) : 0;
  canonical_input.remaining_deadline_ms =
      request->request_deadline->remaining_ms();
  if (req_pb->has_ttft_slo_ms()) {
    canonical_input.ttft_slo_ms = static_cast<uint64_t>(req_pb->ttft_slo_ms());
  }
  if (req_pb->has_tpot_slo_ms()) {
    canonical_input.tpot_slo_ms = static_cast<uint64_t>(req_pb->tpot_slo_ms());
  }
  xllm::proto::CanonicalRequest canonical;
  const provider::ContractResult canonical_result =
      provider::build_canonical_request(canonical_input, &canonical);
  if (!canonical_result.ok()) {
    LOG(ERROR) << "Canonical request validation failed: "
               << canonical_result.message();
    return nullptr;
  }
  request->canonical_request = std::move(canonical);

  return request;
}

namespace {
void handle_get_model_response(brpc::Controller* cntl,
                               std::shared_ptr<CompletionCallData> call_data,
                               google::protobuf::Closure* done,
                               xllm::proto::ModelListResponse* resp_pb) {
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  std::unique_ptr<xllm::proto::ModelListResponse> resp_pb_guard(resp_pb);

  if (cntl->Failed()) {
    LOG(ERROR) << "Fail to send stream generation, " << cntl->ErrorText();
    call_data->finish_with_error(cntl->ErrorText());
    return;
  }
  std::string err_msg;
  std::string json_output;
  if (!json2pb::ProtoMessageToJson(*resp_pb, &json_output, &err_msg)) {
    call_data->finish_with_error(err_msg);
    LOG(ERROR) << "ProtoMessageToJson failed: " << err_msg;
    return;
  }
  LOG(INFO) << "ProtoMessageToJson: " << json_output;
  call_data->write_and_finish(json_output);
}
}  // namespace

void XllmHttpServiceImpl::get_serving_models(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  assert(initialized_);
  ClosureGuard done_guard(done);
  auto cntl = reinterpret_cast<brpc::Controller*>(controller);

  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    cntl->SetFailed("brpc request | respose | controller is null");
    return;
  }
  if (!require_accepting_new_requests(scheduler_, cntl)) {
    return;
  }
  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<::xllm::proto::ModelListRequest>(
          arena);

  // auto call_data = std::make_shared<StreamCallData>(cntl, false,
  // done_guard.release());
  auto call_data = std::make_shared<CompletionCallData>(
      cntl, false, done_guard.release(), nullptr, nullptr);

  auto service_request = std::make_shared<Request>();
  if (!schedule_request(scheduler_, service_request, cntl)) {
    LOG(ERROR) << "Schedule request failed!";
    return;
  }

  // vLLM backend: relay GET /v1/models straight through.
  if (service_request->provider_id == xllm::proto::PROVIDER_ID_VLLM_ASCEND) {
    handle_vllm(call_data,
                scheduler_,
                service_request->routing.prefill_name,
                service_request->prefill_incarnation_id,
                "/v1/models",
                /*is_post=*/false,
                /*body=*/"",
                /*stream=*/false);
    return;
  }

  brpc::Channel* channel_ptr =
      scheduler_->get_channel(service_request->routing.prefill_name).get();

  xllm::proto::XllmAPIService_Stub stub(channel_ptr);
  brpc::Controller* redirect_cntl = new brpc::Controller();
  auto* resp_pb = new xllm::proto::ModelListResponse();
  google::protobuf::Closure* done_callback = brpc::NewCallback(
      &handle_get_model_response, redirect_cntl, call_data, done, resp_pb);
  stub.Models(redirect_cntl, req_pb, resp_pb, done_callback);
}

void XllmHttpServiceImpl::Completions(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  assert(initialized_);
  ClosureGuard done_guard(done);
  auto cntl = reinterpret_cast<brpc::Controller*>(controller);

  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    cntl->SetFailed("brpc request | respose | controller is null");
    return;
  }
  if (!require_accepting_new_requests(scheduler_, cntl)) {
    return;
  }

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<::xllm::proto::CompletionRequest>(
          arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<::xllm::proto::CompletionResponse>(
          arena);

  std::string attachment = std::move(cntl->request_attachment().to_string());
  std::string error;
  auto st = json2pb::JsonToProtoMessage(attachment, req_pb, &error);
  if (!st) {
    cntl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  auto service_request = generate_request(req_pb,
                                          cntl,
                                          xllm::proto::API_KIND_COMPLETIONS,
                                          provider::kOpenAiHttpJsonSchema,
                                          attachment);
  if (service_request == nullptr) {
    cntl->SetFailed("Canonical request validation failed!");
    return;
  }

  if (!req_pb->prompt().empty()) {
    service_request->prompt = req_pb->prompt();
    // select instance for request
    if (!schedule_request(scheduler_, service_request, cntl)) {
      LOG(ERROR) << "Schedule request failed!";
      return;
    }
  } else {
    cntl->SetFailed("Prompt is empty!");
    LOG(ERROR) << "Prompt is empty!";
    return;
  }

  // vLLM backend: relay the raw client JSON over HTTP, skip xllm-only fields.
  if (service_request->provider_id == xllm::proto::PROVIDER_ID_VLLM_ASCEND) {
    const std::string& provider_payload =
        service_request->execution_plan.has_value()
            ? service_request->execution_plan->provider_payload()
            : attachment;
    auto call_data = std::make_shared<CompletionCallData>(
        cntl, service_request->stream, done_guard.release(), req_pb, resp_pb);
    if (!scheduler_->record_new_request(call_data, service_request)) {
      call_data->finish_with_error(
          "Provider request safety guards are unavailable.");
      return;
    }
    handle_vllm(call_data,
                scheduler_,
                service_request->routing.prefill_name,
                service_request->prefill_incarnation_id,
                "/v1/completions",
                /*is_post=*/true,
                provider_payload,
                service_request->stream,
                service_request->correlation,
                service_request);
    return;
  }

  // update request protobuf
  if (!set_request_execution_context(req_pb, *service_request)) {
    cntl->SetFailed("Invalid or expired request context before dispatch.");
    return;
  }
  req_pb->set_source_xservice_addr(options_.service_name());
  req_pb->mutable_token_ids()->Add(service_request->token_ids.begin(),
                                   service_request->token_ids.end());
  req_pb->mutable_routing()->set_prefill_name(
      service_request->routing.prefill_name);
  req_pb->mutable_routing()->set_decode_name(
      service_request->routing.decode_name);
  req_pb->mutable_routing()->set_prefill_incarnation_id(
      service_request->prefill_incarnation_id);
  req_pb->mutable_routing()->set_decode_incarnation_id(
      service_request->decode_incarnation_id);

  auto call_data = std::make_shared<CompletionCallData>(
      cntl, service_request->stream, done_guard.release(), req_pb, resp_pb);
  handle(call_data, service_request);
}

void XllmHttpServiceImpl::ChatCompletions(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  assert(initialized_);
  ClosureGuard done_guard(done);
  auto cntl = reinterpret_cast<brpc::Controller*>(controller);

  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    cntl->SetFailed("brpc request | respose | controller is null");
    return;
  }
  if (!require_accepting_new_requests(scheduler_, cntl)) {
    return;
  }

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<::xllm::proto::ChatRequest>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<::xllm::proto::ChatResponse>(
          arena);

  auto content_len = GetJsonContentLength(cntl);
  if (content_len == (size_t)-1L) {
    cntl->SetFailed("Content-Length header is missing or invalid.");
    return;
  }
  std::string attachment;
  cntl->request_attachment().copy_to(&attachment, content_len, 0);

  auto chat_json = normalize_chat_json(attachment);
  if (!chat_json.ok) {
    cntl->SetFailed(chat_json.error);
    LOG(ERROR) << "normalize chat json failed: " << chat_json.error;
    return;
  }

  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;
  auto status = google::protobuf::util::JsonStringToMessage(
      chat_json.json, req_pb, options);
  if (!status.ok()) {
    cntl->SetFailed(status.ToString());
    LOG(ERROR) << "parse json to proto failed: " << status.ToString();
    return;
  }

  auto service_request =
      generate_request(req_pb,
                       cntl,
                       xllm::proto::API_KIND_CHAT_COMPLETIONS,
                       provider::kOpenAiHttpJsonSchema,
                       attachment);
  if (service_request == nullptr) {
    cntl->SetFailed("Canonical request validation failed!");
    return;
  }

  if (req_pb->messages_size() > 0) {
    service_request->messages.reserve(req_pb->messages_size());
    for (const auto& message : req_pb->messages()) {
      Message msg(message.role(), message.content());
      if (message.has_reasoning_content()) {
        msg.reasoning_content = message.reasoning_content();
      }
      if (!message.tool_call_id().empty()) {
        msg.tool_call_id = message.tool_call_id();
      }
      if (message.tool_calls_size() > 0) {
        Message::ToolCallVec tool_calls;
        tool_calls.reserve(message.tool_calls_size());
        for (const auto& tool_call : message.tool_calls()) {
          Message::ToolCall parsed_tool_call;
          parsed_tool_call.id = tool_call.id();
          parsed_tool_call.type = tool_call.type();
          parsed_tool_call.function.name = tool_call.function().name();
          parsed_tool_call.function.arguments =
              tool_call.function().arguments();
          tool_calls.emplace_back(std::move(parsed_tool_call));
        }
        msg.tool_calls = std::move(tool_calls);
      }
      service_request->messages.emplace_back(std::move(msg));
    }
    if (req_pb->has_chat_template_kwargs()) {
      service_request->chat_template_kwargs =
          proto_struct_to_json(req_pb->chat_template_kwargs());
    }
    service_request->tools = parse_tools_from_proto(req_pb->tools());
    if (req_pb->has_tool_choice()) {
      service_request->tool_choice = req_pb->tool_choice();
    }

    if (!schedule_request(scheduler_, service_request, cntl)) {
      LOG(ERROR) << "Schedule request failed!";
      return;
    }
  } else {
    cntl->SetFailed("Messages is empty!");
    LOG(ERROR) << "Messages is empty!";
    return;
  }

  // vLLM backend: relay the raw client JSON over HTTP, skip xllm-only fields.
  if (service_request->provider_id == xllm::proto::PROVIDER_ID_VLLM_ASCEND) {
    const std::string& provider_payload =
        service_request->execution_plan.has_value()
            ? service_request->execution_plan->provider_payload()
            : attachment;
    auto call_data = std::make_shared<ChatCallData>(
        cntl, service_request->stream, done_guard.release(), req_pb, resp_pb);
    if (!scheduler_->record_new_request(call_data, service_request)) {
      call_data->finish_with_error(
          "Provider request safety guards are unavailable.");
      return;
    }
    handle_vllm(call_data,
                scheduler_,
                service_request->routing.prefill_name,
                service_request->prefill_incarnation_id,
                "/v1/chat/completions",
                /*is_post=*/true,
                provider_payload,
                service_request->stream,
                service_request->correlation,
                service_request);
    return;
  }

  // update request protobuf
  if (!set_request_execution_context(req_pb, *service_request)) {
    cntl->SetFailed("Invalid or expired request context before dispatch.");
    return;
  }
  req_pb->set_source_xservice_addr(options_.service_name());
  req_pb->mutable_token_ids()->Add(service_request->token_ids.begin(),
                                   service_request->token_ids.end());
  req_pb->mutable_routing()->set_prefill_name(
      service_request->routing.prefill_name);
  req_pb->mutable_routing()->set_decode_name(
      service_request->routing.decode_name);
  req_pb->mutable_routing()->set_prefill_incarnation_id(
      service_request->prefill_incarnation_id);
  req_pb->mutable_routing()->set_decode_incarnation_id(
      service_request->decode_incarnation_id);

  auto call_data = std::make_shared<ChatCallData>(
      cntl, service_request->stream, done_guard.release(), req_pb, resp_pb);
  handle(call_data, service_request);
}

void XllmHttpServiceImpl::AnthropicMessages(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  assert(initialized_);
  ClosureGuard done_guard(done);
  auto cntl = reinterpret_cast<brpc::Controller*>(controller);

  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    cntl->SetFailed("brpc request | respose | controller is null");
    return;
  }
  if (!require_accepting_new_requests(scheduler_, cntl)) {
    return;
  }

  auto arena = response->GetArena();
  auto anthropic_req_pb = google::protobuf::Arena::CreateMessage<
      ::xllm::proto::AnthropicMessagesRequest>(arena);
  auto req_pb =
      google::protobuf::Arena::CreateMessage<::xllm::proto::ChatRequest>(arena);
  auto resp_pb = google::protobuf::Arena::CreateMessage<
      ::xllm::proto::AnthropicMessagesResponse>(arena);

  auto content_len = GetJsonContentLength(cntl);
  if (content_len == (size_t)-1L) {
    cntl->SetFailed("Content-Length header is missing or invalid.");
    return;
  }
  std::string attachment;
  cntl->request_attachment().copy_to(&attachment, content_len, 0);

  auto parse_result = parse_anthropic_json(attachment, anthropic_req_pb);
  if (!parse_result.ok) {
    cntl->SetFailed(parse_result.error);
    LOG(ERROR) << "parse anthropic json failed: " << parse_result.error;
    return;
  }

  ChatMessages messages;
  auto adapt_result = fill_chat_req(*anthropic_req_pb, req_pb, &messages);
  if (!adapt_result.ok) {
    cntl->SetFailed(adapt_result.error);
    LOG(ERROR) << "adapt anthropic request failed: " << adapt_result.error;
    return;
  }
  req_pb->set_request_id(new_anthropic_id());

  auto service_request =
      generate_request(req_pb,
                       cntl,
                       xllm::proto::API_KIND_ANTHROPIC_MESSAGES,
                       provider::kAnthropicHttpJsonSchema,
                       attachment);
  if (service_request == nullptr) {
    cntl->SetFailed("Canonical request validation failed!");
    return;
  }
  auto tracer = make_anthropic_tracer(service_request);
  tracer.trace("raw_http_request", attachment);
  tracer.trace("anthropic_request_pb", proto_json(*anthropic_req_pb));
  tracer.trace("chat_request_after_adapt", proto_json(*req_pb));
  service_request->messages = std::move(messages);
  service_request->tools = parse_tools_from_proto(req_pb->tools());
  if (req_pb->has_tool_choice()) {
    service_request->tool_choice = req_pb->tool_choice();
  }

  if (!schedule_request(scheduler_, service_request, cntl)) {
    LOG(ERROR) << "Schedule request failed!";
    return;
  }

  if (!set_request_execution_context(req_pb, *service_request)) {
    cntl->SetFailed("Invalid or expired request context before dispatch.");
    return;
  }
  req_pb->set_source_xservice_addr(options_.service_name());
  req_pb->mutable_token_ids()->Add(service_request->token_ids.begin(),
                                   service_request->token_ids.end());
  req_pb->mutable_routing()->set_prefill_name(
      service_request->routing.prefill_name);
  req_pb->mutable_routing()->set_decode_name(
      service_request->routing.decode_name);
  req_pb->mutable_routing()->set_prefill_incarnation_id(
      service_request->prefill_incarnation_id);
  req_pb->mutable_routing()->set_decode_incarnation_id(
      service_request->decode_incarnation_id);

  auto call_data =
      std::make_shared<AnthropicCallData>(cntl,
                                          service_request->stream,
                                          done_guard.release(),
                                          req_pb,
                                          resp_pb,
                                          service_request->trace_callback);
  if (!call_data->x_request_id.empty()) {
    req_pb->set_x_request_id(call_data->x_request_id);
  }
  if (!call_data->x_request_time.empty()) {
    req_pb->set_x_request_time(call_data->x_request_time);
  }
  handle(call_data, service_request);
}

void XllmHttpServiceImpl::Embeddings(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  assert(initialized_);
  ClosureGuard done_guard(done);
  auto cntl = reinterpret_cast<brpc::Controller*>(controller);

  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    cntl->SetFailed("brpc request | respose | controller is null");
    return;
  }
  if (!require_accepting_new_requests(scheduler_, cntl)) {
    return;
  }

  cntl->SetFailed("not support Embeddings");
  return;
}

void XllmHttpServiceImpl::Models(::google::protobuf::RpcController* controller,
                                 const proto::HttpRequest* request,
                                 proto::HttpResponse* response,
                                 ::google::protobuf::Closure* done) {
  get_serving_models(controller, request, response, done);
}

void XllmHttpServiceImpl::Metrics(::google::protobuf::RpcController* controller,
                                  const proto::HttpRequest* request,
                                  proto::HttpResponse* response,
                                  ::google::protobuf::Closure* done) {
  ClosureGuard done_guard(done);
  // TODO: implement metrics endpoint
}

void XllmHttpServiceImpl::Livez(::google::protobuf::RpcController* controller,
                                const proto::HttpRequest* request,
                                proto::HttpResponse* response,
                                ::google::protobuf::Closure* done) {
  ClosureGuard done_guard(done);
  brpc::Controller* cntl = reinterpret_cast<brpc::Controller*>(controller);
  if (cntl == nullptr || request == nullptr || response == nullptr) {
    return;
  }
  const HealthResponse health = make_liveness_response();
  reply_json(cntl, health.status_code, health.body);
}

void XllmHttpServiceImpl::Readyz(::google::protobuf::RpcController* controller,
                                 const proto::HttpRequest* request,
                                 proto::HttpResponse* response,
                                 ::google::protobuf::Closure* done) {
  ClosureGuard done_guard(done);
  brpc::Controller* cntl = reinterpret_cast<brpc::Controller*>(controller);
  if (cntl == nullptr || request == nullptr || response == nullptr) {
    return;
  }
  const provider::ReadinessSnapshot readiness = scheduler_->readiness_status();
  const HealthResponse health = make_readiness_response(readiness);
  reply_json(cntl, health.status_code, health.body);
}

void XllmHttpServiceImpl::Heartbeat(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  ClosureGuard done_guard(done);
  auto cntl = reinterpret_cast<brpc::Controller*>(controller);
  if (!cntl) {
    return;
  }

  auto reply = [&](int32_t status_code, const char* body) {
    cntl->http_response().set_status_code(status_code);
    cntl->http_response().set_content_type("application/json");
    cntl->response_attachment().append(body);
  };

  // Optional static shared-token auth; skip the check when no token configured
  // (backward compatible with deployments that don't set internal_api_token).
  const std::string& expected = options_.internal_api_token();
  if (!expected.empty()) {
    const std::string* got = cntl->http_request().GetHeader("X-Internal-Token");
    if (got == nullptr || *got != expected) {
      LOG(WARNING) << "Heartbeat rejected: invalid X-Internal-Token";
      reply(401, "{\"error\":\"invalid internal token\"}");
      return;
    }
  }

  // Body is JSON of proto HeartbeatRequest (snake_case field names accepted).
  proto::HeartbeatRequest req;
  const std::string attachment = cntl->request_attachment().to_string();
  google::protobuf::util::JsonParseOptions opts;
  opts.ignore_unknown_fields = true;
  const auto status =
      google::protobuf::util::JsonStringToMessage(attachment, &req, opts);
  if (!status.ok()) {
    LOG(ERROR) << "Heartbeat parse failed: " << status.ToString();
    reply(400, "{\"error\":\"invalid heartbeat json\"}");
    return;
  }
  if (req.name().empty()) {
    reply(400, "{\"error\":\"missing instance name\"}");
    return;
  }

  // Reuse the RPC heartbeat schema/path so HTTP sidecars and brpc
  // backends update scheduler state consistently: liveness/incarnation,
  // LoadMetrics, and LatencyMetrics. vLLM cache_event is empty.
  if (!scheduler_ || !scheduler_->handle_instance_heartbeat(&req)) {
    // Unknown instance or stale incarnation -> ask the sidecar to re-register.
    reply(409, "{\"error\":\"instance not registered or stale incarnation\"}");
    return;
  }

  reply(200, "{\"ok\":true}");
}

}  // namespace xllm_service
