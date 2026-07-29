/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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
#include <brpc/http_status_code.h>
#include <brpc/progressive_reader.h>
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>
#include <json2pb/json_to_pb.h>
#include <json2pb/pb_to_json.h>

#include <cctype>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <nlohmann/json.hpp>

#include "backend_http/request_context.h"
#include "chat.pb.h"
#include "common/call_data.h"
#include "common/closure_guard.h"
#include "common/utils.h"
#include "common/xllm/status.h"
#include "common/xllm/uuid.h"
#include "completion.pb.h"
#include "runtime/runtime_state.h"
#include "scheduler/scheduler.h"
#include "telemetry/prometheus_metrics.h"
#include "xllm_service.pb.h"

namespace xllm_service {

namespace {
thread_local llm::ShortUUID short_uuid;
std::string generate_service_request_id(const std::string& method) {
  std::stringstream ss;
  ss << method << "-";
  ss << std::this_thread::get_id();
  ss << "-";
  ss << short_uuid.random();
  return ss.str();
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
                                         Scheduler* scheduler,
                                         RuntimeState& runtime_state)
    : options_(options),
      scheduler_(scheduler),
      runtime_state_(runtime_state) {
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
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  LOG(INFO) << "Get request: " << request->ping();

  response->set_pong(request->ping());
}

void XllmHttpServiceImpl::Health(::google::protobuf::RpcController* controller,
                                 const proto::HttpRequest* request,
                                 proto::HttpResponse* response,
                                 ::google::protobuf::Closure* done) {
  assert(initialized_);
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    if (controller) {
      reinterpret_cast<brpc::Controller*>(controller)->SetFailed(
          "brpc request | response | controller is null");
    }
    return;
  }

  brpc::Controller* cntl = reinterpret_cast<brpc::Controller*>(controller);
  const RuntimeHealthSnapshot snapshot = runtime_state_.health_snapshot();
  cntl->http_response().set_content_type("text/plain");
  cntl->http_response().set_status_code(
      snapshot.ready ? brpc::HTTP_STATUS_OK
                     : brpc::HTTP_STATUS_SERVICE_UNAVAILABLE);
  cntl->response_attachment().append(snapshot.ready ? "ok\n"
                                                    : "unavailable\n");
}

void XllmHttpServiceImpl::Liveness(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  assert(initialized_);
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    if (controller) {
      reinterpret_cast<brpc::Controller*>(controller)->SetFailed(
          "brpc request | response | controller is null");
    }
    return;
  }

  brpc::Controller* cntl = reinterpret_cast<brpc::Controller*>(controller);
  const RuntimeHealthSnapshot snapshot = runtime_state_.health_snapshot();
  cntl->http_response().set_content_type("text/plain");
  cntl->http_response().set_status_code(
      snapshot.live ? brpc::HTTP_STATUS_OK
                    : brpc::HTTP_STATUS_SERVICE_UNAVAILABLE);
  cntl->response_attachment().append(snapshot.live ? "ok\n" : "stopped\n");
}

void XllmHttpServiceImpl::Readiness(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  assert(initialized_);
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    if (controller) {
      reinterpret_cast<brpc::Controller*>(controller)->SetFailed(
          "brpc request | response | controller is null");
    }
    return;
  }

  brpc::Controller* cntl = reinterpret_cast<brpc::Controller*>(controller);
  const RuntimeHealthSnapshot snapshot = runtime_state_.health_snapshot();
  cntl->http_response().set_content_type("text/plain");
  cntl->http_response().set_status_code(
      snapshot.ready ? brpc::HTTP_STATUS_OK
                     : brpc::HTTP_STATUS_SERVICE_UNAVAILABLE);
  cntl->response_attachment().append(snapshot.ready ? "ready\n"
                                                    : snapshot.reason + "\n");
}

bool XllmHttpServiceImpl::ensure_backend_ready(
    brpc::Controller* controller) const {
  const RuntimeHealthSnapshot snapshot = runtime_state_.health_snapshot();
  if (snapshot.ready) {
    return true;
  }

  nlohmann::json error = {
      {"error",
       {{"message", snapshot.reason.empty() ? "backend is not ready"
                                             : snapshot.reason},
        {"type", "service_unavailable"}}}};
  controller->http_response().set_content_type("application/json");
  controller->http_response().set_status_code(
      brpc::HTTP_STATUS_SERVICE_UNAVAILABLE);
  controller->response_attachment().append(error.dump());
  return false;
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
void handle_first_send_request(brpc::Controller* cntl,
                               Scheduler* scheduler,
                               std::string service_request_id) {
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  if (cntl->Failed()) {
    LOG(ERROR) << "Fail to send stream generation, " << cntl->ErrorText();
    scheduler->handle_transport_failure(service_request_id,
                                        cntl->ErrorText());
    return;
  }
}

template <typename T>
class CustomProgressiveReader : public brpc::ProgressiveReader {
 public:
  explicit CustomProgressiveReader(brpc::Controller* redirect_cntl,
                                   std::shared_ptr<T> call_data)
      : redirect_cntl_(redirect_cntl), call_data_(call_data) {}

  virtual ~CustomProgressiveReader() { delete redirect_cntl_; }

  // Called when one part was read.
  // Error returned is treated as *permanent* and the socket where the
  // data was read will be closed.
  // A temporary error may be handled by blocking this function, which
  // may block the HTTP parsing on the socket.
  virtual butil::Status OnReadOnePart(const void* data, size_t length) {
    call_data_->write(std::string((char*)data, length));
    return butil::Status::OK();
  }

  // Called when there's nothing to read anymore. The `status' is a hint for
  // why this method is called.
  // - status.ok(): the message is complete and successfully consumed.
  // - otherwise: socket was broken or OnReadOnePart() failed.
  // This method will be called once and only once. No other methods will
  // be called after. User can release the memory of this object inside.
  virtual void OnEndOfMessage(const butil::Status& status) { delete this; }

 private:
  brpc::Controller* redirect_cntl_ = nullptr;
  std::shared_ptr<T> call_data_;
};
}  // namespace

namespace {

constexpr char kInferContentLength[] = "Infer-Content-Length";
constexpr char kContentLength[] = "Content-Length";

bool TryParseContentLength(const char* header_name,
                           const std::string& header_value,
                           size_t attachment_size,
                           size_t* content_len) {
  try {
    size_t parsed_size = 0;
    const auto parsed_value = std::stoull(header_value, &parsed_size, 10);
    while (parsed_size < header_value.size() &&
           std::isspace(static_cast<unsigned char>(header_value[parsed_size]))) {
      ++parsed_size;
    }
    if (parsed_size != header_value.size() ||
        parsed_value > std::numeric_limits<size_t>::max()) {
      LOG(WARNING) << "Invalid " << header_name
                   << " header value: " << header_value;
      return false;
    }

    *content_len = static_cast<size_t>(parsed_value);
    if (*content_len > attachment_size) {
      LOG(WARNING) << header_name << " header value " << *content_len
                   << " exceeds request attachment size " << attachment_size
                   << ", use attachment size instead.";
      *content_len = attachment_size;
    }
    return true;
  } catch (const std::exception& e) {
    LOG(WARNING) << "Invalid " << header_name
                 << " header value: " << header_value << ", error: "
                 << e.what();
    return false;
  }
}

size_t GetJsonContentLength(const brpc::Controller* ctrl,
                            size_t attachment_size) {
  const auto infer_content_len =
      ctrl->http_request().GetHeader(kInferContentLength);
  size_t content_len = 0;
  if (infer_content_len != nullptr &&
      TryParseContentLength(kInferContentLength,
                            *infer_content_len,
                            attachment_size,
                            &content_len)) {
    return content_len;
  }

  const auto content_len_header = ctrl->http_request().GetHeader(kContentLength);
  if (content_len_header != nullptr &&
      TryParseContentLength(kContentLength,
                            *content_len_header,
                            attachment_size,
                            &content_len)) {
    return content_len;
  }

  if (attachment_size > 0) {
    LOG(WARNING) << "Content-Length header is missing, use request attachment "
                    "size instead: "
                 << attachment_size;
  }
  return attachment_size;
}

}  // namespace

template <typename T>
void XllmHttpServiceImpl::handle(std::shared_ptr<T> call_data,
                                 std::shared_ptr<Request> request) {
  // record request
  auto& req_pb = call_data->request();
  bool success = scheduler_->record_new_request(call_data, request);
  if (!success) {
    LOG(ERROR) << "rpc service add new request error: "
               << request->service_request_id;
    call_data->finish_with_error("Internal runtime error.");
    return;
  }

  // async redistribute the request and wait the response
  // TODO: optimize the thread pool to async mode.
  auto& target_uri = request->routing.prefill_name;
  brpc::Channel* channel_ptr = scheduler_->get_channel(target_uri).get();
  // use stub
  xllm::proto::XllmAPIService_Stub stub(channel_ptr);
  // xllm::proto::Status* resp_pb = new xllm::proto::Status();
  brpc::Controller* redirect_cntl = new brpc::Controller();
  google::protobuf::Closure* done =
      brpc::NewCallback(&handle_first_send_request,
                        redirect_cntl,
                        scheduler_,
                        request->service_request_id);

  if constexpr (std::is_same_v<T, CompletionCallData>) {
    stub.Completions(redirect_cntl, &req_pb, nullptr, done);
  } else if constexpr (std::is_same_v<T, ChatCallData>) {
    stub.ChatCompletions(redirect_cntl, &req_pb, nullptr, done);
  } else {
    delete redirect_cntl;
    delete done;
    LOG(ERROR) << "Unknown call_data type";
  }
}

template <typename T>
std::shared_ptr<Request> XllmHttpServiceImpl::generate_request(
    T* req_pb,
    const brpc::Controller& controller,
    const std::string& method) {
  std::shared_ptr<Request> request = std::make_shared<Request>();
  request->request_context = parse_request_context(controller);

  const std::string request_body_model = req_pb->model();
  request->model = resolve_effective_model_name(request_body_model,
                                                request->request_context);
  if (request->model != request_body_model) {
    req_pb->set_model(request->model);
  }

  // TODO: add `created_time` fileds etc.
  // create xllm_service request_id: service_request_id
  request->service_request_id = generate_service_request_id(method);

  if (req_pb->has_stream()) {
    request->stream = req_pb->stream();
  }

  if (req_pb->has_stream_options()) {
    request->include_usage = req_pb->stream_options().include_usage();
  }

  if (options_.enable_request_trace()) {
    request->trace_callback =
        [this, service_request_id = request->service_request_id](
            const std::string& message) {
          request_tracer_->log(service_request_id, message);
        };
  }

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
    LOG(ERROR) << "brpc request | response | controller is null";
    if (cntl) {
      cntl->SetFailed("brpc request | response | controller is null");
    }
    return;
  }
  if (!ensure_backend_ready(cntl)) {
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
  if (!scheduler_->schedule(service_request)) {
    cntl->SetFailed("Schedule request failed!");
    LOG(ERROR) << "Schedule request failed!";
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
    LOG(ERROR) << "brpc request | response | controller is null";
    if (cntl) {
      cntl->SetFailed("brpc request | response | controller is null");
    }
    return;
  }
  if (!ensure_backend_ready(cntl)) {
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

  auto service_request = generate_request(req_pb, *cntl, "/v1/completions");

  if (!req_pb->prompt().empty()) {
    service_request->prompt = req_pb->prompt();
    // select instance for request
    if (!scheduler_->schedule(service_request)) {
      cntl->SetFailed("Schedule request failed!");
      LOG(ERROR) << "Schedule request failed!";
      return;
    }
  } else {
    cntl->SetFailed("Prompt is empty!");
    LOG(ERROR) << "Prompt is empty!";
    return;
  }

  // update request protobuf
  req_pb->set_service_request_id(service_request->service_request_id);
  req_pb->set_source_xservice_addr(options_.service_name());
  req_pb->mutable_token_ids()->Add(service_request->token_ids.begin(),
                                   service_request->token_ids.end());
  req_pb->mutable_routing()->set_prefill_name(
      service_request->routing.prefill_name);
  req_pb->mutable_routing()->set_decode_name(
      service_request->routing.decode_name);

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
    LOG(ERROR) << "brpc request | response | controller is null";
    if (cntl) {
      cntl->SetFailed("brpc request | response | controller is null");
    }
    return;
  }
  if (!ensure_backend_ready(cntl)) {
    return;
  }

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<::xllm::proto::ChatRequest>(arena);
  auto resp_pb =
      google::protobuf::Arena::CreateMessage<::xllm::proto::ChatResponse>(
          arena);

  const auto attachment_size = cntl->request_attachment().size();
  auto content_len = GetJsonContentLength(cntl, attachment_size);
  std::string attachment;
  cntl->request_attachment().copy_to(&attachment, content_len, 0);

  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;
  auto status =
      google::protobuf::util::JsonStringToMessage(attachment, req_pb, options);
  if (!status.ok()) {
    cntl->SetFailed(status.ToString());
    LOG(ERROR) << "parse json to proto failed: " << status.ToString();
    return;
  }

  auto service_request = generate_request(req_pb, *cntl, "/v1/chat/completions");

  if (req_pb->messages_size() > 0) {
    service_request->messages.reserve(req_pb->messages_size());
    for (const auto& message : req_pb->messages()) {
      service_request->messages.emplace_back(message.role(), message.content());
    }
    if (req_pb->has_chat_template_kwargs()) {
      service_request->chat_template_kwargs =
          proto_struct_to_json(req_pb->chat_template_kwargs());
    }
    service_request->tools = parse_tools_from_proto(req_pb->tools());
    if (req_pb->has_tool_choice()) {
      service_request->tool_choice = req_pb->tool_choice();
    }

    if (!scheduler_->schedule(service_request)) {
      cntl->SetFailed("Schedule request failed!");
      LOG(ERROR) << "Schedule request failed!";
      return;
    }
  } else {
    cntl->SetFailed("Messages is empty!");
    LOG(ERROR) << "Messages is empty!";
    return;
  }

  // update request protobuf
  req_pb->set_service_request_id(service_request->service_request_id);
  req_pb->set_source_xservice_addr(options_.service_name());
  req_pb->mutable_token_ids()->Add(service_request->token_ids.begin(),
                                   service_request->token_ids.end());
  req_pb->mutable_routing()->set_prefill_name(
      service_request->routing.prefill_name);
  req_pb->mutable_routing()->set_decode_name(
      service_request->routing.decode_name);

  auto call_data = std::make_shared<ChatCallData>(
      cntl, service_request->stream, done_guard.release(), req_pb, resp_pb);
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
    LOG(ERROR) << "brpc request | response | controller is null";
    if (cntl) {
      cntl->SetFailed("brpc request | response | controller is null");
    }
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
  assert(initialized_);
  ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    if (controller) {
      reinterpret_cast<brpc::Controller*>(controller)->SetFailed(
          "brpc request | response | controller is null");
    }
    return;
  }

  brpc::Controller* cntl = reinterpret_cast<brpc::Controller*>(controller);
  nlohmann::json summary = nlohmann::json::object();
  if (scheduler_ != nullptr) {
    summary = scheduler_->debug_summary();
  }

  const RuntimeHealthSnapshot health = runtime_state_.health_snapshot();
  PrometheusMetricsSnapshot snapshot = build_prometheus_metrics_snapshot(
      summary,
      options_.service_name(),
      options_.block_size(),
      health.ready,
      runtime_phase_name(health.phase));
  cntl->http_response().set_content_type("text/plain; version=0.0.4");
  cntl->response_attachment().append(render_prometheus_metrics(snapshot));
}

void XllmHttpServiceImpl::DebugSummary(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  assert(initialized_);
  ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    if (controller) {
      reinterpret_cast<brpc::Controller*>(controller)->SetFailed(
          "brpc request | response | controller is null");
    }
    return;
  }

  brpc::Controller* cntl = reinterpret_cast<brpc::Controller*>(controller);
  nlohmann::json summary = nlohmann::json::object();
  if (scheduler_ != nullptr) {
    summary = scheduler_->debug_summary();
  } else {
    summary["service_name"] = options_.service_name();
  }
  const RuntimeHealthSnapshot health = runtime_state_.health_snapshot();
  summary["live"] = health.live;
  summary["ready"] = health.ready;
  summary["readiness_reason"] = health.reason;
  summary["runtime_phase"] = runtime_phase_name(health.phase);
  cntl->http_response().set_content_type("application/json");
  cntl->response_attachment().append(summary.dump());
}

}  // namespace xllm_service
