/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <brpc/controller.h>
#include <butil/iobuf.h>
#include <glog/logging.h>
#include <json2pb/pb_to_json.h>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>

#include "anthropic.pb.h"
#include "chat.pb.h"
#include "completion.pb.h"

namespace xllm_service {

enum class StreamProtocol {
  kOpenAI,
  kAnthropic,
};

class StreamOutputSink {
 public:
  virtual ~StreamOutputSink() = default;
  virtual int write(const butil::IOBuf& attachment) = 0;
  virtual void notify_on_stopped(google::protobuf::Closure* callback) = 0;
};

class ProgressiveAttachmentSink final : public StreamOutputSink {
 public:
  explicit ProgressiveAttachmentSink(
      butil::intrusive_ptr<brpc::ProgressiveAttachment> attachment)
      : attachment_(std::move(attachment)) {}

  int write(const butil::IOBuf& attachment) override {
    return attachment_->Write(attachment);
  }

  void notify_on_stopped(google::protobuf::Closure* callback) override {
    attachment_->NotifyOnStopped(callback);
  }

 private:
  butil::intrusive_ptr<brpc::ProgressiveAttachment> attachment_;
};

// Interface for the classes that are used to handle grpc requests.
class CallData {
 public:
  virtual ~CallData() = default;

  // returns true if the rpc is ok and the call data is not finished
  // returns false if the call data is finished and can be deleted
  virtual bool proceed(bool rpc_ok) = 0;

  virtual bool is_disconnected() const = 0;

  // The callback is owned and invoked exactly once by brpc, either when the
  // client connection stops or when the completed call is destroyed.
  virtual void notify_on_disconnect(google::protobuf::Closure* callback) = 0;

  void get_x_request_id(std::string& x_request_id, brpc::Controller* ctrl) {
    x_request_id = "";
    if (ctrl->http_request().GetHeader("x-request-id")) {
      x_request_id = *ctrl->http_request().GetHeader("x-request-id");
    } else if (ctrl->http_request().GetHeader("x-ms-client-request-id")) {
      x_request_id = *ctrl->http_request().GetHeader("x-ms-client-request-id");
    }
    return;
  }

  void get_x_request_time(std::string& x_request_time, brpc::Controller* ctrl) {
    x_request_time = "";
    if (ctrl->http_request().GetHeader("x-request-time")) {
      x_request_time = *ctrl->http_request().GetHeader("x-request-time");
    } else if (ctrl->http_request().GetHeader("x-request-timems")) {
      x_request_time = *ctrl->http_request().GetHeader("x-request-timems");
    }
    return;
  }

 public:
  std::string x_request_id;
  std::string x_request_time;
};

template <typename Request, typename Response>
class StreamCallData : public CallData {
 public:
  StreamCallData(
      brpc::Controller* controller,
      bool stream,
      ::google::protobuf::Closure* done,
      Request* request,
      Response* response,
      std::function<void(const std::string&)> trace_callback = nullptr,
      std::shared_ptr<StreamOutputSink> stream_sink = nullptr,
      StreamProtocol stream_protocol = StreamProtocol::kOpenAI,
      bool create_progressive_attachment = true)
      : controller_(controller),
        done_(done),
        request_(request),
        response_(response),
        trace_callback_(std::move(trace_callback)),
        stream_sink_(std::move(stream_sink)),
        stream_protocol_(stream_protocol) {
    stream_ = stream;
    get_x_request_id(x_request_id, controller_);
    get_x_request_time(x_request_time, controller_);

    if (stream_) {
      if (stream_sink_ == nullptr && create_progressive_attachment) {
        butil::intrusive_ptr<brpc::ProgressiveAttachment> attachment =
            controller_->CreateProgressiveAttachment();
        if (attachment != nullptr) {
          stream_sink_ = std::make_shared<ProgressiveAttachmentSink>(
              std::move(attachment));
        }
      }

      controller_->http_response().set_content_type("text/event-stream");
      controller_->http_response().set_status_code(200);
      controller_->http_response().SetHeader("Connection", "keep-alive");
      controller_->http_response().SetHeader("Cache-Control", "no-cache");
      if (stream_sink_ == nullptr) {
        controller_->SetFailed("Progressive stream attachment is unavailable");
        connection_status_.store(-1, std::memory_order_release);
      }
      run_done_once();
    } else {
      controller_->http_response().SetHeader("Content-Type",
                                             "application/json; charset=utf-8");
    }

    json_options_.bytes_to_base64 = false;
    json_options_.jsonify_empty_array = true;
  }

  ~StreamCallData() {
    // For non stream response, call brpc done Run
    if (!stream_) {
      run_done_once();
    }
  }

  bool proceed(bool rpc_ok) override { return true; }

  void trace(const std::string& message) {
    if (trace_callback_) {
      trace_callback_(message);
    }
  }

  // For non stream response
  bool write_and_finish(const std::string& attachment /*json string*/) {
    if (finished_.exchange(true, std::memory_order_acq_rel)) {
      return false;
    }
    std::lock_guard<std::mutex> lock(output_mutex_);
    if (trace_callback_) trace_callback_(attachment);
    controller_->response_attachment() = attachment;
    return true;
  }

  bool write_and_finish(Response& response) {
    if (finished_.exchange(true, std::memory_order_acq_rel)) {
      return false;
    }
    std::lock_guard<std::mutex> lock(output_mutex_);
    butil::IOBufAsZeroCopyOutputStream json_output(
        &controller_->response_attachment());
    std::string err_msg;
    if (!json2pb::ProtoMessageToJson(
            response, &json_output, json_options_, &err_msg)) {
      controller_->response_attachment().clear();
      controller_->SetFailed(err_msg);
      return true;
    }

    if (trace_callback_) {
      std::string str;
      controller_->response_attachment().copy_to(&str);
      trace_callback_(str);
    }

    return true;
  }

  bool finish_with_error(const std::string& error_message) {
    if (finished_.exchange(true, std::memory_order_acq_rel)) {
      return true;
    }
    std::lock_guard<std::mutex> lock(output_mutex_);
    return write_error_locked(error_message);
  }

  // For stream response
  bool write(const butil::IOBuf& attachment_iobuf) {
    std::lock_guard<std::mutex> lock(output_mutex_);
    if (finished_.load(std::memory_order_acquire)) {
      return false;
    }
    return write_stream_payload(attachment_iobuf);
  }

  // For stream response
  bool write(const std::string& attachment) {
    std::lock_guard<std::mutex> lock(output_mutex_);
    if (finished_.load(std::memory_order_acquire)) {
      return false;
    }
    return write_stream_payload(attachment);
  }

  bool write(Response& response) {
    std::lock_guard<std::mutex> lock(output_mutex_);
    if (finished_.load(std::memory_order_acquire)) {
      return false;
    }
    io_buf_.clear();
    io_buf_.append("data: ");
    butil::IOBufAsZeroCopyOutputStream json_output(&io_buf_);
    std::string err_msg;
    if (!json2pb::ProtoMessageToJson(
            response, &json_output, json_options_, &err_msg)) {
      LOG(ERROR) << "Failed to convert proto to json: " << err_msg;
      return false;
    }
    io_buf_.append("\n\n");

    if (trace_callback_) {
      std::string str;
      io_buf_.copy_to(&str);
      trace_callback_(str);
    }

    return write_stream_payload(io_buf_, /*trace=*/false);
  }

  bool finish() {
    if (finished_.exchange(true, std::memory_order_acq_rel)) {
      return true;
    }
    std::lock_guard<std::mutex> lock(output_mutex_);
    if (stream_protocol_ == StreamProtocol::kAnthropic) {
      return stream_sink_ != nullptr &&
             connection_status_.load(std::memory_order_acquire) == 0;
    }
    return write_stream_payload("data: [DONE]\n\n");
  }

  bool is_disconnected() const override {
    if (stream_) {
      return stream_sink_ == nullptr ||
             connection_status_.load(std::memory_order_acquire) != 0;
    } else {
      if (controller_) {
        return controller_->IsCanceled();
      }
      return true;
    }
  }

  void notify_on_disconnect(google::protobuf::Closure* callback) override {
    if (stream_) {
      if (stream_sink_ != nullptr) {
        stream_sink_->notify_on_stopped(callback);
      } else {
        callback->Run();
      }
    } else {
      controller_->NotifyOnCancel(callback);
    }
  }

  Request& request() { return *request_; }
  Response& response() { return *response_; }
  ::google::protobuf::Closure* done() { return done_; }
  bool finished() const { return finished_.load(std::memory_order_acquire); }

 private:
  bool write_error_locked(const std::string& error_message) {
    if (!stream_) {
      controller_->SetFailed(error_message);
      return true;
    }

    nlohmann::json payload;
    std::string error_event;
    if (stream_protocol_ == StreamProtocol::kAnthropic) {
      payload = {
          {"type", "error"},
          {"error", {{"type", "api_error"}, {"message", error_message}}}};
      error_event = "event: error\ndata: " + payload.dump() + "\n\n";
    } else {
      payload = {
          {"error", {{"message", error_message}, {"type", "server_error"}}}};
      error_event = "data: " + payload.dump() + "\n\n";
    }
    const bool error_written = write_stream_payload(error_event);
    if (stream_protocol_ == StreamProtocol::kAnthropic) {
      return error_written;
    }
    return write_stream_payload("data: [DONE]\n\n") && error_written;
  }
  void run_done_once() {
    if (!done_called_.exchange(true, std::memory_order_acq_rel)) {
      done_->Run();
    }
  }

  bool write_stream_payload(const std::string& attachment) {
    io_buf_.clear();
    io_buf_.append(attachment);
    return write_stream_payload(io_buf_);
  }

  bool write_stream_payload(const butil::IOBuf& attachment, bool trace = true) {
    if (trace && trace_callback_) {
      std::string text;
      attachment.copy_to(&text);
      trace_callback_(text);
    }
    if (stream_sink_ == nullptr ||
        connection_status_.load(std::memory_order_acquire) != 0) {
      return false;
    }
    const int status = stream_sink_->write(attachment);
    if (status != 0) {
      connection_status_.store(status, std::memory_order_release);
      return false;
    }
    return true;
  }

  brpc::Controller* controller_;
  ::google::protobuf::Closure* done_;

  Request* request_ = nullptr;
  Response* response_ = nullptr;

  bool stream_ = false;
  butil::IOBuf io_buf_;

  std::atomic<bool> finished_{false};
  std::atomic<bool> done_called_{false};
  // Backend response callbacks, deadline watchdogs, and disconnect callbacks
  // can run on different bthreads/worker threads. Keep each response frame and
  // its terminal state transition in one serialized critical section.
  mutable std::mutex output_mutex_;
  json2pb::Pb2JsonOptions json_options_;
  std::function<void(const std::string&)> trace_callback_;
  std::shared_ptr<StreamOutputSink> stream_sink_;
  StreamProtocol stream_protocol_ = StreamProtocol::kOpenAI;

  std::atomic<int> connection_status_{0};
};

using CompletionCallData = StreamCallData<::xllm::proto::CompletionRequest,
                                          ::xllm::proto::CompletionResponse>;

using ChatCallData =
    StreamCallData<::xllm::proto::ChatRequest, ::xllm::proto::ChatResponse>;

class AnthropicCallData
    : public StreamCallData<::xllm::proto::ChatRequest,
                            ::xllm::proto::AnthropicMessagesResponse> {
 public:
  using Base = StreamCallData<::xllm::proto::ChatRequest,
                              ::xllm::proto::AnthropicMessagesResponse>;

  AnthropicCallData(
      brpc::Controller* controller,
      bool stream,
      ::google::protobuf::Closure* done,
      ::xllm::proto::ChatRequest* request,
      ::xllm::proto::AnthropicMessagesResponse* response,
      std::function<void(const std::string&)> trace_callback = nullptr,
      std::shared_ptr<StreamOutputSink> stream_sink = nullptr,
      bool create_progressive_attachment = true)
      : Base(controller,
             stream,
             done,
             request,
             response,
             std::move(trace_callback),
             std::move(stream_sink),
             StreamProtocol::kAnthropic,
             create_progressive_attachment) {}
};

}  // namespace xllm_service
