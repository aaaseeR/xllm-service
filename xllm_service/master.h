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

#pragma once

#include <brpc/server.h>

#include <atomic>
#include <thread>

#include "common/options.h"
#include "dispatcher/dispatcher.h"
#include "http_service/service.h"
#include "rpc_service/service.h"
#include "runtime/runtime_state.h"
#include "scheduler/scheduler.h"
#include "transport/channel_pool.h"

namespace xllm_service {

class Master {
 public:
  explicit Master(const Options& options);
  ~Master();

  bool start();
  void stop();

 private:
  bool setup_http_server();
  void reconcile_runtime_readiness();
  bool start_rpc_server();

 private:
  Options options_;
  RuntimeState runtime_state_;

  std::shared_ptr<ChannelPool> channel_pool_;
  std::unique_ptr<Scheduler> scheduler_;
  std::unique_ptr<Dispatcher> dispatcher_;

  // 1.For http service
  std::string http_server_address_;
  std::unique_ptr<xllm_service::XllmHttpServiceImpl> http_service_;
  brpc::Server http_server_;
  std::unique_ptr<std::thread> readiness_thread_;
  std::atomic<bool> stopped_{false};
  bool http_server_started_ = false;
  brpc::ServerOptions http_options_;
  butil::EndPoint http_endpoint_;

  // 2.For rpc service
  std::string rpc_server_address_;
  std::unique_ptr<xllm_service::XllmRpcService> rpc_service_;
  brpc::Server rpc_server_;
  bool rpc_server_started_ = false;
};

}  // namespace xllm_service
