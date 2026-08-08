<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================-->

# 双仓 CPU 开发基线

## 基本信息

- Owner：xLLM Service V2
- 状态：CPU_VERIFIED
- 关联设计/Requirement ID：开发规范 §3、§4.3、§7；V2-B0 G-1 测试底座
- 最近验证基线：xLLM `service_dev` observability schema、xllm-service 本状态文档所在提交
- 验证环境和日期：xllm-dev-sandbox，Ubuntu 24.04 ARM64，2026-08-08

## 支持范围

| Provider | Mode | Model/Profile | 支持状态 | 限制与证据 |
| --- | --- | --- | --- | --- |
| xLLM Native | CPU 公共路径 | 协议、Provider/Event wire、资源状态机、流式语义、deadline、exact 首事件、JSON 解析 | CPU_VERIFIED | 98/98 tests passed |
| xLLM Service + xLLM Native | 模板、Provider/Event Contract、请求身份、execution hold、output reorder、deadline、seq=0 Query 恢复与 Service 公共路径 | 当前外层 xLLM `service_dev` | CPU_VERIFIED | 183/183 tests passed；三个生产二进制 build/link verify |
| vLLM-Ascend | Python sidecar 公共逻辑 | 无设备路径 | PARTIAL | Provider Agent 尚未进入 B1-B6 实现 |

## 实现

- 代码入口与核心接口：`tests/xllm_service` 及 xLLM CPU 公共测试目标。
- 状态、资源和错误码权威位置：本批次不增加生产状态或资源协议。
- 跨仓协议与依赖：xllm-service 使用当前外层 xLLM 工作树构建，避免 pinned
  submodule 掩盖跨仓语义差异。
- 明确不支持范围：本状态只证明可重复的双仓 CPU 开发基线，不代表任一 V2
  生产功能完成，也不证明 NPU kernel、CANN、RDMA 或真实推理路径。

## 需求与测试追踪

| Requirement ID | CPU test | Torch CPU test | NPU test | 结果 |
| --- | --- | --- | --- | --- |
| DEV-STYLE | `git diff --check`；xLLM clang-format 规则人工核对 | N/A | N/A | PASS |
| DEV-CPU-XLLM | `xllm-dev xllm-test <xllm> native RelWithDebInfo` 加 exact output adapter 目标 | 当前目标链接 Torch CPU；新增控制核心不含 tensor 逻辑 | N/A | PASS，98/98 |
| DEV-PRODUCTION-XLLM | 所有受影响生产 TU 使用 Clang C++20 `-Werror` 编译；`xllm-build`/`xllm-verify` | 无设备 `PlatformStream` 和 VMM host 类型可编译 | N/A | PASS |
| DEV-CPU-SERVICE | `XLLM_SOURCE_DIR=<xllm> xllm-dev service-test <xllm-service> native Debug` | 新增 execution hold/output reorder/deadline/Query recovery 不含 tensor 逻辑 | N/A | PASS，183/183 |
| DEV-PRODUCTION-SERVICE | `service-build` + `service-verify` | 三个 ARM64 Debug ELF；无缺失动态库 | N/A | PASS |
| DEV-CROSS-REPO | xllm-service 对外层 xLLM `service_dev` override 构建与测试 | N/A | N/A | PASS |

## 完善情况

- 已完成：修复四个未显式固定 DeepSeek V4 thinking/reasoning 模式的陈旧测试；
  同时验证 chat、thinking 和新版默认 high reasoning 契约。Anthropic 模板测试
  显式使用 `tojson`，避免把 Jinja 对象展示格式误当作 JSON wire 契约。
- 已知缺口/风险：已补齐无设备 `PlatformStream`、CPU Platform 默认值、VMM host 类型
  和受影响虚函数，受影响生产 TU 可在 CPU 沙箱严格编译；完整 `scheduler_test` 继续
  暴露既有 host-only `ProcessGroupImpl` 缺失，尚未进入默认 `xllm-test` 六个目标。
  当前 `xllm-verify` 不等价于可启动无设备推理服务，不能把 98 项数量当作完整 V2
  证明。
- 回滚与兼容：请求协议仅做 additive 扩展；旧客户端未携带 correlation 时继续走
  legacy 路径，`service_request_id/service_req_id` 保持为 `request_uid` 的 wire 兼容镜像。
- 性能、容量和观测证据：当前只建立 CPU 正确性与有界 recorder 基线；真实请求
  p99 开销、事件丢失率和跨组件关联覆盖率仍需在 G0 全阶段接入后测量。
- 达到 VERIFIED 仍需完成：按 B1-B10 完成全部 V2 功能和 NPU 专项门禁。
