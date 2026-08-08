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
- 最近验证基线：xLLM `08d6de0c`、xllm-service `service_dev`
- 验证环境和日期：xllm-dev-sandbox，Ubuntu 24.04 ARM64，2026-08-08

## 支持范围

| Provider | Mode | Model/Profile | 支持状态 | 限制与证据 |
| --- | --- | --- | --- | --- |
| xLLM Native | CPU 公共路径 | 协议、流式语义、JSON 解析 | CPU_VERIFIED | 52/52 tests passed |
| xLLM Service + xLLM Native | 模板与 Service 公共路径 | 当前外层 xLLM `08d6de0c` | CPU_VERIFIED | 96/96 tests passed |
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
| DEV-CPU-XLLM | `xllm-dev xllm-test <xllm> native Debug` | 当前目标链接 Torch CPU；本批次无新增 tensor 逻辑 | N/A | PASS，52/52 |
| DEV-CPU-SERVICE | `XLLM_SOURCE_DIR=<xllm> xllm-dev service-test <xllm-service> native Debug` | 本批次无新增 tensor 逻辑 | N/A | PASS，96/96 |
| DEV-CROSS-REPO | xllm-service 对 xLLM `08d6de0c` override 构建与测试 | N/A | N/A | PASS |

## 完善情况

- 已完成：修复四个未显式固定 DeepSeek V4 thinking/reasoning 模式的陈旧测试；
  同时验证 chat、thinking 和新版默认 high reasoning 契约。Anthropic 模板测试
  显式使用 `tojson`，避免把 Jinja 对象展示格式误当作 JSON wire 契约。
- 已知缺口/风险：xLLM CPU 沙箱只覆盖当前公共测试目标；随着 V2 Engine
  状态机、allocator 和 tensor 元数据逻辑进入开发，必须扩展目标而不能沿用
  52 项数量作为完整 V2 证明。
- 回滚与兼容：本批次只修改测试和开发文档，不改变生产请求语义。
- 性能、容量和观测证据：本批次不建立生产性能基线；由 V2-B2/G0 交付。
- 达到 VERIFIED 仍需完成：按 B1-B10 完成全部 V2 功能和 NPU 专项门禁。
