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
- 最近验证基线：xLLM 与 xllm-service `service_dev` 本状态文档所在提交
- 验证环境和日期：xllm-dev-sandbox，Ubuntu 24.04 ARM64，2026-08-10

## 支持范围

| Provider | Mode | Model/Profile | 支持状态 | 限制与证据 |
| --- | --- | --- | --- | --- |
| xLLM Native | CPU 公共路径 | 协议、Provider/Event wire、资源状态机、注册生命周期、流式语义、deadline、exact 首事件、JSON 解析、异步输出投递、production BlockManager adapter 与 simulated HBM | CPU_VERIFIED | `6c9d661e` 本批 adapter/simulator 15/15、admission wire 3/3；前一完整公共基线 118/118；只证明 CPU 链路和容量/所有权，不代表真实 HBM |
| xLLM Service + xLLM Native | 模板、Provider/Event Contract、请求身份、execution hold、output reorder、deadline、断连、seq=0 Query 恢复、首输出前 attempt 替换、流控、K0-K2、模式和可观测 | pinned 与当前外层 xLLM `service_dev` | CPU_VERIFIED | 当前两种构建均为 388/388 tests passed；三个生产二进制 build/link verify |
| vLLM-Ascend | strict Agent/legacy sidecar 公共逻辑 | 无设备路径 | CPU_VERIFIED / NPU_PENDING | strict Descriptor、attempt/deadline/fencing/per-DP State 与 HTTP loopback 通过；真实 NPU/同命待验证 |

## 实现

- 代码入口与核心接口：`tests/xllm_service` 及 xLLM CPU 公共测试目标。
- 状态、资源和错误码权威位置：本批次不增加生产状态或资源协议。
- 跨仓协议与依赖：xllm-service pin 到已推送的 xLLM `6c9d661e`，并同时使用当前
  外层 xLLM 工作树 override 构建，避免两套协议真相漂移。
- 明确不支持范围：本状态只证明可重复的双仓 CPU 开发基线，不代表任一 V2
  生产功能完成，也不证明 NPU kernel、CANN、RDMA 或真实推理路径。

## 需求与测试追踪

| Requirement ID | CPU test | Torch CPU test | NPU test | 结果 |
| --- | --- | --- | --- | --- |
| DEV-STYLE | `git diff --check`；xLLM clang-format 规则人工核对 | N/A | N/A | PASS |
| DEV-CPU-XLLM | 重建 production adapter/simulator 与 request-output adapter；前一公共全量基线 | BlockManager 真实 leaf 共享 free-list；queue 历史覆盖非连续 view retained storage、源 tensor 释放后的所有权及空 tensor | simulated HBM 12 项 + production adapter 3 项；真实 NPU HBM 待验证 | PASS，15/15 与 3/3；15 项重复 100 轮；完整 runtime 新结果受第三方 Mooncake 阻断，旧 118/118 不冒充新提交结果 |
| DEV-PRODUCTION-XLLM | 受影响生产 TU 经 Service 外部/固定依赖编译；完整无设备 runtime build | 无硬件 process-group factory fail closed | N/A | PARTIAL，受影响跨仓生产对象通过；完整 runtime 停于第三方 Mooncake Clang 错误 |
| DEV-CPU-SERVICE | pinned 与 `XLLM_SOURCE_DIR=<xllm>` 两种 `xllm-dev service-test <xllm-service> native Debug` | execution hold/output reorder/deadline/Query recovery/断连/attempt retry 不含 tensor 逻辑 | N/A | PASS，当前均为 388/388；B10 流控/路由/KV/recorder 57 项历史压力各 100 轮，共 5700 次；同 57 项通过 GCC 13 ASan+UBSan（leak detection） |
| DEV-CPU-VLLM-AGENT | 沙箱内 `python -m pytest -q vllm_sidecar/tests` | N/A，Python 控制面 | 待真实 vLLM-Ascend/NPU | PASS，60/60 |
| DEV-PRODUCTION-SERVICE | `service-build` + `service-verify` | 三个 ARM64 Debug ELF；无缺失动态库 | N/A | PASS |
| DEV-CROSS-REPO | xllm-service 对外层 xLLM `service_dev` override 构建与测试 | N/A | N/A | PASS |

## 完善情况

- 已完成：修复四个未显式固定 DeepSeek V4 thinking/reasoning 模式的陈旧测试；
  同时验证 chat、thinking 和新版默认 high reasoning 契约。Anthropic 模板测试
  显式使用 `tojson`，避免把 Jinja 对象展示格式误当作 JSON wire 契约。
- 已知缺口/风险：已补齐无设备 `PlatformStream`、CPU Platform 默认值、VMM host 类型
  和受影响虚函数，无硬件 `ProcessGroupImpl` 工厂也明确 fail closed；完整 runtime
  继续构建后停在第三方 Mooncake thread-safety/incomplete-type 错误。
  当前 `xllm-verify` 不等价于可启动无设备推理服务，不能把 CPU 测试数量当作完整 V2
  证明，更不能替代 KV cache 的 simulated HBM 或真实 NPU HBM 验证。
- 回滚与兼容：请求协议仅做 additive 扩展；旧客户端未携带 correlation 时继续走
  legacy 路径，`service_request_id/service_req_id` 保持为 `request_uid` 的 wire 兼容镜像。
- 性能、容量和观测证据：当前只建立 CPU 正确性与有界 recorder 基线；真实请求
  p99 开销、事件丢失率和跨组件关联覆盖率仍需在 G0 全阶段接入后测量。
- V2 B0-B10 仓库内代码已完成；达到 `VERIFIED` 仍需完成 NPU、真实 HBM/Link/etcd、
  多实例故障矩阵、容量拐点、观测开销和长时 soak 专项门禁。
