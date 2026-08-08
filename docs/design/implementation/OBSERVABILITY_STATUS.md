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

# G0 请求事件协议与观测 CPU 核心

## 基本信息

- Owner：xLLM Service V2
- 状态：PARTIAL
- 关联设计/Requirement ID：G0、02 §3.1/§8.3、D37、D43、D58-D60
- 最近验证基线：xLLM `service_dev` observability schema、xllm-service 本状态文档所在提交
- 验证环境和日期：xllm-dev-sandbox，Ubuntu 24.04 ARM64，2026-08-08

## 支持范围

| Provider | Mode | Model/Profile | 支持状态 | 限制与证据 |
| --- | --- | --- | --- | --- |
| 公共协议 | 全部 V2 mode | event schema v1 | CPU_VERIFIED | 单一 proto、golden wire、field number、optional presence 通过 |
| xLLM Service | Provider-neutral | recorder core | CPU_VERIFIED | 固定容量 ring、非阻塞 producer、显式 drop/invalid counter 通过 |
| xLLM Service | Provider-neutral | TTFT/TPOT/E2E/ITL | CPU_VERIFIED | 仅按本地 monotonic clock；无效样本带原因，不产生负 duration |
| xLLM Service | Completion/Chat/Anthropic/vLLM | 生产请求入口 | CPU_VERIFIED | 上游 ID/traceparent 校验、UUIDv7 request_uid、来源标记和 wire/header 透传；三个生产二进制通过构建和动态链接检查 |
| xLLM Native | Service→P→D | correlation wire/运行对象 | PARTIAL | Completion/Chat 与 DisaggRequest schema、语义校验、RequestParams/Request 复制已实现；协议测试和 request 对象 compile-only 通过，完整无设备运行时目标仍被既有 PlatformStream/VMM CPU 缺口阻塞 |
| Gateway/Service/P/D | 全链路事件 | 生产请求 | IN_PROGRESS | Gateway 代码不在当前双仓；P/D 真实阶段埋点和 exporter 尚未接入 |

## 实现

- 代码入口与核心接口：xLLM `xllm/proto/observability.proto`、
  `xllm/proto/request_correlation.h` 和 P/D `DisaggRequest`；Service
  `xllm_service/observability/request_identity.*`、
  `request_event_recorder.*` 及 HTTP request lifecycle。
- 状态、资源和错误码权威位置：事件/指标 enum 只定义在共享 proto；recorder
  stats 是本地无标签 counter，不创建高基数指标标签。
- 跨仓协议与依赖：xllm-service 通过 `proto_xllm` 编译同一个
  `observability.proto`；`attempt_seq=0` 与缺失、`event_seq=0` 与缺失均可区分。
  `service_request_id/service_req_id` 只镜像 `request_uid`，接收边界校验相等；
  legacy 请求不会被 P 包装成存在但为空的 correlation。
- 执行面隔离：producer 在 schema 校验后使用 `try_lock` 写入构造时预分配的
  ring。锁竞争和容量满分别计数，均立即返回；exporter/drain 不在请求执行线程运行。
- 明确不支持范围：本批次未声明全链路事件关联覆盖率达标；Gateway 生成逻辑、
  P/D 阶段事件和 exporter 尚未完成，也不提供生产性能或 NPU 结论。

## 需求与测试追踪

| Requirement ID | CPU test | Torch CPU test | NPU test | 结果 |
| --- | --- | --- | --- | --- |
| G0 单一事件 schema | xLLM `RequestEventProtocolTest` golden wire、presence、profile/P-D roundtrip、字段号、UUIDv7/source 校验 | N/A，无 tensor 逻辑 | N/A | PASS，7/7 |
| G0 Service 入口身份 | UUIDv7 layout/边界/唯一性；header 输入、traceparent、非法值、来源和 generator fallback | N/A | 待真实 Gateway | PASS，10/10 |
| G0 Service→P→D 透传 | Completion golden wire、DisaggRequest roundtrip、UID mismatch/空对象 fail closed；RequestParams/Request CPU compile-only | N/A | 待真实 P/D | PASS（完整 xLLM CPU target 受既有无设备 platform 类型缺口限制） |
| G0 有界非阻塞 recorder | capacity/drop、zero capacity、ring wrap FIFO、并发 producer、identity 上限 | N/A | N/A | PASS |
| G0 Admission 终态 | ACCEPTED、非法参数后重试、重复 terminal、RAII missing terminal、clock regression | N/A | 待真实 D | PASS |
| G0 指标正确性 | TTFT/TPOT/E2E/ITL、TPOT=0、单 token、缺首 token、时钟回退 | N/A | 待真实请求 | PASS |
| G0 双仓回归 | xLLM 63/63；Service 136/136；Service 三个生产二进制 build/link verify | 当前目标链接 Torch CPU；新增逻辑不含 tensor 运算 | N/A | PASS |

## 完善情况

- 已完成：公共事件和指标 schema；低基数 enum；无原始 prompt/output/token IDs；
  固定容量非阻塞 recorder；长度门禁；Admission RAII 终态；本地单调计时；无效
  TTFT/TPOT/E2E/ITL 的显式原因；Service 请求入口 UUIDv7 与 header/traceparent
  处理；Service→P→D correlation wire 和运行对象；vLLM header 透传；并发和边界
  CPU 测试。
- 已知缺口/风险：correlation 已贯穿当前 Service/P/D 请求对象，但事件尚未覆盖
  真实 Gateway/Service/P/D 全阶段；drop counter 尚未
  接到报警 exporter；RAII 保证本地生成一次终态尝试，但 ring 压力仍可丢失终态，
  必须由 drop rate 和关联覆盖率发布门禁识别。xLLM 的完整无设备 CPU runtime
  仍缺 `PlatformStream`、`VirPtr/PhyMemHandle` 等 fallback；本批只对可达协议目标和
  修改的 request 对象执行 CPU 验证，不把该基础设施限制伪装成已覆盖。
- 回滚与兼容：全新 additive proto，不改变旧请求 wire；未知 schema/enum 在 recorder
  边界 fail closed。禁用 recorder 不影响执行正确性。
- 性能、容量和观测证据：CPU 并发测试证明队列有界和 producer 不等待锁；尚无真实
  workload 的 p99 CPU 开销、drop rate 或关联覆盖率数据。
- 达到 CPU_VERIFIED 仍需完成：Gateway/Service/P/D 最小事件全覆盖，
  exporter/counter 报警，关联完整性和公式门禁的 CPU loopback 集成测试，并补齐
  xLLM 无设备 runtime compile target 或提供不依赖设备平台的专用运行时测试切片。
- 达到 VERIFIED 仍需完成：上述 CPU 门禁通过后，在 NPU 环境验证真实 Prefill、
  KV transfer、Decode、首 token ACK/flush 和资源释放事件及线上开销。
