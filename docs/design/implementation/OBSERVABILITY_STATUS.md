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
- 关联设计/Requirement ID：G0、02 §8.3、D37、D43、D58、D59
- 最近验证基线：xLLM `service_dev` observability schema、xllm-service 本状态文档所在提交
- 验证环境和日期：xllm-dev-sandbox，Ubuntu 24.04 ARM64，2026-08-08

## 支持范围

| Provider | Mode | Model/Profile | 支持状态 | 限制与证据 |
| --- | --- | --- | --- | --- |
| 公共协议 | 全部 V2 mode | event schema v1 | CPU_VERIFIED | 单一 proto、golden wire、field number、optional presence 通过 |
| xLLM Service | Provider-neutral | recorder core | CPU_VERIFIED | 固定容量 ring、非阻塞 producer、显式 drop/invalid counter 通过 |
| xLLM Service | Provider-neutral | TTFT/TPOT/E2E/ITL | CPU_VERIFIED | 仅按本地 monotonic clock；无效样本带原因，不产生负 duration |
| Gateway/Service/P/D | 全链路 | 生产请求 | IN_PROGRESS | ID 生成/透传和真实阶段埋点尚未全部接入 |

## 实现

- 代码入口与核心接口：xLLM `xllm/proto/observability.proto`；Service
  `xllm_service/observability/request_event_recorder.*`。
- 状态、资源和错误码权威位置：事件/指标 enum 只定义在共享 proto；recorder
  stats 是本地无标签 counter，不创建高基数指标标签。
- 跨仓协议与依赖：xllm-service 通过 `proto_xllm` 编译同一个
  `observability.proto`；`attempt_seq=0` 与缺失、`event_seq=0` 与缺失均可区分。
- 执行面隔离：producer 在 schema 校验后使用 `try_lock` 写入构造时预分配的
  ring。锁竞争和容量满分别计数，均立即返回；exporter/drain 不在请求执行线程运行。
- 明确不支持范围：本批次未声明全链路关联覆盖率达标，未接入 Gateway header、
  Service 请求 UID、P/D 阶段事件和 exporter，也不提供生产性能或 NPU 结论。

## 需求与测试追踪

| Requirement ID | CPU test | Torch CPU test | NPU test | 结果 |
| --- | --- | --- | --- | --- |
| G0 单一事件 schema | xLLM `RequestEventProtocolTest` golden wire、presence、profile roundtrip、字段号 | N/A，无 tensor 逻辑 | N/A | PASS，4/4 |
| G0 有界非阻塞 recorder | capacity/drop、zero capacity、ring wrap FIFO、并发 producer、identity 上限 | N/A | N/A | PASS |
| G0 Admission 终态 | ACCEPTED、非法参数后重试、重复 terminal、RAII missing terminal、clock regression | N/A | 待真实 D | PASS |
| G0 指标正确性 | TTFT/TPOT/E2E/ITL、TPOT=0、单 token、缺首 token、时钟回退 | N/A | 待真实请求 | PASS |
| G0 双仓回归 | xLLM 60/60；Service 126/126 | 当前目标链接 Torch CPU；新增逻辑不含 tensor 运算 | N/A | PASS |

## 完善情况

- 已完成：公共事件和指标 schema；低基数 enum；无原始 prompt/output/token IDs；
  固定容量非阻塞 recorder；长度门禁；Admission RAII 终态；本地单调计时；无效
  TTFT/TPOT/E2E/ITL 的显式原因；并发和边界 CPU 测试。
- 已知缺口/风险：事件尚未贯穿真实 Gateway/Service/P/D 请求；drop counter 尚未
  接到报警 exporter；RAII 保证本地生成一次终态尝试，但 ring 压力仍可丢失终态，
  必须由 drop rate 和关联覆盖率发布门禁识别。
- 回滚与兼容：全新 additive proto，不改变旧请求 wire；未知 schema/enum 在 recorder
  边界 fail closed。禁用 recorder 不影响执行正确性。
- 性能、容量和观测证据：CPU 并发测试证明队列有界和 producer 不等待锁；尚无真实
  workload 的 p99 CPU 开销、drop rate 或关联覆盖率数据。
- 达到 CPU_VERIFIED 仍需完成：真实请求 ID 生成/透传，Gateway/Service/P/D 最小事件
  全覆盖，exporter/counter 报警，关联完整性和公式门禁的 CPU loopback 集成测试。
- 达到 VERIFIED 仍需完成：上述 CPU 门禁通过后，在 NPU 环境验证真实 Prefill、
  KV transfer、Decode、首 token ACK/flush 和资源释放事件及线上开销。
