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

# G1 执行资源安全协议 CPU 核心

## 基本信息

- Owner：xLLM Service V2
- 状态：PARTIAL
- 关联设计/Requirement ID：G1、02 §3.2/§5.1/§5.3/§6、09 §2.1、F83
- 最近验证基线：xLLM 与 xllm-service `service_dev` 本状态文档所在提交
- 验证环境和日期：xllm-dev-sandbox，Ubuntu 24.04 ARM64，2026-08-08

## 支持范围

| Provider | Mode | Model/Profile | 支持状态 | 限制与证据 |
| --- | --- | --- | --- | --- |
| xLLM Native | `REMOTE_PD` | Engine attempt/resource protocol CPU core | CPU_VERIFIED | 17 个状态机测试；并发幂等准入、cancel fence、TTL、GenerationCommit、tombstone 和精确释放通过 |
| xLLM Native | `LOCAL_PREFILL_DECODE` / `PREFILL_ONLY` | 公共 attempt schema | PARTIAL | 状态和 reason 可表达；尚未接入对应 allocator/scheduler |
| xLLM Service | Provider-neutral 全模式 | `ExecutionResourceHold` / cleanup capacity | CPU_VERIFIED | 19 个测试；dispatch 前 token、单 hold、候选收敛、旧 attempt/incarnation fencing 和 record/byte 上限通过 |
| vLLM-Ascend | `AGGREGATED` | 公共 hold schema | PARTIAL | cleanup core 可表达；Agent Submit/Query/Cancel 尚未实现 |

这里的 `CPU_VERIFIED` 只描述可独立运行的协议核心，不表示对应生产 mode 已开放。
整个 G1 在生产调用链接入和真实 allocator 完成前保持 `PARTIAL`。

## 实现

- 跨仓 wire 单一真相：xLLM `xllm/proto/provider.proto` 定义结构化三态
  Admission、稳定 reason、attempt lifecycle、target-specific `RequestAttemptKey`
  和 target-neutral `ExecutionAttemptId/ExecutionResourceHold`；
  `disagg_pd.proto` additive 增加 reservation/deadline/incarnation、结构化响应及
  `CommitGeneration/BeginTransfer/CancelRequest/QueryRequest`。
- Engine CPU 核心：xLLM
  `xllm/core/distributed_runtime/attempt_lifecycle_table.*`。allocator、handoff 和
  release callback 与状态转换在同一锁域执行；live/tombstone 与 cancel fence
  使用独立固定容量；unknown Cancel 先安装 negative fence，unknown Query 无副作用。
- Service CPU 核心：`xllm_service/provider/execution_hold.*`。move-only cleanup
  reservation 在 dispatch 前按 record 和 bytes 双上限预留，同一 token 从
  RequestContext 转移到最小 cleanup record，收敛后自动释放。
- hold 安全规则：`likely_holder` 不缩小候选；只有明确 `PROVEN_PRECOMMIT` 或
  `GENERATION_COMMITTED` 证明可以写 confirmed holder 并收窄；所有安全候选必须
  分别取得 terminal/cancel-fence/self-fence/process-terminated/允许的硬时间证明。
  `QUERY_ABSENT` 永远不是证明，硬时间证明默认禁用。
- 明确不支持范围：当前尚未把状态机接入真实 D KV block、decode credit 和 slot
  allocator，新增 RPC 仍只有生成的默认 handler；Service hold 尚未接入生产
  `RequestContext`、Provider dispatch 和后台 cleanup worker；未实现 local/aggregated
  production path，也未提供 NPU、RDMA 或故障注入结论。

## 需求与测试追踪

| Requirement ID | CPU test | Torch CPU test | NPU test | 结果 |
| --- | --- | --- | --- | --- |
| G1 三态准入与稳定 reason | `ReasonsHaveStableThreeWayClassification`；provider golden wire/字段号 | N/A，纯控制协议 | 待真实 allocator | PASS |
| G1 幂等资源分配 | 16 线程同 key retry 只 allocation 一次；参数冲突 fail closed | 当前 xLLM 测试目标链接 Torch CPU；本核心不执行 tensor | 待真实 KV | PASS |
| G1 cancel-before-add | unknown cancel fence、迟到 admission、fence 独立容量与恢复压力 | N/A | 待 RPC race | PASS |
| G1 TTL/终态/释放 | transfer-start TTL、reservation TTL、过期 replay、cancel/destructor 精确 release | N/A | 待真实 stream/KV | PASS |
| G1 GenerationCommit | begin-transfer 门禁、handoff 幂等/冲突、首事件保留、output gate | N/A | 待 P→D transfer | PASS |
| F83 Service 单 hold | 32 线程并发安装只成功一次；resolved 后下一 attempt 可安装 | N/A | 待生产 RequestContext | PASS |
| F83 cleanup 容量 | record/byte backpressure、64 线程精确耗尽、RAII 与跨表 token 拒绝 | N/A | N/A | PASS |
| F83 收敛证明 | 全候选、confirmed 收窄、旧 attempt/incarnation、`QUERY_ABSENT`、hard-bound profile 门禁 | N/A | 待 Provider Query/Cancel | PASS |
| 内存/未定义行为 | GCC 13 ASan+UBSan 定向运行 Engine 17 项与 Service hold 19 项 | N/A | N/A | PASS；Clang sanitizer runtime 未随 ARM64 镜像安装 |
| 双仓回归 | xLLM 84/84；Service 155/155；Service 三个生产二进制 build/link verify | 公共测试目标使用 Torch CPU 环境 | N/A | PASS |

## 完善情况

- 已完成：additive wire、Engine attempt CPU 状态机、Service 单 hold/cleanup capacity
  CPU 核心、wire 稳定性门禁、并发/容量/过期/取消/收敛负向测试。
- 已知缺口/风险：核心 callback 的生产实现必须保持本地、有界、不可重入；request
  fingerprint 必须覆盖全部不可变 admission 参数；cancel fence/tombstone TTL 必须由
  已配置的最大消息寿命证明。任何一项无法证明都必须继续 fail closed。
- 回滚与兼容：旧 `FirstGeneration` 和旧字段保留；新增字段/RPC 都是 additive。
  legacy 路径不会自动获得 V2 资源安全语义，开放开关前必须完成 capability 门禁。
- 性能、容量和观测证据：CPU 测试证明内存逻辑有固定 record/byte/fence 上限；尚无
  真实 workload 的 allocator 锁持有时间、冲突率、cleanup burst 或 1 万次故障数据。
- 达到 CPU_VERIFIED 仍需完成：接入 D allocator 与全部 V2 RPC handler；Service
  RequestContext/dispatch/cleanup worker 接入；补齐 CPU fake-provider 端到端 race、
  deadline 和 fault loop；确保所有 mode 共用一套权威资源账本。
- 达到 VERIFIED 仍需完成：NPU KV/credit/slot 原子分配、P→D transfer、取消/超时/
  进程终止故障矩阵、1 万次故障门禁和容量/性能发布证据。
