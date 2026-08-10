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

# G0 请求事件、Debug 与集群观测状态

## 基本信息

- Owner：xLLM Service V2
- 状态：`CPU_VERIFIED / NPU_AND_CLUSTER_PENDING`
- 关联设计/Requirement ID：G0、02 §3.1/§8.3、D37、D43、D58-D60、V2-B10
- 环境和日期：xllm-dev-sandbox，Ubuntu 24.04 ARM64，Clang 18，2026-08-10

## 支持范围

| 层 | 支持状态 | 已完成机制 | 限制 |
| --- | --- | --- | --- |
| 公共协议 | CPU_VERIFIED | 单一 `observability.proto`、schema v1、optional presence、golden wire、P/D correlation | additive schema；旧消费者忽略新字段 |
| Service 常开指标 | CPU_VERIFIED | 低基数 bvar gauge/counter/histogram，queue/active/mode/failure/output、TTFT/TPOT/E2E/queue wait | 未做线上阈值校准 |
| Service 逐请求事件 | CPU_VERIFIED | `--v=1` 开启；固定容量非阻塞 ring、独立 exporter、稳定 JSON、event sequence、drop/invalid/duplicate 计数 | Gateway 代码不在双仓；并发事件需按 event_seq 重排 |
| Service 集群快照 | CPU_VERIFIED | readiness/saturation/registry/queue/terminal/token/KV/event-loss 的周期 INFO 日志 | 每 Service 本地快照；跨实例聚合由日志/指标平台完成 |
| xLLM Native | CPU_VERIFIED | attempt admission/commit/transfer/cancel/query/terminal 与 P/D stage VLOG | 真实 NPU stage/开销待验 |
| vLLM-Ascend | CPU_VERIFIED | correlation header、严格 Agent attempt/deadline/fencing loopback | 真实 Ascend runtime/同命部署待验 |

## 实现与不变量

- 事件唯一 schema 位于 xLLM `xllm/proto/observability.proto`。Service recorder、JSON
  formatter、Scheduler producer 和 exporter 位于 `xllm_service/observability` 与
  `xllm_service/scheduler`；xLLM attempt/stage producer 位于 Native P/D runtime。
- 单请求关联键为 `global_request_id/trace_id/request_uid/attempt_seq/event_seq`，目标资源
  再加 Engine UID/incarnation；runtime profile 包含 Provider、runtime/plugin/hardware
  runtime、profile digest、model revision 和 execution mode。
- 普通日志和事件禁止 prompt、messages、输出正文、token IDs、工具参数和 KV 内容。
  只允许 token/byte/block 数、枚举原因、稳定身份和 duration。JSON formatter 有永久
  负向断言；prompt 编码错误只记录 request UID、model 和 prompt byte 数。
- 既有 `--enable_request_trace` 是独立 legacy 内容 trace，不属于 V2 事件机制；默认
  关闭，启用时明确 WARNING，生产常规排障/性能分析禁止使用，安全治理见运行手册。
- 详细事件严格服从仓库既有 `VLOG(1)` 风格。`--v=1` 关闭时，生命周期低基数计数仍
  更新，但不构造逐请求 protobuf、不写 ring；避免为默认关闭的 debug 日志支付高额
  请求路径开销。集群摘要和 event loss 告警分别用 INFO/WARNING 常开。
- recorder 在构造时预分配固定 ring，producer schema 校验后 `try_lock`；竞争、容量满
  和非法事件立即返回并计数，不反压推理。容量上限为 1,048,576，export batch 不得
  大于 ring；启动配置非法时 fail closed。
- TTFT/ITL/TPOT 只接受 Provider 返回的累计 generated-token count。响应 chunk 不是
  token；计数缺失、为 0 或重复的 terminal/control chunk 不伪造 token/ITL。E2E 以
  `service_request_terminal` 为边界，TTFT/ITL/TPOT 以
  `service_response_write` 为边界，统一使用 steady clock。
- K1/K2 事件发布 predicted/actual P/D hit、transfer/skipped bytes 和 HOST 命中上界。
  HBM/HOST 分别由 device/host prefix leaf 真实发布；HOST 只在 bounded shortlist 上查询，
  不参与 V2 路由或 admission。SSD/STORE 字段已删除并 reserved，V2 不查询这些 tier。
- `D_ADMISSION` 终态只由 xLLM D 的真实 `AdmissionResult` 回传产生；`RESOURCE_RELEASE`
  覆盖即时释放、deferred cleanup、收敛和 shutdown 未收敛。`ROUTE` 对容量、stale、mode
  和永久不可行发布稳定拒绝原因，缺少终态 fail closed。

## 指标与容量

常开指标包括：

- gauge：`xllm_service_v2_queued_requests`、`dispatched_requests`、
  `queued_prompt_tokens`、`queued_bytes`、`active_requests`、
  `observability_ring_events`，以及 fresh Engine KV reporting engines/DP ranks、max used
  ratio、min/total free blocks；
- counter：按有限 enum 的 lifecycle、failure reason、terminal result、execution mode、
  recorder outcome 和 output sequence outcome；
- histogram：按有限 execution mode 的 queue wait、TTFT、TPOT、E2E；
- 周期快照：Service incarnation/build、readiness/saturation、Engine member/state/link、
  window terminal/token delta、KV predicted/actual/conflict/fallback 和 event loss。

禁止 request、tenant、model、Engine、incarnation 或 profile 作为 bvar label；这些高基数
维度只进入 VLOG 事件，由离线平台按需聚合。完整字段和排障方法见
[观测与性能分析手册](./OBSERVABILITY_RUNBOOK.md)。

## 需求与 CPU 证据

| Requirement | CPU/Torch CPU | NPU/集群 | 结果 |
| --- | --- | --- | --- |
| schema/wire/presence | RequestEvent protocol、字段号、zero-vs-missing、runtime profile、P/D roundtrip | N/A | PASS |
| 身份与隐私 | UUIDv7、header/traceparent、长度/字符门；JSON 无 prompt/output/token IDs | 待真实 Gateway | PASS |
| 有界并发 recorder | capacity/zero/wrap/FIFO、并发 producer、contention/drop、RAII terminal | N/A | PASS |
| 延迟正确性 | TTFT/ITL/TPOT/E2E、单 token、缺 token、0 TPOT、clock regression | 待真实流式 NPU | PASS |
| Scheduler 生产接入 | ingress/queue/route/dispatch/retry/response/terminal/KV producer 编译和全量回归 | 待真实 P/D | PASS |
| xLLM Engine Debug | attempt lifecycle 与 stage 日志所在生产对象、协议/生命周期回归 | 待 NPU stage | PASS |
| 集群解释 | bounded metrics/snapshot、delta、loss warning、运行手册 | 待多 Service 聚合/阈值 | PASS（机制） |

当前验证基线：Service 全量 pinned/override 均为 388/388，三个生产 ELF build/link
通过；xLLM `6c9d661e` 的 request-output admission wire 3/3、resource adapter/simulator
15/15。前一完整公共基线的 RequestEvent protocol 为 14/14。包含 recorder、latency、
KV route、Provider route、namespace 和 flow-control
在内的 57 项各重复 100 轮，共 5700 次，无失败；同 57 项在 GCC 13
ASan+UBSan（含 leak detection）下通过且没有 sanitizer 报告。

## 回滚与达到 VERIFIED 的条件

- `--v=0` 关闭逐请求事件；执行正确性不依赖 recorder。调整 ring/batch/interval 只影响
  观测容量，不得成为请求 backpressure。若 loss counter 增长，该窗口的 trace 覆盖率
  视为不完整，不能用缺失事件证明链路未发生。
- 达到 `VERIFIED` 仍需：真实 Gateway ID 注入、NPU P/D/Agent 全阶段、跨 Service 日志
  聚合、event_seq 重排、开销/丢失率、告警阈值、长时 soak 与故障演练。硬件未完成不
  影响 CPU core 完成状态，但禁止宣称生产性能已经验证。
