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

# G2 跨发送方输出定序与缺口超时 CPU 核心

## 基本信息

- Owner：xLLM Service V2
- 状态：PARTIAL
- 关联设计/Requirement ID：G2、02 §3.2/§5.1/§7/§11/§14.1、F21、F33、F39
- 最近验证基线：xLLM 与 xllm-service `service_dev` 本状态文档所在提交
- 验证环境和日期：xllm-dev-sandbox，Ubuntu 24.04 ARM64，2026-08-08

## 支持范围

| Provider | Mode | Model/Profile | 支持状态 | 限制与证据 |
| --- | --- | --- | --- | --- |
| xLLM Native | `REMOTE_PD` | P/D 输出序号和执行身份 | CPU_VERIFIED | P 从 0、D 从 1 单调生成；wire 字段号、presence、roundtrip 和角色序号测试通过；实际生产 TU 以 Clang C++20 `-Werror` 编译 |
| xLLM Service + xLLM Native | `REMOTE_PD` | 当前 attempt/incarnation 输出接收 | PARTIAL | 已按 `(attempt_seq, sender_engine_uid, sender_incarnation_id)` fencing，并在 request→thread 投递前执行有界重排；尚缺 fake-brpc 跨线程 loopback 和 NPU 数据面验证 |
| xLLM Service | `REMOTE_PD` | 缺口 watchdog 和失败收敛 | PARTIAL | 本地 steady clock、弱引用 watchlist、明确 `DEADLINE_EXCEEDED`、Cancel/hold detach 已接生产路径；只覆盖 output gap，尚未实现整请求 deadline 和 seq=0 Query 恢复 |
| Legacy / 非远程 P-D | 既有输出路径 | 全部 | COMPATIBLE | 未安装 V2 sequencer 时维持原有单发送方亲和线程路径；新增 wire 字段为 additive |
| `LOCAL_PREFILL_DECODE` / `PREFILL_ONLY` / vLLM `AGGREGATED` | V2 首发模式 | 全部 | NOT_IMPLEMENTED | 尚未接各 mode 的生产输出源、deadline 和统一 watchdog |

这里的 `CPU_VERIFIED` 只描述独立的序号生成和重排核心。G2 同时包含整请求
deadline、Engine 本地停止和恢复策略，因此整体在这些生产闭环完成前保持
`PARTIAL`。

## 实现

- 双仓 additive wire：两个 `DisaggStreamGeneration` 均在固定字段 8--11 增加
  `optional output_event_seq`、`optional attempt_seq`、`sender_engine_uid` 和
  `sender_incarnation_id`。Service 有跨 package 二进制解析测试，避免两个 proto
  副本静默漂移。
- xLLM 输出源：`OutputEventSequenceCounter` 是每个 `Request` 的原子计数器；
  `AsyncResponseProcessor` 仅在 service-routed 路径赋值，Prefill/默认/MIX 从 0
  开始，Decode 从 1 开始，并复制当前 correlation 的 `attempt_seq`。失败、batch
  complete 和 batch stream 三条输出路径共用同一入口。`XServiceClient` 在序列化时
  固定写入本 Engine 的 UID/incarnation。
- Service identity fencing：Native `REMOTE_PD` 在 dispatch 前已经绑定 P/D
  incarnation。非错误 P 事件必须带 `finished_on_prefill_instance` 且匹配 P；非错误
  D 事件必须匹配 D；结构化错误可由任一已绑定发送方发出。缺字段、旧 attempt、旧
  incarnation 或错误 sender 均 fail closed。
- Service 有界重排：每请求默认最多缓存 64 个事件、4 MiB 动态 payload，最大序号
  距离 64；同时限制 terminal 越序、重复事件、`UINT64_MAX` sentinel 和缺失序号。
  内存估算覆盖字符串、token、finish reason、logprob 和 top-logprob 的动态 capacity。
- 顺序和终态：跨 P/D 的唯一正确性顺序来自 `output_event_seq`；只有从 0 开始连续的
  vector 才一次性进入原 request affinity thread。重复事件被幂等丢弃；正常 terminal、
  status error、watchdog、实例失效和非法输出在同一个 request output mutex 下竞争
  唯一关闭权，错误也排入相同 affinity thread，避免越过已接受 token。
- gap watchdog：只有实际存在 gap 的请求进入弱引用 watchlist，不全表扫描
  `requests_`。默认每 100 ms 检查一次，以本地 monotonic time 判定 1000 ms gap；
  超时后向仍连接客户端投递明确终态，关闭 sequencer，通过现有 hold 路径向绑定 holder
  Cancel，并从 request、watchlist 和输出线程映射清理。停机共享条件变量并 join。
- 配置：`output_reorder_max_events=64`、`output_reorder_max_bytes=4 MiB`、
  `request_watchdog_interval_ms=100`、`output_gap_timeout_ms=1000`；构造时拒绝零容量、
  非正 timer 和 scan interval 大于 gap timeout。
- 明确不支持范围：未实现 `request_deadline/remaining_deadline_ms` 的端到端传播和
  P/D 本地调度边界停止；未强制 `p_first_event_retry_ub + margin <= gap timeout`；gap
  到期尚未先 Query D 保存的 FirstGeneration 来恢复 seq=0；未知/已终止输出只有既有
  per-item `false`，尚无稳定 reason/metrics；配置尚未接 profile/CLI；无 fake-brpc、
  sanitizer 全链、NPU、RDMA 或真实流式故障矩阵结论。

## 需求与测试追踪

| Requirement ID | CPU test | Torch CPU test | NPU test | 结果 |
| --- | --- | --- | --- | --- |
| G2 wire 与角色序号 | xLLM `RequestEventProtocolTest` 9 项中的 wire/presence/字段号/角色 counter；Service adapter 跨 proto 解析与完整转换 | 公共 xLLM 目标使用 Torch CPU 环境；本逻辑不执行 tensor | 待真实 P/D | PASS |
| G2 attempt/incarnation fencing | `RemotePdIdentityFencesAttemptAndIncarnation`、`ErrorCanComeFromEitherBoundSender` | N/A，纯控制协议 | 待进程重启迟到输出 | PASS |
| G2 连续交付与幂等 | reorder、已交付/已缓存 duplicate、terminal fence/overtake 共 4 项 | N/A | 待双发送方流式输出 | PASS |
| G2 容量与输入边界 | event count、decoded byte、sequence distance、missing、sentinel 共 4 项 | N/A | N/A | PASS |
| G2 本地 gap timer | fake monotonic timeout、填洞后不超时、close 后拒绝 | N/A | 待 wall-clock fault injection | PASS |
| G2 稳定性 | 12 个 sequencer 测试重复 100 轮，共 1200 次 | N/A | N/A | PASS |
| 双仓回归 | xLLM 90/90；Service 173/173；xLLM 两个受影响生产对象严格编译；Service 三个生产二进制 build/link | Torch CPU 依赖可加载；本批无 tensor 数值变化 | N/A | PASS |
| G2 整请求 deadline/Engine stop | 未实现 | 未实现 | 未实现 | OPEN |
| G2 seq=0 Query 恢复与 P retry timer 关系 | 未实现 | N/A | 未实现 | OPEN |

## 完善情况

- 已完成：双仓 output identity wire、P/D 序号源、真实 xLLM RPC 序列化、Service
  attempt/incarnation fencing、有界重排、重复/terminal 防护、弱引用 gap watchdog、
  affinity 有序失败、客户端断连抑制和 execution hold Cancel/detach 收敛接线。
- 已知缺口/风险：当前 gap 超时是安全的明确失败，不是设计要求的优先 Query 恢复；
  缺少整请求 deadline 时，无 gap 但持续慢速输出仍可超过业务时限；生产 Scheduler 的
  并发路径主要由纯核心测试和严格编译覆盖，仍需 fake provider loopback 验证 callback、
  instance failure、watchdog 和正常 terminal 的竞态。
- 回滚与兼容：字段全部 additive；未安装 sequencer 的 legacy/非 REMOTE_PD 请求继续
  使用原路径。Native REMOTE_PD 已安装 sequencer 后严格要求 sequence 和 identity，
  因此双仓必须同步发布或由 capability/version 门禁阻止新旧混跑。
- 性能、容量和观测证据：单请求缓存有 event/byte/distance 三重上限，watchdog 只扫描
  活跃 gap；尚无 1 万并发请求的扫描开销、buffer 水位、gap reason 和迟到事件指标。
- 达到 CPU_VERIFIED 仍需完成：fake-brpc 双发送方/断连/实例失效竞态，seq=0 Query
  恢复，P retry timer 不等式，整请求 deadline 和 P/D 本地 monotonic stop，稳定
  per-item reason/metrics 及 profile 配置。
- 达到 VERIFIED 仍需完成：NPU P→D 流式乱序、Cancel 丢失、Engine 重启/incarnation
  变化、deadline 资源释放和 1 万次故障门禁。
