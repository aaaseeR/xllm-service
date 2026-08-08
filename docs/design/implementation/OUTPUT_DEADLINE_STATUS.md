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

# G2 输出定序、缺口与请求截止时间 CPU 核心

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
| xLLM Service | `REMOTE_PD` | 缺口 watchdog 和失败收敛 | PARTIAL | 本地 steady clock、弱引用 watchlist、seq=0 同 incarnation Query 精确恢复、明确 `DEADLINE_EXCEEDED`、Cancel/hold detach 已接生产路径；尚缺 fake-brpc 竞态和 Query 失败后的预算内重试 |
| xLLM Service + xLLM Native | `REMOTE_PD` | 显式请求截止时间 | PARTIAL | Chat/Completion 显式携带 remaining duration 时，Service、P、D 均转换为本地 monotonic deadline；Service 有界 deadline 队列、P/D hop 重算、D reservation 截断和 Engine 调度边界停止已接生产路径并完成 CPU 编译/单测；尚缺 Gateway/profile 默认值和 loopback 故障矩阵 |
| Legacy / 非远程 P-D | 既有输出路径 | 全部 | COMPATIBLE | 未安装 V2 sequencer 时维持原有单发送方亲和线程路径；新增 wire 字段为 additive |
| `LOCAL_PREFILL_DECODE` / `PREFILL_ONLY` / vLLM `AGGREGATED` | V2 首发模式 | 全部 | NOT_IMPLEMENTED | 尚未接各 mode 的生产输出源和统一 deadline/watchdog；vLLM raw JSON relay 不转发 Native deadline 字段 |

这里的 `CPU_VERIFIED` 只描述独立协议核心和生产编译。G2 仍缺统一默认 deadline、
Query 失败后的完整重试策略和端到端故障矩阵，因此整体在这些生产闭环完成前保持
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
- seq=0 Query 恢复：P 在 `GenerationCommit` 前只生成一次客户端首事件，使用与实时
  `Generations` 相同的权威 adapter 序列化；D 在 fingerprint、attempt、sender 和
  seq=0 校验后最多保留 4 MiB 原始 payload，并通过 live/tombstone `QueryRequest`
  返回。Service 只对 `next_expected_seq=0` 的到期 gap 发起同一 D incarnation Query，
  默认并发批次 8、单 RPC 100 ms，并受整请求剩余 deadline 截断；响应必须处于
  `GENERATION_COMMITTED/RUNNING/DONE` 且完全匹配 D key、P sender、attempt 和 seq=0。
  校验成功后复用现有 sequencer 补洞并释放已缓存的 D 事件，不重新 tokenize、decode
  或 replay parser；实时 P 发送与 Query 恢复竞争时由 seq duplicate 规则幂等收敛。
  P 进程在 commit ACK 后、实时 seq=0 到达前退出时，Service 不再立即删除请求，而是
  将它放入同一个弱引用、有界批次恢复队列；下一轮 watchdog 立即 Query 绑定 D，成功则
  继续请求，无法证明 committed exact event 则明确 `CANCELLED`。Commit/Cancel/Begin
  响应不回显最多 4 MiB payload，只有显式 Query 返回，避免正常路径双倍传输首事件。
- 首事件 timer 关系：Service 以同一个权威策略对象生成 P retry budget 和 dispatch
  margin，默认分别为 700 ms 和 200 ms，并在启动时强制
  `retry_budget + dispatch_margin <= output_gap_timeout`。两个 duration 通过 additive
  optional 字段下发，P 在 `GenerationCommit` 开始时创建本地 monotonic deadline；
  Commit 与歧义后的 Query 共享 700 ms 总预算；失败时先把 P 侧 seq=0 terminal output
  放入同一 per-request FIFO，再执行受 200 ms margin 和整请求剩余 deadline 双重截断
  的 best-effort Cancel。缺失或不完整策略对 V2 attempt fail closed，Service 不再等待
  另一套无关 timer；端到端投递时延仍需 fake-brpc loopback 验证。
- 请求 deadline wire：Chat/Completion 分别以 additive optional 字段 48/44 携带
  `remaining_deadline_ms`。绝对时钟值不跨进程；Service、P 和 D 每次接收都用
  `steady_clock` 建立新的本地 deadline，每次发送都重新计算剩余毫秒。缺字段保持 legacy
  兼容，零值、溢出或到达时已过期 fail closed 为 `DEADLINE_EXCEEDED`。
- Service deadline watchdog：显式 deadline 请求进入独立的有界 multimap 索引，默认
  最多 65,536 条、每轮最多取 1,024 条过期请求；索引持有弱引用，请求正常完成时主动
  删除。有积压时立即按下一有界批次继续追赶，不额外等待扫描周期。watchdog 复用
  output mutex、唯一终态、Cancel 和 hold detach 路径，不扫描 `requests_` 全表；并发
  插入和小批次回收已有 CPU 测试。
- Engine deadline：`Request` 保存本地 monotonic deadline，并在连续调度、统一调度、
  fixed-steps、zero-eviction、REMOTE_PD 和 P/D OOC 的排队/运行边界停止请求并释放 KV。
  P 在 Decode dispatch 前、D admission 后和 `GenerationCommit` 前复查；发往 D 的
  duration 重新计算。D 在 allocator 前拒绝过期请求，并把 reservation TTL 截断到
  剩余业务时限。cancel/deadline 共用 first-writer-wins 原子终止原因，后到 deadline
  不会覆盖先发生的 cancel；最终输出携带对应稳定状态。
- 配置：`output_reorder_max_events=64`、`output_reorder_max_bytes=4 MiB`、
  `request_watchdog_interval_ms=100`、`output_gap_timeout_ms=1000`、
  `output_gap_query_timeout_ms=100`、`output_gap_query_batch_size=8`、
  `p_first_event_retry_ub_ms=700`、`first_event_dispatch_margin_ms=200`；构造时拒绝
  零容量、非正 timer、Query timeout 大于 gap timeout、deadline 零容量/零批次、
  scan interval 大于 gap timeout，或 retry budget 与 margin 之和超过 gap timeout。
- 明确不支持范围：当前只有请求显式携带 `remaining_deadline_ms` 才安装 deadline，
  Gateway、租户策略和 profile 尚未提供统一默认值；vLLM raw relay 和其余 V2 mode 尚未
  接入；seq=0 Query 不可用时尚未按 retry/device/deadline 预算 cancel 旧 D 并创建新
  attempt；未知/已终止输出只有既有 per-item `false`，尚无稳定 reason/metrics；配置
  尚未接 profile/CLI；无 fake-brpc、sanitizer 全链、NPU、RDMA 或真实流式故障矩阵
  结论。

## 需求与测试追踪

| Requirement ID | CPU test | Torch CPU test | NPU test | 结果 |
| --- | --- | --- | --- | --- |
| G2 wire 与角色序号 | xLLM `RequestEventProtocolTest` 14 项中的 wire/presence/字段号/角色 counter；Service adapter 跨 proto 解析与完整转换 | 公共 xLLM 目标使用 Torch CPU 环境；本逻辑不执行 tensor | 待真实 P/D | PASS |
| G2 attempt/incarnation fencing | `RemotePdIdentityFencesAttemptAndIncarnation`、`ErrorCanComeFromEitherBoundSender` | N/A，纯控制协议 | 待进程重启迟到输出 | PASS |
| G2 连续交付与幂等 | reorder、已交付/已缓存 duplicate、terminal fence/overtake 共 4 项 | N/A | 待双发送方流式输出 | PASS |
| G2 容量与输入边界 | event count、decoded byte、sequence distance、missing、sentinel 共 4 项 | N/A | N/A | PASS |
| G2 本地 gap timer | fake monotonic timeout、填洞后不超时、close 后拒绝 | N/A | 待 wall-clock fault injection | PASS |
| G2 稳定性 | 12 个 sequencer 测试重复 100 轮，共 1200 次 | N/A | N/A | PASS |
| G2 request deadline wire/时钟/终止竞态 | optional presence、固定字段号、wire roundtrip、fake monotonic 到期/零值/溢出、cancel/deadline 顺序和 16 线程终止竞争；定向 4 项重复 100 轮共 400 次 | 公共 xLLM 目标使用 Torch CPU 环境；本逻辑不执行 tensor | 待跨机时钟偏差/暂停故障 | PASS（CPU 核心） |
| G2 Service deadline 索引 | 顺序到期、7 条小批次、容量/重复/零配置/弱引用/主动删除/超大 batch 截断；8 线程并发写入 256 条无丢失；6 项重复 100 轮共 600 次 | N/A，纯控制路径 | N/A | PASS |
| G2 P retry timer 关系 | fake monotonic policy、零值/溢出、等号边界和超界；Service 默认值/负值/错误序关系 fail-closed；xLLM 1 项重复 100 轮、Service 2 项重复 100 轮共 200 次；P Commit/Query/Cancel 生产对象以 C++20 `-Werror` 编译 | N/A，纯控制/RPC timeout | 待真实 RPC timeout 与迟到 ACK | PASS（CPU 核心）；loopback 待补 |
| G2 Engine stop/资源边界 | Request parse/output、Service/P/D hop、D admission/reservation cap、六类调度路径共 10 个受影响生产 TU 以 Clang C++20 `-Werror` 编译 | 生产目标链接 Torch CPU；本批无 tensor 数值变化 | 待真实 KV/执行中 cancel | PASS（CPU 编译）；loopback 待补 |
| 双仓回归 | xLLM 98/98；Service 183/183；Service 三个生产二进制 build/link verify | Torch CPU 依赖可加载；本批无 tensor 数值变化 | N/A | PASS |
| G2 seq=0 Query 恢复 | xLLM 首事件 exact adapter、wire field 24、identity/seq/4 MiB 上限；Service UTF-8/token/logprob 精确恢复、state/D/P/attempt/seq/payload/index 拒绝；xLLM protocol 1 项、adapter 2 项和 Service 2 项各重复 100 轮；gap/reorder 3 项重复 100 轮；P/D/Service 生产对象严格编译 | xLLM adapter 目标链接 Torch CPU；本逻辑不执行 tensor 数值计算 | 待 fake-brpc P 退出、RPC timeout、D 重启和真实流 | PASS（CPU 核心与生产接线）；loopback 待补 |

## 完善情况

- 已完成：双仓 output identity wire、P/D 序号源、真实 xLLM RPC 序列化、Service
  attempt/incarnation fencing、有界重排、重复/terminal 防护、弱引用 gap watchdog、
  affinity 有序失败、客户端断连抑制、execution hold Cancel/detach 收敛，以及显式 Native
  request deadline 的 Service/P/D duration 传播、有界 watchdog、Engine 本地停止和
  D reservation cap 接线；P retry budget、dispatch margin 与 Service gap timeout 的
  不等式已由共享策略和生产启动校验强制执行；seq=0 gap 已优先 Query D 保存的 exact
  首事件并经同一 sequencer 恢复。
- 已知缺口/风险：Query 无 payload、超时或 D 已终止时当前会明确失败，尚未实现设计
  允许的预算内 cancel/retry 新 attempt；未携带显式 deadline 的请求仍可能无限等待；
  deadline 与 Query 的生产并发路径主要由纯核心测试和严格编译覆盖，仍需 fake
  provider loopback 验证 callback、P 退出、instance failure、watchdog、正常 terminal
  和 KV 释放的竞态。
- 回滚与兼容：字段全部 additive；未安装 sequencer 的 legacy/非 REMOTE_PD 请求继续
  使用原路径。Native REMOTE_PD 已安装 sequencer 后严格要求 sequence 和 identity，
  因此双仓必须同步发布或由 capability/version 门禁阻止新旧混跑。
- 性能、容量和观测证据：单请求缓存有 event/byte/distance 三重上限，watchdog 只扫描
  活跃 gap 或到期 deadline；deadline 索引有 65,536 record 和 1,024/轮双上限；尚无
  1 万并发请求的扫描开销、buffer/deadline 水位、gap reason 和迟到事件指标。
- 达到 CPU_VERIFIED 仍需完成：fake-brpc 双发送方/断连/实例失效/Query timeout 竞态，
  Query 不可恢复时的预算内 cancel/retry，Gateway/profile 默认 deadline，vLLM/其余
  mode 接入，稳定 per-item reason/metrics 及配置化。
- 达到 VERIFIED 仍需完成：NPU P→D 流式乱序、Cancel 丢失、Engine 重启/incarnation
  变化、deadline 资源释放和 1 万次故障门禁。
