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
- 验证环境和日期：xllm-dev-sandbox，Ubuntu 24.04 ARM64，2026-08-09

## 支持范围

| Provider | Mode | Model/Profile | 支持状态 | 限制与证据 |
| --- | --- | --- | --- | --- |
| xLLM Native | `REMOTE_PD` | P/D 输出序号和执行身份 | CPU_VERIFIED | P 从 0、D 从 1 单调生成；wire 字段号、presence、roundtrip 和角色序号测试通过；实际生产 TU 以 Clang C++20 `-Werror` 编译 |
| xLLM Service + xLLM Native | `REMOTE_PD` | 当前 attempt/incarnation 输出接收与投递 | CPU_VERIFIED | 已按 `(attempt_seq, sender_engine_uid, sender_incarnation_id)` fencing；Service 有界重排，xLLM 以逐请求 FIFO、有界异步队列投递，稳定区分 accepted/rejected/unreachable；真实 brpc 首事件恢复 loopback 和并发竞态测试通过 |
| xLLM Service | `REMOTE_PD` | 缺口 watchdog 和失败收敛 | CPU_VERIFIED | steady clock、弱引用 watchlist、seq=0 同 incarnation Query 精确恢复；Query 失败后仅在首输出、attempt、device-time 和剩余 deadline 四重预算允许时收敛旧 D、递增 attempt 并重选 P/D；迟到旧 attempt 被隔离 |
| xLLM Service + xLLM Native | `REMOTE_PD` | 请求截止时间 | CPU_VERIFIED | Chat/Completion 显式 remaining duration 或 Service 默认 300 s 均转换为逐跳本地 monotonic deadline；Service 有界 deadline/断连队列、P/D hop 重算、D reservation 截断和 Engine 调度边界停止均已接生产路径并完成 CPU 编译/单测 |
| Legacy / 非远程 P-D | 既有输出路径 | 全部 | COMPATIBLE | 未安装 V2 sequencer 时维持原有单发送方亲和线程路径；新增 wire 字段为 additive |
| `LOCAL_PREFILL_DECODE` / `PREFILL_ONLY` / vLLM `AGGREGATED` | V2 首发模式 | 全部 | NOT_IMPLEMENTED | 尚未接各 mode 的生产输出源和统一 deadline/watchdog；vLLM raw JSON relay 不转发 Native deadline 字段 |

这里的 `CPU_VERIFIED` 限定在 xLLM Native `REMOTE_PD` 的 CPU 控制面、真实本地 brpc
loopback、Torch CPU 持有语义和生产编译；不代表 NPU/RDMA 数据面。G2 文档整体仍因
其他 V2 首发 mode、统一 Gateway/profile 策略和 NPU 故障矩阵未完成而保持 `PARTIAL`。

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
- xLLM 非阻塞输出投递：service-routed 输出进入有界 `OutputDeliveryQueue`，每请求
  保证单 worker FIFO、不同请求由 worker pool 隔离；RPC 有 100 ms hard timeout。
  Service 明确接受时出队，未知/已关闭/identity/sequence/commit/terminal/affinity
  拒绝立即终止本地请求，暂时不可达则按 50 ms 重试并在连续 1000 ms 后 abort。
  request/event/byte 三重容量先于无限堆积 fail closed；容量 abort 与在途 callback
  期间保留同 key tombstone，禁止终态请求重新创建队列。动态内存核算覆盖嵌套容器
  capacity 和 Torch tensor 的 retained storage，队列持有输出及 tensor 直到投递完成。
  请求终态与 rate limiter 释放使用一次性 finalizer，不会因网络重试重复记账。
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
  继续请求。真实 brpc client 对 query 做并发 batch、hard timeout、完整 identity/state/
  payload 校验，并覆盖 D 重启与实时首事件竞争。Commit/Cancel/Begin
  响应不回显最多 4 MiB payload，只有显式 Query 返回，避免正常路径双倍传输首事件。
- Query 不可恢复后的 attempt 替换：Service 仅在 `first_token_emitted=false`、重试次数
  未尽、累计浪费 device time 未尽且剩余 deadline 至少 1000 ms 时允许替换。旧 D
  execution hold 必须先取得 Cancel fence 收敛证明，普通 P submission best effort Cancel；
  随后重新选择并绑定 Native P/D、递增 `attempt_seq`、重建 sequencer/hold，再异步派发。
  每个 async HTTP RPC 独占请求 protobuf，避免旧 callback 与新 attempt 并发复用消息；
  旧 attempt 的迟到输出和迟到 dispatch failure 只被拒绝，不关闭当前 attempt。
- 首事件 timer 关系：Service 以同一个权威策略对象生成 P retry budget 和 dispatch
  margin，默认分别为 700 ms 和 200 ms，并在启动时强制
  `retry_budget + dispatch_margin <= output_gap_timeout`。两个 duration 通过 additive
  optional 字段下发，P 在 `GenerationCommit` 开始时创建本地 monotonic deadline；
  Commit 与歧义后的 Query 共享 700 ms 总预算；失败时先把 P 侧 seq=0 terminal output
  放入同一 per-request FIFO，再执行受 200 ms margin 和整请求剩余 deadline 双重截断
  的 best-effort Cancel。缺失或不完整策略对 V2 attempt fail closed，Service 不再等待
  另一套无关 timer。
- 请求 deadline wire：Chat/Completion 分别以 additive optional 字段 48/44 携带
  `remaining_deadline_ms`。绝对时钟值不跨进程；Service、P 和 D 每次接收都用
  `steady_clock` 建立新的本地 deadline，每次发送都重新计算剩余毫秒。缺字段保持 legacy
  兼容，零值、溢出或到达时已过期 fail closed 为 `DEADLINE_EXCEEDED`。
- Service deadline watchdog：显式 duration 或缺省 300 s 的 Native 请求进入独立的
  有界 multimap 索引，默认
  最多 65,536 条、每轮最多取 1,024 条过期请求；索引持有弱引用，请求正常完成时主动
  删除。有积压时立即按下一有界批次继续追赶，不额外等待扫描周期。watchdog 复用
  output mutex、唯一终态、Cancel 和 hold detach 路径，不扫描 `requests_` 全表；并发
  插入和小批次回收已有 CPU 测试。每个请求在 dispatch 前同时预留断连监控容量，brpc
  disconnect callback 只标记已存在槽位，watchdog 有界取走并走同一终止/回收路径。
- Engine deadline：`Request` 保存本地 monotonic deadline，并在连续调度、统一调度、
  fixed-steps、zero-eviction、REMOTE_PD 和 P/D OOC 的排队/运行边界停止请求并释放 KV。
  P 在 Decode dispatch 前、D admission 后和 `GenerationCommit` 前复查；发往 D 的
  duration 重新计算。D 在 allocator 前拒绝过期请求，并把 reservation TTL 截断到
  剩余业务时限。cancel/deadline 共用 first-writer-wins 原子终止原因，后到 deadline
  不会覆盖先发生的 cancel；最终输出携带对应稳定状态。
- 配置：Service 使用 `output_reorder_max_events=64`、
  `output_reorder_max_bytes=4 MiB`、
  `request_watchdog_interval_ms=100`、`output_gap_timeout_ms=1000`、
  `output_gap_query_timeout_ms=100`、`output_gap_query_batch_size=8`、
  `p_first_event_retry_ub_ms=700`、`first_event_dispatch_margin_ms=200`、
  `max_first_output_attempt_retries=1`、
  `max_nonstream_retry_wasted_device_ms=5000`、
  `min_first_output_retry_remaining_ms=1000`、
  `default_request_deadline_ms=300000`。xLLM 使用
  `output_delivery_worker_threads=8`、`output_delivery_rpc_timeout_ms=100`、
  `output_delivery_max_pending_requests=65536`、
  `output_delivery_max_events_per_request=64`、
  `output_delivery_max_bytes_per_request=4 MiB`、
  `output_delivery_retry_interval_ms=50`、
  `subscriber_unreachable_abort_ms=1000`。构造时拒绝
  零容量、非正 timer、Query timeout 大于 gap timeout、deadline 零容量/零批次、
  scan interval 大于 gap timeout、不完整的启用重试预算，或 retry budget 与 margin 之和
  超过 gap timeout。
- 明确不支持范围：Gateway、租户策略和 profile 尚未提供统一 deadline 策略；vLLM raw
  relay 和其余 V2 mode 尚未接入；attempt 替换尚缺包含真实 Scheduler/InstanceMgr 和两组
  Engine 的完整多进程故障 loopback；配置尚未接 profile；无 sanitizer 全链、NPU、RDMA
  或真实流式数据面故障矩阵结论。稳定 per-item delivery reason 已进入 additive wire，
  但对应 reason/重试/device-time 指标仍未完整接入观测系统。

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
| G2 xLLM 非阻塞输出投递 | 14 项覆盖逐请求 FIFO、跨请求隔离、unreachable 重试/超时、稳定 reject、容量 abort、abort 期间同 key 防复活、一次性 abort/finalize 原子门、嵌套容量与 16×32 并发；全组重复 100 轮共 1400 次 | 3 项直接使用 Torch CPU：非连续 view retained storage、源 tensor 释放后的所有权、空 tensor；队列按 storage bytes 计费 | 待真实 NPU tensor/output RPC | PASS |
| G2 delivery reason wire | 双仓固定 enum 名称/数值与字段号；Service proto 双向二进制解析并逐项比较 descriptor；unknown/closed/identity/sequence/commit/terminal/affinity 映射为稳定 per-item code | N/A，纯协议 | N/A | PASS |
| G2 seq=0 Query 恢复 | xLLM 首事件 exact adapter、wire field 24、identity/seq/4 MiB 上限；Service 7 项真实 brpc loopback 覆盖精确 payload、并发 batch、hard timeout、D 重启、live/recovery 竞争和边界拒绝；P/D/Service 生产对象严格编译 | xLLM adapter 目标链接 Torch CPU；本逻辑不执行 tensor 数值计算 | 待真实 P 退出、D 重启和数据流 | PASS（CPU loopback） |
| G2 首输出前 attempt 替换 | 8 项 fake monotonic budget 覆盖次数、首输出、deadline、累计 device time、时钟回退、禁用和非法配置；production 接线覆盖旧 hold Cancel fence、递增 attempt、重选 P/D、sequencer/hold 重建、旧 attempt fencing 和 attempt-scoped dispatch failure | N/A，纯控制/RPC | 待真实两组 P/D 故障注入 | PASS（CPU 核心与生产接线）；完整 Scheduler loopback 待补 |
| G2 主动客户端断连 | 5 项覆盖预留容量、幂等有界通知、弱引用回收、并发 exactly-once 和 close；brpc callback 接入 request watchdog 同一终止路径 | N/A，纯控制 | 待真实客户端断流与 KV 回收 | PASS（CPU 核心与生产接线） |
| G2 Service 关键竞态稳定性 | 首事件 loopback、断连、重试预算、delivery wire 和 pre-dispatch hold 回滚共 22 项各重复 100 轮，共 2200 次 | N/A | N/A | PASS |
| 双仓回归 | xLLM 默认 CPU 六目标 96/96，另有 output queue 14/14、protocol 14/14；Service 205/205；Service 三个生产二进制 build/link verify | Torch CPU queue/所有权测试通过 | N/A | PASS |

## 完善情况

- 已完成：双仓 output identity wire、P/D 序号源、真实 xLLM RPC 序列化、Service
  attempt/incarnation fencing、有界重排、重复/terminal 防护、弱引用 gap watchdog、
  affinity 有序失败、xLLM 有界逐请求 FIFO 与稳定投递 reason、主动客户端断连、
  execution hold Cancel/detach 收敛，以及 Native 默认/显式 request deadline 的
  Service/P/D duration 传播、有界 watchdog、Engine 本地停止和
  D reservation cap 接线；P retry budget、dispatch margin 与 Service gap timeout 的
  不等式已由共享策略和生产启动校验强制执行；seq=0 gap 已优先 Query D 保存的 exact
  首事件并经同一 sequencer 恢复；Query 无 payload、超时或 D 已终止时可在四重预算内
  收敛旧执行并创建新 attempt，否则明确失败。
- 已知缺口/风险：attempt 替换已完成核心预算、生产接线和受影响对象/二进制编译，但
  尚无同时启动 Service Scheduler、InstanceMgr 和两组真实 P/D 的多进程 loopback；
  Gateway/profile 的 deadline 策略、NPU KV 实际释放时延、P/D 真实流式乱序与 Cancel
  丢失仍待对应环境验证。当前 CPU loopback 只覆盖首事件 recovery client 及 wire，不把
  本机 brpc 结果外推成 NPU/RDMA 结论。
- 回滚与兼容：字段全部 additive；未安装 sequencer 的 legacy/非 REMOTE_PD 请求继续
  使用原路径。Native REMOTE_PD 已安装 sequencer 后严格要求 sequence 和 identity，
  因此双仓必须同步发布或由 capability/version 门禁阻止新旧混跑。
- 性能、容量和观测证据：Service 单请求重排有 event/byte/distance 三重上限，xLLM
  投递有 request/event/retained-byte 三重上限；watchdog 只扫描活跃 gap、断连或到期
  deadline，deadline/断连索引有 65,536 record 和 1,024/轮双上限。尚无 1 万并发请求
  扫描开销、buffer/deadline 水位、delivery/gap reason 和迟到事件的完整 metrics。
- 当前 `REMOTE_PD` CPU_VERIFIED 之后仍需完成：完整 Scheduler 双 P/D fault loop、
  Gateway/profile deadline 策略、vLLM/其余 mode 接入、稳定 reason/预算指标和 profile
  配置化；这些是后续大阶段，不回退本批 CPU 结论。
- 达到 VERIFIED 仍需完成：NPU P→D 流式乱序、Cancel 丢失、Engine 重启/incarnation
  变化、deadline 资源释放和 1 万次故障门禁。
