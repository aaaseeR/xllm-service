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
- 验证环境和日期：xllm-dev-sandbox，Ubuntu 24.04 ARM64，2026-08-09

## 支持范围

| Provider | Mode | Model/Profile | 支持状态 | 限制与证据 |
| --- | --- | --- | --- | --- |
| xLLM Native | `REMOTE_PD` | Engine attempt/resource protocol CPU core | CPU_VERIFIED | 20 个状态机测试 + 2 个 fingerprint 测试；并发幂等准入、cancel fence、TTL、GenerationCommit、output gate、tombstone 和精确释放通过 |
| xLLM Native | `REMOTE_PD` | P/D 生产控制路径 | CPU_VERIFIED | 已接真实 D KV allocator/scheduler、P dispatch/handoff、四个 V2 RPC、P/D 输出门禁、Service hold、seq 重排、默认/显式 request deadline、有界输出投递、真实 brpc seq=0 Query 恢复和 Query 失败后的预算内 attempt 替换；尚缺 NPU 数据面故障矩阵 |
| xLLM Native | `LOCAL_PREFILL_DECODE` / `PREFILL_ONLY` | 公共 attempt schema | PARTIAL | 状态和 reason 可表达；尚未接入对应 allocator/scheduler |
| xLLM Service | Provider-neutral 全模式 | `ExecutionResourceHold` / cleanup capacity | CPU_VERIFIED | 26 个测试；dispatch 前 token、pre-dispatch rollback、单 hold、候选收敛、旧 attempt/incarnation fencing、record/byte 上限和有界公平重试批次通过 |
| xLLM Service + xLLM Native | `REMOTE_PD` | 生产 dispatch/输出/取消/进程终止 hold | PARTIAL | Request 直接持有 hold；选定 D incarnation 在 RPC 前安装；P commit、D terminal、Cancel fence、后台 Query/Cancel 重试和精确进程终止驱动收敛；暂缺 loopback race 测试与稳定错误映射 |
| vLLM-Ascend | `AGGREGATED` | strict Agent + Service hold | CPU_VERIFIED / NPU_PENDING | dispatch 前安装单 holder hold；Submit/Query/Cancel、cancel-before-create、deadline 和 cleanup 收敛已完成 CPU loopback |

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
- Engine 生产接线：D `AddNewRequests` 以 `(request_uid, attempt_seq,
  decode_incarnation_id)` 和确定性 SHA-256 fingerprint 执行幂等硬准入，真实
  `try_allocate/decode_schedule` 是 reservation 权威；`BeginTransfer`、
  `CommitGeneration`、`CancelRequest`、`QueryRequest` 已接入 brpc handler。
  reservation 在 commit 前由 lifecycle table 释放，commit 后所有权转给 scheduler，
  terminal finish 不再误 cancel，显式 cancel/析构/sweep 均走唯一清理路径。
- Engine 成员租约：incarnation 与进程内 attempt ledger 不可分离；etcd 权威 key
  丢失后保持 fail-stop，不以连续 missing 延迟 fencing，也不在原进程复用 identity。
  默认且最小 TTL 为 15 秒，heartbeat/reconcile 必须处于 `(0, ttl/3]`，非法配置在
  初始化阶段拒绝；KeepAlive 失败显式记录 key、TTL 和异常，供区分续租故障与最终
  authoritative missing。
- P 生产接线：请求携带绑定的 Decode incarnation、reservation TTL 和 transfer
  mode；临时拒绝只重放同一 immutable attempt，永久拒绝失败关闭；commit ACK
  丢失时 Query 同一 incarnation，只有已 committed/running/done 才开放首事件。
  P 和 D 两端均在 `GenerationCommit` 前阻断输出，D 的并发 output callback 会等待
  commit 原子转换完成。Service 已把 Registry 选中的 P/D incarnation 写入请求 wire。
- 首事件恢复：P 在 commit 前用实时输出路径的同一 adapter 生成并序列化 exact seq=0，
  D 将它纳入 immutable handoff fingerprint，在 4 MiB 上限、attempt/identity/seq 校验
  后保留到 live/tombstone。Service 的 gap watchdog 以 8 条并发批次和 100 ms 单 RPC
  上限 Query 同一 D incarnation，只接受 committed/running/done 与绑定 P sender 完全
  匹配的 payload，再通过原 sequencer 幂等补洞；P 进程在 ACK 后退出也进入同一有界
  恢复队列，不从 token 二次解码客户端文本。payload 只在 Query 返回，正常 commit、
  cancel 和 begin 响应不回显。恢复 client 已用真实本地 brpc 覆盖并发 batch、hard
  timeout、D 重启和 live/recovery 竞争；不可恢复时在四重预算允许且旧 D Cancel fence
  已收敛后递增 attempt、重选 P/D 并重新安装 sequencer/hold。
- deadline 与 reservation：Service、P、D 只跨 wire 传播每跳重新计算的 remaining
  duration，不传播绝对时钟；D 在 allocator 前拒绝到期请求，并把 reservation TTL
  截断到本地剩余 deadline。P 在 dispatch、D admission 后和 GenerationCommit 前
  复查，Engine 的连续/统一/fixed/zero/OOC 调度边界会停止到期请求并释放 KV。Native
  请求未显式携带 duration 时由 Service 安装默认 300 s 本地 deadline 并逐跳传播。
- Service CPU 核心：`xllm_service/provider/execution_hold.*`。move-only cleanup
  reservation 在 dispatch 前按 record 和 bytes 双上限预留，同一 token 从
  RequestContext 转移到最小 cleanup record，收敛后自动释放。
- Service 生产接线：每个推理 `Request` 直接拥有单一 hold；Scheduler 在选定并绑定
  P/D incarnation 后、发出 Prefill RPC 前原子预留 cleanup capacity 并安装选定 D
  holder。P 首个成功事件作为 `GENERATION_COMMITTED` 证明，D 成功终态作为 terminal
  证明；失败、断连和发送失败同步向同一 attempt/incarnation 发 `CancelRequest`，只接受
  key 完全匹配的终态 ACK，否则把最小 hold 连同原 token 转入 cleanup 表。Registry
  精确报告 holder 进程终止时，同时收敛 active 与 detached hold；instance 注销先关闭
  dispatch gate，Request 登记会重新验证绑定 incarnation，登记/移除/detach 与进程终止
  proof 按固定锁序串行，避免 proof-before-adopt。异步 Prefill callback 持有
  `shared_ptr<brpc::Channel>`，不会因 Registry channel 替换而悬空。
- vLLM `AGGREGATED` 复用同一 cleanup reservation/table，不维护第二套资源账本。
  Service 在 Agent HTTP dispatch 前安装单 holder hold，成功响应头确认
  GenerationCommit；正常终态、Agent Query terminal 或 Cancel fence 才能释放。失败、
  断连和 deadline 使用同一 Cancel/detach 路径。Agent ledger 对同 key exactly-once，
  未知 Cancel 安装有界 `CANCELLED_BEFORE_CREATE` fence。Agent 的普通 attempt 与
  negative fence 使用独立容量和 TTL；fence 池满不返回伪 ACK，而是停止新准入并在
  EngineState 发布 `DRAINING`。可取消 HTTP connection 在 Cancel/deadline/fence 时
  shutdown socket，即使 vLLM 尚未返回响应头也能在 CPU loopback 上立即解除代理持有。
- Service detached cleanup worker：默认每秒从 cleanup 表按 round-robin 取最多 8 条
  unresolved record，每个 RPC 使用 100 ms timeout；对每个未收敛 holder 先 Query，
  Query 只有返回精确 attempt/incarnation 的终态才作为证明，否则再发送 Cancel，且只
  接受精确终态 fence ACK。worker 不持有 cleanup 表锁执行 RPC，停机通过条件变量唤醒
  并 join，因此单轮工作量、锁持有时间和退出等待都有界。
- hold 安全规则：`likely_holder` 不缩小候选；只有明确 `PROVEN_PRECOMMIT` 或
  `GENERATION_COMMITTED` 证明可以写 confirmed holder 并收窄；所有安全候选必须
  分别取得 terminal/cancel-fence/self-fence/process-terminated/允许的硬时间证明。
  `QUERY_ABSENT` 永远不是证明，硬时间证明默认禁用。
- 明确不支持范围：Service cleanup capacity、deadline capacity 与后台重试参数目前有
  `Options`/CLI 保守默认值，尚无 profile 配置、worker RPC/失败 metrics 和稳定 HTTP
  `SERVICE_CLEANUP_CAPACITY_RETRYABLE` 映射；当前 5 分钟 Engine
  reservation 上限已在显式 request deadline 存在时截断，但 30 秒 transfer-start TTL
  尚未从 deadline/profile 推导；
  `BeginTransfer` 在 P 接受 admission 后触发，尚未绑定首个 DMA primitive；未实现
  local production hold 和 deadline，也未提供 NPU、RDMA 或端到端故障注入
  结论。REMOTE_PD output seq/reorder、gap watchdog 和默认/显式 request deadline 已进入
  G2 生产路径，详细边界见 `OUTPUT_DEADLINE_STATUS.md`。

## 需求与测试追踪

| Requirement ID | CPU test | Torch CPU test | NPU test | 结果 |
| --- | --- | --- | --- | --- |
| G1 三态准入与稳定 reason | `ReasonsHaveStableThreeWayClassification`；provider golden wire/字段号 | N/A，纯控制协议 | 待真实 allocator | PASS |
| G1 幂等资源分配 | 16 线程同 key retry 只 allocation 一次；参数冲突 fail closed | 当前 xLLM 测试目标链接 Torch CPU；本核心不执行 tensor | 待真实 KV | PASS |
| G1 cancel-before-add | unknown cancel fence、迟到 admission、fence 独立容量与恢复压力 | N/A | 待 RPC race | PASS |
| G1 TTL/终态/释放 | transfer-start TTL、reservation TTL、过期 replay、cancel/destructor 精确 release | N/A | 待真实 stream/KV | PASS |
| G1 GenerationCommit | begin-transfer 门禁、handoff 幂等/冲突、首事件保留、output gate；并发 output 等待 commit 原子转换 | N/A | 待 P→D transfer | PASS |
| G1 D 真实 reservation/释放所有权 | 真实生产 TU 严格编译；commit 前 release、commit 后 cancel、finish 不 cancel 单测 | 当前生产目标链接 Torch CPU；无 tensor 数值变化 | 待真实 KV/credit/slot | PASS（CPU 编译与状态机） |
| G1 P RPC 歧义与首事件屏障 | Commit/Query/Cancel、响应数量异常和空 channel fail-closed；retry budget + dispatch margin 不超过 Service gap timeout，RPC controller 受本地 monotonic budget 截断；生产 TU 严格编译 | N/A | 待 brpc/RDMA 故障注入 | PASS（CPU 核心与编译）；集成故障矩阵待补 |
| G1 incarnation 绑定 | routing 字段号/roundtrip 协议测试；双仓外层 xLLM override build | N/A | 待进程重启竞态 | PASS |
| G1 Engine 成员租约 | PRESENT/MISSING/UNAVAILABLE 三态；15 秒 TTL 下限、reconcile 周期正负边界；KeepAlive 与 xservice client 生产 TU 严格编译 | N/A | 待真实 etcd 选举、分区、lease expiry 与 fleet restart 注入 | PASS（CPU 状态机与编译）；集群可用性待验证 |
| F83 Service 单 hold | 32 线程并发安装只成功一次；resolved 后下一 attempt 可安装；Request 直接持有不可复制 hold | N/A | 待真实请求 | PASS |
| F83 cleanup 容量 | record/byte backpressure、64 线程精确耗尽、RAII 与跨表 token 拒绝 | N/A | N/A | PASS |
| F83 收敛证明 | 全候选、confirmed 收窄、旧 attempt/incarnation、`QUERY_ABSENT`、hard-bound profile 门禁 | N/A | 待 Provider Query/Cancel | PASS |
| F83 后台 cleanup | bounded round-robin batch、公平轮转、已收敛 holder/record 移除；Scheduler Query→Cancel 精确终态路径生产编译 | N/A，纯控制协议 | 待真实 P/D 故障注入 | PASS（CPU 核心与生产编译）；loopback race 待补 |
| F83 Service 生产绑定 | Scheduler/HTTP/Request 生产目标 build/link；dispatch 前安装、incarnation 二次验证、P commit、D terminal、Cancel ACK、断连 detach、精确进程终止串行收敛路径代码审查 | N/A，纯控制协议 | 待真实 P/D | PASS（CPU 编译与核心状态机）；loopback race 待补 |
| F83 aggregated hold | 单 holder install/commit/terminal；Agent 64 路同 key、Cancel-before-create、Cancel/Submit 和 deadline/Submit 竞态；Agent terminal/malformed response parser | N/A，纯控制/HTTP loopback | 待真实 vLLM-Ascend abort/资源释放 | PASS（CPU 核心与生产接线） |
| 内存/未定义行为 | GCC 13 ASan+UBSan 定向运行 Engine 17 项与 Service hold 19 项 | N/A | N/A | PASS；Clang sanitizer runtime 未随 ARM64 镜像安装 |
| deadline 约束 reservation/调度 | optional wire、fake monotonic、Service 有界并发索引；D admission/reservation cap、P 三个边界和六类 Engine 调度路径生产 TU 以 `-Werror` 编译 | 公共测试目标使用 Torch CPU；无 tensor 数值变化 | 待真实 KV/transfer | PASS（CPU 核心与生产编译）；loopback 待补 |
| G1/G2 exact 首事件保留与恢复 | xLLM adapter/protocol/4 MiB/field 24；Service Query state、D/P incarnation、attempt、seq、payload 和 index fail-closed；7 项真实 brpc loopback 覆盖并发、timeout、D restart 和 live race | adapter 目标链接 Torch CPU；无 tensor 数值变化 | 待 P/D 数据面故障注入 | PASS（CPU loopback） |
| 双仓回归 | xLLM 默认七目标 103/103，另有 queue 14/14、protocol 14/14；Service pinned/override 均为 304/304；vLLM Agent/sidecar 60/60；xLLM 受影响生产 TU 严格编译；Service 三个生产二进制 build/link verify | queue 含 Torch CPU retained-storage/ownership 测试 | 本批不修改 KV/HBM 数据面，simulated HBM 不适用 | PASS |

## 完善情况

- 已完成：additive wire、Engine attempt CPU 状态机、Native REMOTE_PD 的 D 硬准入/
  reservation、P handoff、V2 RPC handler、ACK 歧义 Query、两侧首事件门禁、
  scheduler 所有权转移，Service incarnation 路由、单 hold/cleanup capacity CPU 核心，
  以及 REMOTE_PD dispatch、输出、取消、断连、后台有界 Query/Cancel 重试和精确进程
  终止的生产 hold 接线；vLLM `AGGREGATED` 生产 hold、Agent Query/Cancel/deadline 和
  cancel-before-create fence 已接线。
- 已知缺口/风险：lifecycle callback 仍在状态锁内执行，生产实现必须保持本地、
  有界、不可重入；Service 的 outcome-unknown hold 已进入 Native REMOTE_PD 主路径，
  后台 worker 已能周期性 Query/Cancel，但 cleanup worker 尚无多进程 fault loop 与 RPC
  结果 metrics；output event seq/reorder、gap watchdog、默认/显式 Native request
  deadline、P 首事件有界 timer、seq=0 exact Query 恢复和预算内新 attempt 已接线；
  attempt 替换仍缺完整 Scheduler + 两组真实 P/D loopback，cancel fence/tombstone TTL
  还须由最大消息寿命证明。
- 回滚与兼容：旧 `FirstGeneration` 和旧字段保留；新增字段/RPC 都是 additive。
  legacy 路径不会自动获得 V2 资源安全语义，开放开关前必须完成 capability 门禁。
- 性能、容量和观测证据：CPU 测试证明内存逻辑有固定 record/byte/fence 上限；尚无
  真实 workload 的 allocator 锁持有时间、冲突率、cleanup burst 或 1 万次故障数据。
- CPU 证据只证明链路、状态机和 host/Torch CPU 所有权，不替代 HBM。后续修改 KV
  分配、迁移、淘汰或内容时，必须增加 simulated HBM 的固定容量、block 地址/所有权、
  内容/checksum、OOM/碎片和故障回收测试；本批成员租约与 EngineState 协议修复不触碰
  KV/HBM 数据面，因此该层为不适用，而非已验证。
- 达到全 G1 CPU_VERIFIED 仍需完成：cleanup worker metrics 与
  reservation/deadline profile 配置化和 Gateway deadline 策略；补齐 CPU
  多进程端到端 cleanup/deadline/fault loop；接入 local hold，并确保所有开放 mode
  共用一套权威资源账本。
- 达到 VERIFIED 仍需完成：NPU KV/credit/slot 原子分配、P→D transfer、取消/超时/
  进程终止故障矩阵、1 万次故障门禁和容量/性能发布证据。
