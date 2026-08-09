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

# G3 Engine Registry 与 State Stream

## 基本信息

- Owner：xLLM Service V2
- 状态：PARTIAL
- 关联设计/Requirement ID：G3、F64、F65、F67、F78、D13、D15-D18、D34、D41、D52
- 最近验证基线：xLLM `8c8d68a3`；xllm-service `4dfc935`
- 验证环境和日期：xllm-dev-sandbox，Ubuntu 24.04 ARM64，2026-08-09

## 支持范围

| Provider | Mode | Model/Profile | 支持状态 | 限制与证据 |
| --- | --- | --- | --- | --- |
| xLLM Native | `REMOTE_PD/LAYERWISE_PUSH` | contract v1 strict Descriptor | CPU_VERIFIED（Native 生产 + Service 消费核心） | Native P/D 注册真实 Descriptor，heartbeat 发布完整 per-DP EngineState；FULL/DELTA、fencing、TTL、Link READY 门禁通过 |
| xLLM Native | legacy Descriptor-less | BEST_EFFORT | UNSUPPORTED | V2 路由统一 fail closed；完成相应 V2 Descriptor/EngineState producer 后才能加入候选 |
| vLLM-Ascend | `AGGREGATED` | contract v1 strict Descriptor | CPU_VERIFIED / NPU_PENDING | Agent 发布单调、incarnation/profile/model 对齐的 per-DP EngineState；真实指标与 NPU deep health 待验证 |

总体状态保持 PARTIAL：单一协议、xLLM Native 状态生产、权威成员与软状态分离、
master 聚合与有界异步扇出、权威 master incarnation 分发、接收缓存和真实调度消费
链路已经闭环；`NORMAL/STATE_BLIND/REGISTRY_BLIND` 迟滞、统一候选过滤、实际直连证据、
永久 listener 和独立 readiness 已完成 CPU 闭环；Link 周期对账的 Service 侧已经闭环，
vLLM Agent 状态生产已完成 CPU 闭环；真实 etcd 切主注入、vLLM-Ascend 指标校准和
NPU handshake 故障矩阵待最终环境验证。

## 实现

- xLLM `xllm/proto/provider.proto` 是 `StateBatch`、`EngineState`、`LinkState` 和
  `ProviderEngineKey` 的唯一 wire 定义。所有 publish age 都保留 optional presence；
  接收方只用本地 monotonic elapsed 累加，不比较跨主机绝对时钟。
- `xllm_service/provider/engine_registry.*` 分离权威 Registry membership 与可丢失的
  State cache。状态更新不能创建或复活成员；同 engine UID 的新 incarnation 原子替换
  旧成员并清除旧状态/link，同时使当前 FULL 失效。
- `ObservationController` 独立维护 NORMAL、STATE_BLIND、REGISTRY_BLIND。State 失明
  使用同一 stale ratio 单位的 enter/exit 阈值与时间 hold；Registry 不可见立即进入短
  宽限。冷副本未接受首个 FULL 时始终 fail closed，恢复 STATE_BLIND 必须接受当前
  master FULL 并持续满足 exit hold。
- `EngineRegistry::IsSchedulable` 在 NORMAL 使用 state/heartbeat/link hard TTL；
  STATE_BLIND 宽限内只使用未被直连失败推翻的最后良好状态，宽限后要求 TTL 内的直接
  成功证据；REGISTRY_BLIND 只允许短宽限且失败证据立即生效。证据表与成员同生命周期、
  受 `max_members` 硬上限约束，不刷新负载年龄。
- Receiver 只接受 Registry 当前 master incarnation 的单调 snapshot；切主后必须先
  FULL，DELTA 才可应用。FULL 必须精确覆盖当前成员，重复项、未知 enum、容量越界、
  Descriptor/compatibility proof 不一致全部 fail closed。
- `InstanceMgr` 注册、incarnation 替换和最终移除维护 Registry membership；RPC
  `PushEngineState(StateBatch)` 经 `Scheduler` 写入缓存。RR、Provider 选择、静态 peer、
  CAR/SLO 指标、request bind、dispatch/retry 前复验和 readiness 均复用同一个
  `IsSchedulable` 事实。严格 Remote PD 额外要求两端可调度且 incarnation-scoped
  `LinkState=READY`。
- Registry 可见性由启动全前缀读取及一秒周期读取独立维护；master key 缺失或变化只
  更新 State Stream fencing identity，不再伪造成 REGISTRY_BLIND。切主后旧合法快照
  可按实际新鲜度继续保守路由，但新 master DELTA 必须等待其 FULL。
- Registry watch 的成员事件按 etcd revision 串行应用，旧 revision 不能回滚新
  incarnation。权威 DELETE 在下一次选择前关闭 dispatch gate 并移除成员；健康探测、
  heartbeat 和同 incarnation 的后续 PUT 均不能复活成员。revision/tombstone 历史默认
  上限 8192，容量耗尽时进入 REGISTRY_BLIND fail closed，不驱逐历史以换取错误恢复。
- 建链候选只按不可变 Descriptor 兼容矩阵生成，不能反向依赖 READY；路由候选才消费
  LinkState，避免“未 READY 所以不建链”的循环依赖。
- heartbeat 可选携带 `EngineState`。master 校验成员 identity 后聚合状态；成员变化使
  当前 FULL 失效，并在精确覆盖恢复后生成新的权威 FULL。旧客户端不携带该字段时保持
  wire 兼容。
- xLLM Native 仅为 V2 已开放的 P/D `LAYERWISE_PUSH` 模式发布 strict Descriptor；
  DEFAULT/MIX/PULL 等模式继续走 descriptor-less 兼容路径，不伪造 V2 能力。Descriptor
  从已加载模型/tokenizer/renderer、实际 worker 数派生的 TP/DP/CP/KV-split 拓扑、
  BlockManager、scheduler 和 Connector 配置构造，各兼容性契约使用带 domain 的 SHA-256
  digest。缺字段或运行时拓扑不一致时 fail closed。
- xLLM heartbeat 从 `BlockManagerPool` 读取每个 DP 的真实 free/used block，发布单调
  `state_seq`、incarnation/profile/model identity、admission credit 和安全有界的 KV
  使用率。snapshot 不完整时不消耗序号；producer 支持并发调用。EngineState 是
  `HeartbeatRequest` 的 additive field 7，旧 Service/Engine wire 仍兼容。
- vLLM strict Agent 从带 DP label 的 Prometheus 样本生成 per-DP running、waiting、
  deferred、KV ratio 和 admission credit，缺任一配置 DP 时只发布 `PARTIAL`；heartbeat
  携带单调 `state_seq` 和 Descriptor 的 incarnation/profile/model identity。legacy
  sidecar 仍只发布 contract-v0 aggregate metrics，不能伪装 strict State producer。
- `StateStreamOutbox` 为每个订阅者维护一个在途请求和有界 latest-map。新订阅者、发送
  失败、队列溢出和周期检查均强制先发最新 FULL；成功后才继续 DELTA。发布 age 只按
  本地 monotonic elapsed 重基，回拨和溢出均 fail closed。
- master 选举在同一 etcd lease/transaction 内同时写入旧的纯地址 key 和独立 V2
  incarnation key。所有副本同时监听两者；发布线程每轮复验当前地址和 incarnation，
  防止旧 master 继续扇出。
- `StateStreamClient` 先启动一轮所有有效 BRPC，再分别等待结果；单个慢订阅者不会阻止
  其他调用发起，且每个调用都有独立超时。
- 严格 V2 实例先作为权威成员安装，再由 master `LinkReconciler` 对兼容 P/D
  incarnation 做周期差集和幂等 `LinkInstance`。新 pair 先发布 PENDING，单 pair 成功转
  READY、失败转 DEGRADED 并按 1/2/4/.../10 秒有界退避；READY 每 3 秒复验。一个坏
  peer 不回滚成员或其他健康 pair，切主后新 master 强制重新检查。
- master 降级同时关闭 Scheduler、InstanceMgr 和 GlobalKVCacheMgr 的 master 身份，并
  恢复 follower 的 etcd 状态监听；旧 master 停止 State Stream 与 Link 对账发布。
- Native 异步 dispatch、vLLM HTTP 响应/流终态、attempt Query/Cancel 和合法 heartbeat
  都向 Registry 写入 incarnation-scoped 直连成败证据。STATE_BLIND 下另有每轮最多 16
  个、单次 200ms 的轮转轻量 `/health` 探测；用户请求不能作为 NOT_READY 后的探测旁路。
- HTTP/RPC listener 从进程启动到 drain 完成保持运行。`/livez` 恒定表达进程存活；
  `/readyz` 和所有推理入口消费同一 `ReadinessController` 快照。冷启动缺 FULL、blind
  宽限耗尽、无可用直接证据和 draining 会立即 NOT_READY；恢复必须连续稳定 3 秒。
  NORMAL 下瞬时容量不足不摘流，竞争窗口内的新请求稳定返回 HTTP 503
  `SERVICE_NOT_READY`，在飞请求与 heartbeat/health 继续处理。
- 默认容量：4096 members、16384 links；state soft/hard TTL 为 3/10 秒，heartbeat
  hard TTL 与 link hard TTL 为 10 秒；publisher 最多 256 个订阅者、50ms publish tick、
  1 秒周期 FULL、200ms 单 RPC 超时。STATE_BLIND enter/exit ratio 为 0.5/0.2、hold 为
  1/3 秒、宽限 10 秒、直接证据 TTL 3 秒；REGISTRY_BLIND 宽限 3 秒；readiness 检查
  200ms、恢复 hold 3 秒。无界集合、跨时钟 deadline 和软状态驱逐成员均被禁止。

## 需求与测试追踪

| Requirement ID | CPU test | Torch CPU test | NPU test | 结果 |
| --- | --- | --- | --- | --- |
| 单一 additive wire | xLLM `ProviderProtocolTest.StateBatchGoldenWireAndAgesAreStable`、critical field numbers | N/A，无 tensor 逻辑 | N/A | PASS，协议组 8/8 |
| master/FULL/DELTA fencing | `RequiresCurrentMasterFullBeforeScheduling`、旧 master/旧 snapshot/DELTA-before-FULL 负向路径 | N/A | 待多机切主 | PASS |
| membership/incarnation | `IncarnationReplacementCannotBeResurrectedByState`、Descriptor collision、权威 remove | N/A | 待 Engine self-fencing | PASS |
| 权威 DELETE fencing | InstanceMgr 生产路径编译链接；Registry remove/incarnation replacement/state 不能复活旧成员；revision/tombstone 代码检查 | N/A | 待真实 etcd DELETE/revoke、乱序 PUT/DELETE 与 lease 到期注入 | PASS（CPU 核心）；真实 etcd 注入待补 |
| TTL/clock | `UsesReceiverMonotonicAgeAndNeverRegressesState`、overflow/clock regression/missing age | N/A | 待跨机时钟故障注入 | PASS |
| LinkState | proof、role、connector、transfer、lifecycle、TTL 正负路径；PENDING/READY/DEGRADED、退避、周期复验、incarnation replacement、单在途并发；route selector READY list | N/A | 待真实 NPU handshake | PASS；reconciler 5/5 |
| 并发和容量 | 16 线程 membership upsert、64 线程 outbox update、member/link/batch/subscriber 上限、非法配置 | N/A | N/A | PASS；关键并发用例连续 100 轮 |
| master 聚合/发布 | 权威 FULL 精确覆盖、publish-age 重基、单在途、latest-map 合并、失败/溢出恢复 | N/A | 待多机压力 | PASS；Registry 10/10、Outbox 6/6 |
| master identity | 旧地址 key 保持不变、独立 incarnation key、切主后 receiver 先等 FULL | N/A | 待真实 etcd 切主 | PASS（代码与协议）；真实 etcd 故障注入待补 |
| 观测失明与恢复 | `ObservationControllerTest` 8 个阈值/hold/宽限/冷启动/时钟负向用例；`EngineRegistryTest` 覆盖首 FULL、master change、STATE_BLIND 直接证据、REGISTRY_BLIND 失败与到期、FULL 恢复和 Link 共用门禁 | N/A，无 tensor 逻辑 | 待多机断流/断 Registry | PASS；Controller 8/8、Registry 14/14 |
| listener/readiness | `ReadinessControllerTest` 覆盖非法配置、冷启动、恢复 hold、NORMAL 防抖、STATE/REGISTRY blind、drain 和时钟回拨；`HealthResponseTest` 精确验证 live/ready/503 JSON | N/A，无 tensor 逻辑 | 待 LB 摘流与多副本 drain | PASS；Controller 6/6、HTTP 映射 3/3 |
| 生产接入 | RPC descriptor 复用 shared StateBatch；Scheduler/InstanceMgr/route 编译链接；BRPC loopback 成功、8 路并发、超时和非法输入 | N/A | 待多副本 loopback | PASS；client 4/4 |
| xLLM Native 生产 | Descriptor 确定性/非法输入、P/D mode、per-DP 完整/空容量、缺失 capability、snapshot 序号不消耗、32 线程唯一序号；heartbeat field 7 | N/A，无 tensor 数值逻辑 | 待真实 CANN/SOC 版本、NPU block 账本与 P/D heartbeat | PASS；Native 6/6、协议 8/8，并发套件连续 100 轮 |

本批验证：xllm-service Debug 三个生产服务目标编译、动态链接通过，全量 CPU 测试
293/293；ObservationController 8/8；ReadinessController 6/6；HealthResponse 3/3；
EngineRegistry 14/14；StateStreamOutbox 6/6；
StateStreamClient BRPC loopback 4/4；
LinkReconciler 5/5；xLLM Native producer 6/6、Provider 协议 8/8。State Stream、
Link 和 Native producer 的关键并发用例连续 100 轮通过。xLLM 的
`native_provider_runtime.cpp`、`xservice_client.cpp`、`llm_master.cpp` 和
`vlm_master.cpp` 使用真实 Torch CPU/BRPC 编译参数通过。完整 xLLM runtime 目标仍被
既有 `process_group.cpp` 的 CPU `ProcessGroupImpl` 不完整类型错误阻断，该文件不在本批
改动中。没有 tensor 数值逻辑，因此本功能当前无伪造的 Torch CPU 测试项。

## 完善情况

- 已完成：公共协议；Registry/state/link 有界缓存；master/snapshot/incarnation fencing；
  FULL/DELTA 原子验证；本地 monotonic TTL；真实 RPC receiver；全部现有调度入口统一
  consumption；严格 P/D Link READY 门禁；heartbeat ingress；master 权威 FULL 构建；
  单在途、有界 latest-map publisher；兼容的 master identity 分发；BRPC 并发扇出；滚动
  升级前置兼容；Link 周期差集、独立状态机、有界重试与 State Stream 发布；master
  降级后停止旧主发布和恢复 follower 监听；xLLM Native strict Descriptor 与真实
  per-DP heartbeat EngineState 生产；三态观测迟滞、直接证据缓存、主身份与 Registry
  可见性解耦、冷启动/descriptor-less fail closed 和 blind-mode Link 共用门禁；真实
  dispatch/HTTP/Query/heartbeat/轻量 probe 直连证据；权威 DELETE revision fencing 与
  同 incarnation 防复活；永久 listener、`/livez`、`/readyz`、稳定 503 和 readiness
  recovery hold。
- 已知缺口/风险：vLLM Agent EngineState 已在 CPU producer/heartbeat 测试中闭环，但
  尚未与真实 vLLM-Ascend 指标和 NPU deep-health 对账；xLLM 的 NPU 硬件 runtime 版本
  目前只能发布平台与编译期 Torch 版本，仍需接入实际 CANN/驱动
  和 resolved cache dtype/SOC 身份。Link 状态已由真实 `LinkInstance` 结果生成，但尚未
  做 NPU 多 P/D 故障矩阵。master identity 已保持旧 etcd 地址 value 不变，但仍需真实
  etcd 故障注入覆盖两 key 事件乱序、成员 PUT/DELETE/revoke 与 lease 到期；当前 CPU
  只能验证 revision/tombstone 和 fail-closed 核心，不能替代真实 watch 行为。部署必须为
  P/D 提供一致、不可变的 `model_id`，避免本地模型路径被当作 revision 时产生保守的
  不兼容。
- 回滚与兼容：proto 和 RPC 均 additive；V2 首版不保留第一个 FULL 前的旧 runtime-state
  绕过，缺 Descriptor、缺首个 FULL 或不完整快照均 fail closed。软状态超时只停止新
  分配，不执行 deregister、unlink 或在飞资源清理。
- 性能、容量和观测证据：CPU 单测已覆盖有界内存、锁安全、慢订阅者超时、断连合并、
  FULL 恢复与并发调用，不代表生产扇出吞吐；仍需补长时间高频状态 soak 和分发指标。
- 达到完整 G3 CPU_VERIFIED 仍需完成：真实 etcd 切主和成员事件故障注入；xLLM 完整
  CPU runtime 链接还需先修复上述既有 ProcessGroup 编译
  基线。
- 达到 VERIFIED 仍需完成：上述 CPU 门禁全部通过，并在 NPU 多 P/D、多 Service、切主、
  heartbeat 丢失、陈旧状态和 link 故障矩阵中验证。
