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
- 关联设计/Requirement ID：G3、F64、F65、F67、F78、D13、D15-D17、D41、D52
- 最近验证基线：xLLM 与 xllm-service 本状态文档所在提交
- 验证环境和日期：xllm-dev-sandbox，Ubuntu 24.04 ARM64，2026-08-09

## 支持范围

| Provider | Mode | Model/Profile | 支持状态 | 限制与证据 |
| --- | --- | --- | --- | --- |
| xLLM Native | `REMOTE_PD/LAYERWISE_PUSH` | contract v1 strict Descriptor | CPU_VERIFIED（Service 聚合/发布/接收核心） | FULL/DELTA、master/snapshot fencing、有界异步扇出、Engine/Link TTL、incarnation 和 READY 路由门禁通过 |
| xLLM Native | legacy Descriptor-less | BEST_EFFORT | PARTIAL | 第一个有效 FULL 前保留滚动升级兼容；FULL 后不允许与 strict 单边混配 |
| vLLM-Ascend | `AGGREGATED` | contract v1 strict Descriptor | PARTIAL | Registry 核心可表达；Provider Agent 尚未发布真实 EngineState |

总体状态保持 PARTIAL：单一协议、权威成员与软状态分离、master 聚合与有界异步扇出、
权威 master incarnation 分发、接收缓存和真实调度消费链路已经闭环；Engine/Agent 真实
状态生成和观测失明迟滞尚未闭环；Link 周期对账的 Service 侧已经闭环，真实 NPU
handshake 故障矩阵待最终环境验证。

## 实现

- xLLM `xllm/proto/provider.proto` 是 `StateBatch`、`EngineState`、`LinkState` 和
  `ProviderEngineKey` 的唯一 wire 定义。所有 publish age 都保留 optional presence；
  接收方只用本地 monotonic elapsed 累加，不比较跨主机绝对时钟。
- `xllm_service/provider/engine_registry.*` 分离权威 Registry membership 与可丢失的
  State cache。状态更新不能创建或复活成员；同 engine UID 的新 incarnation 原子替换
  旧成员并清除旧状态/link，同时使当前 FULL 失效。
- Receiver 只接受 Registry 当前 master incarnation 的单调 snapshot；切主后必须先
  FULL，DELTA 才可应用。FULL 必须精确覆盖当前成员，重复项、未知 enum、容量越界、
  Descriptor/compatibility proof 不一致全部 fail closed。
- `InstanceMgr` 注册和权威删除维护 Registry membership；RPC
  `PushEngineState(StateBatch)` 经 `Scheduler` 写入缓存。RR、Provider 选择、静态 peer、
  CAR/SLO 指标、request bind、dispatch/retry 前复验和 readiness 均复用同一个
  `IsSchedulable` 事实。严格 Remote PD 额外要求两端可调度且 incarnation-scoped
  `LinkState=READY`。
- 建链候选只按不可变 Descriptor 兼容矩阵生成，不能反向依赖 READY；路由候选才消费
  LinkState，避免“未 READY 所以不建链”的循环依赖。
- heartbeat 可选携带 `EngineState`。master 校验成员 identity 后聚合状态；成员变化使
  当前 FULL 失效，并在精确覆盖恢复后生成新的权威 FULL。旧客户端不携带该字段时保持
  wire 兼容。
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
- 默认容量：4096 members、16384 links；state soft/hard TTL 为 3/10 秒，heartbeat
  hard TTL 与 link hard TTL 为 10 秒；publisher 最多 256 个订阅者、50ms publish tick、
  1 秒周期 FULL、200ms 单 RPC 超时。无界集合、跨时钟 deadline 和软状态驱逐成员均被
  禁止。

## 需求与测试追踪

| Requirement ID | CPU test | Torch CPU test | NPU test | 结果 |
| --- | --- | --- | --- | --- |
| 单一 additive wire | xLLM `ProviderProtocolTest.StateBatchGoldenWireAndAgesAreStable`、critical field numbers | N/A，无 tensor 逻辑 | N/A | PASS，协议组 8/8 |
| master/FULL/DELTA fencing | `RequiresCurrentMasterFullBeforeScheduling`、旧 master/旧 snapshot/DELTA-before-FULL 负向路径 | N/A | 待多机切主 | PASS |
| membership/incarnation | `IncarnationReplacementCannotBeResurrectedByState`、Descriptor collision、权威 remove | N/A | 待 Engine self-fencing | PASS |
| TTL/clock | `UsesReceiverMonotonicAgeAndNeverRegressesState`、overflow/clock regression/missing age | N/A | 待跨机时钟故障注入 | PASS |
| LinkState | proof、role、connector、transfer、lifecycle、TTL 正负路径；PENDING/READY/DEGRADED、退避、周期复验、incarnation replacement、单在途并发；route selector READY list | N/A | 待真实 NPU handshake | PASS；reconciler 5/5 |
| 并发和容量 | 16 线程 membership upsert、64 线程 outbox update、member/link/batch/subscriber 上限、非法配置 | N/A | N/A | PASS；关键并发用例连续 100 轮 |
| master 聚合/发布 | 权威 FULL 精确覆盖、publish-age 重基、单在途、latest-map 合并、失败/溢出恢复 | N/A | 待多机压力 | PASS；Registry 10/10、Outbox 6/6 |
| master identity | 旧地址 key 保持不变、独立 incarnation key、切主后 receiver 先等 FULL | N/A | 待真实 etcd 切主 | PASS（代码与协议）；真实 etcd 故障注入待补 |
| 生产接入 | RPC descriptor 复用 shared StateBatch；Scheduler/InstanceMgr/route 编译链接；BRPC loopback 成功、8 路并发、超时和非法输入 | N/A | 待多副本 loopback | PASS；client 4/4 |

本批验证：xllm-service Debug 三个生产服务目标编译通过，全量 CPU 测试 263/263；
EngineRegistry 10/10；StateStreamOutbox 6/6；StateStreamClient BRPC loopback 4/4；
LinkReconciler 5/5；xLLM Provider 协议 8/8。State Stream 与 Link 的并发、退避和
incarnation 关键用例连续 100 轮通过。没有 tensor 数值逻辑，因此本功能当前无伪造的
Torch CPU 测试项。

## 完善情况

- 已完成：公共协议；Registry/state/link 有界缓存；master/snapshot/incarnation fencing；
  FULL/DELTA 原子验证；本地 monotonic TTL；真实 RPC receiver；全部现有调度入口统一
  consumption；严格 P/D Link READY 门禁；heartbeat ingress；master 权威 FULL 构建；
  单在途、有界 latest-map publisher；兼容的 master identity 分发；BRPC 并发扇出；滚动
  升级前置兼容；Link 周期差集、独立状态机、有界重试与 State Stream 发布；master
  降级后停止旧主发布和恢复 follower 监听。
- 已知缺口/风险：xLLM Native 与 vLLM Agent 尚未在真实 heartbeat 中生成完整
  EngineState，所以当前 ingress 需要调用端补齐后才能形成首个生产 FULL；Link 状态已
  由真实 `LinkInstance` 结果生成，但尚未做 NPU 多 P/D 故障矩阵；状态分发指标、
  `STATE_BLIND/REGISTRY_BLIND` 迟滞尚未实现。master identity 已保持旧 etcd 地址 value
  不变，但仍需真实 etcd 故障注入覆盖两 key 事件乱序与 lease 到期。
- 回滚与兼容：proto 和 RPC 均 additive；第一个当前 FULL 到达前旧 runtime-state 路由
  行为保持不变。FULL 生效后 strict 路由 fail closed，软状态超时只停止新分配，不执行
  deregister、unlink 或在飞资源清理。
- 性能、容量和观测证据：CPU 单测已覆盖有界内存、锁安全、慢订阅者超时、断连合并、
  FULL 恢复与并发调用，不代表生产扇出吞吐；仍需补长时间高频状态 soak 和分发指标。
- 达到完整 G3 CPU_VERIFIED 仍需完成：xLLM/Agent EngineState 生产、真实 etcd 切主故障
  注入、`STATE_BLIND/REGISTRY_BLIND` 迟滞和 readiness 接入。
- 达到 VERIFIED 仍需完成：上述 CPU 门禁全部通过，并在 NPU 多 P/D、多 Service、切主、
  heartbeat 丢失、陈旧状态和 link 故障矩阵中验证。
