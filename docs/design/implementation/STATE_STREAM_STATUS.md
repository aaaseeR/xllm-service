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
| xLLM Native | `REMOTE_PD/LAYERWISE_PUSH` | contract v1 strict Descriptor | CPU_VERIFIED（接收与过滤核心） | FULL/DELTA、master/snapshot fencing、Engine/Link TTL、incarnation 和 READY 路由门禁通过 |
| xLLM Native | legacy Descriptor-less | BEST_EFFORT | PARTIAL | 第一个有效 FULL 前保留滚动升级兼容；FULL 后不允许与 strict 单边混配 |
| vLLM-Ascend | `AGGREGATED` | contract v1 strict Descriptor | PARTIAL | Registry 核心可表达；Provider Agent 尚未发布真实 EngineState |

总体状态保持 PARTIAL：本批完成的是单一协议、权威成员与软状态分离、接收缓存和真实
调度消费链路；master 聚合/异步扇出、非 master 的权威 master incarnation 分发、
Engine/Agent 上报、观测失明迟滞和 Link 周期对账尚未闭环。

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
- 默认容量：4096 members、16384 links；state soft/hard TTL 为 3/10 秒，heartbeat
  hard TTL 与 link hard TTL 为 10 秒。无界集合、跨时钟 deadline 和软状态驱逐成员均被
  禁止。

## 需求与测试追踪

| Requirement ID | CPU test | Torch CPU test | NPU test | 结果 |
| --- | --- | --- | --- | --- |
| 单一 additive wire | xLLM `ProviderProtocolTest.StateBatchGoldenWireAndAgesAreStable`、critical field numbers | N/A，无 tensor 逻辑 | N/A | PASS，协议组 8/8 |
| master/FULL/DELTA fencing | `RequiresCurrentMasterFullBeforeScheduling`、旧 master/旧 snapshot/DELTA-before-FULL 负向路径 | N/A | 待多机切主 | PASS |
| membership/incarnation | `IncarnationReplacementCannotBeResurrectedByState`、Descriptor collision、权威 remove | N/A | 待 Engine self-fencing | PASS |
| TTL/clock | `UsesReceiverMonotonicAgeAndNeverRegressesState`、overflow/clock regression/missing age | N/A | 待跨机时钟故障注入 | PASS |
| LinkState | proof、role、connector、transfer、lifecycle、TTL 正负路径；route selector READY list | N/A | 待真实 handshake | PASS |
| 并发和容量 | 16 线程 membership upsert、member/link/batch 上限、非法配置 | N/A | N/A | PASS |
| 生产接入 | RPC descriptor 复用 shared StateBatch；Scheduler/InstanceMgr/route 编译链接 | N/A | 待多副本 loopback | PASS |

本批验证：xllm-service Debug 生产编译通过，全量 CPU 测试 246/246；EngineRegistry
8/8；xLLM Provider 协议 8/8。没有 tensor 数值逻辑，因此本功能当前无伪造的 Torch CPU
测试项。

## 完善情况

- 已完成：公共协议；Registry/state/link 有界缓存；master/snapshot/incarnation fencing；
  FULL/DELTA 原子验证；本地 monotonic TTL；真实 RPC receiver；全部现有调度入口统一
  consumption；严格 P/D Link READY 门禁；滚动升级前置兼容。
- 已知缺口/风险：master 尚未从 heartbeat/Agent 聚合 EngineState 并向每个 Service
  维护单在途、有界 latest-map publisher；非 master 尚未从 Registry 获得当前 master
  incarnation，因此当前 receiver 只在本地 master view 可用；现有 etcd master value 必须
  保持纯地址，后续 identity 分发不能破坏 Engine watcher；Link PENDING/DEGRADED 周期
  reconciler 和状态分发指标尚未实现。
- 回滚与兼容：proto 和 RPC 均 additive；第一个当前 FULL 到达前旧 runtime-state 路由
  行为保持不变。FULL 生效后 strict 路由 fail closed，软状态超时只停止新分配，不执行
  deregister、unlink 或在飞资源清理。
- 性能、容量和观测证据：CPU 单测证明有界内存与锁安全，不代表生产扇出吞吐；publisher
  完成后需补慢订阅者、断连合并、FULL 恢复和高频状态压力测试。
- 达到 CPU_VERIFIED 仍需完成：master identity 兼容分发、Engine/Agent state ingress、
  master 聚合与异步 publisher、服务成员安全枚举、切主/慢订阅者 loopback、Link 周期对账、
  `STATE_BLIND/REGISTRY_BLIND` 迟滞和 readiness 接入。
- 达到 VERIFIED 仍需完成：上述 CPU 门禁全部通过，并在 NPU 多 P/D、多 Service、切主、
  heartbeat 丢失、陈旧状态和 link 故障矩阵中验证。
