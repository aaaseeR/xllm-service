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

# V2-B6 至 V2-B10 开发门定义与状态

## 结论

B6-B10 是完整 V2 首个交付版本的后半程开发门，不是独立产品版本。每个门只有在
代码、CPU/Torch CPU、涉及 KV 时的 simulated HBM、开发状态文档和受影响回归同时
通过后才能标记 `CPU_VERIFIED`。最终 `VERIFIED` 还要求对应 NPU、真实 P/D/etcd、
故障注入和容量门禁通过；`CPU_VERIFIED` 不得改写为生产硬件已验证。

当前总状态：`IN_PROGRESS`。B6 仓库内代码与 CPU 门已完成，平台 branch protection
和 NPU 保持 pending；B7 正在实现。B7-B10 的范围和 DoD 已冻结，禁止用后续文档
静默缩小。

## 门级范围

| 开发门 | 权威设计范围 | 必须交付的生产能力 | 当前状态 |
| --- | --- | --- | --- |
| V2-B6 | 00 §4、02 G-1 测试底座、11 硬件感知边界 | 真正生效的 pin/CPU 合入入口；device-neutral KV/HBM 资源契约；xLLM simulated HBM test backend；容量、block 地址/所有权、内容/checksum、OOM/碎片、并发与故障回收门禁 | CPU_VERIFIED / PLATFORM_AND_NPU_PENDING；见 [SIMULATED_HBM_STATUS.md](./SIMULATED_HBM_STATUS.md) |
| V2-B7 | 08 V2-K0 | Engine block 真源产生 KV store/remove/clear 事件；独立 KV lane；incarnation/cache epoch/event sequence；有界 shadow index；gap 后分页 snapshot 恢复和对账；只观测不影响路由 | PLANNED |
| V2-B8 | 08 V2-K1 | 统一候选过滤后的 HBM P/D Prefix + load 联合评分；least-load/top-prefix shortlist；pending-work、survival/residence credit；按 bucket 灰度；UNKNOWN 或恢复中自动 load-only | PLANNED |
| V2-B9 | 09 | 每 Service/ModelPool 有界队列；request/token/byte/tenant 硬上限；priority band、tenant flow 公平和 flow 内 FCFS/EDF；整池饱和门；按 capability 选择 REMOTE_PD、LOCAL_PREFILL_DECODE、PREFILL_ONLY | PLANNED |
| V2-B10 | 08 V2-K2、完整 09/11、00 §2.1 | 多模型隔离；优先级/公平与 KV 选择联合；可观测低层 shadow credit；双 Provider 最终 conformance；容量/故障/回退矩阵；完整支持矩阵和 V2 代码完成口径 | PLANNED |

## B6 Definition of Done

1. pin 守卫由仓库脚本提供唯一入口，至少校验 gitlink 与工作树 revision 一致、
   `provider.proto` 和 `observability.proto` 存在；GitHub 镜像对 `main/service_dev`
   push/PR 调用该入口，内部 JD Coding 的受保护分支也必须把同一入口设为必需检查。
2. simulated HBM 位于 xLLM Engine/backend 测试边界，不进入 Service 生产依赖；Service
   只通过 fake Provider 和统一资源语义验证 reservation/hold/transfer/fencing/reclaim。
3. 模拟固定容量与稳定 block/page 地址，禁止重复分配和重复释放；所有 block 明确绑定
   `engine/incarnation/request/attempt/dp/rank` owner。
4. 支持写入、读取和 checksum reference；Prefill 写入、Decode/transfer 读取、迁移、
   淘汰和回收后内容必须可核对，不能只验证计数器。
5. 覆盖 OOM、可控碎片、transfer pending/timeout/cancel/result-unknown、owner fencing、
   进程替换、延迟释放，以及并发 allocate/free/transfer 的容量守恒和线性化结果。
6. xLLM B6 目标、七个公共 CPU contract 目标、Service pinned/override、Agent 测试和
   三个生产服务二进制全部通过；开发状态记录精确命令、commit 和剩余 NPU 证据。

## B7 Definition of Done

1. KVEvent 由 BlockManager/cache store/remove/clear 真源产生，包含 engine/incarnation、
   model/KV namespace、cache epoch、event sequence、block hash/range、tier 和 reason。
2. Engine 的 KV lane 与 EngineState lane 隔离；事件队列和字节数有硬上限，溢出只把
   该 Engine shadow index 置 UNKNOWN 并请求 snapshot，不阻塞 scheduler/heartbeat。
3. Service 对重复事件幂等、乱序/gap fail closed；新 incarnation 或 CLEARED epoch
   立即失效旧位置。FULL/snapshot 完成前 KV 只观测，不能参与 hard routing。
4. 分页 snapshot 受 entry/byte/page/time 上限约束，恢复期间增量事件可重放并在确定
   cutover 后对账；Service/master 重启均能恢复，不使用 etcd 作为 KV 事件总线。
5. CPU 与 simulated HBM 覆盖 store/remove/evict/clear、丢失/重复/乱序、队列溢出、
   Engine/master/Service 重启和 snapshot 竞态；shadow 预测与实际命中结果可对账。

## B8 Definition of Done

1. 所有策略共用同一 `SelectCandidates` 硬过滤；KV-aware 只能在过滤后的候选上排序，
   不得恢复不兼容、非 READY、stale 或未建链 Engine。
2. hash 每请求只计算一次；同时保留 least-load 和 top-prefix shortlist，并计入本 Service
   尚未出现在 State Stream 中的 pending work，近似同分用 request hash 稳定打散。
3. P/D predicted hit、retainable/residence/survival、transfer bytes、Decode headroom 和
   load 使用有量纲的有界成本；UNKNOWN/OOD/snapshot recovery 自动回到 load-only。
4. shadow 阶段记录 predicted/actual hit、有效 Prefill token、跳过 transfer bytes、
   准入冲突和回退原因；达到门禁前不得成为默认策略。
5. fixed-prefix/no-prefix、热点、无命中、索引失效、状态陈旧和多 Service 冲突在 CPU/
   simulated HBM 重放中结果确定、容量守恒且成功率不低于 load-only。

## B9 Definition of Done

1. 队列只保存尚未 dispatch 的规范化请求；一旦产生 ExecutionPlan 即进入现有
   execution hold/attempt/deadline 责任域，不允许跨状态重复计量或丢失取消。
2. Service、ModelPool 和 tenant 的 request/token/byte 上限在入队前原子预留；失败返回
   稳定容量或 deadline reason，队列满本身不改变 readiness。
3. priority band 间严格优先但配置有 starvation bound；band 内 tenant/flow 使用
   work-conserving 公平轮转，flow 内支持 FCFS/EDF，取消/超时立即精确释放配额。
4. earliest-dispatch 使用可证明上界；整池饱和暂停 dispatch 并探测恢复，不产生
   retry storm。负载与 KV 选择只影响出队后的候选排序，不破坏公平账本。
5. REMOTE_PD、LOCAL_PREFILL_DECODE、PREFILL_ONLY 均按公共 capability Resolver 和
   mode-specific GenerationCommit 屏障执行；不支持 profile 稳定拒绝。
6. CPU 测试覆盖容量边界、并发、取消、deadline、tenant 隔离、无饥饿、崩溃上界和
   三种 mode；涉及 KV/credit 的路径同时通过 simulated HBM。

## B10 Definition of Done

1. 所有索引、队列、pending work、负缓存、credit 和观测按 model revision + provider
   profile + KV namespace 隔离；多模型之间不能互相命中、占用配额或污染校准。
2. K2 将 priority/tenant fairness 与 KV/load 选择组合，但 correctness 路径仍是单一
   Resolver + Engine 原子 admission；低层 tier 只发布 shadow credit，不暗示 V2.5
   Store 数据路径已经交付。
3. xLLM Native REMOTE_PD/LOCAL/PREFILL_ONLY 与 vLLM-Ascend AGGREGATED 的支持矩阵、
   稳定拒绝组合、API/SSE/cancel/deadline/fencing/drain 均有 conformance 证据；跨
   Provider P/D 保持关闭并有负向测试。
4. 100% CPU/Torch CPU/simulated HBM 需求追踪闭环；全量回归、sanitizer/压力/确定性
   review 无 blocker；代码、设计、状态、发布与回滚说明一致。
5. NPU/CANN、真实 etcd/Link、Agent 同命、原始端口隔离、长时 soak 和故障矩阵分别
   记录 `NPU_PENDING` 证据与命令。完成这些硬件门后，V2 才从“代码完成”升级为
   `VERIFIED`；禁止用 B10 CPU 完成替代真实硬件结论。

## 验证与提交纪律

- 每个 B 门至少一个独立提交；xLLM 与 xllm-service 分别提交并推送各自
  `service_dev`，Service gitlink 只指向已推送的 xLLM 提交。
- 每个状态文件记录 pinned/override 两种 Service 结果、xLLM 目标结果、simulated HBM
  结果和生产二进制链接结果；失败、skip 或未运行均不能写 PASS。
- 每完成一个门进行至少三轮独立 review：资源/并发，协议/兼容，测试/文档；发现问题
  先增加永久回归再修复。
