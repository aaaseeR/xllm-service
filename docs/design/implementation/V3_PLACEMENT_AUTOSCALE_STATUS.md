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

# V3 Placement/Autoscale 开发状态

## 基本信息

- Owner：xLLM Service
- 状态：IN_PROGRESS；V3-P0/P1 已 `CPU_VERIFIED`，P2 desired store 与 P3
  reconcile/fake actuator 核心已通过 CPU 测试但完整 operation 持久化和集成仍为
  `PARTIAL`，P4-P5 尚在开发，P6 为 `NPU_AND_CLUSTER_PENDING`
- 关联设计：[V3 Placement 与 Autoscale](../14_XLLM_SERVICE_V3_PLACEMENT_AUTOSCALE_DESIGN.md)
- 最近验证：`service_dev` 开发分支，ARM64 Linux Debug 沙箱，2026-08-10

## 当前支持范围

| 能力 | 状态 | 代码与证据 |
| --- | --- | --- |
| `model_revision × provider × role × profile` 领域模型 | CPU_VERIFIED | `xllm_service/placement/placement_types.{h,cpp}` |
| Prefill/Decode/Aggregated 独立容量计算 | CPU_VERIFIED | `xllm_service/placement/placement_planner.{h,cpp}` |
| failure headroom、warm spare、扩缩最大步长 | CPU_VERIFIED | `tests/xllm_service/placement/placement_planner_test.cpp` |
| 快扩、慢缩、hysteresis、cooldown、样本/OOD 门禁 | CPU_VERIFIED | `tests/xllm_service/placement/placement_planner_test.cpp` |
| 模拟 HBM cache-loss 与已确认 Store coverage 经济门禁 | CPU_VERIFIED | `PlacementPlannerTest.ConfirmedStoreCoverageCanUnlockScaleDown` |
| 全局 device budget、保护容量与稳定优先级 | CPU_VERIFIED | `xllm_service/placement/placement_budget_allocator.{h,cpp}` 与对应测试 |
| 生命周期与三重 fencing | CPU_VERIFIED | `xllm_service/placement/placement_lifecycle.{h,cpp}`；覆盖 replay、旧 generation/incarnation、drain commit 与新 incarnation |
| leader-fenced desired store | PARTIAL / CPU_VERIFIED CORE | `placement_desired_store.{h,cpp}` 与 `scheduler/etcd_client`；双 master identity + mod revision CAS、严格 codec、全量快照容量门 |
| 确定性 reconcile 与 operation executor | PARTIAL / CPU_VERIFIED CORE | `placement_reconciler.{h,cpp}`、`placement_actuator.{h,cpp}`；unknown 只 Query、drain/terminate proof、fake actuator |
| command/status 持久化与 Service 集成 | IN_PROGRESS | V3-P2/P3 剩余项及 P5 |
| 真实 NPU/HBM、部署系统与线上闭环 | NPU_AND_CLUSTER_PENDING | V3-P6 线上矩阵 |

当前 CPU Planner 不按 NPU/GPU/MLU 分支。硬件差异只通过不可变 capacity profile 和
Provider lifecycle conformance 输入；CPU 验证控制逻辑，不能替代真实 HBM、CANN、网络、
模型加载和设备释放验证。

## 已验证语义

- 输入中的 NaN、Infinity、非法范围、非法 identity、clock regression 和 observation
  replay 均 fail closed；
- P、D、Aggregated 使用各自工作量和 capacity，不采用固定 P:D 比例；
- forecast 短于 load/warmup p99 时保留 warm spare；
- 扩容由 forecast、queue、reject、TTFT、TPOT 或 KV pressure 触发，已有 operation 时不
  重复动作；
- 缩容必须同时满足样本、稳定窗口、cooldown、低负载和 cache-loss 收益门；
- Store coverage 只作为已确认持久化覆盖的输入；默认模拟完整 HBM cache loss；
- 全局预算先分配 safe-required 保护增量，再按 `priority → SLO risk → pool key` 稳定分配；
- 已有容量超预算时只标记 `OVERCOMMITTED`，不绕过 Planner 强制缩容。
- lifecycle command 同时绑定 operation id、desired generation 与 Engine incarnation；任意已应用
  phase 重放不回退状态，drain commit 后不能取消，重新创建必须使用新 incarnation；
- desired 写入同时比较 Service master address、master incarnation 和 etcd mod revision；损坏、
  超记录/字节预算的 leader 恢复快照 fail closed；
- reconcile 不在存在非终态/未知 operation 时叠加副作用；执行结果不明后只 Query；BEGIN_DRAIN
  成功必须证明 admission closed 且 P/transfer/reservation/D/output/cleanup 全部归零。

## 测试与门禁

| Gate | CPU test | 当前结果 |
| --- | --- | --- |
| V3-P0 领域输入、稳定错误与 lifecycle | types/planner/lifecycle 边界和迁移测试 | CPU_VERIFIED |
| V3-P1 P/D/A、稳定性、cache loss、预算 | planner 与 budget allocator 单测 | CPU_VERIFIED |
| V3-P2 desired store | codec、快照容量、并发 CAS、旧 leader/revision | PARTIAL；核心 CPU_VERIFIED，operation store 待完成 |
| V3-P3 reconcile/actuator | deterministic victim、unknown、proof、容量 | PARTIAL；核心 CPU_VERIFIED，持久恢复/集成待完成 |
| 全仓回归 | `xllm-dev service-test ... native Debug` | 440/440 PASS |
| V3-P4/P5 | Provider lifecycle loopback / Service 集成 | IN_PROGRESS |
| V3-P6 | NPU、真实 HBM、etcd/actuator、阶梯流量、24h+ soak | PENDING |

## 达到 V3 代码完成仍需

1. 在已完成 desired fencing/快照基础上补齐 command/status 持久化和 leader 恢复；
2. 在已完成 reconcile/fake actuator 基础上补齐有界 operation ledger 的持久恢复与终态回收；
3. 完成 xLLM Native 与 vLLM-Ascend drain/query/new-incarnation conformance；
4. 接入 Service SHADOW/create-only/ENFORCED、结构化日志和低基数指标；
5. 通过 P0-P5 全量 CPU/loopback/并发/故障回归并发布线上操作手册；
6. 最终在 P6 真实集群验证后，才可从 `NPU_AND_CLUSTER_PENDING` 升级为 `VERIFIED`。
