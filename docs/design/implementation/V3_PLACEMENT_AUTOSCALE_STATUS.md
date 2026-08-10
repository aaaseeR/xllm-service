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
- 状态：`CPU_AND_OFFLINE_CLUSTER_VERIFIED / NPU_AND_ONLINE_PENDING`
- 范围：V3-P0 至 P5 可移植代码、CPU 单元/loopback 和离线多进程集群验证已完成；
  P6 真实 NPU/HBM、生产 etcd/部署系统和生产流量验证待执行
- 关联设计：[V3 Placement 与 Autoscale](../14_XLLM_SERVICE_V3_PLACEMENT_AUTOSCALE_DESIGN.md)
- 当前能力入口：[V3 当前能力与远端代码索引](../15_XLLM_SERVICE_V3_CURRENT_CAPABILITIES.md)
- 线上入口：[V3 线上验证手册](./V3_ONLINE_VALIDATION_RUNBOOK.md)
- 最近验证：`service_dev`，ARM64 Linux Debug 沙箱，2026-08-10

## P0-P6 状态

| Gate | 状态 | 已完成能力与 CPU 证据 |
| --- | --- | --- |
| V3-P0 | CPU_VERIFIED | 严格配置、pool/profile/observation 领域模型、非法输入和 lifecycle 状态机 |
| V3-P1 | CPU_VERIFIED | P/D/A 独立规划、forecast、快扩慢缩、failure headroom、HBM cache-loss、全局预算 |
| V3-P2 | CPU_VERIFIED | desired/command/status codec、快照容量门、master 三元组 fencing、CAS、leader 恢复 |
| V3-P3 | CPU_VERIFIED | 确定性 reconcile、create/drain/cancel/terminate、unknown-only-query、proof、operation GC |
| V3-P4 | CPU_VERIFIED | xLLM Native BRPC 与 vLLM-Ascend HTTP lifecycle conformance、loopback 和严格 token |
| V3-P5 | CPU_AND_OFFLINE_CLUSTER_VERIFIED | Scheduler 慢环、SHADOW/create-only/ENFORCED、动态回滚、部署 gateway、日志/指标、三 serving binary；双 Service/etcd/Agent/Runtime 多进程压力与故障门通过 |
| V3-P6 | NPU_AND_ONLINE_PENDING | 真实 load/warmup/HBM/drain/设备释放、etcd/leader/部署故障、阶梯流量和 24h+ soak |

## 已验证的关键语义

- CPU 只验证控制链路；HBM 价值通过 KV Shadow 的精确 READY 位置和 simulated cache-loss
  建模验证，不能替代真实 NPU/HBM 证明。
- Planner 不按 NPU/GPU/MLU 分支。硬件、Runtime、TP/DP/PP/EP 和 KV layout 由不可变
  capacity profile 与 Provider lifecycle/deployment actuator 提供。
- desired、command、status 写删均比较 master address、incarnation、epoch 和 etcd mod
  revision；新 leader 全量恢复，未知副作用只 Query。
- operation id 在 leader/generation/pool/action 基础上带 leader-local 唯一 cycle ordinal；
  终态 CREATE 证明过期后生成新的修复 operation，不会重放旧成功记录。
- CANCEL 只作用于未 commit drain，并把旧 BEGIN_DRAIN 收敛为 `CANCELED`；commit 后需求
  反弹采用 replacement-first，容量恢复后才终止旧 incarnation；完成证明按 incarnation
  和 generation 匹配，旧 CANCEL 不能遮蔽后续 drain。
- 终态 command/status 原子、有界回收；BEGIN_DRAIN 先于对应 CANCEL/TERMINATE 证明删除，
  `FAILED/FENCED` 保留并 fail closed。
- 只有 fresh READY、支持 drain、无 reservation/transfer 且 cache value 已知的实例可成为
  victim；已有 DRAINING 只由持久 ledger 驱动。
- `placement_mode_override` 支持在一个 loop interval 内切回 SHADOW；Router 不等待慢环。
- 日志使用稳定 stage/reason/code，指标 label 保持低基数；pool/operation 细节进入 VLOG。

## Placement 容器选型决策

`xllm_service/placement/` 有意统一使用 `std::map/std::set`，包括当前只做查找或去重的位置。Placement 是秒到分钟级有界慢环，`max_pools`、`max_models`、`max_operations_per_pool` 和 operation ledger 都有硬上界；这里优先保证相同 observation/desired/ledger 输入在不同进程、编译器和重启恢复后产生稳定的 pool、victim、intent 与副作用遍历顺序。统一使用有序容器还防止纯查找代码后续增加迭代时无意引入 `unordered_*` 非确定性。该策略是对通用“键序无关时优先 unordered”规则的模块级有意例外；若未来 profiling 证明 O(log n) 成为慢环瓶颈，替换必须同时提供显式排序、确定性重放与故障恢复回归测试，不能直接依赖 unordered iteration。

## CPU 验证门

| 验证项 | 命令/测试 | 结果 |
| --- | --- | --- |
| Placement 全部组件 | `tests/xllm_service/placement/*_test` 13 个二进制 | PASS |
| Reconcile | `placement_reconciler_test` 16 tests | PASS |
| Executor/GC | `placement_actuator_test` 9 tests | PASS |
| Durable ledger | `placement_operation_store_test` 5 tests | PASS |
| Runtime config | `placement_config_test` 8 tests（含文档样例） | PASS |
| Controller | `placement_controller_test` 5 tests | PASS |
| Deployment/Provider loopback | `placement_deployment_actuator_test`、`provider_lifecycle_actuator_test` | PASS |
| 离线多进程集群 | `online_cluster_stress.py --scenario v3 --mode smoke/stress` | PASS；1→3→1、响应丢失、Drain race、Leader SIGKILL、Torch HBM 清零 |
| V2 HBM/KV 与 flow 回归 | KV Shadow、flow control 既有 CPU suite | PASS |
| Serving 链接门 | master、RPC、HTTP 三个 serving binary | PASS |
| 全仓 Service CPU | 沙箱 `xllm-dev service-test ... native Debug` | 512/512 PASS |
| xLLM CPU/provider 协议 | `xllm-dev xllm-test ... native Debug` | 141/141 PASS |
| xLLM simulated HBM/allocator | `simulated_hbm_test` | 15/15 PASS |

## 线上前不可误报的边界

- 未验证真实 NPU 模型加载与 warmup p99、CANN/通信故障、真实 HBM 释放与 cache-loss；
- 未验证真实 deployment gateway 的 operation 幂等保留、Registry lease 延迟与终止证明；
- 已离线验证多 Service leader kill、etcd 短/长 stall、部署响应丢失；尚未验证生产 etcd
  集群的 compact、仲裁丢失、真实网络分区和部署系统超时组合；
- 未用生产短/长 prompt、长 output、多模型与 KV pressure 跑阶梯流量和 24h+ soak；
- 未完成线上 capacity profile、阈值、SLO residual 和低基数 dashboard 校准。

上述 P6 全部通过前，V3 只能标记
`CPU_AND_OFFLINE_CLUSTER_VERIFIED / NPU_AND_ONLINE_PENDING`，不能标记生产 `VERIFIED`。
