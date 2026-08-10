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

# xLLM Service V3 当前能力与远端代码索引

状态：`CPU_AND_OFFLINE_CLUSTER_VERIFIED / NPU_AND_ONLINE_PENDING`

V3 已完成 P0-P5 可移植代码：在 V2 请求快环之外运行 leader-fenced Placement 慢环，按
`provider × model revision × role × profile` 独立预测、规划、持久化和收敛副本。V2 Router
只消费 READY Engine，不等待 Placement。以下路径全部指向远端 `service_dev` 分支。

## 能力与实现

| 能力 | 远端代码 | 测试证据 |
| --- | --- | --- |
| Pool/profile/observation 与严格边界 | [placement_types](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/placement/placement_types.h)、[runtime config](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/placement/placement_config.cpp) | [config tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/xllm_service/placement/placement_config_test.cpp) |
| P/D/A 独立预测、快扩慢缩、HBM cache-loss | [planner](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/placement/placement_planner.cpp) | [planner tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/xllm_service/placement/placement_planner_test.cpp) |
| 全局 device budget 与稳定优先级 | [budget allocator](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/placement/placement_budget_allocator.cpp) | [budget tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/xllm_service/placement/placement_budget_allocator_test.cpp) |
| Engine lifecycle、drain commit 与 incarnation fencing | [lifecycle](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/placement/placement_lifecycle.cpp) | [lifecycle tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/xllm_service/placement/placement_lifecycle_test.cpp) |
| leader-fenced desired/command/status | [desired store](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/placement/placement_desired_store.cpp)、[operation store](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/placement/placement_operation_store.cpp)、[etcd transaction](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/scheduler/etcd_client/etcd_client.cpp) | [store tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/xllm_service/placement/placement_operation_store_test.cpp) |
| 确定性 reconcile、replacement-first 与安全 victim | [reconciler](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/placement/placement_reconciler.cpp) | [reconcile tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/xllm_service/placement/placement_reconciler_test.cpp) |
| Unknown-only-query、CANCELED、proof 与依赖有序 GC | [executor](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/placement/placement_actuator.cpp) | [actuator tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/xllm_service/placement/placement_actuator_test.cpp) |
| xLLM/vLLM lifecycle conformance | [Provider actuator](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/placement/provider_lifecycle_actuator.cpp) | [loopback tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/xllm_service/placement/provider_lifecycle_actuator_test.cpp) |
| 部署系统 create/terminate + Registry 证明 | [deployment actuator](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/placement/placement_deployment_actuator.cpp) | [gateway tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/xllm_service/placement/placement_deployment_actuator_test.cpp) |
| 请求/Engine/KV 观测转慢环输入 | [observation collector](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/placement/placement_observation_collector.cpp)、[input builder](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/placement/placement_input_builder.cpp) | [observation tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/xllm_service/placement/placement_observation_collector_test.cpp)、[builder tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/xllm_service/placement/placement_input_builder_test.cpp) |
| Service 慢环、模式热切换、日志与低基数指标 | [Scheduler integration](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/scheduler/scheduler.cpp)、[metrics](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/common/metrics.cpp) | [controller tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/xllm_service/placement/placement_controller_test.cpp) |
| 离线模拟线上扩缩容硬门 | [cluster E2E](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/e2e/online_cluster_stress.py)、[deployment gateway](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/e2e/mock_deployment_gateway.py)、[mock runtime](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/e2e/mock_vllm_runtime.py) | 真实双 Service/etcd/Agent/Runtime 进程；1→3→1、Drain race、响应丢失 Query、Leader SIGKILL、高并发零失败、每副本 Torch HBM 清零 |

## 四种运行模式

| override | 模式 | 行为 |
| --- | --- | --- |
| `-1` | CONFIG | 使用严格 JSON 配置中的 mode |
| `0` | DISABLED | 停止 Placement 周期，不影响 V2 Router |
| `1` | SHADOW | 只计算 recommendation，不写 desired、不生成新副作用 |
| `2` | ENFORCED_CREATE_ONLY | 写 desired 并允许扩容，不自动缩容 |
| `3` | ENFORCED | 完整 create/drain/cancel/terminate 收敛 |

线上回滚目标是 `1`，最多一个 loop interval 生效；已有未知 operation 继续 Query，不能删除
或盲目重发。配置样例见
[V3 placement config](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/docs/design/examples/xllm_service_v3_placement_config.json)。

## 正确性边界

- CPU/Torch CPU/simulated HBM 证明控制链路，不宣称真实 NPU/HBM 性能和释放正确；
- xLLM Service 感知 Provider capability、profile、lifecycle 和标准资源事实，不感知 CANN/
  CUDA/MLU API、device pointer 或 allocator 实现；
- committed drain 不可取消。需求反弹时先 CREATE replacement，容量恢复后再 TERMINATE；
- 真实 deployment gateway 必须按 operation id 幂等，并返回可验证的 engine uid/incarnation；
- 失败、fenced、容量超限、损坏快照和时钟回退均 fail closed，Router 保持独立运行。

## 剩余 P6

真实 NPU/HBM、模型 load/warmup、设备释放、生产多 Service/etcd quorum/网络、真实部署系统、
生产阶梯流量、故障矩阵、SLO goodput 和 24h+ soak 尚未执行。CPU 沙箱中的双 Service、
真实 etcd、部署网关、严格 Agent/Runtime、多副本扩缩和 Leader 故障已通过 smoke/stress，
但不替代上述线上证据。完成条件和证据模板见
[线上验证手册](./implementation/V3_ONLINE_VALIDATION_RUNBOOK.md)。
