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

# V2/V3 离线多进程 E2E 交付门状态

## 基本信息

- Owner：xLLM Service
- 状态：`CPU_AND_OFFLINE_CLUSTER_VERIFIED / NPU_AND_ONLINE_PENDING`
- 关联设计：V2 交付规范、V2 当前能力、V3 Placement/Autoscale
- 最近验证：`service_dev`，ARM64 Linux Debug 沙箱，2026-08-10
- 交付规则：单元测试、组件 loopback 或单进程 fake 通过不能替代本门；V2/V3 发布候选必须
  先通过离线 `smoke` 与 `stress`，失败时不得进入 NPU/线上验证

## 生产等价范围

| 场景 | 实际进程和协议 | 必须通过的语义 |
| --- | --- | --- |
| V2 | 独立 etcd、两个生产 Service、2P+2D Native Engine、并发 OpenAI HTTP 客户端 | Engine/lease 丢失、短 etcd grace、长 etcd self-fence、new-incarnation 恢复、Leader `SIGKILL`、严格流控、零资源泄漏 |
| V3 | 独立 etcd、两个生产 Service、deployment gateway、多个严格 Agent/Runtime、并发 OpenAI HTTP 客户端 | 1→3→1、create 响应丢失后 Query、Drain race、route 重选、device budget、Leader `SIGKILL`、durable 收敛、零资源泄漏 |

Harness 使用真实 socket、etcd lease/watch、brpc/HTTP 和系统信号；mock 只替代尚不可在 CPU
运行的 Engine/NPU Runtime 与部署平台。每个 Runtime 的固定容量 Torch CPU tensor 模拟 HBM
block 地址、内容、高水位与清零，验证控制链和资源账本，不冒充真实 NPU HBM 性能证明。

## 远端实现和证据入口

- [E2E harness](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/e2e/online_cluster_stress.py)
- [Native Engine](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/e2e/mock_native_engine.cpp)
- [vLLM Runtime](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/e2e/mock_vllm_runtime.py)
- [Deployment gateway](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/e2e/mock_deployment_gateway.py)
- [运行说明](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/e2e/README.md)
- [Coding CI 硬门](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/.coding-ci.yml)

每次运行在 `build/e2e-artifacts/<run-id>/` 生成 `report.json`、独立进程日志、指标快照和
etcd durable 快照。报告必须包含请求量、失败数、P50/P95/P99、实例分布、故障证据、
simulated HBM 高水位/终态、低基数指标和持久状态；只有终态资源全部归零才通过。

## 当前验证结果

| Gate | 负载 | 结果 |
| --- | --- | --- |
| V2 smoke | 120 baseline + 60 Engine loss + 60 short-etcd + 30 long-etcd grace + 120 leader failover | PASS，所有预期成功阶段零失败；长故障直接请求失败且 Service fail closed，三个存活 Engine 以新 incarnation 恢复 |
| V2 stress | 1200 + 600 + 600 + 300 + 1200 | PASS，所有预期成功阶段零失败，Torch HBM 终态全零 |
| V3 smoke | 1→3→1、Drain race、leader failover | PASS，CREATE=3、TERMINATE=2，无重复副作用 |
| V3 stress | 各主要流量阶段 1800 请求、并发 64 | PASS，至少两个副本承载流量，Torch HBM 终态全零 |

## 尚未覆盖和关闭条件

- 真实 NPU HBM、CANN/kernel、DMA/RDMA、模型数值和吞吐容量；
- 生产 etcd quorum/compact、跨机网络分区、真实部署系统和基础设施级故障组合；
- 生产 prompt/output 分布、多模型、多租户、KV pressure 阶梯负载和 24h+ soak；
- 真实告警、dashboard、trace/log 关联和容量阈值校准。

上述项目按线上验证手册完成前，只能标记离线集群已验证，不能标记生产 `VERIFIED`。
