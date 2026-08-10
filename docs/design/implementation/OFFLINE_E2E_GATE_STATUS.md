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
- 交付规则：单元测试、组件 loopback 或单进程 fake 通过不能替代本门；V2/V3 发布候选必须先通过离线 `smoke` 与 `stress`，失败时不得进入 NPU/线上验证

## 生产等价范围

| 场景 | 实际进程和协议 | 必须通过的语义 |
| --- | --- | --- |
| V2 | 独立 etcd、两个生产 Service、2P+2D Native Engine、并发 OpenAI HTTP/SSE 客户端 | 短 deadline、不可满足 deadline、有界过载、Prefill/lease 丢失、短 etcd grace、长 etcd self-fence、new-incarnation 恢复、Leader `SIGKILL`、严格流控、零资源泄漏 |
| V3 | 独立 etcd、两个生产 Service、deployment gateway、多个严格 Agent/Runtime、并发 OpenAI HTTP/SSE 客户端 | 1→3→1、单副本过载与三副本恢复、deadline/断流 Cancel、create 响应丢失后 Query、Drain race、Agent+Runtime `SIGKILL`、route 重选、device budget、Leader `SIGKILL`、durable 收敛、零资源泄漏 |

Harness 使用真实 socket、etcd lease/watch、brpc/HTTP 和系统信号；mock 只替代尚不可在 CPU 运行的 Engine/NPU Runtime 与部署平台。每个 Runtime 的固定容量 Torch CPU tensor 模拟 HBM block 地址、内容、高水位与清零，验证控制链和资源账本，不冒充真实 NPU HBM 性能证明。

## 远端实现和证据入口

- [E2E harness](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/e2e/online_cluster_stress.py)
- [Native Engine](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/e2e/mock_native_engine.cpp)
- [vLLM Runtime](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/e2e/mock_vllm_runtime.py)
- [Deployment gateway](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/e2e/mock_deployment_gateway.py)
- [运行说明](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/e2e/README.md)
- [Coding CI 硬门](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/.coding-ci.yml)

每次运行在 `build/e2e-artifacts/<run-id>/` 生成 `report.json`、独立进程日志、指标快照和 etcd durable 快照。报告必须包含请求量、失败数、P50/P95/P99、实例分布、故障证据、simulated HBM 高水位/终态、低基数指标和持久状态；只有终态资源全部归零才通过。

## 当前验证结果

| Gate | 负载 | 结果 |
| --- | --- | --- |
| V2 smoke | baseline 120/120、SSE 16/16；50 ms deadline 60.6 ms 终止并恢复 16/16；过载 40 成功/56 结构化拒绝并恢复 32/32；Prefill `SIGKILL` 64/64；短/长 etcd 与 failover 安全阶段全成功 | PASS，报告 `1786381540-9005052f` |
| V2 stress | baseline 1200/1200、SSE 160/160；50 ms deadline 68.4 ms；不可满足 deadline 64/64 结构化拒绝；过载 80 成功/880 结构化拒绝；Prefill `SIGKILL` 640/640；短 etcd 120/120；Leader failover 后 1200/1200 | PASS，simulated HBM 高水位 128 blocks，终态 allocation/block 全零；报告 `1786381349-34fd2d1b` |
| V3 smoke | 1→3→1、deadline/真实断流、Drain race、Agent+Runtime 突然丢失与替换、Leader failover | PASS，报告 `1786381573-aa3ddd30`，所有硬断言成立 |
| V3 stress | 单副本高压 446 成功/1354 结构化 backpressure；三副本后 1800/1800；deadline 3/3 预期终止；Drain 重选 12/12；突然丢失 1798/1800 且仅 2 个允许的在途 transport failure；替换后 1200/1200；failover 后 1800/1800 | PASS，实际使用 4 个 incarnation，所有 simulated HBM 终态 allocation/block/tensor 全零；报告 `1786381432-31030c4b` |

同一轮全仓回归为 xllm-service 517/517 CPU CTest 和 xLLM 当前 1069 个 CPU CTest 全量 PASS；63 个并发/状态机高风险用例重复 20 轮累计 1260/1260 PASS，KV State timeout/非法请求测试拆分后目标用例连续 100 轮通过。三轮审查、修复位置和异常复核见 [三轮深度审查状态](./V2_V3_THREE_ROUND_DEEP_REVIEW_STATUS.md)。

## 尚未覆盖和关闭条件

- 真实 NPU HBM、CANN/kernel、DMA/RDMA、模型数值和吞吐容量；
- 生产 etcd quorum/compact、跨机网络分区、真实部署系统和基础设施级故障组合；
- 生产 prompt/output 分布、多模型、多租户、KV pressure 阶梯负载和 24h+ soak；
- 真实告警、dashboard、trace/log 关联和容量阈值校准。

上述项目按线上验证手册完成前，只能标记离线集群已验证，不能标记生产 `VERIFIED`。
