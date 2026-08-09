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

# V2-B0 至 V2-B5 开发门状态

## 结论

V2-B0 至 V2-B5 的**可在 CPU 环境完成的代码与测试门**已经闭环，状态为
`CPU_VERIFIED / NPU_PENDING`。这些编号是内部开发门，不是缩小后的 V1，也不表示
V2 首个交付版本已经完成；V2 仍须继续完成多模型、精确 KV-aware、策略感知有界流控、
优先级与租户公平，并通过 NPU 和故障注入门禁。

| 开发门 | 范围 | CPU 状态 | 本轮主要证据 | 后续硬门禁 |
| --- | --- | --- | --- | --- |
| V2-B0 | 双仓分支、设计/规范、统一 Linux CPU 沙箱和协议测试底座 | CPU_VERIFIED | 外层 xLLM `service_dev` 联编；禁止向 main/master 开发 | NPU/CANN 环境基线 |
| V2-B1 | Provider Contract、CanonicalRequest、RequestCodec、ExecutionPlan 与双 Provider route shape | CPU_VERIFIED | contract-v1 正负校验、Provider 隔离和严格 plan 生产接线 | 真实 Provider conformance |
| V2-B2 | request/attempt/incarnation、事件、deadline、断连和统一 execution hold | CPU_VERIFIED | Native REMOTE_PD 控制面与 bounded cleanup；aggregated hold 已复用同一不变量 | NPU KV/slot 释放与长稳故障矩阵 |
| V2-B3 | Registry/State Stream、blind 状态机、readiness、Link 对账和直连证据 | CPU_VERIFIED | Registry/State/Link/Readiness 生产核心及 vLLM per-DP EngineState | 真实 etcd 切主、网络分区和 NPU handshake |
| V2-B4 | vLLM-Ascend 严格 Provider Agent | CPU_VERIFIED | contract-v1 Descriptor、唯一 ingress、attempt ledger、deadline/cancel、fencing 与 loopback | Agent/vLLM 同命、原始端口隔离、真实 vLLM-Ascend |
| V2-B5 | 双 Provider M0 的 `AGGREGATED` Submit/Query/Cancel 与资源收敛 | CPU_VERIFIED | Service dispatch 前安装 aggregated hold；Agent 精确身份 terminal/cancel fence 才释放；HTTP/stream 接线 | 真实 SSE、abort 资源释放和一万次故障门禁 |

## B4/B5 生产路径

1. vLLM-Ascend Agent 只有在 verified profile、vLLM shallow health、etcd lease 和新
   incarnation 全部成立后才开放 `/readyz` 与推理 ingress。
2. Service 在发出聚合请求前预留 cleanup token，并安装单 holder 的
   `AGGREGATED_EXECUTION` hold；失败时不 dispatch。
3. Agent 对 `(request_uid, attempt_seq)` 原子 `begin`，重复 Submit 稳定拒绝；本地
   deadline、Cancel、断连或 ownership fencing 均终止旧 attempt。
4. Service 只把成功响应头当作同 attempt 的 GenerationCommit；正常终态、Query 中
   `request_uid/attempt_seq/incarnation_id` 精确匹配的 terminal proof，或 Cancel 的
   fence ACK，才能收敛 hold。
5. 请求可以先向客户端失败，但 outcome 未证明时最小 cleanup record 继续存活；不能
   因请求对象析构而丢失资源责任。

## CPU 验证基线

- xllm-service：289/289 CTest 通过，包含 aggregated execution hold 与 Agent
  Query/Cancel 响应证明；三个生产服务二进制完成 Debug 构建和链接。
- vLLM Agent：44/44 Python 测试覆盖 Descriptor、metadata、metrics、lease/heartbeat、
  ledger、HTTP 代理、Cancel/Submit 竞态、本地 deadline 和 incarnation fencing。
- xLLM：本轮未修改外层 xLLM；Service 继续显式使用外层 `service_dev` 构建，避免回退
  到 pinned submodule 的旧协议真相。
- 沙箱：Ubuntu 24.04 ARM64、Clang 18、Python 3.12、PyTorch CPU；`pytest` 与
  `requests` 均为镜像内显式依赖。

## 未完成项

- Agent 尚未实现独立 NPU deep-health；在此之前只发布 shallow health，不能把上游
  `/health` 推断成 `DEEP_HEALTH=HEALTHY`。
- `fate_bound_mode` 与 `raw_ingress_isolated` 当前由 verified deployment profile
  fail-closed 声明，仍须以 SIGKILL、restart-unit 和网络策略实测证明。
- 尚无真实 vLLM-Ascend SSE、abort 后显存/KV/slot 释放时延、etcd 分区和一万次
  Submit 结果不明故障门禁；这些结论不得由 CPU loopback 外推。
- B6 以后继续完成 V2 首个交付版本范围；B0-B5 不得单独发布为 V2。
