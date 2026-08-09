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

# vLLM-Ascend 严格 Provider Agent

## 基本信息

- Owner：xLLM Service V2
- 状态：NPU_PENDING（CPU 核心已验证）
- 关联设计/Requirement ID：G-2、G1-G4/M0、F75、F79、F83、F84、D32、D48
- 最近验证基线：xllm-service 本状态文档所在提交
- 验证环境和日期：xllm-dev-sandbox，Ubuntu 24.04 ARM64，2026-08-09

## 支持范围

| Provider | Mode | 支持状态 | 限制与证据 |
| --- | --- | --- | --- |
| vLLM-Ascend | `AGGREGATED/NONE/SINGLE` contract v1 | CPU_VERIFIED | Descriptor、唯一 Agent ingress、attempt Query/Cancel、deadline、per-DP State 和 Service aggregated hold 已接生产路径 |
| vLLM legacy sidecar | Descriptor-less contract v0 | COMPATIBLE | 保留 lease/aggregate metrics；无严格 V2 语义，不能进入 strict route |
| vLLM-Ascend | `REMOTE_PD` | BLOCKED | V2 固定不开跨 Provider 或 vLLM split topology |
| vLLM-Ascend | NPU deep health | NOT_IMPLEMENTED | 不伪造 `DEEP_HEALTH=HEALTHY`；等待独立设备探活或上游可靠信号 |

## 实现

- `vllm_sidecar/descriptor.py` 对 runtime/model/topology/KV/scheduler 五组 verified
  facts fail closed，生成确定性 profile SHA-256 和完整 contract-v1 Descriptor。
- `vllm_sidecar/agent.py` 是 Chat/Completion 的唯一反向代理入口，要求
  `X-Request-UID`、`X-Attempt-Seq` 和 `X-Remaining-Deadline-Ms`，并向 vLLM 注入稳定
  request ID。健康、lease 或 incarnation 失效会立即 fence 新 ingress。
- `vllm_sidecar/attempts.py` 使用 incarnation-scoped、有 record/terminal TTL 硬上限的
  ledger。Submit exactly-once；未知 Cancel 安装 `CANCELLED_BEFORE_CREATE`；Query
  `ABSENT` 不构成终态证明；deadline、Cancel、fence 和断连关闭持有的 upstream。
- `vllm_sidecar/metrics.py` 保留 vLLM DP label，发布 running/waiting/deferred、KV ratio
  和 admission credit。多 DP 缺失时状态为 `PARTIAL`；KV ratio 取逐 DP/legacy 最大值，
  禁止把比例相加。
- `vllm_sidecar/sidecar.py` 把 Descriptor 写入唯一 etcd Registry，heartbeat 生成单调
  `state_seq` 和 incarnation/profile/model 对齐的 EngineState。lease 重建使用新
  incarnation，并先 fence 旧 Agent 状态。
- Service 的 `attempt_control_client.*` 按 Provider 选择 Native brpc 或 Agent HTTP
  Query/Cancel。vLLM 响应只有 `request_uid/attempt_seq/incarnation_id` 精确匹配且状态
  已知为 terminal 时才形成 terminal proof；错误 HTTP、身份错配、malformed JSON、
  未知 enum 和非终态全部 fail closed。
- Scheduler 在 HTTP dispatch 前安装 `AGGREGATED_EXECUTION` hold；响应头确认
  GenerationCommit，流/非流终态收敛 hold，失败/断连则 Cancel 或把同一 cleanup token
  转入有界后台表。

## 需求与测试追踪

| Requirement | CPU test | NPU/部署 test | 结果 |
| --- | --- | --- | --- |
| strict Descriptor/Profile | 必填字段、非法 topology、raw ingress、身份交叉校验、digest 确定性 | 真实部署 facts 校验 | PASS / PENDING |
| Submit exactly-once | 64 路同 key 并发仅一个成功；duplicate/tombstone/capacity | 真实高并发 vLLM | PASS / PENDING |
| Query/Cancel/fence | lifecycle、Cancel-before-create、stale incarnation、Cancel 与未知 Submit 竞态 | SIGKILL、lease 分区 | PASS / PENDING |
| local deadline | fake clock ledger 与延迟 upstream loopback；超时返回 terminal `EXPIRED` | NPU abort 到资源释放时延 | PASS / PENDING |
| HTTP proxy | Chat/Completion payload/request ID、未知推理路径防旁路、health/livez、成功/重复/错误、流终态基础路径 | 真实 SSE/客户端断流；Anthropic 尚未开放 | PASS / PENDING |
| EngineState | per-DP label、缺 rank `PARTIAL`、ratio 聚合、state sequence/identity | 真实 vLLM-Ascend metrics | PASS / PENDING |
| Service hold | aggregated commit/terminal invariant；Agent 精确身份 terminal/malformed response parser；全量 288/288 | Submit ACK 丢失一万次 | PASS / PENDING |

## 完善情况

- 已完成：B4/B5 的 CPU 可验证核心、生产构建接线、严格/legacy 模式隔离和用户运行文档。
- 已知缺口/风险：Python HTTP runtime 不是高性能数据面，目标部署需通过容量压测决定是否
  替换 transport 实现；上游 abort 的真实资源释放、深层健康、同命和网络隔离尚未证明。
- 回滚与兼容：不提供 `--provider-config` 时继续 contract-v0 legacy sidecar；strict
  Descriptor 与旧 raw 地址不能混用。回滚 strict route 必须先 drain，再撤销注册。
- 达到 VERIFIED 仍需完成：真实 vLLM-Ascend/NPU conformance、Agent-only SIGKILL、原始
  端口旁路检查、etcd ownership 分区、SSE/abort/deadline 故障矩阵与容量/性能门禁。
