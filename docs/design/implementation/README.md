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

# 功能开发状态文档

本目录记录各版本功能的真实支持范围和完善情况。每项功能使用一个
`<feature>_STATUS.md`，并遵循
[V2 代码开发与交付规范](../00_XLLM_SERVICE_V2_DEVELOPMENT_STANDARD.md)第 5 节。

## 状态索引

面向评审和汇报的单页能力入口见
[V2 当前能力与远端代码索引](../12_XLLM_SERVICE_V2_CURRENT_CAPABILITIES.md)；本目录继续保存各功能的细粒度开发证据。

| 功能 | 状态 | 文档 |
| --- | --- | --- |
| 双仓 CPU 开发基线 | CPU_VERIFIED | [V2_BASELINE_STATUS.md](./V2_BASELINE_STATUS.md) |
| V2-B0 至 V2-B5 内部开发门 | CPU_VERIFIED / NPU_PENDING | [B0_B5_STATUS.md](./B0_B5_STATUS.md) |
| V2-B6 至 V2-B10 开发门 | CPU_VERIFIED / NPU_AND_CLUSTER_PENDING | [B6_B10_STATUS.md](./B6_B10_STATUS.md) |
| V2 首版 B10 代码交付 | CPU_VERIFIED / NPU_AND_CLUSTER_PENDING | [B10_V2_DELIVERY_STATUS.md](./B10_V2_DELIVERY_STATUS.md) |
| V2-B7 K0 KV Shadow | CPU_VERIFIED / NPU_AND_CLUSTER_PENDING | [B7_K0_KV_SHADOW_STATUS.md](./B7_K0_KV_SHADOW_STATUS.md) |
| Opus 5 V2 Review 整改 | CPU_VERIFIED / NPU_PENDING | [OPUS5_REVIEW_REMEDIATION.md](./OPUS5_REVIEW_REMEDIATION.md) |
| Provider Contract 与能力门禁核心 | CPU_VERIFIED | [PROVIDER_CONTRACT_STATUS.md](./PROVIDER_CONTRACT_STATUS.md) |
| vLLM-Ascend 严格 Provider Agent | NPU_PENDING | [VLLM_AGENT_STATUS.md](./VLLM_AGENT_STATUS.md) |
| G3 Engine Registry 与 State Stream | CPU_VERIFIED / NPU_AND_CLUSTER_PENDING | [STATE_STREAM_STATUS.md](./STATE_STREAM_STATUS.md) |
| G0 请求事件协议与观测 CPU 核心 | CPU_VERIFIED / NPU_AND_CLUSTER_PENDING | [OBSERVABILITY_STATUS.md](./OBSERVABILITY_STATUS.md) |
| V2 Debug、日志与性能分析手册 | CPU_VERIFIED / CLUSTER_CALIBRATION_PENDING | [OBSERVABILITY_RUNBOOK.md](./OBSERVABILITY_RUNBOOK.md) |
| G1 执行资源安全协议 CPU 核心 | CPU_VERIFIED / NPU_PENDING | [RESOURCE_SAFETY_STATUS.md](./RESOURCE_SAFETY_STATUS.md) |
| G2 输出定序、缺口与请求截止时间 CPU 核心 | CPU_VERIFIED / NPU_AND_CLUSTER_PENDING | [OUTPUT_DEADLINE_STATUS.md](./OUTPUT_DEADLINE_STATUS.md) |
| V3 Placement/Autoscale | CPU_VERIFIED / NPU_AND_CLUSTER_PENDING（P0-P5 完成） | [V3_PLACEMENT_AUTOSCALE_STATUS.md](./V3_PLACEMENT_AUTOSCALE_STATUS.md) |
| V3 线上验证与反馈 | P6 PENDING | [V3_ONLINE_VALIDATION_RUNBOOK.md](./V3_ONLINE_VALIDATION_RUNBOOK.md) |

状态文档使用以下模板：

```markdown
# <功能名称>

## 基本信息

- Owner：
- 状态：PLANNED | IN_PROGRESS | PARTIAL | CPU_VERIFIED | NPU_PENDING |
  VERIFIED | BLOCKED
- 关联设计/Requirement ID：
- 最近验证 commit、环境和日期：

## 支持范围

| Provider | Mode | Model/Profile | 支持状态 | 限制与证据 |
| --- | --- | --- | --- | --- |

## 实现

- 代码入口与核心接口：
- 状态、资源和错误码权威位置：
- 跨仓协议与依赖：
- 明确不支持范围：

## 需求与测试追踪

| Requirement ID | CPU test | Torch CPU test | NPU test | 结果 |
| --- | --- | --- | --- | --- |

## 完善情况

- 已完成：
- 已知缺口/风险：
- 回滚与兼容：
- 性能、容量和观测证据：
- 达到 VERIFIED 仍需完成：
```

没有对应测试证据的支持项不得标记 `CPU_VERIFIED` 或 `VERIFIED`；文档中的
`PARTIAL`、`NPU_PENDING` 和 `BLOCKED` 必须列出具体缺口及关闭条件。
