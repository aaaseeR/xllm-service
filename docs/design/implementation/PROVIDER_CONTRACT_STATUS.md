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

# Provider Contract 与能力门禁核心

## 基本信息

- Owner：xLLM Service V2
- 状态：CPU_VERIFIED
- 关联设计/Requirement ID：G-2、G-1、F66、F72、F73、F80-F82、D50、D52、D56
- 最近验证基线：xLLM `cd91965c`、xllm-service 本状态文档所在提交
- 验证环境和日期：xllm-dev-sandbox，Ubuntu 24.04 ARM64，2026-08-08

## 支持范围

| Provider | Mode | Model/Profile | 支持状态 | 限制与证据 |
| --- | --- | --- | --- | --- |
| xLLM Native | `REMOTE_PD/LAYERWISE_PUSH/P_FIRST` | contract v1 | CPU_VERIFIED | Descriptor、capability、plan role shape 与 deadline 门禁通过 |
| xLLM Native | `LOCAL_PREFILL_DECODE/NONE/D_ONLY` | contract v1 | CPU_VERIFIED | mixed accounting 与 structured admission 是硬能力 |
| xLLM Native | `PREFILL_ONLY/NONE/P_ONLY` | contract v1 | CPU_VERIFIED | 独立 P-only capability 与单 P role shape 是硬门禁 |
| vLLM-Ascend | `AGGREGATED/NONE/SINGLE` | contract v1 | CPU_VERIFIED | 仅证明公共契约；真实 Agent 尚未实现 |
| vLLM-Ascend | `REMOTE_PD/LAYERWISE_PUSH/D_FIRST` | contract v1 | BLOCKED | schema 可表达，V2 固定返回 `MODE_NOT_OPEN` |

## 实现

- 代码入口与核心接口：`xllm_service/provider/provider_contract.*`、
  `provider_adapter.h`、`provider_registry.*`。
- 状态、资源和错误码权威位置：wire 类型和稳定
  `ProviderContractError` 只定义在 xLLM `xllm/proto/provider.proto`；Service
  不复制 enum。Mode/capability 对应关系只定义在公共 Resolver。
- 跨仓协议与依赖：xLLM 是 `provider.proto` 的唯一源；xllm-service 通过
  `proto_xllm` 直接生成和链接同一文件。`renderer_digest` 覆盖 tokenizer +
  template 渲染契约，STRICT 编码结果必须与 Descriptor 相等。
- Adapter registry 按 `(provider_id, profile_digest)` 保存不可变 Adapter；这是
  Provider/profile 级 Adapter 注册表，不替代 G3 的 Engine incarnation Registry。
- 明确不支持范围：本批次没有接入生产 Scheduler，没有实现
  `XllmNativeAdapter`/`VllmAscendAdapter` 的 Submit、Stream、Cancel、Reserve 和
  State Stream，也不声称任何真实 NPU Provider 已通过 conformance。

## 需求与测试追踪

| Requirement ID | CPU test | Torch CPU test | NPU test | 结果 |
| --- | --- | --- | --- | --- |
| G-1/F66 单一 wire 真相 | xLLM `ProviderProtocolTest` golden wire、roundtrip、optional presence、field number | N/A，无 tensor 逻辑 | N/A | PASS，4/4 |
| G-2/F80 mode/capability | `ResolvesEveryCompleteV2CapabilityRow`、未知 mode/transfer 负向测试 | N/A | 待真实 Provider | PASS |
| G-2/F81 Descriptor | 四种开放 profile、缺能力、重复项、runtime alias 负向测试 | N/A | 待真实 Provider | PASS |
| G-2/F82 ExecutionPlan | 四种 role shape、capability、deadline、identity 门禁 | N/A | 待真实 Provider | PASS |
| STRICT renderer | canonical/encoded/model/capability/renderer 一致性正负测试 | N/A | 待真实 tokenizer/runtime | PASS |
| EngineState schema | UNKNOWN 与 0、per-DP、ratio、histogram 负向测试 | N/A | 待 State Stream | PASS |
| Adapter registry | ownership、lookup、重复 key 与非法 Descriptor | N/A | N/A | PASS |

Service 新增 17 个 `ProviderContractTest`，全量 service CPU 回归为 113/113；
xLLM CPU 公共路径为 56/56。

## 完善情况

- 已完成：contract v1 单一 proto；稳定错误码；完整 V2 mode/capability Resolver；
  Provider/open-mode 门禁；Descriptor、Canonical/Encoded Request、EngineState、
  ExecutionPlan 校验；`attempt_seq=0` 与缺失字段可区分；线程安全、只增不删的
  Adapter registry；CPU conformance。
- 已知缺口/风险：G-2 的生产 Adapter 和 `SelectPlans` 接入仍属于后续 Provider
  Pool 批次；G3 的 `(provider_id, profile_digest, incarnation_id)` Engine Registry
  尚未实现；当前只证明公共契约逻辑，不证明硬件 Runtime 行为。
- 回滚与兼容：协议为全新 additive schema；旧二进制不会读取这些消息。V2
  调用方必须对 `contract_version != 1`、未知 enum 与缺能力稳定 fail closed。
- 性能、容量和观测证据：纯 CPU 校验路径，无生产吞吐结论；State Stream 与
  观测基线由 G0/G3 批次交付。
- 达到 VERIFIED 仍需完成：两个真实 Adapter conformance、Scheduler 只依赖
  CanonicalRequest/ExecutionPlan、Engine Registry/State Stream、NPU smoke 与故障门禁。
