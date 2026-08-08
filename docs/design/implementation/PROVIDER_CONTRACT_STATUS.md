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
- 最近验证基线：xLLM `0d9a3f29`、xllm-service 本状态文档所在提交
- 验证环境和日期：xllm-dev-sandbox，Ubuntu 24.04 ARM64，2026-08-09

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
- 已实现可注册的 `XllmNativeAdapter` 与 `VllmAscendAdapter`。Native Adapter
  通过显式 renderer 接口取得原生 payload、精确 token 数和实际 renderer
  digest；vLLM-Ascend Adapter 只接受 `openai.http.json.v1`，保留原始 JSON
  payload，并在 Provider 未返回计数前如实发布 `TOKEN_COUNT_QUALITY_UNKNOWN`。
  Registry 在写入前同时校验 Descriptor 和 Adapter dispatch kind，阻断
  Provider 身份与数据面协议错配。
- 状态、资源和错误码权威位置：wire 类型和稳定
  `ProviderContractError` 只定义在 xLLM `xllm/proto/provider.proto`；Service
  不复制 enum。Mode/capability 对应关系只定义在公共 Resolver。
- 跨仓协议与依赖：xLLM 是 `provider.proto` 的唯一源；xllm-service 通过
  `proto_xllm` 直接生成和链接同一文件。`renderer_digest` 覆盖 tokenizer +
  template 渲染契约，STRICT 编码结果必须与 Descriptor 相等。
- Adapter registry 按 `(provider_id, profile_digest)` 保存不可变 Adapter；这是
  Provider/profile 级 Adapter 注册表，不替代 G3 的 Engine incarnation Registry。
- `InstanceMetaInfo` 已携带 `provider_id`、contract version、profile digest 与可选
  完整 `ProviderDescriptor`。完整 Descriptor 在 JSON 入口执行摘要/incarnation
  交叉校验，并在实例进入生产索引前运行公共 Descriptor validator；旧
  `backend_type` 只在 JSON 兼容入口映射一次，未知值或双身份冲突 fail closed。
  当前 vLLM sidecar 明确发布 contract version 0，继续属于 BEST_EFFORT 兼容桥。
- Scheduler 已按每个请求从可执行 Provider route 中选择 Adapter dispatch kind，
  不再读取进程级 `default_backend_type` 决定 tokenization 或数据面；该 CLI flag
  仅为旧部署参数兼容保留且不参与运行时决策。Round-robin、CAR、SLO 指标入口、
  静态 peer 列表和 Engine Link/Unlink 均按 Provider 隔离。Native `P+D` 与 vLLM
  `AGGREGATED` route shape 由同一个纯 CPU selector 决定，禁止跨 Provider 拼接。
- STRICT Native P/D 使用显式兼容矩阵，不要求两端 `profile_digest` 相等，但要求
  runtime/protocol、model revision/quantization、KV layout/dtype/block/cache group、
  Connector/version、layerwise transfer 和可证明 topology 一致，并生成确定性的
  compatibility proof。STRICT 与 BEST_EFFORT legacy 不能单边混配；矩阵同时用于
  route、readiness、静态 peer 与 Link/Unlink 候选。
- 明确不支持范围：本批次尚未让生产 Scheduler 只消费
  `CanonicalRequest/ExecutionPlan`，也没有实现
  Submit、Stream、Cancel、Reserve 和 State Stream；不声称任何真实 NPU
  Provider 已通过 conformance。

## 需求与测试追踪

| Requirement ID | CPU test | Torch CPU test | NPU test | 结果 |
| --- | --- | --- | --- | --- |
| G-1/F66 单一 wire 真相 | xLLM `ProviderProtocolTest` golden wire、roundtrip、optional presence、field number | N/A，无 tensor 逻辑 | N/A | PASS，7/7（含 G1 additive resource wire） |
| G-2/F80 mode/capability | `ResolvesEveryCompleteV2CapabilityRow`、未知 mode/transfer 负向测试 | N/A | 待真实 Provider | PASS |
| G-2/F81 Descriptor | 四种开放 profile、缺能力、重复项、runtime alias 负向测试 | N/A | 待真实 Provider | PASS |
| G-2/F82 ExecutionPlan | 四种 role shape、capability、deadline、identity 门禁 | N/A | 待真实 Provider | PASS |
| STRICT renderer | canonical/encoded/model/capability/renderer 一致性正负测试 | N/A | 待真实 tokenizer/runtime | PASS |
| EngineState schema | UNKNOWN 与 0、per-DP、ratio、histogram 负向测试 | N/A | 待 State Stream | PASS |
| Adapter registry | ownership、lookup、重复 key 与非法 Descriptor | N/A | N/A | PASS |
| 生产 RequestCodec | Native 精确计数/renderer、vLLM 原始 JSON/UNKNOWN 计数、错误 Provider/schema/空 renderer 负向测试 | N/A，无 tensor 逻辑 | 待真实 tokenizer/runtime | PASS |
| Provider route 隔离 | Native P/D、vLLM SINGLE、跨 Provider、无完整 plan、suspect、未知 Provider、RR cursor 正负测试 | N/A，无 tensor 逻辑 | 待真实混合池 | PASS，8/8 |
| STRICT P/D 兼容 | 不同 profile 正向；model、KV/Connector、topology、runtime 与 strict/legacy 单边混配负向测试 | N/A，无 tensor 逻辑 | 待真实 P/D handshake | PASS |

Service 的 `ProviderContractTest` 当前为 26 项，`InstanceMetaInfoTest` 为 13 项；
当前全量 service CPU 回归为 226/226，vLLM sidecar CPU 回归为 29/29（其中
metadata 11/11），xLLM CPU 公共路径基线为 96/96。

## 完善情况

- 已完成：contract v1 单一 proto；稳定错误码；完整 V2 mode/capability Resolver；
  Provider/open-mode 门禁；Descriptor、Canonical/Encoded Request、EngineState、
  ExecutionPlan 校验；`attempt_seq=0` 与缺失字段可区分；线程安全、只增不删的
  Adapter registry；两个首发 Provider 的 RequestCodec 与 dispatch identity；CPU
  conformance；生产 Scheduler 的 per-request Provider dispatch、跨 Provider route
  隔离与 STRICT P/D 显式兼容矩阵。
- 已知缺口/风险：当前 schema 尚无可校验的 topology-transform proof，因此非相同
  topology 保守拒绝；per-pair `LinkState=READY`、失败隔离和周期对账属于 G3。
  CanonicalRequest、ExecutionPlan 与 Adapter codec 的端到端接入仍属于后续 B1
  批次；G3 的
  `(provider_id, profile_digest, incarnation_id)` Engine Registry 尚未实现；当前
  不证明硬件 Runtime 行为。
- 回滚与兼容：协议为全新 additive schema；旧二进制不会读取这些消息。V2
  调用方必须对 `contract_version != 1`、未知 enum 与缺能力稳定 fail closed。
- 性能、容量和观测证据：纯 CPU 校验路径，无生产吞吐结论；State Stream 与
  观测基线由 G0/G3 批次交付。
- 达到 VERIFIED 仍需完成：两个真实 Adapter conformance、Scheduler 只依赖
  CanonicalRequest/ExecutionPlan、Engine Registry/State Stream、NPU smoke 与故障门禁。
