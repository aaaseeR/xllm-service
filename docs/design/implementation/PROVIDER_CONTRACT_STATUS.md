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
- 最近验证基线：xLLM `446bae12`、xllm-service 本状态文档所在提交
- 验证环境和日期：xllm-dev-sandbox，Ubuntu 24.04 ARM64，2026-08-09

## 支持范围

| Provider | Mode | Model/Profile | 支持状态 | 限制与证据 |
| --- | --- | --- | --- | --- |
| xLLM Native | `REMOTE_PD/LAYERWISE_PUSH/P_FIRST` | contract v1 | CPU_VERIFIED | Descriptor、capability、plan role shape 与 deadline 门禁通过 |
| xLLM Native | `LOCAL_PREFILL_DECODE/NONE/D_ONLY` | contract v1 | CPU_VERIFIED | mixed accounting 与 structured admission 是硬能力 |
| xLLM Native | `PREFILL_ONLY/NONE/P_ONLY` | contract v1 | CPU_VERIFIED | 独立 P-only capability 与单 P role shape 是硬门禁 |
| vLLM-Ascend | `AGGREGATED/NONE/SINGLE` | contract v1 | CPU_VERIFIED | 严格 Agent 已发布完整 Descriptor，并完成唯一 ingress、attempt/deadline/fencing CPU loopback；真实 NPU 待验证 |
| vLLM-Ascend | `REMOTE_PD/LAYERWISE_PUSH/D_FIRST` | contract v1 | BLOCKED | schema 可表达，V2 固定返回 `MODE_NOT_OPEN` |

## 实现

- 代码入口与核心接口：`xllm_service/provider/provider_contract.*`、
  `canonical_request_builder.*`、`provider_adapter.*`、
  `execution_plan_builder.*`、`provider_registry.*`。
- 已实现可注册的 `XllmNativeAdapter` 与 `VllmAscendAdapter`。Native Adapter
  通过显式 renderer 接口取得原生 payload、精确 token 数和实际 renderer
  digest；vLLM-Ascend Adapter 只接受 `openai.http.json.v1`，保留原始 JSON
  payload，并在 Provider 未返回计数前如实发布 `TOKEN_COUNT_QUALITY_UNKNOWN`。
  Registry 在写入前同时校验 Descriptor 和 Adapter dispatch kind，阻断
  Provider 身份与数据面协议错配。
- 状态、资源和错误码权威位置：wire 类型和稳定
  `ProviderContractError` 只定义在 xLLM `xllm/proto/provider.proto`；Service
  不复制 enum。Mode/capability 对应关系只定义在公共 Resolver。
- 多硬件边界已固定为 Engine/Provider 暴露、Service 消费标准化事实。Service 只读取
  Provider/runtime/profile、capability、KV/Connector compatibility、topology、
  admission 和统一 Engine/Link 状态；不调用 CANN/CUDA，不持有 device pointer、真实
  HBM 地址或具体 stream/event/allocator。`soc` 和硬件 Runtime 版本仅参与已验证
  profile 的兼容与发布门禁，不作为硬件特例分支键。
- 真实与 simulated HBM allocator 均位于 Engine/backend 或 fake Provider 边界。
  Service 通过相同 Provider Contract 验证 reservation、execution hold、transfer、
  cancel/fencing 和回收，不在生产 Service 内复制 KV/HBM allocator。B6 实现和 CPU
  证据见 [SIMULATED_HBM_STATUS.md](./SIMULATED_HBM_STATUS.md)。
- 跨仓协议与依赖：xLLM 是 `provider.proto` 的唯一源；xllm-service 通过
  `proto_xllm` 直接生成和链接同一文件。`renderer_digest` 覆盖 tokenizer +
  template 渲染契约，STRICT 编码结果必须与 Descriptor 相等。
- Scheduler 生产路径持有按 `(provider_id, profile_digest)` 索引的 append-only
  Adapter registry。Adapter 对象不可变且可被并发请求复用；token ids、request UID、
  attempt 和 renderer digest 只通过同步 `RequestEncodingContext` 传入，不被 Adapter
  持有。同 key 并发懒注册会合并到同一对象；相同 profile digest 对应不同 Descriptor
  时 fail closed。这是 Provider/profile 级 Adapter 缓存，不替代 G3 的 Engine
  incarnation Registry。
- `InstanceMetaInfo` 已携带 `provider_id`、contract version、profile digest 与可选
  完整 `ProviderDescriptor`。完整 Descriptor 在 JSON 入口执行摘要/incarnation
  交叉校验，并在实例进入生产索引前运行公共 Descriptor validator；旧
  `backend_type` 只在 JSON 兼容入口映射一次，未知值或双身份冲突 fail closed。
  vLLM strict Agent 发布 contract version 1；未提供 verified profile 的 legacy sidecar
  明确发布 contract version 0，继续属于 BEST_EFFORT 兼容桥。
- Scheduler 已按每个请求从可执行 Provider route 中选择 Adapter dispatch kind，
  不再读取进程级 `default_backend_type` 决定 tokenization 或数据面；该 CLI flag
  仅为旧部署参数兼容保留且不参与运行时决策。Round-robin、CAR、SLO 指标入口、
  静态 peer 列表和 Engine Link/Unlink 均按 Provider 隔离。Native `P+D` 与 vLLM
  `AGGREGATED` route shape 由同一个纯 CPU selector 决定，禁止跨 Provider 拼接。
- HTTP ingress 已在解析完成后构造不可变 `CanonicalRequest`，保留 correlation、API
  kind、model revision、原始 schema/payload、有效输出上限、`n/best_of`、priority、
  deadline 与可选 SLO。Scheduler 绑定 P/D Descriptor 快照后调用生产
  `RequestCodec` 并生成 `ExecutionPlan`；retry 会递增 `attempt_seq`、重算剩余
  deadline 并重新编码/建计划，禁止 STRICT 请求降级到 legacy route。
- Native renderer 复用 ingress 已完成的 tokenization，发布精确 token 数，并以
  `--native_renderer_digest` 提供的独立部署摘要与 Descriptor 交叉校验；摘要为空时
  STRICT Native fail closed。vLLM relay 在 STRICT 路径实际发送
  `ExecutionPlan.provider_payload`，prompt token 未知时 KV block 估算保持未知值 0，
  并记录 `prompt-token-count-unknown`，不以低估值做容量承诺。
- Native dispatch 已将 Scheduler 持有的 attempt-scoped `ExecutionPlan` 写入
  Completion/Chat additive wire。每次初始派发或首输出前重试都会重新复制当前 attempt
  的计划和剩余 deadline；HTTP 客户端写入的同名字段会先被清除，legacy route 在没有
  Scheduler 计划时继续保持字段缺失。xLLM Engine 在实际接收点用本机 UID/incarnation
  校验 request identity、attempt、P/D role 与 routing、deadline、capability、资源估算、
  compatibility proof 和 provider payload，STRICT 计划不能投递到错误或已重启的 P。
- STRICT Native P/D 使用显式兼容矩阵，不要求两端 `profile_digest` 相等，但要求
  runtime/protocol、model revision/quantization、KV layout/dtype/block/cache group、
  Connector/version、layerwise transfer 和可证明 topology 一致，并生成确定性的
  compatibility proof。STRICT 与 BEST_EFFORT legacy 不能单边混配；矩阵同时用于
  route、readiness、静态 peer 与 Link/Unlink 候选。
- 明确不支持范围：Descriptor-less Engine 仍作为显式 BEST_EFFORT 兼容入口，不生成
  V2 artifacts。State Stream 发布/接收、切主 identity 分发、Engine/Link 状态门禁和
  失明状态机已完成 CPU 闭环；不声称任何真实 NPU Provider 已通过 conformance。

## 需求与测试追踪

| Requirement ID | CPU test | Torch CPU test | NPU test | 结果 |
| --- | --- | --- | --- | --- |
| G-1/F66 单一 wire 真相 | xLLM `ProviderProtocolTest` golden wire、roundtrip、optional presence、field number | N/A，无 tensor 逻辑 | N/A | PASS，9/9（含 G1 additive resource wire、StateBatch 与关键字段号） |
| G-2/F80 mode/capability | `ResolvesEveryCompleteV2CapabilityRow`、未知 mode/transfer 负向测试 | N/A | 待真实 Provider | PASS |
| G-2/F81 Descriptor | 四种开放 profile、缺能力、重复项、runtime alias 负向测试 | N/A | 待真实 Provider | PASS |
| G-2/F82 ExecutionPlan | 四种 role shape、capability、deadline、identity 门禁 | N/A | 待真实 Provider | PASS |
| STRICT renderer | canonical/encoded/model/capability/renderer 一致性正负测试 | N/A | 待真实 tokenizer/runtime | PASS |
| EngineState schema | UNKNOWN 与 0、per-DP、ratio、stale/state-quality 负向测试；connector tag 12/name reserved | N/A | 待真实 State Stream | PASS；固定 `READY` 占位已删除，Remote PD connector 门禁由 incarnation-scoped `LinkState=READY` 独立表达 |
| 多硬件感知边界 | Descriptor/capability/profile、STRICT KV/Connector/topology compatibility 和跨 Provider 隔离测试 | N/A，无设备 API | 待 NPU 首发及后续 backend conformance | PASS（公共契约）；真实 backend 逐 profile 待验证 |
| Adapter registry | ownership、lookup、重复 key、非法 Descriptor、生产 factory、同 key 并发懒注册合并、profile digest 碰撞拒绝、跨 attempt request context 复用；缓存关键三项重复 100 轮 | N/A | N/A | PASS |
| 生产 RequestCodec | Native 精确计数/renderer、vLLM 原始 JSON/UNKNOWN 计数、错误 Provider/schema/空 renderer 负向测试 | N/A，无 tensor 逻辑 | 待真实 tokenizer/runtime | PASS |
| Canonical/Plan 生产接入 | ingress 语义保留、Native REMOTE_PD 与 vLLM AGGREGATED plan、renderer identity/digest、UNKNOWN KV estimate | N/A，无 tensor 逻辑 | 待真实 Provider wire | PASS，4 项新增测试 |
| Native plan dispatch/Engine ingress | Completion/Chat wire presence/roundtrip/field number；严格计划复制、客户端计划清除、legacy 缺失兼容、缺 retry policy 拒绝；Engine identity/attempt/route/receiver/deadline/capability/resource 正负校验 | RequestParams 与两个 API handler 在 Torch CPU 头文件环境严格编译；逻辑不执行 tensor 数值计算 | 待真实 P/D 数据面 | PASS，Service 3/3，Engine validator 6/6 |
| Provider route 隔离 | Native P/D、vLLM SINGLE、跨 Provider、无完整 plan、suspect、未知 Provider、RR cursor 正负测试 | N/A，无 tensor 逻辑 | 待真实混合池 | PASS，8/8 |
| STRICT P/D 兼容 | 不同 profile 正向；model、KV/Connector、topology、runtime 与 strict/legacy 单边混配负向测试 | N/A，无 tensor 逻辑 | 待真实 P/D handshake | PASS |

Service 的 Provider/Registry 纯 CPU 测试已覆盖 Adapter、route、EngineState 和
LinkState；当前全量 service CPU 回归在 pinned 与外部 xLLM 两种构建下均为
380/380，vLLM Agent/sidecar CPU 回归为 60/60，xLLM CPU 公共路径基线为
118/118（含 simulated HBM 12/12、Provider protocol 9/9）。新增 xLLM Engine plan
validator 为 6/6，相关 protocol allowlist 通过；RequestParams、Completion 与 Chat
生产对象均在 Torch CPU 头文件环境以 `-Werror` 编译通过。完整 RequestParams target
仍受既有 CPU sandbox `ProcessGroupImpl` 不完整类型阻塞，该限制不来自本批变更。

## 完善情况

- 已完成：contract v1 单一 proto；稳定错误码；完整 V2 mode/capability Resolver；
  Provider/open-mode 门禁；Descriptor、Canonical/Encoded Request、EngineState、
  ExecutionPlan 校验；`attempt_seq=0` 与缺失字段可区分；线程安全、只增不删且进入
  Scheduler 生产路径的 Adapter registry；两个首发 Provider 的不可变 RequestCodec、
  request-scoped encoding context 与 dispatch identity；CPU
  conformance；生产 Scheduler 的 per-request Provider dispatch、跨 Provider route
  隔离、STRICT P/D 显式兼容矩阵，以及真实 ingress→CanonicalRequest→RequestCodec→
  ExecutionPlan 构建；vLLM STRICT 数据面已消费 plan payload；Native Completion/Chat
  已携带当前 attempt 计划，Engine 在实际入口 fail closed 校验计划与本机身份。
- 已知缺口/风险：当前 schema 尚无可校验的 topology-transform proof，因此非相同
  topology 保守拒绝；per-pair `LinkState=READY` 的接收和路由门禁已完成，失败隔离、
  publisher 与周期对账已完成 CPU 接线，真实多机切主与 Link 故障注入仍待集群验证。
  动态吞吐、延迟直方图、失败计数和计划预测字段在没有校准 producer 与生产决策
  consumer 前已从 schema 删除并 reserved；当前不声称动态性能排序能力。
  客户端 model alias 到权威 model revision 的映射需由 catalog 明确，当前 STRICT
  路径按字符串完全一致 fail closed。G3 的
  Engine Registry/State cache 已实现且进入候选过滤，但当前不证明硬件 Runtime 行为。
- 回滚与兼容：协议为全新 additive schema；旧二进制不会读取这些消息。V2
  调用方必须对 `contract_version != 1`、未知 enum 与缺能力稳定 fail closed。
- 性能、容量和观测证据：纯 CPU 校验路径，无生产吞吐结论；State Stream 与
  观测基线由 G0/G3 批次交付。
- 达到 VERIFIED 仍需完成：两个真实 Adapter conformance、Scheduler 只依赖
  CanonicalRequest/ExecutionPlan、Engine Registry/State Stream、NPU smoke 与故障门禁；
  后续新增硬件必须复用同一 Contract/Resolver 并按 profile 完成 conformance，不得在
  Service 中增加芯片专属调度路径。
