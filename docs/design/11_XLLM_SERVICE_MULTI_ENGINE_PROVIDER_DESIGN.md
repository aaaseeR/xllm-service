# xLLM Service 多引擎 Provider 与 Adapter 设计

## 1. 文档定位

- 状态：V2 首发的多引擎接入与能力门禁基线
- 日期：2026-08-07
- 代码基线：xllm-service `322bcda03793`、xLLM `8164a701bab7`、vLLM-Ascend `ba58907c6d1c`
- 首批 Provider：xLLM Native、vLLM-Ascend
- 开发与交付门禁：[V2 代码开发与交付规范](./00_XLLM_SERVICE_V2_DEVELOPMENT_STANDARD.md)

本文原有 V1 表述统一解释为 V2-B0 基础能力，不对应独立 V1 产品版本；
Provider 基础门必须与 08/09 的 V2 路由、流控和执行模式一起交付。

本文解决一个具体问题：同一个 xLLM Service 如何同时纳管不同 Runtime，而不把两套内部协议强行做成一套，也不因能力差异破坏准入、SLO 和故障语义。

结论是：**Service 统一的是请求计划、状态语义和能力门禁，不是 Engine 内部实现。** 每种 Runtime 通过 Provider Adapter 接入；资源与数据真相仍由对应 Engine 持有。

## 2. 代码事实与核心结论

### 2.1 两套 Engine 不是同一种接入方式

| 维度 | xLLM Native | vLLM-Ascend |
| --- | --- | --- |
| Runtime 形态 | 独立 xLLM Engine | 上游 vLLM Runtime 的 Ascend 平台插件 |
| 主要数据面 | brpc/protobuf；`AddNewRequests`、`FirstGeneration`、`MultiGenerations` | OpenAI HTTP/SSE；复用 vLLM EngineCore、Scheduler 和 API Server |
| 注册与状态 | Engine 原生注册、heartbeat、xllm-service Registry | 当前 Python sidecar 注册，抓取 `/health`、`/metrics` |
| P/D 方式 | 原生 P→D 逐层 PUSH 与 link RPC | 外部 Proxy 编排；PULL 或 layerwise PUSH，通过 `kv_transfer_params` 交接 |
| KV Connector | xLLM 自有传输协议 | Mooncake、AscendStore、UCM、LMCache、CPU offload 等 vLLM Connector |
| 原子 reservation | 当前协议需按 V2-B0 补齐 | vLLM 内部会分配 KV，但没有对 xLLM Service 暴露等价的硬 reservation/query/fence 协议 |
| 公共可观测性 | heartbeat 与自有事件，需补齐阶段事件 | vLLM 指标已有 running/waiting、KV、TTFT、ITL/TPOT、queue 等；当前 sidecar 只消费少量聚合值 |
| 健康语义 | 需实现 incarnation 与 self-fencing | `/health` 可发现 EngineDeadError；Ascend worker 的 `check_health()` 会吞掉 `npu-smi` 非 OK 异常并无条件返回，不能向调用方提供深层健康失败信号 |

vLLM-Ascend 通过 `vllm.platform_plugins` 注册 `NPUPlatform`，并通过通用插件注册 KV Connector、Model Loader 和模型实现。因此一个可调度实例不能只标记为“vLLM”或“Ascend”，而必须描述完整的 `vLLM Runtime + vLLM-Ascend 插件 + CANN/驱动 + 硬件 + 执行配置` 组合。

### 2.2 当前 xllm-service 的 vLLM 支持只是兼容桥

当前代码已经支持 vLLM HTTP 中继，但还不能安全地组成混合 Provider 动态池：

1. `Scheduler` 按进程级 `default_backend_type` 决定是否 tokenize，而不是按本次请求选中的 Provider 编解码。
2. 多处代码直接判断 `backend_type == "vllm"`；`InstanceMetaInfo` 没有 Provider 版本、模型 revision、拓扑、KV layout、Connector 和能力集合。
3. sidecar 将 Prometheus 指标按基础名称直接求和。`num_requests_waiting` 可以求和，但各 DP 的 `kv_cache_usage_perc` 求和会失真甚至超过 1；延迟平均值也被写入名为 `recent_max_*` 的字段。
4. sidecar lease 只证明 sidecar 存活，不证明 vLLM 进程、NPU 或本地数据入口已经自我隔离。lease 丢失后，原始 vLLM 端口仍可能继续接单和计算。
5. 当前路径没有把 vLLM-Ascend P/D 的 `kv_transfer_params`、Connector 状态和失败语义映射为 xLLM Service 的 reservation、attempt、deadline 与 GenerationCommit。

因此不能在现有 `backend_type` 分支上继续叠加特例。当前 relay 保留为兼容路径；升级后的 Provider Agent 达到对应能力门禁后，才能进入生产调度池。

### 2.3 三条固定原则

1. **公共 SPI 统一语义，不统一 wire。** xLLM Adapter 可以调用 brpc；vLLM-Ascend Adapter 可以代理 HTTP、读取指标并编排 Connector。
2. **按能力生成执行计划，不假设所有 Provider 等价。** 聚合推理、PULL P/D、layerwise PUSH 的绑定时机与提交屏障不同。
3. **未证明的组合 fail closed。** 两端都写着 Mooncake，不代表 KV layout、Connector 版本、TP 转换和握手元数据兼容；跨 Provider P/D 默认禁止。

## 3. 目标架构

```text
Client / Gateway
       |
       v
xLLM Service
  - CanonicalRequest / SchedulingContext
  - Registry + State Stream
  - Compatibility Resolver
  - ExecutionPlan Selector
       |
       +-- XllmNativeAdapter --------> xLLM P / D Engine
       |
       +-- VllmAscendAdapter/Agent --> vLLM API Server + Ascend Plugin
```

Service 不直接理解 Scheduler 类、KV Connector 实现或设备 API。Adapter 负责把公共语义映射到 Provider 能力，并将 Provider 的状态转换为统一状态。Adapter 不伪造不存在的能力。

### 3.1 Provider Adapter SPI

公共 SPI 分为六组：

| 接口组 | 最小语义 |
| --- | --- |
| `Describe` | 返回不可变 Provider/Profile、协议版本和能力集合 |
| `PublishState` | 发布带序号的 lifecycle、健康、队列、KV、吞吐、延迟和 Connector 状态 |
| `Submit/Stream` | 按执行计划提交并返回有序事件；保留 Provider 原生协议 |
| `Cancel/QueryAttempt` | 取消、查询 attempt 终态；能力不足时必须显式返回 `UNSUPPORTED` |
| `Reserve/Transfer/Link` | 仅 P/D Provider 实现 Decode reservation、KV 交接和 pair readiness |
| `Drain/Fence` | 停止新工作、收敛在飞请求，并在 ownership 丢失时阻断旧 incarnation |

这是一组逻辑接口，不要求所有 Provider 采用同一个 RPC。Service 只依赖经过版本化的 Adapter Contract。

### 3.2 ProviderDescriptor

Descriptor 对一个 incarnation 不可变；任何影响兼容性或预测面的字段变化都必须以新 incarnation 注册。

```text
ProviderDescriptor = {
  identity: {
    engine_uid, incarnation_id,
    provider_id, runtime_family,
    runtime_version, plugin_version,
    hardware_runtime_version, protocol_version
  },
  endpoint: {control_transport, data_transport, address},
  serving: {
    role,
    execution_modes: [{mode, transfer_mode, selection_order, binding_stage}],
    api_features[]
  },
  model: {
    model_revision, tokenizer_revision,
    chat_template_digest, quantization
  },
  topology: {soc, device_count, tp, dp, pp, ep, cp},
  kv: {
    kv_layout_digest, cache_dtype, block_size,
    cache_groups, head_shard_mapping,
    connector, connector_version, transfer_modes[],
    storage?: {
      storage_kv_layout_digest, store_serialization_version,
      derivation?
    }
  },
  scheduler: {
    scheduler_class, max_num_seqs,
    max_num_batched_tokens, scheduler_policy_digest
  },
  capabilities[], profile_digest
}
```

首批 `provider_id` 使用 `XLLM_NATIVE` 和 `VLLM_ASCEND`。`runtime_family` 分别为 `xllm` 与 `vllm`。不能用 `provider_id=vllm` 掩盖 Ascend 插件和硬件 Runtime 版本。

`kv_layout_digest` 描述执行与传输侧布局，前像至少包含 cache dtype、block size、cache group、head/shard 映射和 Connector wire version，用于 P/D 与 topology transform 兼容判定。`storage_kv_layout_digest` 描述 Store 中实际落盘字节的布局与序列化版本；只有启用外部 Store 时才要求。两者都不进入 08 §4 的链式 block hash 前像。只有落盘布局完全由执行布局决定时，Provider 才可显式声明 `storage_kv_layout_digest = H(kv_layout_digest || store_serialization_version)`；不能隐含两者相等或可推导。

### 3.3 能力集合

能力是可测试的行为，不是产品名。首批至少包括：

```text
AGGREGATED
REMOTE_PD_PULL
REMOTE_PD_LAYERWISE_PUSH
NATIVE_RESERVATION
ATTEMPT_QUERY
CANCEL_FENCE
ENGINE_LOCAL_DEADLINE
STRUCTURED_ADMISSION
PER_DP_STATE
DEEP_HEALTH
SELF_FENCING
DRAIN
PREFIX_EVENTS
EXTERNAL_KV_STORE
KV_OFFLOAD
```

能力只在 conformance test 通过后发布。版本升级、Connector 变化或 scheduler 配置变化需要重新验证，不能继承旧结果。`EPD` 是未来执行模式而不是 capability；其绑定顺序和提交屏障定义完成前为 `NOT_REGISTRABLE`。

## 4. 统一请求与执行计划

### 4.1 CanonicalRequest 与 RequestCodec

当前 xLLM 路径由 Service tokenize，vLLM 路径把原始 OpenAI JSON 转交 Runtime。混合池不能再通过全局 backend 开关决定这一行为。

Service 保存一份 `CanonicalRequest`，包含原始 API 语义、模型、完整历史、采样参数、输出上限、SLO 和观测标识。每个 Provider 的 `RequestCodec` 负责：

- xLLM：使用已声明 tokenizer/template 生成 token 输入和原生 RPC；
- vLLM-Ascend：生成 OpenAI HTTP 请求并代理 SSE；
- 返回用于调度的 exact 或 bounded token 数，以及可验证的 renderer digest。

STRICT 请求只有在 Service 侧计数所用 tokenizer/template 与 Provider Descriptor 一致时才能进入该 Provider。Provider 特有且无法等价转换的参数必须固定路由，不能在 Provider 间静默降级。

### 4.2 ExecutionPlan

```text
ExecutionPlan = {
  provider_id,
  mode,
  transfer_mode,
  selection_order,
  selected_roles: [{role, engine_uid, incarnation_id, order_index}],
  p_selection_delegated,
  binding_stage,
  compatibility_proof,
  provider_payload,
  prediction,
  deadline_budget,
  reason_codes[]
}
```

绑定顺序由 `(provider_id, mode, transfer_mode)` 决定，而不是由 transfer mode 单独决定：

| Provider | mode / transfer | `selection_order` | `binding_stage` | 首 token 提交屏障 | V2 首发 |
| --- | --- | --- | --- | --- | --- |
| xLLM Native | `REMOTE_PD / LAYERWISE_PUSH` | `P_FIRST`：P + 有序 D 候选 | `BEFORE_PREFILL` | P 获得 D `FirstGeneration` ACK | 开放 |
| vLLM-Ascend | `AGGREGATED / NONE` | `SINGLE`：一个 Agent/Engine | `AT_SUBMIT` | Agent 原子接受完整请求并安装唯一 attempt | 门禁后开放 |
| vLLM-Ascend | `REMOTE_PD / LAYERWISE_PUSH` | `D_FIRST`：先 D，P 由 metaserver 选择 | `BEFORE_PREFILL` | D 完成 KV 预分配并接管 Decode | 关闭 |

`D_FIRST` 计划首项必须是 D，并置 `p_selection_delegated=true`；Adapter 在 attempt status 中回填实际 P。该模式会失去 Service 对 P 的 KV-aware 选择能力，开放前必须显式接受这一代价或补充 metaserver 选择接口。V2 首发不执行 `D_FIRST`，但 Provider Contract 和 conformance test 必须能往返表达并稳定拒绝未开放 profile。`EPD` 不进入本表，直至其绑定和提交屏障有可测试定义。

### 4.3 Mode 与能力矩阵

STRICT Resolver 固定执行 `required_capabilities(mode) ⊆ published_capabilities`。未知 mode、未知 capability、矩阵未分类或要求项缺失都 fail closed。

| mode | 必需 capability | 可选 capability | 不适用 | 结果不明时的执行资源持有 |
| --- | --- | --- | --- | --- |
| `AGGREGATED` | `AGGREGATED`、`ATTEMPT_QUERY`、`CANCEL_FENCE`、`ENGINE_LOCAL_DEADLINE`、`SELF_FENCING`、`DRAIN` | `STRUCTURED_ADMISSION`、`PER_DP_STATE`、`DEEP_HEALTH`、`PREFIX_EVENTS`、`EXTERNAL_KV_STORE`、`KV_OFFLOAD` | `NATIVE_RESERVATION` | 整个 aggregated execution attempt |
| `REMOTE_PD / LAYERWISE_PUSH` | `REMOTE_PD_LAYERWISE_PUSH`、`NATIVE_RESERVATION`、`ATTEMPT_QUERY`、`CANCEL_FENCE`、`ENGINE_LOCAL_DEADLINE`、`SELF_FENCING`、`DRAIN` | 同上 | `AGGREGATED` | remote D reservation |
| `REMOTE_PD / PULL` | `REMOTE_PD_PULL`、`NATIVE_RESERVATION`、`ATTEMPT_QUERY`、`CANCEL_FENCE`、`ENGINE_LOCAL_DEADLINE`、`SELF_FENCING`、`DRAIN` | 同上 | `AGGREGATED` | remote D reservation |

可选观测能力缺失时，对应字段为 `UNKNOWN` 并退出相关硬预测，不能按 0 参与评分。`DEEP_HEALTH` 是否为某个硬件/上线 profile 的额外发布前置，由该 profile 的 release policy 明确；vLLM-Ascend 当前上游接口不能发布该能力，见 §7.2。xLLM Native 的 `LOCAL_PREFILL_DECODE/PREFILL_ONLY` 属于 09 定义的 V2 首发模式；开放前必须以同样规则补全矩阵，这是 V2 交付前置，不能在 Adapter 内写散落特例。

## 5. 状态、指标与容量语义

### 5.1 EngineState

```text
EngineState = {
  incarnation_id, state_seq, observed_at,
  lifecycle, ownership, shallow_health, deep_health,
  per_dp: [{dp_rank, running, waiting_capacity, waiting_deferred,
            kv_used_ratio, kv_free_blocks, admission_credit}],
  latency_histogram_delta, throughput,
  connector_state, failure_counters,
  state_quality
}
```

缺失值是 `UNKNOWN`，不能写成 0。State Stream 仍是软观测；最终容量由 Provider 本地准入决定。

### 5.2 聚合规则

| 指标 | 正确处理 |
| --- | --- |
| running/waiting | 保留 per-DP；集群总量可求和 |
| TP rank KV headroom | 取同一 DP 内最小余量，不能取平均 |
| DP KV usage | 保留每个 DP；需要单值时取选定 DP 值或最坏值，不能求和 |
| counter | 计算带时间窗的 delta/rate 后求和 |
| histogram | 合并 bucket delta 后求分位数，不能平均 p95 或把均值命名为 max |
| health | shallow/deep 分开；任一关键 rank 不健康即该执行单元不可调度 |

vLLM-Ascend Adapter 至少采集 `num_requests_running`、按原因拆分的 waiting、`kv_cache_usage_perc`、TTFT、queue、ITL/TPOT、E2E、preemption 和 prefix cache 指标，并保留 `model_name/engine/DP` 等标签。当前 sidecar 的无标签求和实现只能用于兼容观测，不能用于硬准入或 M1/M2 预测。

## 6. P/D 与 KV 兼容矩阵

P/D compatibility key 为：

```text
(P provider/profile,
 D provider/profile,
 model_revision,
 kv_layout_digest,
 connector_protocol/version,
 transfer_mode,
 topology_transform)
```

兼容矩阵由离线 conformance test 产生并版本化。测试至少覆盖 KV 数值正确性、首 token 一致性、长上下文、边界 block、并发、取消、传输超时、P/D 重启以及允许的 TP 变换。

vLLM-Ascend 当前实现已经表明 Connector 兼容不仅是名称：Mooncake metadata 还包含 engine ID、layer/group 映射、block size、cache 地址/stride/length 和握手信息；部分 P/D 模式要求 `P_TP >= D_TP` 且 `P_TP % D_TP == 0`，混合模型还有更严格限制，异构 A2/A3 P/D 也不是默认支持项。

因此 V2-B0 采用以下规则：

- xLLM P ↔ xLLM D：只开放已验证 profile 组合；
- vLLM-Ascend P ↔ vLLM-Ascend D：由独立 Adapter/Connector 门禁开放；
- xLLM P ↔ vLLM-Ascend D，或反向：默认禁止；只有完整 compatibility key 通过专门转换与故障测试后才能增加，不因同为 Mooncake 自动放行；
- 聚合 Engine 可以在请求提交前参与同模型 Provider 间选择，但首 token 后不跨 Provider 迁移。

## 7. Provider 故障与 fencing

### 7.1 公共要求

Provider 进入生产 STRICT 池必须满足：

1. incarnation、Registry lease、State Stream 和数据面请求使用同一身份；
2. Service 从权威 Registry DELETE/revoke 观测到成员资格丧失或 incarnation 变化后，下一次选择立即排除旧 incarnation；宽限只用于收敛在飞请求，heartbeat、探活或 watch 恢复都不能恢复成员资格；
3. watch 断连、list 歧义或探活失败不等价于成员丧失；Registry 可见性不足时进入 `REGISTRY_BLIND` 并冻结视图，不把旧成员误判为 DELETE；
4. ownership 不确定时 Provider 停止新准入；确认丢失时旧 incarnation 不再接收新请求或输出；
5. 提交 outcome 不明时，Query/cancel/fence 或硬时间证明收敛前不得向另一 Provider 创建新 attempt；
6. deadline 能在 Engine/Agent 本地停止工作，不能只依赖 Service 连接断开；
7. Cancel 有稳定幂等语义，QueryAttempt 能区分运行中、终态、未知和不支持；
8. Agent/Engine 崩溃后由部署系统拉起新 incarnation，Service 不尝试恢复旧 Decode 状态。

### 7.2 vLLM-Ascend Provider Agent

当前 sidecar 只做注册和指标抓取，不足以提供上述 fencing。目标形态中，注册地址必须指向同机 Provider Agent，而不是可绕过的原始 vLLM 端口。Agent 同时承担：

- 代理 OpenAI HTTP/SSE、注入 request/attempt/deadline 元数据；
- 注册完整 Descriptor，采集带标签状态和深层设备健康；
- ownership 丢失时关闭本地 ingress，并取消、drain 或终止受控 vLLM 进程；
- 将断连、abort、EngineDeadError、Connector 失败映射为稳定终态；
- 可选编排 vLLM-Ascend P/D 和 `kv_transfer_params`。

发布 `SELF_FENCING` 必须同时证明两项部署前置：原始 vLLM 端口只允许 Agent/本机访问；Agent 与受控 vLLM 处于同一失效域。V2 首发只接受以下两种同命实现：Agent 作为父进程并在自身死亡时由进程监管机制终止整个 vLLM 进程组，或二者位于同一重启单元且 Agent 退出会重启整个单元。需要 vLLM 自己持有 Agent lease 的第三种方案依赖上游改造，不属于 V2 首发。

`agent_fate_bound` 是从 Agent 消失到 vLLM 停止接单、终止在飞请求并释放资源的硬上界，必须早于 Service 可能创建替代 attempt 的最早时刻。Agent 单独 `SIGKILL` 的负向测试若不能在该上界内关闭原始入口并中止在飞请求，该部署只能作为 BEST_EFFORT。

当前 vLLM-Ascend `check_health()` 会捕获 `parse_text_output()` 抛出的非 OK 异常并无条件返回 `None`，因此上游接口无法提供 `DEEP_HEALTH`。Agent 只能直接调用并验证 `npu-smi`，或等待上游修复；两条路径均未交付前不得发布该 capability。

## 8. 首批 Provider 落地范围

| 能力 | xLLM Native V2 首发 | vLLM-Ascend V2 首发 | 后续 |
| --- | --- | --- | --- |
| 统一注册、Descriptor、State | 必须 | 必须 | 扩展更多 Provider |
| 聚合推理 | 可保留默认/兼容模式 | 首个生产接入模式 | 按请求在已验证 Provider 间选择 |
| 动态远程 P/D | 首个主路径 | 不默认开放 | Provider PD Adapter 通过独立门禁后开放 |
| 原子 reservation/query/fence | Engine 原生补齐 | Agent/Runtime 扩展后开放 | 统一 conformance suite |
| 本地 deadline | Engine 原生补齐 | Agent 必须下传并强制终止 | Runtime 原生支持优先 |
| Prefix/KV event | 后续接入 | Connector 有能力时接入 | 统一 KVIndex namespace |
| 跨 Provider P/D | 禁止 | 禁止 | 只对经过转换和故障测试的组合开放 |

“同时支持两种 Engine”不等于首版要求所有执行模式功能对称。V2 的共同底座必须支持两种 Provider 的发现、状态、请求、取消、观测和能力门禁；xLLM Native 交付严格远程 P/D，并按 09 开放本地模式，vLLM-Ascend 交付严格聚合模式。任何未满足的能力通过门禁排除，不能靠调度器猜测补偿。

## 9. 代码改造顺序

1. **Provider Contract：** 扩展 `InstanceMetaInfo`，增加 Descriptor、capabilities、profile digest 和协议版本；以 `(provider_id, profile_digest, incarnation_id)` 建立 Registry 索引。
2. **统一 Adapter：** 把 `backend_type == "vllm"` 分支迁移到 `XllmNativeAdapter` 与 `VllmAscendAdapter`，调度器只处理 `CanonicalRequest` 和 `ExecutionPlan`。
3. **状态修复：** State Stream 保留 per-DP/rank 标签，按第 5 节聚合；修复 sidecar KV 与延迟指标语义。
4. **vLLM-Ascend Agent：** 让 Agent 成为注册和数据入口，补齐同命部署、ingress fencing、deadline、cancel、独立深层健康和稳定错误映射。
5. **能力门禁：** 建立 Provider conformance、API correctness、KV/P-D compatibility 和故障注入测试；灰度只开放通过的 mode/profile。
6. **PD 扩展：** 在共同底座稳定后，再接入 vLLM-Ascend PULL/layerwise PUSH 编排，不修改公共调度接口。

## 10. 上线门禁

- 同一 Service 可同时发现 xLLM Native 和 vLLM-Ascend，选择结果不依赖进程级 `default_backend_type`。
- Descriptor 不完整、版本未知、能力不足或 profile 未验证时 fail closed；兼容 relay 只能进入显式 BEST_EFFORT bucket。
- tokenizer/template/model revision 不一致时不跨 Provider 调度 STRICT 请求。
- 指标缺失或标签异常时标记 `UNKNOWN`；KV ratio 不求和，histogram 不平均分位数。
- ownership 丢失测试中，旧 Agent/Engine 在 fencing deadline 内停止新接单和旧 incarnation 输出。
- Registry DELETE 或 incarnation 变化后，下一次选择零计划指向旧 incarnation；全池 Registry 失明走 `REGISTRY_BLIND`，不能用探活/heartbeat 恢复旧成员。
- 仅 `SIGKILL` Agent 时，受控 vLLM 在 `agent_fate_bound` 内停止接单、中止在飞请求并释放 KV；否则不得发布 `SELF_FENCING`。
- `AGGREGATED` Submit 已接受但响应超时时，Query/cancel/fence 收敛前不得并发提交替代 attempt；一万次故障注入后无 slot/KV 泄漏。
- mode/capability 矩阵逐行通过 Resolver conformance；`D_FIRST` schema 可往返但 V2 首发 profile 稳定拒绝，`EPD` 不可注册。
- 仅 `kv_layout_digest` 不同必须拒绝 P/D；仅 `storage_kv_layout_digest` 不同必须生成不同 Store key，而 08 §4 的 Router block hash 保持相同。
- xLLM Native 与 vLLM-Ascend 分别通过 API/SSE、取消、deadline、崩溃、drain 和状态陈旧测试。
- vLLM-Ascend 远程 P/D 未通过 reservation、KV 正确性、首 token commit、取消和重启测试前，仅开放聚合模式。
- 跨 Provider P/D 保持关闭，并有负向测试证明无法被同名 Connector 或错误 capability 绕过。
