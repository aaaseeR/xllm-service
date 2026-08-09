# V2-B8 K1 Canonical Hash 与 Namespace 状态

更新时间：2026-08-09
状态：`CPU_VERIFIED / NPU_AND_CLUSTER_PENDING`

## 已完成范围

- xLLM Native `KVDescriptor` 发布不可变 `kv_namespace/hash_version/hash_seed`；
  namespace 由 model/tokenizer/template/renderer、block/hash contract、cache dtype/
  quantization、Runtime cache semantics、cache group 和 isolation domain 的长度分隔
  契约生成。TP/DP/PP/EP/CP、rank、地址和执行侧 `kv_layout_digest` 不进入 namespace。
- hash 前像固定为 `xkvh-v1 || len(namespace) || namespace || parent-present ||
  parent || token-count || tokens-u32-le || len(block-extra) || block-extra`，输出固定为
  `low64-le || high64-le`。Service 使用候选 descriptor 的 seed，不再依赖本地 seed
  恰好相同。
- Native request 将 namespace 从 Service 透传到 P，再由 `DisaggRequest` 原样透传到
  D；Sequence/PrefixCache 的实际 block hash 使用同一个 canonical helper。旧直连请求
  不携带 namespace，继续留在 Engine-local legacy hash domain。
- P/D 的 namespace、hash version 或 seed 任一不一致均在 Service 兼容矩阵和最终
  incarnation binding 中 fail closed。Native Strict ExecutionPlan 缺 namespace 不派发。
- Engine KV journal identity 改为 descriptor namespace，descriptor cache group 与真实
  BlockManager event 的 `group:<cache_group_id>` 完全一致。
- K1 只在全部 hard-filter 后候选具有相同 block size、namespace 和 seed 时计算一次
  block hash；缺失或不一致自动回退 load-only。
- Native 在 D admission 响应中读取 KV group 的 `remote_shared_num`，按实际 block size
  形成带 presence 的 `num_decode_cached_tokens`。该字段随首输出进入 Service；未提供、
  有效 0 和非 0 分别保持 MISSING、VALID_ZERO、VALID_NONZERO，P/D 观测不互相冒充。
- Service 同时对账 P actual hit、D actual hit、有效 Prefill token 和逻辑跳过传输字节；
  bytes-per-token 缺失或乘法溢出时保持 MISSING，不把未知值记成 0。
- `ENFORCED` 需要 `kv_route_enforced_gate_open=true` 且稳定 workload bucket 非 0；
  bucket 使用请求稳定 hash 的万分位，默认 gate 关闭、bucket 为 0，因此即使误配
  `kv_route_mode=ENFORCED` 也仍按 SHADOW 执行。UNKNOWN/recovery/OOD 继续 fail closed。

## CPU 证据

环境：`xllm-dev-sandbox` Ubuntu 24.04 ARM64，Clang 18，PyTorch CPU。

| 范围 | 结果 |
| --- | --- |
| xLLM 默认 CPU gate | 8 个目标、115 项 PASS |
| Engine canonical golden | 1/1 PASS；两级 chained hash 与 Service 字节一致，namespace 隔离 PASS |
| Native Provider contract | 7/7 PASS；namespace/isolation/zero version/zero seed 负向覆盖 |
| xLLM production objects | hasher、request parser、P/D masters、runtime、journal client、两类 scheduler 均编译 PASS |
| xLLM Decode observation wire | 3/3 PASS；presence、有效 0、越界拒绝 |
| xLLM simulated HBM | 12/12 PASS；D 预驻留两 block、只传 suffix、内容 checksum 与容量守恒 |
| Service canonical golden | 1/1 PASS |
| Service actual/gate/wire | metrics 6/6、planner 10/10、RPC wire 12/12 PASS |
| Service production objects | common、Provider contract、InstanceMgr、CAR、Scheduler 编译 PASS |

xLLM 完整 CPU 宿主目标仍会先遇到仓库既有的 `version_singleton` mutex 类型与
`ProcessGroupImpl` 不完整类型错误；本批对所有直接修改的生产对象执行了单目标编译。

## 灰度开放口径

`gate_open` 是运维确认位，不是由单请求自动翻转的开关。开放前必须从同一 model
revision + Provider profile + KV namespace 的 shadow 窗口核对：P/D actual 缺失率、
overprediction、fallback、admission conflict 和 skipped-transfer 对账均达到部署阈值；
先从非 0 小 bucket 开始，异常时把 gate 关闭即可立即回到 SHADOW。仓库默认值不能
改成 ENFORCED，B10 的集群可观测性会把这些门禁指标纳入统一快照和解读手册。
生产入口显式暴露 `--kv_route_mode`、`--kv_route_enforced_gate_open`、
`--kv_route_enforced_bucket_permyriad` 和 `--kv_route_bytes_per_token`；非法 mode、超过
10000 的 bucket，以及未同时配置 CAR + ENFORCED + 非零 bucket 的 open gate 均在
Master 启动时稳定拒绝，避免“配置已生效”的错误假设。

## 明确边界与后续门

- descriptor namespace 仍描述 Engine 不可变 cache semantics；B10 已在其上增加
  `xkvns-request-v1` 的请求级派生域。默认不信任租户头，使用 request UID 派生域且
  关闭跨请求 KV credit；只有部署显式开启 `--trusted_tenant_headers_enabled`、并保证
  Gateway 已鉴权且替换租户/flow 头时，才以 tenant isolation domain 开放同租户复用。
  派生函数已为可信动态 adapter identity 保留长度分隔输入；当前 V2 API 没有动态
  LoRA/adapter 请求字段，因此该输入为空，未来增加时必须先接入该字段再开放 credit。
- canonical Engine helper已支持逐 block `block_extra`；当前 Service V2 ingress 只对纯文本
  开放 K1。未来多模态 ingress 必须先生成相同的 block-local digest/range 才能开放 credit。
- Direct legacy 请求产生的旧 hash 与 Service-routed canonical hash 位于不同键空间，最多
  造成保守 miss，不会产生跨 namespace 命中。
- B8 的仓库内 CPU/simulated HBM 门已完成。真实 NPU HBM、真实多 Service P/D 集群的
  prediction calibration、冲突率和灰度成功率仍标记 `NPU_AND_CLUSTER_PENDING`，不得由
  CPU 结果替代。
