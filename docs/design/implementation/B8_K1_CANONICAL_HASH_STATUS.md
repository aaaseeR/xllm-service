# V2-B8 K1 Canonical Hash 与 Namespace 状态

更新时间：2026-08-09
状态：`IN_PROGRESS`（canonical hash 子门已通过，actual observation 与 bucket
gate 仍在推进）

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

## CPU 证据

环境：`xllm-dev-sandbox` Ubuntu 24.04 ARM64，Clang 18，PyTorch CPU。

| 范围 | 结果 |
| --- | --- |
| xLLM 默认 CPU gate | 8 个目标、115 项 PASS |
| Engine canonical golden | 1/1 PASS；两级 chained hash 与 Service 字节一致，namespace 隔离 PASS |
| Native Provider contract | 7/7 PASS；namespace/isolation/zero version/zero seed 负向覆盖 |
| xLLM production objects | hasher、request parser、P/D masters、runtime、journal client、两类 scheduler 均编译 PASS |
| Service canonical golden | 1/1 PASS |
| Service contract/context/planner | 36 + 4 + 9 项 PASS |
| Service production objects | common、Provider contract、InstanceMgr、CAR、Scheduler 编译 PASS |

xLLM 完整 CPU 宿主目标仍会先遇到仓库既有的 `version_singleton` mutex 类型与
`ProcessGroupImpl` 不完整类型错误；本批对所有直接修改的生产对象执行了单目标编译。

## 明确边界与后续门

- 当前 descriptor namespace 使用 `single-tenant-default`。B10 必须把经鉴权得到的
  tenant isolation salt 和动态 adapter identity 纳入每请求 hash domain；输入不可验证时
  关闭该请求 KV credit，不能使用客户端自报值。
- canonical Engine helper已支持逐 block `block_extra`；当前 Service V2 ingress 只对纯文本
  开放 K1。未来多模态 ingress 必须先生成相同的 block-local digest/range 才能开放 credit。
- Direct legacy 请求产生的旧 hash 与 Service-routed canonical hash 位于不同键空间，最多
  造成保守 miss，不会产生跨 namespace 命中。
- B8 尚需完成 P `num_cached_tokens`、D `remote_shared_num/skipped_transfer_bytes` 的
  MISSING/VALID_ZERO/VALID_NONZERO 观测闭环，以及确定性 bucket 灰度和 simulated HBM
  predicted/actual replay；这些完成前 `ENFORCED` 不得成为默认模式。
