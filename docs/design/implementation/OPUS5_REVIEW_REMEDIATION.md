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

# Opus 5 V2 Review 整改记录

## 基本信息

- Review 输入：工作区根目录 `opus_sevice_review.md`
- 整改分支：xLLM 与 xllm-service 均为 `service_dev`
- xLLM 整改基线：`6c9d661e`
- xllm-service 整改基线：本文档所在提交
- 验证环境：xllm-dev-sandbox，Ubuntu 24.04 ARM64、Clang 18、
  PyTorch 2.10.0+cpu
- 状态：CPU_VERIFIED / NPU_PENDING

## 结论

Review 报告的 2 个 P0、4 个 P1、8 个 P2 和 5 个 P3 均已逐条核实。
其中代码和协议缺陷已修复并增加永久 CPU 回归；P3-3 的安全部分已完成有界
drain，主动删除 Engine etcd 成员没有照搬建议，因为 Service 不拥有 Engine 的
lease，现有接口也没有 incarnation compare，盲删会删除重启后新 incarnation。
该项保留 lease TTL 作为成员性权威，并把具备 compare-delete 的主动注销列为后续
集群故障注入范围。

## 逐项整改

| ID | 结论 | 整改与证据 |
| --- | --- | --- |
| P0-1 | FIXED | xLLM 将 etcd 查询区分为 PRESENT/MISSING/UNAVAILABLE，临时不可用只重试；权威 key 丢失后本进程立即 fail-stop，绝不复用旧 incarnation 继续接流。注册改为 compare-create，避免 check-then-put 覆盖。没有采用运行中轮换 identity 的建议，因为现有 attempt ledger 与已派发请求绑定不可变 incarnation，原地轮换会制造两个逻辑进程。租约默认值和硬下限提升到 15 秒，reconcile 必须在 `(0, ttl/3]`，KeepAlive 异常显式记录。`RegistrationLifecycleTest` 6/6。 |
| P0-2 | FIXED | xllm-service 的 `third_party/xllm` 固定到 `b1cc43dc`，`.gitmodules` 使用内部 xLLM 仓库；CMake 配置阶段逐一检查所需 proto 源存在。默认 pinned 构建和外部工作树覆盖构建均为 304/304。 |
| P1-1 | FIXED | `CallData` 按 OpenAI/Anthropic 方言生成合法错误事件和显式终态。OpenAI 错误后恰好一个 `[DONE]`；Anthropic 使用 `event: error`，不发送 OpenAI marker。写终态后关闭写栅栏。 |
| P1-2 | FIXED | 普通完成/故障清理只转移 execution hold 到有界 cleanup table，不在全局 cleanup mutex 内发阻塞 RPC。首输出 retry 的 Cancel/Query 也在释放全局锁后执行，返回后重新取锁并复核 request/attempt identity。 |
| P1-3 | FIXED | xLLM 将实际绑定端口与可路由本机地址组合后发布，`0.0.0.0` 不再进入 Descriptor；Service JSON 边界同时拒绝 `0.0.0.0`、`::`、`[::]` 和 `*`。xLLM `NetTest` 5/5，Service wildcard 负向测试通过。 |
| P1-4 | FIXED | 流输出抽象为可注入 `StreamOutputSink`，测试无需真实 brpc socket；所有写路径检查 sink/ProgressiveAttachment，不再解引用空 `pa_`。`CallDataTest` 7/7 覆盖非流、两种 SSE、错误、重复终态和缺 sink。 |
| P2-1 | FIXED | 未解决的 `RequestExecutionHold` 析构时通过 table 的弱安全状态自动 adopt；table 析构先使该状态失效，避免悬空回调。永久测试 `DestructorTransfersUnresolvedHoldToCleanup` 通过。 |
| P2-2 | FIXED | `confirm_holder` 收窄候选时保留该 holder 已取得的 convergence proof；Scheduler 将 `kOk` 和“已有证明使 hold 当场释放”的 `kResolved` 都视为 GenerationCommit 成功。永久测试 `ConfirmHolderPreservesExistingConvergenceEvidence` 同时覆盖调用方使用的成功判定。 |
| P2-3 | FIXED | vLLM Agent 启动时要求非空、可打印且有界的 token；推理和所有 attempt 控制接口统一 `hmac.compare_digest` 鉴权。`/livez`、`/health`、`/readyz` 只暴露标准存活/就绪单比特，不返回 attempt、incarnation 或 fence 原因，刻意保持为负载均衡器公开探针。 |
| P2-4 | FIXED | Service 调用 vLLM Agent 前严格校验 token，非法配置 fail closed 并限流记录明确错误；cleanup 失败按 token_invalid/no_channel/rpc_failed/non_terminal 分类计数。Agent 路径不允许空 token。 |
| P2-5 | FIXED | 删除从未赋值的 `LEASE_LOST` 死枚举；权威 DELETE 继续通过现有 ACTIVE/非 ACTIVE 事实关闭调度门。 |
| P2-6 | FIXED | 删除无生产消费者的 Admission evidence、性能直方图/吞吐/失败计数、PlanPrediction 和固定 `READY` 的 Engine connector 占位字段，并 `reserved` 原字段号与名称。Remote PD 的真实 connector 健康只由包含 handshake、版本和 incarnation 的 `LinkState=READY` 表达。 |
| P2-7 | RESOLVED | V2 Provider wire 已改为单一来源：Service 直接编译 pinned xLLM 的同一 `provider.proto`，因此两份 descriptor 的兼容比较不再适用。两仓均断言关键字段号和 reserved 段；xLLM 有 golden wire，Service 增加 Admission golden wire。新增 `V2 CPU Gate` PR 流水线，显式校验 pin 后运行 pinned Service、Agent 和 pinned xLLM CPU 回归。legacy disagg adapter 继续由既有双向 wire compatibility 测试保护。 |
| P2-8 | FIXED | master heartbeat 在配置 token 时使用 OpenSSL `CRYPTO_memcmp` 常量时间比较。Native heartbeat 为滚动升级兼容仍允许部署不配置 token；vLLM Agent 的数据面和控制面始终强制 token。 |
| P3-1 | FIXED | 沙箱 README 更新为 xLLM 103 项公共 CPU 测试和 Service 304/304 双构建基线，并记录无设备 backend 的真实链接边界。 |
| P3-2 | FIXED | Anthropic 以 `message_stop` 结束，不再追加 `data: [DONE]`；错误采用 Anthropic `event: error`。 |
| P3-3 | SAFE PARTIAL | `Master::stop` 先关闭准入并刷新 readiness，再按 `shutdown_drain_timeout_ms` 有界等待；`Scheduler` 析构也强制 draining。未做不安全的主动 etcd 删除：key/lease 归 Engine 所有，缺少 incarnation compare-delete 时删除可能误伤新进程。真实环境继续以 lease TTL 收敛；compare-delete 与 etcd fault injection 待集群验证。 |
| P3-4 | FIXED | 删除基于 payload 子串搜索 `[DONE]` 的终态推断，只有显式 `finish`/`finish_with_error` 能关闭流；终态之后写入被拒。 |
| P3-5 | FIXED | RPC 服务日志从 `server.listen_address()` 读取真实绑定地址，不再打印预先拼接的字符串。 |

## 复审新增项整改

| ID | 结论 | 整改与证据 |
| --- | --- | --- |
| N1 | FIXED | Scheduler 的 GenerationCommit 屏障统一使用 `execution_holder_confirmation_succeeded`，`kOk`/`kResolved` 均成功，其余状态 fail closed；多 holder 已先收敛 confirmed holder 的回归通过。 |
| N2 | FIXED（参数与部署约束） | fail-stop 安全语义不变；默认且最小 TTL 为 15 秒，heartbeat/reconcile 必须不大于 TTL 三分之一，非法配置在建立 etcd 客户端前拒绝。KeepAlive 失败回调记录 key、TTL 和异常。没有采用连续 N 次 authoritative missing，因为成员已被 Service 移除后继续保留旧 incarnation 会破坏 fencing 证明。更复杂的排空 ledger 后原地轮换仍不属于当前版本。 |
| N3 | FIXED | 删除两个生产者都写常量 `READY` 的 Engine connector 字段并 reserved tag 12/name；Registry 不再使用伪信号。严格 Remote PD 仍要求真实 `LinkState=READY`，没有降低 connector handshake 门禁。 |
| N4 | FIXED | `.github/workflows/v2_cpu.yml` 对 PR/push 执行 pinned proto/gitlink 守卫、Service CTest、三个生产服务构建、Agent pytest 和 pinned xLLM 七个 CPU contract 目标。 |
| N5 | REPO FIXED / PLATFORM ACTIVATION PENDING | pin/proto 守卫抽成 `scripts/verify_v2_pin.sh` 唯一入口；GitHub 镜像和根目录 `.coding-ci.yml` 均覆盖 `main/service_dev` 的 push/MR，并显式构建三个 serving 二进制。Coding 首次远端执行与受保护分支 required check 仍需平台确认。 |

## CPU 验证证据

| 范围 | 命令/目标 | 结果 |
| --- | --- | --- |
| xLLM 公共回归 | `xllm-dev xllm-test <xllm> native Debug` | 121/121 PASS，含 simulated HBM/生产 BlockManager adapter 15/15 |
| xLLM Provider wire | `ProviderProtocolTest` | 8/8 PASS |
| xLLM 地址归一化 | `util_test --gtest_filter=NetTest.*` | 5/5 PASS |
| xllm-service pinned xLLM | `xllm-dev service-test <service> native Debug` | 391/391 PASS |
| xllm-service 外部 xLLM | `XLLM_SOURCE_DIR=<xllm> ... service-test` | 391/391 PASS |
| vLLM Agent/sidecar | `python3 -m pytest vllm_sidecar/tests -q` | 60/60 PASS |
| 生产目标 | `xllm-dev service-verify <service> native Debug` | 三个 ARM64 ELF 编译、动态链接 PASS |

xLLM 本次变更涉及的 server、validator、native provider 和测试对象已在 Torch CPU
头文件/库环境编译。完整推理 runtime 在没有 CUDA/CANN backend 时仍受既有
`process_group.cpp` 的 `ProcessGroupImpl` 不完整类型限制；这不是本次整改回归，也不
伪装成 CPU 推理验证。

CPU 结果只证明控制链路、协议、状态机与 host/Torch CPU 所有权。B6 已补齐
simulated HBM 的固定容量、block 所有权、内容/checksum、OOM/碎片、transfer/fencing
和故障回收门禁；它仍不能替代 NPU 真机 HBM 验证。

## 剩余门禁

- NPU/CANN、真实 P/D Link 与 KV 数据面故障矩阵仍为 NPU_PENDING。
- 真实 etcd lease 到期、watch 乱序、compare-delete 和多 Service 切主需在集群环境
  做 fault injection。
- 生产性能、容量与长时间 soak 不由本次 CPU 正确性回归替代。

## §12/§13 声称—实现对齐专项整改

本节对应 `opus_sevice_review.md` §12 的 D1-D8 与 §13 的 D9-D18。结论以生产调用链、
协议消费方和永久测试为准，不把 simulator 自证、字段存在或旧二进制结果当作闭环。

| ID | 状态 | 整改与证据 |
| --- | --- | --- |
| D1 / D9 | CPU CONTRACT FIXED / NPU INTEGRATION GATE | xLLM 新增 `BlockManagerKVResourceBackend`，直接持有并释放真实 `BlockManager` leaf 返回的 RAII `Block`；测试因此与生产 allocator 共享 free-list，而不是平行账本。`BlockManagerPool::kv_resource_backend()` 已进入生产对象图但尚无运行时消费者；不能为“消除残留”让 scheduler 通过第二路径重复记账。NPU 接入时必须让真实资源协议消费该 seam，或删除 accessor；此前只标记 CPU 契约闭环，不宣称运行时 HBM 数据路径完成。 |
| D2 / D17 / D19 | FIXED | 自研客户端可回传 HMAC token；标准 OpenAI `user`/Anthropic `metadata.user_id` 只有在独立的 `trusted_client_identity_headers_enabled` 门开启时，才与 Gateway 剥离并重写的 `x-authenticated-client-id` 派生。Service 不再读取原始 Authorization/API-Key 作为认证身份；开关默认 false，未建立可信边界时伪造 credential+user 仍得到不同 per-request 域和独立 token。 |
| D3 | FIXED | band 内改为 tenant→flow 两级轮转；新 flow 从当前队尾加入，不从终生计数 0 开始，派发后回队尾，空闲即擦除。新增新 flow、轮换 flow-id、取消、过期、return-to-queue 与并发账本测试。 |
| D4 | FIXED | tenant、flow、priority 作为同一个信任单元；未开启可信 Gateway 模式时客户端 priority 一律归一为默认 band。 |
| D5 | FIXED | ENFORCED 启动同时要求 CAR、ENFORCED mode、非零 bucket、`bytes_per_token`、Prefill token cost 和有限正 transfer-byte cost，缺任一项拒绝启动。 |
| D6 | FIXED | `PREFILL_ONLY` 默认关闭，只有部署显式开启且 capability/输出上限均满足时才可选择。 |
| D7 | FIXED | tenant/model/flow 的公平与容量状态在 queued/dispatched 归零后擦除；snapshot 暴露 active account 数，永久测试覆盖 churn。 |
| D8 | FIXED | 删除没有生产者的 `adapter_identity` 参数，namespace domain 升级为 `xkvns-request-v2`；未来动态 adapter 必须先进入受信 Provider profile/contract，不能复活私有占位参数。 |
| D10 | FIXED | V2 Engine 只发布和 Service 只消费真实存在的 HBM/HOST tier。Hierarchy host prefix leaf 与 device leaf 共享 journal，分别发布 HOST/HBM，reset 在两个 tier 都清空后才推进 epoch。SSD/STORE 字段和查询从 V2 删除并 reserved；真实 Store 数据路径属于后续版本。 |
| D11 | FIXED | Service 聚合 fresh EngineState 的 `kv_used_ratio/kv_free_blocks`，导出 reporting engines/DP ranks、max used ratio、min/total free blocks 五个 bvar，并写入周期 cluster snapshot；硬 stale Engine 不参与容量结论。 |
| D12 | FIXED | `ROUTE` 对容量、stale、mode 与永久不可行分别发 REJECTED；`D_ADMISSION` 由 xLLM D 侧真实 `AdmissionResult` 经 P scheduler/输出 wire 回传，包含 disposition/reason/尝试数/累计 RPC 时长，缺终态 fail closed；`RESOURCE_RELEASE` 对即时释放、deferred cleanup、cleanup 收敛和 shutdown 未收敛都形成终态。exporter 退出前排空 ring。 |
| D13 | FIXED | HTTP/RPC 错误保留稳定错误类别、精确 contract/flow 原因和 `request_uid`；容量拒绝附当前 queue snapshot，未知请求、参数错误、deadline 与内部错误不再统一塌缩为 `Internal runtime error`。 |
| D14 | RESOLVED BY BOUNDARY | 不把四种语义不同的 envelope 强并为一个万能状态：`ContractResult` 是跨仓 Provider contract，`KVApplyResult` 是 replica sequence，`FlowControlStatus` 是本地准入，`KVBlockResourceStatus` 是 Engine 物理资源。每个状态只在所属层流动，并在唯一边界转换为稳定 client error/EventReason；禁止跨层透传 free-form message 做控制决策。这样统一的是翻译纪律而非丢失领域语义的类型。 |
| D15 | FIXED | Service 宏全部改为 `XLLM_SERVICE_*` 前缀，queue 私有宏在文件末尾 `#undef`；xLLM detector 显式包含 xLLM 自己的 macro header。Service 不再定义通用 `PROPERTY/CALLBACK_WITH_ERROR`。 |
| D16 | FIXED | planner 成本、reserve、margin、near-equal 等全部开放 gflag并做范围校验；提供 `examples/v2_shadow.flags`，保持 SHADOW、alternate mode 关闭、ENFORCED 所需模型参数为 0。ENFORCED 还拒绝 development/placeholder build ID。会话 secret 只允许部署 secret 注入，不写 flagfile。 |
| D18 | FIXED | token 升级为 `v2.<issued_at>.<session>.<hmac>`，TTL 范围 1 秒至 30 天，拒绝过期和超前超过 60 秒的 token；active key 签发、previous key 仅验证，实现有界轮转。ENFORCED 且无可信 tenant header 时强制所有副本配置同一个 shared secret；shadow/dev 的随机本地 key 只允许 sticky replica 并明确 WARNING。 |

## §14 第五轮复核新增问题闭环

| §14 项 | 结论 | 本轮处理与验证 |
| --- | --- | --- |
| 14.9 主程序编译 P0 | FIXED | `main()` 中 `KVSessionTokenCodec` 补全 `xllm_service::` 限定；`xllm_master_serving`、`xllm_http_serving`、`xllm_rpc_serving` 均完成 Linux ARM64 编译和动态链接检查。 |
| 14.10 CI 平台不匹配 | REPO FIXED / PLATFORM ACTIVATION PENDING | 根目录新增 `.coding-ci.yml`，覆盖 `main/service_dev` push/MR，执行 pin guard、Service 391 项、三个 serving binary、vLLM 60 项和 xLLM CPU contract。YAML 与官方 schema 本地检查通过；首次 Coding 远端运行和 required check 仍需平台确认。 |
| 14.11 公平性无回归 | FIXED | 两个临时 probe 并入正式 `FlowControlQueueTest`：late flow 与 established flow 各获得 10/20 slots 且最大连续派发不超过 2；200 个 drained flow 不残留 active state。两例进入全量 CTest。 |
| 14.6 D13 最后兜底 | FIXED | `Schedule request failed!` 改为稳定 `SCHEDULE_REJECTED; request_uid=...`，保留公开定位键且不泄露内部 message。 |
| 14.8 D19 | FIXED | 原始 Authorization/API-Key 不再作为身份；只有显式可信 Gateway client-id 门才能把标准 SDK `user` 变成跨请求域。默认关闭的负向测试证明伪造 credential+user 不能共享 namespace。 |
| 14.3 D9 残留 | ACCEPTED AS EXPLICIT NPU GATE | accessor 留作真实 BlockManager conformance seam，但尚非运行时数据路径；NPU 接入时消费或删除，当前文档不宣称生产 HBM 闭环。 |
| D14 命名统一 | RESOLVED BY DOMAIN BOUNDARY | 四类状态保持领域强类型，统一公开翻译纪律；强行合成万能 Status 会让 allocator、replica sequence、flow admission 与 Provider contract 错误耦合，因此不做无收益重构。 |

本轮最终证据：xLLM 121/121、xllm-service pinned 391/391、外部 xLLM override
391/391、vLLM sidecar 60/60、三个 serving ELF build/link PASS。

### 状态类型边界

| 领域状态 | 权威层 | 唯一外译边界 | 禁止事项 |
| --- | --- | --- | --- |
| `ContractResult` | Provider/公共 proto | HTTP ingress、plan/dispatch adapter | 以 message 文本驱动重试或路由 |
| `KVApplyResult` | Service KV replica/index | stream consumer 与 snapshot recovery | 作为 Engine allocator 状态返回 |
| `FlowControlStatus` | Service 本地队列 | `admit_flow_control_locked` / client error | 穿过 RPC wire |
| `KVBlockResourceStatus` | xLLM Engine block backend | Provider/Engine resource observation | 让 Service 感知 CANN/CUDA allocator 细节 |

### CPU 与 NPU 证据边界

| CPU/Torch CPU 可验证 | 必须在 NPU 验证 |
| --- | --- |
| block 生命周期、free-list、prefix LRU、抢占与 OOM 决策 | 真实 HBM 碎片与 `aclrtMalloc` 失败模式 |
| `n_blocks` 字节换算与容量守恒 | 权重加载后 `aclrtGetMemInfo` 余量 |
| Torch CPU tensor shape/stride/逐层布局 | NZ format cast 与大页分配 |
| host mirror 和 D2H/H2D 的逻辑配对 | 实际带宽、异步拷贝/计算重叠 |
| 跨请求 prefix 命中判定与 namespace 隔离 | TTFT 收益与多卡 DP/TP 一致性 |

当前 CPU 结论仍为 `CPU_VERIFIED / NPU_PENDING`。完整无设备 runtime build 已修复 xLLM
自身的 `shared_mutex` 锁类型和无硬件 process-group 工厂编译问题，随后停在第三方 Mooncake
Clang thread-safety/incomplete-type 错误；这不影响重新构建的轻量 resource target 或
xllm-service pinned/override 391/391 和三个生产 ELF build/link，但也不被包装成真实
HBM/Torch 全 runtime 证明。xLLM 整改提交 `6c9d661e` 已先推送到远端 `service_dev`，
Service gitlink 再固定到该提交。
