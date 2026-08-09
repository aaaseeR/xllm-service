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
- xLLM 整改基线：`a260e2fc`
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
| P0-1 | FIXED | xLLM 将 etcd 查询区分为 PRESENT/MISSING/UNAVAILABLE，临时不可用只重试；权威 key 丢失后本进程立即 fail-stop，绝不复用旧 incarnation 继续接流。注册改为 compare-create，避免 check-then-put 覆盖。没有采用运行中轮换 identity 的建议，因为现有 attempt ledger 与已派发请求绑定不可变 incarnation，原地轮换会制造两个逻辑进程。`RegistrationLifecycleTest` 3/3。 |
| P0-2 | FIXED | xllm-service 的 `third_party/xllm` 固定到 `a260e2fc`，`.gitmodules` 使用内部 xLLM 仓库；CMake 配置阶段逐一检查所需 proto 源存在。默认 pinned 构建和外部工作树覆盖构建均为 304/304。 |
| P1-1 | FIXED | `CallData` 按 OpenAI/Anthropic 方言生成合法错误事件和显式终态。OpenAI 错误后恰好一个 `[DONE]`；Anthropic 使用 `event: error`，不发送 OpenAI marker。写终态后关闭写栅栏。 |
| P1-2 | FIXED | 普通完成/故障清理只转移 execution hold 到有界 cleanup table，不在全局 cleanup mutex 内发阻塞 RPC。首输出 retry 的 Cancel/Query 也在释放全局锁后执行，返回后重新取锁并复核 request/attempt identity。 |
| P1-3 | FIXED | xLLM 将实际绑定端口与可路由本机地址组合后发布，`0.0.0.0` 不再进入 Descriptor；Service JSON 边界同时拒绝 `0.0.0.0`、`::`、`[::]` 和 `*`。xLLM `NetTest` 5/5，Service wildcard 负向测试通过。 |
| P1-4 | FIXED | 流输出抽象为可注入 `StreamOutputSink`，测试无需真实 brpc socket；所有写路径检查 sink/ProgressiveAttachment，不再解引用空 `pa_`。`CallDataTest` 7/7 覆盖非流、两种 SSE、错误、重复终态和缺 sink。 |
| P2-1 | FIXED | 未解决的 `RequestExecutionHold` 析构时通过 table 的弱安全状态自动 adopt；table 析构先使该状态失效，避免悬空回调。永久测试 `DestructorTransfersUnresolvedHoldToCleanup` 通过。 |
| P2-2 | FIXED | `confirm_holder` 收窄候选时保留该 holder 已取得的 convergence proof。永久测试 `ConfirmHolderPreservesExistingConvergenceEvidence` 通过。 |
| P2-3 | FIXED | vLLM Agent 启动时要求非空、可打印且有界的 token；推理和所有 attempt 控制接口统一 `hmac.compare_digest` 鉴权。`/livez`、`/health`、`/readyz` 只暴露标准存活/就绪单比特，不返回 attempt、incarnation 或 fence 原因，刻意保持为负载均衡器公开探针。 |
| P2-4 | FIXED | Service 调用 vLLM Agent 前严格校验 token，非法配置 fail closed 并限流记录明确错误；cleanup 失败按 token_invalid/no_channel/rpc_failed/non_terminal 分类计数。Agent 路径不允许空 token。 |
| P2-5 | FIXED | 删除从未赋值的 `LEASE_LOST` 死枚举；权威 DELETE 继续通过现有 ACTIVE/非 ACTIVE 事实关闭调度门。 |
| P2-6 | FIXED | 删除无生产消费者的 Admission evidence、性能直方图/吞吐/失败计数和 PlanPrediction 字段，并 `reserved` 原字段号与名称；`connector_state` 保留且已成为 Registry 的硬调度过滤条件，值统一为 READY/NOT_READY。 |
| P2-7 | RESOLVED | V2 Provider wire 已改为单一来源：Service 直接编译 pinned xLLM 的同一 `provider.proto`，因此两份 descriptor 的兼容比较不再适用。两仓均断言关键字段号和 reserved 段；xLLM 有 golden wire，Service 增加 Admission golden wire。legacy disagg adapter 继续由既有双向 wire compatibility 测试保护。 |
| P2-8 | FIXED | master heartbeat 在配置 token 时使用 OpenSSL `CRYPTO_memcmp` 常量时间比较。Native heartbeat 为滚动升级兼容仍允许部署不配置 token；vLLM Agent 的数据面和控制面始终强制 token。 |
| P3-1 | FIXED | 沙箱 README 更新为 xLLM 100 项公共 CPU 测试和 Service 304/304 双构建基线，并记录无设备 backend 的真实链接边界。 |
| P3-2 | FIXED | Anthropic 以 `message_stop` 结束，不再追加 `data: [DONE]`；错误采用 Anthropic `event: error`。 |
| P3-3 | SAFE PARTIAL | `Master::stop` 先关闭准入并刷新 readiness，再按 `shutdown_drain_timeout_ms` 有界等待；`Scheduler` 析构也强制 draining。未做不安全的主动 etcd 删除：key/lease 归 Engine 所有，缺少 incarnation compare-delete 时删除可能误伤新进程。真实环境继续以 lease TTL 收敛；compare-delete 与 etcd fault injection 待集群验证。 |
| P3-4 | FIXED | 删除基于 payload 子串搜索 `[DONE]` 的终态推断，只有显式 `finish`/`finish_with_error` 能关闭流；终态之后写入被拒。 |
| P3-5 | FIXED | RPC 服务日志从 `server.listen_address()` 读取真实绑定地址，不再打印预先拼接的字符串。 |

## CPU 验证证据

| 范围 | 命令/目标 | 结果 |
| --- | --- | --- |
| xLLM 公共回归 | `xllm-dev xllm-test <xllm> native Debug` | 100/100 PASS |
| xLLM Provider wire | `ProviderProtocolTest` | 8/8 PASS |
| xLLM 地址归一化 | `util_test --gtest_filter=NetTest.*` | 5/5 PASS |
| xllm-service pinned xLLM | `xllm-dev service-test <service> native Debug` | 304/304 PASS |
| xllm-service 外部 xLLM | `XLLM_SOURCE_DIR=<xllm> ... service-test` | 304/304 PASS |
| vLLM Agent/sidecar | `python3 -m pytest vllm_sidecar/tests -q` | 60/60 PASS |
| 生产目标 | `xllm-dev service-verify <service> native Debug` | 三个 ARM64 ELF 编译、动态链接 PASS |

xLLM 本次变更涉及的 server、validator、native provider 和测试对象已在 Torch CPU
头文件/库环境编译。完整推理 runtime 在没有 CUDA/CANN backend 时仍受既有
`process_group.cpp` 的 `ProcessGroupImpl` 不完整类型限制；这不是本次整改回归，也不
伪装成 CPU 推理验证。

## 剩余门禁

- NPU/CANN、真实 P/D Link 与 KV 数据面故障矩阵仍为 NPU_PENDING。
- 真实 etcd lease 到期、watch 乱序、compare-delete 和多 Service 切主需在集群环境
  做 fault injection。
- 生产性能、容量与长时间 soak 不由本次 CPU 正确性回归替代。
