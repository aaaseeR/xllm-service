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

# V2/V3 连续深度审查与离线线上模拟状态

## 结论

截至 2026-08-11，V2/V3 已连续完成两组三轮、共六轮代码、并发状态机、分布式容错和离线线上模拟审查，并按 Opus §19 在固定提交 `c474f73b245688df262714f34f19ecd8d05b0cdb` 的干净 worktree 完成第三方要求的 smoke×3 + stress×1 独立复现。发现的问题均已修复并进入永久回归门。当前结论为 `CPU_AND_OFFLINE_CLUSTER_VERIFIED / NPU_AND_ONLINE_PENDING`：CPU、Torch CPU、simulated HBM 和真实多进程协议链路已验证，真实 NPU、CANN/HBM/Link、生产网络与长时流量尚未验证，不能据此宣称硬件生产 `VERIFIED`。

## 三轮审查

| 轮次 | 审查重点 | 发现与处理 | 验证结论 |
| --- | --- | --- | --- |
| 第一轮 | V2 请求状态机、流控、deadline、输出、资源回收与 V3 Placement 并发边界 | 发现 dispatch 失败回滚与 scheduler 并发时，已计入 dispatch 的请求可能绕过 queue/dispatched 上限；将容量判断和回队变为同一锁内的原子决策，无法安全回队时走唯一失败终态 | 高风险 63 项重复 20 轮；全仓 Service CPU 通过 |
| 第二轮 | 双 Service、etcd、Native P/D、严格 vLLM Agent/Runtime、部署 gateway 的组合故障 | 扩展短/长 etcd、Prefill 进程丢失、Leader 切换、create 丢响应、Drain race、扩缩容和 simulated HBM 断言；修正 harness 中可能混淆故障归因的时序 | V2/V3 smoke 与 stress 多进程门通过 |
| 第三轮 | 短 deadline、真实 SSE 断流、有界过载、etcd 状态盲区、Leader/Agent/Runtime 突然丢失、替换与恢复 | 修复 deadline 被 affinity 队列和残留 callback 生命周期放大、输出回调与终态竞争、etcd RPC 阻塞直接探测、并发观测时间采样乱序、路由读锁争用、新 Leader Link 序号纪元不兼容、两次故障注入 lease 余量串扰及 legacy TTFT/ITL 使用墙上时钟等问题 | V2 50 ms deadline 实测 68.4 ms；V3 断流、Cancel、Drain、替换、Leader failover 与资源归零通过；最终日志无负 TTFT/ITL |

第一轮修复见提交 `f101cde`，第二轮组合故障门见提交 `4df5cdf`，前三轮最终收口见提交 `3b4a1a7`；追加三轮与 Opus §19 修复分别由 `ab62a24`、`b037b8d`、`0aaefd3` 和 `c474f73` 交付。

## 追加三轮审查

| 轮次 | 审查重点 | 发现与处理 | 验证结论 |
| --- | --- | --- | --- |
| 第四轮 | V2/V3 请求输出、deadline watchdog、ProgressiveReader 和后端 callback 的终态竞争 | 发现流式 payload、deadline error 和终止帧可能跨 worker 并发写同一个 sink，非流式成功也可能覆盖已成立的 deadline；输出改为串行帧写入，并在等待当前帧前原子抢占唯一终态，终态后的上游流立即以 `EPIPE` 停止消费 | 两个终态竞争用例与 Link ABA 用例累计重复 300/300；全量 Service CPU 520/520 |
| 第五轮 | P/D Link desired remove/re-add、相同 incarnation 和在途握手 ABA | 发现相同 P/D incarnation 被删除再加入时，旧握手完成可能误提交到新 Link；为每次本地握手分配单调 `attempt_id`，完成时同时核对 Link identity 与 attempt generation，旧完成失败关闭 | `SameIncarnationReaddRejectsAbaCompletion` 连续 100/100，Link 全量回归通过 |
| 第六轮 | V3 Agent deadline/请求内存边界、Agent/Runtime 独立故障、Deployment 库存对账和恢复时延 | `AttemptLedger.attach` 在同一锁内再次检查 deadline，禁止连接建立耗时越界后启动 Runtime；注入 `request_id` 后重新执行单请求和聚合 body 容量核算；线上模拟新增 Agent-only、Runtime-only、组合 `SIGKILL`，恢复资源必须与部署库存对账且不得产生第 4 副本；修正压测将固定请求批次总耗时误当恢复窗口的错误，改为独立测量收敛时间 | Agent 0.179 秒、Runtime 1.018 秒、组合替换 3.871 秒，均低于 5 秒；V3 stress 15773 请求、16 阶段、8 类故障通过，最大活跃副本为 3 |

## Opus §19 复核与处理

- **N2 已确认并修复**：请求路径不再在 P×D 循环中反复获取 `EngineRegistry` 独占锁和推进 Observation。每次路由决策只构建一次不可变 `EngineRegistryRoutingSnapshot`，锁内一次性物化 schedulable Engine 与 ready Link，候选生成、P×D 配对、SLO/load/KV 路由和请求 incarnation 复核全部改为锁外快照查询。新增 8P×8D、64 Link、8 并发 reader 的规模回归，连续 50 轮累计 100/100 PASS。
- **N3 已确认并修复**：新增 `--engine_direct_evidence_ttl_ms`，并在 Master 启动和 `EngineRegistry` 构造两层失败关闭；要求 soft State TTL 不超过 hard State TTL，direct-evidence TTL 不超过 hard State 与 heartbeat TTL。离线 E2E 显式设置 1000 ms，避免测试依赖隐式默认值。
- **N4 已补权威复现**：从已推送的固定提交 `c474f73b245688df262714f34f19ecd8d05b0cdb` 建立 detached、无修改 worktree，连续执行 smoke×3 + stress×1，共 8 份 V2/V3 报告全部 `passed=true`。复现过程中先后暴露并修复三类问题：固定 32 并发故障波与 at-most-once 不确定窗口不匹配；权威 3→1 scale-down 错误清除 FULL 标记导致剩余健康副本短暂 `RECOVERY_HOLD`；Runtime 健康 fence 与 Agent attach 并发返回的精确 `CANCELLED + INTERNAL_ERROR` 409 终态证明被门禁误判。前两项分别固化受控故障 blast radius 与 `AuthoritativeRemovalPreservesFullSnapshot` 产品回归，后一项只按完整身份/状态 JSON 严格识别，普通或截断 409 仍失败关闭。此前并发编辑期间产生的失败报告不作为实现证据。

## 关键修复与远端代码

- [FlowControlQueue](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/scheduler/flow_control_queue.cpp) 保证 dispatch 失败回滚仍受 queue、dispatched 和总容量硬上限约束，避免并发回滚造成隐式超卖。
- [Scheduler](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/scheduler/scheduler.cpp) 让 `CANCELLED` 与 `DEADLINE_EXCEEDED` 立即竞争唯一终态，清除 retry/dispatch/output callback 与 `call_data` 引用，避免短 deadline 被共享队列或对象生命周期拖长。
- [Attempt control client](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider/attempt_control_client.cpp) 和 [HTTP service](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/http_service/service.cpp) 对每个 vLLM hop 使用 `min(业务剩余 deadline, Provider hop timeout)`，不允许内部 hop 越过请求时限。
- [全局配置](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/common/global_gflags.cpp) 与 [Master 接线](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/master.cpp) 暴露 Engine State soft/hard TTL 和 heartbeat hard TTL，支持生产环境按网络和采集周期校准而无需改代码。
- [Engine Registry](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider/engine_registry.cpp) 串行归一并发观测时钟、允许 anti-flap entry hold 内由新鲜直接成功证据桥接状态短盲区，并在 Service master incarnation 变化时清空旧 Link 证明，要求新纪元重新握手；权威 scale-down 将已接受 FULL 快照投影到严格剩余成员子集，避免删除健康副本时错误打断剩余容量 readiness，新增/替换成员仍清空 FULL 并失败关闭。
- [Engine Registry 路由快照](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider/engine_registry.h) 将一次请求需要的 Engine/Link 判据物化为不可变视图，避免 P×D 循环重复进入全局 Registry 独占临界区。
- [Instance Manager](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/scheduler/managers/instance_mgr.cpp) 将 direct probe 从可能被 etcd transport timeout 阻塞的 Registry reconcile 中拆出，并以共享拓扑锁加独立 route cursor 锁避免业务流量饿死后台收敛；无路由日志给出候选、可调度和 Link 证明计数。
- [Request 时钟](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/request/request.h) 为 legacy TTFT/ITL 单独持有 `steady_clock`，响应协议所需 Unix 时间继续使用墙上时钟，避免时钟校准产生负延迟指标。
- [CallData](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/common/call_data.h) 以原子唯一终态和串行输出临界区协调 payload、deadline、断连和终止帧，保证终态请求发起后不会再接纳新输出。
- [Link Reconciler](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider/link_reconciler.cpp) 为每个握手分配本地 attempt generation，拒绝相同 incarnation remove/re-add 产生的 ABA 旧完成。
- [vLLM Agent](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/vllm_sidecar/agent.py) 与 [Attempt Ledger](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/vllm_sidecar/attempts.py) 在 Runtime attach 边界原子复核 deadline，并对转换后的真实请求体执行单请求和全局容量硬限制。
- [离线集群 E2E](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/e2e/online_cluster_stress.py) 覆盖真实 HTTP/SSE 客户端、系统信号、etcd watch/lease、多 Service、生产 Agent、部署慢环和 Torch CPU simulated HBM；故障分类器只接受已知 transport loss 或字段完整的 Runtime-fence 终态证明，并由 5 个永久 Python 用例覆盖正反例。

## stress 线上模拟证据

| 场景 | 实测结果 |
| --- | --- |
| V2 正常与 SSE | baseline 1200/1200；SSE 160/160；两个 P/D route 均被使用 |
| V2 deadline 与过载 | 50 ms 在途 deadline 按预期 1/1 失败并在硬边界内终止，恢复 16/16；不可满足 deadline 64/64 结构化拒绝；过载 72 成功、888 个结构化 backpressure，恢复 32/32 |
| V2 故障 | Prefill `SIGKILL` 在途 640/640、恢复 600/600；短 etcd 停顿 120/120；长故障 grace 内 300/300，随后 self-fence/not-ready/new-incarnation 恢复；Leader failover 后 1200/1200 |
| V2 资源 | simulated HBM 高水位 128 blocks，终态 `used_blocks=0`、`active_allocations=0` |
| V3 扩容与过载 | 单副本高压 431 成功、1369 个结构化 backpressure；扩到 3 副本后 1800/1800，三条 route 均被使用 |
| V3 deadline、断流与 Drain | deadline 3/3 按预期终止；真实 SSE 客户端断流后 Cancel/资源收敛，恢复 60/60；select-before-drain 重选 12/12；3 个已接纳请求全部跨 Drain 存活 |
| V3 突发故障 | Agent-only 2399/2400、Runtime-only 2397/2400，恢复后各 1200/1200 覆盖三 route；恢复耗时分别 0.113 秒和 1.255 秒；Agent+Runtime `SIGKILL` 仅 1 个有界在途 transport failure，1799/1800 完成，替换 3.918 秒；Leader failover 后 1800/1800 |
| V3 扩缩与资源 | 1→3→1；scale-down 期间低负载 13/13、无 readiness 抖动，实际使用 4 个 incarnation；placement、Cancel、deadline 指标存在；全部活动/终止 Runtime 的 simulated HBM 均 `used_blocks=0`、`active_allocations=0`、`tensor_zero=true` |

固定提交独立复现报告为 `build/e2e-artifacts-fixed-c474f73/smoke-{1,2,3}/` 下的 V2 `1786416644-09dcb9e4`、`1786416722-46dd9ef1`、`1786416801-42ab30d5` 与 V3 `1786416680-9e889524`、`1786416758-62411124`、`1786416834-eb2a85ee`，以及 `stress-1/` 下的 V2 `1786416919-203ed2c9`、V3 `1786417002-b366fc4c`。产物目录不提交 Git；可复现命令和验收断言由远端 [E2E 运行说明](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/e2e/README.md) 固化。

## 回归门

- xllm-service ARM64 Linux Debug 全量 CPU CTest：523/523 PASS。
- 新增 CallData 唯一终态/串行输出与 Link ABA 三个高风险用例，累计重复 300/300 PASS；vLLM Agent deadline attach 与转换后 body 容量三个边界用例累计重复 60/60 PASS。
- FlowControlQueue、AttemptControlClient、ClientDisconnectMonitor、RequestDeadlineQueue、PlacementReconciler、ProviderPlacementActuator、HTTP/Registry deployment actuator 共 63 个高风险用例，`until-fail:20` 累计 1260/1260 PASS。
- 全量门暴露 `KVStateStreamClientTest` 将“RPC 超时映射”和“非法输入不出网”混合后依赖服务端调度的竞态；已拆成两个独立语义测试，目标用例连续 100/100 PASS 后再取得全量 517/517。
- vLLM Agent/sidecar 与离线故障分类器 Python 回归：73/73 PASS；三个 Service ARM64 Linux Debug 生产 ELF build/link PASS；非法 Engine TTL 组合在建立集群连接前退出并给出精确错误。
- xLLM ARM64 Linux Debug 的 V2/V3 公共 CPU contract 10 个二进制共 141/141 PASS，含 attempt、registration、Provider/RequestEvent wire 和 simulated HBM；CTest 可发现 1069 项，但 `tests/all` 在第三方 Mooncake `PutOperation` incomplete-type/Clang 编译边界被阻断，因此不再把“发现 1069”误写为“全量 1069 PASS”。双仓 `git diff --check` 与 xllm-service 全仓 pre-commit PASS。

Opus §18 遗留的格式门作用域、V3 runbook 登记和跨层日志转义三项已在前序提交闭环；本轮重新核对 `.coding-ci.yml` 全仓 pre-commit、`OBSERVABILITY_RUNBOOK.md` Placement 指标/事件目录和共享 `proto/log_value.h` 契约，未发生回退。

## 进入真实线上验证前仍需完成

- 真实 NPU 模型加载、CANN/kernel、HBM allocator/cache-loss、DMA/RDMA/Link 故障与资源释放时延。
- 生产 etcd quorum/compact/网络分区、真实部署系统幂等、跨机时钟暂停和基础设施组合故障。
- 生产 prompt/output 分布、多模型、多租户、KV pressure 阶梯负载、容量阈值与 24h+ soak。
- 将线上日志、指标、trace 和 MASS 业务效果数据对账，校准告警、SLO residual、扩缩阈值和回滚条件。
