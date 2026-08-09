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

# V2-B7 K0 KV Shadow 开发状态

## 结论

V2-B7 仓库内代码与 CPU/simulated HBM 门已完成，状态为
`CPU_VERIFIED / NPU_AND_CLUSTER_PENDING`。xLLM commit `a730f0f2` 从真实
PrefixCache mutation source 产生 store/remove/clear 事件，并提供独立 KV lane 和
分页 snapshot；xllm-service 建立有界、fail-closed 的 K0 shadow index、Service
复制 lane 和并发恢复控制。

K0 仍严格处于 observation-only：新 shadow index 没有接入任何负载均衡决策，
`UNKNOWN/RECOVERING` 只表示 KV credit 为零，不会让正常 load-only 路由停机。本门
不声明 NPU HBM、真实多 Service/etcd 网络或线上性能已经验证。

## 实现范围

### Engine 真源与独立 KV lane

- `KVStreamIdentity` 以 Provider/profile/Engine/incarnation/model/KV namespace/cache
  epoch 建立隔离和排序域；`KVEvent` 携带 sequence、block/parent hash、token range、
  cache group、tier、DP/device rank、reason 和本地累计 queue age。
- PrefixCache 只在可复用 block 真正插入或淘汰后发布 `STORED/REMOVED`；整个 pool
  确认缓存为空后才发布新 epoch 的 `CLEARED`，不会把仍被共享引用的 block 误报为空。
- `KVEventJournal` 同时限制 pending event/byte、resident entry/byte、snapshot
  session/entry/byte。事件溢出产生显式 gap；resident 无法完整表达时关闭 snapshot，
  不会用截断状态继续宣称命中。
- Engine sender 与 heartbeat/EngineState 线程分离，至多一个 batch 在途；失败重试
  同一 sequence 范围，idle keepalive 发布当前 high watermark。Service 通过
  `GetKVCacheSnapshot` 直接访问 Engine，不使用 etcd 传输 KV 事件。

### Service shadow、复制与恢复

- `KVShadowIndex` 对 stream、entry/byte、per-Engine recovery event/byte 和 snapshot
  staging 设置硬上限；重复事件幂等，乱序、gap、TTL、clock regression、epoch/
  incarnation 变化和容量溢出均先清除 KV credit。
- snapshot 首页固定 `snapshot_id/cache_epoch/base_event_seq`，恢复期间增量事件进入
  有界 buffer，完整分页后按 sequence 重放并原子 cutover；页数、entry、byte、RPC
  timeout 和总生成时间都有上限，零进度页直接失败。
- recovery worker 对 UNKNOWN streams 做稳定排序和 round-robin，跳过 backoff 项时仍会
  扫描后续 Engine；按配置限制并发，失败重试带稳定 jitter，不会让一个坏 Engine
  长期饿死其余 Engine。
- master→replica KV State lane 与 Engine State lane 完全独立。每个 subscriber 有独立
  queue、sequence 和单调 stream epoch；移除再加入时新 epoch 立即 fencing 旧序列，
  master 变化、lane gap 或非法 batch 都清空旧 credit。
- Service 接收 Engine 事件前校验当前 Registry descriptor 的 Provider/profile/Engine/
  incarnation/model 和 K0 namespace，旧进程或错误模型不能污染 shadow。

## K0 边界与 B8 输入

- K0 namespace 暂用保守的 `profile:<profile_digest>`，只用于隔离观察；它不会造成
  跨 profile 假命中，但可能抑制本可共享的前缀。B8 必须落地 request 与 Engine 共用
  的 canonical hash/namespace preimage 后，K1 才能读取该 shadow 排序。
- K0 不修改现有 `CAR/SLO_AWARE/RoundRobin` 选择；旧 CAR 的 heartbeat cache 逻辑与
  新 K0 shadow 互不冒充。只有 B8 shadow gate 达标后，新 KV credit 才能参与排序。
- `tier=HBM` 在 CPU 中由 simulated HBM 的固定容量、owner、内容和 checksum 语义
  验证。生产事件契约不携带 device pointer、CANN/CUDA handle、stream/event 或
  allocator；真实 NPU/其他硬件仍由 Engine/backend 实现。

## CPU 与 simulated HBM 证据

环境：`xllm-dev-sandbox`，Ubuntu 24.04 ARM64，Clang 18，C++20，PyTorch CPU，
日期 2026-08-09。

| 范围 | 命令/目标 | 结果 |
| --- | --- | --- |
| xLLM 默认 CPU gate | `xllm-dev xllm-test <xllm> native Debug` | PASS |
| KV journal 压力 | `kv_event_journal_test --gtest_repeat=100 --gtest_break_on_failure` | 9/9 × 100 轮 PASS |
| simulated HBM | `simulated_hbm_test` | 11/11 PASS，含 HBM 内容→journal→snapshot/remove 对账 |
| xLLM Provider wire | `provider_protocol_test` | 9/9 PASS，含 KV field/RPC descriptor |
| Service pinned 全量 | `xllm-dev service-test <service> native Debug` | 332/332 PASS |
| Service override 全量 | `XLLM_SOURCE_DIR=<xllm> xllm-dev service-test <service> native Debug` | 332/332 PASS |
| Service K0 targets | shadow/outbox/replica/stream client/snapshot client | 28 项 PASS |
| 生产 Service | `XLLM_SOURCE_DIR=<xllm> xllm-dev service-build` + `service-verify` | 三个 ARM64 Debug ELF 构建、动态链接 PASS |

xLLM 的 PrefixCache、BlockManagerPool/Impl/Composite/Concurrent 和
`XServiceClient` 生产对象均在 CPU sandbox 编译通过。尝试把通用 PrefixCache test
链接到完整 xLLM CPU runtime 时，仍会遇到仓库既有的 `ProcessGroupImpl` 不完整类型和
`version_singleton` 的 `shared_mutex`/`lock_guard<mutex>` 类型错误；这两个宿主全链接
问题未由 B7 引入，B10 完整 CPU 构建门会单独关闭，不能把它们写成 PASS。

## 三轮 review 结果

- 资源/并发：补齐 shadow stream 总量上限、snapshot 固定与分页共用生成预算、恢复
  round-robin/并发/jitter；100 轮 journal 并发压力无 sequence 重复或容量泄漏。
- 协议/故障：修复 replica 地址移除再加入后 sequence 重置被旧状态吞掉的问题，引入
  单调 stream epoch，并增加旧 epoch 晚到包的负向回归；master/incarnation/cache
  epoch 都 fail closed。
- 测试/文档：修复 snapshot timeout 测试对 RPC 是否已派发的错误假设；全量 332 项、
  simulated HBM、wire descriptor 和生产二进制全部通过，K0 不参与路由的边界明确。

## 剩余平台门

- 在真实 NPU PrefixCache eviction/reset 和多 rank 布局上核对 event 与 HBM resident；
- 在真实 etcd、多个 Service 和 Engine replacement 下验证 master 切换、replica 重入、
  gap/snapshot 恢复、lane 隔离和长时容量守恒；
- 执行真实网络限速、乱序/超时、snapshot 大缓存预算和 NPU 性能门禁。

上述证据完成前 B7 不标记 `VERIFIED`，但不阻塞 B8-B10 的仓库内 V2 开发。
