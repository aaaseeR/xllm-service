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

# V2-B6 Simulated HBM 开发状态

## 结论

状态为 `CPU_VERIFIED / PLATFORM_AND_NPU_PENDING`，但证据分层必须精确解读：

- simulated HBM 是固定容量、可注入故障、可检查内容的测试 backend，只证明契约级
  owner/generation/transfer/fencing 不变量；
- 生产 `BlockManagerKVResourceBackend` 直接包装 `BlockManagerPool` 的真实 cache leaf，
  与序列调度共享同一 free-list 和物理容量，修复了原先“模拟器与生产路径完全平行”的
  问题；
- 真实 HBM、CANN allocator、NZ layout、DMA/stream/event 和设备性能只能由 NPU 门证明，
  simulator 结果不得作为它们的替代证据。

## 生产接线

- 契约位于 xLLM `framework/kv_cache/kv_block_resource.h`，表达 owner、稳定逻辑地址、
  generation、容量与 transfer 生命周期，不暴露 device pointer 或硬件 handle。
- `BlockManagerKVResourceBackend` 持有实际 leaf 分配的 RAII `Block`。allocate/release
  直接改变真实 leaf 的 `num_free_blocks()`；`BlockManagerPool` 为每个 DP rank 和
  cache group 构造适配器并提供受 pool 生命周期约束的访问入口。
- adapter ledger 对 active transfer、owner fence 与 incarnation fence 均设置硬上限；
  generation 或 fence ledger 耗尽时 fail closed，禁止 ABA 或为了继续服务而遗忘 fence。
- owner 键包含 Engine/incarnation/model/namespace/request/attempt/DP/device 身份。
  transfer 未终结时 pin source 与 destination，unknown outcome 只有在 complete/cancel
  或 fence 收敛后才能回收。
- Service 只消费 Provider 标准化事实，不持有 allocator、HBM 地址或 stream/event。

## 测试分层

快速 `simulated_hbm_test` 共 15 项：12 项 simulator 故障/内容/并发测试，加 3 项生产
adapter 对真实 `BlockManager` 接口的容量、generation、transfer pin、fence 与 ledger
耗尽测试。它刻意不链接整个推理 runtime，避免把 Mooncake/模型依赖是否可编译混入资源
状态机的快速合入门。

既有真实路径补强：

- `block_manager_test.cpp` 直接以 `BlockManagerImpl` 验证适配器共享容量、ABA generation、
  transfer pin 和 fence 后精确回收；
- `hierarchy_block_manager_pool_test.cpp` 验证真实 device/host prefix leaf 分别发布 HBM/HOST，
  两层都清空后才推进 journal epoch；
- `kv_cache_test.cpp` 继续在 `torch::kCPU` 上调用真实 `allocate_kv_caches()`，验证 shape、
  stride/分组与逐层 tensor 布局。

## 当前验证证据

环境：`xllm-dev-sandbox`，Ubuntu 24.04 ARM64，Clang 18，C++20，PyTorch 2.10 CPU，
日期 2026-08-10。

| 范围 | 结果 |
| --- | --- |
| 重建后的 simulated/resource adapter 快速目标 | 15/15 PASS |
| 快速目标并发压力 | 15 项 × 100 轮 PASS |
| xLLM request-output admission wire | 3/3 PASS |
| xllm-service 外部 xLLM 全量回归 | 388/388 PASS |

为跑真实 block/Torch 全目标，已修复 xLLM 自身的 `VersionSingleton` shared-mutex 锁型
错误，并使无硬件 backend 的 process-group 工厂明确 fail closed。继续构建停在第三方
Mooncake 的 Clang thread-safety annotation 和 incomplete `PutOperation` 错误；因此本批
新增的真实 BlockManager/Hierarchy 用例已进入永久目标，但本次无设备环境没有把它们的
新结果伪报为 PASS。原有 Torch CPU 测试代码与历史基线仍在，当前新增改动的可执行证据
以 15/15、3/3 和 Service 388/388 为准。

## CPU 与 NPU 边界

| CPU/Torch CPU | NPU 专项 |
| --- | --- |
| block 生命周期、free-list、prefix LRU、OOM/抢占决策 | HBM 碎片、`aclrtMalloc` 失败 |
| `n_blocks` 换算、容量守恒 | 权重加载后 `aclrtGetMemInfo` 余量 |
| tensor shape/stride/逐层布局 | NZ format cast、大页分配 |
| host mirror、D2H/H2D 逻辑配对 | 实际带宽和 copy/compute overlap |
| prefix 命中与 namespace 隔离逻辑 | TTFT 收益、多卡 DP/TP 一致性 |

剩余硬件门包括固定 HBM 压力下的 OOM/碎片、跨 rank P→D 内容 checksum、超时/取消/
进程替换后的显存回收时延、长时并发、性能容量和故障注入。完成前不升级为
`VERIFIED`。JD Coding 受保护分支的必需检查仍属于平台配置。
