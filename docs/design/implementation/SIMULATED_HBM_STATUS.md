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

V2-B6 仓库内代码与 CPU 门已完成，状态为
`CPU_VERIFIED / PLATFORM_AND_NPU_PENDING`。xLLM commit `49916abc` 提供无设备 API
的 KV block 资源契约和只在测试目标中链接的 simulated HBM backend；xllm-service
本文件所在提交 pin 到该 commit，并将目标加入 V2 CPU gate。该结论证明资源链路、
内容和容量不变量，不表示真实 NPU HBM、CANN stream/event 或 DMA 已验证。

JD Coding 受保护分支仍须由平台管理员把 `scripts/verify_v2_pin.sh` 和完整 CPU gate
设为合入前必需检查。仓库内入口已经唯一化，但仓库提交不能代替服务端 branch
protection，因此该项保持 `PLATFORM_PENDING`，不阻塞后续仓库内 B7-B10 开发。

## 实现边界

- 生产契约位于 xLLM `xllm/core/framework/kv_cache/kv_block_resource.h`，只表达 owner、
  稳定逻辑 block 地址、generation、容量状态和 transfer 生命周期，不暴露 device
  pointer、CANN/CUDA handle、stream、event 或 allocator 实现。
- owner 键包含 `engine_uid/incarnation_id/model_id/kv_namespace/request_uid/attempt_seq/
  dp_rank/device_rank`。model 与 KV namespace 从 B6 起进入隔离键，B10 不需要重写
  底层所有权语义。
- simulator 位于 xLLM `tests/core/framework/kv_cache`，不进入 Service 生产依赖。固定
  block 数和 block bytes，支持连续/离散分配、OOM/碎片区分、ABA generation、内容
  写读/FNV checksum、异步 transfer、result-unknown、cancel、owner/incarnation fence、
  quarantine 延迟释放、snapshot 和内部不变量检查。
- transfer 未终结时同时 pin source 与 destination；timeout 在控制层转为
  result-unknown，再通过 fence + complete/cancel 收敛。结果未决期间不会为回收计数而
  提前复用 block。
- Service 只消费 Provider 的标准化资源事实。现有 fake Provider/ExecutionHold 测试
  验证 dispatch 前 reservation、单 hold、P/D holder 转移责任、attempt/incarnation
  fence、结果未知后的 bounded cleanup 和精确 reclaim；Service 内没有复制 HBM
  allocator。

## 永久测试

simulated HBM 10 个测试覆盖：

1. 固定容量、owner、稳定地址和 generation 防 ABA；
2. 连续分配的可控碎片与离散分配回退；
3. Prefill 内容/checksum 经跨 DP/rank transfer 后保持一致；
4. transfer 期间 source/destination pin、非法 block bytes 拒绝；
5. result-unknown、owner fence、quarantine、complete/cancel 后精确回收；
6. incarnation 替换不继承旧进程 fence；
7. wrong attempt/model/KV namespace 和重复释放 fail closed；
8. active transfer 硬上限不泄漏 block；
9. 8 线程 allocate/write/read/free 容量守恒；
10. 8 线程并发 transfer 的内容和容量守恒。

## 验证证据

环境：`xllm-dev-sandbox`，Ubuntu 24.04 ARM64，Clang 18/GCC 13，C++20，
PyTorch CPU，日期 2026-08-09。

| 范围 | 命令/目标 | 结果 |
| --- | --- | --- |
| xLLM 默认 CPU gate | `xllm-dev xllm-test <xllm> native Debug` | 113/113 PASS，包含 simulated HBM 10/10 |
| Clang 压力 | `simulated_hbm_test --gtest_repeat=100 --gtest_break_on_failure` | 100 轮 PASS |
| GCC ASan+UBSan | `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 simulated_hbm_test --gtest_repeat=100` | 100 轮 PASS，无 sanitizer 报告 |
| Service pinned | `xllm-dev service-test <service> native Debug` | 304/304 PASS |
| Service override | `XLLM_SOURCE_DIR=<xllm> xllm-dev service-test <service> native Debug` | 304/304 PASS |
| Agent/sidecar | `python -m pytest vllm_sidecar/tests -q` | 60/60 PASS |
| 生产 Service | `xllm-dev service-build` + `service-verify` | 三个 ARM64 Debug ELF 构建、动态链接 PASS |
| pin 入口 | `scripts/verify_v2_pin.sh` | gitlink、工作树 revision、Provider/Observability proto PASS |

额外尝试构建通用 `concurrent_block_manager_test/kv_cache_test/
kv_cache_estimation_test` 时，仍分别遇到仓库既有 `ProcessGroupImpl` 不完整类型和
`version_singleton` 的 `shared_mutex`/`lock_guard<mutex>` 类型错误。它们不由 B6
新增目标依赖，默认 113 项门禁不跳过也不掩盖失败；该宿主全量构建限制继续单独跟踪。

## 三轮 review 结果

- 资源/并发：修复并永久测试 transfer pin、quarantine 必须对应已 fence owner、
  错 owner 原子失败和并发 transfer 守恒；未发现泄漏或 double release。
- 协议/兼容：资源接口保持 device-neutral；owner 提前纳入 model/KV namespace；
  generation 阻断旧地址复用；Service 硬件抽象边界未被穿透。
- 测试/文档：默认 gate、100 轮普通压力、100 轮 ASan+UBSan、Service 双构建、Agent
  和生产二进制全部通过；JD branch protection 与真实 NPU 证据明确保持 pending。

## 剩余硬件门

- NPU backend 将同一资源契约映射到真实 KV allocator、CANN stream/event 和 DMA；
- 在固定 HBM 压力下实测 OOM/碎片、跨 rank P→D 内容 checksum、cancel/timeout/进程
  替换后的显存回收时延；
- 执行长时并发、故障注入和性能容量门禁。上述完成前 B6 不标记 `VERIFIED`。
