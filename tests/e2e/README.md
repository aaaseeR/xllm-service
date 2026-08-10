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

# xLLM Service 离线多进程 E2E 交付硬门

本目录不是单元测试替代品。它在 CPU 沙箱启动生产 `xllm_master_serving`、独立 etcd、协议兼容的 Native Engine 或严格 vLLM Agent/Runtime、部署网关和并发 OpenAI HTTP 客户端，用真实进程、socket、watch/lease 和故障信号验证 V2/V3 集群链路。

## 运行

先完成 Debug CPU 构建，再运行：

```bash
/opt/venv/bin/python tests/e2e/online_cluster_stress.py \
  --scenario all --mode smoke \
  --build-dir build/local-arm64-Debug-xllm-override

/opt/venv/bin/python tests/e2e/online_cluster_stress.py \
  --scenario all --mode stress \
  --build-dir build/local-arm64-Debug-xllm-override
```

沙箱可直接执行：

```bash
xllm-dev service-e2e /absolute/path/to/xllm-service native Debug smoke all
xllm-dev service-e2e /absolute/path/to/xllm-service native Debug stress all
```

`smoke` 是每次合入门，`stress` 是版本交付、重要并发/状态机/容错修改和上线前门。任一场景失败，程序返回非零；产物保存在 `build/e2e-artifacts/<run-id>/`，包括 report、进程日志、指标和 etcd 快照。

## V2 必须成立

- 两个真实 Service 进程、2P+2D（故障后 1P+2D）承载并发请求，安全阶段零失败；
- 非流式与真实 OpenAI SSE 均走生产 HTTP 入口；50ms 在途 deadline 必须在 250ms 内终止，随后恢复流量全成功；
- 有界过载必须同时保留业务进展与结构化 backpressure，状态/Ready 收敛后恢复流量全成功；
- P 进程与租约丢失后仍可选剩余完整 P/D 计划；
- 短 etcd 停顿在 ownership grace 内保持数据面连续；
- 长 etcd 停顿越过 deadline 后旧 Engine 物理退出，首次直接失败不能执行成功，Service 随后返回结构化 `SERVICE_NOT_READY`；
- etcd 恢复后 supervisor 以新 incarnation 重启 Engine，Registry/State/Link/READY 全恢复；
- Leader `SIGKILL` 后备节点完成选举、观察恢复并继续零失败服务；
- STRICT 请求在正常 `SATURATED` 下按可信 dispatch-rate 有界排队，`UNKNOWN` 仍失败关闭；
- Torch CPU tensor 模拟固定 HBM block 所有权，出现非零高水位，最终 block、allocation 和 tensor 内容全部归零。

## V3 必须成立

- 真实 Placement 慢环把严格 vLLM 副本从 1 扩到 3，再 Drain/Terminate 回 1；
- 扩容前高并发必须同时产生业务进展与结构化 backpressure；扩到 3 副本后压力流量必须全成功且覆盖 3 条 route；
- create 成功但响应丢失时只 Query，不重复副作用；最终 CREATE=4（包含一次故障替换）、TERMINATE=2；
- 500ms deadline、真实 SSE 客户端断流和 Agent Cancel 必须释放 Service、Runtime 与 simulated HBM 资源，随后恢复流量全成功；
- Drain 与在飞请求重叠时，Agent 返回精确 fenced rejection，Service 排除原 route 并在有界预算内重选；Drain 不杀死已接纳请求；不能超过物理 `max_devices=3`；
- Agent+Runtime `SIGKILL` 期间只允许一个 at-most-once 在途 transport failure，替换后恢复 3 route 全成功；
- Leader `SIGKILL` 后 durable desired/command/status 继续收敛，恢复流量全成功；
- 每个 Runtime 使用独立 Torch CPU simulated HBM，至少两个副本被实际使用，终态全部清零；
- 报告必须包含请求分布、P50/P95/P99、故障、资源、低基数指标和 durable etcd 证据。

## 证据边界

通过只证明 CPU 上的控制协议、并发、资源账本和分布式故障语义。它不证明真实 NPU HBM、CANN/kernel、DMA/RDMA、模型数值、吞吐容量或生产网络 SLO；这些仍需线上 NPU 阶梯负载、故障矩阵和长时 soak。离线 E2E 未通过时，不允许进入这些后续验证。
