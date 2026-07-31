# 编译与运行

## 容器
首先下载我们提供的镜像：
```bash
docker pull xllm-ai/xllm-0.6.0-dev-800I-A3-py3.11-openeuler24.03-lts-aarch64
```
然后创建对应的容器
```bash
sudo docker run -it --ipc=host -u 0 --privileged --name mydocker --network=host  --device=/dev/davinci0  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc -v /var/queue_schedule:/var/queue_schedule -v /mnt/cfs/9n-das-admin/llm_models:/mnt/cfs/9n-das-admin/llm_models -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi -v /usr/local/sbin/:/usr/local/sbin/ -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf -v /var/log/npu/slog/:/var/log/npu/slog -v /export/home:/export/home -w /export/home -v ~/.ssh:/root/.ssh  -v /var/log/npu/profiling/:/var/log/npu/profiling -v /var/log/npu/dump/:/var/log/npu/dump -v /home/:/home/  -v /runtime/:/runtime/  xllm-ai:xllm-0.6.0-dev-800I-A3-py3.11-openeuler24.03-lts-aarch64
```

## 编译
```bash
git clone https://github.com/jd-opensource/xllm-service
cd xllm_service
git submodule init
git submodule update
```

### etcd安装
使用etcd官方提供的[安装脚本](https://github.com/etcd-io/etcd/releases)进行安装，其脚本提供的默认安装路径是`/tmp/etcd-download-test/etcd`，我们可以手动修改其脚本中的安装路径，也可以运行完脚本之后手动迁移：
```bash
mv /tmp/etcd-download-test/etcd /path/to/your/etcd
```

### 添加补丁
etcd_cpp_apiv3 依赖 cpprest 静态库，但 cpprest 编译产生的是动态库，因此需要给 cpprest 的 CMakeLists.txt 加一个补丁：
```bash
bash prepare.sh
```

### xLLM Service编译
再执行编译:
```bash
mkdir -p build
cd build
cmake ..
make -j 8
cd ..
```
!!! warning "可能的错误"
    这里能会遇到关于`boost-locale`和`boost-interprocess`的安装错误：`vcpkg-src/packages/boost-locale_x64-linux/include: No such     file or directory`,`/vcpkg-src/packages/boost-interprocess_x64-linux/include: No such file or directory`
    我们使用`vcpkg`重新安装这些包:
    ```bash
    /path/to/vcpkg remove boost-locale boost-interprocess
    /path/to/vcpkg install boost-locale:x64-linux
    /path/to/vcpkg install boost-interprocess:x64-linux
    ```

## 运行
1. 首先需要启动etcd服务:
```bash 
./etcd-download-test/etcd --listen-peer-urls 'http://localhost:2390'  --listen-client-urls 'http://localhost:2389' --advertise-client-urls  'http://localhost:2391'
```

2. 然后启动service服务:
```bash
ENABLE_DECODE_RESPONSE_TO_SERVICE=0 \
ENABLE_XLLM_DEBUG_LOG=1 \
./build/xllm_service/xllm_master_serving \
    --etcd_addr="127.0.0.1:2389" \
    --http_server_port=9888 \
    --rpc_server_port=9889 \
    --tokenizer_path /path/to/tokenizer_config/
```

xllm-service需要启动一个http服务和一个rpc服务，http服务用于对外接收与处理用户请求，rpc服务用于和xllm实例进行交互。

### 路由模式

默认的 `legacy` 模式保留 xllm-service 的负载均衡和 KV 感知路由能力，
该模式下不能设置 `external_backend_endpoint`。

当 llm-d 已经选定当前适配器 Pod 时，使用 `external` 模式：

```bash
./build/xllm_service/xllm_master_serving \
    --routing_mode=external \
    --external_routing_topology=aggregated \
    --external_backend_endpoint="xllm-runtime:8000" \
    --etcd_addr="127.0.0.1:2389" \
    --http_server_port=9888 \
    --rpc_server_port=9889 \
    --tokenizer_path=/path/to/tokenizer_config/
```

`external_backend_endpoint` 必须与一个聚合式（`DEFAULT`）xLLM Runtime
注册的 `host:port` 名称完全一致。外部模式下 xllm-service 不会重新选择
Endpoint，也不会维护旧的 KV 路由索引；在指定 Runtime 完成注册并恢复健康前，
就绪探针保持不可用。

外部 P/D 选路使用 `--external_routing_topology=pd`。此时配置的后端仍是当前
adapter 自己持有的 Prefill Runtime。Gateway/EPP 必须先删除客户端携带的内部
路由头，再为每个请求注入以下可信头：

| Header | 必需值 |
| --- | --- |
| `x-llm-d-routing-decision-version` | `1` |
| `x-llm-d-prefill-endpoint` | 与 adapter owner 配置完全相同 |
| `x-llm-d-decode-endpoint` | 不同于 Prefill 的已注册 Decode Runtime |
| `x-llm-d-routing-attempt` | 非负重试序号（可选） |

xllm-service 在发送前校验角色、Endpoint 健康状态、incarnation、block size、
KV hash seed 和 KV split size。格式错误或发送到错误 owner 的指令返回 HTTP 400
和 `x-llm-d-error-code: invalid_routing_directive`；发送前已经过期的决策返回
HTTP 503、`x-llm-d-error-code: stale_routing_decision`、
`x-llm-d-retryable: true` 和 `Retry-After: 0`。Gateway 必须在有限重试次数和
总 deadline 内重新请求 EPP 决策，不能复用原内部 P/D 路由头。可执行契约见
[重试 smoke client](../../deploy/smoke/README.md)。

xllm-service 与 xLLM 配套请求协议现在会传递期望的 Decode incarnation。Prefill
在发起分离式 RPC 前拒绝已经变化的 Decode 注册，Decode 也会在构造请求和分配 KV
block 前，用自己的 XService incarnation 校验期望值。仅为兼容旧发送方，空期望值在
滚动升级期间仍被接受；生产 P/D 必须固定匹配的 service 与 Runtime 构建版本，并在
开启功能门禁前通过预发 Decode 替换竞态测试。

### 生产退出

master 进程收到 `SIGTERM` 后会进入 draining 状态，`/readyz` 立即返回 `503`，
同时拒绝新的推理请求，并允许正在执行的请求完成。可通过
`--shutdown_grace_period_s` 设置平台允许的最大退出等待时间（默认 30 秒）。
超过该时间仍未完成的请求会被取消，进程随后退出。Kubernetes 的
`terminationGracePeriodSeconds` 应大于该参数，确保进程有足够时间完成清理。

参数化的 Deployment、Service、PDB、InferencePool 以及可选 HTTPRoute 清单见
[Kubernetes 部署契约](../../deploy/kubernetes/README.md)。

完整的使用流程需要结合xllm一起使用，请查看链接: [xLLM PD分离部署](https://xllm.readthedocs.io/zh-cn/latest/zh/getting_started/PD_disagg/)

### service参数
http服务：用于对外接收以及处理用户请求。
| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| http_server_host | http 服务地址 | "" |
| http_server_port | http 服务端口 | 8888 |
| http_server_idle_timeout_s | http 服务超时时间 | -1 |
| http_server_num_threads | http 服务线程数 | 32 |
| http_server_max_concurrency | http 服务最大请求并发数 | 128 |
| shutdown_grace_period_s | 活动请求的优雅退出等待时间（秒） | 30 |
| routing_mode | 路由权威：`legacy` 或 `external` | legacy |
| external_routing_topology | 外部拓扑：`aggregated` 或 `pd` | aggregated |
| external_backend_endpoint | external 模式下 adapter 持有的 Runtime `host:port` | "" |

rpc服务：用于与xllm之间交互，管理xllm实例集群状态等。
| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| rpc_server_host | rpc 服务地址 | "" |
| rpc_server_port | rpc 服务端口 | 8889 |
| rpc_server_idle_timeout_s | rpc 服务超时时间 | -1 |
| rpc_server_num_threads | rpc 服务线程数 | 32 |
| rpc_server_max_concurrency | rpc 服务最大请求并发数 | 128 |

环境参数:
ENABLE_DECODE_RESPONSE_TO_SERVICE: 在PD分离场景下，是否将解码结果直接返回给service(不需要经过P实例转发)，0表示“否”，1表示“是”。
ENABLE_XLLM_DEBUG_LOG: 是否开启xllm debug log，0表示不开启，1表示开启。
