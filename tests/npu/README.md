# NPU black-box experiments

These scripts validate the xLLM Service and xLLM runtime only. They do not
start, probe, or judge an external deployment component. V3 scripts are
limited to `SHADOW`; CREATE, DRAIN, and TERMINATE are intentionally outside
this suite.

## Prerequisites

Start etcd with its v3 HTTP API enabled. Build both repositories and export a
model path and card list:

```bash
export MODEL_PATH=/path/to/model
export NPU_IDS=0,1
export ETCD_ADDR=127.0.0.1:2379
export ETCD_NAMESPACE=xllm-npu-$(uname -n)-$(date +%s)
```

The namespace must be reused by all scripts in one experiment. It is deleted
only when `CLEAN_NAMESPACE=1` (the default), and only exact keys under that
namespace are removed.

## Local sequence

Run the following on one host:

```bash
cd /export/home/zhangyi.932/new_service/xllm-service/tests/npu
bash 00_preflight.sh
bash 01_single_native.sh
bash 02_single_pd.sh
bash 04_v3_shadow.sh
bash cleanup.sh
```

`01_single_native.sh` uses one card. `02_single_pd.sh` uses two cards and
tests PREFILL and DECODE; it deliberately does not start MIX.

For V3 SHADOW, set `V3_CONFIG` to a strict config whose Native P/D pool
`model_revision` and `profile_digest` match the running engines:

```bash
export V3_CONFIG=/absolute/path/to/xllm_service_v3_placement_config.json
bash 04_v3_shadow.sh
```

The script checks leader election, placement metrics, request availability,
etcd command records, and leader failover. It does not contact any external
deployment component.

## Cross-host P/D

Run the same script on both hosts with the same `ETCD_ADDR`,
`ETCD_NAMESPACE`, `MODEL_PATH`, and `REQUEST_MODEL`. Use real routable host
addresses in `HOST_IP`.

On the PREFILL host:

```bash
ROLE=PREFILL HOST_IP=10.0.0.11 NPU_ID=0 START_SERVICE=1 \
  bash 03_cross_host_pd.sh
```

On the DECODE host:

```bash
ROLE=DECODE HOST_IP=10.0.0.12 NPU_ID=0 START_SERVICE=0 CLEAN_NAMESPACE=0 \
  bash 03_cross_host_pd.sh
```

The PREFILL host waits for both roles, sends the request to its local Service,
and stores the cross-host etcd, request, process, port, and NPU snapshots.

## Multi-node tensor parallel

`05_multinode_tp.sh` is a worker script. Run it concurrently on every rank,
using the same `ETCD_NAMESPACE`, `MASTER_NODE_ADDR`, `NNODES`, model, and
network configuration. Only rank 0 starts the Service and sends the request:

```bash
# host 10.0.0.11
NNODES=2 NODE_RANK=0 HOST_IP=10.0.0.11 MASTER_NODE_ADDR=10.0.0.11:32100 \
NPU_ID=0 START_SERVICE=1 CLEAN_NAMESPACE=1 RUN_REQUEST=1 \
bash 05_multinode_tp.sh

# host 10.0.0.12
NNODES=2 NODE_RANK=1 HOST_IP=10.0.0.12 MASTER_NODE_ADDR=10.0.0.11:32100 \
NPU_ID=0 START_SERVICE=0 CLEAN_NAMESPACE=0 \
bash 05_multinode_tp.sh
```

## Automatic runner and report

`run_suite.sh` waits for etcd and, when `NPU_IDS` is not set, waits for the
requested number of cards. The default `REQUIRED_CARDS=2` is enough for the
local Native/P/D/SHADOW sequence:

```bash
export REQUIRED_CARDS=2
export WAIT_INTERVAL=30
./run_suite.sh
```

Card discovery uses `npu-smi info` and treats cards below
`NPU_FREE_HBM_MB` used HBM as free. If the site has a different `npu-smi`
format, set `NPU_DISCOVERY_CMD` to a command that prints one available card ID
per line. `RESOURCE_WAIT_TIMEOUT=0` (the default) waits indefinitely.

The runner keeps the same `RUN_ID` and namespace across stages. It writes
`functional_verification_report.md` and
`functional_verification_report.json` after cleanup. Set `STOP_ON_FAIL=0` to
continue collecting later stages after an earlier failure.

## Collected evidence

Each run writes under `run/<RUN_ID>/`: command lines and PIDs, Service logs,
request JSON/response/status/latency, Prometheus metrics, decoded etcd keys and
values, process tables, listening ports, and `npu-smi` output. Every script
returns nonzero when an assertion fails.

During each running stage, `telemetry/<stage>/` records periodic Service and
engine metrics, readiness codes, namespace-scoped etcd snapshots, and NPU status. The
default interval is five seconds and can be changed with
`TELEMETRY_INTERVAL`.

The scripts do not use `pkill` or broad process matching. `cleanup.sh` only
terminates PID groups recorded in the current run directory.
