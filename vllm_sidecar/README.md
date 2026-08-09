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

# vLLM-Ascend Provider Agent

`vllm_sidecar` has two deliberately separate operating modes:

- **strict V2 Agent** (production target): the only externally reachable vLLM
  ingress, contract-v1 registration, incarnation fencing, bounded attempt
  Query/Cancel, request deadline enforcement and per-DP EngineState.
- **legacy sidecar** (compatibility only): lease registration plus aggregate
  metrics. It publishes contract version 0 and is never a strict V2 route.

The strict Agent runs in the same failure domain as one local vLLM-Ascend
runtime. Raw vLLM ingress must be isolated from xllm-service and clients; all
inference traffic goes through the Agent endpoint.

## Strict V2 lifecycle

1. Load and validate a verified provider profile. Missing identity, model,
   topology, KV, scheduler or failure-domain facts fail startup.
2. Start the Agent fenced and wait for the local vLLM `/health` endpoint.
3. Grant an etcd lease, publish the complete contract-v1 Descriptor, then
   activate a fresh `incarnation_id`.
4. Accept each `(request_uid, attempt_seq)` at most once. The bounded ledger
   preserves terminal tombstones and implements Query, Cancel and
   Cancel-before-create.
5. Forward only admitted inference traffic, inject a stable vLLM request ID,
   enforce the remaining deadline locally, and close the upstream request on
   cancellation, client disconnect, expiry or fencing.
6. Publish monotonic EngineState heartbeats with per-DP state. A lost lease,
   unhealthy runtime, stale registration or shutdown fences ingress before
   deregistration/re-registration.

If the Agent is killed without cleanup, the lease expires within
`--lease-ttl`. Deployment must additionally enforce the configured
`runtime.fate_bound_mode` (`parent_death_signal` or `same_restart_unit`) so an
orphaned raw runtime cannot remain reachable.

## Install and CPU test

```bash
pip install -r vllm_sidecar/requirements.txt
python -m pytest -q vllm_sidecar/tests
```

The CPU suite uses a loopback HTTP runtime and covers descriptor validation,
profile digest stability, duplicate submission, cancellation races, terminal
tombstones, deadlines, incarnation replacement, streaming proxying, per-DP
metric parsing and strict heartbeat production. It does not claim NPU runtime
or deployment failure-domain validation.

## Run strict V2 Agent

Copy `provider_config.example.json`, replace every digest/version/topology
field with deployment facts, and start:

```bash
python -m vllm_sidecar.sidecar \
  --provider-config vllm_sidecar/provider_config.json \
  --vllm-url http://127.0.0.1:18000 \
  --agent-listen 0.0.0.0:18100 \
  --register-addr 0.0.0.0:18100 \
  --etcd-endpoints 127.0.0.1:2379 \
  --xllm-service-url http://127.0.0.1:9998
```

`--agent-listen` and `--register-addr` must be identical. The latter is a bare
`host:port`, not a URL. For a real deployment, advertise a routable address
instead of `0.0.0.0`; the strict equality requirement prevents accidentally
registering the raw vLLM port.

### Strict inference contract

xllm-service sends these headers on Chat/Completion traffic:

| Header | Meaning |
| --- | --- |
| `X-Request-UID` | globally stable request identity |
| `X-Attempt-Seq` | attempt identity; duplicate submission is rejected |
| `X-Incarnation-ID` | exact target Agent incarnation; stale submissions are rejected atomically |
| `X-Remaining-Deadline-Ms` | remaining local duration; zero/expired is rejected |

The Agent exposes incarnation-scoped control endpoints:

```text
POST /v1/internal/attempt/query
POST /v1/internal/attempt/cancel
{"request_uid":"...","attempt_seq":0,"incarnation_id":"..."}
```

Only a terminal Query result whose `request_uid`, `attempt_seq` and
`incarnation_id` exactly match the queried attempt, or an acknowledged Cancel
fence for that identity, allows xllm-service to release an unresolved
aggregated execution hold.

Strict mode proxies only `/v1/chat/completions`, `/v1/completions`, and the
read-only `/v1/models`. Other vLLM paths, including `/v1/messages`,
`/v1/responses`, embeddings and vendor raw generation endpoints, return 404
instead of bypassing the attempt ledger. They may be opened only together with
their Service Adapter, API capability filter, deadline/cancel path and tests.

### Health endpoints

| Endpoint | Contract |
| --- | --- |
| `/livez` | Agent process is alive |
| `/health`, `/readyz` | current incarnation is registered and accepting |

## Provider profile

The verified JSON contains five required sections: `runtime`, `model`,
`topology`, `kv`, and `scheduler`. The Agent canonicalizes the complete profile
and publishes its SHA-256 digest in both Descriptor and EngineState. It refuses
unknown fate-binding modes, raw-ingress exposure, zero capacities and missing
compatibility facts.

The example is a schema sample, not a production attestation. In particular,
model/tokenizer/template/renderer, KV layout/sharding, Connector and scheduler
digests must be generated from the actual deployment.

## Metrics and EngineState

The Agent scrapes vLLM Prometheus metrics and preserves DP labels
(`data_parallel_rank`, `dp_rank`, or `engine`). It publishes per-DP running,
waiting, deferred, KV usage and admission credit. State quality is `FULL` only
when all configured DP ranks are present; otherwise it is `PARTIAL`. Ratios are
never summed across DP ranks.

Heartbeats are sent to `/v1/internal/heartbeat`. Configure the same internal
token on xllm-service and the Agent (`--internal-token` or
`XLLM_INTERNAL_TOKEN`) and expose this endpoint only on a trusted network.
Liveness remains lease-based; metrics failures do not rewrite the etcd key.

## Legacy compatibility mode

Omit `--provider-config` and `--agent-listen` to retain the old lease/aggregate
metrics bridge:

```bash
python -m vllm_sidecar.sidecar \
  --vllm-url http://127.0.0.1:18000 \
  --register-addr 127.0.0.1:18000 \
  --etcd-endpoints 127.0.0.1:2379
```

This mode intentionally registers the raw vLLM address, has no attempt ledger,
and publishes contract version 0. It is BEST_EFFORT compatibility only and
must not be promoted to a strict V2 route.

## Remaining validation

CPU verification proves the bounded control logic and loopback HTTP behavior.
Before `VERIFIED`, run real vLLM-Ascend/NPU conformance, raw-ingress network
isolation checks, SIGKILL/fate-binding tests, etcd lease-loss/restart injection,
deadline/cancel resource-release tests and sustained load/capacity validation.
