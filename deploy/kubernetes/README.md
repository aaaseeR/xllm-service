# xllm-service Kubernetes deployment

This directory is a deployment contract for the P0 external-routing path. It
targets the llm-d Gateway API Inference Extension (`InferencePool` v1) and
keeps xllm-service as the model-server adapter selected by the EPP.

The base manifests are a validation contract. Production overlays must still
provide cluster-specific images, runtime resources, secrets, and policy.

## Required inputs

Before applying `base/`, replace the values in `base/runtime.env` and
`base/deployment.yaml` for the target environment:

- `XLLM_ETCD_ADDR`: reachable etcd client address.
- `XLLM_ETCD_NAMESPACE`: namespace prefix shared by the service and runtime.
- `XLLM_RUNTIME_PORT`: the port registered by the pod-local aggregated xLLM
  runtime. The Deployment combines it with the Downward API `status.podIP` to
  form the exact `--external_backend_endpoint`. The platform overlay must add
  the xLLM Runtime container to the same pod and start it with that Pod IP as
  `--host` and the configured port as `--port`.
- `XLLM_TOKENIZER_PATH`: tokenizer path available inside the image or mounted
  volume.
- `XLLM_SHUTDOWN_GRACE_PERIOD_S`: request drain window; keep
  `terminationGracePeriodSeconds` larger than this value.
- `xllm-service:REPLACE_WITH_RELEASE_TAG`: replace this base placeholder with an
  immutable image digest containing `xllm_master_serving` on `PATH`.
- `xllm-service-epp:9002`: the EPP Service name and port installed by llm-d.
- CPU and memory requests/limits: the base starts at `1 CPU / 2Gi` requested
  and `4 CPU / 8Gi` limited; size these from tokenizer and concurrency load
  tests before production rollout.

The runtime ConfigMap is generated with a content hash. Changing
`base/runtime.env` updates the name referenced by the pod template and
therefore triggers a Deployment rollout instead of leaving running processes
with stale startup configuration.

The `httproute.yaml` is optional and assumes a Gateway named
`inference-gateway`; apply it only when that Gateway is installed.

## Validate

Install the validation dependency and check the parameterized base before
opening a change:

```bash
python3 -m pip install -r deploy/kubernetes/requirements.txt
python3 deploy/kubernetes/validate.py \
  --base-dir deploy/kubernetes/base \
  --http-route deploy/kubernetes/httproute.yaml
```

The validator is also registered with CTest. It checks the selector, port,
probe, external Runtime identity, rollout, security, NetworkPolicy, PDB,
InferencePool, HTTPRoute, and graceful-shutdown invariants.

For a production overlay, materialize its effective values in a copy of this
base and run the same command with `--production`. Production validation
requires the image to use `registry/repository@sha256:<digest>`, rejects the
base image placeholder, and requires the HTTP 8888 NetworkPolicy rule to name
explicit platform sources. Use `--expected-topology=pd` for a P/D overlay; the
default expected topology is `aggregated`. Kubernetes API schema validation
remains a separate release step and must run against the target cluster CRDs.

## Apply

```bash
kubectl apply -k deploy/kubernetes/base
kubectl apply -f deploy/kubernetes/httproute.yaml
```

Verify rollout and readiness before sending traffic:

```bash
kubectl rollout status deployment/xllm-service
kubectl get inferencepool/xllm-service
kubectl get pods -l app.kubernetes.io/name=xllm-service
```

The Service and InferencePool expose only HTTP port 8888. Port 8889 is the
internal xllm-service RPC endpoint and is not exposed through the Service or
used as the Gateway backend. The container handles `SIGTERM` by returning
`503` from `/readyz`, rejecting new
inference requests, and waiting up to `--shutdown_grace_period_s` for active
requests. Keep `terminationGracePeriodSeconds` larger than that value.

No shell-based `preStop` hook is required: the binary is PID 1 and handles the
Kubernetes `SIGTERM` directly. This keeps the base compatible with distroless
release images. The PDB keeps at least one adapter available during voluntary
disruption when the default two replicas are used. A soft hostname topology
spread keeps replicas on different nodes when capacity permits.

The pod does not mount a Kubernetes API token. The adapter does not query the
API server; InferencePool discovery belongs to the EPP. The container also
drops Linux capabilities, disallows privilege escalation, and uses the runtime
default seccomp profile. Put accelerator access in the xLLM Runtime workload,
not in this adapter Deployment.

This is intentionally a parameterized base, not a release image or an EPP
installation. The image digest, tokenizer volume, etcd service, Gateway, and
EPP chart remain platform-owned inputs.

Do not replace the Pod IP endpoint with a load-balancing ClusterIP Service.
External routing validates the exact runtime identity and incarnation selected
for a request. A non-local runtime deployment therefore needs a stable one-to-one
endpoint for each adapter pod, not a Service that can resolve to multiple runtime
pods.

The optional HTTPRoute matches only the supported chat, completion, and models
`/v1` APIs. Embeddings are not exposed because the current HTTP implementation
returns unsupported. The route does not expose `/metrics`, `/debug/summary`,
`/livez`, or `/readyz` through the Gateway. Metrics collection and operational
diagnostics must use cluster-local access with platform authorization.

For P/D topology, the ingress Gateway/EPP must remove client-supplied
`x-llm-d-routing-decision-version`, `x-llm-d-prefill-endpoint`,
`x-llm-d-decode-endpoint`, and `x-llm-d-routing-attempt` headers before it
injects a trusted version-1 decision. The selected prefill endpoint must equal
the adapter's configured Pod IP Runtime owner. A typed stale 503 must be sent
back through Gateway for a fresh EPP decision; retrying the same internal
routing headers is invalid.

The base NetworkPolicy permits inbound TCP 8888. It permits RPC 8889 only from
pods in the same namespace with the adapter pool labels, because the current
xLLM Runtime connects to the elected adapter master for heartbeat and control
traffic. RPC 8889 is still absent from the Service and HTTPRoute. A same-pod
Runtime inherits the adapter pod labels and satisfies this rule. Production
overlays must narrow HTTP sources to the Gateway, EPP, monitoring, and probe
identities using their real namespace and pod labels.
