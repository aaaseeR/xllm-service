# Gateway retry smoke client

`retry_client.py` is an executable reference for the request retry contract
between a caller and the llm-d Gateway. It is deliberately implemented with
the Python standard library so it can run in CI and release environments
without installing a client SDK.

It retries only an HTTP 503 response carrying both
`x-llm-d-retryable: true` and
`x-llm-d-error-code: stale_routing_decision`. The total number of attempts and
the total deadline are bounded. The request body and `x-request-id` remain
stable across attempts. Any other status, including an untyped 503, is
returned immediately.

Each retry targets the Gateway again so EPP can produce a fresh routing
decision. The client rejects internal P/D routing headers supplied on its
command line; those headers must be stripped from untrusted ingress and
injected by Gateway/EPP. Once an HTTP 200 response is received, the response
stream is returned without buffering and is never replayed.

Example:

```bash
python3 deploy/smoke/retry_client.py \
  --url http://gateway.example/v1/completions \
  --body-file request.json \
  --header 'Authorization:Bearer token' \
  --max-attempts 2 \
  --timeout-s 30
```

Run the local fault-injection suite with:

```bash
python3 deploy/smoke/retry_client_test.py
```
