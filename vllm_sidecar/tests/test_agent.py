# Copyright 2026 The xLLM Authors. All Rights Reserved.

from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import pytest
import requests

from vllm_sidecar.agent import AgentRuntime


class UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self._reply({"path": self.path})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        self._reply(body)

    def _reply(self, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args) -> None:
        pass


class DelayedUpstreamHandler(UpstreamHandler):
    received = threading.Event()
    release = threading.Event()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        self.received.set()
        self.release.wait(timeout=2.0)
        self._reply(body)


def _post_attempt(base: str, path: str, uid: str, seq: int) -> requests.Response:
    return requests.post(
        base + path,
        json={
            "request_uid": uid,
            "attempt_seq": seq,
            "incarnation_id": "inc-1",
        },
        timeout=2.0,
    )


def test_agent_proxy_attempt_query_cancel_and_fencing() -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream.daemon_threads = True
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    upstream_base = f"http://127.0.0.1:{upstream.server_address[1]}"

    agent = AgentRuntime("127.0.0.1:0", upstream_base)
    agent.start()
    agent.activate("inc-1")
    base = "http://" + agent.listen_address
    try:
        assert requests.get(base + "/health", timeout=2.0).status_code == 200
        models = requests.get(base + "/v1/models?scope=all", timeout=2.0)
        assert models.status_code == 200
        assert models.json()["path"] == "/v1/models?scope=all"
        assert (
            requests.post(base + "/v1/models", json={}, timeout=2.0).status_code
            == 405
        )
        response = requests.post(
            base + "/v1/chat/completions",
            json={"model": "m", "messages": []},
            headers={
                "X-Request-UID": "request-1",
                "X-Attempt-Seq": "0",
                "X-Incarnation-ID": "inc-1",
                "X-Remaining-Deadline-Ms": "2000",
            },
            timeout=2.0,
        )
        assert response.status_code == 200
        assert response.json()["request_id"] == "xllm-request-1-0"
        query = _post_attempt(
            base, "/v1/internal/attempt/query", "request-1", 0
        )
        query_body = query.json()
        assert query_body["state"] == "ATTEMPT_LIFECYCLE_STATE_DONE"
        assert query_body["request_uid"] == "request-1"
        assert query_body["attempt_seq"] == 0
        assert query_body["incarnation_id"] == "inc-1"

        duplicate = requests.post(
            base + "/v1/chat/completions",
            json={"model": "m"},
            headers={
                "X-Request-UID": "request-1",
                "X-Attempt-Seq": "0",
                "X-Incarnation-ID": "inc-1",
                "X-Remaining-Deadline-Ms": "2000",
            },
            timeout=2.0,
        )
        assert duplicate.status_code == 409

        cancel = _post_attempt(
            base, "/v1/internal/attempt/cancel", "request-2", 0
        )
        cancel_body = cancel.json()
        assert cancel_body["state"] == (
            "ATTEMPT_LIFECYCLE_STATE_CANCELLED_BEFORE_CREATE"
        )
        assert cancel_body["request_uid"] == "request-2"
        assert cancel_body["attempt_seq"] == 0
        assert cancel_body["incarnation_id"] == "inc-1"
        stale = requests.post(
            base + "/v1/internal/attempt/query",
            json={
                "request_uid": "request-1",
                "attempt_seq": 0,
                "incarnation_id": "old-inc",
            },
            timeout=2.0,
        )
        assert stale.status_code == 409
        invalid_identity = requests.post(
            base + "/v1/internal/attempt/cancel",
            json={
                "request_uid": "",
                "attempt_seq": True,
                "incarnation_id": "inc-1",
            },
            timeout=2.0,
        )
        assert invalid_identity.status_code == 400
        assert (
            requests.delete(
                base + "/v1/internal/attempt/cancel", timeout=2.0
            ).status_code
            == 405
        )
        stale_submit = requests.post(
            base + "/v1/chat/completions",
            json={"model": "m"},
            headers={
                "X-Request-UID": "stale-submit",
                "X-Attempt-Seq": "0",
                "X-Incarnation-ID": "old-inc",
                "X-Remaining-Deadline-Ms": "2000",
            },
            timeout=2.0,
        )
        assert stale_submit.status_code == 409
        assert agent.ledger.query("stale-submit", 0, "inc-1").state == (
            "ATTEMPT_LIFECYCLE_STATE_ABSENT"
        )
        invalid_submit = requests.post(
            base + "/v1/completions",
            json={"model": "m", "prompt": "p"},
            headers={
                "X-Request-UID": "bounded-submit",
                "X-Attempt-Seq": str(1 << 64),
                "X-Incarnation-ID": "inc-1",
                "X-Remaining-Deadline-Ms": "2000",
            },
            timeout=2.0,
        )
        assert invalid_submit.status_code == 400
        missing_incarnation = requests.post(
            base + "/v1/completions",
            json={"model": "m", "prompt": "p"},
            headers={
                "X-Request-UID": "missing-incarnation",
                "X-Attempt-Seq": "0",
                "X-Remaining-Deadline-Ms": "2000",
            },
            timeout=2.0,
        )
        assert missing_incarnation.status_code == 400
        invalid_json = requests.post(
            base + "/v1/completions",
            data=b"{",
            headers={
                "Content-Type": "application/json",
                "X-Request-UID": "invalid-json",
                "X-Attempt-Seq": "0",
                "X-Incarnation-ID": "inc-1",
                "X-Remaining-Deadline-Ms": "2000",
            },
            timeout=2.0,
        )
        assert invalid_json.status_code == 400
        assert agent.ledger.query("invalid-json", 0, "inc-1").state == (
            "ATTEMPT_LIFECYCLE_STATE_FAILED"
        )
        unsupported = requests.post(
            base + "/v1/messages",
            json={"model": "m", "messages": []},
            headers={
                "X-Request-UID": "raw-bypass",
                "X-Attempt-Seq": "0",
                "X-Incarnation-ID": "inc-1",
                "X-Remaining-Deadline-Ms": "2000",
            },
            timeout=2.0,
        )
        assert unsupported.status_code == 404
        assert agent.ledger.query("raw-bypass", 0, "inc-1").state == (
            "ATTEMPT_LIFECYCLE_STATE_ABSENT"
        )

        agent.fence()
        assert requests.get(base + "/health", timeout=2.0).status_code == 503
        assert requests.get(base + "/livez", timeout=2.0).status_code == 200
    finally:
        agent.stop()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=2.0)


def test_cancel_wins_while_submit_result_is_unknown() -> None:
    DelayedUpstreamHandler.received.clear()
    DelayedUpstreamHandler.release.clear()
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), DelayedUpstreamHandler)
    upstream.daemon_threads = True
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    agent = AgentRuntime(
        "127.0.0.1:0", f"http://127.0.0.1:{upstream.server_address[1]}"
    )
    agent.start()
    agent.activate("inc-1")
    base = "http://" + agent.listen_address
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            submission = executor.submit(
                requests.post,
                base + "/v1/chat/completions",
                json={"model": "m", "messages": []},
                headers={
                    "X-Request-UID": "cancel-race",
                    "X-Attempt-Seq": "3",
                    "X-Incarnation-ID": "inc-1",
                    "X-Remaining-Deadline-Ms": "2000",
                },
                timeout=2.0,
            )
            assert DelayedUpstreamHandler.received.wait(timeout=1.0)
            cancel = _post_attempt(
                base, "/v1/internal/attempt/cancel", "cancel-race", 3
            )
            assert cancel.json()["state"] == "ATTEMPT_LIFECYCLE_STATE_CANCELLED"
            assert submission.result(timeout=0.5).status_code == 409

        duplicate = requests.post(
            base + "/v1/chat/completions",
            json={"model": "m", "messages": []},
            headers={
                "X-Request-UID": "cancel-race",
                "X-Attempt-Seq": "3",
                "X-Incarnation-ID": "inc-1",
                "X-Remaining-Deadline-Ms": "2000",
            },
            timeout=2.0,
        )
        assert duplicate.status_code == 409
    finally:
        DelayedUpstreamHandler.release.set()
        agent.stop()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=2.0)


def test_local_deadline_fences_delayed_upstream_acceptance() -> None:
    DelayedUpstreamHandler.received.clear()
    DelayedUpstreamHandler.release.clear()
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), DelayedUpstreamHandler)
    upstream.daemon_threads = True
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    agent = AgentRuntime(
        "127.0.0.1:0", f"http://127.0.0.1:{upstream.server_address[1]}"
    )
    agent.start()
    agent.activate("inc-1")
    base = "http://" + agent.listen_address
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            submission = executor.submit(
                requests.post,
                base + "/v1/chat/completions",
                json={"model": "m", "messages": []},
                headers={
                    "X-Request-UID": "deadline-race",
                    "X-Attempt-Seq": "1",
                    "X-Incarnation-ID": "inc-1",
                    "X-Remaining-Deadline-Ms": "50",
                },
                timeout=2.0,
            )
            assert DelayedUpstreamHandler.received.wait(timeout=1.0)
            deadline = time.monotonic() + 1.0
            state = ""
            while time.monotonic() < deadline:
                state = _post_attempt(
                    base, "/v1/internal/attempt/query", "deadline-race", 1
                ).json()["state"]
                if state == "ATTEMPT_LIFECYCLE_STATE_EXPIRED":
                    break
                time.sleep(0.01)
            assert state == "ATTEMPT_LIFECYCLE_STATE_EXPIRED"
            response = submission.result(timeout=0.5)
            assert response.status_code == 504
            assert response.json()["state"] == "ATTEMPT_LIFECYCLE_STATE_EXPIRED"
    finally:
        DelayedUpstreamHandler.release.set()
        agent.stop()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=2.0)


def test_agent_bounds_inflight_proxy_requests() -> None:
    DelayedUpstreamHandler.received.clear()
    DelayedUpstreamHandler.release.clear()
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), DelayedUpstreamHandler)
    upstream.daemon_threads = True
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    agent = AgentRuntime(
        "127.0.0.1:0",
        f"http://127.0.0.1:{upstream.server_address[1]}",
        max_inflight_requests=1,
    )
    agent.start()
    agent.activate("inc-1")
    base = "http://" + agent.listen_address
    headers = {
        "X-Attempt-Seq": "0",
        "X-Incarnation-ID": "inc-1",
        "X-Remaining-Deadline-Ms": "2000",
    }
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(
                requests.post,
                base + "/v1/completions",
                json={"model": "m", "prompt": "p"},
                headers={**headers, "X-Request-UID": "inflight-1"},
                timeout=2.0,
            )
            assert DelayedUpstreamHandler.received.wait(timeout=1.0)
            rejected = requests.post(
                base + "/v1/completions",
                json={"model": "m", "prompt": "p"},
                headers={**headers, "X-Request-UID": "inflight-2"},
                timeout=2.0,
            )
            assert rejected.status_code == 503
            assert agent.ledger.query("inflight-2", 0, "inc-1").state == (
                "ATTEMPT_LIFECYCLE_STATE_ABSENT"
            )
            DelayedUpstreamHandler.release.set()
            assert first.result(timeout=2.0).status_code == 200
    finally:
        DelayedUpstreamHandler.release.set()
        agent.stop()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=2.0)


def test_agent_bounds_aggregate_inflight_body_bytes() -> None:
    DelayedUpstreamHandler.received.clear()
    DelayedUpstreamHandler.release.clear()
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), DelayedUpstreamHandler)
    upstream.daemon_threads = True
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    agent = AgentRuntime(
        "127.0.0.1:0",
        f"http://127.0.0.1:{upstream.server_address[1]}",
        max_inflight_requests=2,
        max_request_body_bytes=128,
        inflight_body_capacity_bytes=128,
    )
    agent.start()
    agent.activate("inc-1")
    base = "http://" + agent.listen_address
    headers = {
        "X-Attempt-Seq": "0",
        "X-Incarnation-ID": "inc-1",
        "X-Remaining-Deadline-Ms": "2000",
    }
    payload = {"model": "m", "prompt": "x" * 80}
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(
                requests.post,
                base + "/v1/completions",
                json=payload,
                headers={**headers, "X-Request-UID": "bytes-1"},
                timeout=2.0,
            )
            assert DelayedUpstreamHandler.received.wait(timeout=1.0)
            rejected = requests.post(
                base + "/v1/completions",
                json=payload,
                headers={**headers, "X-Request-UID": "bytes-2"},
                timeout=2.0,
            )
            assert rejected.status_code == 503
            assert agent.ledger.query("bytes-2", 0, "inc-1").state == (
                "ATTEMPT_LIFECYCLE_STATE_ABSENT"
            )
            DelayedUpstreamHandler.release.set()
            assert first.result(timeout=2.0).status_code == 200
    finally:
        DelayedUpstreamHandler.release.set()
        agent.stop()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=2.0)


def test_cancel_fence_capacity_never_returns_false_ack() -> None:
    agent = AgentRuntime(
        "127.0.0.1:0",
        "http://127.0.0.1:1",
        max_cancel_fences=1,
        negative_fence_ttl_seconds=0.1,
    )
    agent.start()
    agent.activate("inc-1")
    base = "http://" + agent.listen_address
    try:
        installed = _post_attempt(
            base, "/v1/internal/attempt/cancel", "fence-1", 0
        )
        assert installed.status_code == 200
        assert installed.json()["accepted"] is True
        assert requests.get(base + "/health", timeout=2.0).status_code == 503

        rejected = _post_attempt(
            base, "/v1/internal/attempt/cancel", "fence-2", 0
        )
        assert rejected.status_code == 503
        assert rejected.json()["accepted"] is False

        deadline = time.monotonic() + 1.0
        health = 503
        while time.monotonic() < deadline:
            health = requests.get(base + "/health", timeout=2.0).status_code
            if health == 200:
                break
            time.sleep(0.01)
        assert health == 200
        assert _post_attempt(
            base, "/v1/internal/attempt/cancel", "fence-2", 0
        ).status_code == 200
    finally:
        agent.stop()


def test_agent_configuration_fails_closed_and_unstarted_stop_is_safe() -> None:
    with pytest.raises(ValueError):
        AgentRuntime(
            "127.0.0.1:0",
            "http://127.0.0.1:1",
            max_inflight_requests=0,
        )
    with pytest.raises(ValueError):
        AgentRuntime(
            "127.0.0.1:0",
            "http://user:password@127.0.0.1:1",
        )
    with pytest.raises(ValueError):
        AgentRuntime("127.0.0.1:not-a-port", "http://127.0.0.1:1")
    with pytest.raises(ValueError):
        AgentRuntime(
            "127.0.0.1:0",
            "http://127.0.0.1:1",
            internal_token="bad\ntoken",
        )
    with pytest.raises(ValueError):
        AgentRuntime("[::1]:0", "http://127.0.0.1:1")

    agent = AgentRuntime("127.0.0.1:0", "http://127.0.0.1:1")
    agent.stop()


def test_agent_requires_constant_time_internal_authentication() -> None:
    agent = AgentRuntime(
        "127.0.0.1:0",
        "http://127.0.0.1:1",
        internal_token="secret-token",
    )
    agent.start()
    agent.activate("inc-1")
    base = "http://" + agent.listen_address
    body = {
        "request_uid": "auth-query",
        "attempt_seq": 0,
        "incarnation_id": "inc-1",
    }
    try:
        assert requests.post(
            base + "/v1/internal/attempt/query", json=body, timeout=2.0
        ).status_code == 401
        assert requests.post(
            base + "/v1/internal/attempt/query",
            json=body,
            headers={"X-Internal-Token": "wrong-token"},
            timeout=2.0,
        ).status_code == 401
        authorized = requests.post(
            base + "/v1/internal/attempt/query",
            json=body,
            headers={"X-Internal-Token": "secret-token"},
            timeout=2.0,
        )
        assert authorized.status_code == 200
        assert authorized.json()["state"] == "ATTEMPT_LIFECYCLE_STATE_ABSENT"
        assert requests.get(base + "/livez", timeout=2.0).status_code == 200
    finally:
        agent.stop()
