#!/usr/bin/env python3

# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gateway client implementing the xllm-service typed stale retry contract."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


STALE_ERROR_CODE = "stale_routing_decision"
RESERVED_ROUTING_HEADERS = frozenset(
    {
        "x-llm-d-routing-decision-version",
        "x-llm-d-prefill-endpoint",
        "x-llm-d-decode-endpoint",
        "x-llm-d-routing-attempt",
    }
)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 2
    total_timeout_s: float = 30.0

    def validate(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.total_timeout_s <= 0:
            raise ValueError("total_timeout_s must be positive")


@dataclass(frozen=True)
class RetryOutcome:
    response: object
    attempts: int
    retry_budget_exhausted: bool = False


RequestFn = Callable[[urllib.request.Request, float], object]


def _default_request(request: urllib.request.Request, timeout_s: float) -> object:
    try:
        return urllib.request.urlopen(request, timeout=timeout_s)
    except urllib.error.HTTPError as error:
        return error


def _status(response: object) -> int:
    status = getattr(response, "status", None)
    if status is None:
        status = getattr(response, "code")
    return int(status)


def _headers(response: object) -> Mapping[str, str]:
    return getattr(response, "headers")


def is_typed_stale_response(response: object) -> bool:
    headers = _headers(response)
    return (
        _status(response) == 503
        and headers.get("x-llm-d-retryable", "").lower() == "true"
        and headers.get("x-llm-d-error-code", "").lower()
        == STALE_ERROR_CODE
    )


def _retry_after_s(response: object) -> float:
    value = _headers(response).get("Retry-After", "0").strip()
    try:
        delay = int(value, 10)
    except ValueError:
        return 0.0
    return float(max(0, delay))


def _validate_headers(headers: Mapping[str, str]) -> None:
    supplied = RESERVED_ROUTING_HEADERS.intersection(
        name.lower() for name in headers
    )
    if supplied:
        names = ", ".join(sorted(supplied))
        raise ValueError(
            "internal routing headers must be injected by Gateway/EPP: "
            + names
        )


def send_with_typed_stale_retry(
    url: str,
    body: bytes,
    headers: Mapping[str, str],
    policy: RetryPolicy,
    *,
    request_fn: RequestFn = _default_request,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> RetryOutcome:
    """Send one logical request and retry only a typed stale 503 response."""

    policy.validate()
    _validate_headers(headers)
    request_headers = dict(headers)
    if not any(name.lower() == "x-request-id" for name in request_headers):
        request_headers["x-request-id"] = str(uuid.uuid4())

    deadline = clock() + policy.total_timeout_s
    for attempt in range(1, policy.max_attempts + 1):
        remaining_s = deadline - clock()
        if remaining_s <= 0:
            raise TimeoutError("request retry deadline expired before send")

        request = urllib.request.Request(
            url,
            data=body,
            headers=request_headers,
            method="POST",
        )
        response = request_fn(request, remaining_s)
        if not is_typed_stale_response(response):
            return RetryOutcome(response=response, attempts=attempt)
        if attempt == policy.max_attempts:
            return RetryOutcome(
                response=response,
                attempts=attempt,
                retry_budget_exhausted=True,
            )

        delay_s = _retry_after_s(response)
        if delay_s >= deadline - clock():
            return RetryOutcome(
                response=response,
                attempts=attempt,
                retry_budget_exhausted=True,
            )
        response.close()
        if delay_s > 0:
            sleep(delay_s)

    raise AssertionError("retry loop terminated without a response")


def _parse_header(value: str) -> tuple[str, str]:
    name, separator, header_value = value.partition(":")
    if not separator or not name.strip():
        raise argparse.ArgumentTypeError("header must use NAME:VALUE format")
    return name.strip(), header_value.strip()


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send an inference request through Gateway with a bounded "
        "typed-stale retry policy."
    )
    parser.add_argument("--url", required=True)
    parser.add_argument("--body-file", type=argparse.FileType("rb"))
    parser.add_argument(
        "--header",
        action="append",
        type=_parse_header,
        default=[],
    )
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    body = args.body_file.read() if args.body_file is not None else b"{}"
    headers = dict(args.header)
    headers.setdefault("Content-Type", "application/json")
    try:
        outcome = send_with_typed_stale_retry(
            args.url,
            body,
            headers,
            RetryPolicy(args.max_attempts, args.timeout_s),
        )
    except (OSError, TimeoutError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2

    response = outcome.response
    with response:
        shutil.copyfileobj(response, sys.stdout.buffer)
    return 0 if 200 <= _status(response) < 300 else 1


if __name__ == "__main__":
    raise SystemExit(main())
