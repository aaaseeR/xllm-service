# Copyright 2025-2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm-service/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Parse labelled vLLM Prometheus metrics into V2 EngineState inputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import re

import requests

logger = logging.getLogger("vllm_sidecar.metrics")

_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][\w:]*)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?[0-9.eE+-]+|NaN|[-+]?Inf)\s*"
)
_LABEL_RE = re.compile(r'(?P<name>[a-zA-Z_][\w]*)\s*=\s*(?P<value>"(?:\\.|[^"\\])*")')

_RUNNING = "vllm:num_requests_running"
_WAITING = "vllm:num_requests_waiting"
_WAITING_CAPACITY = (
    "vllm:num_requests_waiting_for_capacity",
    "vllm:num_requests_waiting_capacity",
)
_WAITING_DEFERRED = (
    "vllm:num_requests_waiting_deferred",
    "vllm:num_requests_swapped",
)
_GPU_CACHE = ("vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc")
_TTFT = "vllm:time_to_first_token_seconds"
_TBT = "vllm:time_per_output_token_seconds"
_DP_LABELS = ("data_parallel_rank", "dp_rank", "engine")


@dataclass(frozen=True)
class MetricSample:
    name: str
    labels: tuple[tuple[str, str], ...]
    value: float

    def label(self, name: str) -> str | None:
        for key, value in self.labels:
            if key == name:
                return value
        return None


def _parse_labels(text: str) -> tuple[tuple[str, str], ...]:
    labels = []
    for match in _LABEL_RE.finditer(text):
        try:
            value = json.loads(match.group("value"))
        except json.JSONDecodeError:
            continue
        labels.append((match.group("name"), value))
    return tuple(sorted(labels))


def _parse_labeled_samples(text: str) -> list[MetricSample]:
    samples = []
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        match = _SAMPLE_RE.match(line)
        if match is None:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        samples.append(
            MetricSample(
                name=match.group("name"),
                labels=_parse_labels(match.group("labels") or ""),
                value=value,
            )
        )
    return samples


def _parse_samples(text: str) -> dict[str, float]:
    """Compatibility helper: sum labelled samples by base metric name."""
    totals: dict[str, float] = {}
    for sample in _parse_labeled_samples(text):
        totals[sample.name] = totals.get(sample.name, 0.0) + sample.value
    return totals


def _rank(sample: MetricSample, dp_size: int) -> int | None:
    for name in _DP_LABELS:
        value = sample.label(name)
        if value is None:
            continue
        match = re.search(r"(\d+)$", value)
        if match is None:
            return None
        rank = int(match.group(1))
        return rank if rank < dp_size else None
    return 0 if dp_size == 1 else None


def _samples_by_name(samples: list[MetricSample]) -> dict[str, list[MetricSample]]:
    result: dict[str, list[MetricSample]] = {}
    for sample in samples:
        result.setdefault(sample.name, []).append(sample)
    return result


def _rank_values(
    by_name: dict[str, list[MetricSample]],
    names: tuple[str, ...],
    dp_size: int,
    aggregate: str,
) -> dict[int, float]:
    result: dict[int, float] = {}
    for name in names:
        for sample in by_name.get(name, []):
            rank = _rank(sample, dp_size)
            if rank is None:
                continue
            if aggregate == "max":
                result[rank] = max(result.get(rank, 0.0), sample.value)
            else:
                result[rank] = result.get(rank, 0.0) + sample.value
        if result:
            break
    return result


class VllmMetricsScraper:
    def __init__(
        self,
        metrics_url: str,
        timeout: float = 3.0,
        dp_size: int = 1,
        max_num_seqs: int = 1,
    ) -> None:
        if dp_size <= 0 or max_num_seqs <= 0:
            raise ValueError("dp_size and max_num_seqs must be positive")
        self._url = metrics_url
        self._timeout = timeout
        self._dp_size = dp_size
        self._max_num_seqs = max_num_seqs
        self._prev: dict[str, tuple[float, float]] = {}
        self._session = requests.Session()

    def _interval_avg_ms(self, samples: dict[str, float], base: str) -> int:
        current = (samples.get(base + "_sum", 0.0), samples.get(base + "_count", 0.0))
        previous = self._prev.get(base)
        self._prev[base] = current
        if previous is None:
            return 0
        count_delta = current[1] - previous[1]
        sum_delta = current[0] - previous[0]
        if count_delta <= 0 or sum_delta < 0:
            return 0
        return max(0, int((sum_delta / count_delta) * 1000.0))

    def _per_dp(self, samples: list[MetricSample]) -> tuple[list[dict], bool]:
        by_name = _samples_by_name(samples)
        running = _rank_values(by_name, (_RUNNING,), self._dp_size, "sum")
        waiting = _rank_values(by_name, _WAITING_CAPACITY, self._dp_size, "sum")
        if not waiting:
            waiting = _rank_values(by_name, (_WAITING,), self._dp_size, "sum")
        deferred = _rank_values(by_name, _WAITING_DEFERRED, self._dp_size, "sum")
        kv_used = _rank_values(by_name, _GPU_CACHE, self._dp_size, "max")

        per_dp = []
        complete = True
        for rank in range(self._dp_size):
            item: dict[str, int | float] = {"dp_rank": rank}
            if rank in running:
                item["running"] = max(0, int(running[rank]))
            else:
                complete = False
            if rank in waiting:
                item["waiting_capacity"] = max(0, int(waiting[rank]))
            else:
                complete = False
            if rank in deferred:
                item["waiting_deferred"] = max(0, int(deferred[rank]))
            if rank in kv_used and 0.0 <= kv_used[rank] <= 1.0:
                item["kv_used_ratio"] = kv_used[rank]
            else:
                complete = False
            if rank in running and rank in waiting:
                occupied = max(0, int(running[rank])) + max(0, int(waiting[rank]))
                item["admission_credit"] = max(0, self._max_num_seqs - occupied)
            per_dp.append(item)
        return per_dp, complete

    def scrape(self) -> dict | None:
        try:
            response = self._session.get(self._url, timeout=self._timeout)
            response.raise_for_status()
        except requests.RequestException as error:
            logger.debug("metrics scrape failed: %s", error)
            return None

        labeled = _parse_labeled_samples(response.text)
        totals: dict[str, float] = {}
        for sample in labeled:
            totals[sample.name] = totals.get(sample.name, 0.0) + sample.value
        per_dp, complete = self._per_dp(labeled)
        cache_values = [
            item["kv_used_ratio"] for item in per_dp if "kv_used_ratio" in item
        ]
        return {
            "load_metrics": {
                "waiting_requests_num": sum(
                    int(item.get("waiting_capacity", 0)) for item in per_dp
                ),
                # The legacy scalar is worst-DP usage. Summing ratios is invalid.
                "gpu_cache_usage_perc": max(cache_values, default=0.0),
            },
            "latency_metrics": {
                "recent_max_ttft": self._interval_avg_ms(totals, _TTFT),
                "recent_max_tbt": self._interval_avg_ms(totals, _TBT),
            },
            "per_dp": per_dp,
            "state_quality": (
                "STATE_QUALITY_FULL" if complete else "STATE_QUALITY_PARTIAL"
            ),
        }
