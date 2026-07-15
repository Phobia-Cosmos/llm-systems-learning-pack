"""Small, dependency-free Prometheus collector used by miniagent benchmarks.

The module intentionally implements only the text exposition features needed by
the vLLM and SGLang ``/metrics`` endpoints.  Values are aggregated over labels
so a multi-worker server yields one time series per metric.  Histogram buckets
retain their ``le`` boundary while all other labels are summed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, Literal
from urllib.request import Request, urlopen


MetricKind = Literal["gauge", "counter", "histogram"]
MetricSpec = Mapping[str, MetricKind]
MetricSample = dict[str, Any]


VLLM_METRIC_SPEC: dict[str, MetricKind] = {
    "vllm:num_requests_running": "gauge",
    "vllm:num_requests_waiting": "gauge",
    "vllm:kv_cache_usage_perc": "gauge",
    "vllm:num_preemptions": "counter",
    "vllm:time_to_first_token_seconds": "histogram",
    "vllm:request_queue_time_seconds": "histogram",
    "vllm:request_prefill_time_seconds": "histogram",
    "vllm:request_decode_time_seconds": "histogram",
}


SGLANG_METRIC_SPEC: dict[str, MetricKind] = {
    "sglang:num_running_reqs": "gauge",
    "sglang:num_queue_reqs": "gauge",
    "sglang:token_usage": "gauge",
    "sglang:num_retracted_reqs": "gauge",
    "sglang:num_retracted_requests_total": "counter",
    "sglang:queue_time_seconds": "histogram",
    "sglang:time_to_first_token_seconds": "histogram",
    "sglang:is_cuda_graph": "gauge",
    "sglang:cuda_graph_passes_total": "counter",
}


_METRIC_NAME_RE = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")
_LABEL_RE = re.compile(
    r'(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"'
    r'(?P<value>(?:\\.|[^"\\])*)"'
)
_HISTOGRAM_SUFFIXES = ("_bucket", "_count", "_sum")


@dataclass(frozen=True)
class ParsedSnapshot:
    """One aggregated scrape.

    ``values`` contains ordinary samples and histogram ``_sum``/``_count``
    samples.  ``buckets`` is keyed first by histogram family, then by the
    Prometheus ``le`` label.
    """

    values: dict[str, float]
    buckets: dict[str, dict[str, float]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "values": dict(self.values),
            "buckets": {
                family: dict(boundaries)
                for family, boundaries in self.buckets.items()
            },
        }


def _unescape_label(value: str) -> str:
    """Decode the escape sequences defined by Prometheus text exposition."""

    output: list[str] = []
    index = 0
    while index < len(value):
        char = value[index]
        if char != "\\" or index + 1 >= len(value):
            output.append(char)
            index += 1
            continue
        escaped = value[index + 1]
        output.append("\n" if escaped == "n" else escaped)
        index += 2
    return "".join(output)


def _parse_labels(raw_labels: str) -> dict[str, str]:
    return {
        match.group("name"): _unescape_label(match.group("value"))
        for match in _LABEL_RE.finditer(raw_labels)
    }


def _split_sample_line(line: str) -> tuple[str, dict[str, str], float] | None:
    """Parse a single metric line, ignoring comments and malformed samples."""

    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    name_match = _METRIC_NAME_RE.match(stripped)
    if name_match is None:
        return None
    name = name_match.group(0)
    position = name_match.end()
    labels: dict[str, str] = {}

    if position < len(stripped) and stripped[position] == "{":
        position += 1
        label_start = position
        in_quotes = False
        escaped = False
        while position < len(stripped):
            char = stripped[position]
            if escaped:
                escaped = False
            elif char == "\\" and in_quotes:
                escaped = True
            elif char == '"':
                in_quotes = not in_quotes
            elif char == "}" and not in_quotes:
                break
            position += 1
        if position >= len(stripped) or stripped[position] != "}":
            return None
        labels = _parse_labels(stripped[label_start:position])
        position += 1

    remainder = stripped[position:].strip()
    if not remainder:
        return None
    value_token = remainder.split(None, 1)[0]
    try:
        value = float(value_token)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return name, labels, value


def _selected_sample(name: str, selected: set[str] | None) -> bool:
    if selected is None or name in selected:
        return True
    for suffix in _HISTOGRAM_SUFFIXES:
        if name.endswith(suffix) and name[: -len(suffix)] in selected:
            return True
    # prometheus_client exposes a Counter named ``x`` as ``x_total``.  Permit
    # callers to select either the collector name or its wire-format name.
    return name.endswith("_total") and name[:-6] in selected


def parse_prometheus_text(
    text: str | bytes,
    selected_metrics: Iterable[str] | None = None,
) -> ParsedSnapshot:
    """Parse and label-aggregate Prometheus text exposition.

    Args:
        text: A response body from a Prometheus ``/metrics`` endpoint.
        selected_metrics: Collector/family names to retain.  Selecting a
            histogram family also selects its ``_bucket``, ``_sum`` and
            ``_count`` samples.  ``None`` retains every finite sample.

    Non-finite and malformed values are ignored.  Multiple samples with the
    same name are summed across labels.  For histogram buckets, only the ``le``
    label remains as a dimension.
    """

    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    selected = None if selected_metrics is None else set(selected_metrics)
    values: dict[str, float] = {}
    buckets: dict[str, dict[str, float]] = {}

    for line in text.splitlines():
        parsed = _split_sample_line(line)
        if parsed is None:
            continue
        name, labels, value = parsed
        if not _selected_sample(name, selected):
            continue

        if name.endswith("_bucket") and "le" in labels:
            family = name[: -len("_bucket")]
            boundary = labels["le"]
            family_buckets = buckets.setdefault(family, {})
            family_buckets[boundary] = family_buckets.get(boundary, 0.0) + value
        else:
            values[name] = values.get(name, 0.0) + value

    return ParsedSnapshot(values=values, buckets=buckets)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _value_series(
    samples: Sequence[Mapping[str, Any]],
    candidates: Sequence[str],
) -> list[float]:
    series: list[float] = []
    for sample in samples:
        values = sample.get("values", {})
        if not isinstance(values, Mapping):
            continue
        for candidate in candidates:
            value = _finite_number(values.get(candidate))
            if value is not None:
                series.append(value)
                break
    return series


def _empty_scalar_summary(kind: MetricKind) -> dict[str, Any]:
    if kind == "gauge":
        return {
            "kind": kind,
            "observations": 0,
            "start": None,
            "end": None,
            "max": None,
            "mean": None,
        }
    return {
        "kind": kind,
        "observations": 0,
        "start": None,
        "end": None,
        "delta": None,
        "reset_detected": False,
    }


def _summarize_gauge(series: Sequence[float]) -> dict[str, Any]:
    if not series:
        return _empty_scalar_summary("gauge")
    return {
        "kind": "gauge",
        "observations": len(series),
        "start": series[0],
        "end": series[-1],
        "max": max(series),
        "mean": sum(series) / len(series),
    }


def _summarize_counter(series: Sequence[float]) -> dict[str, Any]:
    if not series:
        return _empty_scalar_summary("counter")
    return {
        "kind": "counter",
        "observations": len(series),
        "start": series[0],
        "end": series[-1],
        "delta": series[-1] - series[0],
        "reset_detected": any(
            current < previous for previous, current in zip(series, series[1:])
        ),
    }


def _bucket_sort_key(boundary: str) -> tuple[int, float | str]:
    if boundary in {"+Inf", "Inf", "+inf", "inf"}:
        return (2, 0.0)
    try:
        return (0, float(boundary))
    except ValueError:
        return (1, boundary)


def _bucket_deltas(
    samples: Sequence[Mapping[str, Any]],
    family: str,
) -> dict[str, float]:
    by_boundary: dict[str, list[float]] = {}
    for sample in samples:
        all_buckets = sample.get("buckets", {})
        if not isinstance(all_buckets, Mapping):
            continue
        family_buckets = all_buckets.get(family, {})
        if not isinstance(family_buckets, Mapping):
            continue
        for raw_boundary, raw_value in family_buckets.items():
            value = _finite_number(raw_value)
            if value is not None:
                by_boundary.setdefault(str(raw_boundary), []).append(value)

    deltas = {
        boundary: series[-1] - series[0]
        for boundary, series in by_boundary.items()
        if len(series) >= 2
    }
    return dict(sorted(deltas.items(), key=lambda item: _bucket_sort_key(item[0])))


def _summarize_histogram(
    samples: Sequence[Mapping[str, Any]],
    family: str,
) -> dict[str, Any]:
    count_series = _value_series(samples, [f"{family}_count"])
    sum_series = _value_series(samples, [f"{family}_sum"])
    count_delta = (
        count_series[-1] - count_series[0] if len(count_series) >= 2 else None
    )
    sum_delta = sum_series[-1] - sum_series[0] if len(sum_series) >= 2 else None
    delta_mean = None
    if (
        count_delta is not None
        and sum_delta is not None
        and count_delta > 0
        and math.isfinite(sum_delta / count_delta)
    ):
        delta_mean = sum_delta / count_delta

    return {
        "kind": "histogram",
        "count_observations": len(count_series),
        "sum_observations": len(sum_series),
        "count_start": count_series[0] if count_series else None,
        "count_end": count_series[-1] if count_series else None,
        "count_delta": count_delta,
        "sum_start": sum_series[0] if sum_series else None,
        "sum_end": sum_series[-1] if sum_series else None,
        "sum_delta": sum_delta,
        "delta_mean": delta_mean,
        "bucket_deltas": _bucket_deltas(samples, family),
    }


def summarize_samples(
    samples: Sequence[Mapping[str, Any]],
    metric_spec: MetricSpec,
) -> dict[str, Any]:
    """Summarize scrape records according to their Prometheus metric type."""

    timestamps = [
        value
        for sample in samples
        if (value := _finite_number(sample.get("timestamp_unix"))) is not None
    ]
    elapsed = [
        value
        for sample in samples
        if (value := _finite_number(sample.get("elapsed_seconds"))) is not None
    ]
    metrics: dict[str, dict[str, Any]] = {}

    for name, kind in metric_spec.items():
        if kind == "gauge":
            metrics[name] = _summarize_gauge(_value_series(samples, [name]))
        elif kind == "counter":
            candidates = [name] if name.endswith("_total") else [name, f"{name}_total"]
            metrics[name] = _summarize_counter(_value_series(samples, candidates))
        elif kind == "histogram":
            metrics[name] = _summarize_histogram(samples, name)
        else:
            raise ValueError(f"unsupported metric kind for {name!r}: {kind!r}")

    duration_seconds = None
    if len(elapsed) >= 2:
        duration_seconds = elapsed[-1] - elapsed[0]
    elif len(timestamps) >= 2:
        duration_seconds = timestamps[-1] - timestamps[0]

    return {
        "sample_count": len(samples),
        "first_timestamp_unix": timestamps[0] if timestamps else None,
        "last_timestamp_unix": timestamps[-1] if timestamps else None,
        "duration_seconds": duration_seconds,
        "metrics": metrics,
    }


Fetcher = Callable[[str, float], str | bytes]


def _http_fetch(metrics_url: str, timeout_seconds: float) -> bytes:
    request = Request(
        metrics_url,
        headers={"Accept": "text/plain; version=0.0.4", "User-Agent": "miniagent/1"},
    )
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        return response.read()


class PrometheusSampler:
    """Poll a Prometheus endpoint on a background thread.

    ``start`` performs an immediate scrape.  ``stop`` wakes the thread and, by
    default, performs a final scrape so histogram/counter deltas cover the
    complete measured subprocess interval.
    """

    def __init__(
        self,
        metrics_url: str,
        interval_seconds: float = 0.1,
        selected_metrics: Iterable[str] | None = None,
        *,
        metric_spec: MetricSpec | None = None,
        timeout_seconds: float = 1.0,
        fetcher: Fetcher | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        normalized_spec = dict(metric_spec or {})
        invalid = {
            name: kind
            for name, kind in normalized_spec.items()
            if kind not in {"gauge", "counter", "histogram"}
        }
        if invalid:
            raise ValueError(f"unsupported metric kinds: {invalid}")

        self.metrics_url = metrics_url
        self.metric_spec = normalized_spec
        self.selected_metrics = (
            set(selected_metrics)
            if selected_metrics is not None
            else (set(normalized_spec) if normalized_spec else None)
        )
        self.interval_seconds = float(interval_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self._fetcher = fetcher or _http_fetch
        self._samples: list[MetricSample] = []
        self._errors: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_monotonic: float | None = None

    @property
    def samples(self) -> list[MetricSample]:
        with self._lock:
            return [
                {
                    **sample,
                    "values": dict(sample["values"]),
                    "buckets": {
                        family: dict(boundaries)
                        for family, boundaries in sample["buckets"].items()
                    },
                }
                for sample in self._samples
            ]

    @property
    def errors(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(error) for error in self._errors]

    def _timing(self) -> tuple[float, float, str]:
        timestamp = time.time()
        monotonic = time.monotonic()
        started = self._started_monotonic
        elapsed = 0.0 if started is None else max(0.0, monotonic - started)
        timestamp_iso = datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
        return timestamp, elapsed, timestamp_iso

    def sample_once(self, *, raise_on_error: bool = False) -> dict[str, Any] | None:
        """Synchronously collect one scrape and append it to ``samples``."""

        try:
            payload = self._fetcher(self.metrics_url, self.timeout_seconds)
            parsed = parse_prometheus_text(payload, self.selected_metrics)
            timestamp, elapsed, timestamp_iso = self._timing()
            record = {
                "timestamp_unix": timestamp,
                "timestamp_utc": timestamp_iso,
                "elapsed_seconds": elapsed,
                **parsed.as_dict(),
            }
            with self._lock:
                self._samples.append(record)
            return record
        except Exception as exc:  # Polling errors must not kill the benchmark.
            timestamp, elapsed, timestamp_iso = self._timing()
            error = {
                "timestamp_unix": timestamp,
                "timestamp_utc": timestamp_iso,
                "elapsed_seconds": elapsed,
                "type": type(exc).__name__,
                "message": str(exc),
            }
            with self._lock:
                self._errors.append(error)
            if raise_on_error:
                raise
            return None

    def _poll(self) -> None:
        while not self._stop_event.is_set():
            iteration_started = time.monotonic()
            self.sample_once()
            remaining = self.interval_seconds - (time.monotonic() - iteration_started)
            if remaining > 0:
                self._stop_event.wait(remaining)

    def start(self) -> "PrometheusSampler":
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("sampler is already running")
        self._stop_event.clear()
        self._started_monotonic = time.monotonic()
        self._thread = threading.Thread(
            target=self._poll,
            name="miniagent-prometheus-sampler",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self, *, collect_final: bool = True) -> list[MetricSample]:
        thread = self._thread
        if thread is None:
            return self.samples
        self._stop_event.set()
        thread.join(timeout=max(1.0, self.timeout_seconds + self.interval_seconds))
        if thread.is_alive():
            raise RuntimeError("Prometheus sampler thread did not stop")
        self._thread = None
        if collect_final:
            self.sample_once()
        return self.samples

    def __enter__(self) -> "PrometheusSampler":
        return self.start()

    def __exit__(self, *_exc_info: object) -> None:
        self.stop()

    def summary(self, metric_spec: MetricSpec | None = None) -> dict[str, Any]:
        return summarize_samples(self.samples, metric_spec or self.metric_spec)

    def as_dict(self, metric_spec: MetricSpec | None = None) -> dict[str, Any]:
        samples = self.samples
        effective_spec = dict(metric_spec or self.metric_spec)
        return {
            "schema_version": 1,
            "metrics_url": self.metrics_url,
            "interval_seconds": self.interval_seconds,
            "timeout_seconds": self.timeout_seconds,
            "selected_metrics": (
                sorted(self.selected_metrics)
                if self.selected_metrics is not None
                else None
            ),
            "metric_spec": effective_spec,
            "samples": samples,
            "errors": self.errors,
            "summary": summarize_samples(samples, effective_spec),
        }

    def write_json(
        self,
        path: str | os.PathLike[str],
        metric_spec: MetricSpec | None = None,
    ) -> dict[str, Any]:
        """Atomically persist raw samples, scrape errors, and their summary."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.as_dict(metric_spec)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                json.dump(payload, temporary, indent=2, sort_keys=True)
                temporary.write("\n")
            os.replace(temporary_name, destination)
        finally:
            if temporary_name is not None and os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return payload


__all__ = [
    "MetricKind",
    "MetricSample",
    "MetricSpec",
    "ParsedSnapshot",
    "PrometheusSampler",
    "SGLANG_METRIC_SPEC",
    "VLLM_METRIC_SPEC",
    "parse_prometheus_text",
    "summarize_samples",
]
