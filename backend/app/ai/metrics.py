"""Per-call instrumentation for the LLM layer.

Answers the operational questions you cannot judge by reading a transcript:
how often structured output actually works, how slow each call is, and how
many tokens (hence how much money) a game burns.

`ParsePath` is the important one: the provider tries strict JSON-schema
first and degrades to loose JSON mode plus a permissive parser. Collapsing
that into a single "did it work" boolean would hide the fact that a model
never honours strict schema and is being carried entirely by the fallback.
"""

from __future__ import annotations

import statistics
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from time import perf_counter
from typing import Any


class ParsePath(StrEnum):
    STRICT_SCHEMA = "strict_schema"
    JSON_OBJECT = "json_object"
    PERMISSIVE = "permissive"
    FAILED = "failed"


@dataclass
class CallRecord:
    schema: str
    path: ParsePath
    latency_seconds: float
    attempt: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.path is not ParsePath.FAILED


@dataclass
class MetricsCollector:
    """Thread-safe because the coordinator fans calls out with asyncio.gather
    and the provider may be shared across concurrent games."""

    records: list[CallRecord] = field(default_factory=list)
    discussion_attempts: int = 0
    discussion_skips: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, record: CallRecord) -> None:
        with self._lock:
            self.records.append(record)

    def record_discussion_result(self, *, skipped: bool) -> None:
        with self._lock:
            self.discussion_attempts += 1
            self.discussion_skips += int(skipped)

    @contextmanager
    def time_call(self) -> Iterator[list[float]]:
        """Yields a one-element list that receives the elapsed seconds."""
        holder: list[float] = []
        started = perf_counter()
        try:
            yield holder
        finally:
            holder.append(perf_counter() - started)

    # -- aggregates --

    def summary(
        self,
        price_per_1m_input: float = 0.0,
        price_per_1m_output: float = 0.0,
    ) -> dict[str, Any]:
        with self._lock:
            records = list(self.records)
            discussion_attempts = self.discussion_attempts
            discussion_skips = self.discussion_skips

        total = len(records)
        if total == 0:
            return {"total_calls": 0}

        by_path: dict[str, int] = {}
        for path in ParsePath:
            by_path[path.value] = sum(1 for r in records if r.path is path)

        latencies = sorted(r.latency_seconds for r in records)
        prompt_tokens = sum(r.prompt_tokens or 0 for r in records)
        completion_tokens = sum(r.completion_tokens or 0 for r in records)
        tokens_reported = any(r.prompt_tokens is not None for r in records)

        by_schema: dict[str, dict[str, Any]] = {}
        for schema in sorted({r.schema for r in records}):
            subset = [r for r in records if r.schema == schema]
            by_schema[schema] = {
                "calls": len(subset),
                "success_rate": sum(1 for r in subset if r.succeeded) / len(subset),
                "mean_latency_seconds": round(
                    statistics.fmean(r.latency_seconds for r in subset), 3
                ),
            }

        summary: dict[str, Any] = {
            "total_calls": total,
            "success_rate": sum(1 for r in records if r.succeeded) / total,
            "parse_path_counts": by_path,
            "strict_schema_rate": by_path[ParsePath.STRICT_SCHEMA.value] / total,
            "retry_calls": sum(1 for r in records if r.attempt > 0),
            "latency_seconds": {
                "mean": round(statistics.fmean(latencies), 3),
                "p50": round(_percentile(latencies, 0.50), 3),
                "p95": round(_percentile(latencies, 0.95), 3),
                "max": round(latencies[-1], 3),
            },
            "by_schema": by_schema,
            "errors": _top_errors(records),
            "discussion_generation_attempts": discussion_attempts,
            "discussion_skips": discussion_skips,
            "discussion_skip_rate": discussion_skips / max(discussion_attempts, 1),
        }

        if tokens_reported:
            cost = (
                prompt_tokens / 1_000_000 * price_per_1m_input
                + completion_tokens / 1_000_000 * price_per_1m_output
            )
            summary["tokens"] = {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": prompt_tokens + completion_tokens,
            }
            summary["estimated_cost"] = round(cost, 6)
            summary["pricing_supplied"] = bool(price_per_1m_input or price_per_1m_output)
        else:
            # The mock provider reports no usage, and a real endpoint may
            # omit it too -- say so rather than implying zero spend.
            summary["tokens"] = None
            summary["estimated_cost"] = None
            summary["pricing_supplied"] = False

        return summary


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = min(int(round(q * (len(sorted_values) - 1))), len(sorted_values) - 1)
    return sorted_values[index]


def _top_errors(records: list[CallRecord], limit: int = 5) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for record in records:
        if record.error:
            counts[record.error] = counts.get(record.error, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return [{"error": message, "count": count} for message, count in ranked]
