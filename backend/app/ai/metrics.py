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
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from time import perf_counter
from typing import Any

from app.ai.provider.pricing import is_same_model


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
    # Logical generations and HTTP requests are deliberately separate: a
    # strict-schema failure may require a second response-mode request, and
    # either request may be retried at the transport/API boundary.
    http_requests: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    # Why the model stopped, and how much of the output budget its private
    # reasoning consumed. Without these, "response was not parseable as JSON"
    # cannot distinguish a model that emits prose from a reasoning model whose
    # thinking exhausted max_completion_tokens and left the content empty or
    # cut off mid-object -- two failures with opposite fixes.
    finish_reason: str | None = None
    reasoning_tokens: int | None = None
    # Anthropic prompt-cache usage. `prompt_tokens` stays the *total* input
    # (uncached + cache writes + cache reads) so token counts remain
    # comparable across providers; these say how much of it the cache served.
    cache_creation_tokens: int | None = None
    cache_read_tokens: int | None = None
    # The provider's own stop value (`end_turn`, `max_tokens`, `refusal`, ...).
    # `finish_reason` is the normalized form `truncated` reads; keeping both
    # means a provider-specific stop is never lost in translation.
    stop_reason: str | None = None
    refusal_category: str | None = None
    # Which model actually answered, and which one was asked for. A
    # server-side refusal fallback can serve a turn on a different -- and
    # differently priced -- model than the one requested.
    served_model: str | None = None
    requested_model: str | None = None
    request_id: str | None = None
    # USD for this logical call across all its HTTP attempts, priced by the
    # provider from `app/ai/provider/pricing.py`. None when the provider does
    # not price its own calls (mock, luna).
    estimated_cost: float | None = None
    effort: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.path is not ParsePath.FAILED

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    @property
    def refused(self) -> bool:
        return self.finish_reason == "refusal" or self.stop_reason == "refusal"

    @property
    def served_by_fallback(self) -> bool:
        """Did a different model answer than the one asked for?

        Only when both are known. A dated snapshot of the requested model
        (`claude-opus-5-5-20261101` for `claude-opus-5-5`) is the same model,
        not a fallback; `claude-opus-5` for `claude-opus-5-5` is a fallback.
        """
        if not self.served_model or not self.requested_model:
            return False
        return not is_same_model(self.served_model, self.requested_model)


@dataclass
class MetricsCollector:
    """Thread-safe because the coordinator fans calls out with asyncio.gather
    and the provider may be shared across concurrent games."""

    records: list[CallRecord] = field(default_factory=list)
    discussion_attempts: int = 0
    discussion_skips: int = 0
    total_game_wall_time: float = 0.0
    public_utterances: int = 0
    game_days: int = 0
    time_spent_in_solver: float = 0.0
    solver_query_count: int = 0
    solver_cache_hits: int = 0
    # Named counters for things the code did *instead of* the model -- a vote
    # resolved in code, a fallback line posted, a speech discarded. A rate of
    # these is how a run that "succeeded" is shown to have been carried by
    # safety nets rather than by the model.
    events: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def merged(cls, collectors: Iterable[MetricsCollector]) -> MetricsCollector:
        """One collector holding everything the given ones recorded.

        The evaluation harness keeps a collector per game, so one game's
        failures or cost are never averaged away by another's, and merges them
        for the run-wide figures. Summing the counters (wall time, utterances,
        days, solver time) rather than averaging them keeps every ratio in
        `summary()` -- seconds per utterance, per day -- a pooled ratio over
        all the games, the same as if one collector had recorded them all.
        The inputs are copied, not shared: recording into a source afterwards
        does not change the merge.
        """
        merged = cls()
        for collector in collectors:
            with collector._lock:
                merged.records.extend(collector.records)
                merged.discussion_attempts += collector.discussion_attempts
                merged.discussion_skips += collector.discussion_skips
                merged.total_game_wall_time += collector.total_game_wall_time
                merged.public_utterances += collector.public_utterances
                merged.game_days += collector.game_days
                merged.time_spent_in_solver += collector.time_spent_in_solver
                merged.solver_query_count += collector.solver_query_count
                merged.solver_cache_hits += collector.solver_cache_hits
                for name, count in collector.events.items():
                    merged.events[name] = merged.events.get(name, 0) + count
        return merged

    def record(self, record: CallRecord) -> None:
        with self._lock:
            self.records.append(record)

    def record_event(self, name: str) -> None:
        with self._lock:
            self.events[name] = self.events.get(name, 0) + 1

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
            events = dict(sorted(self.events.items()))

        total = len(records)
        if total == 0:
            # Events can fire with no LLM call at all (everything resolved in
            # code), which is exactly the case where hiding them would mislead.
            return {"total_calls": 0, "events": events} if events else {"total_calls": 0}

        by_path: dict[str, int] = {}
        for path in ParsePath:
            by_path[path.value] = sum(1 for r in records if r.path is path)

        latencies = sorted(r.latency_seconds for r in records)
        prompt_tokens = sum(r.prompt_tokens or 0 for r in records)
        completion_tokens = sum(r.completion_tokens or 0 for r in records)
        tokens_reported = any(r.prompt_tokens is not None for r in records)
        cache_creation_tokens = sum(r.cache_creation_tokens or 0 for r in records)
        cache_read_tokens = sum(r.cache_read_tokens or 0 for r in records)

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
            "http_requests": sum(r.http_requests for r in records),
            "success_rate": sum(1 for r in records if r.succeeded) / total,
            "parse_path_counts": by_path,
            "strict_schema_rate": by_path[ParsePath.STRICT_SCHEMA.value] / total,
            "strict_schema_failures": total - by_path[ParsePath.STRICT_SCHEMA.value],
            "json_object_successes": (
                by_path[ParsePath.JSON_OBJECT.value] + by_path[ParsePath.PERMISSIVE.value]
            ),
            "complete_failures": by_path[ParsePath.FAILED.value],
            "retry_calls": sum(1 for r in records if r.attempt > 0),
            "finish_reason_counts": _finish_reason_counts(records),
            # Of the calls that produced nothing usable, how many were cut off
            # rather than malformed. A high share here is an output-budget
            # problem, not a prompting one.
            "truncated_calls": sum(1 for r in records if r.truncated),
            "truncated_failure_rate": (
                sum(1 for r in records if r.truncated and not r.succeeded)
                / max(sum(1 for r in records if not r.succeeded), 1)
            ),
            "reasoning_tokens": sum(r.reasoning_tokens or 0 for r in records),
            "reasoning_token_share": (
                sum(r.reasoning_tokens or 0 for r in records) / max(completion_tokens, 1)
            ),
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
            "total_game_wall_time": self.total_game_wall_time,
            "time_per_game_day": self.total_game_wall_time / max(self.game_days, 1),
            "time_per_public_utterance": self.total_game_wall_time
            / max(self.public_utterances, 1),
            "time_waiting_for_llm": sum(latencies),
            "time_spent_in_solver": self.time_spent_in_solver,
            "solver_query_count": self.solver_query_count,
            "solver_cache_hit_rate": self.solver_cache_hits
            / max(self.solver_query_count, 1),
            "events": events,
            "cache_creation_tokens": cache_creation_tokens,
            "cache_read_tokens": cache_read_tokens,
            # Share of all input the prompt cache served. Cache reads cost a
            # fraction of fresh input and do not count toward the input-token
            # rate limit, so this is the first number to check when a run is
            # slower or dearer than expected.
            "cache_read_share": cache_read_tokens / max(prompt_tokens, 1),
            "refusals": _refusal_counts(records),
            "served_model_counts": _served_model_counts(records),
            "fallback_served_calls": sum(1 for r in records if r.served_by_fallback),
        }

        token_totals = (
            {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": prompt_tokens + completion_tokens,
            }
            if tokens_reported
            else None
        )
        if _priced_by_provider(records):
            # The provider priced every call that used tokens from its own
            # table -- per served model and with cache rates, which one flat
            # input/output price applied to the totals cannot reproduce.
            summary["tokens"] = token_totals
            summary["estimated_cost"] = round(sum(r.estimated_cost or 0.0 for r in records), 6)
            summary["pricing_supplied"] = True
            summary["pricing_source"] = "table"
        elif tokens_reported:
            cost = (
                prompt_tokens / 1_000_000 * price_per_1m_input
                + completion_tokens / 1_000_000 * price_per_1m_output
            )
            summary["tokens"] = token_totals
            summary["estimated_cost"] = round(cost, 6)
            summary["pricing_supplied"] = bool(price_per_1m_input or price_per_1m_output)
            summary["pricing_source"] = "arguments" if summary["pricing_supplied"] else None
        else:
            # The mock provider reports no usage, and a real endpoint may
            # omit it too -- say so rather than implying zero spend.
            summary["tokens"] = None
            summary["estimated_cost"] = None
            summary["pricing_supplied"] = False
            summary["pricing_source"] = None

        return summary


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = min(int(round(q * (len(sorted_values) - 1))), len(sorted_values) - 1)
    return sorted_values[index]


def _priced_by_provider(records: list[CallRecord]) -> bool:
    """True when the provider priced its own calls and left none that used
    tokens unpriced -- a sum over only some of them would understate spend."""
    if not any(r.estimated_cost is not None for r in records):
        return False
    return all(
        r.estimated_cost is not None
        for r in records
        if r.prompt_tokens is not None or r.completion_tokens is not None
    )


def _refusal_counts(records: list[CallRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if record.refused:
            category = record.refusal_category or "uncategorized"
            counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def _served_model_counts(records: list[CallRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if record.served_model:
            counts[record.served_model] = counts.get(record.served_model, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def _finish_reason_counts(records: list[CallRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if record.finish_reason:
            counts[record.finish_reason] = counts.get(record.finish_reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def _top_errors(records: list[CallRecord], limit: int = 5) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for record in records:
        if record.error:
            counts[record.error] = counts.get(record.error, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return [{"error": message, "count": count} for message, count in ranked]
