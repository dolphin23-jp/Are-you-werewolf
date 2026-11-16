#!/usr/bin/env python3
"""Which model / effort / speed can answer a game turn in about five seconds?

The v3 target is a wait of ~5 s per public utterance (approach-reset §6, §11.5
Phase 1b). The discussion model and effort have to be chosen *before* the
Claude baseline is taken: changing them afterwards means re-taking the
baseline and re-tuning the doctrine. This script measures the candidates on
the prompts a game really sends.

    # 計画だけ表示(通信なし・費用ゼロ)
    python scripts/latency_sweep.py --dry-run

    # 実測(有料。--max-cost は必須)
    WEREWOLF_ANTHROPIC_API_KEY=... python scripts/latency_sweep.py --max-cost 5

Steps:
  1. capture (free): one offline all-AI v3 game for --seed is played with the
     scenario double, and every prompt it sends is recorded
     (`app/eval/prompt_capture.py`).
  2. sample: --prompts N (default 60) of them, stratified by schema (about
     40% discussion, 30% vote, 15% night action, 15% wolf/freemason chat),
     deterministic for the seed.
  3. replay (paid): every prompt is sent once per config, the configs of one
     prompt next to each other and in a rotating order, so every comparison
     is made within one run under the same conditions. Each request has the
     shape the provider sends -- model, the `max_tokens` floor, effort and the
     JSON-schema format -- with refusal fallbacks and betas left off so the
     timing is the model's alone, and `speed="fast"` only on fast configs.
     Streaming measures the time to the first event, to the first visible
     text and to the end.

TIMING ONLY: the sweep does not judge whether an answer is any good. A model
or effort change must still pass the game evaluation (`scripts/evaluate.py`)
before it is adopted, and switching the game to a different model is the
owner's decision, not this script's.

Writes to --out (default `eval-out/latency-sweep/`, git-ignored with the rest
of `eval-out/`): plan.md / plan.json / prompts.jsonl always; after a replay
also rows.jsonl (appended as each request finishes, so an interrupted run
keeps what it measured), rows.csv, summary.md and summary.json.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic  # noqa: E402
from pydantic import BaseModel, ValidationError  # noqa: E402

# Private on purpose in the provider; shared here so the sweep's headers,
# reservations and failed-attempt accounting are exactly the game's.
from app.ai.provider.anthropic_claude import (  # noqa: E402
    _ESTIMATED_TOKENS_PER_CHAR,
    _ambient_header_overrides,
    _failed_attempt_cost,
)
from app.ai.provider.budget import BudgetExceeded, EvaluationBudget  # noqa: E402
from app.ai.provider.pricing import (  # noqa: E402
    FAST_PRICES,
    ModelPrice,
    cost_usd,
    is_same_model,
    price_for,
)
from app.config import ANTHROPIC_EFFORT_LEVELS, Settings  # noqa: E402
from app.eval.prompt_capture import CapturedPrompt, capture_offline_game  # noqa: E402

DEFAULT_CONFIGS = (
    "claude-opus-5-5:low,claude-opus-5-5:medium,claude-opus-5-5:low:fast,"
    "claude-sonnet-5-5:low,claude-haiku-5-5:low"
)
DEFAULT_PROMPTS = 60
# Inside the git-ignored `eval-out/`, so a run never shows up as a change.
DEFAULT_OUT = Path("eval-out/latency-sweep")
DEFAULT_CONCURRENCY = 4
MAX_CONCURRENCY = 16
MAX_PROMPTS = 1000

# The approach-reset target for one public utterance (§6).
TARGET_SECONDS = 5.0

# Share of the sample per schema. Discussion is the turn the player waits on
# one at a time, so it gets the largest share; votes run in parallel but are
# the most frequent call; night actions and private chat fill the rest.
SCHEMA_SHARES: dict[str, float] = {
    "ChatDiscussionOutput": 0.40,
    "VoteOutput": 0.30,
    "NightActionOutput": 0.15,
    "WolfChatOutput": 0.15,
}

# Rough output tokens (thinking included) per call, by effort, for the
# *expected* cost line of the plan only. Assumptions, not measurements: the
# worst case (every call using its whole `max_tokens`) is printed beside it,
# and the budget reserves the worst case, so a wrong guess here can mislead
# the plan but never overspend.
EXPECTED_OUTPUT_TOKENS: dict[str, int] = {
    "low": 800,
    "medium": 2000,
    "high": 4000,
    "xhigh": 6000,
    "max": 8000,
}

_MODEL_ID_RE = re.compile(r"claude-[a-z0-9]+(?:-[a-z0-9]+)*")
_ERROR_TEXT_LIMIT = 200


# --- configs -----------------------------------------------------------------


@dataclass(frozen=True)
class SweepConfig:
    """One candidate: a model, an effort, and standard or fast speed."""

    model: str
    effort: str
    fast: bool = False

    @property
    def label(self) -> str:
        return f"{self.model}:{self.effort}" + (":fast" if self.fast else "")

    @property
    def speed(self) -> str | None:
        return "fast" if self.fast else None

    def price(self) -> tuple[ModelPrice, bool]:
        return price_for(self.model, self.speed)


def parse_configs(text: str) -> list[SweepConfig]:
    """Parse `model:effort[:fast],...`, refusing anything the API would reject
    on the first paid request (or that would make the comparison ambiguous)."""
    configs: list[SweepConfig] = []
    seen: set[str] = set()
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError(f"config {item!r} is not model:effort or model:effort:fast")
        model, effort = parts[0].lower(), parts[1].lower()
        if not _MODEL_ID_RE.fullmatch(model):
            raise ValueError(f"config {item!r}: {model!r} is not a Claude model id")
        if effort not in ANTHROPIC_EFFORT_LEVELS:
            levels = sorted(ANTHROPIC_EFFORT_LEVELS)
            raise ValueError(f"config {item!r}: effort {effort!r} is not one of {levels}")
        fast = False
        if len(parts) == 3:
            if parts[2].lower() != "fast":
                raise ValueError(f"config {item!r}: the third part may only be 'fast'")
            if model not in FAST_PRICES:
                # Fast mode exists only on some models; this table lists the
                # ones whose fast price is known.
                raise ValueError(
                    f"config {item!r}: fast mode is only measured for {sorted(FAST_PRICES)}"
                )
            fast = True
        config = SweepConfig(model=model, effort=effort, fast=fast)
        if config.label in seen:
            raise ValueError(f"config {config.label!r} is listed twice")
        seen.add(config.label)
        configs.append(config)
    if not configs:
        raise ValueError("no configs given")
    return configs


# --- sampling ----------------------------------------------------------------


def _quotas(n: int, available: dict[str, int], shares: dict[str, float]) -> dict[str, int]:
    """Split `n` by `shares` (largest remainder), capped by what is available.

    A schema with too few prompts gives its shortfall to the others in share
    order, so the sample still has `n` prompts whenever the capture does.
    """
    names = [name for name in shares if available.get(name, 0) > 0]
    total_share = sum(shares[name] for name in names)
    if not names or total_share <= 0:
        return {}
    exact = {name: n * shares[name] / total_share for name in names}
    quotas = {name: math.floor(value) for name, value in exact.items()}
    remainder = n - sum(quotas.values())
    by_fraction = sorted(names, key=lambda name: (-(exact[name] - quotas[name]), names.index(name)))
    for name in by_fraction[:remainder]:
        quotas[name] += 1
    # Cap at availability and hand the shortfall on, largest share first.
    shortfall = 0
    for name in names:
        if quotas[name] > available[name]:
            shortfall += quotas[name] - available[name]
            quotas[name] = available[name]
    for name in sorted(names, key=lambda name: -shares[name]):
        if shortfall <= 0:
            break
        room = available[name] - quotas[name]
        take = min(room, shortfall)
        quotas[name] += take
        shortfall -= take
    return quotas


def stratified_sample(
    prompts: Sequence[CapturedPrompt],
    n: int,
    seed: int,
    shares: dict[str, float] | None = None,
) -> list[CapturedPrompt]:
    """Pick `n` prompts stratified by schema, the same ones for the same seed.

    Schemas outside `shares` (a summary, a morning intent) are never picked:
    they are not turns a player waits on. The result is in game order.
    """
    shares = shares or SCHEMA_SHARES
    pools: dict[str, list[CapturedPrompt]] = {}
    for prompt in prompts:
        if prompt.schema in shares:
            pools.setdefault(prompt.schema, []).append(prompt)
    quotas = _quotas(n, {name: len(pool) for name, pool in pools.items()}, shares)
    rng = random.Random(f"latency-sweep:{seed}")
    picked: list[CapturedPrompt] = []
    for name in shares:
        quota = quotas.get(name, 0)
        if quota:
            picked.extend(rng.sample(pools[name], quota))
    return sorted(picked, key=lambda prompt: prompt.index)


def replay_order(sample: Sequence[CapturedPrompt], seed: int) -> list[CapturedPrompt]:
    """The order to send the sample in: schemas evenly mixed, days spread.

    A run can stop at its cost cap part-way through. Sent in game order, the
    measured part would be the early days' prompts only -- shorter boards,
    so faster answers, so a biased result. Instead each schema's prompts are
    shuffled (seeded) and the schemas are merged in proportion, the k-th of
    a schema with q prompts sitting at (k + 0.5) / q, so every prefix of the
    run is close to the sample's own mix of schemas and of days.
    """
    rng = random.Random(f"latency-sweep-order:{seed}")
    by_schema: dict[str, list[CapturedPrompt]] = {}
    for prompt in sample:
        by_schema.setdefault(prompt.schema, []).append(prompt)
    keyed: list[tuple[float, int, int, CapturedPrompt]] = []
    for rank, name in enumerate(sorted(by_schema)):
        group = list(by_schema[name])
        rng.shuffle(group)
        for k, prompt in enumerate(group):
            keyed.append(((k + 0.5) / len(group), rank, prompt.index, prompt))
    keyed.sort(key=lambda item: item[:3])
    return [item[3] for item in keyed]


# --- requests and cost -------------------------------------------------------

_SCHEMA_CACHE: dict[type[BaseModel], dict[str, Any]] = {}


def _strict_schema(response_schema: type[BaseModel]) -> dict[str, Any]:
    cached = _SCHEMA_CACHE.get(response_schema)
    if cached is None:
        cached = anthropic.transform_schema(response_schema)
        _SCHEMA_CACHE[response_schema] = cached
    return cached


def max_tokens_sent(prompt: CapturedPrompt, floor: int) -> int:
    """The provider sends the larger of the caller's budget and the floor."""
    return max(prompt.max_tokens, floor)


def build_request(prompt: CapturedPrompt, config: SweepConfig, floor: int) -> dict[str, Any]:
    """The request the provider would send for this prompt, minus the refusal
    fallback (its beta and a possible second model would muddy the timing).

    `tests/scripts/test_latency_sweep.py` pins this against the provider's own
    request builder, so the two cannot drift apart unnoticed.
    """
    request: dict[str, Any] = {
        "model": config.model,
        "max_tokens": max_tokens_sent(prompt, floor),
        "messages": prompt.wire_messages(),
        "output_config": {
            "effort": config.effort,
            "format": {"type": "json_schema", "schema": _strict_schema(prompt.response_schema)},
        },
    }
    if prompt.system:
        request["system"] = prompt.system
    if config.fast:
        request["speed"] = "fast"
    return request


# For the *expected* cost line only: the prompts are mostly Japanese, at
# about one token per character. The reservation keeps the provider's
# deliberately high ratio.
_EXPECTED_TOKENS_PER_CHAR = 1.0


def _prompt_chars(prompt: CapturedPrompt) -> int:
    # The structured-output schema reaches the model as injected input too.
    schema_chars = len(json.dumps(_strict_schema(prompt.response_schema), ensure_ascii=False))
    return prompt.input_chars + schema_chars


def estimated_input_tokens(prompt: CapturedPrompt) -> int:
    """Input tokens on the safe (high) side, with the provider's own ratio."""
    return math.ceil(_ESTIMATED_TOKENS_PER_CHAR * _prompt_chars(prompt))


def worst_case_cost(prompt: CapturedPrompt, config: SweepConfig, floor: int) -> float:
    """What one request could cost at most: all of `max_tokens` used."""
    price, _known = config.price()
    return cost_usd(
        price,
        input_tokens=estimated_input_tokens(prompt),
        output_tokens=max_tokens_sent(prompt, floor),
    )


def input_only_cost(prompt: CapturedPrompt, config: SweepConfig) -> float:
    price, _known = config.price()
    return cost_usd(price, input_tokens=estimated_input_tokens(prompt))


def expected_cost(prompt: CapturedPrompt, config: SweepConfig, floor: int) -> float:
    """A rough central estimate; see `EXPECTED_OUTPUT_TOKENS`."""
    price, _known = config.price()
    output = min(EXPECTED_OUTPUT_TOKENS[config.effort], max_tokens_sent(prompt, floor))
    input_tokens = math.ceil(_EXPECTED_TOKENS_PER_CHAR * _prompt_chars(prompt))
    return cost_usd(price, input_tokens=input_tokens, output_tokens=output)


@dataclass(frozen=True)
class SweepJob:
    """One request of the sweep: a prompt, a config, and its place in the run."""

    order: int
    prompt: CapturedPrompt
    config: SweepConfig
    # Where this config fell among the configs of the same prompt (0 = first).
    position: int


def interleave(prompts: Sequence[CapturedPrompt], configs: Sequence[SweepConfig]) -> list[SweepJob]:
    """Every config for prompt 1, then every config for prompt 2, ...

    The config order rotates by one per prompt, so no config is always the
    first (cold connection) or the last (behind the others) to be sent.
    """
    jobs: list[SweepJob] = []
    k = len(configs)
    for i, prompt in enumerate(prompts):
        shift = i % k if k else 0
        rotated = list(configs[shift:]) + list(configs[:shift])
        for position, config in enumerate(rotated):
            jobs.append(SweepJob(order=len(jobs), prompt=prompt, config=config, position=position))
    return jobs


@dataclass
class ConfigEstimate:
    label: str
    requests: int
    worst_case_usd: float
    expected_usd: float
    price_known: bool


@dataclass
class SweepPlan:
    seed: int
    configs: list[SweepConfig]
    prompts: list[CapturedPrompt]
    max_tokens_floor: int
    captured_prompts: int
    captured_schema_counts: dict[str, int]
    estimates: list[ConfigEstimate] = field(default_factory=list)

    @property
    def requests(self) -> int:
        return len(self.configs) * len(self.prompts)

    @property
    def worst_case_usd(self) -> float:
        return sum(e.worst_case_usd for e in self.estimates)

    @property
    def expected_usd(self) -> float:
        return sum(e.expected_usd for e in self.estimates)

    def prompts_within(self, max_cost: float) -> int:
        """How many sample prompts (each under every config) the expected cost
        says fit under `max_cost` -- a planning hint, not a promise."""
        if not self.prompts or self.expected_usd <= 0:
            return len(self.prompts)
        per_prompt = self.expected_usd / len(self.prompts)
        return min(len(self.prompts), math.floor(max_cost / per_prompt))

    def sample_schema_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for prompt in self.prompts:
            counts[prompt.schema] = counts.get(prompt.schema, 0) + 1
        return {name: counts[name] for name in SCHEMA_SHARES if name in counts}


def make_plan(
    *,
    seed: int,
    configs: list[SweepConfig],
    sample: list[CapturedPrompt],
    max_tokens_floor: int,
    captured: Sequence[CapturedPrompt],
) -> SweepPlan:
    captured_counts: dict[str, int] = {}
    for prompt in captured:
        captured_counts[prompt.schema] = captured_counts.get(prompt.schema, 0) + 1
    plan = SweepPlan(
        seed=seed,
        configs=configs,
        prompts=sample,
        max_tokens_floor=max_tokens_floor,
        captured_prompts=len(captured),
        captured_schema_counts=dict(sorted(captured_counts.items())),
    )
    for config in configs:
        plan.estimates.append(
            ConfigEstimate(
                label=config.label,
                requests=len(sample),
                worst_case_usd=sum(worst_case_cost(p, config, max_tokens_floor) for p in sample),
                expected_usd=sum(expected_cost(p, config, max_tokens_floor) for p in sample),
                price_known=config.price()[1],
            )
        )
    return plan


# --- measurement -------------------------------------------------------------


@dataclass
class SweepRow:
    """One request's measurements. Seconds are from just before the request
    was issued; None means the moment never came (an error, or no text)."""

    order: int
    prompt_index: int
    schema: str
    day: int | None
    phase: str | None
    config: str
    model: str
    effort: str
    speed: str
    position: int
    max_tokens: int
    first_event_s: float | None = None
    first_text_s: float | None = None
    total_s: float | None = None
    input_tokens: int | None = None
    cache_creation_tokens: int | None = None
    cache_read_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    stop_reason: str | None = None
    refusal_category: str | None = None
    served_model: str | None = None
    served_speed: str | None = None
    request_id: str | None = None
    json_valid: bool = False
    text_chars: int | None = None
    cost_usd: float = 0.0
    status_code: int | None = None
    error: str | None = None

    @property
    def completed(self) -> bool:
        """The stream ran to its end (whatever the answer said)."""
        return self.error is None and self.total_s is not None

    @property
    def succeeded(self) -> bool:
        return self.completed and self.json_valid

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"


ROW_FIELDS: tuple[str, ...] = tuple(SweepRow.__dataclass_fields__)


class BudgetGate:
    """Serialises reservations so concurrent requests share one budget.

    A reservation that does not fit while other requests are still in flight
    waits for them to settle and tries again -- their worst cases are almost
    never what they really cost, so giving up at once would strand most of
    the budget. Only when nothing is in flight and the next worst case still
    does not fit is the budget really spent, and `BudgetExceeded` is raised.

    From then on the gate stays shut, even for a cheaper request that would
    still fit: the run stops where it stopped, so every config keeps the
    same prompts and the comparison stays like for like.
    """

    def __init__(self, budget: EvaluationBudget) -> None:
        self.budget = budget
        self._condition = asyncio.Condition()
        self._in_flight = 0
        self._closed: str | None = None

    async def reserve(self, worst_case: float) -> None:
        async with self._condition:
            while True:
                if self._closed is not None:
                    raise BudgetExceeded(self._closed)
                try:
                    self.budget.reserve(worst_case)
                except BudgetExceeded as exc:
                    if self._in_flight == 0:
                        self._closed = str(exc)
                        raise
                    await self._condition.wait()
                    continue
                self._in_flight += 1
                return

    async def settle(self, reserved: float, actual: float) -> None:
        async with self._condition:
            self.budget.settle(reserved, actual)
            self._in_flight -= 1
            self._condition.notify_all()


def _response_cost(message: Any) -> float:
    usage = message.usage
    price, _known = price_for(str(message.model), usage.speed)
    return cost_usd(
        price,
        input_tokens=usage.input_tokens,
        cache_creation_tokens=usage.cache_creation_input_tokens or 0,
        cache_read_tokens=usage.cache_read_input_tokens or 0,
        output_tokens=usage.output_tokens,
    )


def _fill_from_message(row: SweepRow, message: Any, response_schema: type[BaseModel]) -> None:
    usage = message.usage
    row.input_tokens = usage.input_tokens
    row.cache_creation_tokens = usage.cache_creation_input_tokens or 0
    row.cache_read_tokens = usage.cache_read_input_tokens or 0
    row.output_tokens = usage.output_tokens
    details = usage.output_tokens_details
    row.thinking_tokens = details.thinking_tokens if details is not None else None
    row.stop_reason = message.stop_reason
    if message.stop_reason == "refusal" and message.stop_details is not None:
        row.refusal_category = message.stop_details.category
    row.served_model = str(message.model)
    row.served_speed = usage.speed
    text = next((block.text for block in message.content if block.type == "text"), None)
    if text is not None:
        row.text_chars = len(text)
        try:
            response_schema.model_validate_json(text)
            row.json_valid = True
        except ValidationError:
            row.json_valid = False


async def measure(
    job: SweepJob,
    *,
    client: anthropic.AsyncAnthropic,
    gate: BudgetGate,
    max_tokens_floor: int,
    clock: Callable[[], float] = time.perf_counter,
) -> SweepRow:
    """Send one request through the beta streaming API and time it.

    `BudgetExceeded` from the reservation propagates: nothing was sent. Any
    error after that becomes the row's `error`; nothing is retried, because a
    retried request's time is not one request's time.
    """
    prompt, config = job.prompt, job.config
    request = build_request(prompt, config, max_tokens_floor)
    row = SweepRow(
        order=job.order,
        prompt_index=prompt.index,
        schema=prompt.schema,
        day=prompt.day,
        phase=prompt.phase,
        config=config.label,
        model=config.model,
        effort=config.effort,
        speed=config.speed or "standard",
        position=job.position,
        max_tokens=request["max_tokens"],
    )
    worst = worst_case_cost(prompt, config, max_tokens_floor)
    await gate.reserve(worst)
    # Until a response says otherwise -- including if this task is cancelled
    # mid-stream -- the attempt is assumed to have cost its input.
    actual = input_only_cost(prompt, config)
    started = clock()
    try:
        async with client.beta.messages.stream(**request) as stream:
            row.request_id = stream.request_id
            async for event in stream:
                elapsed = clock() - started
                if row.first_event_s is None:
                    row.first_event_s = elapsed
                if (
                    row.first_text_s is None
                    and event.type == "content_block_delta"
                    and event.delta.type == "text_delta"
                    and event.delta.text
                ):
                    row.first_text_s = elapsed
            message = await stream.get_final_message()
        row.total_s = clock() - started
        _fill_from_message(row, message, prompt.response_schema)
        actual = _response_cost(message)
    except Exception as exc:
        row.error = f"{type(exc).__name__}: {exc}"[:_ERROR_TEXT_LIMIT]
        if isinstance(exc, anthropic.APIStatusError):
            row.status_code = exc.status_code
            row.request_id = row.request_id or exc.request_id
        actual = _failed_attempt_cost(exc, actual)
    finally:
        row.cost_usd = actual
        await gate.settle(worst, actual)
    return row


@dataclass
class SweepOutcome:
    rows: list[SweepRow]
    planned: int
    stopped_reason: str | None = None

    @property
    def sent(self) -> int:
        return len(self.rows)


async def run_sweep(
    jobs: Sequence[SweepJob],
    *,
    client: anthropic.AsyncAnthropic,
    budget: EvaluationBudget,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_tokens_floor: int = 12000,
    clock: Callable[[], float] = time.perf_counter,
    on_row: Callable[[SweepRow], None] | None = None,
) -> SweepOutcome:
    """Run the jobs in order with at most `concurrency` in flight.

    On `BudgetExceeded` no further job is started; the ones already in flight
    finish and are kept. Rows come back in job order.
    """
    gate = BudgetGate(budget)
    pending = iter(jobs)
    rows: list[SweepRow] = []
    stop: list[str] = []

    async def worker() -> None:
        while not stop:
            job = next(pending, None)
            if job is None:
                return
            try:
                row = await measure(
                    job, client=client, gate=gate, max_tokens_floor=max_tokens_floor, clock=clock
                )
            except BudgetExceeded as exc:
                if not stop:
                    stop.append(str(exc))
                return
            rows.append(row)
            if on_row is not None:
                on_row(row)

    workers = max(1, min(concurrency, len(jobs) or 1))
    await asyncio.gather(*(worker() for _ in range(workers)))
    rows.sort(key=lambda row: row.order)
    return SweepOutcome(rows=rows, planned=len(jobs), stopped_reason=stop[0] if stop else None)


# --- statistics and reports --------------------------------------------------


def quantile(values: Sequence[float], q: float) -> float | None:
    """Linear interpolation between closest ranks (numpy's default); None for
    no values. Small samples make p90 coarse -- n is printed beside it."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _mean(values: Iterable[float | int | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return statistics.fmean(present) if present else None


@dataclass
class GroupStats:
    config: str
    schema: str
    n: int
    succeeded: int
    refusals: int
    errors: int
    truncated: int
    within_target: int
    completed: int
    first_text_p50: float | None
    first_text_p90: float | None
    total_p50: float | None
    total_p90: float | None
    mean_output_tokens: float | None
    mean_thinking_tokens: float | None
    mean_cost_usd: float | None
    fallback_served: int

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.n if self.n else 0.0


ALL_SCHEMAS = "(全体)"


def group_stats(rows: Sequence[SweepRow], config: str, schema: str) -> GroupStats:
    completed = [row for row in rows if row.completed]
    first_text = [row.first_text_s for row in completed if row.first_text_s is not None]
    totals = [row.total_s for row in completed if row.total_s is not None]
    return GroupStats(
        config=config,
        schema=schema,
        n=len(rows),
        succeeded=sum(1 for row in rows if row.succeeded),
        refusals=sum(1 for row in rows if row.refused),
        errors=sum(1 for row in rows if row.error is not None),
        truncated=sum(1 for row in rows if row.stop_reason == "max_tokens"),
        within_target=sum(1 for total in totals if total <= TARGET_SECONDS),
        completed=len(completed),
        first_text_p50=quantile(first_text, 0.5),
        first_text_p90=quantile(first_text, 0.9),
        total_p50=quantile(totals, 0.5),
        total_p90=quantile(totals, 0.9),
        mean_output_tokens=_mean(row.output_tokens for row in completed),
        mean_thinking_tokens=_mean(row.thinking_tokens for row in completed),
        mean_cost_usd=_mean(row.cost_usd for row in rows),
        # Fallbacks are off, so this should stay 0; a dated snapshot of the
        # requested model is the same model, not a substitute.
        fallback_served=sum(
            1
            for row in completed
            if row.served_model is not None and not is_same_model(row.served_model, row.model)
        ),
    )


def summarize(rows: Sequence[SweepRow], configs: Sequence[SweepConfig]) -> list[GroupStats]:
    """Per config: every schema present, then the config as a whole."""
    stats: list[GroupStats] = []
    schemas = [name for name in SCHEMA_SHARES if any(row.schema == name for row in rows)]
    schemas += sorted({row.schema for row in rows} - set(schemas))
    for config in configs:
        mine = [row for row in rows if row.config == config.label]
        if not mine:
            continue
        for schema in schemas:
            subset = [row for row in mine if row.schema == schema]
            if subset:
                stats.append(group_stats(subset, config.label, schema))
        stats.append(group_stats(mine, config.label, ALL_SCHEMAS))
    return stats


CAVEAT_LINES = (
    "> **この測定は時間だけを測る。発言の質は見ていない。** JSON がスキーマに合うかは数えるが、",
    "> 推理や日本語の良し悪しは判定しない。モデルや effort を変えるときは、必ずゲームの評価",
    "> (`scripts/evaluate.py`)を通してから採用する。別のモデルに替えるかどうかはオーナーが決める。",
)


def _fmt_s(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _fmt_num(value: float | None, digits: int = 0) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _fmt_usd(value: float | None) -> str:
    return "—" if value is None else f"${value:.4f}"


def render_plan(plan: SweepPlan, *, max_cost: float | None) -> str:
    lines = [
        "# レイテンシ測定の計画",
        "",
        *CAVEAT_LINES,
        "",
        f"- シード: {plan.seed}(オフラインの v3 対戦1試合から記録したプロンプト "
        f"{plan.captured_prompts}件)",
        "- 記録されたスキーマ: "
        + "、".join(f"{name} {count}" for name, count in plan.captured_schema_counts.items()),
        f"- 標本: {len(plan.prompts)}件("
        + "、".join(f"{name} {count}" for name, count in plan.sample_schema_counts().items())
        + ")",
        f"- 条件: {len(plan.configs)}個 / 要求: {plan.requests}件(条件 × 標本)",
        f"- max_tokens の下限: {plan.max_tokens_floor}(思考も含む。使われた分だけ課金)",
        "- フォールバック・beta: 送らない(時間をそのモデルだけのものにするため)",
        "",
        "| 条件 | 要求 | 最悪の費用 | 見込みの費用 | 単価 |",
        "|---|---:|---:|---:|---|",
    ]
    for estimate in plan.estimates:
        lines.append(
            f"| `{estimate.label}` | {estimate.requests} | {_fmt_usd(estimate.worst_case_usd)} "
            f"| {_fmt_usd(estimate.expected_usd)} | "
            f"{'表どおり' if estimate.price_known else '表に無い(最も高い行で見積もり)'} |"
        )
    lines += [
        f"| **合計** | {plan.requests} | **{_fmt_usd(plan.worst_case_usd)}** "
        f"| **{_fmt_usd(plan.expected_usd)}** | |",
        "",
        "最悪の費用は、すべての要求が max_tokens を使い切った場合。見込みの費用は effort ごとの"
        "出力トークンの仮定("
        + "・".join(f"{k} {v}" for k, v in EXPECTED_OUTPUT_TOKENS.items())
        + ")による概算で、実測ではない。入力トークンは、最悪の費用では文字数 × 1.2、"
        "見込みでは文字数 × 1.0 とした。",
    ]
    if max_cost is not None:
        fits = plan.prompts_within(max_cost)
        lines += [
            "",
            f"費用の上限(`--max-cost`): ${max_cost:.2f}。各要求の前に最悪の費用を予約し、"
            "入りきらなくなったところで止める(送った分の結果は残る)。",
        ]
        if fits < len(plan.prompts):
            lines.append(
                f"見込みどおりなら、この上限で測れるのは標本 {len(plan.prompts)}件のうち約 {fits}件"
                "(各件を全条件で)。送る順はスキーマと日を混ぜてあるので、途中で止まっても標本の"
                "構成は崩れにくい。全件を測るには、見込みの費用より少し高い上限を渡す。"
            )
    return "\n".join(lines) + "\n"


def render_summary(
    plan: SweepPlan,
    outcome: SweepOutcome,
    stats: Sequence[GroupStats],
    *,
    budget: EvaluationBudget,
    max_cost: float,
    commit: str | None = None,
) -> str:
    lines = [
        "# レイテンシ測定の結果",
        "",
        *CAVEAT_LINES,
        "",
        f"- シード: {plan.seed} / 標本: {len(plan.prompts)}件 / 条件: "
        + "、".join(f"`{config.label}`" for config in plan.configs),
        f"- 送った要求: {outcome.sent} / 計画 {outcome.planned}",
        f"- 推定費用: ${budget.spent_cost:.4f}(上限 ${max_cost:.2f})",
    ]
    if commit:
        lines.append(f"- コミット: `{commit}`")
    if outcome.stopped_reason:
        lines += [
            "",
            f"**費用の上限で止めた**(未送信 {outcome.planned - outcome.sent}件): "
            f"{outcome.stopped_reason}",
        ]
    lines += [
        "",
        f"目標は1発言あたり {TARGET_SECONDS:.0f} 秒以下(approach-reset §6)。議論"
        "(`ChatDiscussionOutput`)の行が主に見る行。投票・夜行動は席どうしで並列に走る。",
        "",
        "| 条件 | スキーマ | n | 成功 | 拒否 | エラー | 打ち切り | 最初の文字 p50 / p90 (秒) "
        f"| 完了 p50 / p90 (秒) | {TARGET_SECONDS:.0f}秒以内 | 出力トークン平均 | うち思考 "
        "| 1回の費用 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stat in stats:
        schema = f"**{stat.schema}**" if stat.schema == ALL_SCHEMAS else f"`{stat.schema}`"
        within = f"{stat.within_target}/{stat.completed}" if stat.completed else "—"
        lines.append(
            f"| `{stat.config}` | {schema} | {stat.n} | {stat.succeeded} "
            f"({stat.success_rate:.0%}) | {stat.refusals} | {stat.errors} | {stat.truncated} "
            f"| {_fmt_s(stat.first_text_p50)} / {_fmt_s(stat.first_text_p90)} "
            f"| {_fmt_s(stat.total_p50)} / {_fmt_s(stat.total_p90)} | {within} "
            f"| {_fmt_num(stat.mean_output_tokens)} | {_fmt_num(stat.mean_thinking_tokens)} "
            f"| {_fmt_usd(stat.mean_cost_usd)} |"
        )
    fallback = sum(stat.fallback_served for stat in stats if stat.schema == ALL_SCHEMAS)
    lines += [
        "",
        "- 成功: ストリームが最後まで届き、本文の JSON がスキーマに合った要求。",
        "- 最初の文字: 最初の本文(text)の断片が届くまで。思考のあいだは何も表示されない。",
        "- 拒否は `stop_reason=refusal`。種類は rows.csv の `refusal_category`。",
        "- p90 は n が小さいと粗い(線形補間)。",
    ]
    if fallback:
        lines.append(
            f"- 要求と違うモデルが応答した要求が {fallback}件ある(rows.csv の `served_model`)。"
        )
    lines += [
        "",
        "**この表で決めてよいのは「速さの候補」まで。** 候補の effort・モデルでゲームの評価を"
        "回し、品質が落ちないことを確かめてから既定を変える。モデルの変更はオーナーが決める。",
    ]
    return "\n".join(lines) + "\n"


def _row_dict(row: SweepRow) -> dict[str, Any]:
    return asdict(row)


def write_rows_csv(rows: Sequence[SweepRow], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROW_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_dict(row))


def _plan_dict(plan: SweepPlan, max_cost: float | None) -> dict[str, Any]:
    return {
        "seed": plan.seed,
        "configs": [config.label for config in plan.configs],
        "requests": plan.requests,
        "max_tokens_floor": plan.max_tokens_floor,
        "captured_prompts": plan.captured_prompts,
        "captured_schema_counts": plan.captured_schema_counts,
        "sample_schema_counts": plan.sample_schema_counts(),
        "sample_prompt_indices": [prompt.index for prompt in plan.prompts],
        "worst_case_usd": round(plan.worst_case_usd, 6),
        "expected_usd": round(plan.expected_usd, 6),
        "estimates": [asdict(estimate) for estimate in plan.estimates],
        "expected_output_tokens_assumption": EXPECTED_OUTPUT_TOKENS,
        "max_cost": max_cost,
        "timing_only": True,
    }


def write_plan_files(plan: SweepPlan, out: Path, *, max_cost: float | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.md").write_text(render_plan(plan, max_cost=max_cost), encoding="utf-8")
    (out / "plan.json").write_text(
        json.dumps(_plan_dict(plan, max_cost), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out / "prompts.jsonl").open("w", encoding="utf-8") as handle:
        for prompt in plan.prompts:
            handle.write(json.dumps(prompt.to_dict(), ensure_ascii=False) + "\n")


# --- CLI ---------------------------------------------------------------------


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive number, got {text!r}")
    return value


def _bounded_int(low: int, high: int) -> Callable[[str], int]:
    def parse(text: str) -> int:
        value = int(text)
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}, got {value}")
        return value

    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--configs", default=DEFAULT_CONFIGS, help="model:effort[:fast] のカンマ区切り"
    )
    parser.add_argument(
        "--prompts", type=_bounded_int(1, MAX_PROMPTS), default=DEFAULT_PROMPTS, help="標本の件数"
    )
    parser.add_argument("--seed", type=int, default=1, help="記録する対戦と標本のシード")
    parser.add_argument(
        "--max-cost", type=_positive_float, default=None, help="推定費用の上限(USD)。実測では必須"
    )
    parser.add_argument("--dry-run", action="store_true", help="計画と費用の見積もりだけ(通信なし)")
    parser.add_argument(
        "--concurrency",
        type=_bounded_int(1, MAX_CONCURRENCY),
        default=DEFAULT_CONCURRENCY,
        help="同時に送る要求の数",
    )
    parser.add_argument(
        "--max-tokens-floor",
        type=_bounded_int(1, 128000),
        default=None,
        help="max_tokens の下限(既定: WEREWOLF_ANTHROPIC_MAX_TOKENS)",
    )
    parser.add_argument(
        "--timeout", type=_positive_float, default=None, help="1要求のタイムアウト秒(既定: 設定値)"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def build_client(settings: Settings, timeout: float) -> anthropic.AsyncAnthropic:
    """Credentials and endpoint from the app's own settings, passed explicitly.

    The SDK's ambient `ANTHROPIC_*` variables are never read: they usually
    belong to some other tool on the machine. Retries are off -- a retried
    request's time is not one request's time.
    """
    api_key = settings.werewolf_anthropic_api_key
    return anthropic.AsyncAnthropic(
        api_key=api_key,
        base_url=settings.werewolf_anthropic_base_url,
        timeout=timeout,
        max_retries=0,
        default_headers=_ambient_header_overrides(api_key),
        open_telemetry=False,
    )


async def _capture(seed: int) -> list[CapturedPrompt]:
    result = await capture_offline_game(seed, engine="v3")
    if not result.finished:
        print(f"  [--] 記録用の対戦が終局しなかった(シード {seed})。記録できた分を使う", flush=True)
    return result.prompts


async def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    client: anthropic.AsyncAnthropic | None = None,
    capture: Callable[[int], Awaitable[list[CapturedPrompt]]] | None = None,
) -> int:
    """`settings`, `client` and `capture` are injectable so the whole flow can
    be tested offline; the command line never passes them."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        configs = parse_configs(args.configs)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.dry_run and args.max_cost is None:
        parser.error("--max-cost is required unless --dry-run is given (this run is billed)")

    settings = settings or Settings()
    floor = args.max_tokens_floor or settings.werewolf_anthropic_max_tokens
    timeout = args.timeout or settings.werewolf_anthropic_timeout_seconds

    print(f"==> 記録: オフラインの v3 対戦(シード {args.seed})", flush=True)
    captured = await (capture or _capture)(args.seed)
    sample = stratified_sample(captured, args.prompts, args.seed)
    if not sample:
        print("記録されたプロンプトに測れるスキーマが無かった", file=sys.stderr)
        return 1
    plan = make_plan(
        seed=args.seed,
        configs=configs,
        sample=sample,
        max_tokens_floor=floor,
        captured=captured,
    )
    write_plan_files(plan, args.out, max_cost=args.max_cost)
    print(render_plan(plan, max_cost=args.max_cost), flush=True)
    for estimate in plan.estimates:
        if not estimate.price_known:
            print(f"  [--] {estimate.label}: 単価表に無いモデル。最も高い行で見積もる", flush=True)

    if args.dry_run:
        print(f"==> dry-run: 送信はしていない。計画は {args.out}/plan.md", flush=True)
        return 0

    if client is None:
        if not settings.werewolf_anthropic_api_key:
            print(
                "WEREWOLF_ANTHROPIC_API_KEY is not set (the SDK's own ANTHROPIC_API_KEY is "
                "deliberately not read). Use --dry-run to see the plan without a key.",
                file=sys.stderr,
            )
            return 2
        client = build_client(settings, timeout)

    budget = EvaluationBudget(max_requests=plan.requests, max_estimated_cost=args.max_cost)
    jobs = interleave(replay_order(plan.prompts, args.seed), plan.configs)
    rows_path = args.out / "rows.jsonl"
    rows_path.write_text("", encoding="utf-8")

    def on_row(row: SweepRow) -> None:
        with rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_row_dict(row), ensure_ascii=False) + "\n")
        status = "ok" if row.succeeded else (row.error or row.stop_reason or "invalid json")
        print(
            f"  [{row.order + 1}/{plan.requests}] {row.config} {row.schema} "
            f"first_text={_fmt_s(row.first_text_s)}s total={_fmt_s(row.total_s)}s {status}",
            flush=True,
        )

    print(f"==> 実測: {plan.requests}件(同時 {args.concurrency})", flush=True)
    outcome = await run_sweep(
        jobs,
        client=client,
        budget=budget,
        concurrency=args.concurrency,
        max_tokens_floor=floor,
        on_row=on_row,
    )
    stats = summarize(outcome.rows, configs)
    write_rows_csv(outcome.rows, args.out / "rows.csv")
    summary = render_summary(
        plan,
        outcome,
        stats,
        budget=budget,
        max_cost=args.max_cost,
        commit=os.environ.get("GITHUB_SHA"),
    )
    (args.out / "summary.md").write_text(summary, encoding="utf-8")
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "plan": _plan_dict(plan, args.max_cost),
                "sent": outcome.sent,
                "planned": outcome.planned,
                "stopped_reason": outcome.stopped_reason,
                "spent_usd": round(budget.spent_cost, 6),
                "stats": [asdict(stat) for stat in stats],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(summary, flush=True)
    print(f"==> 出力先: {args.out}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
