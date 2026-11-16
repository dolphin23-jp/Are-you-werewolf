"""`scripts/latency_sweep.py`, fully offline.

The replay tests drive the real `anthropic.AsyncAnthropic` streaming client
over an `httpx2.MockTransport` that answers with server-sent events, so the
streaming method, its event types and the request the sweep sends are the
installed SDK's own -- a wrong name fails here, not on the first paid run.
"""

from __future__ import annotations

import asyncio
import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

from app.ai.provider.anthropic_claude import AnthropicClaudeProvider
from app.ai.provider.base import Message
from app.ai.provider.budget import BudgetExceeded, EvaluationBudget
from app.ai.provider.pricing import cost_usd, price_for
from app.ai.schemas import (
    ChatDiscussionOutput,
    NightActionOutput,
    SummaryOutput,
    VoteOutput,
    WolfChatOutput,
)
from app.config import Settings
from app.eval.prompt_capture import CapturedPrompt
from scripts import latency_sweep as sweep

API_KEY = "test-key"
BASE_URL = "https://example.invalid"
VOTE_JSON = '{"vote_target": "p1", "reason": "ok"}'

SCHEMAS = {
    "ChatDiscussionOutput": ChatDiscussionOutput,
    "VoteOutput": VoteOutput,
    "NightActionOutput": NightActionOutput,
    "WolfChatOutput": WolfChatOutput,
}


def _prompt(index: int, schema: type = VoteOutput, *, day: int = 1, chars: int = 100):
    return CapturedPrompt(
        index=index,
        response_schema=schema,
        system="あなたは人狼ゲームのプレイヤーです。",
        messages=(Message(role="user", content="盤" * chars),),
        max_tokens=800,
        day=day,
        phase="voting",
    )


def _captured(per_schema: dict[str, int]) -> list[CapturedPrompt]:
    prompts: list[CapturedPrompt] = []
    for name, count in per_schema.items():
        for _ in range(count):
            prompts.append(_prompt(len(prompts), SCHEMAS[name], day=1 + len(prompts) % 6))
    return prompts


# --- configs ---------------------------------------------------------------


def test_default_configs_parse_into_the_five_candidates():
    configs = sweep.parse_configs(sweep.DEFAULT_CONFIGS)
    assert [c.label for c in configs] == [
        "claude-opus-5-5:low",
        "claude-opus-5-5:medium",
        "claude-opus-5-5:low:fast",
        "claude-sonnet-5-5:low",
        "claude-haiku-5-5:low",
    ]
    fast = configs[2]
    assert fast.fast and fast.speed == "fast"
    assert configs[0].speed is None


def test_configs_are_normalised_and_whitespace_tolerant():
    configs = sweep.parse_configs(" Claude-Opus-5-5 : LOW , claude-haiku-5-5:max ,")
    assert [c.label for c in configs] == ["claude-opus-5-5:low", "claude-haiku-5-5:max"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("claude-opus-5-5", "model:effort"),
        ("claude-opus-5-5:fastest", "effort"),
        ("claude-opus-5-5:low:slow", "third part"),
        ("claude-sonnet-5-5:low:fast", "fast mode"),
        ("gpt-6-luna:low", "Claude model id"),
        ("claude-opus-5-5:low,claude-opus-5-5:low", "twice"),
        (" , ", "no configs"),
        ("claude-opus-5-5:low:fast:extra", "model:effort"),
    ],
)
def test_bad_configs_are_refused_before_anything_is_sent(text: str, message: str):
    with pytest.raises(ValueError, match=message):
        sweep.parse_configs(text)


# --- sampling --------------------------------------------------------------


def test_sample_follows_the_schema_shares():
    captured = _captured(
        {
            "ChatDiscussionOutput": 57,
            "VoteOutput": 90,
            "NightActionOutput": 14,
            "WolfChatOutput": 20,
        }
    )
    sample = sweep.stratified_sample(captured, 60, seed=1)
    counts = {name: sum(1 for p in sample if p.schema == name) for name in SCHEMAS}
    assert counts == {
        "ChatDiscussionOutput": 24,
        "VoteOutput": 18,
        "NightActionOutput": 9,
        "WolfChatOutput": 9,
    }
    assert [p.index for p in sample] == sorted(p.index for p in sample)
    assert len({p.index for p in sample}) == 60


def test_sample_is_deterministic_by_seed():
    captured = _captured({"ChatDiscussionOutput": 50, "VoteOutput": 50})
    first = [p.index for p in sweep.stratified_sample(captured, 20, seed=7)]
    again = [p.index for p in sweep.stratified_sample(captured, 20, seed=7)]
    other = [p.index for p in sweep.stratified_sample(captured, 20, seed=8)]
    assert first == again
    assert first != other


def test_a_short_schema_hands_its_shortfall_to_the_others():
    captured = _captured(
        {"ChatDiscussionOutput": 40, "VoteOutput": 40, "NightActionOutput": 2, "WolfChatOutput": 3}
    )
    sample = sweep.stratified_sample(captured, 20, seed=1)
    counts = {name: sum(1 for p in sample if p.schema == name) for name in SCHEMAS}
    assert len(sample) == 20
    # 3 night actions were due (15%) but only 2 exist; the gap goes to the
    # largest shares, never to a schema that has run out.
    assert counts["NightActionOutput"] == 2
    assert counts["WolfChatOutput"] == 3
    assert counts["ChatDiscussionOutput"] + counts["VoteOutput"] == 15


def test_sample_never_exceeds_the_capture_and_skips_unmeasured_schemas():
    captured = _captured({"VoteOutput": 3, "WolfChatOutput": 1})
    captured.append(_prompt(len(captured), ChatDiscussionOutput))
    captured.append(_prompt(len(captured), SummaryOutput))
    sample = sweep.stratified_sample(captured, 60, seed=1)
    assert len(sample) == 5
    assert all(p.schema != "SummaryOutput" for p in sample)


def test_replay_order_mixes_schemas_so_any_prefix_is_balanced():
    captured = _captured(
        {
            "ChatDiscussionOutput": 57,
            "VoteOutput": 90,
            "NightActionOutput": 14,
            "WolfChatOutput": 20,
        }
    )
    sample = sweep.stratified_sample(captured, 60, seed=3)
    order = sweep.replay_order(sample, seed=3)
    assert sorted(p.index for p in order) == sorted(p.index for p in sample)
    assert [p.index for p in sweep.replay_order(sample, seed=3)] == [p.index for p in order]
    # A run stopped by its cap after 20 prompts still saw every schema in
    # roughly its share (24/18/9/9 of 60 -> 8/6/3/3 of 20).
    prefix = order[:20]
    counts = {name: sum(1 for p in prefix if p.schema == name) for name in SCHEMAS}
    assert counts == {
        "ChatDiscussionOutput": 8,
        "VoteOutput": 6,
        "NightActionOutput": 3,
        "WolfChatOutput": 3,
    }


# --- interleaving and cost ------------------------------------------------


def test_interleave_runs_every_config_per_prompt_in_a_rotating_order():
    configs = sweep.parse_configs("claude-opus-5-5:low,claude-opus-5-5:medium,claude-haiku-5-5:low")
    prompts = [_prompt(i) for i in range(4)]
    jobs = sweep.interleave(prompts, configs)
    assert [job.order for job in jobs] == list(range(12))
    for i, prompt in enumerate(prompts):
        block = jobs[i * 3 : i * 3 + 3]
        assert {job.prompt.index for job in block} == {prompt.index}
        assert sorted(job.config.label for job in block) == sorted(c.label for c in configs)
        assert [job.position for job in block] == [0, 1, 2]
    firsts = [jobs[i * 3].config.label for i in range(4)]
    assert firsts == [configs[0].label, configs[1].label, configs[2].label, configs[0].label]


def test_worst_case_prices_the_whole_max_tokens_at_the_config_rate():
    prompt = _prompt(0, chars=1000)
    standard, fast = sweep.parse_configs("claude-opus-5-5:low,claude-opus-5-5:low:fast")
    tokens_in = sweep.estimated_input_tokens(prompt)
    assert sweep.worst_case_cost(prompt, standard, 12000) == pytest.approx(
        cost_usd(price_for("claude-opus-5-5")[0], input_tokens=tokens_in, output_tokens=12000)
    )
    # Fast mode is billed from its own row (2x on Opus 5.5).
    assert sweep.worst_case_cost(prompt, fast, 12000) == pytest.approx(
        2 * sweep.worst_case_cost(prompt, standard, 12000)
    )
    # The caller's budget wins when it is above the floor.
    assert sweep.max_tokens_sent(prompt, 500) == 800


def test_plan_totals_and_how_much_fits_under_a_cap():
    configs = sweep.parse_configs("claude-opus-5-5:low,claude-haiku-5-5:low")
    sample = [_prompt(i, chars=2000) for i in range(10)]
    plan = sweep.make_plan(
        seed=1, configs=configs, sample=sample, max_tokens_floor=12000, captured=sample
    )
    assert plan.requests == 20
    assert plan.worst_case_usd == pytest.approx(
        sum(sweep.worst_case_cost(p, c, 12000) for p in sample for c in configs)
    )
    assert plan.expected_usd < plan.worst_case_usd
    assert all(e.price_known for e in plan.estimates)
    assert plan.prompts_within(plan.expected_usd * 10) == 10
    assert plan.prompts_within(plan.expected_usd / 2) == 5
    rendered = sweep.render_plan(plan, max_cost=plan.expected_usd / 2)
    assert "時間だけを測る" in rendered
    assert "約 5件" in rendered


def test_request_matches_what_the_provider_sends_without_fallbacks():
    """The sweep must time the request the game sends; pinned against the
    provider's own builder so a change there shows up here."""
    floor = 12000
    for schema in SCHEMAS.values():
        prompt = _prompt(0, schema)
        for config in sweep.parse_configs("claude-opus-5-5:low,claude-sonnet-5-5:medium"):
            provider = AnthropicClaudeProvider(
                api_key=API_KEY,
                base_url=BASE_URL,
                model=config.model,
                max_tokens_floor=floor,
                fallbacks=False,
                client=_client(lambda request: httpx2.Response(500)),
            )
            expected = provider._build_request(
                prompt.system,
                prompt.wire_messages(),
                schema,
                sweep.max_tokens_sent(prompt, floor),
                config.effort,
                degraded=False,
            )
            assert sweep.build_request(prompt, config, floor) == expected
    (fast,) = sweep.parse_configs("claude-opus-5-5:low:fast")
    request = sweep.build_request(_prompt(0), fast, floor)
    assert request["speed"] == "fast"
    assert "betas" not in request and "fallbacks" not in request


# --- replay over a mock transport -----------------------------------------


def _sse(events: list[dict[str, Any]]) -> bytes:
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        for event in events
    ).encode()


def _stream_events(
    text: str = VOTE_JSON,
    *,
    model: str = "claude-opus-5-5",
    speed: str = "standard",
    stop_reason: str = "end_turn",
    stop_details: dict[str, Any] | None = None,
    input_tokens: int = 1000,
    output_tokens: int = 300,
    thinking_tokens: int = 200,
) -> list[dict[str, Any]]:
    halves = [text[: len(text) // 2], text[len(text) // 2 :]] if text else []
    delta: dict[str, Any] = {"stop_reason": stop_reason, "stop_sequence": None}
    if stop_details is not None:
        delta["stop_details"] = stop_details
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 1,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "speed": speed,
                },
            },
        },
        # Opus 5.5 thinks first; its thinking block streams no visible text.
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""},
        },
        *(
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": h}}
            for h in halves
        ),
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": delta,
            "usage": {
                "output_tokens": output_tokens,
                "output_tokens_details": {"thinking_tokens": thinking_tokens},
            },
        },
        {"type": "message_stop"},
    ]


def _stream_response(events: list[dict[str, Any]], request_id: str = "req_s") -> httpx2.Response:
    return httpx2.Response(
        200,
        headers={"content-type": "text/event-stream", "request-id": request_id},
        content=_sse(events),
    )


def _error(status: int, message: str = "nope") -> httpx2.Response:
    return httpx2.Response(
        status,
        headers={"request-id": f"req_{status}"},
        json={"type": "error", "error": {"type": "api_error", "message": message}},
    )


class _Server:
    """Answers each request with `respond(body)`; records every request."""

    def __init__(self, respond: Callable[[dict[str, Any]], httpx2.Response]) -> None:
        self.respond = respond
        self.requests: list[httpx2.Request] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0)
            return self.respond(json.loads(request.content))
        finally:
            self.in_flight -= 1

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]


def _client(handler: Callable[[httpx2.Request], Any]) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        api_key=API_KEY,
        base_url=BASE_URL,
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )


def _vote_server(**overrides: Any) -> _Server:
    def respond(body: dict[str, Any]) -> httpx2.Response:
        speed = body.get("speed") or "standard"
        return _stream_response(_stream_events(model=body["model"], speed=speed, **overrides))

    return _Server(respond)


class _StepClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


def _job(prompt: CapturedPrompt, label: str = "claude-opus-5-5:low") -> sweep.SweepJob:
    (config,) = sweep.parse_configs(label)
    return sweep.SweepJob(order=0, prompt=prompt, config=config, position=0)


async def _measure(server: _Server, job: sweep.SweepJob, budget: EvaluationBudget | None = None):
    budget = budget or EvaluationBudget(max_requests=10, max_estimated_cost=100.0)
    row = await sweep.measure(
        job,
        client=_client(server.handler),
        gate=sweep.BudgetGate(budget),
        max_tokens_floor=12000,
        clock=_StepClock(),
    )
    return row, budget


async def test_a_streamed_answer_is_timed_and_its_usage_recorded():
    server = _vote_server()
    row, budget = await _measure(server, _job(_prompt(3, day=2)))

    assert row.error is None
    assert row.succeeded and row.json_valid
    # First event, then (after the silent thinking block) the first text,
    # then the end of the stream -- strictly in that order.
    assert row.first_event_s is not None and row.first_text_s is not None
    assert row.total_s is not None
    assert 0 < row.first_event_s < row.first_text_s < row.total_s
    assert (row.input_tokens, row.output_tokens, row.thinking_tokens) == (1000, 300, 200)
    assert row.stop_reason == "end_turn" and row.refusal_category is None
    assert row.served_model == "claude-opus-5-5" and row.served_speed == "standard"
    assert row.request_id == "req_s"
    assert row.text_chars == len(VOTE_JSON)
    assert (row.prompt_index, row.day, row.schema) == (3, 2, "VoteOutput")
    expected = cost_usd(price_for("claude-opus-5-5")[0], input_tokens=1000, output_tokens=300)
    assert row.cost_usd == pytest.approx(expected)
    assert budget.spent_cost == pytest.approx(expected) and budget.reserved_cost == 0

    (request,) = server.requests
    body = server.bodies[0]
    assert body["stream"] is True
    assert body["model"] == "claude-opus-5-5"
    assert body["max_tokens"] == 12000
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["output_config"]["format"]["schema"] == anthropic.transform_schema(VoteOutput)
    # Clean timing: no fallback beta, no fallbacks, no sampling parameters.
    assert "fallbacks" not in body and "speed" not in body and "temperature" not in body
    assert "anthropic-beta" not in request.headers
    # This SDK sends the key as a bearer token; either way, ours and only ours.
    assert (request.headers.get("x-api-key") or request.headers["authorization"]).endswith(API_KEY)


async def test_fast_configs_ask_for_fast_and_are_priced_at_the_fast_row():
    server = _vote_server()
    row, _ = await _measure(server, _job(_prompt(0), "claude-opus-5-5:low:fast"))
    assert server.bodies[0]["speed"] == "fast"
    assert row.speed == "fast" and row.served_speed == "fast"
    assert row.cost_usd == pytest.approx(
        cost_usd(price_for("claude-opus-5-5", "fast")[0], input_tokens=1000, output_tokens=300)
    )


async def test_a_refusal_is_recorded_with_its_category_and_is_not_a_success():
    server = _vote_server(
        text="",
        stop_reason="refusal",
        stop_details={"type": "refusal", "category": "reasoning_extraction", "explanation": None},
    )
    row, _ = await _measure(server, _job(_prompt(0)))
    assert row.refused and not row.succeeded
    assert row.refusal_category == "reasoning_extraction"
    assert row.first_text_s is None
    assert row.completed  # the stream itself ended normally


async def test_invalid_json_is_counted_as_a_failure_not_an_error():
    server = _vote_server(text='{"reason": "no target"}')
    row, _ = await _measure(server, _job(_prompt(0)))
    assert row.completed and not row.json_valid and not row.succeeded
    assert row.error is None


async def test_an_answer_that_did_not_validate_keeps_its_text_for_diagnosis():
    """The first sweep had one fast-mode discussion answer that ended normally
    but did not validate (154 characters), and nothing of it was kept, so the
    cause could not be read. A row that fails validation keeps its text (a
    bounded prefix) and the first validation error."""
    text = '{"reason": "no target"}'
    row, _ = await _measure(_vote_server(text=text), _job(_prompt(0)))

    assert row.invalid_text == text
    assert row.validation_error is not None and "vote_target" in row.validation_error


async def test_a_long_invalid_answer_keeps_a_bounded_prefix():
    text = '{"reason": "' + "長" * 5000
    row, _ = await _measure(_vote_server(text=text), _job(_prompt(0)))

    assert row.invalid_text is not None
    assert len(row.invalid_text) == sweep.INVALID_TEXT_LIMIT
    assert text.startswith(row.invalid_text)


async def test_a_valid_answer_keeps_no_text():
    row, _ = await _measure(_vote_server(), _job(_prompt(0)))

    assert row.json_valid
    assert row.invalid_text is None and row.validation_error is None
    assert sweep.row_status(row) == "ok"


async def test_the_progress_line_says_an_end_turn_answer_did_not_validate():
    row, _ = await _measure(_vote_server(text='{"reason": "x"}'), _job(_prompt(0)))

    assert row.stop_reason == "end_turn"
    assert sweep.row_status(row).startswith("invalid json: vote_target")


async def test_http_errors_become_rows_and_are_charged_like_the_provider_charges_them():
    prompt = _prompt(0, chars=1000)
    job = _job(prompt)
    rejected, budget = await _measure(_Server(lambda body: _error(400, "bad request")), job)
    assert rejected.status_code == 400 and "BadRequestError" in (rejected.error or "")
    assert rejected.cost_usd == 0.0  # a 4xx is refused before any inference
    assert budget.reserved_cost == 0

    overloaded, _ = await _measure(_Server(lambda body: _error(529, "overloaded")), job)
    assert overloaded.status_code == 529 and not overloaded.completed
    assert overloaded.total_s is None
    # Possibly processed after we stopped listening: charged its input.
    assert overloaded.cost_usd == pytest.approx(sweep.input_only_cost(prompt, job.config))


async def test_the_sweep_stops_cleanly_when_the_budget_cannot_fit_another_request():
    configs = sweep.parse_configs("claude-opus-5-5:low,claude-haiku-5-5:low")
    prompts = [_prompt(i) for i in range(5)]
    jobs = sweep.interleave(prompts, configs)
    worst_opus = sweep.worst_case_cost(prompts[0], configs[0], 12000)
    # Room for the first request's worst case, and not much more.
    budget = EvaluationBudget(max_requests=len(jobs), max_estimated_cost=worst_opus * 1.01)
    server = _vote_server(input_tokens=10, output_tokens=12000, thinking_tokens=11000)
    outcome = await sweep.run_sweep(
        jobs,
        client=_client(server.handler),
        budget=budget,
        concurrency=1,
        max_tokens_floor=12000,
        clock=_StepClock(),
    )
    assert outcome.stopped_reason is not None and "budget" in outcome.stopped_reason
    assert 0 < outcome.sent < outcome.planned
    assert len(server.requests) == outcome.sent
    assert budget.spent_cost <= budget.max_estimated_cost
    assert [row.order for row in outcome.rows] == list(range(outcome.sent))


async def test_a_reservation_waits_for_in_flight_requests_instead_of_stopping_early():
    configs = sweep.parse_configs("claude-opus-5-5:low")
    prompts = [_prompt(i) for i in range(6)]
    jobs = sweep.interleave(prompts, configs)
    worst = sweep.worst_case_cost(prompts[0], configs[0], 12000)
    # Only one worst case fits at a time, but each request really costs a
    # fraction of it: with four workers every request must still be sent,
    # one after another, rather than the run stopping at the second.
    budget = EvaluationBudget(max_requests=len(jobs), max_estimated_cost=worst * 1.5)
    server = _vote_server(input_tokens=10, output_tokens=20, thinking_tokens=5)
    seen: list[int] = []
    outcome = await sweep.run_sweep(
        jobs,
        client=_client(server.handler),
        budget=budget,
        concurrency=4,
        max_tokens_floor=12000,
        clock=_StepClock(),
        on_row=lambda row: seen.append(row.order),
    )
    assert outcome.stopped_reason is None
    assert outcome.sent == outcome.planned == 6
    assert server.max_in_flight == 1
    assert sorted(seen) == list(range(6))


async def test_a_cost_stop_leaves_every_config_on_the_same_prompts():
    """A dear request waiting for room must not be overtaken by cheaper later
    ones: the run stops at a point in job order, so the configs are compared
    on the same prompts (all but the last, which the stop can split)."""
    configs = sweep.parse_configs("claude-opus-5-5:low:fast,claude-haiku-5-5:low")
    prompts = [_prompt(i) for i in range(30)]
    jobs = sweep.interleave(prompts, configs)
    worst_opus = sweep.worst_case_cost(prompts[0], configs[0], 12000)
    # One fast-Opus worst case fits beside the cheap requests, two do not;
    # each answer really costs a quarter of its worst case, so the cap is
    # reached part-way through.
    budget = EvaluationBudget(max_requests=len(jobs), max_estimated_cost=worst_opus * 1.5)
    server = _vote_server(input_tokens=10, output_tokens=3000, thinking_tokens=100)

    outcome = await sweep.run_sweep(
        jobs,
        client=_client(server.handler),
        budget=budget,
        concurrency=4,
        max_tokens_floor=12000,
        clock=_StepClock(),
    )

    assert outcome.stopped_reason is not None
    assert 0 < outcome.sent < outcome.planned
    # Admitted strictly in job order: what was sent is a prefix of the plan.
    assert [row.order for row in outcome.rows] == list(range(outcome.sent))
    by_config = {
        config.label: {row.prompt_index for row in outcome.rows if row.config == config.label}
        for config in configs
    }
    sets = list(by_config.values())
    union, common = set.union(*sets), set.intersection(*sets)
    assert len(union - common) <= 1  # only the prompt the stop fell in
    # The summary compares the configs on the prompts every one of them saw.
    stats = sweep.summarize(outcome.rows, configs)
    whole = [stat for stat in stats if stat.schema == sweep.ALL_SCHEMAS]
    assert [stat.n for stat in whole] == [len(common)] * len(configs)


def test_prompts_not_measured_under_every_config_are_left_out_of_the_comparison():
    configs = sweep.parse_configs("claude-opus-5-5:low,claude-haiku-5-5:low")
    rows = [
        _row("claude-opus-5-5:low", "VoteOutput", 1.0, prompt_index=0, order=0),
        _row("claude-haiku-5-5:low", "VoteOutput", 1.0, prompt_index=0, order=1),
        _row("claude-haiku-5-5:low", "VoteOutput", 9.0, prompt_index=1, order=2),
    ]
    assert sweep.comparable_prompts(rows, configs) == {0}
    stats = sweep.summarize(rows, configs)
    assert [(s.config, s.n) for s in stats if s.schema == sweep.ALL_SCHEMAS] == [
        ("claude-opus-5-5:low", 1),
        ("claude-haiku-5-5:low", 1),
    ]
    plan = sweep.SweepPlan(
        seed=1,
        configs=configs,
        prompts=[_prompt(0), _prompt(1)],
        max_tokens_floor=12000,
        captured_prompts=2,
        captured_schema_counts={"VoteOutput": 2},
    )
    outcome = sweep.SweepOutcome(rows=rows, planned=4, stopped_reason="budget would be exceeded")
    text = sweep.render_summary(
        plan,
        outcome,
        stats,
        budget=EvaluationBudget(max_requests=4, max_estimated_cost=1.0),
        max_cost=1.0,
    )
    assert "全条件で測れなかった標本 1件" in text


async def test_once_the_budget_is_spent_the_gate_stays_shut_even_for_a_cheap_request():
    """Stopping mid-way must leave every config with the same prompts, so a
    cheaper config may not keep going after a dearer one was refused."""
    gate = sweep.BudgetGate(EvaluationBudget(max_requests=10, max_estimated_cost=1.0))
    await gate.reserve(0.6)
    await gate.settle(0.6, 0.6)
    with pytest.raises(BudgetExceeded):
        await gate.reserve(0.5)  # nothing in flight, does not fit: spent
    with pytest.raises(BudgetExceeded):
        await gate.reserve(0.01)  # would fit, but the run has stopped
    assert gate.budget.used_requests == 1


async def test_concurrency_is_bounded():
    configs = sweep.parse_configs("claude-opus-5-5:low,claude-haiku-5-5:low")
    jobs = sweep.interleave([_prompt(i) for i in range(6)], configs)
    gate_open = asyncio.Event()
    server = _vote_server()
    in_flight = 0
    peak = 0

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        if in_flight >= 3:
            gate_open.set()
        await asyncio.wait_for(gate_open.wait(), timeout=5)
        in_flight -= 1
        return server.respond(json.loads(request.content))

    outcome = await sweep.run_sweep(
        jobs,
        client=_client(handler),
        budget=EvaluationBudget(max_requests=len(jobs), max_estimated_cost=1000.0),
        concurrency=3,
        max_tokens_floor=12000,
    )
    assert outcome.sent == 12
    assert peak == 3


# --- statistics ------------------------------------------------------------


def test_quantile_interpolates_between_ranks():
    assert sweep.quantile([], 0.5) is None
    assert sweep.quantile([4.0], 0.9) == 4.0
    assert sweep.quantile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert sweep.quantile([10.0, 1.0, 2.0, 3.0, 4.0], 0.9) == pytest.approx(7.6)


def _row(config: str, schema: str, total: float | None, **fields: Any) -> sweep.SweepRow:
    base: dict[str, Any] = {
        "order": 0,
        "prompt_index": 0,
        "schema": schema,
        "day": 1,
        "phase": "discussion",
        "config": config,
        "model": config.split(":")[0],
        "effort": config.split(":")[1],
        "speed": "standard",
        "position": 0,
        "max_tokens": 12000,
        "first_text_s": None if total is None else total / 2,
        "total_s": total,
        "output_tokens": None if total is None else 100,
        "thinking_tokens": None if total is None else 60,
        "served_model": None if total is None else config.split(":")[0],
        "json_valid": total is not None,
        "cost_usd": 0.01,
    }
    base.update(fields)
    return sweep.SweepRow(**base)


def test_group_statistics_count_successes_refusals_and_the_five_second_target():
    rows = [
        _row("claude-opus-5-5:low", "ChatDiscussionOutput", 2.0),
        _row("claude-opus-5-5:low", "ChatDiscussionOutput", 4.0),
        _row("claude-opus-5-5:low", "ChatDiscussionOutput", 8.0, output_tokens=400),
        _row(
            "claude-opus-5-5:low",
            "ChatDiscussionOutput",
            6.0,
            json_valid=False,
            stop_reason="refusal",
            refusal_category="cyber",
        ),
        _row("claude-opus-5-5:low", "ChatDiscussionOutput", None, error="APITimeoutError: x"),
    ]
    stats = sweep.group_stats(rows, "claude-opus-5-5:low", "ChatDiscussionOutput")
    assert (stats.n, stats.succeeded, stats.refusals, stats.errors) == (5, 3, 1, 1)
    assert stats.completed == 4
    assert stats.within_target == 2  # 2.0 and 4.0 s
    assert stats.total_p50 == pytest.approx(5.0)
    assert stats.total_p90 == pytest.approx(7.4)
    assert stats.first_text_p50 == pytest.approx(2.5)
    assert stats.mean_output_tokens == pytest.approx(175.0)
    assert stats.mean_thinking_tokens == pytest.approx(60.0)
    assert stats.mean_cost_usd == pytest.approx(0.01)
    assert stats.success_rate == pytest.approx(0.6)
    assert stats.fallback_served == 0


def test_a_different_served_model_is_flagged_but_a_dated_snapshot_is_not():
    rows = [
        _row("claude-opus-5-5:low", "VoteOutput", 1.0, served_model="claude-opus-5"),
        _row("claude-opus-5-5:low", "VoteOutput", 1.0, served_model="claude-opus-5-5-20261101"),
    ]
    assert sweep.group_stats(rows, "claude-opus-5-5:low", "VoteOutput").fallback_served == 1


def test_summary_is_per_config_and_schema_with_a_whole_config_row():
    configs = sweep.parse_configs("claude-opus-5-5:low,claude-haiku-5-5:low")
    rows = [
        _row("claude-haiku-5-5:low", "VoteOutput", 1.0),
        _row("claude-opus-5-5:low", "VoteOutput", 3.0),
        _row("claude-opus-5-5:low", "ChatDiscussionOutput", 6.0),
    ]
    stats = sweep.summarize(rows, configs)
    assert [(s.config, s.schema) for s in stats] == [
        ("claude-opus-5-5:low", "ChatDiscussionOutput"),
        ("claude-opus-5-5:low", "VoteOutput"),
        ("claude-opus-5-5:low", sweep.ALL_SCHEMAS),
        ("claude-haiku-5-5:low", "VoteOutput"),
        ("claude-haiku-5-5:low", sweep.ALL_SCHEMAS),
    ]
    assert stats[2].n == 2


# --- the command line -----------------------------------------------------


def _fake_capture(prompts: list[CapturedPrompt]):
    async def capture(seed: int) -> list[CapturedPrompt]:
        return prompts

    return capture


def _capture_prompts() -> list[CapturedPrompt]:
    return _captured(
        {"ChatDiscussionOutput": 10, "VoteOutput": 10, "NightActionOutput": 4, "WolfChatOutput": 4}
    )


async def test_dry_run_writes_the_plan_and_needs_no_key_or_network(tmp_path: Path, capsys):
    code = await sweep.main(
        ["--dry-run", "--prompts", "10", "--out", str(tmp_path)],
        settings=Settings(werewolf_anthropic_api_key=""),
        capture=_fake_capture(_capture_prompts()),
    )
    assert code == 0
    assert {p.name for p in tmp_path.iterdir()} == {"plan.md", "plan.json", "prompts.jsonl"}
    plan = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
    assert plan["requests"] == 50 and plan["timing_only"] is True
    assert plan["worst_case_usd"] > plan["expected_usd"] > 0
    lines = (tmp_path / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10 and json.loads(lines[0])["schema"]
    assert "dry-run" in capsys.readouterr().out


async def test_a_paid_run_requires_max_cost():
    with pytest.raises(SystemExit) as exit_info:
        await sweep.main(["--prompts", "4"], capture=_fake_capture(_capture_prompts()))
    assert exit_info.value.code == 2


async def test_a_paid_run_without_the_app_key_refuses(tmp_path: Path, capsys):
    code = await sweep.main(
        ["--max-cost", "1", "--prompts", "4", "--out", str(tmp_path)],
        settings=Settings(werewolf_anthropic_api_key=""),
        capture=_fake_capture(_capture_prompts()),
    )
    assert code == 2
    assert "WEREWOLF_ANTHROPIC_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "rows.csv").exists()


def test_invalid_cli_values_are_refused():
    parser = sweep.build_parser()
    for argv in (
        ["--max-cost", "0"],
        ["--max-cost", "nan"],
        ["--prompts", "0"],
        ["--concurrency", "99"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


async def test_a_full_offline_replay_writes_rows_and_a_summary(tmp_path: Path):
    def respond(body: dict[str, Any]) -> httpx2.Response:
        schema = body["output_config"]["format"]["schema"]
        text = VOTE_JSON
        if "public_message" in schema.get("properties", {}):
            text = '{"public_message": "こんにちは"}'
        elif "message" in schema.get("properties", {}):
            text = '{"message": "様子見"}'
        elif "target" in schema.get("properties", {}):
            text = '{"target": "p2", "reason": "r"}'
        return _stream_response(_stream_events(text, model=body["model"]))

    server = _Server(respond)
    code = await sweep.main(
        [
            "--max-cost",
            "50",
            "--prompts",
            "8",
            "--configs",
            "claude-opus-5-5:low,claude-haiku-5-5:low",
            "--out",
            str(tmp_path),
        ],
        settings=Settings(werewolf_anthropic_api_key=""),
        client=_client(server.handler),
        capture=_fake_capture(_capture_prompts()),
    )
    assert code == 0
    assert len(server.requests) == 16
    with (tmp_path / "rows.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 16
    assert set(rows[0]) == set(sweep.ROW_FIELDS)
    assert all(row["json_valid"] == "True" for row in rows)
    assert len((tmp_path / "rows.jsonl").read_text(encoding="utf-8").splitlines()) == 16
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "時間だけを測る" in summary
    assert "オーナーが決める" in summary
    assert "`claude-haiku-5-5:low`" in summary
    data = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert data["sent"] == data["planned"] == 16
    assert data["stopped_reason"] is None


def test_the_client_takes_the_app_settings_never_the_sdk_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://ambient.invalid")
    client = sweep.build_client(
        Settings(
            werewolf_anthropic_api_key="app-key",
            werewolf_anthropic_base_url="https://app.invalid",
        ),
        timeout=30.0,
    )
    assert client.api_key == "app-key"
    assert str(client.base_url).startswith("https://app.invalid")
    assert client.max_retries == 0
