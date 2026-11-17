"""`app/eval/call_log.py`: every model call of a game, with its prompt and answer.

These pin that each call is logged with the seat that made it, that the
provider's own record is joined to it even when calls overlap, that a failing
call is logged and still raised, and the three files a game leaves.
"""

from __future__ import annotations

import asyncio
import gzip
import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from app.ai.call_context import calling_as, current_call_context
from app.ai.metrics import CallRecord, MetricsCollector, ParsePath
from app.ai.provider.base import Message
from app.eval.call_log import CallLogProvider, call_rows, write_call_log


class _Answer(BaseModel):
    text: str


class _Inner:
    """Records a CallRecord the way a real provider does, after a delay."""

    def __init__(self, metrics: MetricsCollector, delays: dict[str, float] | None = None) -> None:
        self._metrics = metrics
        self.model = "fake-model"
        self._delays = delays or {}

    async def generate_structured(
        self, *, system, messages, response_schema, max_tokens=800, temperature=0.9
    ):  # type: ignore[no-untyped-def]
        await asyncio.sleep(self._delays.get(system, 0))
        if system == "boom":
            raise RuntimeError("API が落ちた")
        self._metrics.record(
            CallRecord(
                schema=response_schema.__name__,
                path=ParsePath.STRICT_SCHEMA,
                latency_seconds=0.5,
                attempt=0,
                prompt_tokens=100,
                completion_tokens=20,
                estimated_cost=0.001,
                stop_reason="end_turn",
            )
        )
        return response_schema(text=f"answer to {system}")


def _clock() -> float:
    return 0.0


def test_each_call_is_logged_with_its_seat_prompt_and_answer() -> None:
    metrics = MetricsCollector()
    provider = CallLogProvider(_Inner(metrics), context=lambda: (2, "discussion"), clock=_clock)

    async def run() -> None:
        with calling_as(seat_id="p3"):
            await provider.generate_structured(
                system="sys",
                messages=[Message(role="user", content="hello")],
                response_schema=_Answer,
            )

    asyncio.run(run())

    [entry] = provider.entries
    assert (entry.index, entry.seat_id, entry.day, entry.phase, entry.schema) == (
        0,
        "p3",
        2,
        "discussion",
        "_Answer",
    )
    assert entry.messages == [{"role": "user", "content": "hello"}]
    assert entry.output == {"text": "answer to sys"}
    [record] = metrics.records
    assert (record.seat_id, record.call_index) == ("p3", 0)


def test_overlapping_calls_each_join_their_own_record() -> None:
    metrics = MetricsCollector()
    # The first call finishes last: joining by order would swap them.
    provider = CallLogProvider(_Inner(metrics, delays={"a": 0.05, "b": 0.0}))

    async def call(seat: str, system: str) -> None:
        with calling_as(seat_id=seat):
            await provider.generate_structured(system=system, messages=[], response_schema=_Answer)

    async def run() -> None:
        await asyncio.gather(call("p1", "a"), call("p2", "b"))

    asyncio.run(run())

    rows = call_rows(provider.entries, list(metrics.records), {"p1": "ハルト", "p2": "サクラ"})
    by_seat = {row["seat_name"]: row for row in rows}
    assert by_seat["ハルト"]["output"] == {"text": "answer to a"}
    assert by_seat["ハルト"]["records"][0]["call_index"] == by_seat["ハルト"]["index"]
    assert by_seat["サクラ"]["records"][0]["call_index"] == by_seat["サクラ"]["index"]
    assert current_call_context().seat_id is None


def test_a_failing_call_is_logged_and_still_raised() -> None:
    metrics = MetricsCollector()
    provider = CallLogProvider(_Inner(metrics))

    with pytest.raises(RuntimeError):
        asyncio.run(
            provider.generate_structured(system="boom", messages=[], response_schema=_Answer)
        )

    [entry] = provider.entries
    assert entry.error == "RuntimeError: API が落ちた"
    assert entry.output is None


def test_a_game_leaves_the_call_table_the_rows_and_the_prompts(tmp_path: Path) -> None:
    metrics = MetricsCollector()
    provider = CallLogProvider(_Inner(metrics), context=lambda: (1, "voting"))

    async def run() -> None:
        with calling_as(seat_id="p1"):
            await provider.generate_structured(
                system="全文のシステムプロンプト",
                messages=[Message(role="user", content="盤面")],
                response_schema=_Answer,
            )

    asyncio.run(run())
    write_call_log(provider.entries, list(metrics.records), {"p1": "ハルト"}, tmp_path, 4)

    rows = [json.loads(line) for line in (tmp_path / "calls-seed4.jsonl").read_text().splitlines()]
    assert rows[0]["seat_name"] == "ハルト"
    assert rows[0]["records"][0]["prompt_tokens"] == 100
    assert rows[0]["records"][0]["path"] == "strict_schema"
    table = (tmp_path / "calls-seed4.md").read_text()
    assert "| 0 |" in table and "ハルト" in table and "end_turn" in table and "0.0010" in table
    with gzip.open(tmp_path / "prompts-seed4.jsonl.gz", "rt", encoding="utf-8") as handle:
        prompt = json.loads(handle.readline())
    assert prompt["system"] == "全文のシステムプロンプト"
    assert prompt["messages"] == [{"role": "user", "content": "盤面"}]


def test_a_seat_s_agent_labels_its_calls() -> None:
    from app.ai.personalities import assign_personalities
    from app.ai.player_agent import AIPlayerAgent

    metrics = MetricsCollector()
    provider = CallLogProvider(_Inner(metrics))
    personality = next(iter(assign_personalities(["p5"], seed=1).values()))
    agent = AIPlayerAgent(provider, personality, metrics=metrics, seat_id="p5")

    asyncio.run(agent.generate_night_action("sys", [], ["p1"]))

    assert provider.entries[0].seat_id == "p5"
