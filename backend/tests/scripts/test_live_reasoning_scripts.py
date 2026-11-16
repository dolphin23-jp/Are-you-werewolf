"""The live qualification scripts, offline.

Nothing here reaches a model: games are played by the offline doubles, or
the game itself is replaced by a stub that returns a report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from app.ai.metrics import MetricsCollector
from app.ai.provider.budget import EvaluationBudget
from app.ai.provider.mock import MockProvider
from app.config import Settings
from scripts import live_ab_reasoning_check as live_ab
from scripts import live_reasoning_check

# --- the schedule ---------------------------------------------------------------


def test_the_paired_schedule_keeps_its_formal_order() -> None:
    # Exactly the order the hardcoded v2/legacy schedule produced.
    assert live_ab.build_schedule([11, 12, 13], ["legacy", "v2"], 3) == [
        (11, "v2"),
        (11, "legacy"),
        (12, "legacy"),
        (12, "v2"),
        (13, "legacy"),
        (13, "v2"),
    ]


@pytest.mark.parametrize("engine", ["v3", "v2", "legacy"])
def test_any_single_engine_actually_plays(engine: str) -> None:
    assert live_ab.build_schedule([11, 12, 13], [engine], 2) == [(11, engine), (12, engine)]


def test_repeated_engines_are_played_once() -> None:
    assert live_ab.build_schedule([5], ["v3", "v3", "v2"], 3) == [(5, "v3"), (5, "v2")]
    assert live_ab.candidate_engines(["legacy", "v3", "v3"]) == ["v3"]


# --- execute, with the game stubbed out -----------------------------------------


def _args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 12])
    parser.add_argument("--engines", nargs="+", default=["v2"])
    parser.add_argument("--output-dir", type=Path, default=tmp_path)
    parser.add_argument("--max-http-requests", type=int, default=4000)
    parser.add_argument("--max-estimated-cost", type=float, default=20.0)
    parser.add_argument("--max-games", type=int, default=3)
    parser.add_argument("--input-price-per-million", type=float, default=None)
    parser.add_argument("--output-price-per-million", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--review-dir", type=Path, default=None)
    return parser.parse_args(list(extra))


class _StubRun:
    """Stands in for `live_reasoning_check.run`: settles a fixed cost on the
    shared budget, as the Anthropic provider would, and returns a report."""

    def __init__(self, cost: float = 1.5) -> None:
        self.calls: list[tuple[int, str]] = []
        self.cost = cost

    async def __call__(
        self,
        seed: int,
        transcript: Path | None,
        engine: str,
        budget: EvaluationBudget | None = None,
    ) -> dict[str, Any]:
        self.calls.append((seed, engine))
        if budget is not None:
            budget.reserve(self.cost)
            budget.settle(self.cost, self.cost)
        return {
            "seed": seed,
            "engine": engine,
            "game_id": f"live-{seed}-{engine}",
            "model": "claude-opus-5-5",
            "operational_metrics": {"total_calls": 10, "complete_failures": 0},
        }


def test_v3_alone_is_scheduled_measured_and_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _StubRun()
    monkeypatch.setattr(live_ab, "run", stub)

    rows = asyncio.run(live_ab.execute(_args(tmp_path, "--engines", "v3")))

    assert stub.calls == [(11, "v3"), (12, "v3")]
    assert [row["engine"] for row in rows] == ["v3", "v3"]
    aggregate = json.loads((tmp_path / "aggregate.json").read_text(encoding="utf-8"))
    assert aggregate["engines"] == ["v3"]
    assert aggregate["models"] == ["claude-opus-5-5"]
    # The v3 games count toward the gate's coverage bar (two seeds) instead of
    # leaving it at zero as the v2-only filter did.
    assert not any(reason.startswith("live_games=") for reason in aggregate["release_reasons"])
    assert aggregate["stages"]["stage_a_v2_smoke"]["game_ids"] == ["live-11-v3"]
    assert aggregate["budget"]["spent_cost"] == pytest.approx(3.0)
    assert "Qualified engines: v3" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_a_resumed_run_skips_finished_games_and_keeps_the_dollars_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _StubRun()
    monkeypatch.setattr(live_ab, "run", first)
    asyncio.run(live_ab.execute(_args(tmp_path, "--engines", "v3", "--seeds", "11")))
    assert first.calls == [(11, "v3")]

    second = _StubRun()
    monkeypatch.setattr(live_ab, "run", second)
    asyncio.run(
        live_ab.execute(_args(tmp_path, "--engines", "v3", "--seeds", "11", "12", "--resume"))
    )

    assert second.calls == [(12, "v3")]
    budget = json.loads((tmp_path / "aggregate.json").read_text(encoding="utf-8"))["budget"]
    # Without carrying the settled dollars over, a resumed anthropic run would
    # start its cap from zero again.
    assert budget["spent_cost"] == pytest.approx(3.0)
    assert budget["estimated_cost"] == pytest.approx(3.0)


def test_the_cap_carried_over_stops_a_resumed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "aggregate.json").write_text(
        json.dumps({"budget": {"used_requests": 100, "spent_cost": 19.0}}), encoding="utf-8"
    )
    stub = _StubRun(cost=1.5)
    monkeypatch.setattr(live_ab, "run", stub)

    rows = asyncio.run(live_ab.execute(_args(tmp_path, "--engines", "v3", "--resume")))

    # 19 + 1.5 > 20: the first game's reservation is refused before it starts.
    assert rows[0]["status"] == "budget_exhausted"
    assert len(rows) == 1


def test_a_hard_failure_stops_the_candidates_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing(
        seed: int, transcript: Path | None, engine: str, budget: Any = None
    ) -> dict[str, Any]:
        return {"seed": seed, "engine": engine, "game_id": f"g{seed}"}

    def analyze(self: object, transcript: object) -> Any:
        class _Quality:
            def to_dict(self) -> dict[str, int]:
                # Not in the old hand-kept list; it is in the gate's.
                return {"public_fact_flip_count": 1}

        return _Quality()

    monkeypatch.setattr(live_ab, "run", failing)

    def write_transcript(seed: int, engine: str) -> None:
        (tmp_path / f"{seed}-{engine}-transcript.json").write_text("{}", encoding="utf-8")

    for seed in (11, 12):
        write_transcript(seed, "v3")
    monkeypatch.setattr(live_ab.ReasoningTranscriptAnalyzer, "analyze", analyze)

    rows = asyncio.run(live_ab.execute(_args(tmp_path, "--engines", "v3")))

    assert len(rows) == 1  # stopped after the first seed


# --- live_reasoning_check -------------------------------------------------------


class _Labelled(MockProvider):
    model = "claude-opus-5-5"


def test_the_checked_game_is_labelled_with_the_model_that_played(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        live_reasoning_check,
        "get_settings",
        lambda: Settings(werewolf_llm_provider="anthropic"),
    )

    def factory(
        settings: Settings, *, seed: int | None = None, metrics: MetricsCollector | None = None
    ) -> MockProvider:
        return _Labelled(seed=seed, metrics=metrics)

    monkeypatch.setattr(live_reasoning_check, "build_llm_provider", factory)
    transcript = tmp_path / "t.json"
    budget = EvaluationBudget(max_requests=100_000, max_estimated_cost=10.0)

    report = asyncio.run(live_reasoning_check.run(3, transcript, "legacy", budget))

    assert report["model"] == "claude-opus-5-5"
    assert report["provider"] == "anthropic"
    # The budget proxy is not what was measured.
    assert json.loads(transcript.read_text(encoding="utf-8"))["provider"] == "_Labelled"


def test_an_offline_double_is_labelled_as_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        live_reasoning_check,
        "get_settings",
        lambda: Settings(werewolf_llm_provider="scenario"),
    )

    report = asyncio.run(live_reasoning_check.run(3, None, "legacy"))

    assert report["model"] == "(mock)"


def test_anthropic_needs_a_cap_for_a_checked_game(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        live_reasoning_check,
        "get_settings",
        lambda: Settings(werewolf_llm_provider="anthropic"),
    )

    with pytest.raises(SystemExit) as stopped:
        live_reasoning_check.main(["--seed", "1"])

    assert stopped.value.code == 2
    assert "--max-cost" in capsys.readouterr().err
