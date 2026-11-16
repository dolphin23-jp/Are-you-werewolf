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


# --- resuming after the process died mid-game -----------------------------------


class _Crash(Exception):
    """Stands in for whatever ends the process mid-game: Ctrl-C, an OOM, a bug."""


class _SpendingRun:
    """Spends `requests` x `cost` on the shared budget, one reserve/settle per
    request as the Anthropic provider does, and can die in `crash_seed` after
    `crash_after` requests (all of them: just before the game would have
    returned). Records what each game found already spent when it started."""

    def __init__(
        self,
        *,
        cost: float = 0.75,
        requests: int = 10,
        crash_seed: int | None = None,
        crash_after: int = 0,
    ) -> None:
        self.cost = cost
        self.requests = requests
        self.crash_seed = crash_seed
        self.crash_after = crash_after
        self.spent_at_start: dict[int, float] = {}
        self.ledger_seen: list[float] = []
        self.ledger: Path | None = None

    async def __call__(
        self,
        seed: int,
        transcript: Path | None,
        engine: str,
        budget: EvaluationBudget | None = None,
    ) -> dict[str, Any]:
        assert budget is not None
        self.spent_at_start[seed] = budget.spent_cost
        for index in range(self.requests + 1):
            if seed == self.crash_seed and index == self.crash_after:
                raise _Crash("process died")
            if index == self.requests:
                break
            budget.reserve(self.cost)
            budget.settle(self.cost, self.cost)
            if self.ledger is not None:
                # What a SIGKILL right now would leave on disk.
                state = json.loads(self.ledger.read_text(encoding="utf-8"))
                self.ledger_seen.append(state["spent_cost"])
        return {"seed": seed, "engine": engine, "game_id": f"live-{seed}-{engine}"}


def test_a_resumed_run_carries_the_spend_of_a_game_that_died_mid_play(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crashing = _SpendingRun(crash_seed=12, crash_after=10)
    monkeypatch.setattr(live_ab, "run", crashing)
    with pytest.raises(_Crash):
        asyncio.run(
            live_ab.execute(_args(tmp_path, "--engines", "v3", "--seeds", "11", "12", "13"))
        )
    # Seed 11 finished ($7.50); seed 12 spent $7.50 more and then died.

    resumed = _SpendingRun()
    monkeypatch.setattr(live_ab, "run", resumed)
    asyncio.run(
        live_ab.execute(
            _args(tmp_path, "--engines", "v3", "--seeds", "11", "12", "13", "--resume")
        )
    )

    assert sorted(resumed.spent_at_start) == [12]  # 11 was kept; 12 is replayed
    # The replay starts from everything really spent, the dead game included.
    assert resumed.spent_at_start[12] == pytest.approx(15.0)
    budget = json.loads((tmp_path / "aggregate.json").read_text(encoding="utf-8"))["budget"]
    # 15 + 6 x 0.75 = 19.5; the seventh request would cross $20. Real total
    # spend is therefore at most the cap, not the $27 the old books allowed.
    assert budget["spent_cost"] == pytest.approx(19.5)
    assert budget["spent_cost"] <= 20.0


def test_the_spend_is_on_disk_after_every_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SIGKILL or OOM runs no handler, so the ledger must already be current."""
    spending = _SpendingRun(requests=3)
    spending.ledger = tmp_path / live_ab.BUDGET_LEDGER
    monkeypatch.setattr(live_ab, "run", spending)

    asyncio.run(live_ab.execute(_args(tmp_path, "--engines", "v3", "--seeds", "11")))

    assert spending.ledger_seen == pytest.approx([0.75, 1.5, 2.25])


def test_requests_in_flight_at_a_crash_are_counted_at_their_worst_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / live_ab.BUDGET_LEDGER).write_text(
        json.dumps({"used_requests": 6, "spent_cost": 0.0, "reserved_cost": 1.8}),
        encoding="utf-8",
    )
    resumed = _SpendingRun(requests=1)
    monkeypatch.setattr(live_ab, "run", resumed)

    asyncio.run(live_ab.execute(_args(tmp_path, "--engines", "v3", "--seeds", "11", "--resume")))

    # Nobody knows what the six in-flight calls cost, so they count in full.
    assert resumed.spent_at_start[11] == pytest.approx(1.8)


def test_a_restored_snapshot_counts_its_reservations_as_spent() -> None:
    budget = live_ab.restore_budget(
        _args(Path(".")), {"used_requests": 6, "spent_cost": 0.5, "reserved_cost": 1.8}
    )
    assert budget.spent_cost == pytest.approx(2.3)
    assert budget.reserved_cost == 0.0
    assert budget.estimated_cost == pytest.approx(2.3)


def test_a_ledger_that_cannot_be_written_never_fails_a_paid_call(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    budget = live_ab.LedgeredBudget(
        max_requests=10, max_estimated_cost=5.0, ledger=tmp_path / "missing" / "budget.json"
    )
    for _ in range(2):
        budget.reserve(1.0)
        budget.settle(1.0, 0.5)  # the response is already paid for: must not raise

    assert budget.spent_cost == pytest.approx(1.0)
    assert capsys.readouterr().err.count("を書けません") == 1  # said once, not per request


def test_a_fresh_run_starts_its_ledger_from_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / live_ab.BUDGET_LEDGER).write_text(
        json.dumps({"used_requests": 6, "spent_cost": 9.0}), encoding="utf-8"
    )
    spending = _SpendingRun(requests=1)
    monkeypatch.setattr(live_ab, "run", spending)

    asyncio.run(live_ab.execute(_args(tmp_path, "--engines", "v3", "--seeds", "11")))

    assert spending.spent_at_start[11] == 0.0
    ledger = json.loads((tmp_path / live_ab.BUDGET_LEDGER).read_text(encoding="utf-8"))
    assert ledger["spent_cost"] == pytest.approx(0.75)


# --- a real (offline) game stopped by the cap -----------------------------------


class _PricedMock(MockProvider):
    """MockProvider's answers, budgeted like the Anthropic provider: reserve
    before each call, settle afterwards even when cancelled, and let
    `BudgetExceeded` propagate. No network, no SDK."""

    model = "claude-opus-5-5"

    def __init__(
        self, seed: int | None, metrics: MetricsCollector | None, cost: float, in_flight: float
    ) -> None:
        super().__init__(seed=seed, metrics=metrics)
        self._cost = cost
        # Seconds "on the wire" for a ballot: the seats' votes really overlap,
        # each holding its reservation, and outlast the refused one.
        self._in_flight = in_flight
        self._budget: EvaluationBudget | None = None
        self.schemas: list[str] = []

    def set_request_budget(self, budget: EvaluationBudget) -> None:
        self._budget = budget

    async def generate_structured(self, **kwargs: Any) -> Any:
        if self._budget is not None:
            self._budget.reserve(self._cost)
        name = kwargs["response_schema"].__name__
        self.schemas.append(name)
        try:
            if self._in_flight and name == "VoteOutput":
                await asyncio.sleep(self._in_flight)
            return await super().generate_structured(**kwargs)
        finally:
            if self._budget is not None:
                self._budget.settle(self._cost, self._cost)


def _install_priced_mock(
    monkeypatch: pytest.MonkeyPatch, cost: float, in_flight: float = 0.0
) -> list[_PricedMock]:
    built: list[_PricedMock] = []

    def factory(
        settings: Settings, *, seed: int | None = None, metrics: MetricsCollector | None = None
    ) -> _PricedMock:
        provider = _PricedMock(seed, metrics, cost, in_flight)
        built.append(provider)
        return provider

    monkeypatch.setattr(
        live_reasoning_check, "get_settings", lambda: Settings(werewolf_llm_provider="anthropic")
    )
    monkeypatch.setattr(live_reasoning_check, "build_llm_provider", factory)
    return built


# Exactly representable, so the sums the cap is compared with are exact.
_COST = 0.25


def _cap_mid_first_vote(monkeypatch: pytest.MonkeyPatch) -> float:
    """A cap with room for four of the first round's sixteen ballots."""
    probe = _install_priced_mock(monkeypatch, cost=0.0)
    budget = EvaluationBudget(max_requests=100_000, max_estimated_cost=1000.0)
    asyncio.run(live_reasoning_check.run(3, None, "legacy", budget))
    first_vote = probe[0].schemas.index("VoteOutput")
    return (first_vote + 4) * _COST


def test_a_stop_at_the_cap_keeps_the_partial_transcript_and_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cap = _cap_mid_first_vote(monkeypatch)
    _install_priced_mock(monkeypatch, cost=_COST)
    transcript = tmp_path / "t.json"
    budget = EvaluationBudget(max_requests=100_000, max_estimated_cost=cap)

    report = asyncio.run(live_reasoning_check.run(3, transcript, "legacy", budget))

    assert report["status"] == "budget_exhausted"
    assert "budget" in report["budget_stop"]
    assert report["finished"] is False
    assert report["operational_metrics"]["total_calls"] > 0
    saved = json.loads(transcript.read_text(encoding="utf-8"))
    assert saved["utterances"]  # the paid partial game can be read
    assert saved["metrics"]["total_calls"] == report["operational_metrics"]["total_calls"]
    assert budget.spent_cost <= cap


def test_the_command_line_reports_a_capped_game_and_exits_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cap = _cap_mid_first_vote(monkeypatch)
    _install_priced_mock(monkeypatch, cost=_COST)
    transcript = tmp_path / "t.json"

    argv = ["--seed", "3", "--engine", "legacy", "--max-cost", str(cap)]
    code = live_reasoning_check.main([*argv, "--transcript", str(transcript)])

    assert code == live_reasoning_check.EXIT_BUDGET_STOPPED
    out = capsys.readouterr()
    assert live_reasoning_check.BUDGET_STOP_BANNER in out.err
    assert json.loads(out.out)["status"] == "budget_exhausted"
    assert transcript.exists()


def test_any_other_failure_still_leaves_the_partial_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def stop_at_the_vote(self: object, session: object) -> None:
        raise RuntimeError("stopped")

    _install_priced_mock(monkeypatch, cost=_COST)
    monkeypatch.setattr(
        live_reasoning_check.AICoordinator, "generate_all_votes", stop_at_the_vote
    )
    transcript = tmp_path / "t.json"
    budget = EvaluationBudget(max_requests=100_000, max_estimated_cost=1000.0)

    with pytest.raises(RuntimeError, match="stopped"):
        asyncio.run(live_reasoning_check.run(3, transcript, "legacy", budget))

    saved = json.loads(transcript.read_text(encoding="utf-8"))
    assert saved["utterances"]
    assert saved["metrics"]["total_calls"] > 0


def test_a_capped_game_settles_every_seat_before_the_aggregate_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cap = _cap_mid_first_vote(monkeypatch)
    built = _install_priced_mock(monkeypatch, cost=_COST, in_flight=0.2)
    out = tmp_path / "out"

    rows = asyncio.run(
        live_ab.execute(
            _args(
                out,
                "--engines",
                "legacy",
                "--seeds",
                "3",
                "--output-dir",
                str(out),
                "--max-estimated-cost",
                str(cap),
            )
        )
    )

    assert "VoteOutput" in built[0].schemas
    aggregate = json.loads((out / "aggregate.json").read_text(encoding="utf-8"))
    budget = aggregate["budget"]
    # The other seats' ballots were cancelled and settled before the file was
    # written: nothing is left reserved, and every request sent is paid for.
    assert budget["reserved_cost"] == 0.0
    assert budget["used_requests"] == len(built[0].schemas)
    assert budget["spent_cost"] == pytest.approx(budget["used_requests"] * _COST)
    assert budget["spent_cost"] <= cap
    # The capped game kept what a reader needs to audit the spend.
    (row,) = rows
    assert row["status"] == "budget_exhausted"
    assert "operational_metrics" in row and "reasoning_quality" in row
    assert (out / "3-legacy-transcript.json").exists()


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
