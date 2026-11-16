import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from app.ai.metrics import CallRecord, MetricsCollector, ParsePath
from app.ai.provider.base import Message
from app.ai.provider.budget import EvaluationBudget
from app.ai.provider.mock import MockProvider
from app.config import Settings
from app.eval.release_gate import HARD_FAILURE_FIELDS
from app.eval.report import BUDGET_STOP_BANNER, CALL_HEALTH_HEADING, HARD_CORRECTNESS_HEADING
from app.eval.transcript import DecisionAuditRecord, GameTranscript
from scripts import evaluate
from scripts.evaluate import _make_specs, play_one_game


def test_evaluation_replaces_human_with_seventeenth_ai():
    specs = _make_specs()

    assert len(specs) == 17
    assert [spec.player_id for spec in specs] == [f"p{i}" for i in range(17)]
    assert all(not spec.is_human for spec in specs)
    assert all(spec.name != "観戦席" for spec in specs)


def test_watching_a_game_does_not_change_how_it_is_played(tmp_path: Path) -> None:
    # `legacy` has no reasoning runtime, so this stays fast; the progress output
    # does not depend on the engine.
    settings = Settings(werewolf_llm_provider="mock", werewolf_reasoning_engine="legacy")
    plain = asyncio.run(play_one_game(3, settings, MetricsCollector()))

    lines: list[str] = []
    watched = asyncio.run(
        play_one_game(
            3,
            settings,
            MetricsCollector(),
            emit=lines.append,
            out_dir=tmp_path,
            heartbeat_seconds=0,
        )
    )

    assert [(u.kind, u.player_id, u.text) for u in watched.utterances] == [
        (u.kind, u.player_id, u.text) for u in plain.utterances
    ]
    assert watched.final_state.get("winner") == plain.final_state.get("winner")
    assert any("昼の議論" in line for line in lines)
    assert any("終了" in line for line in lines)
    # The last snapshot taken before the game ended stays on disk, labelled as
    # partial: the caller writes the finished transcript together with the report.
    assert "途中経過です" in (tmp_path / "transcript-seed3.md").read_text(encoding="utf-8")


def test_a_game_that_stops_midway_leaves_what_was_said(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def stop_at_the_vote(self: object, session: object) -> None:
        raise RuntimeError("stopped")

    monkeypatch.setattr("scripts.evaluate.AICoordinator.generate_all_votes", stop_at_the_vote)
    settings = Settings(werewolf_llm_provider="mock", werewolf_reasoning_engine="legacy")
    lines: list[str] = []

    with pytest.raises(RuntimeError, match="stopped"):
        asyncio.run(
            play_one_game(
                3,
                settings,
                MetricsCollector(),
                emit=lines.append,
                out_dir=tmp_path,
                heartbeat_seconds=0,
            )
        )

    saved = (tmp_path / "transcript-seed3.md").read_text(encoding="utf-8")
    assert "途中経過です" in saved
    assert saved.count("- **") >= 5  # the first day's discussion was kept
    assert any("中断" in line for line in lines)


# --- per-game metrics, the spending cap, hard correctness -----------------

_EFFORT = {"ChatDiscussionOutput": "low", "DiscussionOutput": "low", "VoteOutput": "medium"}


class _PricedMock:
    """Answers like `MockProvider`, but prices and budgets every call the way the
    Anthropic provider does: reserve before sending, settle afterwards, and let
    `BudgetExceeded` propagate. No network, no SDK."""

    model = "claude-opus-5-5"

    def __init__(
        self,
        seed: int | None,
        metrics: MetricsCollector | None,
        cost: float,
        *,
        in_flight: float = 0.0,
    ) -> None:
        self._inner = MockProvider(seed=seed)
        self._metrics = metrics
        self._cost = cost
        # Seconds "on the wire". Non-zero makes concurrent calls (the seats'
        # votes) really overlap, each holding its reservation meanwhile, and
        # long enough that the caller would resume before they finish.
        self._in_flight = in_flight
        self._budget: EvaluationBudget | None = None
        self.sent = 0
        self.schemas: list[str] = []

    def set_request_budget(self, budget: EvaluationBudget) -> None:
        self._budget = budget

    async def generate_structured(
        self,
        *,
        system: str,
        messages: list[Message],
        response_schema: Any,
        max_tokens: int = 800,
        temperature: float = 0.9,
    ) -> Any:
        if self._budget is not None:
            self._budget.reserve(self._cost)
        self.sent += 1
        self.schemas.append(response_schema.__name__)
        try:
            if self._in_flight and response_schema.__name__ == "VoteOutput":
                await asyncio.sleep(self._in_flight)
            result = await self._inner.generate_structured(
                system=system, messages=messages, response_schema=response_schema
            )
        finally:
            # Settled even when cancelled, as the real provider does.
            if self._budget is not None:
                self._budget.settle(self._cost, self._cost)
        if self._metrics is not None:
            self._metrics.record(
                CallRecord(
                    schema=response_schema.__name__,
                    path=ParsePath.STRICT_SCHEMA,
                    latency_seconds=0.0,
                    attempt=0,
                    http_requests=1,
                    prompt_tokens=100,
                    completion_tokens=10,
                    stop_reason="end_turn",
                    finish_reason="end_turn",
                    served_model=self.model,
                    requested_model=self.model,
                    estimated_cost=self._cost,
                    effort=_EFFORT.get(response_schema.__name__, "medium"),
                )
            )
        return result


def _install_priced_mock(
    monkeypatch: pytest.MonkeyPatch, cost: float, *, in_flight: float = 0.0
) -> list[_PricedMock]:
    built: list[_PricedMock] = []

    def factory(
        settings: Settings, *, seed: int | None = None, metrics: MetricsCollector | None = None
    ) -> _PricedMock:
        provider = _PricedMock(seed, metrics, cost, in_flight=in_flight)
        built.append(provider)
        return provider

    monkeypatch.setattr(evaluate, "build_llm_provider", factory)
    return built


def _run_main(argv: list[str]) -> int:
    return asyncio.run(evaluate.main(argv))


def test_anthropic_cannot_run_without_a_cost_cap(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as stopped:
        _run_main(["--provider", "anthropic", "--out", str(tmp_path)])

    assert stopped.value.code == 2
    assert "--max-cost" in capsys.readouterr().err
    assert not (tmp_path / "report.md").exists()


def test_the_cap_is_required_for_the_resolved_provider_not_just_the_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # WEREWOLF_LLM_PROVIDER=anthropic spends exactly as much as --provider does,
    # and a model id typed into the provider field resolves to anthropic.
    monkeypatch.setenv("WEREWOLF_LLM_PROVIDER", "claude-opus-5-5")

    with pytest.raises(SystemExit) as stopped:
        _run_main(["--out", str(tmp_path)])

    assert stopped.value.code == 2
    assert "必須" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["--provider", "mock", "--max-cost", "0"],
        ["--provider", "mock", "--max-cost", "-1"],
        # luna prices its usage with --price-in/--price-out; without them the
        # cap could never fire, which is worse than no cap at all.
        ["--provider", "luna", "--max-cost", "5"],
        ["--provider", "mock", "--games", "0"],
    ],
)
def test_unenforceable_caps_are_refused(argv: list[str], tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as stopped:
        _run_main([*argv, "--out", str(tmp_path)])

    assert stopped.value.code == 2


def test_model_label_prefers_the_provider_then_the_setting() -> None:
    anthropic = Settings(werewolf_llm_provider="anthropic", werewolf_anthropic_model="claude-x")
    luna = Settings(werewolf_llm_provider="luna", luna_model="gpt-6-luna")

    assert evaluate.model_label(None, anthropic) == "claude-x"
    assert evaluate.model_label(None, luna) == "gpt-6-luna"
    assert evaluate.model_label(None, Settings(werewolf_llm_provider="scenario")) == "(mock)"
    assert evaluate.model_label(_PricedMock(1, None, 0.0), anthropic) == "claude-opus-5-5"
    assert evaluate.model_label(MockProvider(), Settings()) == "(mock)"


def test_judge_model_swaps_only_the_judges_model() -> None:
    parser = evaluate.build_parser()
    args = parser.parse_args(["--judge-model", "claude-sonnet-5-5"])
    game = Settings(werewolf_llm_provider="anthropic")

    anthropic = evaluate.judge_settings(args, game)
    luna = evaluate.judge_settings(args, Settings(werewolf_llm_provider="luna"))

    assert anthropic.werewolf_anthropic_model == "claude-sonnet-5-5"
    assert game.werewolf_anthropic_model == "claude-opus-5-5"
    assert luna.luna_model == "claude-sonnet-5-5"
    assert evaluate.judge_settings(parser.parse_args([]), game) is game


def test_merged_collectors_pool_every_game() -> None:
    first, second = MetricsCollector(), MetricsCollector()
    first.record(
        CallRecord(
            schema="VoteOutput", path=ParsePath.STRICT_SCHEMA, latency_seconds=1.0, attempt=0
        )
    )
    second.record(
        CallRecord(schema="VoteOutput", path=ParsePath.FAILED, latency_seconds=3.0, attempt=1)
    )
    first.record_event("vote_code_resolved")
    second.record_event("vote_code_resolved")
    second.record_event("brief_fallback")
    first.record_discussion_result(skipped=False)
    second.record_discussion_result(skipped=True)
    first.total_game_wall_time, first.public_utterances, first.game_days = 100.0, 40, 4
    second.total_game_wall_time, second.public_utterances, second.game_days = 60.0, 40, 4

    merged = MetricsCollector.merged([first, second])
    summary = merged.summary()

    assert summary["total_calls"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["events"] == {"brief_fallback": 1, "vote_code_resolved": 2}
    assert summary["discussion_skips"] == 1
    # A pooled ratio, not the mean of the two per-game ratios.
    assert summary["time_per_public_utterance"] == pytest.approx(160.0 / 80)
    assert summary["time_per_game_day"] == pytest.approx(160.0 / 8)
    # The sources are copied, not shared.
    first.record_event("vote_code_resolved")
    assert merged.events["vote_code_resolved"] == 2


def test_each_game_keeps_its_own_metrics_and_the_report_pools_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("WEREWOLF_REASONING_ENGINE", "legacy")

    code = _run_main(["--provider", "mock", "--games", "2", "--seed", "4", "--out", str(tmp_path)])

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert code == 0
    assert [game["seed"] for game in summary["per_game"]] == [4, 5]
    calls = [game["metrics"]["total_calls"] for game in summary["per_game"]]
    assert summary["metrics"]["total_calls"] == sum(calls)
    for game in summary["per_game"]:
        assert game["finished"] is True
        assert game["winner"]
        assert game["public_utterances"] > 0 and game["days"] > 0
        assert game["wall_time_seconds"] > 0
        assert game["hard_failures"] is None  # legacy: not measured
        # The figure the report labels "generation time per public utterance".
        assert game["metrics"]["time_per_public_utterance"] == pytest.approx(
            game["metrics"]["total_game_wall_time"] / game["public_utterances"]
        )
    assert summary["model"] == "(mock)"
    assert summary["reasoning_engine"] == "legacy"
    assert summary["stopped_by_budget"] is False and summary["budget"] is None
    assert "commit" in summary and "settings" not in summary
    assert CALL_HEALTH_HEADING in report
    assert "全席AI" in report  # the per-utterance time says what it measures
    assert BUDGET_STOP_BANNER not in report


def test_the_spending_cap_stops_the_run_and_keeps_what_was_played(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("WEREWOLF_REASONING_ENGINE", "legacy")
    built = _install_priced_mock(monkeypatch, cost=0.01)

    code = _run_main(
        ["--provider", "anthropic", "--max-cost", "0.3", "--games", "3", "--out", str(tmp_path)]
    )

    assert code == evaluate.EXIT_BUDGET_STOPPED
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert BUDGET_STOP_BANNER in report.split("\n- プロバイダ")[0]  # above everything else
    assert summary["stopped_by_budget"] is True
    assert summary["games_played"] == 1  # the second game never started
    assert summary["per_game"][0]["finished"] is False
    assert summary["per_game"][0]["budget_stop"]
    # Never a request past the cap, and nothing left in flight.
    budget = summary["budget"]
    assert budget["spent_cost"] <= 0.3 + 1e-9
    assert budget["reserved_cost"] == pytest.approx(0.0)
    assert sum(provider.sent for provider in built) == budget["used_requests"]
    # The partial game is on disk and says so.
    assert "途中経過です" in (tmp_path / "transcript-seed1.md").read_text(encoding="utf-8")
    assert (tmp_path / "transcript-seed1.json").exists()
    # Labelled as what was asked for, with the settings a baseline needs.
    assert summary["model"] == "claude-opus-5-5"
    assert summary["settings"]["anthropic_effort_by_kind"]["discussion"] == "low"
    assert summary["metrics"]["pricing_source"] == "table"
    by_schema = summary["metrics"]["by_schema"]
    assert by_schema
    for schema, data in by_schema.items():
        assert data["effort"] == _EFFORT.get(schema, "medium")


def test_a_stop_in_the_middle_of_the_votes_leaves_nothing_in_flight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("WEREWOLF_REASONING_ENGINE", "legacy")
    # A free probe game finds where the first vote round starts.
    probe = _install_priced_mock(monkeypatch, cost=0.0)
    assert _run_main(["--provider", "mock", "--out", str(tmp_path / "probe")]) == 0
    first_vote = probe[0].schemas.index("VoteOutput")

    built = _install_priced_mock(monkeypatch, cost=0.01, in_flight=0.2)
    cap = (first_vote + 4) * 0.01  # room for four of the sixteen ballots
    code = _run_main(
        ["--provider", "anthropic", "--max-cost", f"{cap:.2f}", "--out", str(tmp_path / "run")]
    )

    assert code == evaluate.EXIT_BUDGET_STOPPED
    budget = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))["budget"]
    assert "VoteOutput" in built[0].schemas
    # The other seats' ballots were cancelled and settled before the report
    # read the budget: nothing reserved, nothing over the cap.
    assert budget["reserved_cost"] == pytest.approx(0.0)
    assert budget["spent_cost"] <= cap + 1e-9
    assert budget["used_requests"] == built[0].sent


def test_the_judge_shares_the_cap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WEREWOLF_REASONING_ENGINE", "legacy")
    built = _install_priced_mock(monkeypatch, cost=0.0)

    code = _run_main(
        ["--provider", "anthropic", "--max-cost", "1", "--judge", "--out", str(tmp_path)]
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert code == 0
    judge_provider = built[-1]
    assert judge_provider.sent > 0
    # The judge's calls are counted against the one budget, but kept out of
    # the game's own metrics.
    assert summary["budget"]["used_requests"] == sum(provider.sent for provider in built)
    assert summary["judge"]["metrics"]["total_calls"] == judge_provider.sent
    assert summary["metrics"]["total_calls"] == built[0].sent


def test_a_cap_that_is_not_reached_changes_nothing() -> None:
    settings = Settings(werewolf_llm_provider="mock", werewolf_reasoning_engine="legacy")
    plain = asyncio.run(evaluate.play_game(3, settings, MetricsCollector()))
    budget = EvaluationBudget(max_requests=10_000, max_estimated_cost=100.0)
    capped = asyncio.run(evaluate.play_game(3, settings, MetricsCollector(), budget=budget))

    assert capped.budget_stop is None and capped.finished
    assert [(u.kind, u.player_id, u.text) for u in capped.transcript.utterances] == [
        (u.kind, u.player_id, u.text) for u in plain.transcript.utterances
    ]
    # The proxy is plumbing: the transcript still names the provider under test.
    assert capped.transcript.provider == "MockProvider"
    assert budget.used_requests == len(capped.metrics.records)


def test_hard_correctness_is_the_release_gate_view_of_the_audits() -> None:
    transcript = GameTranscript(game_id="g")
    transcript.decision_audits.append(
        DecisionAuditRecord(
            game_id="g",
            day=2,
            phase="voting",
            player_id="p3",
            decision_target="p4",
            displayed_target="p4",
            vote_target="p4",
            ally_vote=True,
            ally_vote_planned=False,
        )
    )
    run = evaluate.GameRun(
        seed=1,
        transcript=transcript,
        metrics=MetricsCollector(),
        model="(mock)",
        engine="v3",
        finished=True,
        wall_time_seconds=1.0,
        validation={},
    )

    hard, quality = evaluate.hard_correctness(run)

    assert hard is not None and quality is not None
    assert set(hard) == set(HARD_FAILURE_FIELDS)
    assert hard["unplanned_wolf_ally_vote_count"] == 1
    assert sum(hard.values()) == 1
    assert quality["wolf_ally_vote_count"] == 1
    # legacy keeps no audit records: "not measured", never a vacuous pass.
    run.engine = "legacy"
    assert evaluate.hard_correctness(run) == (None, None)


def test_a_v3_game_reports_its_hard_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WEREWOLF_REASONING_ENGINE", "v3")

    assert _run_main(["--provider", "mock", "--out", str(tmp_path)]) == 0

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    game = summary["per_game"][0]
    assert summary["reasoning_engine"] == "v3"
    assert set(game["hard_failures"]) == set(HARD_FAILURE_FIELDS)
    assert game["reasoning_quality"]["decision_count"] > 0
    assert set(game["validation"]) == {
        "dead_target_selections",
        "public_fact_flips",
        "vote_history_misreads",
        "vote_plan_mismatches",
    }
    section = report.split(HARD_CORRECTNESS_HEADING)[1].split("## 4.")[0]
    assert "計測していません" not in section
    assert ("**合格**" in section) == (sum(game["hard_failures"].values()) == 0)


def test_commit_sha_prefers_the_actions_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    assert evaluate.commit_sha() == "abc123"


def test_commit_sha_tolerates_a_missing_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_SHA", raising=False)

    def no_git(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("git")

    monkeypatch.setattr(evaluate.subprocess, "run", no_git)
    assert evaluate.commit_sha() is None
