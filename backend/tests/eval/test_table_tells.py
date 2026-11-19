"""The report's tells a repeat human player could learn (owner's review of run 14, X7).

Plain counts pooled over a run's games, read from the record: the order of
the day-1 claims (genuine against fake), seats that conceded a point and
whether they were executed that day or the next, and executions by
personality. Kept out of the release gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.eval.analyzers import AnalysisResult
from app.eval.release_gate import HARD_FAILURE_FIELDS
from app.eval.report import render_report
from app.eval.table_tells import (
    TABLE_TELLS_HEADING,
    claim_order,
    concedes,
    concessions,
    personality_executions,
    render_table_tells,
)
from app.eval.transcript import GameTranscript

RUN14 = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "live-v3-claude-sonnet55-seeds1-3-2026-11-17"
)
NAMES = {"p1": "ユイ", "p2": "マコト", "p3": "リク"}


def _run14() -> list[GameTranscript]:
    return [
        GameTranscript.from_dict(
            json.loads(
                (RUN14 / f"seed{seed}" / f"transcript-seed{seed}.json").read_text(encoding="utf-8")
            )
        )
        for seed in (1, 2, 3)
    ]


# -- what counts as conceding --


@pytest.mark.parametrize(
    "sentence",
    [
        "確かめようがないのは認める、そこは引くよ！",
        "マコトさん、理由が薄いのは認めます",  # a call, not a subject
        "決め手は発言の薄さと票の流れだけで、自分でも弱かったと思う",
        "私の推理は弱かった",
        "ここは引きます",
        "さっきの疑いは撤回する",
    ],
)
def test_a_concession_of_ones_own(sentence: str):
    assert concedes(sentence, NAMES)


@pytest.mark.parametrize(
    "sentence",
    [
        "流れ任せは認めたので、ここは変えません",  # past: mostly someone else's
        "カイト、認めてくれてありがとう",
        "ユイは根拠が弱いと認める",  # another seat is the subject
        "リクの昨日の理由は確かに弱いけど",  # someone else's weakness
        "そこが引っかかる",
        "認めないよ",
    ],
)
def test_not_a_concession(sentence: str):
    assert not concedes(sentence, NAMES)


# -- run 14, pooled --


def test_run14s_day1_claim_order():
    """The genuine seer claimed first in all three games; every fake came second or later."""
    order = claim_order(_run14())

    assert order.genuine[("seer", 1)] == 3
    assert order.fake[("seer", 1, "werewolf")] == 0
    assert order.fake[("seer", 2, "werewolf")] == 3
    assert order.fake[("seer", 3, "werewolf")] == 1
    assert order.genuine[("medium", 1)] == 2
    assert order.fake[("medium", 1, "werewolf")] == 1


def test_run14s_concessions_and_executions():
    found = concessions(_run14())

    assert (found.conceded["village"], found.conceded_executed_same["village"]) == (5, 2)
    assert found.conceded_executed_next["village"] == 2
    assert (found.conceded["werewolf"], found.conceded_executed_same["werewolf"]) == (3, 2)
    assert (found.conceded["fox"], found.conceded_executed_same["fox"]) == (1, 1)
    assert any(
        sentence == "確かめようがないのは認める、そこは引くよ！" for *_, sentence in found.examples
    )


def test_run14s_executions_by_personality_add_up_to_its_executions():
    games = _run14()
    found = personality_executions(games)
    executions = sum(
        death.get("cause") == "executed"
        for game in games
        for death in game.final_state["death_records"]
    )

    assert sum(found.executed.values()) == executions
    assert sum(found.seated.values()) == sum(len(game.personalities) for game in games)
    assert found.executed[("策略家", "fox")] == 2


def test_the_report_carries_the_section_outside_the_gate():
    games = _run14()
    page = render_report(
        games=[(game, AnalysisResult()) for game in games],
        metrics_summary={},
        judge_summary=None,
        provider="AnthropicClaudeProvider",
        model="m",
    )

    assert f"## 7. {TABLE_TELLS_HEADING}" in page
    section = page.split(TABLE_TELLS_HEADING, 1)[1]
    assert "| 占い師 | 1番目 | 3 | 0 | 0 | 0 |" in section
    assert "| 村人陣営 | 5 | 2 | 2 |" in section
    assert "例として読んで" in section
    assert not any("tell" in name or "concession" in name for name in HARD_FAILURE_FIELDS)


def test_no_section_without_a_recorded_game():
    assert render_table_tells([GameTranscript()]) == ""
