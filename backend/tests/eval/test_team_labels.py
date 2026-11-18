"""The winning faction reads the same everywhere a person sees it.

The record said 「妖狐陣営」, while the report's per-game table and the live
log of `scripts/evaluate.py` still printed `fox` / `winner=fox`. One table of
faction labels now serves all three.
"""

from __future__ import annotations

from typing import Any

from app.eval.report import CALL_HEALTH_HEADING, render_report, render_transcript
from app.eval.transcript import GameTranscript
from app.eval.transcript_render import TEAM_LABELS, team_label
from scripts.evaluate import winner_line


def test_one_table_names_the_three_factions():
    assert TEAM_LABELS == {"village": "村人陣営", "werewolf": "人狼陣営", "fox": "妖狐陣営"}
    assert team_label("fox") == "妖狐陣営"
    # Anything else stays as it was, rather than vanishing.
    assert team_label("draw") == "draw"


def _per_game_row(winner: Any, *, finished: bool = True) -> str:
    report = render_report(
        games=[],
        metrics_summary={},
        judge_summary=None,
        provider="AnthropicClaudeProvider",
        model="m",
        per_game=[
            {
                "seed": 1,
                "winner": winner,
                "finished": finished,
                "wall_time_seconds": 60.0,
                "public_utterances": 10,
                "days": 2,
                "metrics": {},
                "hard_failures": None,
            }
        ],
    )
    section = report.split(CALL_HEALTH_HEADING, 1)[1]
    return next(line for line in section.splitlines() if line.startswith("| 1 |"))


def test_the_reports_per_game_table_names_the_faction():
    assert _per_game_row("fox").startswith("| 1 | 妖狐陣営 |")
    assert _per_game_row("werewolf").startswith("| 1 | 人狼陣営 |")
    assert _per_game_row(None, finished=False).startswith("| 1 | (中断) |")


def test_the_live_log_names_the_faction():
    assert winner_line("fox") == "    勝者: 妖狐陣営"
    assert winner_line(None) == "    勝者: (未決着)"


def test_the_record_names_the_faction_from_the_same_table():
    transcript = GameTranscript(
        seed=3,
        names={"p1": "ハルト"},
        roles={"p1": "fox"},
        teams={"p1": "fox"},
        final_state={"winner": "fox", "victory_reason": "妖狐が生存"},
    )

    page = render_transcript(transcript)

    assert "勝者: 妖狐陣営（fox）" in page
    assert "| ハルト (`p1`) | 妖狐 | 妖狐陣営 |" in page
