import asyncio
from pathlib import Path

import pytest

from app.ai.metrics import MetricsCollector
from app.config import Settings
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
