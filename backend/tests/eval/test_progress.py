"""The live view of a long evaluation run.

A real game takes over an hour, and until the report is written the run page used
to show nothing. These pin what the log says, that a saved partial transcript
says it is unfinished, and that watching a game can never cost the run.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.ai.metrics import CallRecord, MetricsCollector, ParsePath
from app.engine.phases import Phase
from app.eval.progress import (
    LiveTranscriptRecorder,
    RunProgress,
    clock_text,
    describe_calls,
    format_utterance,
)
from app.eval.report import render_transcript
from app.eval.transcript import GameTranscript, Utterance

NAMES = {"p1": "ハルト", "p2": "サクラ"}


def _utterance(kind: str = "discussion", **changes: object) -> Utterance:
    fields: dict[str, object] = {
        "day": 1,
        "phase": "discussion",
        "kind": kind,
        "player_id": "p1",
        "player_name": "ハルト",
        "role": "seer",
        "team": "village",
        "personality": "穏やか",
        "deception_role": None,
    }
    fields.update(changes)
    return Utterance(**fields)  # type: ignore[arg-type]


def _call(
    *, ok: bool = True, cut: bool = False, seconds: float = 10.0, **tokens: int
) -> CallRecord:
    return CallRecord(
        schema="DiscussionOutput",
        path=ParsePath.JSON_OBJECT if ok else ParsePath.FAILED,
        latency_seconds=seconds,
        attempt=1,
        finish_reason="length" if cut else "stop",
        **tokens,  # type: ignore[arg-type]
    )


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_clock_switches_to_hours_after_an_hour() -> None:
    assert clock_text(65) == "01:05"
    assert clock_text(3725) == "1:02:05"


def test_a_discussion_line_is_one_short_line() -> None:
    text = "最初の行\n二行目 " + "あ" * 200
    line = format_utterance(_utterance(text=text), NAMES)

    assert line is not None
    assert "\n" not in line
    assert line.startswith("1日目 ハルト: 最初の行 二行目")
    assert line.endswith("…")
    assert len(line.split(": ", 1)[1]) <= 90


def test_a_fallback_line_is_flagged_so_a_failing_model_shows_up_live() -> None:
    line = format_utterance(_utterance(text="定型文", used_fallback=True), NAMES)

    assert line == "1日目 ハルト [フォールバック]: 定型文"


def test_votes_and_night_actions_name_the_target_not_its_id() -> None:
    vote = format_utterance(_utterance("vote", target="p2"), NAMES)
    night = format_utterance(_utterance("night_action", target="p2"), NAMES)

    assert vote == "1日目 投票 ハルト → サクラ"
    assert night == "1日目 夜行動 ハルト(seer) → サクラ"


def test_an_unknown_target_falls_back_to_its_id_and_a_missing_one_to_a_question_mark() -> None:
    assert format_utterance(_utterance("vote", target="p9"), NAMES) == "1日目 投票 ハルト → p9"
    assert format_utterance(_utterance("vote", target=None), NAMES) == "1日目 投票 ハルト → ?"


def test_private_chat_is_labelled_by_channel() -> None:
    wolf = format_utterance(_utterance("wolf_chat", text="今夜は静かに"), NAMES)
    mason = format_utterance(_utterance("freemason_chat", text="様子を見よう"), NAMES)

    assert wolf == "1日目 [人狼] ハルト: 今夜は静かに"
    assert mason == "1日目 [共有] ハルト: 様子を見よう"


def test_internal_records_print_nothing() -> None:
    assert format_utterance(_utterance("summary", text="要約"), NAMES) is None
    assert format_utterance(_utterance("vote_change", text="変更"), NAMES) is None


def test_call_health_reports_failures_and_truncation() -> None:
    metrics = MetricsCollector()
    assert describe_calls(metrics) == "呼び出し 0回"

    metrics.record(_call(seconds=10.0, completion_tokens=100, reasoning_tokens=70))
    metrics.record(_call(seconds=20.0, completion_tokens=100, reasoning_tokens=70))
    metrics.record(_call(ok=False, cut=True, seconds=30.0, completion_tokens=100))

    assert describe_calls(metrics) == (
        "呼び出し 3回(失敗 1・途中切れ 1) 平均 20.0秒 推論トークン 47%"
    )


def test_the_recorder_prints_as_it_records_and_still_keeps_the_transcript() -> None:
    clock = _Clock()
    lines: list[str] = []
    recorder = LiveTranscriptRecorder(lines.append, clock=clock)
    recorder.set_roster(
        names=NAMES, roles={}, teams={}, personalities={}, deception={}, seed=1, provider="mock"
    )

    clock.now = 5
    recorder.record(_utterance(text="こんにちは"))
    clock.now = 3725
    recorder.record(_utterance("vote", target="p2"))
    recorder.record(_utterance("summary", text="内部"))

    assert lines == [
        "[00:05] 1日目 ハルト: こんにちは",
        "[1:02:05] 1日目 投票 ハルト → サクラ",
    ]
    assert [u.kind for u in recorder.transcript.utterances] == ["discussion", "vote", "summary"]


def test_a_phase_is_marked_once_and_a_partial_transcript_is_saved(tmp_path: Path) -> None:
    lines: list[str] = []
    recorder = LiveTranscriptRecorder(lines.append, clock=_Clock())
    metrics = MetricsCollector()
    progress = RunProgress(recorder, metrics, seed=7, out_dir=tmp_path, heartbeat_seconds=0)

    progress.phase(1, Phase.DISCUSSION, lambda: {"phase": "discussion"})
    progress.phase(1, Phase.DISCUSSION, lambda: {"phase": "discussion"})
    progress.phase(1, Phase.VOTING, lambda: {"phase": "voting"})

    assert [line for line in lines if "──" in line] == [
        "[00:00] ── 1日目 昼の議論 ── 呼び出し 0回",
        "[00:00] ── 1日目 投票 ── 呼び出し 0回",
    ]
    saved = (tmp_path / "transcript-seed7.md").read_text(encoding="utf-8")
    assert "途中経過です" in saved
    assert "(対戦は未完了)" in saved
    assert json.loads((tmp_path / "transcript-seed7.json").read_text(encoding="utf-8"))[
        "final_state"
    ] == {"phase": "voting"}


def test_the_finished_game_is_left_for_the_caller_to_write(tmp_path: Path) -> None:
    lines: list[str] = []
    recorder = LiveTranscriptRecorder(lines.append, clock=_Clock())
    progress = RunProgress(
        recorder, MetricsCollector(), seed=7, out_dir=tmp_path, heartbeat_seconds=0
    )

    progress.phase(5, Phase.GAME_OVER, lambda: {})

    assert any("終了" in line for line in lines)
    assert list(tmp_path.iterdir()) == []


def test_without_an_out_dir_nothing_is_written(tmp_path: Path) -> None:
    recorder = LiveTranscriptRecorder(lambda line: None, clock=_Clock())
    progress = RunProgress(recorder, MetricsCollector(), seed=7, heartbeat_seconds=0)

    progress.phase(1, Phase.NIGHT, lambda: {})

    assert list(tmp_path.iterdir()) == []


def test_an_interrupted_game_is_saved_as_partial_right_away(tmp_path: Path) -> None:
    lines: list[str] = []
    recorder = LiveTranscriptRecorder(lines.append, clock=_Clock())
    progress = RunProgress(
        recorder, MetricsCollector(), seed=7, out_dir=tmp_path, heartbeat_seconds=0
    )
    recorder.record(_utterance(text="途中までの発言"))

    progress.interrupted(lambda: {"phase": "discussion"})

    assert any("中断" in line for line in lines)
    saved = (tmp_path / "transcript-seed7.md").read_text(encoding="utf-8")
    assert "途中経過です" in saved
    assert "途中までの発言" in saved


def test_a_closed_stdout_does_not_end_the_game() -> None:
    def broken_pipe(line: str) -> None:
        raise BrokenPipeError(line)

    recorder = LiveTranscriptRecorder(broken_pipe, clock=_Clock())
    recorder.record(_utterance(text="まだ対戦中"))

    assert [u.text for u in recorder.transcript.utterances] == ["まだ対戦中"]


def test_a_failed_snapshot_is_reported_but_never_stops_the_run(tmp_path: Path) -> None:
    lines: list[str] = []
    recorder = LiveTranscriptRecorder(lines.append, clock=_Clock())
    progress = RunProgress(
        recorder, MetricsCollector(), seed=7, out_dir=tmp_path, heartbeat_seconds=0
    )

    def broken() -> dict[str, object]:
        raise RuntimeError("boom")

    progress.phase(1, Phase.DISCUSSION, broken)

    assert any("途中経過の保存に失敗" in line and "boom" in line for line in lines)


def test_the_heartbeat_speaks_only_after_a_quiet_interval() -> None:
    async def scenario() -> tuple[list[str], list[str]]:
        clock = _Clock()
        lines: list[str] = []
        recorder = LiveTranscriptRecorder(lines.append, clock=clock)
        progress = RunProgress(recorder, MetricsCollector(), seed=1, heartbeat_seconds=0.01)
        task = asyncio.create_task(progress.heartbeat())
        try:
            await asyncio.sleep(0.05)
            while_busy = list(lines)  # the clock has not moved: nothing was quiet yet
            clock.now = 500.0
            await asyncio.sleep(0.05)
            return while_busy, list(lines)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    while_busy, after_silence = asyncio.run(scenario())

    assert while_busy == []
    assert after_silence
    assert "生成中(最後の出来事から 08:20)" in after_silence[0]


def test_a_zero_interval_turns_the_heartbeat_off() -> None:
    progress = RunProgress(
        LiveTranscriptRecorder(lambda line: None, clock=_Clock()),
        MetricsCollector(),
        seed=1,
        heartbeat_seconds=0,
    )

    asyncio.run(asyncio.wait_for(progress.heartbeat(), timeout=1))


def test_only_a_partial_transcript_says_it_is_unfinished() -> None:
    transcript = GameTranscript(seed=3, final_state={"winner": "village"})

    finished = render_transcript(transcript)
    partial = render_transcript(transcript, partial=True)

    assert "途中経過" not in finished
    assert "勝者: village" in finished
    assert "途中経過です" in partial
    assert "勝者" not in partial
    assert "(対戦は未完了)" in partial
