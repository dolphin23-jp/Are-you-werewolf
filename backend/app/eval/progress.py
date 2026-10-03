"""Live progress for long evaluation runs.

A real 17-player game against a reasoning model can take over an hour. The run
used to print three header lines and then nothing until the report was written,
so a stalled run looked exactly like a slow one, and cancelling it threw away
everything the game had produced. This turns the step log into a live view of
the table and keeps a readable partial transcript on disk.

It only observes. Nothing here feeds back into play, so a game is identical with
or without it -- and a failure here must never cost the run.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from app.ai.metrics import MetricsCollector
from app.engine.phases import Phase
from app.eval.report import render_transcript
from app.eval.transcript import GameTranscript, TranscriptRecorder, Utterance

TEXT_LIMIT = 90
HEARTBEAT_SECONDS = 120.0

_PHASE_LABEL = {
    Phase.NIGHT: "夜",
    Phase.DAWN: "夜明け",
    Phase.DISCUSSION: "昼の議論",
    Phase.VOTING: "投票",
    Phase.RUNOFF: "決選投票",
    Phase.VOTE_RESULT: "投票結果",
    Phase.GAME_OVER: "終了",
}


def clock_text(seconds: float) -> str:
    """mm:ss under an hour, h:mm:ss after, so a long run still lines up."""
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _shorten(text: str, limit: int = TEXT_LIMIT) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def format_utterance(utterance: Utterance, names: Mapping[str, str]) -> str | None:
    """One log line for something worth watching; None for internal records."""
    day = f"{utterance.day}日目"
    who = utterance.player_name
    target = names.get(utterance.target or "", utterance.target or "?")
    if utterance.kind == "discussion":
        flag = " [フォールバック]" if utterance.used_fallback else ""
        return f"{day} {who}{flag}: {_shorten(utterance.text)}"
    if utterance.kind == "vote":
        return f"{day} 投票 {who} → {target}"
    if utterance.kind == "night_action":
        return f"{day} 夜行動 {who}({utterance.role}) → {target}"
    if utterance.kind == "wolf_chat":
        return f"{day} [人狼] {who}: {_shorten(utterance.text)}"
    if utterance.kind == "freemason_chat":
        return f"{day} [共有] {who}: {_shorten(utterance.text)}"
    return None


def describe_calls(metrics: MetricsCollector) -> str:
    """Call health so far. Truncation shows up here long before the report."""
    records = list(metrics.records)
    if not records:
        return "呼び出し 0回"
    failed = sum(1 for record in records if not record.succeeded)
    cut = sum(1 for record in records if record.truncated)
    mean = sum(record.latency_seconds for record in records) / len(records)
    text = f"呼び出し {len(records)}回(失敗 {failed}・途中切れ {cut}) 平均 {mean:.1f}秒"
    completion = sum(record.completion_tokens or 0 for record in records)
    reasoning = sum(record.reasoning_tokens or 0 for record in records)
    if completion > 0 and reasoning > 0:
        text += f" 推論トークン {reasoning / completion:.0%}"
    return text


def write_transcript_files(
    transcript: GameTranscript, out_dir: Path, seed: int, *, partial: bool
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"transcript-seed{seed}.json").write_text(
        json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / f"transcript-seed{seed}.md").write_text(
        render_transcript(transcript, partial=partial), encoding="utf-8"
    )


class LiveTranscriptRecorder(TranscriptRecorder):
    """A recorder that also prints each utterance the moment it is recorded."""

    def __init__(
        self,
        emit: Callable[[str], None],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self._emit = emit
        self._clock = clock
        self._started_at = clock()
        self._last_event_at = self._started_at

    def log(self, line: str) -> None:
        # A closed stdout must not end a game that is hours into its run.
        with contextlib.suppress(OSError):
            self._emit(f"[{clock_text(self._clock() - self._started_at)}] {line}")

    def record(self, utterance: Utterance) -> None:
        super().record(utterance)
        self._last_event_at = self._clock()
        line = format_utterance(utterance, self.transcript.names)
        if line is not None:
            self.log(line)

    def seconds_since_last_event(self) -> float:
        return self._clock() - self._last_event_at


class RunProgress:
    """Phase markers, a heartbeat and partial snapshots for one game."""

    def __init__(
        self,
        recorder: LiveTranscriptRecorder,
        metrics: MetricsCollector,
        *,
        seed: int,
        out_dir: Path | None = None,
        heartbeat_seconds: float = HEARTBEAT_SECONDS,
    ) -> None:
        self._recorder = recorder
        self._metrics = metrics
        self._seed = seed
        self._out_dir = out_dir
        self._heartbeat_seconds = heartbeat_seconds
        self._last_phase: tuple[int, Phase] | None = None

    def phase(self, day: int, phase: Phase, snapshot: Callable[[], dict[str, Any]]) -> None:
        """Marks a phase change and saves what the table has produced so far."""
        if (day, phase) == self._last_phase:
            return
        self._last_phase = (day, phase)
        label = _PHASE_LABEL.get(phase, phase.value)
        self._recorder.log(f"── {day}日目 {label} ── {describe_calls(self._metrics)}")
        # The finished game is written by the caller, with its report.
        if phase is not Phase.GAME_OVER:
            self.save_partial(snapshot)

    def interrupted(self, snapshot: Callable[[], dict[str, Any]]) -> None:
        """The game stopped before it ended: cancelled, timed out or crashed.

        The last phase marker is up to a whole discussion old by now, so save
        again; this is the moment the table's words are about to be lost.
        """
        self._recorder.log("── 中断 ── ここまでの記録を保存します")
        self.save_partial(snapshot)

    def save_partial(self, snapshot: Callable[[], dict[str, Any]]) -> None:
        if self._out_dir is None:
            return
        try:
            self._recorder.finalize(snapshot())
            write_transcript_files(
                self._recorder.transcript, self._out_dir, self._seed, partial=True
            )
        except Exception as exc:  # observing must never end the run it observes
            self._recorder.log(f"(途中経過の保存に失敗しました: {type(exc).__name__}: {exc})")

    async def heartbeat(self) -> None:
        """Says so while a model call is in flight and nothing else is printing.

        Utterances already prove the run is alive, so this only speaks after a
        whole interval of silence: that is the moment "slow" and "stuck" need
        telling apart.
        """
        if self._heartbeat_seconds <= 0:
            return
        while True:
            await asyncio.sleep(self._heartbeat_seconds)
            quiet = self._recorder.seconds_since_last_event()
            if quiet >= self._heartbeat_seconds:
                self._recorder.log(
                    f"… 生成中(最後の出来事から {clock_text(quiet)}) "
                    f"{describe_calls(self._metrics)}"
                )
