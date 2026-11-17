"""Every model call of an evaluation game, with its prompt and its answer.

The game record says what each seat said; it does not say what the seat was
shown, how long the call took, what it cost or how the model stopped. The
first Claude runs kept only per-schema averages (`summary.json`), so reading a
turn meant reconstructing its prompt from code and guessing its latency from
the job log's timestamps. `CallLogProvider` sits in front of the game's
provider, notes each request and its answer, and labels the call so the
provider's own `CallRecord` (tokens, cost, stop reason, served model, request
id) can be joined to it. `write_call_log` writes three files per game:

- `calls-seedN.jsonl`: one line per call -- seat, day, phase, schema, timing,
  the record's fields and the parsed answer -- for scripts;
- `calls-seedN.md`: the same calls as a table, for people;
- `prompts-seedN.jsonl.gz`: the exact system prompt and messages of each call
  (several MB uncompressed, so compressed), joined by `index`.

It only observes: the request and the answer pass through unchanged.
"""

from __future__ import annotations

import gzip
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from app.ai.call_context import calling_as, current_call_context
from app.ai.metrics import CallRecord
from app.ai.provider.base import LLMProvider, Message, SchemaT

# (day, phase) at the moment of a call; whoever owns the game state supplies it.
GameContext = tuple[int | None, str | None]


@dataclass
class CallLogEntry:
    index: int
    seat_id: str | None
    day: int | None
    phase: str | None
    schema: str
    max_tokens: int
    system: str
    messages: list[dict[str, str]]
    started_seconds: float
    seconds: float | None = None
    output: dict[str, Any] | None = None
    error: str | None = None

    @property
    def prompt_chars(self) -> int:
        return len(self.system) + sum(len(m["content"]) for m in self.messages)


@dataclass
class CallLogProvider:
    """An `LLMProvider` that logs every call, then delegates it unchanged.

    `_metrics` and `model` are forwarded for the same reason the budget proxy
    forwards them: the coordinator finds the metrics collector by duck typing
    on the provider it is given.
    """

    inner: LLMProvider
    context: Callable[[], GameContext] | None = None
    clock: Callable[[], float] = time.monotonic
    entries: list[CallLogEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._started = self.clock()

    @property
    def _metrics(self) -> object | None:
        return getattr(self.inner, "_metrics", None)

    @property
    def model(self) -> str | None:
        return getattr(self.inner, "model", None)

    async def generate_structured(
        self,
        *,
        system: str,
        messages: list[Message],
        response_schema: type[SchemaT],
        max_tokens: int = 800,
        temperature: float = 0.9,
    ) -> SchemaT | None:
        day, phase = self._where()
        started = self.clock()
        entry = CallLogEntry(
            index=len(self.entries),
            seat_id=current_call_context().seat_id,
            day=day,
            phase=phase,
            schema=response_schema.__name__,
            max_tokens=max_tokens,
            system=system,
            # Copied: the caller owns the list and may keep appending to it.
            messages=[{"role": m.role, "content": m.content} for m in messages],
            started_seconds=round(started - self._started, 3),
        )
        self.entries.append(entry)
        try:
            with calling_as(call_index=entry.index):
                result = await self.inner.generate_structured(
                    system=system,
                    messages=messages,
                    response_schema=response_schema,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
        except BaseException as exc:
            entry.seconds = round(self.clock() - started, 3)
            entry.error = f"{type(exc).__name__}: {exc}"
            raise
        entry.seconds = round(self.clock() - started, 3)
        entry.output = result.model_dump(mode="json") if result is not None else None
        return result

    def _where(self) -> GameContext:
        if self.context is None:
            return None, None
        try:
            return self.context()
        except Exception:  # labelling must never cost the call
            return None, None


def _record_dict(record: CallRecord) -> dict[str, Any]:
    data = asdict(record)
    for key, value in data.items():
        if isinstance(value, Enum):
            data[key] = value.value
    data["succeeded"] = record.succeeded
    return data


def call_rows(
    entries: Sequence[CallLogEntry],
    records: Sequence[CallRecord],
    names: Mapping[str, str],
) -> list[dict[str, Any]]:
    """One row per logged call, joined to the provider's record by call index."""
    by_index: dict[int, list[CallRecord]] = {}
    for record in records:
        if record.call_index is not None:
            by_index.setdefault(record.call_index, []).append(record)
    rows: list[dict[str, Any]] = []
    for entry in entries:
        matched = by_index.get(entry.index, [])
        rows.append(
            {
                "index": entry.index,
                "seat_id": entry.seat_id,
                "seat_name": names.get(entry.seat_id or "", entry.seat_id),
                "day": entry.day,
                "phase": entry.phase,
                "schema": entry.schema,
                "max_tokens": entry.max_tokens,
                "started_seconds": entry.started_seconds,
                "seconds": entry.seconds,
                "prompt_chars": entry.prompt_chars,
                "error": entry.error,
                "records": [_record_dict(record) for record in matched],
                "output": entry.output,
            }
        )
    return rows


def write_call_log(
    entries: Sequence[CallLogEntry],
    records: Sequence[CallRecord],
    names: Mapping[str, str],
    out_dir: Path,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = call_rows(entries, records, names)
    (out_dir / f"calls-seed{seed}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (out_dir / f"calls-seed{seed}.md").write_text(render_call_table(rows, seed), encoding="utf-8")
    with gzip.open(out_dir / f"prompts-seed{seed}.jsonl.gz", "wt", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(
                json.dumps(
                    {
                        "index": entry.index,
                        "seat_id": entry.seat_id,
                        "day": entry.day,
                        "phase": entry.phase,
                        "schema": entry.schema,
                        "max_tokens": entry.max_tokens,
                        "system": entry.system,
                        "messages": entry.messages,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def render_call_table(rows: Sequence[Mapping[str, Any]], seed: int) -> str:
    """Every call as one table row, in the order the calls were made."""
    lines = [
        f"# 呼び出しの記録 (seed={seed})\n",
        "> ゲーム中のモデル呼び出しを、送った順にすべて並べています。秒はゲーム開始からの時刻と、"
        "その呼び出しにかかった時間（順番待ちを含む）。トークンは入力/出力/思考、"
        "費用は推定。プロンプトの全文は `prompts-seed"
        f"{seed}.jsonl.gz`、答えの全文は `calls-seed{seed}.jsonl` の同じ index の行にあります。\n",
        "| # | 開始(秒) | 所要(秒) | 日 | 場面 | 席 | 種類 | effort | 入力 | 出力 | 思考 | "
        "費用($) | 停止 | 応答モデル | 結果 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        records = row.get("records") or []
        record = records[-1] if records else {}
        status = "ok" if record.get("succeeded") else ("—" if not record else "失敗")
        if row.get("error"):
            status = f"例外: {row['error']}"
        elif record.get("error"):
            status = f"{status}: {record['error']}"
        cost = record.get("estimated_cost")
        cells = [
            row["index"],
            row.get("started_seconds"),
            row.get("seconds"),
            row.get("day"),
            row.get("phase"),
            row.get("seat_name") or "-",
            row.get("schema"),
            record.get("effort") or "-",
            record.get("prompt_tokens") if record.get("prompt_tokens") is not None else "-",
            record.get("completion_tokens") if record.get("completion_tokens") is not None else "-",
            record.get("reasoning_tokens") if record.get("reasoning_tokens") is not None else "-",
            f"{cost:.4f}" if isinstance(cost, int | float) else "-",
            record.get("stop_reason") or record.get("finish_reason") or "-",
            record.get("served_model") or "-",
            str(status).replace("|", "｜"),
        ]
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    return "\n".join(lines) + "\n"
