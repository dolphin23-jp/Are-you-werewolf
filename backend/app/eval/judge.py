"""LLM-scored subjective metrics: Japanese naturalness and persona
consistency.

Two caveats worth stating plainly, because they bound how much these
numbers are worth:

1. **Self-evaluation bias.** By default the judge is the same model that
   produced the text. A model rating its own output tends to be generous.
   Point `--judge-model` at a different model when you can, and treat the
   absolute values as soft; the useful signal is the *relative* change
   between runs after a prompt edit.
2. **Extra spend.** Judging costs additional calls on top of the game
   itself, so it is opt-in (`--judge`) rather than always-on.

The judge never sees hidden roles, only what a player said plus the
persona they were supposed to embody -- so it scores presentation, not
strategic correctness (that is `analyzers.py`'s job).
"""

from __future__ import annotations

import asyncio
import statistics
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from app.ai.provider.base import LLMProvider, Message
from app.eval.transcript import GameTranscript

MAX_LINES_PER_PLAYER = 8


class PersonaScore(BaseModel):
    naturalness: int = Field(ge=1, le=5)
    persona_consistency: int = Field(ge=1, le=5)
    comment: str = ""


@dataclass
class JudgeResult:
    per_player: dict[str, dict[str, Any]]
    summary: dict[str, Any]


_SYSTEM = (
    "あなたは日本語の対話品質を評価する審査員です。"
    "人狼ゲームのプレイヤー1人の発言集と、その人物が演じるはずだった人格設定が与えられます。"
    "以下の2軸を1〜5の整数で評価してください。\n"
    "- naturalness: 日本語として自然か。翻訳調・機械的な繰り返し・不自然な敬語は減点。\n"
    "- persona_consistency: 指定された人格(口調・思考スタイル・議論スタイル・感情傾向)を"
    "一貫して保てているか。\n"
    "推理内容の当否は評価対象外です。表現のみを見てください。"
)

_OUTPUT_INSTRUCTION = (
    "以下のJSON形式で回答してください: "
    '{"naturalness": 1〜5の整数, "persona_consistency": 1〜5の整数, '
    '"comment": "60文字以内の講評"}'
)


async def judge_transcript(
    transcript: GameTranscript,
    provider: LLMProvider,
    *,
    max_concurrency: int = 4,
) -> JudgeResult:
    speakers = sorted({u.player_id for u in transcript.utterances if u.kind == "discussion"})
    semaphore = asyncio.Semaphore(max_concurrency)

    async def score_one(player_id: str) -> tuple[str, dict[str, Any] | None]:
        lines = [
            u.text
            for u in transcript.by_player(player_id)
            if u.kind == "discussion" and u.text.strip()
        ][:MAX_LINES_PER_PLAYER]
        if not lines:
            return player_id, None

        persona = transcript.personalities.get(player_id, "(不明)")
        body = (
            f"【演じるはずだった人格】{persona}\n\n"
            "【このプレイヤーの発言】\n"
            + "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines))
            + f"\n\n{_OUTPUT_INSTRUCTION}"
        )
        async with semaphore:
            scored = await provider.generate_structured(
                system=_SYSTEM,
                messages=[Message(role="user", content=body)],
                response_schema=PersonaScore,
                max_tokens=300,
                temperature=0.0,
            )
        if scored is None:
            return player_id, None
        return player_id, {
            "persona": persona,
            "lines_judged": len(lines),
            "naturalness": scored.naturalness,
            "persona_consistency": scored.persona_consistency,
            "comment": scored.comment,
        }

    scored_pairs = await asyncio.gather(*(score_one(pid) for pid in speakers))
    per_player = {pid: data for pid, data in scored_pairs if data is not None}
    unscored = [pid for pid, data in scored_pairs if data is None]

    if not per_player:
        return JudgeResult(per_player={}, summary={"judged_players": 0, "unscored": unscored})

    naturalness = [d["naturalness"] for d in per_player.values()]
    consistency = [d["persona_consistency"] for d in per_player.values()]
    summary = {
        "judged_players": len(per_player),
        "unscored": unscored,
        "naturalness_mean": round(statistics.fmean(naturalness), 2),
        "naturalness_min": min(naturalness),
        "persona_consistency_mean": round(statistics.fmean(consistency), 2),
        "persona_consistency_min": min(consistency),
        "scale": "1-5 (higher is better)",
    }
    return JudgeResult(per_player=per_player, summary=summary)
