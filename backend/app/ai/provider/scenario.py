"""Offline provider that exercises the *structured* half of the discussion
contract, which `MockProvider` never touches.

`MockProvider` picks one of nine canned lines and fills in `overall_thought`.
That is enough to prove a game completes, but it means `reply_to`, `quote`,
`directed_questions`, `agrees_with`, `public_results` and
`reasoning_memo.execution_target` are all left empty -- so the reply/quote
plumbing, the pending-question lifecycle, the key-point ledger and the
pressured-candidate machinery never run in an end-to-end test, only in narrow
unit tests with hand-written doubles.

This provider fills those fields, so `tests/e2e/test_scenario_ai_full_game.py`
drives the same code paths a real model would. It is still completely offline
and seed-deterministic: no network, no cost, safe in CI. `MockProvider` stays
the default and is deliberately left untouched, since the rest of the suite
pins its behaviour.

Targets are drawn from ids the prompt actually mentions -- the same discipline
`MockProvider` follows -- because a hallucinated id would be rejected
downstream and the interesting paths would never be reached.
"""

from __future__ import annotations

import asyncio
import random
import re
from time import perf_counter

from app.ai.metrics import CallRecord, MetricsCollector, ParsePath
from app.ai.provider.base import Message, SchemaT
from app.ai.schemas import (
    DirectedQuestion,
    DiscussionOutput,
    MorningIntentOutput,
    NightActionOutput,
    PublicResultClaim,
    ReasoningMemo,
    SummaryOutput,
    VoteOutput,
    WolfChatOutput,
)

_PLAYER_ID_RE = re.compile(r"\bp\d+\b")
_MESSAGE_ID_RE = re.compile(r"\[(m\d+)")
_SELF_ID_RE = re.compile(r"あなた自身のplayer_idは (p\d+) です")
# The pending-question layer renders one `[mN] 名前(pN) →あなた:「...」` line per
# open question, and falls back to a literal "(ありません)" when there are none.
_PENDING_BLOCK_RE = re.compile(r"【あなたへの未回答の質問】\n(.*?)\n最初にこれへ", re.DOTALL)

_OPENING_LINES = [
    "占い師CO、初日は様子見でした。",
    "霊媒師です。COします。",
    "共有者です。COします。",
    "今日は発言量の差を見ていきたいです。",
    "グレーを順番に詰めていきましょう。",
]

_ARGUMENT_LINES = [
    "その主張は投票履歴と噛み合っていません。",
    "占い先の理由をもう少し具体的に聞かせてください。",
    "内訳を整理すると、そちらの方が自然だと思います。",
    "昨日からの意見の変化が気になっています。",
]

_AGREEMENT_LINES = [
    "同意です。",
    "その見方でいいと思います。",
    "把握しました。",
]

_WOLF_LINES = [
    "今夜は様子を見て潜伏を続けよう。",
    "怪しまれている人を狙うのはどうだろう。",
    "占い師っぽい人を早めに処理したい。",
]

_CO_ROLE_BY_PREFIX = {
    "占い師CO": "seer",
    "霊媒師": "medium",
    "共有者": "freemason",
}


class ScenarioProvider:
    """Deterministic given a seed. Emits schema-valid, *structurally populated*
    responses so the full discussion machinery runs offline."""

    def __init__(
        self,
        seed: int | None = None,
        metrics: MetricsCollector | None = None,
        latency_seconds: float = 0.0,
    ) -> None:
        self._rng = random.Random(seed)
        self._metrics = metrics
        self._latency_seconds = latency_seconds

    async def generate_structured(
        self,
        *,
        system: str,
        messages: list[Message],
        response_schema: type[SchemaT],
        max_tokens: int = 800,
        temperature: float = 0.9,
    ) -> SchemaT | None:
        started = perf_counter()
        if self._latency_seconds:
            await asyncio.sleep(self._latency_seconds)

        text = system + "\n" + "\n".join(m.content for m in messages)
        result = self._build(text, response_schema)

        if self._metrics is not None:
            self._metrics.record(
                CallRecord(
                    schema=response_schema.__name__,
                    path=ParsePath.STRICT_SCHEMA if result is not None else ParsePath.FAILED,
                    latency_seconds=perf_counter() - started,
                    attempt=0,
                    prompt_tokens=None,
                    completion_tokens=None,
                )
            )
        return result  # type: ignore[return-value]

    # -- response construction --

    def _build(self, text: str, response_schema: type[SchemaT]) -> object:
        self_id = self._self_id(text)
        others = self._other_player_ids(text, self_id)
        pick = self._rng.choice(others) if others else self_id

        if response_schema is DiscussionOutput:
            return self._discussion(text, self_id, others)
        if response_schema is MorningIntentOutput:
            return MorningIntentOutput()
        if response_schema is VoteOutput:
            return VoteOutput(vote_target=pick, reason="シナリオの投票理由です。")
        if response_schema is NightActionOutput:
            return NightActionOutput(target=pick, reason="シナリオの夜行動理由です。")
        if response_schema is WolfChatOutput:
            return WolfChatOutput(message=self._rng.choice(_WOLF_LINES))
        if response_schema is SummaryOutput:
            return SummaryOutput(summary="シナリオの要約です。")
        return None

    def _discussion(self, text: str, self_id: str, others: list[str]) -> DiscussionOutput:
        answering = self._pending_question_message_id(text)
        message_ids = self._message_ids(text)
        target = self._rng.choice(others) if others else self_id

        # An open question addressed to us is answered first, mirroring the
        # instruction the prompt gives -- this is what closes the pending-question
        # lifecycle end to end.
        if answering is not None:
            return DiscussionOutput(
                public_message="ご質問にお答えします。理由は発言量の差です。",
                key_point="質問への回答を提示した",
                reply_to=answering,
                quote="ご指摘の点について",
                reasoning_memo=ReasoningMemo(
                    overall_thought="質問に答えました。",
                    execution_target=target,
                    suspects=[target],
                ),
                ready_to_vote=False,
            )

        # Agreement with no new argument: exercises the reaction path and the
        # sentence-boundary shortening that goes with it.
        if message_ids and self._rng.random() < 0.3:
            return DiscussionOutput(
                public_message=self._rng.choice(_AGREEMENT_LINES),
                key_point="",
                agrees_with=[self._rng.choice(message_ids)],
                reasoning_memo=ReasoningMemo(
                    overall_thought="既出の論点に同意しました。",
                    execution_target=target,
                ),
                ready_to_vote=True,
            )

        first_turn = not message_ids
        line = (
            self._rng.choice(_OPENING_LINES) if first_turn else self._rng.choice(_ARGUMENT_LINES)
        )
        claim_role = next(
            (role for prefix, role in _CO_ROLE_BY_PREFIX.items() if line.startswith(prefix)),
            None,
        )
        results: list[PublicResultClaim] = []
        if claim_role in ("seer", "medium") and others:
            # The coordinator only registers a claim whose target is named in the
            # public text, so the message has to carry the id it reports.
            result_target = self._rng.choice(others)
            line = f"{line} {result_target}は白です。"
            results = [
                PublicResultClaim(
                    result_type=claim_role, target_id=result_target, is_werewolf=False
                )
            ]

        return DiscussionOutput(
            public_message=line,
            key_point=f"{target}への疑いを提示した",
            reply_to=self._rng.choice(message_ids) if message_ids else None,
            public_claim_role=claim_role,
            contains_co_claim=claim_role is not None,
            public_results=results,
            directed_questions=[
                DirectedQuestion(
                    target_id=target,
                    question="その根拠を具体的に説明してください。",
                    source_message_id=message_ids[-1] if message_ids else None,
                )
            ],
            reasoning_memo=ReasoningMemo(
                overall_thought="シナリオの思考メモです。",
                execution_target=target,
                suspects=[target],
            ),
            ready_to_vote=self._rng.random() < 0.5,
            needs_another_statement=self._rng.random() < 0.3,
        )

    # -- prompt scraping --

    @staticmethod
    def _self_id(text: str) -> str:
        match = _SELF_ID_RE.search(text)
        return match.group(1) if match else "p0"

    @staticmethod
    def _other_player_ids(text: str, self_id: str) -> list[str]:
        """Never returns the speaker itself: self-suspicion is both a prompt
        violation and would distort the pressured-candidate machinery."""
        return sorted({pid for pid in _PLAYER_ID_RE.findall(text) if pid != self_id})

    @staticmethod
    def _message_ids(text: str) -> list[str]:
        seen: dict[str, None] = {}
        for message_id in _MESSAGE_ID_RE.findall(text):
            seen.setdefault(message_id, None)
        return list(seen)

    @staticmethod
    def _pending_question_message_id(text: str) -> str | None:
        match = _PENDING_BLOCK_RE.search(text)
        if match is None:
            return None
        ids = _MESSAGE_ID_RE.findall(match.group(1))
        return ids[0] if ids else None
