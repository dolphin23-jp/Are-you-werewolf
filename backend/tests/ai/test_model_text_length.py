"""A turn's length is measured on what the model wrote, not on what code added.

In v3 the recorded limit is the stage cap the prompt named (30/60/120). Code
puts the canonical CO and result sentences in front of a line after the model
has written it, so measuring the posted text counted those ~26 characters
against the model: a seer's own morning line under its 120 cap was reported
over it (the review's len_repro.py). The coordinator now records the length of
the model's own part next to the cap, and a turn code wrote whole records 0.
"""

from __future__ import annotations

import asyncio

from app.ai.coordinator import AICoordinator
from app.ai.personalities import chat_stage_max_chars
from app.ai.provider.base import Message, SchemaT
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.schemas import (
    MorningIntentOutput,
    SummaryOutput,
    VoteOutput,
)
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.eval.analyzers import analyze
from app.eval.transcript import TranscriptRecorder
from tests.ai.reasoning.solver import boards
from tests.conftest import make_controller

AI_IDS = [f"p{i}" for i in range(1, 17)]

# The model's own morning line: no role or verdict words, so code must add both.
MODEL_LINE = (
    "おはよう。昨夜見た先は人間側だったので、今日はグレーから吊りたい。"
    "発言の少ない人から順に話を聞きたいので、まずは昨日あまり話していなかった人の意見を待つ。"
    "反対があれば今のうちに言ってほしい。"
)


class _ScriptedLine:
    """Writes `message` for every discussion turn, or fails when it is None."""

    def __init__(self, message: str | None) -> None:
        self.message = message

    async def generate_structured(
        self, *, system: str, messages: list[Message], response_schema: type[SchemaT], **kwargs
    ):  # type: ignore[no-untyped-def]
        del system, messages, kwargs
        if response_schema is MorningIntentOutput:
            return MorningIntentOutput()
        # Matched by name, so the full, chat and brief contracts all fail alike.
        if "Discussion" in response_schema.__name__:
            if self.message is None:
                return None
            return response_schema.model_validate({"public_message": self.message})
        if response_schema is VoteOutput:
            return VoteOutput(vote_target="p2", reason="様子見")
        return SummaryOutput(summary="要約")


def _seer_morning(message: str | None):  # type: ignore[no-untyped-def]
    controller = make_controller(seed=3)
    state = controller.state
    state.phase = Phase.DISCUSSION
    state.day = 1
    seer = next(p.player_id for p in state.players_by_role(RoleName.SEER))
    target = next(
        pid for pid in AI_IDS if pid != seer and state.players[pid].role is RoleName.VILLAGER
    )
    boards.divine(state, seer, target, night=0)
    runtime = ReasoningRuntime(state, AI_IDS, seed=3)
    runtime.refresh(state)
    recorder = TranscriptRecorder()
    coordinator = AICoordinator(
        state,
        AI_IDS,
        _ScriptedLine(message),
        seed=3,
        reasoning=runtime,
        model_decides=True,
        recorder=recorder,
        pacing_scale=0.0,
    )
    asyncio.run(coordinator._speak(controller, state, seer, "immediate"))
    return recorder.transcript


def test_the_code_written_co_and_result_do_not_count_against_the_stage_cap():
    cap = chat_stage_max_chars("immediate")
    assert len(MODEL_LINE) <= cap

    transcript = _seer_morning(MODEL_LINE)

    turn = transcript.utterances[-1]
    assert turn.text.endswith(MODEL_LINE)
    assert turn.text.startswith("占いCO。占い結果、"), "code never added the fact sentences"
    assert len(turn.text) > cap
    assert turn.effective_length_limit == cap
    assert turn.model_text_chars == len(MODEL_LINE)
    stats = analyze(transcript).stats["speech"]
    assert stats["over_length_limit"] == 0
    assert stats["mean_length"] == len(MODEL_LINE)


def test_a_turn_code_wrote_whole_records_no_model_part():
    """A failed generation still posts the seer's result, in code's words."""
    transcript = _seer_morning(None)

    turn = transcript.utterances[-1]
    assert turn.used_fallback is True
    assert turn.text.startswith("占いCO。占い結果、")
    assert turn.model_text_chars == 0
    assert analyze(transcript).stats["speech"]["over_length_limit"] == 0
