"""The publication audit must report results that were withheld, and only those.

Both failures here were found by the seed-11 live A/B run, where a single false
`missing_required_result` tripped the release gate's hard-failure check and
stopped the whole qualification after one game.
"""

from __future__ import annotations

import asyncio

from app.ai.coordinator import AICoordinator, _omitted_result_ids
from app.ai.provider.base import Message, SchemaT
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.schemas import (
    DiscussionOutput,
    MorningIntentOutput,
    PublicResultClaim,
    SummaryOutput,
    VoteOutput,
)
from app.engine.game import GameController, PlayerSpec
from app.engine.roles import RoleName
from app.engine.state import MediumRecord
from app.eval.transcript import TranscriptRecorder
from tests.ai.reasoning.solver import boards

AI_SEATS = [f"p{i}" for i in range(1, 17)]


def test_a_published_result_is_not_omitted_just_because_the_day_is_unstated():
    """Seed 11 day 5: required `medium:None:p13` against published
    `medium:4:p13`, for a result the medium had just stated out loud."""
    assert _omitted_result_ids(("medium:None:p13",), ("medium:4:p13",)) == ()
    assert _omitted_result_ids(("medium:4:p13",), ("medium:None:p13",)) == ()


def test_a_genuinely_withheld_result_is_still_reported():
    assert _omitted_result_ids(("seer:2:p9",), ("seer:2:p13",)) == ("seer:2:p9",)
    assert _omitted_result_ids(("seer:2:p9",), ()) == ("seer:2:p9",)
    # A different result type about the same seat is not the same disclosure.
    assert _omitted_result_ids(("medium:1:p9",), ("seer:1:p9",)) == ("medium:1:p9",)


class _VolunteersAResult:
    """A model that mentions a result it has already published, without pinning
    the day -- the turn that produced the seed-11 false positive."""

    def __init__(self) -> None:
        self.discussion_calls = 0

    async def generate_structured(
        self, *, system: str, messages: list[Message], response_schema: type[SchemaT], **kwargs
    ):  # type: ignore[no-untyped-def]
        del system, messages, kwargs
        if response_schema is MorningIntentOutput:
            return MorningIntentOutput()
        if response_schema is SummaryOutput:
            return SummaryOutput()
        if response_schema is VoteOutput:
            return VoteOutput(vote_target="p2", reason="様子見")
        if response_schema is DiscussionOutput:
            self.discussion_calls += 1
            return DiscussionOutput(
                public_message="さきほどの霊媒結果のとおり、p13は人狼ではありませんでした。",
                public_results=[
                    PublicResultClaim(
                        result_type="medium",
                        target_id="p13",
                        is_werewolf=False,
                        referenced_day=None,
                    )
                ],
            )
        return None


def test_restating_an_already_published_result_is_not_audited_as_withholding_it():
    """Seed 11, p14, day 5. The medium publishes its p13 result, then mentions it
    again later in the same day without naming the day. The engine de-duplicates
    the restatement, so that second turn registers no ledger entry -- and the
    audit used to read the model's own `public_results` as the requirement,
    producing a `missing_required_result` for a result already stated out loud.
    """
    state = boards.deal({"p14": RoleName.MEDIUM, "p1": RoleName.WEREWOLF}, day=5)
    boards.execute(state, "p13", day=4)
    state.medium_records.append(
        MediumRecord(medium_id="p14", target_id="p13", day=4, is_werewolf=False)
    )
    controller = _controller_on(state)

    runtime = ReasoningRuntime(state, AI_SEATS, seed=11)
    runtime.refresh(state)
    recorder = TranscriptRecorder()
    provider = _VolunteersAResult()
    coordinator = AICoordinator(
        state, AI_SEATS, provider, seed=11, recorder=recorder, reasoning=runtime, pacing_scale=0.0
    )

    async def play() -> None:
        # Two turns for the same seat on the same day: the duty, then the echo.
        await coordinator._speak(controller, state, "p14", "opening")
        runtime.refresh(state)
        await coordinator._speak(controller, state, "p14", "rebuttal")

    asyncio.run(play())

    assert provider.discussion_calls >= 2, "the scenario never reached the second turn"
    published = {
        (claim.result_type, claim.target_id) for claim in state.public_result_claims
    }
    assert ("medium", "p13") in published, "the result was never published at all"
    omissions = [
        record
        for record in recorder.transcript.result_publication_audits
        if record.omitted_result_ids
    ]
    assert omissions == [], f"a published result was audited as withheld: {omissions}"


def _controller_on(state):  # type: ignore[no-untyped-def]
    """A real GameController driving a hand-dealt board.

    The engine's own claim de-duplication is the whole point: `record_speech_event`
    returns None for a same-day restatement of an identical verdict, so the
    second message registers no ledger entry of its own. A stub controller would
    quietly record one and the regression would not reproduce.
    """
    specs = [PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=(i == 0)) for i in range(17)]
    controller = GameController(session_id="audit-fixture", player_specs=specs, seed=11)
    controller.state = state
    return controller
