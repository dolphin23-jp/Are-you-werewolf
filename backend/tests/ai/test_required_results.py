"""Publishing a result is decided in code, not left to the model remembering.

`SpeechGoal.PUBLISH_RESULT` used to be a hint. The model could still return
`public_results=[]`, and the seer's result simply did not happen. Now the
decision carries the results -- and the CO they need behind them -- and the
coordinator publishes exactly those.

The CO half was a real gap: a forced result from a seer who had not claimed
landed in the ledger as a verdict from nobody in particular.
"""

from __future__ import annotations

import asyncio

from app.ai.coordinator import AICoordinator
from app.ai.provider.base import Message, SchemaT
from app.ai.reasoning.dialogue import SpeechGoal
from app.ai.reasoning.facts import PublicFactLedger
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.schemas import DiscussionOutput, PublicResultClaim
from app.engine.game import GameController
from app.engine.roles import RoleName
from tests.conftest import make_controller

AI_IDS = [f"p{i}" for i in range(1, 17)]


class ScriptedProvider:
    """Returns one fixed discussion output. Defaults to saying nothing."""

    def __init__(self, output: DiscussionOutput | None = None) -> None:
        self.output = output

    async def generate_structured(
        self, *, system: str, messages: list[Message], response_schema: type[SchemaT], **kwargs
    ):  # type: ignore[no-untyped-def]
        del system, messages, kwargs
        if response_schema is DiscussionOutput:
            return self.output or DiscussionOutput(public_message="様子を見ます。")
        return response_schema()  # type: ignore[call-arg]


def _day_one_with_a_seer_holding_a_result() -> tuple[GameController, str, str]:
    """A real game at day 1: an AI seer alive with an unpublished night-0 look."""
    for seed in range(1, 80):
        controller = make_controller(seed=seed)
        state = controller.state
        seer = next(p.player_id for p in state.players.values() if p.role is RoleName.SEER)
        if seer == "p0":
            continue
        controller.start_game()
        target = next(
            pid for pid in state.alive_ids() if pid not in (seer, state.first_victim_id)
        )
        controller.submit_night_action(seer, "divine", target)
        controller.resolve_night()
        if state.players[seer].alive:
            controller.start_discussion()
            return controller, seer, target
    raise AssertionError("no seed produced a live AI seer holding a result")


def _speak(controller: GameController, seer: str, output: DiscussionOutput | None = None):  # type: ignore[no-untyped-def]
    runtime = ReasoningRuntime(controller.state, AI_IDS, seed=1)
    coordinator = AICoordinator(
        controller.state, AI_IDS, ScriptedProvider(output), seed=1, reasoning=runtime
    )
    asyncio.run(coordinator._speak(controller, controller.state, seer, "initial_view"))
    return runtime


def _shown(controller: GameController, seer: str) -> str:
    return next(m.content for m in reversed(controller.state.chat_log) if m.author_id == seer)


def test_the_decision_carries_the_result_and_the_claim():
    controller, seer, target = _day_one_with_a_seer_holding_a_result()
    runtime = ReasoningRuntime(controller.state, AI_IDS, seed=1)

    decision = runtime.discussion_decision(controller.state, seer)

    assert decision.speech_goal is SpeechGoal.PUBLISH_RESULT
    required = [
        (r.result_type, r.target_id, r.referenced_day)
        for r in decision.required_public_results
    ]
    assert required == [("seer", target, 0)]
    assert decision.required_claim_role is RoleName.SEER


def test_a_model_that_forgets_the_field_still_publishes_the_result():
    controller, seer, target = _day_one_with_a_seer_holding_a_result()

    _speak(controller, seer)  # ScriptedProvider returns public_results=[]

    result = PublicFactLedger(controller.state).find_result(seer, "seer", target)
    assert result is not None
    assert result.referenced_day == 0
    assert controller.state.players[target].name in _shown(controller, seer)


def test_the_result_is_published_with_a_claim_behind_it():
    controller, seer, _ = _day_one_with_a_seer_holding_a_result()

    _speak(controller, seer)

    assert PublicFactLedger(controller.state).claimed_role_of(seer) is RoleName.SEER
    assert "占いCO" in _shown(controller, seer)


def test_a_fabricated_extra_result_is_not_published():
    """The decision replaces the model's list; it does not merge into it."""
    controller, seer, target = _day_one_with_a_seer_holding_a_result()
    invented = next(
        pid for pid in controller.state.alive_ids() if pid not in (seer, target)
    )
    output = DiscussionOutput(
        public_message="結果を伝えます。",
        public_results=[
            PublicResultClaim(result_type="seer", target_id=invented, is_werewolf=True)
        ],
    )

    _speak(controller, seer, output)

    ledger = PublicFactLedger(controller.state)
    assert ledger.find_result(seer, "seer", target) is not None
    assert ledger.find_result(seer, "seer", invented) is None


def test_an_already_published_result_is_not_required_again():
    """Including one published through free text with no stated night, which a
    night-keyed check would treat as unpublished and repeat every turn."""
    controller, seer, target = _day_one_with_a_seer_holding_a_result()
    controller.co(seer, "seer")
    controller.public_result(seer, "seer", target, False)  # referenced_day=None
    runtime = ReasoningRuntime(controller.state, AI_IDS, seed=1)

    decision = runtime.discussion_decision(controller.state, seer)

    assert runtime.unpublished_results(controller.state, seer) == ()
    assert decision.required_public_results == ()
    assert decision.required_claim_role is None


def test_a_seat_is_never_told_to_publish_someone_elses_result():
    controller, seer, _ = _day_one_with_a_seer_holding_a_result()
    runtime = ReasoningRuntime(controller.state, AI_IDS, seed=1)
    others = [pid for pid in AI_IDS if pid != seer and controller.state.players[pid].alive]

    for other in others:
        assert runtime.unpublished_results(controller.state, other) == ()
        assert runtime.discussion_decision(controller.state, other).required_public_results == ()
