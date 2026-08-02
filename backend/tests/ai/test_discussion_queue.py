from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.ai.coordinator import _MAX_CONSECUTIVE_SPEECH_FAILURES, AICoordinator
from app.ai.player_agent import DEFAULT_MAX_RETRIES
from app.ai.provider.base import Message, SchemaT
from app.ai.schemas import DirectedQuestion, DiscussionOutput, MorningIntentOutput
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.sessions.models import DiscussionRoundState
from tests.conftest import make_controller


def test_public_claim_registration_falls_back_to_spoken_message():
    controller = make_controller(seed=4)
    coordinator = AICoordinator(controller.state, ["p1"], MorningPriorityProvider(), seed=1)
    output = DiscussionOutput(
        public_message="霊媒師CO。現時点で処刑結果はありません。",
        public_claim_role=None,
        contains_co_claim=False,
    )

    coordinator.register_public_claim(controller, "p1", output)

    claims = [(claim.player_id, claim.claimed_role) for claim in controller.state.co_declarations]
    assert claims == [("p1", RoleName.MEDIUM)]


def test_named_freemason_partner_is_prompted_and_confirmation_closes_line():
    controller = make_controller(seed=4)
    coordinator = AICoordinator(controller.state, ["p1", "p2"], MorningPriorityProvider(), seed=1)

    coordinator.register_public_claim(
        controller,
        "p1",
        DiscussionOutput(public_message="共有者CO、相方はPlayer2(p2)です。"),
        "m1",
    )

    relation = controller.state.freemason_partner_claims[0]
    assert (relation.claimant_id, relation.partner_id, relation.confirmed) == ("p1", "p2", False)
    assert controller.state.pending_questions["p2"][0].source_message_id == "m1"

    coordinator.register_public_claim(
        controller,
        "p2",
        DiscussionOutput(
            public_message="Player1(p1)の共有者CO、相方は私Player2で間違いありません。"
        ),
        "m2",
    )

    assert relation.confirmed is True


def test_named_ai_partner_speaks_before_remaining_initial_order():
    controller = make_controller(seed=4)
    coordinator = AICoordinator(
        controller.state, ["p1", "p2", "p3"], MorningPriorityProvider(), seed=1
    )
    coordinator.register_public_claim(
        controller,
        "p1",
        DiscussionOutput(public_message="共有者CO、相方はPlayer2(p2)です。"),
        "m1",
    )
    round_state = DiscussionRoundState(day=1, order=["p3", "p2"], max_total=10)

    speaker, stage = coordinator._next_discussion_speaker(controller.state, round_state)

    assert (speaker, stage) == ("p2", "freemason_confirmation")
    assert round_state.order == ["p3"]


def test_question_topic_groups_equivalent_execution_questions():
    assert AICoordinator._question_topic("今日の処刑候補は誰ですか") == "execution_candidate"
    assert AICoordinator._question_topic("一番怪しい灰は誰ですか") == "execution_candidate"
    assert AICoordinator._question_topic("狐候補を挙げてください") == "fox_candidate"


def test_concentrated_pressure_schedules_a_minority_review_before_summary():
    controller = make_controller(seed=4)
    coordinator = AICoordinator(
        controller.state, ["p1", "p2", "p3", "p4"], MorningPriorityProvider(), seed=1
    )
    outputs = []
    for player_id in ("p1", "p2", "p3", "p4"):
        output = DiscussionOutput(public_message="p0を疑います")
        output.reasoning_memo.execution_target = "p0"
        outputs.append((player_id, output))
    round_state = DiscussionRoundState(
        day=1,
        order=[],
        outputs=outputs,
        speech_counts={player_id: 1 for player_id, _output in outputs},
        major_targets_ready=True,
        max_total=20,
    )

    speaker, stage = coordinator._next_discussion_speaker(controller.state, round_state)

    assert speaker in {"p1", "p2", "p3", "p4"}
    assert stage == "minority_review:p0"
    assert round_state.summary_done is False


class ReplyLoopProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_structured(
        self,
        *,
        system: str,
        messages: list[Message],
        response_schema: type[SchemaT],
        max_tokens: int = 800,
        temperature: float = 0.9,
    ) -> SchemaT | None:
        del system, messages, max_tokens, temperature
        if response_schema is MorningIntentOutput:
            return MorningIntentOutput()  # type: ignore[return-value]
        assert response_schema is DiscussionOutput
        self.calls += 1
        lines = {
            1: "まず意見を述べます。",
            2: "Player1さん、理由を説明してください。",
        }
        line = lines.get(self.calls, "Player2さんも説明してください。")
        target = "p1" if self.calls == 2 else "p2"
        return DiscussionOutput(
            public_message=line,
            directed_questions=[DirectedQuestion(target_id=target, question="説明して")],
        )  # type: ignore[return-value]


def test_direct_reply_queue_is_globally_bounded_and_returns_control():
    controller = make_controller(seed=4)
    controller.state.phase = Phase.DISCUSSION
    provider = ReplyLoopProvider()
    coordinator = AICoordinator(
        controller.state,
        ["p1", "p2"],
        provider,
        seed=1,
        max_discussion_followups=1,
    )
    session = SimpleNamespace(controller=controller, discussion_lock=asyncio.Lock())

    asyncio.run(coordinator.run_discussion_round(session))

    assert provider.calls <= 5
    # Questions now reach targets even before their first turn; the target can
    # therefore be queued for a focused follow-up rather than being discarded.
    authors = [message.author_id for message in controller.state.chat_log]
    assert set(authors[:2]) == {"p1", "p2"}
    assert len(authors) >= 3


class MorningPriorityProvider:
    async def generate_structured(
        self,
        *,
        system: str,
        messages: list[Message],
        response_schema: type[SchemaT],
        max_tokens: int = 800,
        temperature: float = 0.9,
    ) -> SchemaT | None:
        del messages, max_tokens, temperature
        if response_schema is MorningIntentOutput:
            timing = "immediate" if "プレイヤー「Player2」" in system else "normal"
            return MorningIntentOutput(timing=timing)  # type: ignore[return-value]
        return DiscussionOutput(public_message="見解を述べます。", ready_to_vote=True)  # type: ignore[return-value]


def test_immediate_morning_intent_speaks_before_normal_seating_order():
    controller = make_controller(seed=4)
    controller.state.phase = Phase.DISCUSSION
    coordinator = AICoordinator(
        controller.state,
        ["p1", "p2"],
        MorningPriorityProvider(),
        seed=1,
    )
    session = SimpleNamespace(controller=controller, discussion_lock=asyncio.Lock())

    asyncio.run(coordinator.run_discussion_round(session))

    assert [message.author_id for message in controller.state.chat_log][:2] == ["p2", "p1"]


def test_major_targets_reply_twice_before_one_consensus_summary():
    controller = make_controller(seed=4)
    coordinator = AICoordinator(controller.state, ["p1", "p2"], MorningPriorityProvider(), seed=1)
    accused = DiscussionOutput(public_message="accuse")
    accused.reasoning_memo.execution_target = "p2"
    ready = DiscussionOutput(public_message="ready", ready_to_vote=True)
    round_state = DiscussionRoundState(
        day=1,
        order=["p1", "p2"],
        cursor=2,
        outputs=[("p1", accused), ("p2", ready)],
        speech_counts={"p1": 1, "p2": 1},
        max_total=6,
    )

    speaker, stage = coordinator._next_discussion_speaker(controller.state, round_state)
    assert (speaker, stage) == ("p2", "rebuttal_or_reassessment")
    round_state.speech_counts["p2"] = 2
    round_state.reply_queue.clear()
    round_state.queued.clear()

    speaker, stage = coordinator._next_discussion_speaker(controller.state, round_state)
    assert stage == "consensus_summary"
    assert speaker in {"p1", "p2"}
    assert round_state.summary_done is True


def _round_state_owing_a_rebuttal() -> DiscussionRoundState:
    """A round where p2 is the pressured execution candidate and still owes the
    table a rebuttal -- the state that used to hand the same speaker back forever."""
    accused = DiscussionOutput(public_message="accuse")
    accused.reasoning_memo.execution_target = "p2"
    return DiscussionRoundState(
        day=1,
        order=["p1", "p2"],
        cursor=2,
        outputs=[("p1", accused)],
        speech_counts={"p1": 1},
        max_total=8,
    )


def test_speaker_selection_terminates_when_a_pressured_target_never_speaks():
    controller = make_controller(seed=4)
    coordinator = AICoordinator(controller.state, ["p1", "p2"], MorningPriorityProvider(), seed=1)
    round_state = _round_state_owing_a_rebuttal()

    # Simulates `_speak` yielding nothing every time: speech_counts never advances.
    # Selection must still run dry instead of returning p2 indefinitely.
    seen = []
    for _ in range(40):
        speaker, _stage = coordinator._next_discussion_speaker(controller.state, round_state)
        if speaker is None:
            break
        seen.append(speaker)
    else:  # pragma: no cover - only reached if selection never terminates
        raise AssertionError("speaker selection never ran out of candidates")

    assert seen.count("p2") <= 3


class SilentProvider:
    """Never returns a discussion output, mimicking total generation failure."""

    def __init__(self) -> None:
        self.discussion_calls = 0

    async def generate_structured(
        self,
        *,
        system: str,
        messages: list[Message],
        response_schema: type[SchemaT],
        max_tokens: int = 800,
        temperature: float = 0.9,
    ) -> SchemaT | None:
        del system, messages, max_tokens, temperature
        if response_schema is MorningIntentOutput:
            return MorningIntentOutput()  # type: ignore[return-value]
        self.discussion_calls += 1
        return None


def test_segment_gives_up_instead_of_retrying_a_silent_speaker_forever():
    controller = make_controller(seed=4)
    controller.state.phase = Phase.DISCUSSION
    provider = SilentProvider()
    coordinator = AICoordinator(controller.state, ["p1", "p2"], provider, seed=1)
    session = SimpleNamespace(controller=controller, discussion_lock=asyncio.Lock())

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(session), timeout=5))

    assert controller.state.chat_log == []
    # Each failed slot costs the full contract's attempts plus the brief fallback's,
    # and the segment gives up after a bounded number of consecutive silent slots.
    # Derived rather than hardcoded so the ceiling tracks the constants, while an
    # unbounded retry loop still trips it.
    attempts_per_slot = (DEFAULT_MAX_RETRIES + 1) * 2
    assert provider.discussion_calls <= _MAX_CONSECUTIVE_SPEECH_FAILURES * attempts_per_slot


class PhaseEndingProvider:
    """Ends the discussion phase from inside the first generated speech, the way a
    human clicking 「投票へ進む」 does mid-segment."""

    def __init__(self, controller: object) -> None:
        self._controller = controller
        self.discussion_calls = 0

    async def generate_structured(
        self,
        *,
        system: str,
        messages: list[Message],
        response_schema: type[SchemaT],
        max_tokens: int = 800,
        temperature: float = 0.9,
    ) -> SchemaT | None:
        del system, messages, max_tokens, temperature
        if response_schema is MorningIntentOutput:
            return MorningIntentOutput()  # type: ignore[return-value]
        self.discussion_calls += 1
        output = DiscussionOutput(public_message="疑わしいのはPlayer2さんです。")
        output.reasoning_memo.execution_target = "p2"
        self._controller.state.phase = Phase.VOTING  # type: ignore[attr-defined]
        return output  # type: ignore[return-value]


def test_segment_stops_when_the_phase_leaves_discussion_mid_round():
    # `_speak` returns None on a non-discussion phase without ever awaiting, so a
    # non-advancing retry here would spin without releasing the event loop.
    controller = make_controller(seed=4)
    controller.state.phase = Phase.DISCUSSION
    provider = PhaseEndingProvider(controller)
    coordinator = AICoordinator(controller.state, ["p1", "p2"], provider, seed=1)
    session = SimpleNamespace(controller=controller, discussion_lock=asyncio.Lock())

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(session), timeout=5))

    assert controller.state.phase == Phase.VOTING
    assert provider.discussion_calls <= 4


def test_the_human_seat_is_never_scheduled_as_a_speaker():
    # AIs routinely question the human or name them as an execution target.
    # Scheduling that seat reached `self._personalities[human_id]` and raised
    # KeyError inside the fire-and-forget discussion task, which killed the round
    # with no error surfacing anywhere.
    controller = make_controller(seed=4)
    coordinator = AICoordinator(controller.state, ["p1", "p2"], MorningPriorityProvider(), seed=1)
    accusing_human = DiscussionOutput(public_message="p0が怪しい")
    accusing_human.reasoning_memo.execution_target = "p0"
    round_state = DiscussionRoundState(
        day=1,
        order=["p1", "p2"],
        cursor=2,
        outputs=[("p1", accusing_human), ("p2", accusing_human)],
        speech_counts={"p1": 1, "p2": 1},
        max_total=8,
    )

    coordinator._round_queue_reply(controller.state, round_state, "p0")
    assert "p0" not in round_state.reply_queue

    for _ in range(20):
        speaker, _stage = coordinator._next_discussion_speaker(controller.state, round_state)
        if speaker is None:
            break
        assert speaker != "p0"
        round_state.speech_counts[speaker] = round_state.speech_counts.get(speaker, 0) + 1
    assert "p0" not in round_state.major_targets
