"""What a discussion turn does when generation fails, stalls, or arrives too late.

Three failures, one rule: the table never loses what code guarantees, and never
hears what it should not.

* A seer whose request fails still owes the table its result. The result, the
  CO behind it and their canonical sentences were never the model's to write,
  so a failed generation used to cost a whole day's information for nothing.
* A turn that finishes after the vote opened, the day turned or the speaker
  died is dropped. The engine's `chat` checks only liveness, so a slow
  discussion line used to land in the middle of the vote.
* A segment where nobody can be made to speak hands the floor back to the
  human with a running countdown, instead of leaving the table silent with
  nothing scheduled to ask for the next segment.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from app.ai.coordinator import AICoordinator
from app.ai.provider.base import Message, SchemaT
from app.ai.reasoning.facts import PublicFactLedger
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.schemas import BriefDiscussionOutput, MorningIntentOutput
from app.engine.game import GameController
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.eval.transcript import TranscriptRecorder
from tests.conftest import make_controller

AI_IDS = [f"p{i}" for i in range(1, 17)]


class _Metrics:
    """The two collector calls the coordinator makes, recorded."""

    def __init__(self) -> None:
        self.events: Counter[str] = Counter()
        self.discussion_results: list[bool] = []

    def record_event(self, name: str) -> None:
        self.events[name] += 1

    def record_discussion_result(self, *, skipped: bool) -> None:
        self.discussion_results.append(skipped)


class _MetricsWithoutEvents:
    """A collector from before `record_event` existed."""

    def record_discussion_result(self, *, skipped: bool) -> None:
        del skipped


def _is_discussion(schema: type) -> bool:
    # Matched by name so the providers serve the full contract, the chat
    # contract and the brief fallback alike, whichever the agent asks for.
    return "Discussion" in schema.__name__


class FailingProvider:
    """Every discussion request fails, whichever contract it asks for."""

    def __init__(self) -> None:
        self._metrics: object = _Metrics()
        self.discussion_calls = 0

    async def generate_structured(
        self, *, system: str, messages: list[Message], response_schema: type[SchemaT], **kwargs
    ):  # type: ignore[no-untyped-def]
        del system, messages, kwargs
        if _is_discussion(response_schema):
            self.discussion_calls += 1
            return None
        if response_schema is MorningIntentOutput:
            return MorningIntentOutput()
        return None


class SlowProvider:
    """Holds every discussion request until released, so the table can move on."""

    def __init__(self, message: str = "少し考えましたが、まだ決め手がありません。") -> None:
        self._metrics = _Metrics()
        self.message = message
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate_structured(
        self, *, system: str, messages: list[Message], response_schema: type[SchemaT], **kwargs
    ):  # type: ignore[no-untyped-def]
        del system, messages, kwargs
        if _is_discussion(response_schema):
            self.started.set()
            await self.release.wait()
            return response_schema.model_validate({"public_message": self.message})
        if response_schema is MorningIntentOutput:
            return MorningIntentOutput()
        return None


def _seer_holding_a_result() -> tuple[GameController, str, str]:
    """A real day 1: an AI seer alive with an unpublished night-0 look, no CO yet."""
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


def _public_by(controller: GameController, player_id: str) -> list[str]:
    return [
        m.content
        for m in controller.state.chat_log
        if m.author_id == player_id and m.channel.value == "public"
    ]


# -- obligations survive a failed generation --


@pytest.mark.parametrize("model_decides", [False, True], ids=["v2", "v3"])
def test_a_failed_generation_still_publishes_the_seers_result(model_decides: bool) -> None:
    controller, seer, target = _seer_holding_a_result()
    state = controller.state
    provider = FailingProvider()
    recorder = TranscriptRecorder()
    coordinator = AICoordinator(
        state,
        AI_IDS,
        provider,
        seed=1,
        reasoning=ReasoningRuntime(state, AI_IDS, seed=1),
        model_decides=model_decides,
        recorder=recorder,
    )

    output = asyncio.run(coordinator._speak(controller, state, seer, "immediate"))

    assert provider.discussion_calls > 0  # generation really was attempted, and failed
    assert output is not None
    ledger = PublicFactLedger(state)
    result = ledger.find_result(seer, "seer", target)
    assert result is not None and result.referenced_day == 0
    assert ledger.claimed_role_of(seer) is RoleName.SEER
    (shown,) = _public_by(controller, seer)
    assert state.players[target].name in shown
    assert recorder.transcript.result_publication_audits[-1].omitted_result_ids == ()
    assert recorder.transcript.utterances[-1].used_fallback is True
    assert provider._metrics.events["obligation_posted_by_code"] == 1  # type: ignore[attr-defined]


def test_a_failed_turn_owing_nothing_still_posts_nothing() -> None:
    controller, seer, _target = _seer_holding_a_result()
    state = controller.state
    provider = FailingProvider()
    coordinator = AICoordinator(
        state, AI_IDS, provider, seed=1, reasoning=ReasoningRuntime(state, AI_IDS, seed=1)
    )
    quiet = next(
        pid for pid in AI_IDS if state.players[pid].alive and pid != seer
        and state.players[pid].role is RoleName.VILLAGER
    )
    before = len(state.chat_log)

    assert asyncio.run(coordinator._speak(controller, state, quiet, "initial_view")) is None

    assert len(state.chat_log) == before
    assert provider._metrics.events["obligation_posted_by_code"] == 0  # type: ignore[attr-defined]


def test_a_turn_written_in_code_keeps_the_seats_own_notes_and_statement() -> None:
    """The code-written turn says nothing about candidates, so it must not erase
    what the model last thought or what the seat last told the table."""
    controller, seer, target = _seer_holding_a_result()
    state = controller.state
    runtime = ReasoningRuntime(state, AI_IDS, seed=1)
    coordinator = AICoordinator(
        state, AI_IDS, FailingProvider(), seed=1, reasoning=runtime, model_decides=True
    )
    named = next(pid for pid in state.alive_ids() if pid not in (seer, target, "p0"))
    notes = {"execution_target": named, "overall_thought": "昨日の票の流れが気になる"}
    coordinator._context.set_reasoning_memo(seer, dict(notes))
    runtime.record_stated_target(seer, named, state.day)

    assert asyncio.run(coordinator._speak(controller, state, seer, "immediate")) is not None

    assert coordinator._context.get_reasoning_memo(seer) == notes
    assert runtime.stated_target(seer, state.day) == named


class BriefOnlyProvider:
    """The full contract fails (a truncated JSON, say); the brief one answers."""

    def __init__(self) -> None:
        self._metrics = _Metrics()

    async def generate_structured(
        self, *, system: str, messages: list[Message], response_schema: type[SchemaT], **kwargs
    ):  # type: ignore[no-untyped-def]
        del system, messages, kwargs
        if response_schema is BriefDiscussionOutput:
            return BriefDiscussionOutput(public_message="昨日の票の流れを見直したい。")
        if response_schema is MorningIntentOutput:
            return MorningIntentOutput()
        return None


def test_a_brief_fallback_turn_keeps_the_seats_own_notes_and_statement() -> None:
    """The brief contract asks for the sentence alone, so its turn carries no
    notes. Stored as the seat's memo it wiped the previous plan, and recorded as
    the stated target it erased the candidate the seat argued earlier today --
    the same harm a code-written turn was already guarded against."""
    controller, seer, target = _seer_holding_a_result()
    state = controller.state
    runtime = ReasoningRuntime(state, AI_IDS, seed=1)
    provider = BriefOnlyProvider()
    coordinator = AICoordinator(
        state, AI_IDS, provider, seed=1, reasoning=runtime, model_decides=True
    )
    quiet = next(
        pid for pid in AI_IDS if state.players[pid].alive and pid != seer
        and state.players[pid].role is RoleName.VILLAGER
    )
    named = next(pid for pid in state.alive_ids() if pid not in (quiet, seer, target, "p0"))
    notes = {"execution_target": named, "overall_thought": "昨日の票の流れが気になる"}
    coordinator._context.set_reasoning_memo(quiet, dict(notes))
    runtime.record_stated_target(quiet, named, state.day)

    output = asyncio.run(coordinator._speak(controller, state, quiet, "initial_view"))

    assert output is not None
    assert provider._metrics.events["brief_fallback"] == 1  # the brief path really ran
    assert _public_by(controller, quiet) == ["昨日の票の流れを見直したい。"]
    assert coordinator._context.get_reasoning_memo(quiet) == notes
    assert runtime.stated_target(quiet, state.day) == named


# -- a turn that arrives after the table moved on --


def _end_discussion(controller: GameController, _speaker: str) -> None:
    controller.end_discussion()


def _speaker_dies(controller: GameController, speaker: str) -> None:
    controller.state.players[speaker].alive = False


def _day_turns(controller: GameController, _speaker: str) -> None:
    controller.state.day += 1


@pytest.mark.parametrize(
    "move_on",
    [_end_discussion, _speaker_dies, _day_turns],
    ids=["vote-opened", "speaker-died", "day-turned"],
)
def test_a_turn_finished_after_the_table_moved_on_is_dropped(
    move_on: Callable[[GameController, str], None],
) -> None:
    controller, seer, target = _seer_holding_a_result()
    state = controller.state
    provider = SlowProvider()
    runtime = ReasoningRuntime(state, AI_IDS, seed=1)
    recorder = TranscriptRecorder()
    coordinator = AICoordinator(
        state,
        AI_IDS,
        provider,
        seed=1,
        reasoning=runtime,
        model_decides=True,
        recorder=recorder,
    )

    async def scenario() -> tuple[object, int]:
        turn = asyncio.create_task(coordinator._speak(controller, state, seer, "immediate"))
        await provider.started.wait()
        move_on(controller, seer)
        posted = len(state.chat_log)
        provider.release.set()
        return await turn, posted

    result, posted = asyncio.run(scenario())

    assert result is None
    assert len(state.chat_log) == posted
    # Nothing recorded as said: no result, no claim, no stated target, no utterance.
    ledger = PublicFactLedger(state)
    assert ledger.find_result(seer, "seer", target) is None
    assert ledger.claimed_role_of(seer) is None
    assert runtime.stated_target(seer) is None
    assert not [u for u in recorder.transcript.utterances if u.kind == "discussion"]
    assert provider._metrics.events["speech_discarded_after_phase_end"] == 1


def test_the_round_posts_nothing_once_the_vote_opens_mid_generation() -> None:
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.DISCUSSION
    state.day = 2
    provider = SlowProvider()
    coordinator = AICoordinator(
        state,
        AI_IDS,
        provider,
        seed=4,
        reasoning=ReasoningRuntime(state, AI_IDS, seed=4),
        model_decides=True,
    )
    session = SimpleNamespace(controller=controller, discussion_lock=asyncio.Lock())

    async def scenario() -> int:
        round_task = asyncio.create_task(coordinator.run_discussion_round(session))
        await provider.started.wait()
        controller.end_discussion()
        posted = len(state.chat_log)
        provider.release.set()
        await asyncio.wait_for(round_task, timeout=5)
        return posted

    posted = asyncio.run(scenario())

    assert state.phase is Phase.VOTING
    assert len(state.chat_log) == posted


# -- a segment nobody can speak in --


def _stalling_table(
    metrics: object,
) -> tuple[GameController, AICoordinator, FailingProvider, SimpleNamespace]:
    controller = make_controller(seed=4)
    controller.state.phase = Phase.DISCUSSION
    provider = FailingProvider()
    provider._metrics = metrics
    coordinator = AICoordinator(controller.state, ["p1", "p2", "p3", "p4"], provider, seed=1)
    session = SimpleNamespace(
        controller=controller, discussion_lock=asyncio.Lock(), human_id="p0"
    )
    return controller, coordinator, provider, session


def test_a_stalled_segment_hands_the_floor_back_to_the_human() -> None:
    metrics = _Metrics()
    controller, coordinator, provider, session = _stalling_table(metrics)
    asyncio.run(coordinator.advance_discussion(session))
    # Nobody opens the morning, so the human is given the floor first.
    assert session.discussion_round.awaiting_human is True
    coordinator.resume_after_human(session)

    asyncio.run(coordinator.advance_discussion(session))

    round_state = session.discussion_round
    assert controller.state.chat_log == []
    assert provider.discussion_calls > 0
    assert round_state.complete is False
    # Waiting again, with a fresh countdown, so the client's auto-pass retries.
    assert round_state.awaiting_human is True
    assert round_state.awaiting_since is not None
    assert metrics.events["discussion_stalled"] == 1


def test_a_stall_without_an_event_counter_is_still_handled() -> None:
    controller, coordinator, _provider, session = _stalling_table(_MetricsWithoutEvents())
    asyncio.run(coordinator.advance_discussion(session))
    coordinator.resume_after_human(session)

    asyncio.run(coordinator.advance_discussion(session))

    assert session.discussion_round.awaiting_human is True


def test_an_offline_stall_never_waits_for_a_human() -> None:
    metrics = _Metrics()
    controller, coordinator, _provider, session = _stalling_table(metrics)

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(session), timeout=5))

    assert session.discussion_round.awaiting_human is False
    assert metrics.events["discussion_stalled"] >= 1
