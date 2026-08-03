"""The turn a seat speaks and the ballot it casts come from the same place.

Before this, two reasoning systems ran side by side: the runtime picked the
vote, the model's free-text memo picked what the discussion argued about, and
nothing reconciled them. An AI could push one name all day and vote for another.
"""

from __future__ import annotations

import asyncio

from app.ai.coordinator import AICoordinator
from app.ai.provider.base import Message, SchemaT
from app.ai.reasoning.dialogue import (
    ConclusionType,
    DiscussionDecision,
    SpeechGoal,
    parse_argument,
)
from app.ai.reasoning.facts import PublicFactLedger
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.schemas import (
    DiscussionOutput,
    MorningIntentOutput,
    NightActionOutput,
    SummaryOutput,
    VoteOutput,
)
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.state import VoteRecord
from tests.ai.reasoning.solver import boards
from tests.conftest import make_controller

AI_IDS = [f"p{i}" for i in range(1, 17)]


class DisobedientProvider:
    """Answers every discussion turn by naming a seat of its own choosing.

    Stands in for the real failure: a model that ignores the analysis and
    asserts its own conclusion. The enforcement has to survive that.
    """

    def __init__(self, rogue_target: str = "p16") -> None:
        self.rogue_target = rogue_target
        self.prompts: list[str] = []

    async def generate_structured(
        self, *, system: str, messages: list[Message], response_schema: type[SchemaT], **kwargs
    ):  # type: ignore[no-untyped-def]
        del system, kwargs
        self.prompts.append("\n".join(message.content for message in messages))
        if response_schema is MorningIntentOutput:
            return MorningIntentOutput()
        if response_schema is DiscussionOutput:
            output = DiscussionOutput(public_message="私の考えを述べます。")
            output.reasoning_memo.execution_target = self.rogue_target
            output.alternative_execution_target = self.rogue_target
            return output
        if response_schema is VoteOutput:
            return VoteOutput(vote_target=self.rogue_target)
        if response_schema is NightActionOutput:
            return NightActionOutput(target=self.rogue_target)
        if response_schema is SummaryOutput:
            return SummaryOutput(summary="")
        return None


def _board(day: int = 2):  # type: ignore[no-untyped-def]
    state = boards.deal(
        {"p1": RoleName.WEREWOLF, "p4": RoleName.SEER, "p6": RoleName.FOX}, day=day
    )
    boards.claim(state, "p4", RoleName.SEER, day=1)
    boards.verdict(state, "p4", "seer", "p9", True, day=1)
    return state


class _FakeController:
    """Only the attribute `_start_discussion_round` reaches for."""

    def __init__(self, state) -> None:  # type: ignore[no-untyped-def]
        self.state = state


def _runtime(state, seed: int = 5) -> ReasoningRuntime:  # type: ignore[no-untyped-def]
    runtime = ReasoningRuntime(state, AI_IDS, seed=seed)
    runtime.refresh(state)
    return runtime


# -- the decision is fixed before the words exist --


def test_a_turn_states_the_conclusion_the_belief_engine_reached():
    state = _board()
    runtime = _runtime(state)

    decision = runtime.discussion_decision(state, "p2")

    assert isinstance(decision, DiscussionDecision)
    assert decision.execution_target == runtime.seats["p2"].belief.state.current_execution_target
    assert decision.supporting_evidence
    assert decision.target_confidence_band


def test_the_brief_states_the_facts_and_forbids_revising_them():
    state = _board()
    runtime = _runtime(state)

    brief = runtime.discussion_decision(state, "p2").render_brief()

    assert "第一処刑候補" in brief
    assert "上の結論・根拠・数値は変更しないでください" in brief
    assert "口調" in brief  # what the model *is* allowed to choose


def test_the_speech_goal_follows_the_situation():
    state = _board()
    boards.divine(state, "p4", "p11", night=1)
    runtime = _runtime(state)

    holder = runtime.discussion_decision(state, "p4")
    questioned = runtime.discussion_decision(state, "p2", pending_question=True)
    pressed = runtime.discussion_decision(state, "p3", under_pressure=True)

    assert holder.speech_goal is SpeechGoal.PUBLISH_RESULT
    assert questioned.speech_goal is SpeechGoal.ANSWER_QUESTION
    assert pressed.speech_goal is SpeechGoal.DEFEND


def test_a_model_cannot_substitute_its_own_execution_target():
    controller = make_controller(seed=4)
    controller.state.phase = Phase.DISCUSSION
    controller.state.day = 2
    provider = DisobedientProvider(rogue_target="p16")
    runtime = ReasoningRuntime(controller.state, ["p1", "p2"], seed=4)
    coordinator = AICoordinator(
        controller.state, ["p1", "p2"], provider, seed=4, reasoning=runtime
    )

    output = asyncio.run(
        coordinator._speak(controller, controller.state, "p1", "initial_view")
    )

    assert output is not None
    decided = runtime.seats["p1"].belief.state.current_execution_target
    # The model said p16; the analysis said otherwise, and the analysis stands.
    assert output.reasoning_memo.execution_target == decided
    assert output.reasoning_memo.execution_target != "p16"
    # And the brief actually reached the prompt.
    assert any("第一処刑候補" in prompt for prompt in provider.prompts)


def test_the_vote_is_checked_against_what_was_said_out_loud():
    controller = make_controller(seed=4)
    controller.state.phase = Phase.DISCUSSION
    controller.state.day = 2
    runtime = ReasoningRuntime(controller.state, ["p1", "p2"], seed=4)
    coordinator = AICoordinator(
        controller.state, ["p1", "p2"], DisobedientProvider(), seed=4, reasoning=runtime
    )

    asyncio.run(coordinator._speak(controller, controller.state, "p1", "initial_view"))
    stated = runtime.stated_target("p1")
    controller.state.phase = Phase.VOTING
    asyncio.run(coordinator._cast_vote(controller, controller.state, "p1"))

    # Same source for both, so the two agree and nothing is flagged.
    assert stated is not None
    assert controller.state.pending_votes["p1"] == stated
    assert coordinator.validation.vote_plan_mismatches == []


def test_a_changed_candidate_is_recorded_with_its_reason():
    controller = make_controller(seed=4)
    controller.state.phase = Phase.VOTING
    controller.state.day = 2
    recorder_seen: list[str] = []

    class _Recorder:
        def set_roster(self, **kwargs):  # type: ignore[no-untyped-def]
            del kwargs

        def record(self, utterance):  # type: ignore[no-untyped-def]
            recorder_seen.append(utterance.kind)

    runtime = ReasoningRuntime(controller.state, ["p1", "p2"], seed=4)
    coordinator = AICoordinator(
        controller.state,
        ["p1", "p2"],
        DisobedientProvider(),
        seed=4,
        reasoning=runtime,
        recorder=_Recorder(),  # type: ignore[arg-type]
    )
    # Claim publicly to have been chasing someone the ballot will not name.
    runtime.refresh(controller.state)
    ballot, _ = runtime.vote_decision(
        "p1", [pid for pid in controller.state.alive_ids() if pid != "p1"]
    )
    stated = next(pid for pid in controller.state.alive_ids() if pid not in ("p1", ballot))
    runtime.record_stated_target("p1", stated)

    asyncio.run(coordinator._cast_vote(controller, controller.state, "p1"))

    assert coordinator.validation.vote_plan_mismatches
    assert "vote_change" in recorder_seen


# -- the speaking order --


def test_only_the_seats_with_something_to_say_open_the_day():
    controller = make_controller(seed=6)
    controller.state.phase = Phase.DISCUSSION
    controller.state.day = 1
    runtime = ReasoningRuntime(controller.state, AI_IDS, seed=6)
    coordinator = AICoordinator(
        controller.state, AI_IDS, DisobedientProvider(), seed=6, reasoning=runtime
    )

    round_state = asyncio.run(coordinator._start_discussion_round(controller.state))

    # Not all sixteen: a morning of sixteen opening statements is a wall of text.
    assert 0 < len(round_state.order) < len(AI_IDS)
    assert len(set(round_state.order)) == len(round_state.order)


def test_a_seat_holding_a_result_is_never_left_out_of_the_opening():
    state = _board()
    state.phase = Phase.DISCUSSION
    # p4 is the seer; the night-1 black is already out, but the night-2 look is not.
    boards.divine(state, "p4", "p11", night=2)
    controller = _FakeController(state)
    runtime = ReasoningRuntime(state, AI_IDS, seed=6)
    coordinator = AICoordinator(
        state, AI_IDS, DisobedientProvider(), seed=6, reasoning=runtime
    )
    del controller

    round_state = asyncio.run(coordinator._start_discussion_round(state))

    # A duty speaker is never dropped by the value ranking, and goes first.
    assert "p4" in round_state.order
    assert round_state.order[0] == "p4"


# -- human arguments, parsed once --


def test_a_strategic_claim_is_structured_rather_than_ignored():
    state = _board()
    ledger = PublicFactLedger(state)
    daiki = ledger.name_of("p9")

    argument = parse_argument(
        f"{daiki}を処刑する必要はありません。", ledger, "p0", "m1"
    )

    assert argument is not None
    assert argument.conclusion_type is ConclusionType.DEFENCE
    assert argument.conclusion_target_id == "p9"
    assert argument.strategic_claims[0].claim_type == "spare_target"


def test_a_bandwagon_accusation_is_structured():
    state = _board()
    ledger = PublicFactLedger(state)
    yui = ledger.name_of("p11")

    argument = parse_argument(f"{yui}は処刑に便乗しています。", ledger, "p0", "m2")

    assert argument is not None
    assert argument.conclusion_type is ConclusionType.ACCUSATION
    assert argument.conclusion_target_id == "p11"


def test_a_closed_world_challenge_is_structured():
    state = _board()
    ledger = PublicFactLedger(state)

    argument = parse_argument(
        "その内訳では真の占い師はどこにいるのですか。", ledger, "p0", "m3"
    )

    assert argument is not None
    assert argument.conclusion_type is ConclusionType.CLOSED_WORLD_CHALLENGE


def test_a_fact_correction_still_takes_priority_over_rhetoric():
    state = _board()
    state.vote_records.append(VoteRecord(voter_id="p0", target_id="p3", day=2, round=1))
    ledger = PublicFactLedger(state)
    daiki = ledger.name_of("p12")

    argument = parse_argument(
        f"2日目、私は{daiki}には投票していません。", ledger, "p0", "m4"
    )

    assert argument is not None
    assert argument.conclusion_type is ConclusionType.FACT_CORRECTION
    assert argument.factual_claims[0].denied == "p12"


def test_ordinary_conversation_is_not_forced_into_a_claim():
    state = _board()
    ledger = PublicFactLedger(state)

    assert parse_argument("よろしくお願いします。", ledger, "p0", "m5") is None
    assert parse_argument("難しい盤面ですね。", ledger, "p0", "m6") is None


def test_one_human_message_is_parsed_once_and_weighed_per_seat():
    state = _board()
    runtime = _runtime(state)
    yui = state.players["p11"].name
    before = {
        pid: seat.belief.state.public_suspicion_scores.get("p11", 0.0)
        for pid, seat in runtime.seats.items()
    }

    runtime.apply_human_message(state, "p0", f"{yui}は処刑に便乗しています。", "m7")

    assert len(runtime.argument_log) == 1
    after = {
        pid: seat.belief.state.public_suspicion_scores.get("p11", 0.0)
        for pid, seat in runtime.seats.items()
    }
    moved = [pid for pid in after if after[pid] > before[pid]]
    assert len(moved) > 1
    # Weighed per seat, so the shifts are not all identical.
    assert len({round(after[pid] - before[pid], 4) for pid in moved}) > 1


def test_an_argument_never_moves_the_speakers_own_belief():
    state = _board()
    runtime = ReasoningRuntime(state, AI_IDS, seed=5)
    runtime.refresh(state)
    yui = state.players["p11"].name

    runtime.apply_human_message(state, "p2", f"{yui}は便乗しています。", "m8")

    assert "p2" not in {
        record.subject_id
        for record in runtime.seats["p2"].belief.active_evidence()
        if record.category == "accusation"
    }
    assert not any(
        record.evidence_id.startswith("arg:p2")
        for record in runtime.seats["p2"].belief.evidence
    )


# -- reassessment after a correction --


def test_seats_whose_reasons_were_withdrawn_are_queued_to_speak():
    from app.ai.reasoning.belief import EvidenceRecord, vote_fact_id

    controller = make_controller(seed=4)
    controller.state.phase = Phase.DISCUSSION
    controller.state.day = 2
    controller.state.vote_records.append(
        VoteRecord(voter_id="p0", target_id="p3", day=2, round=1)
    )
    runtime = ReasoningRuntime(controller.state, ["p1", "p2"], seed=4)
    coordinator = AICoordinator(
        controller.state, ["p1", "p2"], DisobedientProvider(), seed=4, reasoning=runtime
    )
    runtime.refresh(controller.state)
    runtime.seats["p1"].belief.add_evidence(
        EvidenceRecord(
            evidence_id="recalled:p0",
            subject_id="p0",
            category="misremembered_vote",
            source_event_ids=(vote_fact_id("p0", 2, 1, "p12"),),
            weight=1.5,
            explanation="p0はp12へ投票していた。",
        )
    )
    runtime.seats["p1"].belief.recompute(PublicFactLedger(controller.state))
    name = controller.state.players["p12"].name

    moved = coordinator.note_human_message(
        controller.state, "p0", f"2日目、私は{name}には投票していません。", "m9"
    )

    assert moved == ["p1"]
    # Queued, not discarded: a correction that lands in silence is a correction
    # the table never sees acted on.
    assert runtime.take_reassessment_speakers() == ["p1"]
