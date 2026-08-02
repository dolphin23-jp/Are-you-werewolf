"""How speech becomes events: declarations bind, free text is only read."""

from __future__ import annotations

from app.ai.reasoning import PublicFactLedger
from app.ai.reasoning.claims import (
    build_claim_drafts,
    ensure_fact_sentences,
    register_claim_drafts,
    render_result_sentence,
)
from app.ai.schemas import DiscussionOutput, PublicResultClaim
from app.engine.roles import RoleName
from app.engine.speech_events import (
    SpeechEventType,
    current_role_claim,
    result_versions,
)
from tests.ai.reasoning.fixtures import declare_co, execute, make_state
from tests.conftest import make_controller


def _drafts(output: DiscussionOutput, state, speaker_id: str):  # type: ignore[no-untyped-def]
    return build_claim_drafts(output, PublicFactLedger(state), speaker_id=speaker_id)


# -- structured output is the declaration --


def test_a_structured_co_registers_without_the_prose_saying_it():
    state = make_state()
    output = DiscussionOutput(public_message="今日は静観します。", public_claim_role="seer")

    drafts = _drafts(output, state, "p3")

    assert [(d.event_type, d.role) for d in drafts] == [
        (SpeechEventType.ROLE_CLAIM, RoleName.SEER)
    ]
    assert drafts[0].confidence == 1.0


def test_a_structured_result_registers_without_the_prose_naming_the_target():
    state = make_state()
    output = DiscussionOutput(
        public_message="結果を報告します。",
        public_claim_role="seer",
        public_results=[
            PublicResultClaim(result_type="seer", target_id="p5", is_werewolf=True)
        ],
    )

    drafts = _drafts(output, state, "p3")

    assert [(d.event_type, d.target_id, d.result_is_werewolf) for d in drafts] == [
        (SpeechEventType.ROLE_CLAIM, None, None),
        (SpeechEventType.ABILITY_RESULT, "p5", True),
    ]


def test_free_text_is_only_consulted_where_the_structured_output_is_silent():
    state = make_state()
    declare_co(state, "p3", RoleName.SEER)
    # The prose names p7; the declared field names p5. The declaration wins.
    output = DiscussionOutput(
        public_message="占い結果、Player7(p7)は人狼でした。",
        public_results=[
            PublicResultClaim(result_type="seer", target_id="p5", is_werewolf=True)
        ],
    )

    drafts = _drafts(output, state, "p3")

    assert [d.target_id for d in drafts] == ["p5"]


def test_a_spoken_co_still_registers_when_the_model_omits_the_field():
    state = make_state()
    output = DiscussionOutput(public_message="霊媒師CO。現時点で処刑結果はありません。")

    drafts = _drafts(output, state, "p3")

    assert [(d.event_type, d.role) for d in drafts] == [
        (SpeechEventType.ROLE_CLAIM, RoleName.MEDIUM)
    ]


def test_two_spoken_results_in_one_message_both_become_events():
    state = make_state()
    output = DiscussionOutput(
        public_message="占いCO。Player5(p5)は黒、Player6(p6)は白でした。"
    )

    drafts = _drafts(output, state, "p3")

    assert [(d.target_id, d.result_is_werewolf) for d in drafts[1:]] == [
        ("p5", True),
        ("p6", False),
    ]


# -- what free text must not turn into a claim --


def test_waiting_for_someone_elses_co_is_not_a_claim():
    state = make_state()
    output = DiscussionOutput(public_message="占い師のCOを待ってから動きましょう。")

    assert _drafts(output, state, "p3") == []


def test_denying_a_role_is_not_a_claim():
    state = make_state()
    output = DiscussionOutput(public_message="私は占い師ではないので静観します。")

    assert _drafts(output, state, "p3") == []


def test_quoting_someone_elses_co_does_not_claim_it_for_the_speaker():
    state = make_state()
    named = DiscussionOutput(public_message="Player5(p5)は占い師ですと言っていました。")
    unnamed = DiscussionOutput(public_message="さっき「占い師CO」と言っていましたよね。")

    # Naming the subject first makes it a report, so nothing claim-shaped remains.
    assert _drafts(named, state, "p3") == []
    # Without a name the sentence still looks like a claim, so it is recorded --
    # but the reporting form keeps it below the promotion threshold.
    quoted = _drafts(unnamed, state, "p3")
    assert [(d.role, d.confidence) for d in quoted] == [(RoleName.SEER, 0.4)]


def test_a_hedged_self_claim_is_recorded_but_not_promoted():
    controller = make_controller(seed=4)
    controller.state.day = 1
    output = DiscussionOutput(public_message="たぶん占い師です、と言えたらいいのですが。")

    drafts = _drafts(output, controller.state, "p1")
    register_claim_drafts(controller, "p1", drafts, "m1")

    assert [draft.confidence for draft in drafts] == [0.4]
    assert controller.state.co_declarations == ()
    assert controller.state.speech_events[0].event_type is SpeechEventType.ROLE_CLAIM
    assert controller.state.speech_events[0].is_binding is False


def test_p1_and_p11_stay_distinct_when_reading_a_spoken_result():
    state = make_state()
    output = DiscussionOutput(public_message="占いCO。Player11(p11)は黒でした。")

    drafts = _drafts(output, state, "p3")

    assert [d.target_id for d in drafts[1:]] == ["p11"]


# -- keeping the prose and the ledger in agreement --


def test_a_declared_claim_missing_from_the_prose_is_written_into_it():
    state = make_state()
    ledger = PublicFactLedger(state)
    output = DiscussionOutput(
        public_message="結果を報告します。",
        public_claim_role="seer",
        public_results=[
            PublicResultClaim(result_type="seer", target_id="p5", is_werewolf=True)
        ],
    )
    drafts = _drafts(output, state, "p3")

    fixed = ensure_fact_sentences(output.public_message, drafts, ledger, speaker_id="p3")

    assert fixed == "占いCO。占い結果、Player5(p5)は黒(人狼)です。結果を報告します。"


def test_a_claim_the_prose_already_states_is_not_repeated():
    state = make_state()
    ledger = PublicFactLedger(state)
    message = "占いCO。" + render_result_sentence(ledger, "seer", "p5", True)
    output = DiscussionOutput(
        public_message=message,
        public_claim_role="seer",
        public_results=[
            PublicResultClaim(result_type="seer", target_id="p5", is_werewolf=True)
        ],
    )
    drafts = _drafts(output, state, "p3")

    assert ensure_fact_sentences(message, drafts, ledger, speaker_id="p3") == message


def test_the_written_sentence_is_read_back_as_the_same_event():
    state = make_state(day=2)
    execute(state, "p4", day=1)
    ledger = PublicFactLedger(state)
    sentence = render_result_sentence(ledger, "medium", "p4", False)

    spoken = DiscussionOutput(public_message="霊媒師CO。" + sentence)
    drafts = build_claim_drafts(spoken, ledger, speaker_id="p3")

    assert [(d.event_type, d.target_id, d.result_is_werewolf) for d in drafts] == [
        (SpeechEventType.ROLE_CLAIM, None, None),
        (SpeechEventType.ABILITY_RESULT, "p4", False),
    ]


# -- registration goes through the engine --


def test_registered_drafts_carry_the_message_they_were_said_in():
    controller = make_controller(seed=4)
    controller.state.day = 1
    output = DiscussionOutput(
        public_message="占いCO。",
        public_claim_role="seer",
        public_results=[
            PublicResultClaim(result_type="seer", target_id="p5", is_werewolf=True)
        ],
    )

    drafts = _drafts(output, controller.state, "p3")
    register_claim_drafts(controller, "p3", drafts, "m12")

    assert current_role_claim(controller.state.speech_events, "p3").source_message_id == "m12"
    assert result_versions(controller.state.speech_events)[0].source_message_id == "m12"
