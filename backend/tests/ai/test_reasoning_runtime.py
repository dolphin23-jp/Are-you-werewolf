"""The reasoning layer driving an actual game, with the model left the wording.

The regression scenario the roadmap asks for runs at the bottom: a human
corrects their own vote record, some AIs drop the suspicion, one keeps it on
independent evidence, and the table does not collapse onto a single name.
"""

from __future__ import annotations

import asyncio

from app.ai.coordinator import AICoordinator
from app.ai.provider.mock import MockProvider
from app.ai.reasoning.belief import EvidenceRecord, vote_fact_id
from app.ai.reasoning.runtime import (
    MAX_OPENING_SPEAKERS,
    MIN_OPENING_SPEAKERS,
    ReasoningRuntime,
)
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.state import ChatChannel, ChatMessage, VoteRecord
from tests.ai.reasoning.solver import boards
from tests.conftest import make_controller

AI_IDS = [f"p{i}" for i in range(1, 17)]


def _runtime(state, seed: int = 3) -> ReasoningRuntime:  # type: ignore[no-untyped-def]
    runtime = ReasoningRuntime(state, AI_IDS, seed=seed)
    runtime.refresh(state)
    return runtime


def _played_board(day: int = 2):  # type: ignore[no-untyped-def]
    state = boards.deal(
        {"p1": RoleName.WEREWOLF, "p4": RoleName.SEER, "p6": RoleName.FOX}, day=day
    )
    boards.claim(state, "p4", RoleName.SEER, day=1)
    boards.verdict(state, "p4", "seer", "p9", True, day=1)
    for voter in ("p2", "p3", "p5", "p7"):
        state.vote_records.append(
            VoteRecord(voter_id=voter, target_id="p11", day=1, round=1)
        )
    return state


# -- decisions that no longer cost a request --


def test_a_vote_is_decided_from_evidence_and_explains_itself():
    state = _played_board()
    runtime = _runtime(state)

    target, reason = runtime.vote_decision("p2", [f"p{i}" for i in range(1, 17)])

    assert target in {"p9", "p11"}
    assert reason
    # The reason is the evidence, not a sentence a model produced.
    assert "判定" in reason or "投票" in reason


def test_night_targets_chase_suspicion_and_guard_protects_trust():
    state = _played_board()
    runtime = _runtime(state)
    candidates = [f"p{i}" for i in range(2, 16)]

    divined = runtime.night_target("p4", "divine", candidates)
    guarded = runtime.night_target("p4", "guard", candidates)

    scores = runtime.seats["p4"].belief.state.public_suspicion_scores
    assert scores.get(divined, 0.0) >= scores.get(guarded, 0.0)


def test_the_same_board_and_seed_decide_identically():
    state = _played_board()

    first = _runtime(state).vote_decision("p2", AI_IDS)
    second = _runtime(state).vote_decision("p2", AI_IDS)

    assert first == second


def test_a_repeat_refresh_on_an_unchanged_board_does_no_work():
    state = _played_board()
    runtime = _runtime(state)
    before = runtime.seats["p2"].belief.state.public_suspicion_scores.copy()

    runtime.refresh(state)

    assert runtime.seats["p2"].belief.state.public_suspicion_scores == before


# -- the speech scheduler --


def test_only_a_handful_of_seats_open_the_day():
    state = _played_board()
    runtime = _runtime(state)

    speakers = runtime.select_opening_speakers(state)

    assert MIN_OPENING_SPEAKERS <= len(speakers) <= MAX_OPENING_SPEAKERS
    assert len(set(speakers)) == len(speakers)


def test_speech_value_prefers_the_seats_with_something_to_say():
    state = _played_board()
    runtime = _runtime(state)

    ranked = runtime.speech_candidates(
        state, pending_question_targets=["p12"], pressured_ids=["p9"]
    )
    by_id = {item.player_id: item for item in ranked}

    assert by_id["p12"].value >= 3.0
    assert "名指しで質問された" in by_id["p12"].reasons
    assert by_id["p9"].value >= 2.5
    assert ranked[0].value >= ranked[-1].value


def test_the_speaking_order_is_seeded_and_reproducible():
    state = _played_board()

    first = _runtime(state, seed=9).select_opening_speakers(state)
    second = _runtime(state, seed=9).select_opening_speakers(state)

    assert first == second


def test_holding_an_unpublished_result_outranks_everything_else():
    state = _played_board()
    boards.divine(state, "p4", "p13", night=1)
    runtime = _runtime(state)

    ranked = runtime.speech_candidates(state)

    assert ranked[0].player_id == "p4"
    assert "未公開の能力結果を持つ" in ranked[0].reasons


# -- one human message, no per-AI parsing --


def test_a_human_correction_is_matched_once_for_the_whole_table():
    state = _played_board()
    state.vote_records.append(VoteRecord(voter_id="p0", target_id="p3", day=2, round=1))
    runtime = _runtime(state)
    for seat in runtime.seats.values():
        seat.belief.add_evidence(
            EvidenceRecord(
                evidence_id="recalled:p0",
                subject_id="p0",
                category="misremembered_vote",
                source_event_ids=(vote_fact_id("p0", 2, 1, "p12"),),
                weight=1.5,
                explanation="p0はp12へ投票していた。",
            )
        )
    name = state.players["p12"].name
    other = state.players["p3"].name

    outcomes = runtime.apply_human_message(
        state, "p0", f"2日目、私は{name}には投票していません。{other}へ投票しました。"
    )

    assert outcomes
    # One parse, every seat updated: the misreading is inactive table-wide.
    assert all(
        not next(
            record
            for record in seat.belief.evidence
            if record.evidence_id == "recalled:p0"
        ).active
        for seat in runtime.seats.values()
    )
    assert runtime.metrics()["human_messages_considered"] == 1


def test_only_the_seats_that_used_the_wrong_fact_are_worth_hearing_from():
    state = _played_board()
    state.vote_records.append(VoteRecord(voter_id="p0", target_id="p3", day=2, round=1))
    runtime = _runtime(state)
    misled = runtime.seats["p2"]
    misled.belief.add_evidence(
        EvidenceRecord(
            evidence_id="recalled:p0",
            subject_id="p0",
            category="misremembered_vote",
            source_event_ids=(vote_fact_id("p0", 2, 1, "p12"),),
            weight=1.5,
            explanation="p0はp12へ投票していた。",
        )
    )
    name = state.players["p12"].name

    outcomes = runtime.apply_human_message(
        state, "p0", f"2日目、私は{name}には投票していません。"
    )

    assert runtime.seats_that_moved(outcomes) == ["p2"]


# -- what the layer reports about itself --


def test_the_table_does_not_all_name_the_same_person():
    state = _played_board()
    runtime = _runtime(state)

    spread = runtime.opinion_spread()
    distribution = runtime.target_distribution()

    assert spread > 0.0
    assert len(distribution) > 1


def test_personalities_produce_visibly_different_ballots():
    state = _played_board()
    runtime = _runtime(state)

    by_profile = runtime.distribution_by_profile()

    assert len(by_profile) > 1
    chosen = {target for bucket in by_profile.values() for target in bucket}
    assert len(chosen) > 1


def test_conflict_points_come_from_structured_events():
    state = _played_board()
    boards.claim(state, "p8", RoleName.SEER, day=2)
    runtime = _runtime(state)

    points = runtime.conflict_points(state)

    # Labelled the way the board memo labels roles, not with the enum value.
    assert any("占いCO対抗" in point for point in points)
    assert not any("seer" in point for point in points)
    assert any("p4→p9=黒" in point for point in points)


def test_a_freemason_pair_is_not_reported_as_a_counter_claim():
    """Two freemasons is the composition. Only claims past a role's seat count
    are a contest -- the rule the board memo already states."""
    state = boards.deal({"p6": RoleName.FREEMASON, "p7": RoleName.FREEMASON}, day=2)
    boards.claim(state, "p6", RoleName.FREEMASON, day=1)
    boards.claim(state, "p7", RoleName.FREEMASON, day=1)
    boards.claim(state, "p4", RoleName.SEER, day=1)
    runtime = _runtime(state)

    points = runtime.conflict_points(state)

    assert not any("CO対抗" in point for point in points)


def test_a_third_freemason_claim_is_a_counter_claim():
    state = boards.deal({"p6": RoleName.FREEMASON, "p7": RoleName.FREEMASON}, day=2)
    for player_id in ("p6", "p7", "p8"):
        boards.claim(state, player_id, RoleName.FREEMASON, day=1)
    runtime = _runtime(state)

    assert "共有CO対抗: p6、p7、p8" in runtime.conflict_points(state)


def test_no_seats_private_target_reaches_the_shared_summary():
    """The seer's unpublished black is what its belief engine chases. The
    summary every seat reads used to list it as a 'split' nobody had voiced."""
    state = boards.deal({"p4": RoleName.SEER, "p9": RoleName.WEREWOLF}, day=2)
    boards.divine(state, "p4", "p9", night=1)
    runtime = _runtime(state)
    private_targets = {
        seat.belief.state.current_execution_target for seat in runtime.seats.values()
    }
    assert runtime.seats["p4"].belief.state.current_execution_target == "p9"
    assert len(private_targets - {None}) > 1  # the belief engines do disagree

    points = runtime.conflict_points(state)

    assert not any("割れている" in point for point in points)
    assert not any("p9" in point for point in points)


def _says(state, speaker: str, target: str | None, day: int) -> None:  # type: ignore[no-untyped-def]
    """`speaker` names `target` in public on `day`, and the runtime records it."""
    text = f"Player{target[1:]}が怪しい。" if target else "まだ決めきれない。"
    state.chat_log.append(
        ChatMessage(
            message_id=f"m{len(state.chat_log)}",
            author_id=speaker,
            content=text,
            channel=ChatChannel.PUBLIC,
            day=day,
        )
    )


def test_a_publicly_stated_split_is_reported_from_todays_statements_only():
    state = _played_board()
    boards.execute(state, "p11", day=2)
    runtime = _runtime(state)
    stated = [
        ("p2", "p5", 2),
        ("p3", "p5", 2),
        ("p7", "p8", 2),
        ("p10", "p12", 1),  # yesterday's: not today's split
        ("p13", "p11", 2),  # executed since: no longer a choice
        ("p14", None, 2),  # named nobody
    ]
    for speaker, target, day in stated:
        _says(state, speaker, target, day)
        runtime.record_stated_target(speaker, target, day)

    points = runtime.conflict_points(state)

    assert "吊り先の意見が割れている: p5←p2・p3、p8←p7" in points


def test_one_publicly_stated_candidate_is_not_a_split():
    state = _played_board()
    runtime = _runtime(state)
    for speaker, target, day in (("p2", "p5", 2), ("p3", "p5", 2), ("p7", "p8", 1)):
        _says(state, speaker, target, day)
        runtime.record_stated_target(speaker, target, day)

    assert not any("割れている" in point for point in runtime.conflict_points(state))


def test_a_candidate_only_in_a_seats_structured_answer_stays_private():
    """Under v3 the stated target is the model's structured answer. A seat that
    never named it out loud has not told the table, so the summary must not."""
    state = _played_board()
    runtime = _runtime(state)
    _says(state, "p2", "p5", 2)
    runtime.record_stated_target("p2", "p5", 2)
    _says(state, "p3", None, 2)  # spoke, but named nobody
    runtime.record_stated_target("p3", "p8", 2)

    points = runtime.conflict_points(state)

    assert not any("割れている" in point for point in points)
    assert not any("p8" in point for point in points)


def test_a_stated_target_belongs_to_the_day_it_was_stated():
    state = _played_board()
    runtime = _runtime(state)
    runtime.record_stated_target("p2", "p5", 1)

    assert runtime.stated_target("p2", 1) == "p5"
    assert runtime.stated_target("p2", 2) is None
    # Undated, it is simply the last thing the seat said.
    assert runtime.stated_target("p2") == "p5"
    assert runtime.stated_targets_on(1) == {"p2": "p5"}
    assert runtime.stated_targets_on(2) == {}


def test_hypotheses_are_reported_in_bands_not_percentages():
    state = _played_board()
    runtime = _runtime(state)

    summary = runtime.hypothesis_summary("p2")

    assert summary
    assert "%" not in summary
    assert "本線" in summary


# -- the coordinator actually uses it --


def test_v2_casts_a_vote_without_calling_the_model():
    controller = make_controller(seed=4)
    controller.state.phase = Phase.VOTING
    controller.state.day = 2
    provider = MockProvider(seed=4)
    runtime = ReasoningRuntime(controller.state, ["p1", "p2"], seed=4)
    coordinator = AICoordinator(
        controller.state, ["p1", "p2"], provider, seed=4, reasoning=runtime
    )

    class _Refusing:
        async def generate_structured(self, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("v2 must not spend a request on a vote")

    coordinator._agents["p1"]._provider = _Refusing()  # type: ignore[attr-defined]

    asyncio.run(coordinator._cast_vote(controller, controller.state, "p1"))

    assert controller.state.pending_votes["p1"] in controller.state.alive_ids()


def test_v3_is_the_default_and_the_older_engines_remain_selectable():
    from app.config import Settings

    assert Settings().werewolf_reasoning_engine == "v3"
    assert Settings(werewolf_reasoning_engine="v2").werewolf_reasoning_engine == "v2"
    assert Settings(werewolf_reasoning_engine="legacy").werewolf_reasoning_engine == "legacy"
    # AICoordinator itself doesn't read Settings -- app/api/routes_game.py is
    # what bridges the config flag to `reasoning=`, so what stays true here is
    # just that the old path (no reasoning runtime at all) is still reachable
    # by construction, for whichever caller wants it.
    controller = make_controller(seed=4)
    coordinator = AICoordinator(controller.state, ["p1"], MockProvider(seed=4), seed=4)
    assert coordinator.reasoning is None


# -- the required regression scenario --


def test_a_human_correction_splits_the_table_rather_than_resetting_it():
    """The scenario from the roadmap, end to end.

    A human corrects their vote record. Seats that leaned on the wrong reading
    drop it; a seat holding an independent reason keeps some suspicion; and the
    table does not converge on one name afterwards.
    """
    state = _played_board()
    state.vote_records.append(VoteRecord(voter_id="p0", target_id="p3", day=2, round=1))
    runtime = _runtime(state)
    misreading = EvidenceRecord(
        evidence_id="recalled:p0",
        subject_id="p0",
        category="misremembered_vote",
        source_event_ids=(vote_fact_id("p0", 2, 1, "p12"),),
        weight=2.0,
        explanation="p0はp12へ投票していた。",
    )
    # p3 is excluded on purpose: p0 voted for them, and a seat that knows its
    # own card reads "you voted for someone provably not a wolf" as its own
    # small piece of evidence, which the correction does not touch.
    for seat_id in ("p2", "p7", "p5"):
        runtime.seats[seat_id].belief.add_evidence(misreading)
    # p5 also has a reason that survives the correction.
    runtime.seats["p5"].belief.add_evidence(
        EvidenceRecord(
            evidence_id="independent:p0",
            subject_id="p0",
            category="accusation",
            source_event_ids=("independent",),
            weight=1.0,
            explanation="p0の1日目の発言が矛盾していた。",
        )
    )
    for seat_id in ("p2", "p7", "p5"):
        runtime.seats[seat_id].belief.recompute_public(state)
    name = state.players["p12"].name

    runtime.apply_human_message(
        state, "p0", f"2日目、私は{name}には投票していません。"
    )

    # Conceded and dropped.
    assert runtime.seats["p2"].belief.state.public_suspicion_scores["p0"] == 0.0
    assert runtime.seats["p7"].belief.state.public_suspicion_scores["p0"] == 0.0
    # Conceded but not cleared -- the other reason is still standing, and is the
    # only one now cited.
    assert runtime.seats["p5"].belief.state.public_suspicion_scores["p0"] > 0.0
    assert runtime.seats["p5"].belief.state.reasons_for("p0") == ("independent:p0",)
    # A seat that never used the wrong fact is untouched.
    assert runtime.seats["p8"].belief.state.public_suspicion_scores["p0"] == 0.0
    assert runtime.seats["p8"].belief.state.reasons_for("p0") == ()
    # And the table still disagrees about who to execute.
    assert runtime.opinion_spread() > 0.0
