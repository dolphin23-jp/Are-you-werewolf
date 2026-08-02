"""Beliefs that can be withdrawn, because their reasons can be.

The scenario these tests exist for: an AI suspects someone because it thinks
they voted a certain way, the human corrects the record, and the AI concedes the
point -- then has to actually stop using it.
"""

from __future__ import annotations

from app.ai.reasoning.belief import (
    HARD_EXCLUDED_SCORE,
    PUBLISHED_BLACK_WEIGHT,
    TRUST_STEP,
    BeliefEngine,
    CorrectionKind,
    CorrectionStatus,
    EvidenceRecord,
    FactCorrection,
    parse_fact_corrections,
    verify,
    vote_fact_id,
)
from app.ai.reasoning.facts import PublicFactLedger
from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import CommonPublicPerspective, PlayerPrivatePerspective
from app.ai.reasoning.solver import build_solver
from app.engine.roles import RoleName
from app.engine.state import VoteRecord
from tests.ai.reasoning.solver import boards


def _board(day: int = 2):  # type: ignore[no-untyped-def]
    return boards.deal(
        {"p1": RoleName.WEREWOLF, "p4": RoleName.SEER, "p6": RoleName.FOX}, day=day
    )


def _vote(state, voter: str, target: str, day: int = 1, round_number: int = 1) -> None:  # type: ignore[no-untyped-def]
    state.vote_records.append(
        VoteRecord(voter_id=voter, target_id=target, day=day, round=round_number)
    )


def _misremembered_vote_evidence(
    subject: str, wrong_target: str, day: int = 1
) -> EvidenceRecord:
    """An AI's own recollection of a ballot -- which may be wrong.

    The target is baked into the source id, so a correction can retract exactly
    this reading and nothing else.
    """
    return EvidenceRecord(
        evidence_id=f"recalled_vote:{subject}:{day}",
        subject_id=subject,
        category="misremembered_vote",
        source_event_ids=(vote_fact_id(subject, day, 1, wrong_target),),
        weight=1.5,
        explanation=f"{day}日目、{subject}は{wrong_target}へ投票していた。",
    )


def _engine(player_id: str = "p2") -> BeliefEngine:
    return BeliefEngine(player_id, CommonPublicPerspective())


# -- provenance --


def test_every_point_of_a_suspicion_traces_to_named_evidence():
    state = _board()
    boards.claim(state, "p4", RoleName.SEER)
    boards.verdict(state, "p4", "seer", "p9", True)
    ledger = PublicFactLedger(state)
    engine = _engine()

    engine.observe(ledger)

    assert engine.state.wolf_scores["p9"] == PUBLISHED_BLACK_WEIGHT
    reasons = engine.state.reasons_for("p9")
    assert reasons == ("verdict:p4:seer:p9:黒",)
    record = next(r for r in engine.evidence if r.evidence_id == reasons[0])
    assert "黒と公開された" in record.explanation


def test_re_observing_the_same_fact_does_not_double_count_it():
    state = _board()
    boards.claim(state, "p4", RoleName.SEER)
    boards.verdict(state, "p4", "seer", "p9", True)
    ledger = PublicFactLedger(state)
    engine = _engine()

    engine.observe(ledger)
    engine.observe(ledger)

    assert engine.state.wolf_scores["p9"] == PUBLISHED_BLACK_WEIGHT


def test_contested_claims_raise_suspicion_on_every_claimant():
    state = _board()
    boards.claim(state, "p4", RoleName.SEER)
    boards.claim(state, "p8", RoleName.SEER)
    engine = _engine()

    engine.observe(PublicFactLedger(state))

    # Two seer COs means at least one is lying, so both carry the cost.
    assert engine.state.wolf_scores["p4"] > 0
    assert engine.state.wolf_scores["p8"] > 0
    assert engine.state.reasons_for("p4") == ("contested:seer:p4",)


# -- corrections --


def test_a_confirmed_correction_retracts_the_evidence_that_rested_on_it():
    state = _board()
    _vote(state, "p0", "p3", day=1)
    ledger = PublicFactLedger(state)
    engine = _engine()
    engine.add_evidence(_misremembered_vote_evidence("p0", "p12"))
    engine.recompute(ledger)
    assert engine.state.wolf_scores["p0"] == 1.5

    outcome = engine.apply_correction(
        FactCorrection(
            kind=CorrectionKind.VOTE_TARGET,
            source_player_id="p0",
            subject_id="p0",
            day=1,
            asserted="p3",
            denied="p12",
        ),
        ledger,
    )

    assert outcome.verdict.status is CorrectionStatus.CONFIRMED
    assert outcome.invalidated_evidence_ids == ("recalled_vote:p0:1",)
    assert engine.state.wolf_scores["p0"] == 0.0
    assert engine.state.reasons_for("p0") == ()


def test_a_conclusion_may_survive_a_correction_only_on_its_other_evidence():
    state = _board()
    _vote(state, "p0", "p3", day=1)
    boards.claim(state, "p4", RoleName.SEER)
    boards.verdict(state, "p4", "seer", "p0", True)
    ledger = PublicFactLedger(state)
    engine = _engine()
    engine.observe(ledger)
    engine.add_evidence(_misremembered_vote_evidence("p0", "p12"))
    engine.recompute(ledger)

    engine.apply_correction(
        FactCorrection(
            kind=CorrectionKind.VOTE_TARGET,
            source_player_id="p0",
            subject_id="p0",
            day=1,
            asserted="p3",
            denied="p12",
        ),
        ledger,
    )

    # The suspicion survives, but only on the published black -- the withdrawn
    # reading contributes nothing to the score and nothing to the reasons.
    assert engine.state.wolf_scores["p0"] == PUBLISHED_BLACK_WEIGHT
    assert engine.state.reasons_for("p0") == ("verdict:p4:seer:p0:黒",)


def test_a_retracted_reason_is_never_offered_as_a_reason_again():
    state = _board()
    _vote(state, "p0", "p3", day=1)
    ledger = PublicFactLedger(state)
    engine = _engine()
    engine.add_evidence(_misremembered_vote_evidence("p0", "p12"))
    engine.recompute(ledger)

    engine.apply_correction(
        FactCorrection(
            kind=CorrectionKind.VOTE_TARGET,
            source_player_id="p0",
            subject_id="p0",
            day=1,
            asserted="p3",
            denied="p12",
        ),
        ledger,
    )

    active_ids = {record.evidence_id for record in engine.active_evidence()}
    assert "recalled_vote:p0:1" not in active_ids
    # Kept, but marked withdrawn: the audit trail is the point of not deleting.
    withdrawn = next(r for r in engine.evidence if r.evidence_id == "recalled_vote:p0:1")
    assert withdrawn.active is False
    assert "撤回" in withdrawn.explanation


def test_a_false_correction_is_refuted_and_costs_its_source_credibility():
    state = _board()
    _vote(state, "p0", "p12", day=1)
    ledger = PublicFactLedger(state)
    engine = _engine()
    engine.add_evidence(_misremembered_vote_evidence("p0", "p12"))
    engine.recompute(ledger)

    outcome = engine.apply_correction(
        FactCorrection(
            kind=CorrectionKind.VOTE_TARGET,
            source_player_id="p0",
            subject_id="p0",
            day=1,
            asserted="p3",
            denied="p12",
        ),
        ledger,
    )

    # The record says otherwise, so nothing is retracted and the claim costs.
    assert outcome.verdict.status is CorrectionStatus.REFUTED
    assert engine.state.wolf_scores["p0"] == 1.5
    assert engine.state.source_trust["p0"] == -TRUST_STEP


def test_a_correct_correction_earns_its_source_credibility():
    state = _board()
    _vote(state, "p0", "p3", day=1)
    ledger = PublicFactLedger(state)
    engine = _engine()

    engine.apply_correction(
        FactCorrection(
            kind=CorrectionKind.VOTE_TARGET,
            source_player_id="p0",
            subject_id="p0",
            day=1,
            asserted="p3",
            denied="p12",
        ),
        ledger,
    )

    assert engine.state.source_trust["p0"] == TRUST_STEP


def test_an_uncheckable_correction_moves_nothing():
    state = _board()
    ledger = PublicFactLedger(state)
    engine = _engine()

    outcome = engine.apply_correction(
        FactCorrection(
            kind=CorrectionKind.VOTE_TARGET,
            source_player_id="p0",
            subject_id="p0",
            day=1,
            asserted="p3",
            denied="p12",
        ),
        ledger,
    )

    assert outcome.verdict.status is CorrectionStatus.UNVERIFIABLE
    assert engine.state.source_trust == {}


def test_a_correction_reaches_only_the_seats_that_used_the_wrong_fact():
    state = _board()
    _vote(state, "p0", "p3", day=1)
    ledger = PublicFactLedger(state)
    correction = FactCorrection(
        kind=CorrectionKind.VOTE_TARGET,
        source_player_id="p0",
        subject_id="p0",
        day=1,
        asserted="p3",
        denied="p12",
    )

    misled_a, misled_b = _engine("p2"), _engine("p5")
    for engine in (misled_a, misled_b):
        engine.add_evidence(_misremembered_vote_evidence("p0", "p12"))
        engine.recompute(ledger)
    unaffected = _engine("p7")
    unaffected.add_evidence(
        EvidenceRecord(
            evidence_id="own_read:p0",
            subject_id="p0",
            category="misremembered_vote",
            source_event_ids=(vote_fact_id("p0", 1, 1, "p3"),),
            weight=0.5,
            explanation="p0はp3へ投票していた。",
        )
    )
    unaffected.recompute(ledger)
    before = dict(unaffected.state.wolf_scores)

    outcomes = [engine.apply_correction(correction, ledger) for engine in (misled_a, misled_b)]
    untouched = unaffected.apply_correction(correction, ledger)

    # Both misled seats update independently; the one that had it right is not
    # disturbed, and each seat's state is its own.
    assert all(outcome.changed_anything for outcome in outcomes)
    assert misled_a.state.wolf_scores["p0"] == misled_b.state.wolf_scores["p0"] == 0.0
    assert untouched.changed_anything is False
    assert unaffected.state.wolf_scores == before


def test_belief_states_are_never_shared_between_seats():
    state = _board()
    boards.claim(state, "p4", RoleName.SEER)
    boards.verdict(state, "p4", "seer", "p9", True)
    ledger = PublicFactLedger(state)
    first, second = _engine("p2"), _engine("p5")

    first.observe(ledger)
    first.add_evidence(_misremembered_vote_evidence("p3", "p12"))
    first.recompute(ledger)
    second.observe(ledger)

    assert first.state is not second.state
    assert first.state.wolf_scores["p3"] == 1.5
    assert second.state.wolf_scores["p3"] == 0.0
    assert second.state.player_id == "p5"


# -- hard facts outrank soft ones --


def test_a_seat_the_rules_exclude_is_never_the_execution_target():
    state = _board()
    boards.kill_first_victim(state, "p3")
    ledger = PublicFactLedger(state)
    solver = build_solver(ObservationSet.from_state(state), CommonPublicPerspective())
    engine = _engine()
    # Pile soft suspicion on the one seat the rules have cleared.
    engine.add_evidence(
        EvidenceRecord(
            evidence_id="hunch:p3",
            subject_id="p3",
            category="misremembered_vote",
            source_event_ids=("hunch",),
            weight=99.0,
            explanation="強い勘。",
        )
    )

    engine.observe(ledger, solver)

    assert engine.state.wolf_scores["p3"] == HARD_EXCLUDED_SCORE
    assert engine.state.current_execution_target != "p3"


def test_a_solver_certainty_pins_the_target_regardless_of_soft_weight():
    state = _board()
    ledger = PublicFactLedger(state)
    solver = build_solver(
        ObservationSet.from_state(state), PlayerPrivatePerspective("p1")
    )
    engine = BeliefEngine("p1", PlayerPrivatePerspective("p1"))

    engine.observe(ledger, solver)

    # p1 is a wolf and knows their team, so those seats are settled and lead.
    assert engine.state.confidence == 1.0
    assert engine.state.current_execution_target in solver.observations.player_ids
    assert engine.state.wolf_scores[engine.state.current_execution_target] > 0


def test_the_dead_drop_out_of_the_current_candidates():
    state = _board()
    boards.claim(state, "p4", RoleName.SEER)
    boards.verdict(state, "p4", "seer", "p9", True)
    engine = _engine()
    engine.observe(PublicFactLedger(state))
    assert engine.state.current_execution_target == "p9"

    boards.execute(state, "p9", day=1)
    engine.recompute(PublicFactLedger(state))

    assert engine.state.current_execution_target != "p9"
    # The reason survives -- the corpse is not evidence that the black was wrong.
    assert engine.state.reasons_for("p9") == ("verdict:p4:seer:p9:黒",)


def test_a_player_never_nominates_themselves():
    state = _board()
    boards.claim(state, "p4", RoleName.SEER)
    boards.verdict(state, "p4", "seer", "p2", True)
    engine = _engine("p2")

    engine.observe(PublicFactLedger(state))

    assert engine.state.wolf_scores["p2"] > 0
    assert engine.state.current_execution_target != "p2"


# -- determinism --


def test_the_same_inputs_produce_the_same_beliefs():
    state = _board()
    boards.claim(state, "p4", RoleName.SEER)
    boards.claim(state, "p8", RoleName.SEER)
    boards.verdict(state, "p4", "seer", "p9", True)
    boards.verdict(state, "p8", "seer", "p9", False)
    _vote(state, "p0", "p9", day=1)
    ledger = PublicFactLedger(state)
    solver = build_solver(ObservationSet.from_state(state), CommonPublicPerspective())

    first, second = _engine(), _engine()
    first.observe(ledger, solver)
    second.observe(ledger, solver)

    assert first.state.wolf_scores == second.state.wolf_scores
    assert first.state.current_execution_target == second.state.current_execution_target
    assert first.state.active_hypotheses == second.state.active_hypotheses


# -- parsing and verification --


def test_a_spoken_correction_is_read_and_checked_against_the_record():
    state = _board()
    _vote(state, "p0", "p3", day=1)
    ledger = PublicFactLedger(state)
    name = ledger.name_of("p12")
    other = ledger.name_of("p3")

    corrections = parse_fact_corrections(
        f"1日目、私は{name}には投票していません。{other}へ投票しました。",
        ledger,
        speaker_id="p0",
    )

    assert len(corrections) == 1
    correction = corrections[0]
    assert (correction.denied, correction.asserted, correction.day) == ("p12", "p3", 1)
    assert verify(correction, ledger).status is CorrectionStatus.CONFIRMED


def test_an_ordinary_complaint_is_not_read_as_a_correction():
    state = _board()
    _vote(state, "p0", "p3", day=1)
    ledger = PublicFactLedger(state)

    assert parse_fact_corrections("その言い方は不当だと思います。", ledger, "p0") == []
    assert parse_fact_corrections("投票先を変えるか迷っています。", ledger, "p0") == []


def test_other_checkable_facts_are_verified_against_the_ledger():
    state = _board()
    boards.execute(state, "p9", day=1)
    ledger = PublicFactLedger(state)

    executed = verify(
        FactCorrection(
            kind=CorrectionKind.EXECUTED, source_player_id="p2", subject_id="p9", day=1
        ),
        ledger,
    )
    never_executed = verify(
        FactCorrection(
            kind=CorrectionKind.EXECUTED,
            source_player_id="p2",
            subject_id="p8",
            denied="executed",
        ),
        ledger,
    )
    wrong_liveness = verify(
        FactCorrection(
            kind=CorrectionKind.ALIVE,
            source_player_id="p2",
            subject_id="p9",
            asserted="alive",
        ),
        ledger,
    )

    assert executed.status is CorrectionStatus.CONFIRMED
    assert never_executed.status is CorrectionStatus.CONFIRMED
    assert wrong_liveness.status is CorrectionStatus.REFUTED


def test_a_role_claim_correction_checks_the_standing_claim():
    state = _board()
    boards.claim(state, "p4", RoleName.SEER)
    ledger = PublicFactLedger(state)

    denied_wrongly = verify(
        FactCorrection(
            kind=CorrectionKind.ROLE_CLAIM,
            source_player_id="p4",
            subject_id="p4",
            denied="seer",
        ),
        ledger,
    )
    denied_correctly = verify(
        FactCorrection(
            kind=CorrectionKind.ROLE_CLAIM,
            source_player_id="p8",
            subject_id="p8",
            denied="medium",
        ),
        ledger,
    )

    assert denied_wrongly.status is CorrectionStatus.REFUTED
    assert denied_correctly.status is CorrectionStatus.CONFIRMED
