"""When a result happened, and what happens to a reason when its fact is gone.

Two failures this file pins down.

The first: a verdict carried a colour and a publication day but never the night
it came from, so a seer sitting on a night-1 black until day 3 was read as
having looked that morning. Everything downstream inherited the error -- the
bluffer's own story most of all, because it reconstructed its fake schedule from
publication days.

The second: evidence outlived its fact. A published black created a reason, the
claimant later retracted or corrected it, and the reason stayed. The seat went
on citing a verdict the table could no longer find in the record, which is the
same category of failure as maintaining a wrong vote history.
"""

from __future__ import annotations

from app.ai.reasoning.belief import BeliefEngine
from app.ai.reasoning.belief.engine import (
    PUBLISHED_BLACK_WEIGHT,
    result_fact_id,
)
from app.ai.reasoning.facts import PublicFactLedger
from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import (
    ClaimedStoryPerspective,
    PlayerPrivatePerspective,
)
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.reasoning.solver import AccurateTimeline, build_solver, has_role
from app.ai.reasoning.solver.backend import Certainty
from app.ai.reasoning.timeline import ConflictKind, find_timeline_conflicts
from app.engine.roles import RoleName
from tests.ai.reasoning.solver import boards

ALL_SEATS = [f"p{i}" for i in range(17)]


def _seer_board(day: int = 3):  # type: ignore[no-untyped-def]
    state = boards.deal(
        {"p1": RoleName.WEREWOLF, "p4": RoleName.SEER, "p6": RoleName.FOX}, day=day
    )
    boards.claim(state, "p4", RoleName.SEER, day=1)
    return state


def _engine(state, player_id: str = "p2") -> BeliefEngine:  # type: ignore[no-untyped-def]
    engine = BeliefEngine(player_id, PlayerPrivatePerspective(player_id))
    observations = ObservationSet.from_state(state)
    engine.observe(
        PublicFactLedger(state),
        build_solver(observations, PlayerPrivatePerspective(player_id)),
        observations,
    )
    return engine


# -- when the result happened --


def test_a_result_carries_the_night_it_came_from_all_the_way_down():
    state = _seer_board()
    boards.verdict(state, "p4", "seer", "p9", True, day=3, referenced_day=1)

    ledger_fact = PublicFactLedger(state).public_results()[0]
    verdict = ObservationSet.from_state(state).verdicts[0]

    assert ledger_fact.referenced_day == 1
    assert ledger_fact.source_night == 1
    assert verdict.referenced_day == 1
    assert verdict.source_night == 1


def test_an_unstated_night_still_falls_back_to_the_night_before():
    state = _seer_board(day=2)
    boards.verdict(state, "p4", "seer", "p9", True, day=2)

    assert PublicFactLedger(state).public_results()[0].source_night == 1


def test_a_bluffers_story_uses_the_night_claimed_not_the_day_published():
    """A held-back result is a normal play, and it used to corrupt the story.

    p2 fake-claims seer and publishes a night-1 look on day 3. Reading the night
    off the publication day would put the look on night 2 -- a night the story
    never claimed -- and the bluffer would then reason about corpses against a
    schedule they never told anybody.
    """
    state = _seer_board()
    boards.claim(state, "p2", RoleName.SEER, day=1)
    boards.verdict(state, "p2", "seer", "p9", True, day=3, referenced_day=1)
    observations = ObservationSet.from_state(state)

    story = ClaimedStoryPerspective("p2", RoleName.SEER)
    nights = story.known_night_actions(observations).divines

    assert nights == {1: "p9"}


# -- conflicts with the public calendar --


def test_an_honest_account_produces_no_conflicts():
    state = _seer_board(day=2)
    boards.execute(state, "p9", day=1)
    boards.verdict(state, "p4", "seer", "p11", False, day=2, referenced_day=1)

    assert find_timeline_conflicts(ObservationSet.from_state(state)) == ()


def test_two_looks_on_one_night_is_a_conflict():
    state = _seer_board()
    boards.verdict(state, "p4", "seer", "p9", True, day=2, referenced_day=1)
    boards.verdict(state, "p4", "seer", "p11", False, day=3, referenced_day=1)

    kinds = {c.kind for c in find_timeline_conflicts(ObservationSet.from_state(state))}

    assert ConflictKind.DUPLICATE_NIGHT in kinds


def test_a_look_on_a_night_that_has_not_happened_is_a_conflict():
    state = _seer_board(day=2)
    boards.verdict(state, "p4", "seer", "p9", True, day=2, referenced_day=5)

    kinds = {c.kind for c in find_timeline_conflicts(ObservationSet.from_state(state))}

    assert ConflictKind.FUTURE_NIGHT in kinds


def test_a_look_at_someone_already_buried_is_a_conflict():
    state = _seer_board(day=4)
    boards.execute(state, "p9", day=1)
    boards.verdict(state, "p4", "seer", "p9", True, day=4, referenced_day=3)

    conflicts = find_timeline_conflicts(ObservationSet.from_state(state))

    assert any(c.kind == ConflictKind.TARGET_ALREADY_DEAD for c in conflicts)


def test_a_look_on_the_night_the_seer_was_attacked_is_not_a_conflict():
    """Conservative on purpose: those two events are simultaneous.

    From outside, "divined then died" and "died before divining" are the same
    night. Reporting a conflict here would accuse a real seer of lying on the
    strength of the game's own resolution order.
    """
    state = _seer_board(day=3)
    boards.die_by_attack(state, "p4", night=2)
    boards.verdict(state, "p4", "seer", "p11", False, day=2, referenced_day=2)

    conflicts = find_timeline_conflicts(ObservationSet.from_state(state))

    assert not [c for c in conflicts if c.kind == ConflictKind.AFTER_OWN_DEATH]


def test_a_medium_result_for_a_day_with_no_execution_is_a_conflict():
    state = _seer_board(day=2)
    boards.claim(state, "p5", RoleName.MEDIUM, day=1)
    boards.verdict(state, "p5", "medium", "p9", True, day=2, referenced_day=1)

    kinds = {c.kind for c in find_timeline_conflicts(ObservationSet.from_state(state))}

    assert ConflictKind.NO_EXECUTION in kinds


def test_a_medium_result_naming_the_wrong_body_is_a_conflict():
    state = _seer_board(day=2)
    boards.claim(state, "p5", RoleName.MEDIUM, day=1)
    boards.execute(state, "p9", day=1)
    boards.verdict(state, "p5", "medium", "p11", True, day=2, referenced_day=1)

    kinds = {c.kind for c in find_timeline_conflicts(ObservationSet.from_state(state))}

    assert ConflictKind.WRONG_EXECUTION in kinds


# -- what a conflict is allowed to conclude --


def test_a_timeline_conflict_is_not_a_hard_rule_on_its_own():
    """A real seer can lie about *when*. Nothing here may settle a role by itself."""
    state = _seer_board()
    boards.claim(state, "p2", RoleName.SEER, day=1)
    boards.verdict(state, "p2", "seer", "p9", True, day=2, referenced_day=1)
    boards.verdict(state, "p2", "seer", "p11", False, day=3, referenced_day=1)
    observations = ObservationSet.from_state(state)

    solver = build_solver(observations, PlayerPrivatePerspective("p7"))

    assert solver.assess(has_role("p2", RoleName.SEER)) is not Certainty.IMPOSSIBLE


def test_supposing_the_timing_is_accurate_does_rule_the_claim_out():
    state = _seer_board()
    boards.claim(state, "p2", RoleName.SEER, day=1)
    boards.verdict(state, "p2", "seer", "p9", True, day=2, referenced_day=1)
    boards.verdict(state, "p2", "seer", "p11", False, day=3, referenced_day=1)
    observations = ObservationSet.from_state(state)

    solver = build_solver(
        observations,
        PlayerPrivatePerspective("p7"),
        assumptions=(AccurateTimeline("p2"),),
    )

    assert solver.assess(has_role("p2", RoleName.SEER)) is Certainty.IMPOSSIBLE


def test_a_conflict_raises_suspicion_of_the_claimant_not_the_named_player():
    state = _seer_board()
    boards.claim(state, "p2", RoleName.SEER, day=1)
    boards.verdict(state, "p2", "seer", "p9", True, day=2, referenced_day=1)
    boards.verdict(state, "p2", "seer", "p11", False, day=3, referenced_day=1)

    engine = _engine(state, "p7")
    conflict_evidence = [
        record
        for record in engine.active_evidence()
        if record.category == "timeline_conflict"
    ]

    assert conflict_evidence
    assert {record.subject_id for record in conflict_evidence} == {"p2"}


# -- evidence that outlived its fact --


def test_a_published_black_becomes_a_reason_to_suspect():
    state = _seer_board(day=2)
    boards.verdict(state, "p4", "seer", "p9", True, day=2, referenced_day=1)

    engine = _engine(state)

    assert engine.state.public_suspicion_scores.get("p9", 0.0) > 0
    assert result_fact_id("p4", "seer", "p9", True) in {
        record.origin.fact_id
        for record in engine.active_evidence()
        if record.origin is not None
    }


def test_retracting_the_verdict_retracts_the_reason_and_moves_the_score():
    state = _seer_board(day=2)
    boards.verdict(state, "p4", "seer", "p9", True, day=2, referenced_day=1)
    engine = _engine(state)
    before = engine.state.public_suspicion_scores.get("p9", 0.0)

    boards.retract_verdict(state, "p4", "seer", "p9", day=2)
    observations = ObservationSet.from_state(state)
    engine.observe(
        PublicFactLedger(state),
        build_solver(observations, PlayerPrivatePerspective("p2")),
        observations,
    )

    after = engine.state.public_suspicion_scores.get("p9", 0.0)
    assert before > 0
    assert after < before
    assert not [
        record
        for record in engine.active_evidence()
        if record.category == "published_black" and record.subject_id == "p9"
    ]


def test_a_correction_retires_the_old_colour_and_installs_the_new_one():
    state = _seer_board(day=2)
    boards.verdict(state, "p4", "seer", "p9", True, day=2, referenced_day=1)
    engine = _engine(state)

    boards.correct_verdict(state, "p4", "seer", "p9", False, day=2, referenced_day=1)
    observations = ObservationSet.from_state(state)
    engine.observe(
        PublicFactLedger(state),
        build_solver(observations, PlayerPrivatePerspective("p2")),
        observations,
    )

    categories = {
        record.category
        for record in engine.active_evidence()
        if record.subject_id == "p9"
    }
    assert "published_white" in categories
    assert "published_black" not in categories
    # The withdrawn reason stays in the record; it is inactive, not deleted.
    assert any(
        record.category == "published_black" and not record.active
        for record in engine.evidence
    )


def test_a_withdrawn_co_takes_the_contested_claim_reason_with_it():
    state = _seer_board(day=2)
    boards.claim(state, "p2", RoleName.SEER, day=1)
    engine = _engine(state, "p7")
    assert [
        record
        for record in engine.active_evidence()
        if record.category == "contested_claim"
    ]

    boards.retract_claim(state, "p2", day=2)
    observations = ObservationSet.from_state(state)
    engine.observe(
        PublicFactLedger(state),
        build_solver(observations, PlayerPrivatePerspective("p7")),
        observations,
    )

    assert not [
        record
        for record in engine.active_evidence()
        if record.category == "contested_claim"
    ]


# -- how much a verdict weighs depends on who published it --


def test_a_distrusted_claimants_verdict_weighs_less_without_changing_sign():
    state = _seer_board(day=2)
    boards.verdict(state, "p4", "seer", "p9", True, day=2, referenced_day=1)

    trusting = _engine(state)
    doubting = BeliefEngine("p3", PlayerPrivatePerspective("p3"))
    doubting.state.source_trust["p4"] = -1.5
    observations = ObservationSet.from_state(state)
    doubting.observe(
        PublicFactLedger(state),
        build_solver(observations, PlayerPrivatePerspective("p3")),
        observations,
    )

    def black_weight(engine: BeliefEngine) -> float:
        return next(
            record.weight
            for record in engine.active_evidence()
            if record.category == "published_black" and record.subject_id == "p9"
        )

    assert 0 < black_weight(doubting) < black_weight(trusting)
    assert black_weight(trusting) == PUBLISHED_BLACK_WEIGHT


def test_distrust_never_turns_a_black_into_an_argument_for_innocence():
    state = _seer_board(day=2)
    boards.verdict(state, "p4", "seer", "p9", True, day=2, referenced_day=1)

    engine = BeliefEngine("p3", PlayerPrivatePerspective("p3"))
    engine.state.source_trust["p4"] = -50.0
    observations = ObservationSet.from_state(state)
    engine.observe(
        PublicFactLedger(state),
        build_solver(observations, PlayerPrivatePerspective("p3")),
        observations,
    )

    weight = next(
        record.weight
        for record in engine.active_evidence()
        if record.category == "published_black"
    )
    assert weight > 0


# -- the bluffer's story reaches a real game --


def test_a_fake_claim_gets_a_public_story_without_anyone_passing_it_in():
    """The connection that was missing: nothing in production ever supplied
    `claimed_roles`, so a wolf claiming seer reasoned with its real card."""
    state = _seer_board(day=2)
    boards.claim(state, "p1", RoleName.SEER, day=1)  # p1 is the wolf
    runtime = ReasoningRuntime(state, ALL_SEATS, seed=3)

    runtime.refresh(state)

    deception = runtime.seats["p1"].deception
    assert deception is not None
    assert deception.is_bluffing
    assert deception.claimed_role is RoleName.SEER


def test_a_genuine_claim_gets_no_public_story():
    state = _seer_board(day=2)
    runtime = ReasoningRuntime(state, ALL_SEATS, seed=3)

    runtime.refresh(state)

    seer = runtime.seats["p4"].deception
    assert seer is None or not seer.is_bluffing


def test_the_public_story_still_cannot_see_the_wolf_team():
    state = _seer_board(day=2)
    boards.claim(state, "p1", RoleName.SEER, day=1)
    runtime = ReasoningRuntime(state, ALL_SEATS, seed=3)
    runtime.refresh(state)

    story = runtime.seats["p1"].deception
    assert story is not None and story.public_story is not None
    known = story.public_story.known_roles(ObservationSet.from_state(state))

    assert known == {"p1": RoleName.SEER}
