"""Knowing who the wolves are is not the same as wanting them executed.

The bug this file pins down: a werewolf's private certainty about its partners
was read by a village-shaped utility function, so the wolves opened every game
by nominating each other. Certainty, public suspicion and preference are now
three separate things, and these tests hold them apart.
"""

from __future__ import annotations

import pytest

from app.ai.reasoning.belief import (
    BeliefEngine,
    EvidenceRecord,
    RoleCertainty,
    UtilityInputs,
    attack_utility,
    divine_utility,
    execution_utility,
    guard_utility,
)
from app.ai.reasoning.belief.utility import ALLY_PROTECTION
from app.ai.reasoning.facts import PublicFactLedger
from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import PlayerPrivatePerspective
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.reasoning.solver import build_solver, has_role
from app.engine.roles import RoleName
from app.engine.state import MediumRecord
from tests.ai.reasoning.solver import boards

ALL_SEATS = [f"p{i}" for i in range(17)]


def _wolf_board(day: int = 1):  # type: ignore[no-untyped-def]
    return boards.deal(
        {
            "p1": RoleName.WEREWOLF,
            "p2": RoleName.WEREWOLF,
            "p3": RoleName.WEREWOLF,
            "p4": RoleName.SEER,
            "p5": RoleName.MEDIUM,
            "p6": RoleName.HUNTER,
            "p7": RoleName.FOX,
            "p8": RoleName.MADMAN,
        },
        day=day,
    )


def _engine(state, player_id: str) -> BeliefEngine:  # type: ignore[no-untyped-def]
    observations = ObservationSet.from_state(state)
    perspective = PlayerPrivatePerspective(player_id)
    engine = BeliefEngine(player_id, perspective)
    engine.observe(
        PublicFactLedger(state), build_solver(observations, perspective), observations
    )
    return engine


def _runtime(state, seed: int = 3) -> ReasoningRuntime:  # type: ignore[no-untyped-def]
    runtime = ReasoningRuntime(state, ALL_SEATS[1:], seed=seed)
    runtime.refresh(state)
    return runtime


# -- the reproduction --


def test_a_wolf_does_not_nominate_a_teammate_on_an_empty_board():
    state = _wolf_board()
    engine = _engine(state, "p1")

    # It still knows exactly who they are -- that knowledge is correct and must
    # not be thrown away to fix the vote.
    assert engine.state.certainty_of("p2") is RoleCertainty.CONFIRMED
    assert engine.state.certainty_of("p3") is RoleCertainty.CONFIRMED
    assert engine.state.current_execution_target not in ("p1", "p2", "p3")


def test_a_wolf_votes_outside_its_own_team():
    state = _wolf_board()
    runtime = _runtime(state)

    for wolf in ("p1", "p2", "p3"):
        target, _reason = runtime.vote_decision(wolf, ALL_SEATS[1:])
        assert target not in ("p1", "p2", "p3"), f"{wolf} voted for a teammate"


def test_betrayal_stays_reachable_when_it_actually_pays():
    inputs = UtilityInputs(
        actor_id="p1",
        actor_role=RoleName.WEREWOLF,
        ally_ids=frozenset({"p2", "p3"}),
        wolf_certainty={"p2": RoleCertainty.CONFIRMED},
        # The table is already convinced about p2 by a wide margin.
        public_suspicion={"p2": 40.0, "p9": 1.0},
        fox_suspicion={},
        claim_trust={},
        claimed_roles={},
        alive_ids=frozenset(ALL_SEATS),
    )

    # Priced, not forbidden: enough public pressure and the wolf goes along.
    assert execution_utility(inputs, "p2") > execution_utility(inputs, "p9")
    assert ALLY_PROTECTION > 0


def test_a_villager_still_wants_the_publicly_suspicious_seat_executed():
    state = _wolf_board(day=2)
    boards.claim(state, "p4", RoleName.SEER, day=1)
    boards.verdict(state, "p4", "seer", "p9", True, day=1)
    engine = _engine(state, "p10")

    assert engine.state.current_execution_target == "p9"


def test_a_madman_pushes_at_the_seats_the_table_trusts():
    inputs = UtilityInputs(
        actor_id="p8",
        actor_role=RoleName.MADMAN,
        ally_ids=frozenset(),
        wolf_certainty={},
        public_suspicion={"p9": 3.0, "p10": 0.0},
        fox_suspicion={},
        claim_trust={"p10": 1.0},
        claimed_roles={"p10": RoleName.SEER},
        alive_ids=frozenset(ALL_SEATS),
    )

    # A misexecution is the job, so the trusted seat is worth more than the
    # one the village is already about to hang.
    assert execution_utility(inputs, "p10") > execution_utility(inputs, "p9")


# -- who knows what --


def test_a_madman_is_never_told_where_the_wolves_are():
    state = _wolf_board()
    engine = _engine(state, "p8")

    for wolf in ("p1", "p2", "p3"):
        assert engine.state.certainty_of(wolf) is not RoleCertainty.CONFIRMED


def test_a_fox_has_no_allies_and_no_confirmed_wolves():
    state = _wolf_board()
    engine = _engine(state, "p7")

    assert all(
        engine.state.certainty_of(pid) is not RoleCertainty.CONFIRMED
        for pid in ALL_SEATS
        if pid != "p7"
    )


def test_a_villager_does_not_know_the_real_wolf_seats():
    state = _wolf_board()
    engine = _engine(state, "p10")

    for wolf in ("p1", "p2", "p3"):
        assert engine.state.certainty_of(wolf) is RoleCertainty.UNKNOWN


# -- private ability results --


def test_a_true_seer_treats_their_own_black_as_settled():
    state = _wolf_board(day=2)
    boards.divine(state, "p4", "p2", night=1)
    engine = _engine(state, "p4")

    # Nothing has been published; the seer alone holds it.
    assert engine.state.certainty_of("p2") is RoleCertainty.CONFIRMED
    assert engine.state.current_execution_target == "p2"


def test_an_unpublished_result_reaches_nobody_else():
    state = _wolf_board(day=2)
    boards.divine(state, "p4", "p2", night=1)

    for seat in ("p9", "p10", "p8", "p7"):
        engine = _engine(state, seat)
        assert engine.state.certainty_of("p2") is RoleCertainty.UNKNOWN
    observations = ObservationSet.from_state(state)
    assert PlayerPrivatePerspective("p9").known_divine_results(observations) == ()


def test_a_seers_white_rules_out_a_wolf_and_nothing_more():
    state = _wolf_board(day=2)
    boards.divine(state, "p4", "p8", night=1)  # the madman divines white
    observations = ObservationSet.from_state(state)
    perspective = PlayerPrivatePerspective("p4")
    solver = build_solver(observations, perspective)

    assert solver.is_possible(has_role("p8", RoleName.WEREWOLF)) is False
    # White means "not a werewolf", not "villager": the madman is on the wolf
    # team and divines white, and treating white as village-confirmed would be
    # the single most damaging mistake available here.
    assert solver.is_possible(has_role("p8", RoleName.MADMAN)) is True
    assert solver.is_possible(has_role("p8", RoleName.VILLAGER)) is True


def test_a_true_medium_treats_their_own_result_as_settled():
    state = _wolf_board(day=2)
    boards.execute(state, "p2", day=1)
    state.medium_records.append(
        MediumRecord(medium_id="p5", target_id="p2", day=1, is_werewolf=True)
    )
    engine = _engine(state, "p5")

    assert engine.state.certainty_of("p2") is RoleCertainty.CONFIRMED


def test_publishing_a_result_does_not_cost_its_owner_the_certainty():
    state = _wolf_board(day=2)
    boards.divine(state, "p4", "p2", night=1)
    boards.claim(state, "p4", RoleName.SEER, day=2)
    boards.verdict(state, "p4", "seer", "p2", True, day=2)
    engine = _engine(state, "p4")

    assert engine.state.certainty_of("p2") is RoleCertainty.CONFIRMED


# -- night actions have their own preferences --


def test_a_seer_does_not_look_at_the_same_seat_twice():
    state = _wolf_board(day=3)
    boards.divine(state, "p4", "p9", night=1)
    boards.divine(state, "p4", "p10", night=2)
    runtime = _runtime(state)

    target = runtime.night_target("p4", "divine", [f"p{i}" for i in range(9, 16)])

    assert target not in ("p9", "p10")


def test_a_hunter_covers_a_power_role_rather_than_the_least_suspicious_seat():
    state = _wolf_board(day=2)
    boards.claim(state, "p4", RoleName.SEER, day=1)
    boards.verdict(state, "p4", "seer", "p9", True, day=1)
    runtime = _runtime(state)

    target = runtime.night_target("p6", "guard", ["p4", "p11", "p12", "p13"])

    assert target == "p4"


def test_wolves_bite_a_threat_and_never_a_teammate():
    state = _wolf_board(day=2)
    boards.claim(state, "p4", RoleName.SEER, day=1)
    boards.verdict(state, "p4", "seer", "p9", True, day=1)
    runtime = _runtime(state)

    target = runtime.night_target("p1", "attack", ["p2", "p3", "p4", "p11"])

    assert target == "p4"
    assert target not in ("p2", "p3")


def test_attack_does_not_simply_reuse_public_suspicion():
    inputs = UtilityInputs(
        actor_id="p1",
        actor_role=RoleName.WEREWOLF,
        ally_ids=frozenset({"p2"}),
        wolf_certainty={},
        # The table's favourite suspect is the wolves' best cover.
        public_suspicion={"p9": 5.0, "p4": 0.0},
        fox_suspicion={},
        claim_trust={"p4": 1.0},
        claimed_roles={"p4": RoleName.SEER},
        alive_ids=frozenset(ALL_SEATS),
    )

    assert attack_utility(inputs, "p4") > attack_utility(inputs, "p9")
    assert attack_utility(inputs, "p2") == float("-inf")


def test_the_four_utilities_disagree_with_each_other():
    inputs = UtilityInputs(
        actor_id="p1",
        actor_role=RoleName.WEREWOLF,
        ally_ids=frozenset({"p2"}),
        wolf_certainty={"p2": RoleCertainty.CONFIRMED},
        public_suspicion={"p9": 4.0, "p4": 0.0},
        fox_suspicion={"p7": 2.0},
        claim_trust={"p4": 1.0},
        claimed_roles={"p4": RoleName.SEER},
        alive_ids=frozenset(ALL_SEATS),
        already_divined=frozenset({"p9"}),
    )

    # One number could not have produced all four of these.
    assert execution_utility(inputs, "p2") < 0
    assert divine_utility(inputs, "p9") == float("-inf")
    assert guard_utility(inputs, "p1") == float("-inf")
    assert attack_utility(inputs, "p2") == float("-inf")


def test_night_targets_never_include_the_dead():
    state = _wolf_board(day=2)
    boards.execute(state, "p9", day=1)
    runtime = _runtime(state)
    alive = [pid for pid in ALL_SEATS[1:] if state.players[pid].alive]

    for action in ("divine", "guard", "attack"):
        target = runtime.night_target("p4", action, alive)
        assert target != "p9"


# -- the hard/soft boundary survives all of it --


def test_ordinary_soft_evidence_cannot_make_a_wolf_nominate_a_partner():
    state = _wolf_board()
    engine = _engine(state, "p1")
    engine.add_evidence(
        EvidenceRecord(
            evidence_id="hunch:p2",
            subject_id="p2",
            category="misremembered_vote",
            source_event_ids=("hunch",),
            weight=8.0,
            explanation="やや強い違和感。",
        )
    )
    engine.recompute(PublicFactLedger(state))

    # The suspicion is registered and still loses to protecting the team. Only a
    # genuinely overwhelming public case clears `ALLY_PROTECTION` -- which is
    # the point: betrayal is priced, not forbidden.
    assert engine.state.public_suspicion_scores["p2"] > 0
    assert engine.state.current_execution_target != "p2"


@pytest.mark.parametrize("engine_value", ["legacy", "v2", " V2 "])
def test_the_engine_setting_accepts_only_the_two_real_values(engine_value: str):
    from app.config import Settings

    assert Settings(werewolf_reasoning_engine=engine_value).werewolf_reasoning_engine in (
        "legacy",
        "v2",
    )


def test_an_unknown_engine_value_fails_at_startup():
    import pydantic

    from app.config import Settings

    # Falling through to legacy in silence meant a deployment that asked for v2
    # quietly ran the old engine and looked like v2 had changed nothing.
    with pytest.raises(pydantic.ValidationError):
        Settings(werewolf_reasoning_engine="v3")
    assert Settings(werewolf_reasoning_engine="").werewolf_reasoning_engine == "legacy"
