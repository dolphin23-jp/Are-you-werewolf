"""Night events read differently from every seat, which is the whole point.

Two causes of death overnight -- one attack, one curse -- and the public death
notice collapses them. Everything the table can work out from a corpse follows
from that ambiguity; everything the wolves, hunter and seer can work out follows
from knowing which action was theirs.
"""

from __future__ import annotations

import json

from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import (
    CommonPublicPerspective,
    PlayerPrivatePerspective,
    TrueWorldPerspective,
)
from app.ai.reasoning.solver import Hypothesis, RoleIs, build_solver, has_role, not_role
from app.ai.reasoning.solver.builder import ConstraintBuilder
from app.ai.reasoning.solver.rules import (
    AttackRuleModule,
    DeathRuleModule,
    FoxCurseRuleModule,
    GuardRuleModule,
)
from app.engine.roles import RoleName
from tests.ai.reasoning.solver import boards


def _solver(state, perspective):  # type: ignore[no-untyped-def]
    return build_solver(boards.observe(state), perspective)


def _public(state):  # type: ignore[no-untyped-def]
    return _solver(state, CommonPublicPerspective())


def _wolf_board(**extra: RoleName):  # type: ignore[no-untyped-def]
    assignment = {
        "p1": RoleName.WEREWOLF,
        "p2": RoleName.WEREWOLF,
        "p3": RoleName.WEREWOLF,
        "p4": RoleName.SEER,
        "p5": RoleName.HUNTER,
        "p6": RoleName.FOX,
    }
    assignment.update(extra)
    return boards.deal(assignment, day=2)


# -- the table cannot tell the causes apart --


def test_one_night_death_leaves_both_causes_open():
    state = _wolf_board()
    boards.die_by_attack(state, "p9", night=1)
    solver = _public(state)

    # Attacked or cursed: the corpse looks the same either way, so nothing about
    # p9's role is settled.
    assert solver.certain_role("p9") is None
    assert solver.is_possible(has_role("p9", RoleName.FOX)) is True
    assert solver.is_possible(has_role("p9", RoleName.VILLAGER)) is True


def test_two_deaths_in_one_night_put_the_fox_among_them():
    state = _wolf_board()
    boards.die_by_curse(state, "p6", night=1)
    boards.die_by_attack(state, "p9", night=1)
    solver = _public(state)

    # One attack and one curse per night, so exactly one of the pair was cursed
    # -- publicly derivable, and neither seat is named.
    either = Hypothesis(
        claims=(RoleIs("p6", RoleName.FOX),), label="p6 is the fox"
    )
    assert solver.is_forced(either) is False
    assert solver.is_possible(either) is True
    assert solver.is_possible(
        Hypothesis(claims=(RoleIs("p9", RoleName.FOX),))
    ) is True
    # But the fox cannot be anybody else.
    assert solver.is_possible(has_role("p12", RoleName.FOX)) is False


def test_a_quiet_night_proves_neither_a_guard_nor_a_fox_bite():
    state = _wolf_board()
    boards.attack(state, "p1", "p6", night=1, succeeded=False)
    solver = _public(state)

    # No corpse can mean a guarded target or a bitten fox. The table sees the
    # absence and must keep both alive as explanations.
    assert solver.observations.deaths_on(1) == ()
    assert solver.is_possible(has_role("p5", RoleName.HUNTER)) is True
    assert solver.is_possible(has_role("p6", RoleName.FOX)) is True
    assert solver.certain_role("p6") is None


# -- the wolves know where they struck --


def test_a_wolf_reads_the_second_corpse_as_the_fox():
    state = _wolf_board()
    boards.attack(state, "p1", "p9", night=1, succeeded=True)
    boards.die_by_attack(state, "p9", night=1)
    boards.die_by_curse(state, "p6", night=1)

    wolf = _solver(state, PlayerPrivatePerspective("p1"))

    # Their own kill explains p9; nothing but the curse explains p6.
    assert wolf.is_forced(has_role("p6", RoleName.FOX)) is True
    assert wolf.certain_role("p6") is RoleName.FOX
    # The table, lacking the attack target, can only say "one of the two".
    assert _public(state).is_forced(has_role("p6", RoleName.FOX)) is False


def test_a_wolfs_own_target_stays_ambiguous_to_them():
    state = _wolf_board()
    boards.attack(state, "p1", "p6", night=1, succeeded=False)
    boards.die_by_curse(state, "p6", night=1)

    wolf = _solver(state, PlayerPrivatePerspective("p1"))

    # p6 was both bitten and divined that night. The wolf cannot distinguish
    # "my attack landed" from "the seer got there first", so no conclusion.
    assert wolf.is_forced(has_role("p6", RoleName.FOX)) is False
    assert wolf.is_possible(has_role("p6", RoleName.FOX)) is True


def test_the_attack_target_never_reaches_the_public_view():
    state = _wolf_board()
    boards.attack(state, "p1", "p9", night=1, succeeded=True)
    boards.die_by_attack(state, "p9", night=1)
    boards.die_by_curse(state, "p6", night=1)
    observations = boards.observe(state)

    knowledge = CommonPublicPerspective().known_night_actions(observations)

    assert knowledge.is_empty is True
    assert knowledge.attacks == {}


def test_a_teammate_cannot_be_the_unattacked_corpse():
    state = _wolf_board()
    boards.attack(state, "p1", "p9", night=1, succeeded=True)
    boards.die_by_attack(state, "p9", night=1)
    boards.die_by_curse(state, "p6", night=1)

    wolf = _solver(state, PlayerPrivatePerspective("p1"))

    # The corpse the wolves did not target must be the fox, so it cannot be one
    # of them: an attack alone never kills a wolf.
    assert wolf.is_possible(has_role("p6", RoleName.WEREWOLF)) is False


# -- the hunter knows who they covered --


def test_a_guarded_seat_that_died_anyway_must_be_the_fox():
    state = _wolf_board()
    boards.guard(state, "p5", "p6", night=1)
    boards.die_by_curse(state, "p6", night=1)

    hunter = _solver(state, PlayerPrivatePerspective("p5"))

    assert hunter.is_forced(has_role("p6", RoleName.FOX)) is True


def test_a_guarded_death_makes_any_other_role_for_that_seat_impossible():
    state = _wolf_board()
    boards.guard(state, "p5", "p6", night=1)
    boards.die_by_curse(state, "p6", night=1)
    hunter = _solver(state, PlayerPrivatePerspective("p5"))

    # The hunter can rule the impossible reading out explicitly, which is what
    # "my guard failed and they were not the fox" would have to be.
    assert hunter.is_possible(not_role("p6", RoleName.FOX)) is False
    assert hunter.explain_contradiction(not_role("p6", RoleName.FOX)).is_contradictory


def test_the_guard_target_never_reaches_the_public_view():
    state = _wolf_board()
    boards.guard(state, "p5", "p6", night=1)
    boards.die_by_curse(state, "p6", night=1)
    observations = boards.observe(state)

    assert CommonPublicPerspective().known_night_actions(observations).guards == {}
    # And the deduction it enables does not happen without it.
    assert _public(state).is_forced(has_role("p6", RoleName.FOX)) is False


# -- the seer knows where they looked --


def test_a_divined_survivor_is_not_the_fox():
    state = _wolf_board()
    boards.divine(state, "p4", "p9", night=1)

    seer = _solver(state, PlayerPrivatePerspective("p4"))

    # A divined fox always dies that night, so surviving settles it.
    assert seer.is_possible(has_role("p9", RoleName.FOX)) is False
    assert _public(state).is_possible(has_role("p9", RoleName.FOX)) is True


def test_a_divined_fox_dies_that_night():
    state = _wolf_board()
    boards.divine(state, "p4", "p6", night=1)
    boards.die_by_curse(state, "p6", night=1)

    seer = _solver(state, PlayerPrivatePerspective("p4"))

    # Still consistent with the fox being there, unlike every other seat the
    # seer looked at and left alive.
    assert seer.is_possible(has_role("p6", RoleName.FOX)) is True


def test_a_corpse_the_seer_did_not_look_at_is_not_the_fox():
    state = _wolf_board()
    boards.divine(state, "p4", "p12", night=1)
    boards.die_by_attack(state, "p9", night=1)

    seer = _solver(state, PlayerPrivatePerspective("p4"))

    # Nothing cursed p9, and an attack cannot kill a fox.
    assert seer.is_possible(has_role("p9", RoleName.FOX)) is False
    assert seer.is_possible(has_role("p12", RoleName.FOX)) is False


def test_a_fox_survives_an_ordinary_bite():
    state = _wolf_board()
    boards.attack(state, "p1", "p6", night=1, succeeded=False)
    boards.divine(state, "p4", "p9", night=1)

    wolf = _solver(state, PlayerPrivatePerspective("p1"))

    # No corpse despite a bite: the fox is one live explanation, and the wolves
    # cannot rule it out from their side.
    assert wolf.observations.deaths_on(1) == ()
    assert wolf.is_possible(has_role("p6", RoleName.FOX)) is True


# -- what the rules must not conclude --


def test_a_bitten_seer_claimant_is_not_forced_to_be_real():
    state = _wolf_board()
    boards.claim(state, "p9", RoleName.SEER, day=1)
    boards.die_by_attack(state, "p9", night=1)
    solver = _public(state)

    # "The wolves bit the seer claimant, so they were real or the madman" is
    # sound play and terrible logic. It stays out of the hard rules.
    assert solver.is_possible(has_role("p9", RoleName.VILLAGER)) is True
    assert solver.is_possible(has_role("p9", RoleName.SEER)) is True
    assert solver.is_possible(has_role("p9", RoleName.MADMAN)) is True


def test_claiming_a_village_role_you_do_not_hold_stays_possible():
    state = _wolf_board()
    boards.claim(state, "p9", RoleName.HUNTER, day=1)
    solver = _public(state)

    # No rule forbids a villager claiming hunter, so the solver must not invent
    # one -- only the wolf and fox cards are structurally excluded anywhere.
    for role in (RoleName.VILLAGER, RoleName.HUNTER, RoleName.MEDIUM, RoleName.MADMAN):
        assert solver.is_possible(has_role("p9", role)) is True


def test_the_real_world_stays_consistent_with_every_private_view():
    state = _wolf_board()
    boards.divine(state, "p4", "p12", night=1)
    boards.guard(state, "p5", "p9", night=1)
    boards.attack(state, "p1", "p13", night=1, succeeded=True)
    boards.die_by_attack(state, "p13", night=1)
    observations = boards.observe(state)
    truth = Hypothesis(
        claims=tuple(
            RoleIs(player_id=pid, role=role)
            for pid, role in sorted(observations.true_roles.items())
        ),
        label="true world",
    )

    for seat in observations.player_ids:
        solver = build_solver(observations, PlayerPrivatePerspective(seat))
        assert solver.is_possible(truth) is True, f"{seat} rules out what actually happened"


def test_night_knowledge_is_scoped_to_the_seat_that_acted():
    state = _wolf_board()
    boards.divine(state, "p4", "p12", night=1)
    boards.guard(state, "p5", "p9", night=1)
    boards.attack(state, "p1", "p13", night=1, succeeded=True)
    observations = boards.observe(state)

    seer = PlayerPrivatePerspective("p4").known_night_actions(observations)
    hunter = PlayerPrivatePerspective("p5").known_night_actions(observations)
    wolf = PlayerPrivatePerspective("p2").known_night_actions(observations)
    villager = PlayerPrivatePerspective("p12").known_night_actions(observations)

    assert (seer.divines, seer.guards, seer.attacks) == ({1: "p12"}, {}, {})
    assert (hunter.divines, hunter.guards, hunter.attacks) == ({}, {1: "p9"}, {})
    # Any wolf may reason from the team's attack, not only the alpha who sent it.
    assert wolf.attacks == {1: "p13"}
    assert villager.is_empty is True


def test_no_night_action_leaks_through_the_public_projection():
    state = _wolf_board()
    boards.divine(state, "p4", "p12", night=1)
    boards.guard(state, "p5", "p9", night=1)
    boards.attack(state, "p1", "p13", night=1, succeeded=True)
    boards.die_by_attack(state, "p13", night=1)
    observations = boards.observe(state)

    public_constraints = []
    for module in (
        DeathRuleModule(),
        AttackRuleModule(),
        GuardRuleModule(),
        FoxCurseRuleModule(),
    ):
        builder = ConstraintBuilder()
        module.add_hard_constraints(builder, CommonPublicPerspective(), observations)
        public_constraints.extend(builder.constraints)

    # A single night death and no viewpoint knowledge: the public modules have
    # nothing to say, and in particular say nothing about p9 or p13.
    assert public_constraints == []


def test_the_debug_view_sees_every_night_action():
    state = _wolf_board()
    boards.divine(state, "p4", "p12", night=1)
    boards.guard(state, "p5", "p9", night=1)
    boards.attack(state, "p1", "p13", night=1, succeeded=True)
    observations = boards.observe(state)

    knowledge = TrueWorldPerspective().known_night_actions(observations)

    assert knowledge.divines == {1: "p12"}
    assert knowledge.guards == {1: "p9"}
    assert knowledge.attacks == {1: "p13"}


def test_soft_evidence_is_defined_but_deliberately_empty():
    state = _wolf_board()
    boards.attack(state, "p1", "p9", night=1, succeeded=True)
    boards.die_by_attack(state, "p9", night=1)
    observations = boards.observe(state)

    for module in (DeathRuleModule(), AttackRuleModule(), GuardRuleModule()):
        assert (
            module.extract_soft_evidence(PlayerPrivatePerspective("p1"), observations) == []
        )


def test_night_events_change_the_board_version():
    state = _wolf_board()
    before = ObservationSet.from_state(state).board_version
    boards.die_by_attack(state, "p9", night=1)
    after = ObservationSet.from_state(state).board_version

    # Cached answers from before the corpse must not be reused after it.
    assert before != after
    assert len(json.dumps({"before": before, "after": after})) > 0
