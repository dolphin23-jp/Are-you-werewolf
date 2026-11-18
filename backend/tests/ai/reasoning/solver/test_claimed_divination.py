"""What a seer claim commits its owner to on a night the curse is visible.

The first live Claude game turned on one public deduction the solver could not
make: two corpses after night 1, and ダイキ published a night-1 target (シオン)
who was alive. One of the two corpses was the cursed fox, so the real seer's
night-1 target is one of them -- ダイキ was not the real seer, from public
information alone. These tests pin that deduction and, as importantly, the
boards where it must *not* be drawn.
"""

from __future__ import annotations

from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import (
    ClaimedStoryPerspective,
    CommonPublicPerspective,
    PlayerPrivatePerspective,
    TrueWorldPerspective,
)
from app.ai.reasoning.solver import Hypothesis, RoleIs, build_solver, has_role
from app.ai.reasoning.solver.builder import ConstraintBuilder
from app.ai.reasoning.solver.rules import ClaimedDivinationRuleModule
from app.ai.reasoning.solver.rules.claimed_divination import seer_claim_refutations
from app.engine.roles import RoleName
from app.engine.state import GameState
from tests.ai.reasoning import game1_boards as g1
from tests.ai.reasoning.solver import boards


def _public(state: GameState):  # type: ignore[no-untyped-def]
    return build_solver(boards.observe(state), CommonPublicPerspective())


def _truth(observations: ObservationSet) -> Hypothesis:
    return Hypothesis(
        claims=tuple(
            RoleIs(player_id=pid, role=role)
            for pid, role in sorted(observations.true_roles.items())
        ),
        label="true world",
    )


def _module_constraints(state: GameState, perspective=None):  # type: ignore[no-untyped-def]
    builder = ConstraintBuilder()
    ClaimedDivinationRuleModule().add_hard_constraints(
        builder, perspective or CommonPublicPerspective(), boards.observe(state)
    )
    return builder.constraints


# -- the first game's day-2 morning --


def test_a_claimant_whose_curse_night_target_survived_cannot_be_the_real_seer():
    solver = _public(g1.day_two_morning())

    assert solver.is_possible(has_role(g1.DAIKI, RoleName.SEER)) is False


def test_the_other_claimants_stay_possible_seers():
    solver = _public(g1.day_two_morning())

    # アオイ claimed the corpse (her story is the cursed-fox one); ケント died
    # before publishing night 1, so nothing he said conflicts with the corpses.
    assert solver.is_possible(has_role(g1.AOI, RoleName.SEER)) is True
    assert solver.is_possible(has_role(g1.KENT, RoleName.SEER)) is True


def test_the_refutation_names_the_night_the_target_and_the_corpses():
    observations = boards.observe(g1.day_two_morning())

    refutations = seer_claim_refutations(observations)

    assert [(r.claimant_id, r.night, r.target_id, r.corpse_ids) for r in refutations] == [
        (g1.DAIKI, 1, g1.SHION, (g1.SOTA, g1.KENT))
    ]


def test_the_unsat_core_explains_the_refutation_in_the_modules_own_words():
    solver = _public(g1.day_two_morning())

    result = solver.explain_contradiction(has_role(g1.DAIKI, RoleName.SEER))

    assert result.is_contradictory
    assert any("シオン" in text or g1.SHION in text for text in result.explanations)
    assert any(cid.startswith("claimed_divination:") for cid in result.constraint_ids)
    assert any(cid.startswith("two_deaths_include_fox:") for cid in result.constraint_ids)


def test_the_true_world_survives_every_viewpoint():
    observations = boards.observe(g1.day_two_morning())
    truth = _truth(observations)

    assert build_solver(observations, CommonPublicPerspective()).is_possible(truth)
    assert build_solver(observations, TrueWorldPerspective()).is_possible(truth)
    for seat in observations.player_ids:
        solver = build_solver(observations, PlayerPrivatePerspective(seat))
        assert solver.is_possible(truth), f"{seat} rules out what actually happened"


def test_the_fox_is_publicly_dead_on_the_day_two_morning():
    solver = _public(g1.day_two_morning())
    alive = [pid for pid in solver.observations.player_ids if solver.observations.is_alive(pid)]

    for pid in alive:
        assert solver.is_possible(has_role(pid, RoleName.FOX)) is False, pid


def test_a_claimant_whose_night_zero_target_outlived_a_night_zero_curse_is_not_the_seer():
    state = boards.deal(
        {
            "p1": RoleName.WEREWOLF,
            "p2": RoleName.WEREWOLF,
            "p3": RoleName.WEREWOLF,
            "p4": RoleName.SEER,
            "p6": RoleName.FOX,
        }
    )
    boards.kill_first_victim(state, "p16")
    boards.divine(state, "p4", "p6", night=0)
    boards.die_by_curse(state, "p6", night=0)
    boards.claim(state, "p4", RoleName.SEER, day=1)
    boards.verdict(state, "p4", "seer", "p6", False, day=1, referenced_day=0)
    boards.claim(state, "p9", RoleName.SEER, day=1)
    boards.verdict(state, "p9", "seer", "p10", False, day=1, referenced_day=0)
    state.day = 1
    solver = _public(state)

    # p6 was cursed on night 0, so the real seer's night-0 target is p6: the
    # claimant who named a living p10 for night 0 is not the real seer.
    assert solver.is_possible(has_role("p9", RoleName.SEER)) is False
    assert solver.is_possible(has_role("p4", RoleName.SEER)) is True


# -- where the deduction must not be drawn --


def test_an_unpublished_curse_night_refutes_nobody():
    state = g1.day_two_morning()
    # Without his night-1 claim, ダイキ might have divined ケント (the fox in
    # that world) and simply not said so yet.
    state.speech_events[:] = [
        event
        for event in state.speech_events
        if not (event.actor_id == g1.DAIKI and event.target_id == g1.SHION)
    ]

    assert _public(state).is_possible(has_role(g1.DAIKI, RoleName.SEER)) is True


def test_a_single_corpse_night_settles_nothing_about_the_claimant():
    state = boards.deal({"p1": RoleName.WEREWOLF, "p4": RoleName.SEER, "p6": RoleName.FOX})
    boards.kill_first_victim(state, "p16")
    boards.claim(state, "p5", RoleName.SEER, day=1)
    boards.verdict(state, "p5", "seer", "p8", False, day=2, referenced_day=1)
    boards.die_by_attack(state, "p9", night=1)
    state.day = 2

    solver = _public(state)
    real = has_role("p5", RoleName.SEER)

    # One corpse is an attack or a curse; p5 may be real, with p8 and p9 simply
    # not the fox in that world.
    assert solver.is_possible(real) is True
    assert solver.is_possible(real & has_role("p9", RoleName.FOX)) is False
    assert solver.is_possible(real & has_role("p8", RoleName.FOX)) is False
    # Without p5 being real, either could still be the fox.
    assert solver.is_possible(has_role("p9", RoleName.FOX)) is True
    assert solver.is_possible(has_role("p8", RoleName.FOX)) is True


def test_claiming_the_corpse_is_consistent_with_being_real():
    state = boards.deal({"p1": RoleName.WEREWOLF, "p4": RoleName.SEER, "p6": RoleName.FOX})
    boards.kill_first_victim(state, "p16")
    boards.claim(state, "p4", RoleName.SEER, day=1)
    boards.divine(state, "p4", "p6", night=1)
    boards.verdict(state, "p4", "seer", "p6", False, day=2, referenced_day=1)
    boards.die_by_curse(state, "p6", night=1)
    boards.die_by_attack(state, "p9", night=1)
    state.day = 2

    solver = _public(state)
    real = has_role("p4", RoleName.SEER)

    # p4 real means p6 was the cursed one, so the other corpse is not the fox.
    assert solver.is_possible(real) is True
    assert solver.is_possible(real & has_role("p9", RoleName.FOX)) is False


def test_a_verdict_with_no_stated_night_is_not_dated_by_guesswork():
    state = g1.day_two_morning()
    boards.verdict(state, g1.YUI, "seer", g1.SAKURA, False, day=2)  # no night given

    constraints = _module_constraints(state)

    assert not any(c.constraint_id.startswith(f"claimed_divination:{g1.YUI}:") for c in constraints)


def test_an_impossible_schedule_is_left_to_the_timeline_check():
    state = g1.day_two_morning()
    # ノゾミ died on night 0, so "divined ノゾミ on night 1" is a timeline
    # conflict, not a curse fact; the premise the rule rests on is already void.
    boards.verdict(state, g1.MIO, "seer", g1.NOZOMI, False, day=2, referenced_day=1)

    constraints = _module_constraints(state)

    assert not any(c.constraint_id.startswith(f"claimed_divination:{g1.MIO}:") for c in constraints)


def test_a_restated_verdict_adds_no_new_constraint():
    state = g1.day_two_morning()
    before = sorted(c.constraint_id for c in _module_constraints(state))
    boards.verdict(state, g1.DAIKI, "seer", g1.SHION, False, day=3, referenced_day=1)

    after = sorted(c.constraint_id for c in _module_constraints(state))

    assert after == before


# -- whose claims a viewpoint reasons about --


def test_a_seat_never_applies_the_rule_to_its_own_claims():
    state = g1.day_two_morning()

    own = _module_constraints(state, PlayerPrivatePerspective(g1.DAIKI))

    assert own
    assert not any(c.constraint_id.startswith(f"claimed_divination:{g1.DAIKI}:") for c in own)


def test_a_real_seer_who_misreported_their_target_keeps_their_own_world():
    state = boards.deal({"p1": RoleName.WEREWOLF, "p4": RoleName.SEER, "p6": RoleName.FOX})
    boards.kill_first_victim(state, "p16")
    boards.claim(state, "p4", RoleName.SEER, day=1)
    boards.divine(state, "p4", "p6", night=1)
    boards.die_by_curse(state, "p6", night=1)
    boards.die_by_attack(state, "p9", night=1)
    # The real seer names someone else for night 1. The table may now rule them
    # out -- that is what their account says -- but their own seat knows better.
    boards.verdict(state, "p4", "seer", "p8", False, day=2, referenced_day=1)
    state.day = 2
    observations = boards.observe(state)

    seer = build_solver(observations, PlayerPrivatePerspective("p4"))

    assert seer.is_possible(_truth(observations)) is True
    assert _public(state).is_possible(has_role("p4", RoleName.SEER)) is False


def test_a_bluffers_story_check_is_unchanged():
    state = g1.day_two_morning()
    observations = boards.observe(state)

    story = build_solver(observations, ClaimedStoryPerspective(g1.DAIKI, RoleName.SEER))

    # ダイキ's own story ("I am the seer and looked at シオン") still collapses
    # against the corpses, through the curse rule on the story's own divines.
    assert story.is_possible(Hypothesis()) is False


# -- the logic layer's standing checks (docs/real_game_log_reasoning_plan.md) --


def _curse_night_board(claimant: str, target: str, corpses: tuple[str, str]) -> GameState:
    state = boards.deal({"p1": RoleName.WEREWOLF, "p4": RoleName.SEER, "p6": RoleName.FOX})
    boards.kill_first_victim(state, "p16")
    boards.claim(state, claimant, RoleName.SEER, day=1)
    boards.verdict(state, claimant, "seer", target, False, day=2, referenced_day=1)
    boards.die_by_curse(state, corpses[0], night=1)
    boards.die_by_attack(state, corpses[1], night=1)
    state.day = 2
    return state


def test_the_same_board_under_other_seats_gives_the_same_answer():
    first = _public(_curse_night_board("p5", "p8", ("p6", "p9")))
    relabelled = _public(_curse_night_board("p11", "p13", ("p6", "p10")))

    assert first.is_possible(has_role("p5", RoleName.SEER)) is False
    assert relabelled.is_possible(has_role("p11", RoleName.SEER)) is False
    assert first.is_possible(has_role("p8", RoleName.SEER)) is True
    assert relabelled.is_possible(has_role("p13", RoleName.SEER)) is True


def test_unrelated_events_do_not_move_the_conclusion():
    state = g1.day_two_morning()
    before = sorted(c.constraint_id for c in _module_constraints(state))
    boards.claim(state, g1.MIO, RoleName.HUNTER, day=2)
    boards.verdict(state, g1.AKARI, "medium", g1.HARUTO, True, day=2, referenced_day=1)

    assert sorted(c.constraint_id for c in _module_constraints(state)) == before
    assert _public(state).is_possible(has_role(g1.DAIKI, RoleName.SEER)) is False
