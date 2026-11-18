"""The game still running, as a solver rule (run #14, X1).

Every victory check the engine ran and passed says the living held at least
one wolf and fewer wolves than non-wolves. The rule has to be exact: never
stronger than the engine (the real world must always satisfy it), and strong
enough that an account leaving no living wolf is impossible.
"""

from __future__ import annotations

from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import CommonPublicPerspective, TrueWorldPerspective
from app.ai.reasoning.solver import (
    GenuineClaim,
    HonestResults,
    Hypothesis,
    RoleCountAmong,
    build_solver,
    has_role,
)
from app.ai.reasoning.solver.rules import (
    DERIVED_RULE_MODULE_IDS,
    GameRunningRuleModule,
    default_rule_modules,
    plain_rule_modules,
)
from app.ai.reasoning.solver.rules.game_running import passed_victory_checks, victory_checks
from app.engine.roles import RoleName, Team
from tests.ai.reasoning import run14_boards as r14
from tests.ai.reasoning.solver import boards


def test_the_checks_are_the_ones_the_engine_ran():
    """Seed 1, day 6: a dawn and an execution each day, the first victim at
    the first dawn."""
    observations = ObservationSet.from_state(r14.board(1, 6))
    checks = victory_checks(observations)

    assert [(check.label, len(check.alive)) for check in checks] == [
        ("1日目の朝", 16),
        ("1日目の処刑後", 15),
        ("2日目の朝", 14),
        ("2日目の処刑後", 13),
        ("3日目の朝", 12),
        ("3日目の処刑後", 11),
        ("4日目の朝", 10),
        ("4日目の処刑後", 9),
        ("5日目の朝", 8),
        ("5日目の処刑後", 7),
        ("6日目の朝", 6),
    ]
    assert checks[-1].alive == tuple(
        pid for pid in observations.player_ids if observations.is_alive(pid)
    )


def test_the_real_world_always_satisfies_the_rule():
    """At every moment of the three run #14 games, all roles known: the rule
    never says more than the engine checked."""
    for seed in (1, 2, 3):
        for day in range(1, 7):
            for at in ("morning", "vote", "night"):
                observations = ObservationSet.from_state(r14.board(seed, day, at))
                solver = build_solver(observations, TrueWorldPerspective())
                assert solver.is_possible(Hypothesis()), (seed, day, at)


def test_a_finished_game_drops_the_check_that_ended_it():
    state = r14.board(1, 6, "night")  # ダイキ, the last wolf, hanged
    state.winner = Team.VILLAGE
    observations = ObservationSet.from_state(state)

    assert len(passed_victory_checks(observations)) == len(victory_checks(observations)) - 1
    solver = build_solver(observations, TrueWorldPerspective())
    assert solver.is_possible(Hypothesis())


def test_a_running_game_has_a_living_wolf_and_fewer_wolves_than_the_rest():
    observations = ObservationSet.from_state(r14.board(3, 6))
    solver = build_solver(observations, CommonPublicPerspective())
    alive = tuple(pid for pid in observations.player_ids if observations.is_alive(pid))

    def living_wolves(count: int) -> Hypothesis:
        return Hypothesis(
            claims=(RoleCountAmong(alive, RoleName.WEREWOLF, at_least=count, at_most=count),)
        )

    assert len(alive) == 6
    assert not solver.is_possible(living_wolves(0))
    assert solver.is_possible(living_wolves(1))
    assert solver.is_possible(living_wolves(2))
    assert not solver.is_possible(living_wolves(3))


def test_without_the_rule_the_seed3_account_settled_nothing():
    """The gap run #14 found: レン genuine and truthful allowed no living wolf."""
    observations = ObservationSet.from_state(r14.board(3, 6, "vote"))
    rules = [m for m in default_rule_modules() if m.module_id != GameRunningRuleModule.module_id]
    supposition = (GenuineClaim(r14.REN, RoleName.SEER), HonestResults(r14.REN))
    tsumugi = has_role(r14.TSUMUGI, RoleName.WEREWOLF)

    without = build_solver(
        observations, CommonPublicPerspective(), rules=rules, assumptions=supposition
    )
    with_rule = build_solver(observations, CommonPublicPerspective(), assumptions=supposition)

    assert not without.is_forced(tsumugi)
    assert with_rule.is_forced(tsumugi)


def test_the_derived_rules_are_left_out_of_the_plain_ones():
    plain = {module.module_id for module in plain_rule_modules()}
    every = {module.module_id for module in default_rule_modules()}

    assert DERIVED_RULE_MODULE_IDS == {"claimed_divination", "game_running"}
    assert plain == every - DERIVED_RULE_MODULE_IDS


def test_a_count_over_no_seats_is_trivially_bounded():
    state = boards.deal({"p0": RoleName.SEER})
    solver = build_solver(boards.observe(state), CommonPublicPerspective())

    assert solver.is_possible(Hypothesis(claims=(RoleCountAmong((), RoleName.WEREWOLF),)))
    assert not solver.is_possible(
        Hypothesis(claims=(RoleCountAmong((), RoleName.WEREWOLF, at_least=1),))
    )
