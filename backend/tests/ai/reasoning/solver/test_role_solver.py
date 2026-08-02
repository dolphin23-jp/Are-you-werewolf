"""Role deduction over the fixed composition, without an LLM anywhere."""

from __future__ import annotations

import pytest

from app.ai.reasoning.perspectives import (
    CommonPublicPerspective,
    GenericVillagerPerspective,
    PlayerPrivatePerspective,
    TrueWorldPerspective,
)
from app.ai.reasoning.solver import (
    MAX_MODEL_LIMIT,
    Hypothesis,
    RoleIs,
    SolverCache,
    build_solver,
    has_role,
    not_role,
    role_count,
)
from app.engine.roles import ROLE_DEFINITIONS, RoleName
from tests.ai.reasoning.solver.conftest import Board


def _solver(board: Board, perspective, **kwargs):  # type: ignore[no-untyped-def]
    return build_solver(board.observations, perspective, **kwargs)


def _true_world(board: Board) -> Hypothesis:
    return Hypothesis(
        claims=tuple(
            RoleIs(player_id=pid, role=role)
            for pid, role in sorted(board.observations.true_roles.items())
        ),
        label="true world",
    )


# -- the composition itself --


def test_the_dealt_composition_is_satisfiable():
    controller_board_roles = ROLE_DEFINITIONS
    assert sum(d.count for d in controller_board_roles.values()) == 17


def test_a_board_with_no_extra_knowledge_admits_worlds(board: Board):
    solver = _solver(board, CommonPublicPerspective())

    assert solver.is_possible(Hypothesis()) is True
    assert solver.representative_models([], 1)


def test_four_werewolves_is_unsatisfiable(board: Board):
    solver = _solver(board, CommonPublicPerspective())

    assert solver.is_possible(role_count(RoleName.WEREWOLF, 4)) is False
    assert solver.is_forced(role_count(RoleName.WEREWOLF, 3)) is True


def test_two_seers_is_unsatisfiable(board: Board):
    solver = _solver(board, CommonPublicPerspective())

    assert solver.is_possible(role_count(RoleName.SEER, 2)) is False
    assert solver.is_possible(role_count(RoleName.SEER, 0)) is False


def test_the_first_victim_can_be_neither_wolf_nor_fox(board: Board):
    solver = _solver(board, CommonPublicPerspective())
    victim = board.first_victim_id

    assert solver.is_possible(has_role(victim, RoleName.WEREWOLF)) is False
    assert solver.is_possible(has_role(victim, RoleName.FOX)) is False
    assert solver.is_forced(not_role(victim, RoleName.WEREWOLF)) is True
    # Still an ordinary unknown otherwise -- the rule narrows, it does not solve.
    assert solver.is_possible(has_role(victim, RoleName.VILLAGER)) is True


# -- what each seat knows --


def test_a_villager_cannot_be_their_own_wolf_hypothesis(board: Board):
    seat = board.one(RoleName.VILLAGER)
    solver = _solver(board, GenericVillagerPerspective(seat))

    assert solver.is_possible(has_role(seat, RoleName.WEREWOLF)) is False
    assert solver.certain_role(seat) is RoleName.VILLAGER


def test_a_werewolf_has_their_whole_team_settled(board: Board):
    wolves = board.seats(RoleName.WEREWOLF)
    solver = _solver(board, PlayerPrivatePerspective(wolves[0]))

    assert all(solver.is_forced(has_role(wolf, RoleName.WEREWOLF)) for wolf in wolves)
    assert all(solver.certain_role(wolf) is RoleName.WEREWOLF for wolf in wolves)


def test_a_madman_is_not_told_where_the_wolves_are(board: Board):
    madman = board.one(RoleName.MADMAN)
    wolves = board.seats(RoleName.WEREWOLF)
    solver = _solver(board, PlayerPrivatePerspective(madman))

    assert solver.certain_role(madman) is RoleName.MADMAN
    # The madman is on the werewolf team and still has to guess like everyone
    # else; handing them the positions is what turns them into an oracle.
    for wolf in wolves:
        assert solver.is_forced(has_role(wolf, RoleName.WEREWOLF)) is False
        assert solver.is_possible(has_role(wolf, RoleName.VILLAGER)) is True


def test_a_fox_learns_nothing_about_anybody_else(board: Board):
    fox = board.one(RoleName.FOX)
    solver = _solver(board, PlayerPrivatePerspective(fox))
    others = [pid for pid in board.observations.player_ids if pid != fox]

    assert solver.certain_role(fox) is RoleName.FOX
    assert all(solver.certain_role(pid) is None for pid in others)


def test_a_freemason_has_their_partner_settled(board: Board):
    first, second = board.seats(RoleName.FREEMASON)
    solver = _solver(board, PlayerPrivatePerspective(first))

    assert solver.is_forced(has_role(second, RoleName.FREEMASON)) is True
    assert solver.certain_role(second) is RoleName.FREEMASON


# -- leakage --


def test_the_public_view_cannot_see_an_unpublished_freemason_pair(board: Board):
    first, second = board.seats(RoleName.FREEMASON)
    solver = _solver(board, CommonPublicPerspective())

    assert solver.is_forced(has_role(first, RoleName.FREEMASON)) is False
    assert solver.is_possible(has_role(first, RoleName.WEREWOLF)) is True
    assert solver.certain_role(second) is None
    # Nothing at all is settled from the public view except the first victim's
    # excluded roles, which is a rule rather than a leak.
    settled = [
        pid
        for pid in board.observations.player_ids
        if solver.certain_role(pid) is not None
    ]
    assert settled == []


def test_no_private_seat_can_settle_a_role_it_was_not_dealt(board: Board):
    for seat in board.observations.player_ids:
        dealt = board.observations.true_roles[seat]
        solver = _solver(board, PlayerPrivatePerspective(seat))
        settled = {
            pid
            for pid in board.observations.player_ids
            if solver.certain_role(pid) is not None
        }
        allowed = {seat} | set(board.observations.seat_knowledge(seat).ally_ids)
        assert settled == allowed, f"{seat} ({dealt.value}) settled {settled - allowed}"


def test_the_true_assignment_is_consistent_with_every_private_view(board: Board):
    truth = _true_world(board)

    for seat in board.observations.player_ids:
        solver = _solver(board, PlayerPrivatePerspective(seat))
        assert solver.is_possible(truth) is True, f"{seat} rules out the real world"


# -- implication --


def test_implication_carries_a_settled_conclusion_through_a_premise(board: Board):
    solver = _solver(board, CommonPublicPerspective())
    wolves = board.seats(RoleName.WEREWOLF)
    victim = board.first_victim_id

    # Pinning all three wolf seats forces every other seat to be a non-wolf.
    premise = Hypothesis(
        claims=tuple(RoleIs(player_id=wolf, role=RoleName.WEREWOLF) for wolf in wolves)
    )
    others = [pid for pid in board.observations.player_ids if pid not in wolves]

    assert solver.is_possible(premise) is True
    assert solver.implies(premise, not_role(others[0], RoleName.WEREWOLF)) is True
    assert solver.implies(Hypothesis(), not_role(victim, RoleName.FOX)) is True


def test_nothing_is_forced_by_a_board_that_admits_no_world(board: Board):
    solver = _solver(board, CommonPublicPerspective())
    impossible = role_count(RoleName.SEER, 2)

    # An impossible premise materially implies anything, which is exactly why
    # `is_forced` refuses to answer from a contradictory board.
    assert solver.implies(impossible, has_role("p0", RoleName.FOX)) is True
    assert solver.is_possible(impossible) is False


# -- representative models --


def test_model_enumeration_is_bounded_and_distinct(board: Board):
    solver = _solver(board, CommonPublicPerspective())

    models = solver.representative_models([], 5)

    assert len(models) == 5
    assert len({model.roles for model in models}) == 5
    for model in models:
        assert len(model.players_with(RoleName.WEREWOLF)) == 3
        assert model.role_of(board.first_victim_id) is not RoleName.WEREWOLF


def test_an_over_large_model_request_is_capped(board: Board):
    solver = _solver(board, CommonPublicPerspective())

    models = solver.representative_models([], MAX_MODEL_LIMIT + 500)

    assert len(models) <= MAX_MODEL_LIMIT


def test_model_enumeration_repeats_exactly_across_independent_solvers(board: Board):
    first = _solver(board, CommonPublicPerspective(), cache=SolverCache())
    second = _solver(board, CommonPublicPerspective(), cache=SolverCache())

    # Separate solvers, separate caches: equality here is the enumeration itself
    # being deterministic, not a memoized answer being handed back.
    assert first.representative_models([], 3) == second.representative_models([], 3)


def test_a_fully_known_world_has_exactly_one_model(board: Board):
    solver = _solver(board, TrueWorldPerspective())

    models = solver.representative_models([], 4)

    assert len(models) == 1
    assert dict(models[0].roles) == dict(board.observations.true_roles)


# -- caching --


def test_the_same_question_is_answered_from_cache(board: Board):
    cache = SolverCache()
    solver = _solver(board, CommonPublicPerspective(), cache=cache)
    question = has_role(board.first_victim_id, RoleName.WEREWOLF)

    first = solver.is_possible(question)
    checks_after_first = solver.stats.checks
    second = solver.is_possible(question)

    assert first is second is False
    assert solver.stats.checks == checks_after_first  # no second solver call
    assert cache.hits == 1


def test_a_different_perspective_does_not_reuse_another_ones_answer(board: Board):
    cache = SolverCache()
    wolf = board.one(RoleName.WEREWOLF)
    question = has_role(wolf, RoleName.WEREWOLF)

    public = _solver(board, CommonPublicPerspective(), cache=cache)
    private = _solver(board, PlayerPrivatePerspective(wolf), cache=cache)

    assert public.is_forced(question) is False
    assert private.is_forced(question) is True


def test_the_cache_is_bounded(board: Board):
    cache = SolverCache(capacity=4)
    solver = _solver(board, CommonPublicPerspective(), cache=cache)

    for role in RoleName:
        solver.is_possible(has_role("p0", role))

    assert len(cache) == 4


@pytest.mark.parametrize("role", list(RoleName))
def test_every_role_remains_individually_possible_for_an_unknown_seat(
    board: Board, role: RoleName
):
    solver = _solver(board, CommonPublicPerspective())
    seat = next(pid for pid in board.observations.player_ids if pid != board.first_victim_id)

    assert solver.is_possible(has_role(seat, role)) is True
