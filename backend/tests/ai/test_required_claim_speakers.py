"""A seat with a claim to make is never dropped from the day.

Only the freemason leader used to be passed in as a planned claimant. A wolf
planning a seer counter-CO, a seer holding a result with no CO behind it, a
bluffer whose story had collapsed -- none of them were guaranteed a turn, so a
plan could exist and simply never be said.
"""

from __future__ import annotations

import asyncio

from app.ai.coordinator import AICoordinator
from app.ai.provider.mock import MockProvider
from app.ai.reasoning.belief import StoryStatus
from app.ai.reasoning.runtime import ReasoningRuntime
from app.engine.phases import Phase
from app.engine.roles import RoleName
from tests.ai.reasoning.solver import boards
from tests.conftest import make_controller

AI_IDS = [f"p{i}" for i in range(1, 17)]


def _board(day: int = 2):  # type: ignore[no-untyped-def]
    return boards.deal(
        {"p1": RoleName.WEREWOLF, "p4": RoleName.SEER, "p6": RoleName.FREEMASON}, day=day
    )


def test_the_freemason_leader_is_required_until_the_claim_is_made():
    state = _board()
    runtime = ReasoningRuntime(state, AI_IDS, seed=1)

    assert "p6" in runtime.required_claim_speakers(state, freemason_leader="p6")
    boards.claim(state, "p6", RoleName.FREEMASON, day=2)
    assert "p6" not in runtime.required_claim_speakers(state, freemason_leader="p6")


def test_a_planned_fake_claimant_is_required_until_the_claim_is_made():
    state = _board()
    runtime = ReasoningRuntime(state, AI_IDS, seed=1)
    plan = {"p1": RoleName.SEER}

    assert "p1" in runtime.required_claim_speakers(state, planned_fake_roles=plan)
    boards.claim(state, "p1", RoleName.SEER, day=2)
    assert "p1" not in runtime.required_claim_speakers(state, planned_fake_roles=plan)


def test_a_result_holder_without_its_co_is_required():
    state = _board()
    boards.divine(state, "p4", "p9", night=1)
    runtime = ReasoningRuntime(state, AI_IDS, seed=1)

    assert "p4" in runtime.required_claim_speakers(state)


def test_a_bluffer_whose_story_collapsed_is_required():
    """Two looks on one night: under the bluffer's own stated timing the story
    cannot be a real seer's, so it has to be retracted or slid."""
    state = _board()
    boards.claim(state, "p1", RoleName.SEER, day=1)
    boards.verdict(state, "p1", "seer", "p9", True, day=2, referenced_day=1)
    boards.verdict(state, "p1", "seer", "p11", False, day=2, referenced_day=1)
    runtime = ReasoningRuntime(state, AI_IDS, seed=1)
    runtime.refresh(state)

    deception = runtime.seats["p1"].deception
    assert deception is not None and deception.status is StoryStatus.COLLAPSED
    assert "p1" in runtime.required_claim_speakers(state)


def test_a_seer_whose_published_colour_contradicts_its_own_result_is_required():
    state = _board()
    boards.divine(state, "p4", "p9", night=1)  # p9 is a villager: really white
    boards.claim(state, "p4", RoleName.SEER, day=2)
    boards.verdict(state, "p4", "seer", "p9", True, day=2, referenced_day=1)  # said black
    runtime = ReasoningRuntime(state, AI_IDS, seed=1)

    assert "p4" in runtime.required_claim_speakers(state)


def test_a_seat_with_nothing_to_claim_is_not_required():
    state = _board()
    runtime = ReasoningRuntime(state, AI_IDS, seed=1)

    assert runtime.required_claim_speakers(state) == ()


def test_a_dead_seat_is_never_required():
    state = _board()
    boards.die_by_attack(state, "p6", night=1)
    runtime = ReasoningRuntime(state, AI_IDS, seed=1)

    assert "p6" not in runtime.required_claim_speakers(state, freemason_leader="p6")


def test_a_planned_fake_claimant_gets_a_turn_in_a_real_round():
    """Through `_start_discussion_round`, with the coordinator's own plan."""
    for seed in range(1, 80):
        controller = make_controller(seed=seed)
        state = controller.state
        state.phase = Phase.DISCUSSION
        state.day = 1
        runtime = ReasoningRuntime(state, AI_IDS, seed=seed)
        coordinator = AICoordinator(
            state, AI_IDS, MockProvider(seed=seed), seed=seed, reasoning=runtime
        )
        planners = [
            pid
            for pid in coordinator._wolf_deception.fake_role_by_player
            if pid in AI_IDS and state.players[pid].alive
        ]
        if planners:
            break
    else:
        raise AssertionError("no seed produced a planned fake claimant")

    round_state = asyncio.run(coordinator._start_discussion_round(state))

    # Being in the order is not enough -- the value ranking can happen to pick a
    # planner anyway. A guaranteed turn means sitting in the duty prefix, which
    # the ranking cannot reorder or drop.
    duty = round_state.order[: round_state.immediate_count]
    for planner in planners:
        assert planner in duty, f"{planner} planned a claim but was not a duty speaker"
