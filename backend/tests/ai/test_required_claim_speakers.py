"""A seat with a claim to make is never dropped from the day.

Only the freemason leader used to be passed in as a planned claimant. A wolf
planning a seer counter-CO, a seer holding a result with no CO behind it, a
bluffer whose story had collapsed -- none of them were guaranteed a turn, so a
plan could exist and simply never be said.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.ai.coordinator import AICoordinator
from app.ai.provider.mock import MockProvider
from app.ai.reasoning.belief import StoryStatus
from app.ai.reasoning.runtime import ReasoningRuntime
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.sessions.models import DiscussionRoundState
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


def _day_one_with_an_unclaimed_seer_result():  # type: ignore[no-untyped-def]
    """A real day 1 with a living human: an AI seer holding its night-0 look, no CO yet."""
    for seed in range(1, 80):
        controller = make_controller(seed=seed)
        state = controller.state
        seer = next(p.player_id for p in state.players.values() if p.role is RoleName.SEER)
        if seer == "p0":
            continue
        controller.start_game()
        target = next(
            pid for pid in state.alive_ids() if pid not in (seer, state.first_victim_id)
        )
        controller.submit_night_action(seer, "divine", target)
        controller.resolve_night()
        if state.players[seer].alive and state.players["p0"].alive:
            controller.start_discussion()
            return controller, seer, seed
    raise AssertionError("no seed produced a live AI seer holding a result")


def test_a_seat_with_two_duties_takes_one_morning_turn():
    """The seer is a result holder *and* a required claimant. Counted twice, the
    morning stage waited for one more speaker than it held: past the seer it
    handed the floor to an ordinary opener before the human ever got it."""
    controller, seer, seed = _day_one_with_an_unclaimed_seer_result()
    state = controller.state
    alive_ai = [pid for pid in AI_IDS if state.players[pid].alive]
    runtime = ReasoningRuntime(state, alive_ai, seed=seed)
    coordinator = AICoordinator(
        state, alive_ai, MockProvider(seed=seed), seed=seed, reasoning=runtime
    )
    plan = coordinator._freemason_public_plan
    claimants = runtime.required_claim_speakers(
        state,
        planned_fake_roles={
            **coordinator._wolf_deception.fake_role_by_player,
            **coordinator._madman_fake_role_by_player,
        },
        freemason_leader=plan[0] if plan is not None else None,
    )
    holders = [pid for pid in alive_ai if runtime.holds_unpublished_result(state, pid)]
    assert seer in holders and seer in claimants
    duty_seats = set(holders) | {pid for pid in claimants if pid in alive_ai}
    session = SimpleNamespace(
        controller=controller, discussion_lock=asyncio.Lock(), human_id="p0"
    )

    # Drained the way `after_discussion_phase_entered` drains the opening.
    for _ in range(10):
        asyncio.run(coordinator.advance_discussion(session))
        round_state = session.discussion_round
        if round_state.complete or round_state.awaiting_human or round_state.stage != "immediate":
            break

    assert round_state.immediate_count == len(duty_seats)
    assert round_state.stage == "initial_view"
    assert round_state.awaiting_human is True
    # The floor passes to the human after the duty speakers -- not one seat later.
    spoken = [m.author_id for m in state.chat_log if m.channel.value == "public"]
    assert sorted(spoken) == sorted(duty_seats)


def test_the_morning_stage_cannot_outrun_the_order():
    """Even a count that exceeds the order ends the stage once the order is spoken."""
    round_state = DiscussionRoundState(day=1, order=["p1"], cursor=1, immediate_count=2)

    assert AICoordinator._close_immediate_stage(round_state) is True
    assert round_state.stage == "initial_view"
    assert AICoordinator._close_immediate_stage(round_state) is False


def test_a_duty_speaker_confirming_early_is_taken_off_the_morning_count():
    controller = make_controller(seed=4)
    coordinator = AICoordinator(
        controller.state, ["p1", "p2", "p3"], MockProvider(seed=4), seed=1
    )
    coordinator._forced_partner_confirmations.add("p2")
    round_state = DiscussionRoundState(
        day=1, order=["p1", "p2", "p3"], cursor=1, immediate_count=2, max_total=10
    )

    speaker, stage = coordinator._next_discussion_speaker(controller.state, round_state)

    assert (speaker, stage) == ("p2", "freemason_confirmation")
    assert round_state.order == ["p1", "p3"]
    assert round_state.immediate_count == 1
    # p1 and p2 have both spoken: the morning is over, and p3 opens the views.
    assert AICoordinator._close_immediate_stage(round_state) is True
    assert coordinator._next_discussion_speaker(controller.state, round_state) == (
        "p3",
        "initial_view",
    )
