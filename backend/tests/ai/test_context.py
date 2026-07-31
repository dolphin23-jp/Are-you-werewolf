"""Prompt-assembly regressions.

Every machine-readable answer the model gives is a player_id, but the parts
of the prompt it reasons from -- the chat log, its wolf allies, its freemason
partner -- were written purely with names. Nothing tied the two together, so
the model was effectively voting blind. These tests keep the roster mapping
in place.
"""

from __future__ import annotations

from app.ai.context import ContextBuilder, DaySummaryManager
from app.ai.deception import FakeClaimGuard, WolfDeceptionAssignment
from app.ai.personalities import assign_personalities
from app.ai.strategy import StrategyAnalyzer, render_board_analysis
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.vote import VoteManager
from tests.conftest import make_controller


def _builder(state) -> ContextBuilder:
    wolf_ids = [p.player_id for p in state.players_by_role(RoleName.WEREWOLF)]
    return ContextBuilder(
        personalities=assign_personalities(list(state.players), seed=1),
        day_summaries=DaySummaryManager(),
        wolf_deception=WolfDeceptionAssignment(
            pattern_name="all_lurk",
            pattern_label="全潜伏",
            fake_role_by_player={},
            lurking_player_ids=wolf_ids,
        ),
        madman_fake_role=None,
        fake_claim_guard=FakeClaimGuard(wolf_team_ids=set(wolf_ids)),
    )


def test_vote_candidates_carry_both_name_and_id():
    controller = make_controller(seed=4)
    state = controller.state
    builder = _builder(state)

    _, messages = builder.build_vote_context(state, "p1", ["p2", "p3"])
    body = messages[0].content

    assert "Player2(p2)" in body
    assert "Player3(p3)" in body


def test_wolf_allies_carry_ids_so_a_wolf_can_avoid_voting_for_a_partner():
    controller = make_controller(seed=4)
    state = controller.state
    wolves = [p.player_id for p in state.players_by_role(RoleName.WEREWOLF)]
    builder = _builder(state)

    system, _ = builder.build_vote_context(state, wolves[0], state.alive_ids()[1:])

    for ally in wolves[1:]:
        assert f"{state.players[ally].name}({ally})" in system


def test_board_analysis_lists_the_living_roster():
    controller = make_controller(seed=4)
    state = controller.state
    rendered = render_board_analysis(StrategyAnalyzer().analyze(state), state)

    for pid in state.alive_ids():
        assert f"{state.players[pid].name}({pid})" in rendered


def test_runoff_prompt_explains_why_the_field_shrank():
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.VOTING
    ids = state.alive_ids()

    # Force an exact tie so the manager narrows the field.
    state.pending_votes.update({ids[2]: ids[0], ids[3]: ids[1]})
    VoteManager().tally(state)
    assert state.runoff_candidates

    builder = _builder(state)
    _, messages = builder.build_vote_context(state, ids[4], state.votable_ids(ids[4]))
    body = messages[0].content

    assert "決選投票" in body
    assert "同数" in body


def test_ordinary_round_does_not_mention_a_runoff():
    controller = make_controller(seed=4)
    state = controller.state
    builder = _builder(state)

    _, messages = builder.build_vote_context(state, "p1", ["p2", "p3"])

    assert "決選投票" not in messages[0].content
