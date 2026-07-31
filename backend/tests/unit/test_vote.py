import pytest

from app.engine.game import GameController
from app.engine.vote import VoteManager
from tests.conftest import make_player_specs


def _fresh_state_at_voting():
    controller = GameController(session_id="s", player_specs=make_player_specs(), seed=3)
    controller.state.phase = controller.state.phase  # noqa: PLW0127
    from app.engine.phases import Phase

    controller.state.phase = Phase.VOTING
    return controller


def test_single_max_executes_player():
    controller = _fresh_state_at_voting()
    ids = controller.state.alive_ids()
    target = ids[0]
    for voter in ids:
        if voter == target:
            continue
        controller.vote(voter, target)
    result = VoteManager().tally(controller.state)
    assert result.executed_player_id == target
    assert controller.state.players[target].alive is False


def _cast_even_split_tie(controller, ids: list[str]) -> None:
    """Produce an exact 8-8 tie between ids[0] and ids[1]: 7 of the other 15
    alive players vote ids[0], 8 vote ids[1], and ids[1] itself votes for
    ids[0] (bringing ids[0] to 8 as well). ids[0] does not need to vote."""
    target_a, target_b = ids[0], ids[1]
    others = ids[2:17]
    votes: dict[str, str] = {target_b: target_a}
    for i, voter in enumerate(others):
        votes[voter] = target_a if i < 7 else target_b
    controller.state.pending_votes.update(votes)


def test_tie_triggers_runoff_until_max_rounds_then_draw():
    controller = _fresh_state_at_voting()
    ids = controller.state.alive_ids()
    manager = VoteManager(max_vote_rounds=2)

    _cast_even_split_tie(controller, ids)
    result1 = manager.tally(controller.state)
    assert result1.tied_player_ids is not None
    assert set(result1.tied_player_ids) == {ids[0], ids[1]}
    assert controller.state.vote_round == 2

    # round 2: still tied -> draw (max_vote_rounds=2)
    _cast_even_split_tie(controller, ids)
    result2 = manager.tally(controller.state)
    assert result2.is_draw is True


def test_cannot_vote_for_self():
    controller = _fresh_state_at_voting()
    ids = controller.state.alive_ids()
    with pytest.raises(ValueError):
        controller.vote(ids[0], ids[0])
