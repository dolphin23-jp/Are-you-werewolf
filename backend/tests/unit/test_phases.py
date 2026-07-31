from app.engine.phases import Phase, PhaseEvent, is_legal, next_phase


def test_happy_path_cycle():
    assert next_phase(Phase.WAITING, PhaseEvent.START_GAME) == Phase.NIGHT
    assert next_phase(Phase.NIGHT, PhaseEvent.RESOLVE_NIGHT) == Phase.DAWN
    assert next_phase(Phase.DAWN, PhaseEvent.START_DISCUSSION) == Phase.DISCUSSION
    assert next_phase(Phase.DISCUSSION, PhaseEvent.END_DISCUSSION) == Phase.VOTING
    assert next_phase(Phase.VOTING, PhaseEvent.VOTE_RESOLVED) == Phase.VOTE_RESULT
    assert next_phase(Phase.VOTE_RESULT, PhaseEvent.START_NIGHT) == Phase.NIGHT


def test_runoff_cycle():
    assert next_phase(Phase.VOTING, PhaseEvent.VOTE_TIE) == Phase.RUNOFF
    assert next_phase(Phase.RUNOFF, PhaseEvent.VOTE_TIE) == Phase.RUNOFF
    assert next_phase(Phase.RUNOFF, PhaseEvent.RUNOFF_EXHAUSTED) == Phase.VOTE_RESULT
    assert next_phase(Phase.RUNOFF, PhaseEvent.VOTE_RESOLVED) == Phase.VOTE_RESULT


def test_victory_decided_wins_from_any_phase():
    for phase in Phase:
        assert next_phase(phase, PhaseEvent.VICTORY_DECIDED) == Phase.GAME_OVER


def test_illegal_transition_returns_none():
    assert next_phase(Phase.WAITING, PhaseEvent.END_DISCUSSION) is None
    assert is_legal(Phase.WAITING, PhaseEvent.END_DISCUSSION) is False
    assert is_legal(Phase.WAITING, PhaseEvent.START_GAME) is True
