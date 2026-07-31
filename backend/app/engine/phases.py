"""Explicit phase-transition table for the game state machine.

Kept as a pure function so legal transitions are documented and testable in
one place, independent of `GameController`.
"""

from __future__ import annotations

from enum import StrEnum


class Phase(StrEnum):
    WAITING = "waiting"
    NIGHT = "night"
    DAWN = "dawn"
    DISCUSSION = "discussion"
    VOTING = "voting"
    RUNOFF = "runoff"
    VOTE_RESULT = "vote_result"
    GAME_OVER = "game_over"


class PhaseEvent(StrEnum):
    START_GAME = "start_game"
    RESOLVE_NIGHT = "resolve_night"
    START_DISCUSSION = "start_discussion"
    END_DISCUSSION = "end_discussion"
    VOTE_TIE = "vote_tie"
    VOTE_RESOLVED = "vote_resolved"
    RUNOFF_EXHAUSTED = "runoff_exhausted"
    START_NIGHT = "start_night"
    VICTORY_DECIDED = "victory_decided"


_TRANSITIONS: dict[tuple[Phase, PhaseEvent], Phase] = {
    (Phase.WAITING, PhaseEvent.START_GAME): Phase.NIGHT,
    (Phase.NIGHT, PhaseEvent.RESOLVE_NIGHT): Phase.DAWN,
    (Phase.DAWN, PhaseEvent.START_DISCUSSION): Phase.DISCUSSION,
    (Phase.DISCUSSION, PhaseEvent.END_DISCUSSION): Phase.VOTING,
    (Phase.VOTING, PhaseEvent.VOTE_TIE): Phase.RUNOFF,
    (Phase.RUNOFF, PhaseEvent.VOTE_TIE): Phase.RUNOFF,
    (Phase.RUNOFF, PhaseEvent.VOTE_RESOLVED): Phase.VOTE_RESULT,
    (Phase.RUNOFF, PhaseEvent.RUNOFF_EXHAUSTED): Phase.VOTE_RESULT,
    (Phase.VOTING, PhaseEvent.VOTE_RESOLVED): Phase.VOTE_RESULT,
    (Phase.VOTE_RESULT, PhaseEvent.START_NIGHT): Phase.NIGHT,
}

# Any phase transitions to GAME_OVER once victory is decided.
_ANY_PHASE_TO_GAME_OVER = PhaseEvent.VICTORY_DECIDED


def next_phase(current: Phase, event: PhaseEvent) -> Phase | None:
    """Return the resulting phase for `event` fired while in `current`,
    or None if the transition is illegal."""
    if event == _ANY_PHASE_TO_GAME_OVER:
        return Phase.GAME_OVER
    return _TRANSITIONS.get((current, event))


def is_legal(current: Phase, event: PhaseEvent) -> bool:
    return next_phase(current, event) is not None
