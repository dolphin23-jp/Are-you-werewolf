"""Hand-dealt 17-seat boards.

The solver tests need a specific arrangement -- a freemason as the first victim,
a wolf sitting on an un-divined seat -- which a seeded random deal cannot be
asked for. `deal` takes the assignment directly and fills the rest with
villagers, so each test states only the seats its scenario turns on.
"""

from __future__ import annotations

from app.ai.reasoning.observations import ObservationSet
from app.engine.phases import Phase
from app.engine.roles import ROLE_DEFINITIONS, RoleName
from app.engine.speech_events import SpeechEventType, result_role
from app.engine.state import DeathCause, DeathRecord, GameState, PlayerState

PLAYER_IDS = tuple(f"p{i}" for i in range(17))


def deal(assignment: dict[str, RoleName], *, day: int = 1) -> GameState:
    """Build a board from a partial assignment, completing the composition.

    Raises if the given seats already break the fixed composition -- a test that
    starts from an impossible board proves nothing.
    """
    roles = dict(assignment)
    remaining: list[RoleName] = []
    for role, definition in ROLE_DEFINITIONS.items():
        used = sum(1 for assigned in roles.values() if assigned == role)
        if used > definition.count:
            raise ValueError(f"{role.value} appears {used} times, limit {definition.count}")
        remaining.extend([role] * (definition.count - used))
    free = [pid for pid in PLAYER_IDS if pid not in roles]
    if len(free) != len(remaining):
        raise ValueError("assignment does not fit the 17-player composition")
    # Deterministic completion: seats fill in seating order from the leftover
    # pool in role-definition order.
    roles.update(dict(zip(free, remaining, strict=True)))
    players = {
        pid: PlayerState(player_id=pid, name=f"Player{index}", role=roles[pid])
        for index, pid in enumerate(PLAYER_IDS)
    }
    return GameState(
        session_id="solver-fixture", players=players, phase=Phase.DISCUSSION, day=day
    )


def kill_first_victim(state: GameState, player_id: str) -> None:
    state.first_victim_id = player_id
    _kill(state, player_id, DeathCause.FIRST_VICTIM, day=0)


def execute(state: GameState, player_id: str, day: int) -> None:
    _kill(state, player_id, DeathCause.EXECUTED, day)


def _kill(state: GameState, player_id: str, cause: DeathCause, day: int) -> None:
    player = state.players[player_id]
    player.alive = False
    player.death_cause = cause
    player.death_day = day
    state.death_records.append(DeathRecord(player_id=player_id, cause=cause, day=day))


def claim(state: GameState, player_id: str, role: RoleName, day: int = 1) -> None:
    _event(state, day, player_id, SpeechEventType.ROLE_CLAIM, role=role)


def verdict(
    state: GameState,
    claimant_id: str,
    result_type: str,
    target_id: str,
    is_werewolf: bool,
    day: int = 1,
) -> None:
    _event(
        state,
        day,
        claimant_id,
        SpeechEventType.ABILITY_RESULT,
        role=result_role(result_type),
        target_id=target_id,
        result_is_werewolf=is_werewolf,
    )


def _event(
    state: GameState, day: int, actor_id: str, event_type: SpeechEventType, **kwargs: object
) -> None:
    original = state.day
    state.day = day
    try:
        state.append_speech_event(actor_id, event_type, **kwargs)  # type: ignore[arg-type]
    finally:
        state.day = original


def observe(state: GameState) -> ObservationSet:
    return ObservationSet.from_state(state)
