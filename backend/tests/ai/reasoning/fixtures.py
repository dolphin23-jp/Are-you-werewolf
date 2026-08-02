"""Small synthetic boards for state-consistency tests.

Deliberately not a captured real game: the bugs these tests pin down (a dead
player put back on the execution block, a medium result about someone who was
never executed, a vote history remembered wrong) each need three or four facts
to reproduce, and a 17-seat transcript would bury them. Twelve seats is enough
to cover p1-vs-p11 confusion.
"""

from __future__ import annotations

from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.state import (
    ChatChannel,
    ChatMessage,
    CoDeclaration,
    DeathCause,
    DeathRecord,
    GameState,
    PlayerState,
    PublicResultClaim,
    VoteRecord,
)

# p0..p11, so p1 and p11 both exist and any id handling that ignores token
# boundaries fails loudly.
DEFAULT_PLAYER_COUNT = 12


def make_state(
    day: int = 1,
    player_count: int = DEFAULT_PLAYER_COUNT,
    phase: Phase = Phase.DISCUSSION,
    roles: dict[str, RoleName] | None = None,
) -> GameState:
    players = {
        f"p{i}": PlayerState(
            player_id=f"p{i}",
            name=f"Player{i}",
            role=(roles or {}).get(f"p{i}", RoleName.VILLAGER),
        )
        for i in range(player_count)
    }
    return GameState(session_id="fixture", players=players, phase=phase, day=day)


def execute(state: GameState, player_id: str, day: int) -> None:
    _kill(state, player_id, DeathCause.EXECUTED, day)


def kill_at_night(state: GameState, player_id: str, day: int) -> None:
    _kill(state, player_id, DeathCause.ATTACKED, day)


def make_first_victim(state: GameState, player_id: str) -> None:
    state.first_victim_id = player_id
    _kill(state, player_id, DeathCause.FIRST_VICTIM, 0)


def _kill(state: GameState, player_id: str, cause: DeathCause, day: int) -> None:
    player = state.players[player_id]
    player.alive = False
    player.death_cause = cause
    player.death_day = day
    state.death_records.append(DeathRecord(player_id=player_id, cause=cause, day=day))


def declare_co(state: GameState, player_id: str, role: RoleName, day: int = 1) -> None:
    state.co_declarations.append(
        CoDeclaration(player_id=player_id, claimed_role=role, day=day)
    )


def publish_result(
    state: GameState,
    claimant_id: str,
    result_type: str,
    target_id: str,
    is_werewolf: bool,
    day: int = 1,
) -> None:
    state.public_result_claims.append(
        PublicResultClaim(
            claimant_id=claimant_id,
            result_type=result_type,
            target_id=target_id,
            is_werewolf=is_werewolf,
            day=day,
        )
    )


def cast_vote(
    state: GameState, voter_id: str, target_id: str, day: int = 1, round_number: int = 1
) -> None:
    state.vote_records.append(
        VoteRecord(voter_id=voter_id, target_id=target_id, day=day, round=round_number)
    )


def say(state: GameState, author_id: str, content: str, day: int = 1) -> str:
    message_id = f"m{state.next_message_number}"
    state.next_message_number += 1
    state.chat_log.append(
        ChatMessage(
            message_id=message_id,
            author_id=author_id,
            content=content,
            channel=ChatChannel.PUBLIC,
            day=day,
        )
    )
    return message_id
