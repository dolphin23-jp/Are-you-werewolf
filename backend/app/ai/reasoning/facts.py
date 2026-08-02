"""A read-only projection of facts available to every player.

The ledger is rebuilt from ``GameState``. It never becomes a second mutable
source of truth and intentionally does not expose roles or private night actions.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.engine.roles import RoleName
from app.engine.state import DeathCause, GameState, PublicDeathCause


@dataclass(frozen=True)
class PublicPlayerFact:
    player_id: str
    name: str
    alive: bool
    death_day: int | None
    death_cause: PublicDeathCause | None


@dataclass(frozen=True)
class PublicClaimFact:
    player_id: str
    claimed_role: RoleName
    day: int


@dataclass(frozen=True)
class PublicResultFact:
    claimant_id: str
    result_type: str
    target_id: str
    is_werewolf: bool
    day: int


@dataclass(frozen=True)
class PublicVoteFact:
    voter_id: str
    target_id: str
    day: int
    round: int


@dataclass(frozen=True)
class PublicFactLedger:
    day: int
    players: tuple[PublicPlayerFact, ...]
    claims: tuple[PublicClaimFact, ...]
    results: tuple[PublicResultFact, ...]
    votes: tuple[PublicVoteFact, ...]

    @classmethod
    def from_state(cls, state: GameState) -> PublicFactLedger:
        players = tuple(
            PublicPlayerFact(
                player_id=player.player_id,
                name=player.name,
                alive=player.alive,
                death_day=player.death_day,
                death_cause=_public_cause(player.death_cause),
            )
            for player in state.players.values()
        )
        return cls(
            day=state.day,
            players=players,
            claims=tuple(
                PublicClaimFact(item.player_id, item.claimed_role, item.day)
                for item in state.co_declarations
            ),
            results=tuple(
                PublicResultFact(
                    item.claimant_id,
                    item.result_type,
                    item.target_id,
                    item.is_werewolf,
                    item.day,
                )
                for item in state.public_result_claims
            ),
            votes=tuple(
                PublicVoteFact(item.voter_id, item.target_id, item.day, item.round)
                for item in state.vote_records
            ),
        )

    @property
    def alive_ids(self) -> tuple[str, ...]:
        return tuple(item.player_id for item in self.players if item.alive)

    @property
    def dead_ids(self) -> tuple[str, ...]:
        return tuple(item.player_id for item in self.players if not item.alive)

    def vote_target(self, voter_id: str, day: int, round: int = 1) -> str | None:
        return next(
            (
                vote.target_id
                for vote in self.votes
                if vote.voter_id == voter_id and vote.day == day and vote.round == round
            ),
            None,
        )


def _public_cause(cause: DeathCause | None) -> PublicDeathCause | None:
    if cause is None:
        return None
    if cause == DeathCause.EXECUTED:
        return PublicDeathCause.EXECUTED
    if cause == DeathCause.FIRST_VICTIM:
        return PublicDeathCause.FIRST_VICTIM
    return PublicDeathCause.NIGHT
