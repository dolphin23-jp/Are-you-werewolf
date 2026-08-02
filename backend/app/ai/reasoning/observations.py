"""Everything the solver may be told about a board, before any viewpoint filter.

`ObservationSet` deliberately holds the true assignment: the same object serves
a werewolf reasoning about their own team, a villager who knows only their own
card, and an evaluation harness checking that the real world is consistent. What
each of those is allowed to see is decided by the `Perspective`, never by the
observations and never by a rule module reading around them.

Nothing here is a *conclusion*. Observations are what happened; hypotheses about
what it means live in the solver.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field

from app.ai.reasoning.facts import PublicFactLedger
from app.engine.roles import RoleName
from app.engine.state import GameState


@dataclass(frozen=True)
class SeatKnowledge:
    """What one seat legitimately knows about roles from the deal alone.

    `ally_ids` is empty for every role except werewolf and freemason. The madman
    in particular knows their own card and nothing else -- handing them the wolf
    positions is the single most common way a werewolf-team AI stops behaving
    like a player and starts behaving like an oracle.
    """

    player_id: str
    self_role: RoleName
    ally_ids: tuple[str, ...] = ()

    def as_role_map(self) -> dict[str, RoleName]:
        return {self.player_id: self.self_role, **{ally: self.self_role for ally in self.ally_ids}}


_SHARED_ROLE_KNOWLEDGE = (RoleName.WEREWOLF, RoleName.FREEMASON)


@dataclass(frozen=True)
class PublicClaim:
    """Someone said they hold a role. Whether that is true is a separate question."""

    player_id: str
    role: RoleName
    day: int
    source_message_id: str = ""


@dataclass(frozen=True)
class PublicVerdict:
    """Someone published a seer/medium result. Likewise: said, not established."""

    claimant_id: str
    result_type: str
    target_id: str
    is_werewolf: bool
    day: int
    source_message_id: str = ""


@dataclass(frozen=True)
class ObservationSet:
    player_ids: tuple[str, ...]
    true_roles: Mapping[str, RoleName]
    first_victim_id: str | None = None
    board_version: str = ""
    day: int = 0
    alive: Mapping[str, bool] = field(default_factory=dict)
    claims: tuple[PublicClaim, ...] = ()
    verdicts: tuple[PublicVerdict, ...] = ()
    executed_ids: tuple[str, ...] = ()

    @classmethod
    def from_state(cls, state: GameState) -> ObservationSet:
        ledger = PublicFactLedger(state)
        return cls(
            player_ids=tuple(state.players),
            true_roles={pid: player.role for pid, player in state.players.items()},
            first_victim_id=state.first_victim_id,
            board_version=board_version(state),
            day=state.day,
            alive={pid: player.alive for pid, player in state.players.items()},
            claims=tuple(
                PublicClaim(
                    player_id=claim.player_id,
                    role=claim.claimed_role,
                    day=claim.day,
                    source_message_id=claim.source_message_id,
                )
                for claim in ledger.co_declarations()
            ),
            verdicts=tuple(
                PublicVerdict(
                    claimant_id=result.claimant_id,
                    result_type=result.result_type,
                    target_id=result.target_id,
                    is_werewolf=result.is_werewolf,
                    day=result.day,
                )
                for result in ledger.public_results()
            ),
            executed_ids=ledger.executed_ids(),
        )

    # -- public-claim lookups --

    def claimed_role_of(self, player_id: str) -> RoleName | None:
        return next(
            (claim.role for claim in self.claims if claim.player_id == player_id), None
        )

    def claimants_of(self, role: RoleName) -> tuple[str, ...]:
        return tuple(claim.player_id for claim in self.claims if claim.role == role)

    def verdicts_by(self, claimant_id: str) -> tuple[PublicVerdict, ...]:
        return tuple(
            verdict for verdict in self.verdicts if verdict.claimant_id == claimant_id
        )

    def is_alive(self, player_id: str) -> bool:
        return self.alive.get(player_id, True)

    def seat_knowledge(self, player_id: str) -> SeatKnowledge:
        """The deal-time knowledge of one seat. Wolves see wolves, freemasons see
        their partner; everyone else sees only their own card."""
        role = self.true_roles[player_id]
        allies: tuple[str, ...] = ()
        if role in _SHARED_ROLE_KNOWLEDGE:
            allies = tuple(
                pid
                for pid in self.player_ids
                if pid != player_id and self.true_roles[pid] == role
            )
        return SeatKnowledge(player_id=player_id, self_role=role, ally_ids=allies)


def board_version(state: GameState) -> str:
    """A short digest that changes whenever anything the solver reads changes.

    Used as part of the cache key. It covers more than the constraints currently
    consume (deaths, claims, votes) on purpose: a later rule module that starts
    reading one of those must not silently inherit answers cached before it did.
    """
    parts = [
        state.session_id,
        str(state.day),
        str(state.phase),
        str(state.first_victim_id),
        ",".join(f"{pid}:{int(player.alive)}" for pid, player in state.players.items()),
        f"deaths={len(state.death_records)}",
        f"events={len(state.speech_events)}",
        f"votes={len(state.vote_records)}",
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
