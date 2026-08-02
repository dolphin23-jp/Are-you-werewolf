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
from dataclasses import dataclass

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
class ObservationSet:
    player_ids: tuple[str, ...]
    true_roles: Mapping[str, RoleName]
    first_victim_id: str | None = None
    board_version: str = ""

    @classmethod
    def from_state(cls, state: GameState) -> ObservationSet:
        return cls(
            player_ids=tuple(state.players),
            true_roles={pid: player.role for pid, player in state.players.items()},
            first_victim_id=state.first_victim_id,
            board_version=board_version(state),
        )

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
