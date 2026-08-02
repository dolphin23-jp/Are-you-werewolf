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
from app.engine.state import DeathCause, GameState


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
class NightDeath:
    """Someone died overnight. Which of the two night causes it was is not
    public -- attacked and cursed look identical from the table."""

    player_id: str
    night: int


@dataclass(frozen=True)
class NightAction:
    actor_id: str
    target_id: str
    night: int


@dataclass(frozen=True)
class NightKnowledge:
    """Night actions a viewpoint legitimately knows, keyed by night.

    Empty for the public view by construction, so the real guard and attack
    targets have no route into a village-side deduction.
    """

    divines: Mapping[int, str] = field(default_factory=dict)
    guards: Mapping[int, str] = field(default_factory=dict)
    attacks: Mapping[int, str] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (self.divines or self.guards or self.attacks)


EMPTY_NIGHT_KNOWLEDGE = NightKnowledge()


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
    night_deaths: tuple[NightDeath, ...] = ()
    # True night actions. Reachable only through a perspective, exactly like
    # `true_roles` -- never read directly by a rule module.
    divine_actions: tuple[NightAction, ...] = ()
    guard_actions: tuple[NightAction, ...] = ()
    attack_actions: tuple[NightAction, ...] = ()

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
            night_deaths=tuple(
                NightDeath(player_id=record.player_id, night=record.day)
                for record in state.death_records
                if record.cause in (DeathCause.ATTACKED, DeathCause.CURSED)
            ),
            divine_actions=tuple(
                NightAction(actor_id=r.seer_id, target_id=r.target_id, night=r.day)
                for r in state.divine_records
            ),
            guard_actions=tuple(
                NightAction(actor_id=r.hunter_id, target_id=r.target_id, night=r.day)
                for r in state.guard_records
            ),
            attack_actions=tuple(
                NightAction(actor_id=r.wolf_id, target_id=r.target_id, night=r.day)
                for r in state.attack_records
            ),
        )

    # -- night lookups --

    def deaths_on(self, night: int) -> tuple[NightDeath, ...]:
        return tuple(death for death in self.night_deaths if death.night == night)

    def nights_with_deaths(self) -> tuple[int, ...]:
        return tuple(sorted({death.night for death in self.night_deaths}))

    def night_knowledge_for(self, player_id: str) -> NightKnowledge:
        """What this seat did at night. Wolves share the team's attack, so any
        wolf may reason from it, not only the alpha who submitted it."""
        role = self.true_roles[player_id]
        wolf_ids = {
            pid for pid, dealt in self.true_roles.items() if dealt == RoleName.WEREWOLF
        }
        return NightKnowledge(
            divines={
                action.night: action.target_id
                for action in self.divine_actions
                if action.actor_id == player_id
            },
            guards={
                action.night: action.target_id
                for action in self.guard_actions
                if action.actor_id == player_id
            },
            attacks={
                action.night: action.target_id
                for action in self.attack_actions
                if role is RoleName.WEREWOLF and action.actor_id in wolf_ids
            },
        )

    def all_night_knowledge(self) -> NightKnowledge:
        """Everything. Debug and evaluation only, via `TrueWorldPerspective`."""
        return NightKnowledge(
            divines={action.night: action.target_id for action in self.divine_actions},
            guards={action.night: action.target_id for action in self.guard_actions},
            attacks={action.night: action.target_id for action in self.attack_actions},
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
