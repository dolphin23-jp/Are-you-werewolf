"""`PublicFactLedger`: the one read-only projection of `GameState` that the AI
reasoning layer is allowed to treat as established fact.

Two properties matter more than convenience here:

* **No second copy.** The ledger holds a reference to the live `GameState` and
  derives every view on demand. `co_declarations`, `public_result_claims` and
  `vote_records` stay the single source of truth -- the ledger is a reader, so
  it can never drift out of sync with the engine the way a snapshot would.
* **No private truth.** True roles, real death causes, night actions and other
  players' private ability results never cross this boundary. Deaths are
  reported through :func:`app.engine.state.public_death_cause`, which collapses
  attacked/cursed into an indistinguishable night death.

Everything above this module -- validation, deterministic summaries -- reads the
board through the ledger, so a state-consistency rule only has to be written once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.speech_events import (
    MEDIUM_RESULT,
    RESULT_TYPES,
    SEER_RESULT,
    ResultVersion,
    RoleClaimState,
    SpeechEvent,
    current_role_claim,
    events_for_message,
    result_versions,
    role_claim_history,
)
from app.engine.state import DeathCause, GameState, PublicDeathCause, public_death_cause

# Re-exported so the reasoning layer has one import site for public-claim
# vocabulary, whether it comes from the engine's event model or from here.
__all__ = [
    "MEDIUM_RESULT",
    "RESULT_TYPES",
    "SEER_RESULT",
    "PublicCoFact",
    "PublicExecutionFact",
    "PublicFactLedger",
    "PublicPlayerFact",
    "PublicResultFact",
    "PublicVoteFact",
    "ResultVersion",
    "RoleClaimState",
    "SpeechEvent",
    "mentions_player",
]


@dataclass(frozen=True)
class PublicPlayerFact:
    player_id: str
    name: str
    alive: bool
    death_cause: PublicDeathCause | None = None
    death_day: int | None = None

    @property
    def label(self) -> str:
        return f"{self.name}({self.player_id})"


@dataclass(frozen=True)
class PublicCoFact:
    player_id: str
    claimed_role: RoleName
    day: int
    source_message_id: str = ""


@dataclass(frozen=True)
class PublicResultFact:
    claimant_id: str
    result_type: str
    target_id: str
    is_werewolf: bool
    day: int
    source_message_id: str = ""


@dataclass(frozen=True)
class PublicVoteFact:
    voter_id: str
    target_id: str
    day: int
    round: int


@dataclass(frozen=True)
class PublicExecutionFact:
    player_id: str
    day: int


def mentions_player(text: str, player_id: str, name: str = "") -> bool:
    """Whether `text` names this specific player.

    A plain substring test makes `p1` match `p11` (and `Player1` match
    `Player11`), which is exactly the misidentification that turns one player's
    published result into a claim about somebody else. Both forms are matched
    with a trailing-digit guard so neighbouring ids stay distinct.
    """
    patterns = [re.escape(player_id)]
    if name:
        patterns.append(re.escape(name))
    return any(
        re.search(rf"(?<![0-9A-Za-z]){pattern}(?![0-9])", text) is not None
        for pattern in patterns
    )


class PublicFactLedger:
    """Read-only public view over a live `GameState`."""

    def __init__(self, state: GameState) -> None:
        self._state = state

    # -- board basics --

    @property
    def day(self) -> int:
        return self._state.day

    @property
    def phase(self) -> Phase:
        return self._state.phase

    @property
    def vote_round(self) -> int:
        return self._state.vote_round

    @property
    def first_victim_id(self) -> str | None:
        return self._state.first_victim_id

    @property
    def runoff_candidates(self) -> tuple[str, ...]:
        return tuple(self._state.runoff_candidates)

    # -- players --

    def players(self) -> tuple[PublicPlayerFact, ...]:
        return tuple(
            PublicPlayerFact(
                player_id=player.player_id,
                name=player.name,
                alive=player.alive,
                death_cause=public_death_cause(player.death_cause),
                death_day=player.death_day,
            )
            for player in self._state.players.values()
        )

    def player(self, player_id: str) -> PublicPlayerFact | None:
        player = self._state.players.get(player_id)
        if player is None:
            return None
        return PublicPlayerFact(
            player_id=player.player_id,
            name=player.name,
            alive=player.alive,
            death_cause=public_death_cause(player.death_cause),
            death_day=player.death_day,
        )

    def known_player_ids(self) -> tuple[str, ...]:
        """Seating order, not sorted(): `sorted` reads as p0, p1, p10, p11, p2."""
        return tuple(self._state.players)

    def is_known(self, player_id: str | None) -> bool:
        return player_id is not None and player_id in self._state.players

    def is_alive(self, player_id: str | None) -> bool:
        player = self._state.players.get(player_id or "")
        return player is not None and player.alive

    def name_of(self, player_id: str) -> str:
        player = self._state.players.get(player_id)
        return player.name if player is not None else player_id

    def label_of(self, player_id: str) -> str:
        player = self._state.players.get(player_id)
        return f"{player.name}({player_id})" if player is not None else player_id

    def alive_ids(self) -> tuple[str, ...]:
        return tuple(pid for pid, p in self._state.players.items() if p.alive)

    def dead_ids(self) -> tuple[str, ...]:
        return tuple(pid for pid, p in self._state.players.items() if not p.alive)

    def mentioned_player_ids(self, text: str) -> tuple[str, ...]:
        return tuple(
            pid
            for pid, player in self._state.players.items()
            if mentions_player(text, pid, player.name)
        )

    # -- deaths --

    def executions(self) -> tuple[PublicExecutionFact, ...]:
        """Executions in the order they happened. Announced publicly, so this
        is fact, not inference."""
        return tuple(
            PublicExecutionFact(player_id=record.player_id, day=record.day)
            for record in self._state.death_records
            if record.cause == DeathCause.EXECUTED
        )

    def executed_ids(self) -> tuple[str, ...]:
        return tuple(execution.player_id for execution in self.executions())

    def execution_day(self, player_id: str) -> int | None:
        for execution in self.executions():
            if execution.player_id == player_id:
                return execution.day
        return None

    def was_executed(self, player_id: str) -> bool:
        return self.execution_day(player_id) is not None

    def night_death_ids(self, day: int | None = None) -> tuple[str, ...]:
        return tuple(
            record.player_id
            for record in self._state.death_records
            if public_death_cause(record.cause) == PublicDeathCause.NIGHT
            and (day is None or record.day == day)
        )

    # -- claims --

    def co_declarations(self) -> tuple[PublicCoFact, ...]:
        return tuple(
            PublicCoFact(
                player_id=claim.player_id,
                claimed_role=claim.claimed_role,
                day=claim.day,
                source_message_id=claim.source_message_id,
            )
            for claim in self._state.co_declarations
        )

    def claimed_role_of(self, player_id: str) -> RoleName | None:
        """The claim standing now: None if never made, retracted, or the new role
        after a slide."""
        claim = current_role_claim(self._state.speech_events, player_id)
        return claim.role if claim is not None else None

    def speech_events(self) -> tuple[SpeechEvent, ...]:
        """The raw public-claim log. Every derived view below comes from it, and
        every claim in it points back at the message that made it."""
        return tuple(self._state.speech_events)

    def events_for_message(self, message_id: str) -> tuple[SpeechEvent, ...]:
        return events_for_message(self._state.speech_events, message_id)

    def role_claim_history(self, player_id: str | None = None) -> tuple[RoleClaimState, ...]:
        """Claimed, retracted, slid -- in the order it happened. Separate from
        `co_declarations`, which is only what stands right now."""
        return role_claim_history(self._state.speech_events, player_id)

    def current_role_claim(self, player_id: str) -> RoleClaimState | None:
        return current_role_claim(self._state.speech_events, player_id)

    def result_versions(
        self,
        *,
        claimant_id: str | None = None,
        result_type: str | None = None,
        target_id: str | None = None,
    ) -> tuple[ResultVersion, ...]:
        """Including superseded and retracted ones, so "what did they say before
        the correction" stays answerable."""
        return result_versions(
            self._state.speech_events,
            claimant_id=claimant_id,
            result_type=result_type,
            target_id=target_id,
        )

    def public_results(self) -> tuple[PublicResultFact, ...]:
        return tuple(
            PublicResultFact(
                claimant_id=claim.claimant_id,
                result_type=claim.result_type,
                target_id=claim.target_id,
                is_werewolf=claim.is_werewolf,
                day=claim.day,
                source_message_id=claim.source_message_id,
            )
            for claim in self._state.public_result_claims
        )

    def find_result(
        self, claimant_id: str, result_type: str, target_id: str
    ) -> PublicResultFact | None:
        """The verdict standing right now for this exact claim.

        A restatement that disagrees with it is corruption; changing it requires
        an explicit correction, which supersedes rather than overwrites.
        """
        for result in self.public_results():
            if (
                result.claimant_id == claimant_id
                and result.result_type == result_type
                and result.target_id == target_id
            ):
                return result
        return None

    # -- votes --

    def votes(self) -> tuple[PublicVoteFact, ...]:
        return tuple(
            PublicVoteFact(
                voter_id=vote.voter_id,
                target_id=vote.target_id,
                day=vote.day,
                round=vote.round,
            )
            for vote in self._state.vote_records
        )

    def votes_on(self, day: int, round_number: int | None = None) -> tuple[PublicVoteFact, ...]:
        return tuple(
            vote
            for vote in self.votes()
            if vote.day == day and (round_number is None or vote.round == round_number)
        )

    def vote_of(
        self, voter_id: str, day: int, round_number: int | None = None
    ) -> PublicVoteFact | None:
        """Who this player actually voted for -- the last ballot they cast in the
        requested day/round, never an inference from what they said."""
        found: PublicVoteFact | None = None
        for vote in self.votes_on(day, round_number):
            if vote.voter_id == voter_id:
                found = vote
        return found

    def votable_ids(self, voter_id: str) -> tuple[str, ...]:
        return tuple(self._state.votable_ids(voter_id))

    # -- diagnostics --

    def to_public_dict(self) -> dict[str, object]:
        """Whole ledger as plain data. Used by tests to assert that no private
        role, night action or unpublished ability result can reach a public view."""
        return {
            "day": self.day,
            "phase": self.phase.value,
            "vote_round": self.vote_round,
            "first_victim_id": self.first_victim_id,
            "runoff_candidates": list(self.runoff_candidates),
            "players": [
                {
                    "player_id": p.player_id,
                    "name": p.name,
                    "alive": p.alive,
                    "death_cause": p.death_cause.value if p.death_cause else None,
                    "death_day": p.death_day,
                }
                for p in self.players()
            ],
            "co_declarations": [
                {"player_id": c.player_id, "claimed_role": c.claimed_role.value, "day": c.day}
                for c in self.co_declarations()
            ],
            "public_results": [
                {
                    "claimant_id": r.claimant_id,
                    "result_type": r.result_type,
                    "target_id": r.target_id,
                    "is_werewolf": r.is_werewolf,
                    "day": r.day,
                }
                for r in self.public_results()
            ],
            "votes": [
                {
                    "voter_id": v.voter_id,
                    "target_id": v.target_id,
                    "day": v.day,
                    "round": v.round,
                }
                for v in self.votes()
            ],
            "executions": [
                {"player_id": e.player_id, "day": e.day} for e in self.executions()
            ],
        }
