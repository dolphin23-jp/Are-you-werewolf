"""Central game state: single source of truth, no LLM knowledge."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.engine.phases import Phase
from app.engine.roles import ROLE_DEFINITIONS, RoleName, Team


class DeathCause(StrEnum):
    EXECUTED = "executed"
    ATTACKED = "attacked"
    CURSED = "cursed"
    FIRST_VICTIM = "first_victim"


class PublicDeathCause(StrEnum):
    EXECUTED = "executed"
    NIGHT = "night_death"
    FIRST_VICTIM = "first_victim"


class ChatChannel(StrEnum):
    PUBLIC = "public"
    WOLF = "wolf"
    FREEMASON = "freemason"


@dataclass
class PlayerState:
    player_id: str
    name: str
    role: RoleName
    is_human: bool = False
    alive: bool = True
    death_cause: DeathCause | None = None
    death_day: int | None = None

    @property
    def team(self) -> Team:
        return ROLE_DEFINITIONS[self.role].team


@dataclass
class ChatMessage:
    message_id: str
    author_id: str
    content: str
    channel: ChatChannel
    day: int
    reply_to: str | None = None
    quote: str | None = None
    references: list[str] = field(default_factory=list)


@dataclass
class PendingQuestion:
    asker: str
    target: str
    question: str
    source_message_id: str
    day: int
    topic: str = ""


@dataclass
class DivineRecord:
    seer_id: str
    target_id: str
    day: int
    is_werewolf: bool


@dataclass
class MediumRecord:
    medium_id: str
    target_id: str
    day: int
    is_werewolf: bool


@dataclass
class GuardRecord:
    hunter_id: str
    target_id: str
    day: int


@dataclass
class AttackRecord:
    wolf_id: str
    target_id: str
    day: int
    succeeded: bool


@dataclass
class VoteRecord:
    voter_id: str
    target_id: str
    day: int
    round: int


@dataclass
class DeathRecord:
    player_id: str
    cause: DeathCause
    day: int


@dataclass
class CoDeclaration:
    player_id: str
    claimed_role: RoleName
    day: int


@dataclass
class FreemasonPartnerClaim:
    claimant_id: str
    partner_id: str
    day: int
    confirmed: bool = False


@dataclass
class PublicResultClaim:
    claimant_id: str
    result_type: str
    target_id: str
    is_werewolf: bool
    day: int


@dataclass
class GameState:
    session_id: str
    players: dict[str, PlayerState]
    phase: Phase = Phase.WAITING
    day: int = 0
    vote_round: int = 1
    chat_log: list[ChatMessage] = field(default_factory=list)
    next_message_number: int = 1
    pending_questions: dict[str, list[PendingQuestion]] = field(default_factory=dict)
    divine_records: list[DivineRecord] = field(default_factory=list)
    medium_records: list[MediumRecord] = field(default_factory=list)
    guard_records: list[GuardRecord] = field(default_factory=list)
    attack_records: list[AttackRecord] = field(default_factory=list)
    vote_records: list[VoteRecord] = field(default_factory=list)
    death_records: list[DeathRecord] = field(default_factory=list)
    co_declarations: list[CoDeclaration] = field(default_factory=list)
    freemason_partner_claims: list[FreemasonPartnerClaim] = field(default_factory=list)
    public_result_claims: list[PublicResultClaim] = field(default_factory=list)
    first_victim_id: str | None = None
    typing_channels: dict[str, str] = field(default_factory=dict)
    winner: Team | None = None
    victory_reason: str = ""
    is_draw: bool = False

    # -- pending per-night submissions (cleared after resolution) --
    pending_divine: tuple[str, str] | None = None  # (seer_id, target_id)
    pending_guard: tuple[str, str] | None = None  # (hunter_id, target_id)
    pending_attack: tuple[str, str] | None = None  # (wolf_id, target_id)
    pending_votes: dict[str, str] = field(default_factory=dict)  # voter_id -> target_id
    # In a runoff, only the players tied for the most votes remain eligible.
    # Empty means an ordinary round where anyone alive can be voted for.
    # Without this a runoff is just "vote again over the whole field", which
    # rarely converges and burns through max_vote_rounds into a false draw.
    runoff_candidates: list[str] = field(default_factory=list)

    def votable_ids(self, voter_id: str) -> list[str]:
        """Who this player may vote for right now."""
        pool = self.runoff_candidates or self.alive_ids()
        return [pid for pid in pool if pid != voter_id and self.players[pid].alive]

    def alive_players(self) -> list[PlayerState]:
        return [p for p in self.players.values() if p.alive]

    def alive_ids(self) -> list[str]:
        return [p.player_id for p in self.alive_players()]

    def players_by_role(self, role: RoleName) -> list[PlayerState]:
        return [p for p in self.players.values() if p.role == role]

    def alive_by_team(self, team: Team) -> list[PlayerState]:
        return [p for p in self.alive_players() if p.team == team]

    def get_player_view(self, viewer_id: str) -> dict[str, Any]:
        """Per-viewer filtered view: wolves see wolves, freemasons see their
        partner, everyone else only sees public information."""
        viewer = self.players.get(viewer_id)
        public_players = [
            {
                "player_id": p.player_id,
                "name": p.name,
                "alive": p.alive,
                "death_cause": public_death_cause(p.death_cause),
                "death_day": p.death_day,
            }
            for p in self.players.values()
        ]

        allies: list[str] = []
        if viewer is not None:
            if viewer.role == RoleName.WEREWOLF:
                allies = [
                    p.player_id
                    for p in self.players_by_role(RoleName.WEREWOLF)
                    if p.player_id != viewer_id
                ]
            elif viewer.role == RoleName.FREEMASON:
                allies = [
                    p.player_id
                    for p in self.players_by_role(RoleName.FREEMASON)
                    if p.player_id != viewer_id
                ]

        public_chat = [m for m in self.chat_log if m.channel == ChatChannel.PUBLIC]
        private_chat: list[ChatMessage] = []
        if viewer is not None:
            if viewer.role == RoleName.WEREWOLF:
                private_chat += [m for m in self.chat_log if m.channel == ChatChannel.WOLF]
            if viewer.role == RoleName.FREEMASON:
                private_chat += [m for m in self.chat_log if m.channel == ChatChannel.FREEMASON]

        my_divine = [r for r in self.divine_records if viewer and r.seer_id == viewer_id]
        my_medium = [r for r in self.medium_records if viewer and r.medium_id == viewer_id]

        return {
            "session_id": self.session_id,
            "phase": self.phase,
            "day": self.day,
            "vote_round": self.vote_round,
            "runoff_candidates": list(self.runoff_candidates),
            "your_player_id": viewer_id,
            "your_role": viewer.role if viewer else None,
            "allies": allies,
            "players": public_players,
            "public_chat": [_chat_dict(m) for m in public_chat],
            "private_chat": [_chat_dict(m) for m in private_chat],
            "pending_questions": [
                _pending_question_dict(question)
                for question in self.pending_questions.get(viewer_id, [])
            ],
            "your_divine_results": [_divine_dict(r) for r in my_divine],
            "your_medium_results": [_medium_dict(r) for r in my_medium],
            "co_declarations": [
                {"player_id": c.player_id, "claimed_role": c.claimed_role, "day": c.day}
                for c in self.co_declarations
            ],
            "freemason_partner_claims": [
                {
                    "claimant_id": claim.claimant_id,
                    "partner_id": claim.partner_id,
                    "day": claim.day,
                    "confirmed": claim.confirmed,
                }
                for claim in self.freemason_partner_claims
            ],
            "public_result_claims": [
                {
                    "claimant_id": claim.claimant_id,
                    "result_type": claim.result_type,
                    "target_id": claim.target_id,
                    "is_werewolf": claim.is_werewolf,
                    "day": claim.day,
                }
                for claim in self.public_result_claims
            ],
            "vote_history": [
                {"voter_id": v.voter_id, "target_id": v.target_id, "day": v.day, "round": v.round}
                for v in self.vote_records
            ],
            "first_victim_id": self.first_victim_id,
            "has_voted_current_round": viewer_id in self.pending_votes,
            "typing_player_ids": [
                player_id
                for player_id, channel in self.typing_channels.items()
                if channel == "public"
                or (channel == "wolf" and viewer and viewer.role == RoleName.WEREWOLF)
                or (channel == "freemason" and viewer and viewer.role == RoleName.FREEMASON)
            ],
            "winner": self.winner,
            "victory_reason": self.victory_reason,
            "is_draw": self.is_draw,
        }

    def get_debug_view(self) -> dict[str, Any]:
        """Full, unfiltered spectator/debug dump."""
        return {
            "session_id": self.session_id,
            "phase": self.phase,
            "day": self.day,
            "vote_round": self.vote_round,
            "runoff_candidates": list(self.runoff_candidates),
            "players": [
                {
                    "player_id": p.player_id,
                    "name": p.name,
                    "role": p.role,
                    "team": p.team,
                    "alive": p.alive,
                    "death_cause": p.death_cause,
                    "death_day": p.death_day,
                    "is_human": p.is_human,
                }
                for p in self.players.values()
            ],
            "chat_log": [_chat_dict(m) for m in self.chat_log],
            "pending_questions": {
                target: [_pending_question_dict(question) for question in questions]
                for target, questions in self.pending_questions.items()
                if questions
            },
            "divine_records": [_divine_dict(r) for r in self.divine_records],
            "medium_records": [_medium_dict(r) for r in self.medium_records],
            "guard_records": [
                {"hunter_id": r.hunter_id, "target_id": r.target_id, "day": r.day}
                for r in self.guard_records
            ],
            "attack_records": [
                {
                    "wolf_id": r.wolf_id,
                    "target_id": r.target_id,
                    "day": r.day,
                    "succeeded": r.succeeded,
                }
                for r in self.attack_records
            ],
            "vote_records": [
                {"voter_id": v.voter_id, "target_id": v.target_id, "day": v.day, "round": v.round}
                for v in self.vote_records
            ],
            "death_records": [
                {"player_id": d.player_id, "cause": d.cause, "day": d.day}
                for d in self.death_records
            ],
            "co_declarations": [
                {"player_id": c.player_id, "claimed_role": c.claimed_role, "day": c.day}
                for c in self.co_declarations
            ],
            "freemason_partner_claims": [
                {
                    "claimant_id": claim.claimant_id,
                    "partner_id": claim.partner_id,
                    "day": claim.day,
                    "confirmed": claim.confirmed,
                }
                for claim in self.freemason_partner_claims
            ],
            "winner": self.winner,
            "victory_reason": self.victory_reason,
            "is_draw": self.is_draw,
        }


def _chat_dict(m: ChatMessage) -> dict[str, Any]:
    return {
        "message_id": m.message_id,
        "author_id": m.author_id,
        "content": m.content,
        "channel": m.channel,
        "day": m.day,
        "reply_to": m.reply_to,
        "quote": m.quote,
        "references": list(m.references),
    }


def _pending_question_dict(question: PendingQuestion) -> dict[str, Any]:
    return {
        "asker": question.asker,
        "target": question.target,
        "question": question.question,
        "source_message_id": question.source_message_id,
        "day": question.day,
        "topic": question.topic,
    }


def public_death_cause(cause: DeathCause | None) -> PublicDeathCause | None:
    """The only death information the table is allowed to see.

    Attacked and cursed both surface as a plain night death; anything reading
    a public projection must go through here rather than `PlayerState.death_cause`.
    """
    if cause is None:
        return None
    if cause == DeathCause.EXECUTED:
        return PublicDeathCause.EXECUTED
    if cause == DeathCause.FIRST_VICTIM:
        return PublicDeathCause.FIRST_VICTIM
    return PublicDeathCause.NIGHT


def _divine_dict(r: DivineRecord) -> dict[str, Any]:
    return {
        "seer_id": r.seer_id,
        "target_id": r.target_id,
        "day": r.day,
        "is_werewolf": r.is_werewolf,
    }


def _medium_dict(r: MediumRecord) -> dict[str, Any]:
    return {
        "medium_id": r.medium_id,
        "target_id": r.target_id,
        "day": r.day,
        "is_werewolf": r.is_werewolf,
    }
