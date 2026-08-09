"""Structured actions used by self-play and by the human semantic bridge.

These types describe *what can be said or done*, not strategic doctrine. A
policy may choose a poor, deceptive, late, contradictory, or unconventional
action as long as the underlying game rules permit it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum

from app.engine.roles import RoleName


class ActionType(StrEnum):
    PASS = "pass"
    CLAIM = "claim"
    REPORT = "report"
    EVALUATE = "evaluate"
    DECLARE = "declare"
    PROPOSE = "propose"
    QUESTION = "question"
    REACT = "react"
    RETRACT = "retract"
    CORRECT = "correct"
    VOTE = "vote"
    NIGHT_ACTION = "night_action"
    PRIVATE_PLAN = "private_plan"


class TimingBucket(IntEnum):
    """Logical speech timing; wall-clock latency must never become strategy."""

    IMMEDIATE = 0
    EARLY = 1
    NORMAL = 2
    LATE = 3
    HOLD = 4


class Topic(StrEnum):
    GENERAL = "general"
    ROLE = "role"
    PARTNER = "partner"
    SEER_RESULT = "seer_result"
    MEDIUM_RESULT = "medium_result"
    GUARD = "guard"
    ATTACK = "attack"
    EXECUTION = "execution"
    VOTE = "vote"
    DIVINE = "divine"
    CO_REQUEST = "co_request"
    CO_INTENTION = "co_intention"
    VOTE_REASON = "vote_reason"
    SEER_AUTHENTICITY = "seer_authenticity"
    MEDIUM_AUTHENTICITY = "medium_authenticity"
    HUNTER_AUTHENTICITY = "hunter_authenticity"
    FREEMASON_AUTHENTICITY = "freemason_authenticity"
    WOLF = "wolf"
    WOLF_COUNT = "wolf_count"
    FOX = "fox"
    MADMAN = "madman"


class Scope(StrEnum):
    NONE = "none"
    SELF = "self"
    ALL = "all"
    ALIVE = "alive"
    UNCLAIMED = "unclaimed"
    SEER_CLAIMANTS = "seer_claimants"
    MEDIUM_CLAIMANTS = "medium_claimants"
    HUNTER_CLAIMANTS = "hunter_claimants"
    FREEMASON_CLAIMANTS = "freemason_claimants"
    FREE_CHOICE = "free_choice"


class Stance(StrEnum):
    ASSERT = "assert"
    TRUST = "trust"
    SUSPECT = "suspect"
    SUPPORT = "support"
    OPPOSE = "oppose"
    NEUTRAL = "neutral"


class ResultValue(StrEnum):
    WHITE = "white"
    BLACK = "black"
    UNKNOWN = "unknown"


class Channel(StrEnum):
    PUBLIC = "public"
    WOLF = "wolf"
    FREEMASON = "freemason"


@dataclass(frozen=True)
class SemanticAction:
    action_type: ActionType
    topic: Topic | None = None
    target_id: str | None = None
    secondary_target_id: str | None = None
    role: RoleName | None = None
    result: ResultValue | None = None
    quantity: int | None = None
    referenced_day: int | None = None
    scope: Scope | None = None
    stance: Stance | None = None
    channel: Channel = Channel.PUBLIC
    reference_event_id: str | None = None


@dataclass(frozen=True)
class SpeechBundle:
    """One human-like turn, composed from at most three semantic atoms."""

    atoms: tuple[SemanticAction, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.atoms) <= 3:
            raise ValueError("a speech bundle must contain between one and three atoms")
        forbidden = {ActionType.VOTE, ActionType.NIGHT_ACTION}
        if any(atom.action_type in forbidden for atom in self.atoms):
            raise ValueError("vote/night execution actions cannot be embedded in speech")


@dataclass(frozen=True)
class TimedSemanticEvent:
    event_id: str
    actor_id: str
    day: int
    discussion_tick: int
    action: SemanticAction
