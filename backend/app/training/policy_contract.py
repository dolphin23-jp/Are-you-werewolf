"""Framework-agnostic output contract for learned policies.

A future neural model may implement these heads with PyTorch, JAX or another
library. Keeping the contract here lets the game/training environment remain
independent from that choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.engine.roles import RoleName
from app.training.actions import (
    ActionType,
    ResultValue,
    Scope,
    Stance,
    TimingBucket,
    Topic,
)
from app.training.encoding import (
    MAX_SEATS,
    MAX_SEMANTIC_EVENTS,
    EncodedPolicyObservation,
)

MAX_REFERENCED_DAYS = 16
MAX_QUANTITY = 3


@dataclass(frozen=True)
class PolicyHeadSizes:
    timing: int = len(TimingBucket)
    action_type: int = len(ActionType)
    topic: int = len(Topic)
    target: int = MAX_SEATS
    secondary_target: int = MAX_SEATS
    role: int = len(RoleName)
    result: int = len(ResultValue)
    quantity: int = MAX_QUANTITY + 1
    referenced_day: int = MAX_REFERENCED_DAYS + 1
    scope: int = len(Scope)
    stance: int = len(Stance)
    reference_event: int = MAX_SEMANTIC_EVENTS
    vote_target: int = MAX_SEATS
    night_topic: int = 3
    night_target: int = MAX_SEATS


@dataclass(frozen=True)
class PolicyLogits:
    """One model forward pass before legal masks and sampling are applied."""

    timing: tuple[float, ...]
    action_type: tuple[float, ...]
    topic: tuple[float, ...]
    target: tuple[float, ...]
    secondary_target: tuple[float, ...]
    role: tuple[float, ...]
    result: tuple[float, ...]
    quantity: tuple[float, ...]
    referenced_day: tuple[float, ...]
    scope: tuple[float, ...]
    stance: tuple[float, ...]
    reference_event: tuple[float, ...]
    vote_target: tuple[float, ...]
    night_topic: tuple[float, ...]
    night_target: tuple[float, ...]
    value: float

    def validate(self, sizes: PolicyHeadSizes | None = None) -> None:
        sizes = sizes or PolicyHeadSizes()
        actual = (
            ("timing", len(self.timing), sizes.timing),
            ("action_type", len(self.action_type), sizes.action_type),
            ("topic", len(self.topic), sizes.topic),
            ("target", len(self.target), sizes.target),
            ("secondary_target", len(self.secondary_target), sizes.secondary_target),
            ("role", len(self.role), sizes.role),
            ("result", len(self.result), sizes.result),
            ("quantity", len(self.quantity), sizes.quantity),
            ("referenced_day", len(self.referenced_day), sizes.referenced_day),
            ("scope", len(self.scope), sizes.scope),
            ("stance", len(self.stance), sizes.stance),
            ("reference_event", len(self.reference_event), sizes.reference_event),
            ("vote_target", len(self.vote_target), sizes.vote_target),
            ("night_topic", len(self.night_topic), sizes.night_topic),
            ("night_target", len(self.night_target), sizes.night_target),
        )
        for name, width, expected_width in actual:
            if width != expected_width:
                raise ValueError(
                    f"{name} head has width {width}; expected {expected_width}"
                )


class LearnedPolicyModel(Protocol):
    """Minimal interface expected from a trainable policy/value network."""

    def forward(self, observation: EncodedPolicyObservation) -> PolicyLogits: ...
