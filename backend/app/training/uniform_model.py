"""Uniform-logit model used to validate the learned-policy execution path."""

from __future__ import annotations

from app.training.encoding import EncodedPolicyObservation
from app.training.policy_contract import PolicyHeadSizes, PolicyLogits


def _zeros(width: int) -> tuple[float, ...]:
    return (0.0,) * width


class UniformPolicyModel:
    """Return equal logits for every head and a zero value estimate."""

    def __init__(self, sizes: PolicyHeadSizes | None = None) -> None:
        self.sizes = sizes or PolicyHeadSizes()

    def forward(self, observation: EncodedPolicyObservation) -> PolicyLogits:
        del observation
        sizes = self.sizes
        return PolicyLogits(
            timing=_zeros(sizes.timing),
            action_type=_zeros(sizes.action_type),
            topic=_zeros(sizes.topic),
            target=_zeros(sizes.target),
            secondary_target=_zeros(sizes.secondary_target),
            role=_zeros(sizes.role),
            result=_zeros(sizes.result),
            quantity=_zeros(sizes.quantity),
            referenced_day=_zeros(sizes.referenced_day),
            scope=_zeros(sizes.scope),
            stance=_zeros(sizes.stance),
            reference_event=_zeros(sizes.reference_event),
            vote_target=_zeros(sizes.vote_target),
            night_topic=_zeros(sizes.night_topic),
            night_target=_zeros(sizes.night_target),
            value=0.0,
        )
