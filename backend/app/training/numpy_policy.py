"""Small NumPy policy/value network for Phase-1 self-play smoke training.

This is intentionally not the final architecture. It proves that the complete
observation -> logits -> masked action -> terminal reward -> parameter update
loop works before a heavier Transformer implementation is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.training.encoding import EncodedPolicyObservation
from app.training.policy_contract import PolicyHeadSizes, PolicyLogits

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PolicyHeadLayout:
    names: tuple[str, ...]
    widths: tuple[int, ...]
    starts: tuple[int, ...]
    output_size: int

    def span(self, name: str) -> slice:
        index = self.names.index(name)
        start = self.starts[index]
        return slice(start, start + self.widths[index])


def policy_head_layout(sizes: PolicyHeadSizes | None = None) -> PolicyHeadLayout:
    sizes = sizes or PolicyHeadSizes()
    items = (
        ("timing", sizes.timing),
        ("action_type", sizes.action_type),
        ("topic", sizes.topic),
        ("target", sizes.target),
        ("secondary_target", sizes.secondary_target),
        ("role", sizes.role),
        ("result", sizes.result),
        ("quantity", sizes.quantity),
        ("referenced_day", sizes.referenced_day),
        ("scope", sizes.scope),
        ("stance", sizes.stance),
        ("reference_event", sizes.reference_event),
        ("vote_target", sizes.vote_target),
        ("night_topic", sizes.night_topic),
        ("night_target", sizes.night_target),
    )
    names = tuple(name for name, _ in items)
    widths = tuple(width for _, width in items)
    starts: list[int] = []
    cursor = 0
    for width in widths:
        starts.append(cursor)
        cursor += width
    return PolicyHeadLayout(names, widths, tuple(starts), cursor + 1)


def flatten_observation(observation: EncodedPolicyObservation) -> FloatArray:
    """Flatten all encoded information, including explicit history masks."""
    values: list[float] = []
    values.extend(observation.global_features)
    for token in observation.player_tokens:
        values.extend(token)
    for token in observation.semantic_tokens:
        values.extend(token)
    values.extend(observation.semantic_mask)
    for token in observation.vote_tokens:
        values.extend(token)
    values.extend(observation.vote_mask)
    for token in observation.dawn_tokens:
        values.extend(token)
    values.extend(observation.dawn_mask)
    raw = np.asarray(values, dtype=np.float64)
    # Encoded categories and logical times are small non-negative integers. A
    # fixed squashing scale keeps gradients stable without dataset statistics.
    return np.tanh(raw / 16.0)


@dataclass(frozen=True)
class NetworkForward:
    observation_vector: FloatArray
    hidden: FloatArray
    output: FloatArray


class NumpyMLPPolicy:
    """One-hidden-layer shared policy/value network with manual backprop."""

    def __init__(
        self,
        *,
        hidden_size: int = 64,
        seed: int | None = None,
        sizes: PolicyHeadSizes | None = None,
    ) -> None:
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.sizes = sizes or PolicyHeadSizes()
        self.layout = policy_head_layout(self.sizes)
        self.hidden_size = hidden_size
        self._rng = np.random.default_rng(seed)
        self._initialized = False
        self.w1 = np.empty((0, 0), dtype=np.float64)
        self.b1 = np.zeros(hidden_size, dtype=np.float64)
        self.w2 = np.empty((0, 0), dtype=np.float64)
        self.b2 = np.zeros(self.layout.output_size, dtype=np.float64)

    @property
    def initialized(self) -> bool:
        return self._initialized

    def forward(self, observation: EncodedPolicyObservation) -> PolicyLogits:
        forward = self.forward_vector(observation)
        return self._vector_to_logits(forward.output)

    def forward_vector(self, observation: EncodedPolicyObservation) -> NetworkForward:
        x = flatten_observation(observation)
        self._ensure_initialized(x.size)
        hidden = np.tanh(x @ self.w1 + self.b1)
        output = hidden @ self.w2 + self.b2
        return NetworkForward(x, hidden, output)

    def apply_output_gradients(
        self,
        examples: list[tuple[NetworkForward, FloatArray]],
        *,
        learning_rate: float,
        max_grad_norm: float = 5.0,
    ) -> float:
        """Backprop gradients d(loss)/d(output) through the tiny MLP."""
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not examples:
            return 0.0
        if not self._initialized:
            raise RuntimeError("network must be initialized before updating")

        grad_w1 = np.zeros_like(self.w1)
        grad_b1 = np.zeros_like(self.b1)
        grad_w2 = np.zeros_like(self.w2)
        grad_b2 = np.zeros_like(self.b2)

        for forward, grad_output in examples:
            if grad_output.shape != (self.layout.output_size,):
                raise ValueError("output gradient has unexpected width")
            grad_w2 += np.outer(forward.hidden, grad_output)
            grad_b2 += grad_output
            grad_hidden = self.w2 @ grad_output
            grad_pre_hidden = grad_hidden * (1.0 - np.square(forward.hidden))
            grad_w1 += np.outer(forward.observation_vector, grad_pre_hidden)
            grad_b1 += grad_pre_hidden

        scale = 1.0 / len(examples)
        grad_w1 *= scale
        grad_b1 *= scale
        grad_w2 *= scale
        grad_b2 *= scale

        norm = float(
            np.sqrt(
                np.sum(np.square(grad_w1))
                + np.sum(np.square(grad_b1))
                + np.sum(np.square(grad_w2))
                + np.sum(np.square(grad_b2))
            )
        )
        if max_grad_norm > 0 and norm > max_grad_norm:
            clip_scale = max_grad_norm / norm
            grad_w1 *= clip_scale
            grad_b1 *= clip_scale
            grad_w2 *= clip_scale
            grad_b2 *= clip_scale
            norm = max_grad_norm

        self.w1 -= learning_rate * grad_w1
        self.b1 -= learning_rate * grad_b1
        self.w2 -= learning_rate * grad_w2
        self.b2 -= learning_rate * grad_b2
        return norm

    def parameter_vector(self) -> FloatArray:
        if not self._initialized:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(
            (self.w1.ravel(), self.b1, self.w2.ravel(), self.b2)
        ).copy()

    def _ensure_initialized(self, input_size: int) -> None:
        if self._initialized:
            if self.w1.shape[0] != input_size:
                raise ValueError("observation width changed after network initialization")
            return
        scale1 = np.sqrt(2.0 / (input_size + self.hidden_size))
        scale2 = np.sqrt(2.0 / (self.hidden_size + self.layout.output_size))
        self.w1 = self._rng.normal(0.0, scale1, (input_size, self.hidden_size))
        self.w2 = self._rng.normal(
            0.0,
            scale2,
            (self.hidden_size, self.layout.output_size),
        )
        self._initialized = True

    def _vector_to_logits(self, output: FloatArray) -> PolicyLogits:
        if output.shape != (self.layout.output_size,):
            raise ValueError("network output has unexpected width")

        def head(name: str) -> tuple[float, ...]:
            return tuple(float(value) for value in output[self.layout.span(name)])

        value = float(output[-1])
        return PolicyLogits(
            timing=head("timing"),
            action_type=head("action_type"),
            topic=head("topic"),
            target=head("target"),
            secondary_target=head("secondary_target"),
            role=head("role"),
            result=head("result"),
            quantity=head("quantity"),
            referenced_day=head("referenced_day"),
            scope=head("scope"),
            stance=head("stance"),
            reference_event=head("reference_event"),
            vote_target=head("vote_target"),
            night_topic=head("night_topic"),
            night_target=head("night_target"),
            value=value,
        )
