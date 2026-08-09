"""Minimal clipped-PPO optimizer for the Phase-1 NumPy policy baseline.

Only terminal team reward is used. There is no role-specific bonus, CO bonus,
kill bonus, or other hand-authored strategic shaping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.training.numpy_policy import FloatArray, NumpyMLPPolicy
from app.training.policy_sampling import HeadChoice, PolicySampleTrace
from app.training.trajectory import EpisodeTrajectory


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    epochs: int = 2
    max_grad_norm: float = 5.0


@dataclass(frozen=True)
class PPOUpdateStats:
    decisions: int
    epochs: int
    mean_policy_loss: float
    mean_value_loss: float
    mean_ratio: float
    clip_fraction: float
    gradient_norm: float


class NumpyPPOTrainer:
    """Update a shared policy/value network from completed self-play episodes."""

    def __init__(
        self,
        model: NumpyMLPPolicy,
        config: PPOConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or PPOConfig()
        if self.config.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.config.clip_ratio < 1:
            raise ValueError("clip_ratio must be in [0, 1)")
        if self.config.value_coefficient < 0:
            raise ValueError("value_coefficient cannot be negative")
        if self.config.epochs <= 0:
            raise ValueError("epochs must be positive")

    def update(self, trajectories: list[EpisodeTrajectory]) -> PPOUpdateStats:
        decisions = [
            decision
            for trajectory in trajectories
            for decision in trajectory.decisions
            if decision.policy_trace is not None
        ]
        if not decisions:
            return PPOUpdateStats(0, 0, 0.0, 0.0, 1.0, 0.0, 0.0)
        if any(not trajectory.finalized for trajectory in trajectories):
            raise ValueError("PPO update requires finalized trajectories")

        policy_losses: list[float] = []
        value_losses: list[float] = []
        ratios: list[float] = []
        clipped: list[float] = []
        gradient_norm = 0.0

        for _ in range(self.config.epochs):
            examples = []
            for decision in decisions:
                trace = decision.policy_trace
                if trace is None:
                    continue
                forward = self.model.forward_vector(decision.observation)
                grad_output = np.zeros(self.model.layout.output_size, dtype=np.float64)
                current_log_prob = self._current_log_prob_and_gradient(
                    forward.output,
                    trace,
                    grad_output,
                    coefficient=0.0,
                )
                log_ratio = max(-20.0, min(20.0, current_log_prob - trace.log_prob))
                ratio = math.exp(log_ratio)
                advantage = decision.reward - trace.value_estimate
                active = self._ppo_gradient_is_active(ratio, advantage)
                policy_coefficient = -advantage * ratio if active else 0.0

                if policy_coefficient != 0.0:
                    grad_output.fill(0.0)
                    self._current_log_prob_and_gradient(
                        forward.output,
                        trace,
                        grad_output,
                        coefficient=policy_coefficient,
                    )

                value = float(forward.output[-1])
                value_error = value - decision.reward
                grad_output[-1] += self.config.value_coefficient * value_error

                clipped_ratio = min(
                    1.0 + self.config.clip_ratio,
                    max(1.0 - self.config.clip_ratio, ratio),
                )
                surrogate = min(ratio * advantage, clipped_ratio * advantage)
                policy_losses.append(-surrogate)
                value_losses.append(
                    0.5 * self.config.value_coefficient * value_error * value_error
                )
                ratios.append(ratio)
                clipped.append(float(not active))
                examples.append((forward, grad_output))

            gradient_norm = self.model.apply_output_gradients(
                examples,
                learning_rate=self.config.learning_rate,
                max_grad_norm=self.config.max_grad_norm,
            )

        return PPOUpdateStats(
            decisions=len(decisions),
            epochs=self.config.epochs,
            mean_policy_loss=float(np.mean(policy_losses)),
            mean_value_loss=float(np.mean(value_losses)),
            mean_ratio=float(np.mean(ratios)),
            clip_fraction=float(np.mean(clipped)),
            gradient_norm=gradient_norm,
        )

    def _current_log_prob_and_gradient(
        self,
        output: FloatArray,
        trace: PolicySampleTrace,
        grad_output: FloatArray,
        *,
        coefficient: float,
    ) -> float:
        log_prob = 0.0
        for choice in trace.choices:
            span = self.model.layout.span(choice.head)
            head_logits = output[span]
            choice_log_prob, probabilities = _masked_log_prob(head_logits, choice)
            log_prob += choice_log_prob
            if coefficient == 0.0:
                continue
            start = span.start or 0
            for index, probability in zip(
                choice.valid_indices,
                probabilities,
                strict=True,
            ):
                derivative = (1.0 if index == choice.index else 0.0) - probability
                grad_output[start + index] += coefficient * derivative
        return log_prob

    def _ppo_gradient_is_active(self, ratio: float, advantage: float) -> bool:
        if advantage >= 0:
            return ratio <= 1.0 + self.config.clip_ratio
        return ratio >= 1.0 - self.config.clip_ratio


def _masked_log_prob(
    logits: FloatArray,
    choice: HeadChoice,
) -> tuple[float, tuple[float, ...]]:
    if not choice.valid_indices:
        raise ValueError(f"{choice.head} trace has no legal indices")
    if choice.index not in choice.valid_indices:
        raise ValueError(f"{choice.head} selected index is not legal")
    selected_logits = np.asarray(
        [logits[index] for index in choice.valid_indices],
        dtype=np.float64,
    )
    peak = float(np.max(selected_logits))
    weights = np.exp(selected_logits - peak)
    probabilities_array = weights / np.sum(weights)
    selected_offset = choice.valid_indices.index(choice.index)
    probability = float(probabilities_array[selected_offset])
    return math.log(probability), tuple(float(value) for value in probabilities_array)
