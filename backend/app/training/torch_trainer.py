"""Batched clipped-PPO optimizer for the PyTorch Transformer policy.

The trainer reuses the framework-agnostic rollout traces. Only indices that were
legal at decision time participate in each masked log-probability. Rewards are
still sparse terminal faction rewards; no strategic shaping is added here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from app.training.numpy_trainer import PPOUpdateStats
from app.training.policy_sampling import PolicySampleTrace
from app.training.torch_policy import TorchPolicyTensorOutput, TorchTransformerPolicy
from app.training.trajectory import EpisodeTrajectory, RecordedDecision


@dataclass(frozen=True)
class TorchPPOConfig:
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    epochs: int = 2
    max_grad_norm: float = 5.0
    minibatch_size: int = 64

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.clip_ratio < 1:
            raise ValueError("clip_ratio must be in [0, 1)")
        if self.value_coefficient < 0:
            raise ValueError("value_coefficient cannot be negative")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.max_grad_norm < 0:
            raise ValueError("max_grad_norm cannot be negative")
        if self.minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")


class TorchPPOTrainer:
    """Update a Transformer policy/value network from finalized trajectories."""

    def __init__(
        self,
        model: TorchTransformerPolicy,
        config: TorchPPOConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.config = config or TorchPPOConfig()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)

    def update(self, trajectories: list[EpisodeTrajectory]) -> PPOUpdateStats:
        if any(not trajectory.finalized for trajectory in trajectories):
            raise ValueError("PPO update requires finalized trajectories")
        decisions = [
            decision
            for trajectory in trajectories
            for decision in trajectory.decisions
            if decision.policy_trace is not None
        ]
        if not decisions:
            return PPOUpdateStats(0, 0, 0.0, 0.0, 1.0, 0.0, 0.0)

        policy_losses: list[float] = []
        value_losses: list[float] = []
        ratios: list[float] = []
        clipped: list[float] = []
        gradient_norm = 0.0
        self.model.train()

        for _ in range(self.config.epochs):
            permutation = torch.randperm(len(decisions), generator=self._generator).tolist()
            for start in range(0, len(permutation), self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                batch = [decisions[index] for index in indices]
                batch_stats = self._update_minibatch(batch)
                policy_losses.append(batch_stats[0])
                value_losses.append(batch_stats[1])
                ratios.extend(batch_stats[2])
                clipped.extend(batch_stats[3])
                gradient_norm = batch_stats[4]

        return PPOUpdateStats(
            decisions=len(decisions),
            epochs=self.config.epochs,
            mean_policy_loss=_mean(policy_losses),
            mean_value_loss=_mean(value_losses),
            mean_ratio=_mean(ratios),
            clip_fraction=_mean(clipped),
            gradient_norm=gradient_norm,
        )

    def _update_minibatch(
        self,
        decisions: list[RecordedDecision],
    ) -> tuple[float, float, list[float], list[float], float]:
        output = self.model.forward_batch(
            tuple(decision.observation for decision in decisions)
        )
        traces = [decision.policy_trace for decision in decisions]
        if any(trace is None for trace in traces):
            raise RuntimeError("Torch PPO minibatch contains a decision without a trace")
        typed_traces = [trace for trace in traces if trace is not None]

        current_log_probs = torch.stack(
            [
                _trace_log_prob(output, batch_index, trace)
                for batch_index, trace in enumerate(typed_traces)
            ]
        )
        device = output.value.device
        old_log_probs = torch.tensor(
            [trace.log_prob for trace in typed_traces],
            dtype=output.value.dtype,
            device=device,
        )
        rewards = torch.tensor(
            [decision.reward for decision in decisions],
            dtype=output.value.dtype,
            device=device,
        )
        advantages = torch.tensor(
            [
                decision.reward - trace.value_estimate
                for decision, trace in zip(decisions, typed_traces, strict=True)
            ],
            dtype=output.value.dtype,
            device=device,
        )

        log_ratios = torch.clamp(current_log_probs - old_log_probs, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratios)
        clipped_ratio = torch.clamp(
            ratio,
            min=1.0 - self.config.clip_ratio,
            max=1.0 + self.config.clip_ratio,
        )
        surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
        policy_loss = -surrogate.mean()
        value_error = output.value - rewards
        value_loss = (
            0.5 * self.config.value_coefficient * torch.square(value_error).mean()
        )
        loss = policy_loss + value_loss

        self.optimizer.zero_grad(set_to_none=True)
        # PyTorch 2.13's Tensor.backward stub is untyped even though the runtime
        # method is the standard autograd entry point; keep the exception local.
        loss.backward()  # type: ignore[no-untyped-call]
        raw_norm = clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.max_grad_norm
            if self.config.max_grad_norm > 0
            else float("inf"),
        )
        self.optimizer.step()

        active = torch.where(
            advantages >= 0,
            ratio <= 1.0 + self.config.clip_ratio,
            ratio >= 1.0 - self.config.clip_ratio,
        )
        norm_value = float(raw_norm.detach().cpu())
        if self.config.max_grad_norm > 0:
            norm_value = min(norm_value, self.config.max_grad_norm)
        return (
            float(policy_loss.detach().cpu()),
            float(value_loss.detach().cpu()),
            [float(value) for value in ratio.detach().cpu().tolist()],
            [float(value) for value in (~active).float().detach().cpu().tolist()],
            norm_value,
        )


def _trace_log_prob(
    output: TorchPolicyTensorOutput,
    batch_index: int,
    trace: PolicySampleTrace,
) -> Tensor:
    total: Tensor | None = None
    for choice in trace.choices:
        if not choice.valid_indices:
            raise ValueError(f"{choice.head} trace has no legal indices")
        if choice.index not in choice.valid_indices:
            raise ValueError(f"{choice.head} selected index is not legal")
        logits = output.head(choice.head)[batch_index]
        valid_indices = torch.tensor(
            choice.valid_indices,
            dtype=torch.long,
            device=logits.device,
        )
        valid_logits = logits.index_select(0, valid_indices)
        selected_offset = choice.valid_indices.index(choice.index)
        log_prob = torch.log_softmax(valid_logits, dim=0)[selected_offset]
        total = log_prob if total is None else total + log_prob
    if total is None:
        raise ValueError("policy trace has no sampled heads")
    return total


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
