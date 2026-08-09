"""Batched clipped-PPO optimizer for the PyTorch Transformer policy.

The trainer reuses the framework-agnostic rollout traces. Only indices that were
legal at decision time participate in each masked log-probability. Rewards are
still sparse terminal faction rewards; no strategic shaping is added here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from app.training.numpy_trainer import PPOUpdateStats
from app.training.policy_sampling import HeadChoice, PolicySampleTrace
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
    gamma: float = 1.0
    gae_lambda: float = 1.0
    normalize_advantages: bool = False

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
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")


@dataclass(frozen=True)
class _TorchPPOSample:
    decision: RecordedDecision
    advantage: float
    value_target: float


class TorchPPOTrainer:
    """Update a Transformer policy/value network from finalized trajectories."""

    def __init__(
        self,
        model: TorchTransformerPolicy,
        config: TorchPPOConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        if model.config.dropout != 0.0:
            raise ValueError(
                "Torch PPO currently requires dropout=0 so rollout/update log-probs match"
            )
        self.model = model
        self.config = config or TorchPPOConfig()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)

    def checkpoint_state(self) -> tuple[dict[str, Any], Tensor]:
        """Return in-memory optimizer and minibatch RNG state for safe encoding."""

        return self.optimizer.state_dict(), self._generator.get_state().clone()

    def restore_checkpoint_state(
        self,
        optimizer_state: dict[str, Any],
        generator_state: Tensor,
    ) -> None:
        """Restore optimizer moments and the deterministic minibatch RNG stream."""

        if generator_state.dtype is not torch.uint8 or generator_state.ndim != 1:
            raise ValueError("trainer generator state must be a 1D uint8 tensor")
        self.optimizer.load_state_dict(optimizer_state)
        self._generator.set_state(generator_state.detach().cpu())

    def update(self, trajectories: list[EpisodeTrajectory]) -> PPOUpdateStats:
        if any(not trajectory.finalized for trajectory in trajectories):
            raise ValueError("PPO update requires finalized trajectories")
        samples = _prepare_samples(trajectories, self.config)
        if not samples:
            return PPOUpdateStats(0, 0, 0.0, 0.0, 1.0, 0.0, 0.0)

        policy_losses: list[float] = []
        value_losses: list[float] = []
        ratios: list[float] = []
        clipped: list[float] = []
        gradient_norm = 0.0
        self.model.train()

        for _ in range(self.config.epochs):
            permutation = torch.randperm(len(samples), generator=self._generator).tolist()
            for start in range(0, len(permutation), self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                batch = [samples[index] for index in indices]
                batch_stats = self._update_minibatch(batch)
                policy_losses.append(batch_stats[0])
                value_losses.append(batch_stats[1])
                ratios.extend(batch_stats[2])
                clipped.extend(batch_stats[3])
                gradient_norm = batch_stats[4]

        return PPOUpdateStats(
            decisions=len(samples),
            epochs=self.config.epochs,
            mean_policy_loss=_mean(policy_losses),
            mean_value_loss=_mean(value_losses),
            mean_ratio=_mean(ratios),
            clip_fraction=_mean(clipped),
            gradient_norm=gradient_norm,
        )

    def _update_minibatch(
        self,
        samples: list[_TorchPPOSample],
    ) -> tuple[float, float, list[float], list[float], float]:
        decisions = [sample.decision for sample in samples]
        output = self.model.forward_batch(
            tuple(decision.observation for decision in decisions)
        )
        traces = [decision.policy_trace for decision in decisions]
        if any(trace is None for trace in traces):
            raise RuntimeError("Torch PPO minibatch contains a decision without a trace")
        typed_traces = [trace for trace in traces if trace is not None]

        current_log_probs = _trace_log_probs(output, typed_traces)
        device = output.value.device
        old_log_probs = torch.tensor(
            [trace.log_prob for trace in typed_traces],
            dtype=output.value.dtype,
            device=device,
        )
        value_targets = torch.tensor(
            [sample.value_target for sample in samples],
            dtype=output.value.dtype,
            device=device,
        )
        advantages = torch.tensor(
            [sample.advantage for sample in samples],
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
        value_error = output.value - value_targets
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


def _prepare_samples(
    trajectories: list[EpisodeTrajectory],
    config: TorchPPOConfig,
) -> list[_TorchPPOSample]:
    samples: list[_TorchPPOSample] = []
    for trajectory in trajectories:
        traced_decisions = [
            decision
            for decision in trajectory.decisions
            if decision.policy_trace is not None
        ]
        by_player: dict[str, list[RecordedDecision]] = {}
        for decision in traced_decisions:
            by_player.setdefault(decision.player_id, []).append(decision)

        targets_by_decision: dict[int, tuple[float, float]] = {}
        for player_id, decisions in by_player.items():
            targets = _player_gae_targets(
                decisions,
                terminal_reward=trajectory.terminal_rewards.get(player_id, 0.0),
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )
            for decision, target in zip(decisions, targets, strict=True):
                targets_by_decision[id(decision)] = target

        for decision in traced_decisions:
            advantage, value_target = targets_by_decision[id(decision)]
            samples.append(
                _TorchPPOSample(
                    decision=decision,
                    advantage=advantage,
                    value_target=value_target,
                )
            )

    if config.normalize_advantages and len(samples) > 1:
        mean = sum(sample.advantage for sample in samples) / len(samples)
        variance = sum(
            (sample.advantage - mean) ** 2 for sample in samples
        ) / len(samples)
        scale = math.sqrt(variance + 1e-8)
        samples = [
            replace(sample, advantage=(sample.advantage - mean) / scale)
            for sample in samples
        ]
    return samples


def _player_gae_targets(
    decisions: list[RecordedDecision],
    *,
    terminal_reward: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[tuple[float, float], ...]:
    """Compute per-seat GAE without introducing intermediate shaped rewards.

    A seat receives zero reward between its decisions and its faction terminal
    payoff after its final recorded decision. With ``gamma=1`` and
    ``gae_lambda=1`` this telescopes exactly to the previous Monte-Carlo target:
    every value target is the terminal faction payoff and every advantage is
    ``terminal_reward - rollout_value``.
    """

    if not decisions:
        return ()
    reversed_targets: list[tuple[float, float]] = []
    next_value = 0.0
    gae = 0.0
    last_index = len(decisions) - 1
    for index in range(last_index, -1, -1):
        decision = decisions[index]
        trace = decision.policy_trace
        if trace is None:
            raise ValueError("GAE requires traced decisions")
        reward = terminal_reward if index == last_index else 0.0
        value = trace.value_estimate
        delta = reward + gamma * next_value - value
        gae = delta + gamma * gae_lambda * gae
        reversed_targets.append((gae, gae + value))
        next_value = value
    return tuple(reversed(reversed_targets))


def _trace_log_probs(
    output: TorchPolicyTensorOutput,
    traces: list[PolicySampleTrace],
) -> Tensor:
    """Compute factorized masked log-probs in head-sized vectorized groups.

    Rollout traces can contain different heads because timing-only, speech, vote,
    and night decisions factorize differently. Grouping every sampled choice by
    head replaces one tiny device operation per choice with at most one masked
    log-softmax per policy head in the minibatch.
    """

    if not traces:
        return output.value.new_empty((0,))

    grouped: dict[str, list[tuple[int, HeadChoice]]] = {}
    for batch_index, trace in enumerate(traces):
        if not trace.choices:
            raise ValueError("policy trace has no sampled heads")
        for choice in trace.choices:
            if not choice.valid_indices:
                raise ValueError(f"{choice.head} trace has no legal indices")
            if choice.index not in choice.valid_indices:
                raise ValueError(f"{choice.head} selected index is not legal")
            grouped.setdefault(choice.head, []).append((batch_index, choice))

    total = output.value.new_zeros((len(traces),))
    for head, entries in grouped.items():
        head_logits = output.head(head)
        if head_logits.ndim != 2 or head_logits.shape[0] != len(traces):
            raise ValueError(f"{head} output shape does not match PPO minibatch")
        width = head_logits.shape[1]

        legal_rows: list[list[bool]] = []
        batch_indices: list[int] = []
        selected_indices: list[int] = []
        for batch_index, choice in entries:
            if any(index < 0 or index >= width for index in choice.valid_indices):
                raise ValueError(f"{head} legal index exceeds head width")
            legal = set(choice.valid_indices)
            legal_rows.append([index in legal for index in range(width)])
            batch_indices.append(batch_index)
            selected_indices.append(choice.index)

        device = head_logits.device
        rows = head_logits.index_select(
            0,
            torch.tensor(batch_indices, dtype=torch.long, device=device),
        )
        legal_mask = torch.tensor(legal_rows, dtype=torch.bool, device=device)
        selected = torch.tensor(selected_indices, dtype=torch.long, device=device)
        masked_log_probs = torch.log_softmax(
            rows.masked_fill(~legal_mask, float("-inf")),
            dim=1,
        )
        choice_log_probs = masked_log_probs.gather(1, selected.unsqueeze(1)).squeeze(1)
        total = total.index_add(
            0,
            torch.tensor(batch_indices, dtype=torch.long, device=device),
            choice_log_probs,
        )
    return total


def _trace_log_prob(
    output: TorchPolicyTensorOutput,
    batch_index: int,
    trace: PolicySampleTrace,
) -> Tensor:
    """Scalar reference implementation retained for equivalence tests."""

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
