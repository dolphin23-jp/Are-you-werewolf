from __future__ import annotations

import pytest

from app.engine.game import PlayerSpec
from app.engine.roles import RoleName
from app.training.encoding import ObservationEncoder
from app.training.env import WerewolfTrainingEnv
from app.training.policy_sampling import HeadChoice, PolicySampleTrace
from app.training.trajectory import DecisionKind, EpisodeTrajectory, RecordedDecision

torch_trainer = pytest.importorskip("app.training.torch_trainer")
TorchPPOConfig = torch_trainer.TorchPPOConfig


def _observation():
    specs = [PlayerSpec(player_id=f"p{i}", name=f"Player{i}") for i in range(17)]
    env = WerewolfTrainingEnv(
        specs,
        seed=1501,
        forced_roles={"p0": RoleName.VILLAGER},
    )
    return ObservationEncoder().encode(env.observe("p0"))


def _decision(value_estimate: float, target_id: str) -> RecordedDecision:
    trace = PolicySampleTrace(
        choices=(
            HeadChoice(
                head="vote_target",
                index=1,
                valid_indices=(1, 2),
                log_prob=-0.5,
            ),
        ),
        value_estimate=value_estimate,
    )
    return RecordedDecision(
        player_id="p0",
        kind=DecisionKind.VOTE,
        observation=_observation(),
        target_id=target_id,
        policy_trace=trace,
    )


def _trajectory() -> EpisodeTrajectory:
    trajectory = EpisodeTrajectory("gae")
    trajectory.append(_decision(0.2, "p1"))
    trajectory.append(_decision(0.4, "p2"))
    trajectory.finalize({"p0": 1.0})
    return trajectory


def test_default_gae_preserves_previous_terminal_monte_carlo_targets():
    samples = torch_trainer._prepare_samples([_trajectory()], TorchPPOConfig())

    assert [sample.value_target for sample in samples] == pytest.approx([1.0, 1.0])
    assert [sample.advantage for sample in samples] == pytest.approx([0.8, 0.6])


def test_gae_lambda_can_bootstrap_without_adding_intermediate_reward():
    samples = torch_trainer._prepare_samples(
        [_trajectory()],
        TorchPPOConfig(gamma=1.0, gae_lambda=0.0),
    )

    assert [sample.value_target for sample in samples] == pytest.approx([0.4, 1.0])
    assert [sample.advantage for sample in samples] == pytest.approx([0.2, 0.6])


def test_advantage_normalization_is_optional_and_batch_wide():
    samples = torch_trainer._prepare_samples(
        [_trajectory()],
        TorchPPOConfig(normalize_advantages=True),
    )

    assert sum(sample.advantage for sample in samples) == pytest.approx(0.0, abs=1e-7)
