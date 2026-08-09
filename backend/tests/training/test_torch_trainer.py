import pytest

from app.engine.game import PlayerSpec
from app.engine.roles import RoleName
from app.training.actions import ActionType
from app.training.encoding import ObservationEncoder
from app.training.env import WerewolfTrainingEnv
from app.training.legal import LegalActionMask
from app.training.policy_sampling import MaskedPolicySampler
from app.training.trajectory import DecisionKind, EpisodeTrajectory, RecordedDecision

torch = pytest.importorskip("torch")
torch_policy = pytest.importorskip("app.training.torch_policy")
torch_trainer = pytest.importorskip("app.training.torch_trainer")
TorchTransformerPolicy = torch_policy.TorchTransformerPolicy
TransformerPolicyConfig = torch_policy.TransformerPolicyConfig
TorchPPOConfig = torch_trainer.TorchPPOConfig
TorchPPOTrainer = torch_trainer.TorchPPOTrainer


def _env() -> WerewolfTrainingEnv:
    specs = [PlayerSpec(player_id=f"p{i}", name=f"Player{i}") for i in range(17)]
    return WerewolfTrainingEnv(
        specs,
        seed=501,
        forced_roles={"p0": RoleName.VILLAGER},
    )


def _model():
    return TorchTransformerPolicy(
        TransformerPolicyConfig(
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.0,
        )
    )


def _parameters(model):
    return torch.cat(
        [parameter.detach().cpu().flatten() for parameter in model.parameters()]
    )


def _vote_trajectory(model, *, reward: float, episode_id: str) -> EpisodeTrajectory:
    env = _env()
    public_observation = env.observe("p0")
    encoded = ObservationEncoder().encode(public_observation)
    logits = model.forward(encoded)
    sampled = MaskedPolicySampler(seed=503).sample_vote(
        public_observation,
        LegalActionMask(
            action_types=(ActionType.VOTE,),
            vote_target_ids=("p1", "p2", "p3"),
        ),
        logits,
    )
    trajectory = EpisodeTrajectory(episode_id)
    trajectory.append(
        RecordedDecision(
            player_id="p0",
            kind=DecisionKind.VOTE,
            observation=encoded,
            target_id=sampled.target_id,
            policy_trace=sampled.trace,
        )
    )
    trajectory.finalize({"p0": reward})
    return trajectory


def test_torch_ppo_changes_transformer_parameters_from_terminal_reward():
    torch.manual_seed(505)
    model = _model()
    trajectory = _vote_trajectory(model, reward=1.0, episode_id="torch-ppo")
    before = _parameters(model).clone()

    stats = TorchPPOTrainer(
        model,
        TorchPPOConfig(
            learning_rate=1e-3,
            epochs=2,
            minibatch_size=1,
        ),
        seed=507,
    ).update([trajectory])
    after = _parameters(model)

    assert stats.decisions == 1
    assert stats.epochs == 2
    assert torch.isfinite(torch.tensor(stats.mean_policy_loss))
    assert torch.isfinite(torch.tensor(stats.mean_value_loss))
    assert stats.gradient_norm > 0.0
    assert not torch.allclose(before, after)


def test_torch_ppo_batches_multiple_traced_decisions():
    torch.manual_seed(509)
    model = _model()
    positive = _vote_trajectory(model, reward=1.0, episode_id="positive")
    negative = _vote_trajectory(model, reward=-1.0, episode_id="negative")

    stats = TorchPPOTrainer(
        model,
        TorchPPOConfig(
            learning_rate=1e-3,
            epochs=1,
            minibatch_size=2,
        ),
        seed=511,
    ).update([positive, negative])

    assert stats.decisions == 2
    assert stats.epochs == 1
    assert stats.mean_ratio > 0.0
    assert 0.0 <= stats.clip_fraction <= 1.0
    assert stats.gradient_norm > 0.0


def test_torch_ppo_rejects_unfinalized_trajectory():
    torch.manual_seed(513)
    model = _model()
    trajectory = _vote_trajectory(model, reward=1.0, episode_id="unfinished")
    trajectory.finalized = False

    with pytest.raises(ValueError, match="requires finalized"):
        TorchPPOTrainer(model).update([trajectory])
