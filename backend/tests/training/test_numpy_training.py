import numpy as np

from app.engine.game import PlayerSpec
from app.engine.roles import RoleName
from app.training.encoding import ObservationEncoder
from app.training.env import WerewolfTrainingEnv
from app.training.legal import LegalActionMask
from app.training.numpy_policy import NumpyMLPPolicy, flatten_observation
from app.training.numpy_trainer import NumpyPPOTrainer, PPOConfig
from app.training.policy_sampling import MaskedPolicySampler
from app.training.trajectory import DecisionKind, EpisodeTrajectory, RecordedDecision
from app.training.actions import ActionType


def _env() -> WerewolfTrainingEnv:
    specs = [PlayerSpec(player_id=f"p{i}", name=f"Player{i}") for i in range(17)]
    return WerewolfTrainingEnv(
        specs,
        seed=47,
        forced_roles={"p0": RoleName.VILLAGER},
    )


def test_numpy_policy_has_deterministic_valid_output_shape():
    observation = ObservationEncoder().encode(_env().observe("p0"))
    first = NumpyMLPPolicy(seed=7, hidden_size=16)
    second = NumpyMLPPolicy(seed=7, hidden_size=16)

    first_logits = first.forward(observation)
    second_logits = second.forward(observation)

    first_logits.validate()
    second_logits.validate()
    assert np.allclose(first.parameter_vector(), second.parameter_vector())
    assert np.isfinite(flatten_observation(observation)).all()


def test_ppo_update_changes_parameters_from_sparse_terminal_reward():
    env = _env()
    public_observation = env.observe("p0")
    encoded = ObservationEncoder().encode(public_observation)
    model = NumpyMLPPolicy(seed=11, hidden_size=16)
    logits = model.forward(encoded)
    sampler = MaskedPolicySampler(seed=13)
    mask = LegalActionMask(
        action_types=(ActionType.VOTE,),
        vote_target_ids=("p1", "p2"),
    )
    sampled = sampler.sample_vote(public_observation, mask, logits)
    trajectory = EpisodeTrajectory("synthetic")
    trajectory.append(
        RecordedDecision(
            player_id="p0",
            kind=DecisionKind.VOTE,
            observation=encoded,
            target_id=sampled.target_id,
            policy_trace=sampled.trace,
        )
    )
    trajectory.finalize({"p0": 1.0})
    before = model.parameter_vector()

    stats = NumpyPPOTrainer(
        model,
        PPOConfig(learning_rate=1e-2, epochs=2),
    ).update([trajectory])
    after = model.parameter_vector()

    assert stats.decisions == 1
    assert np.isfinite(stats.mean_policy_loss)
    assert np.isfinite(stats.mean_value_loss)
    assert stats.gradient_norm > 0
    assert not np.allclose(before, after)
