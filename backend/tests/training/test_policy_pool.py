from pathlib import Path

import numpy as np

from app.engine.game import PlayerSpec
from app.training.encoding import ObservationEncoder
from app.training.env import WerewolfTrainingEnv
from app.training.numpy_policy import NumpyMLPPolicy
from app.training.policy_pool import NumpyPolicyPool


def _initialized_model(seed: int) -> NumpyMLPPolicy:
    specs = [PlayerSpec(player_id=f"p{i}", name=f"Player{i}") for i in range(17)]
    env = WerewolfTrainingEnv(specs, seed=seed)
    observation = ObservationEncoder().encode(env.observe("p0"))
    model = NumpyMLPPolicy(seed=seed, hidden_size=8)
    model.forward(observation)
    return model


def test_policy_pool_persists_generations_and_parent_links(tmp_path: Path):
    pool = NumpyPolicyPool(tmp_path / "pool")
    first_model = _initialized_model(101)
    first = pool.add(first_model)
    second_model = _initialized_model(102)
    second = pool.add(second_model, parent_id=first.policy_id)

    reloaded = NumpyPolicyPool(tmp_path / "pool")
    restored = reloaded.load(second.policy_id)

    assert first.policy_id == "g000000"
    assert second.policy_id == "g000001"
    assert second.parent_id == first.policy_id
    assert reloaded.next_generation == 2
    assert reloaded.latest() == second
    assert np.array_equal(restored.parameter_vector(), second_model.parameter_vector())
