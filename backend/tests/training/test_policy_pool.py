import json
from pathlib import Path

import numpy as np

from app.engine.game import PlayerSpec
from app.engine.roles import Team
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
    second = pool.add(
        second_model,
        parent_id=first.policy_id,
        specialized_team=Team.VILLAGE,
    )

    reloaded = NumpyPolicyPool(tmp_path / "pool")
    restored = reloaded.load(second.policy_id)

    assert first.policy_id == "g000000"
    assert second.policy_id == "g000001"
    assert second.parent_id == first.policy_id
    assert second.specialized_team is Team.VILLAGE
    assert reloaded.next_generation == 2
    assert reloaded.latest() == second
    assert np.array_equal(restored.parameter_vector(), second_model.parameter_vector())


def test_policy_pool_separates_team_specialists_but_keeps_generalists(tmp_path: Path):
    pool = NumpyPolicyPool(tmp_path / "pool")
    general = pool.add(_initialized_model(111))
    village = pool.add(_initialized_model(112), specialized_team=Team.VILLAGE)
    wolf = pool.add(_initialized_model(113), specialized_team=Team.WEREWOLF)
    fox = pool.add(_initialized_model(114), specialized_team=Team.FOX)

    assert pool.policy_ids_for_team(Team.VILLAGE) == (
        general.policy_id,
        village.policy_id,
    )
    assert pool.policy_ids_for_team(Team.WEREWOLF) == (
        general.policy_id,
        wolf.policy_id,
    )
    assert pool.policy_ids_for_team(Team.FOX) == (
        general.policy_id,
        fox.policy_id,
    )
    assert pool.policy_ids_for_team(
        Team.VILLAGE,
        include_general=False,
    ) == (village.policy_id,)
    assert pool.policy_ids_for_team(Team.FOX, last=1) == (fox.policy_id,)


def test_legacy_manifest_without_specialized_team_is_generalist(tmp_path: Path):
    root = tmp_path / "legacy-pool"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "entries": [
                    {
                        "policy_id": "g000000",
                        "generation": 0,
                        "checkpoint": "g000000.npz",
                        "parent_id": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    pool = NumpyPolicyPool(root)

    assert pool.entries[0].specialized_team is None
    for team in Team:
        assert pool.policy_ids_for_team(team) == ("g000000",)
