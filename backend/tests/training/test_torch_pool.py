from pathlib import Path

import pytest

from app.engine.roles import Team

torch = pytest.importorskip("torch")
torch_policy = pytest.importorskip("app.training.torch_policy")
torch_pool = pytest.importorskip("app.training.torch_pool")
TorchTransformerPolicy = torch_policy.TorchTransformerPolicy
TransformerPolicyConfig = torch_policy.TransformerPolicyConfig
TorchPolicyPool = torch_pool.TorchPolicyPool


def _model(seed: int):
    torch.manual_seed(seed)
    return TorchTransformerPolicy(
        TransformerPolicyConfig(
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
    ).eval()


def test_torch_pool_persists_faction_specialists(tmp_path: Path):
    pool = TorchPolicyPool(tmp_path / "pool")
    general = pool.add(_model(901))
    village = pool.add(
        _model(902),
        parent_id=general.policy_id,
        specialized_team=Team.VILLAGE,
    )
    wolf = pool.add(
        _model(903),
        parent_id=village.policy_id,
        specialized_team=Team.WEREWOLF,
    )

    restored = TorchPolicyPool(tmp_path / "pool")
    loaded = restored.load(village.policy_id)

    assert restored.next_generation == 3
    assert restored.latest() == wolf
    assert restored.policy_ids_for_team(Team.VILLAGE) == (
        general.policy_id,
        village.policy_id,
    )
    assert restored.policy_ids_for_team(Team.WEREWOLF) == (
        general.policy_id,
        wolf.policy_id,
    )
    assert restored.policy_ids_for_team(
        Team.VILLAGE,
        include_general=False,
    ) == (village.policy_id,)
    for original, reloaded in zip(
        pool.load(village.policy_id).parameters(),
        loaded.parameters(),
        strict=True,
    ):
        assert torch.equal(original, reloaded)
