from pathlib import Path

import pytest

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.population_payoff import PolicyProfile, PopulationPayoffTable

torch = pytest.importorskip("torch")
torch_policy = pytest.importorskip("app.training.torch_policy")
torch_pool = pytest.importorskip("app.training.torch_pool")
torch_population = pytest.importorskip("app.training.torch_population")
TorchTransformerPolicy = torch_policy.TorchTransformerPolicy
TransformerPolicyConfig = torch_policy.TransformerPolicyConfig
TorchPolicyPool = torch_pool.TorchPolicyPool
evaluate_torch_policy_profile = torch_population.evaluate_torch_policy_profile


def _specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=False)
        for i in range(17)
    ]


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


def test_torch_population_profile_records_terminal_outcome(tmp_path: Path):
    pool = TorchPolicyPool(tmp_path / "pool")
    village = pool.add(_model(1101), specialized_team=Team.VILLAGE)
    wolf = pool.add(_model(1102), specialized_team=Team.WEREWOLF)
    fox = pool.add(_model(1103), specialized_team=Team.FOX)
    profile = PolicyProfile(village.policy_id, wolf.policy_id, fox.policy_id)
    table = PopulationPayoffTable(tmp_path / "payoffs.json")

    record = evaluate_torch_policy_profile(
        _specs(),
        pool,
        table,
        profile,
        seeds=(1104,),
        max_discussion_ticks=0,
    )

    assert record.games == 1
    assert record.village_wins + record.werewolf_wins + record.fox_wins + record.draws == 1
    assert table.has_complete_cube(
        (village.policy_id,),
        (wolf.policy_id,),
        (fox.policy_id,),
    )
