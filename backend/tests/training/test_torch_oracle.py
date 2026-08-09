from pathlib import Path

import pytest

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.meta_strategy import PolicyWeight, PopulationMetaStrategy

torch = pytest.importorskip("torch")
torch_oracle = pytest.importorskip("app.training.torch_oracle")
torch_policy = pytest.importorskip("app.training.torch_policy")
torch_pool = pytest.importorskip("app.training.torch_pool")
torch_trainer = pytest.importorskip("app.training.torch_trainer")
train_torch_population_oracle = torch_oracle.train_torch_population_oracle
TorchTransformerPolicy = torch_policy.TorchTransformerPolicy
TransformerPolicyConfig = torch_policy.TransformerPolicyConfig
TorchPolicyPool = torch_pool.TorchPolicyPool
TorchPPOConfig = torch_trainer.TorchPPOConfig


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


def test_torch_oracle_adds_faction_specialized_generation(tmp_path: Path):
    pool = TorchPolicyPool(tmp_path / "pool")
    base = pool.add(_model(1201))
    alternative = pool.add(_model(1202))
    strategy = PopulationMetaStrategy(
        village=(
            PolicyWeight(base.policy_id, 0.8),
            PolicyWeight(alternative.policy_id, 0.2),
        ),
        werewolf=(PolicyWeight(base.policy_id, 1.0),),
        fox=(PolicyWeight(base.policy_id, 1.0),),
    )

    stats = train_torch_population_oracle(
        _specs(),
        pool,
        strategy,
        team=Team.VILLAGE,
        episodes=1,
        start_seed=1203,
        opponent_seed=1204,
        trainer_seed=1205,
        max_discussion_ticks=0,
        ppo_config=TorchPPOConfig(
            learning_rate=1e-3,
            epochs=1,
            minibatch_size=64,
        ),
    )

    assert stats.parent_policy_id == base.policy_id
    assert stats.oracle_entry.parent_id == base.policy_id
    assert stats.oracle_entry.specialized_team is Team.VILLAGE
    assert stats.update.decisions > 0
    assert stats.wins + stats.losses + stats.draws == 1
    assert stats.oracle_entry.policy_id in pool.policy_ids_for_team(
        Team.VILLAGE,
        include_general=False,
    )
