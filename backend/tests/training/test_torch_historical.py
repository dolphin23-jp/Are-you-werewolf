from pathlib import Path

import pytest

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.meta_strategy import PolicyWeight, PopulationMetaStrategy

torch = pytest.importorskip("torch")
torch_historical = pytest.importorskip("app.training.torch_historical")
torch_policy = pytest.importorskip("app.training.torch_policy")
torch_pool = pytest.importorskip("app.training.torch_pool")
torch_trainer = pytest.importorskip("app.training.torch_trainer")
TorchHistoricalTrainingLoop = torch_historical.TorchHistoricalTrainingLoop
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


def test_torch_historical_updates_only_learner_faction(tmp_path: Path):
    pool = TorchPolicyPool(tmp_path / "pool")
    general = pool.add(_model(1001))
    learner = _model(1002)
    loop = TorchHistoricalTrainingLoop(
        _specs(),
        learner,
        pool,
        opponent_seed=1003,
        max_discussion_ticks=0,
        trainer_seed=1004,
        ppo_config=TorchPPOConfig(
            learning_rate=1e-3,
            epochs=1,
            minibatch_size=64,
        ),
    )

    stats = loop.train_batch(
        learner_team=Team.VILLAGE,
        start_seed=1005,
        episodes=1,
    )

    assert stats.episodes == 1
    assert stats.wins + stats.losses + stats.draws == 1
    assert stats.update.decisions > 0
    assert stats.mean_decisions == stats.update.decisions
    assert set(stats.opponent_policy_ids) == {
        f"werewolf:{general.policy_id}",
        f"fox:{general.policy_id}",
    }


def test_torch_historical_uses_meta_strategy_and_team_eligibility(tmp_path: Path):
    pool = TorchPolicyPool(tmp_path / "meta-pool")
    general = pool.add(_model(1011))
    wolf = pool.add(_model(1012), specialized_team=Team.WEREWOLF)
    fox = pool.add(_model(1013), specialized_team=Team.FOX)
    strategy = PopulationMetaStrategy(
        village=(PolicyWeight(general.policy_id, 1.0),),
        werewolf=(PolicyWeight(wolf.policy_id, 1.0),),
        fox=(PolicyWeight(fox.policy_id, 1.0),),
    )
    loop = TorchHistoricalTrainingLoop(
        _specs(),
        _model(1014),
        pool,
        opponent_strategy=strategy,
        max_discussion_ticks=0,
        ppo_config=TorchPPOConfig(epochs=1, minibatch_size=64),
    )

    stats = loop.train_batch(
        learner_team=Team.VILLAGE,
        start_seed=1015,
        episodes=1,
    )

    assert set(stats.opponent_policy_ids) == {
        f"werewolf:{wolf.policy_id}",
        f"fox:{fox.policy_id}",
    }
