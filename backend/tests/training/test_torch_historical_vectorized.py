from pathlib import Path

import pytest

from app.engine.game import PlayerSpec
from app.engine.roles import Team

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


class BatchCountingTransformer(TorchTransformerPolicy):
    def __init__(self) -> None:
        super().__init__(
            TransformerPolicyConfig(
                d_model=16,
                nhead=4,
                num_layers=1,
                dim_feedforward=32,
                dropout=0.0,
            )
        )
        self.batch_sizes: list[int] = []

    def forward_batch(self, observations):
        self.batch_sizes.append(len(observations))
        return super().forward_batch(observations)


def _specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=False)
        for i in range(17)
    ]


def _model(seed: int) -> TorchTransformerPolicy:
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


def test_historical_batch_vectorizes_games_and_loads_unique_opponent_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pool = TorchPolicyPool(tmp_path / "pool")
    general = pool.add(_model(1401))
    torch.manual_seed(1403)
    learner = BatchCountingTransformer().eval()

    original_load = pool.load
    load_calls: list[str] = []

    def counting_load(policy_id: str):
        load_calls.append(policy_id)
        return original_load(policy_id)

    monkeypatch.setattr(pool, "load", counting_load)
    loop = TorchHistoricalTrainingLoop(
        _specs(),
        learner,
        pool,
        opponent_seed=1405,
        max_discussion_ticks=0,
        max_parallel_games=3,
        trainer_seed=1407,
        ppo_config=TorchPPOConfig(
            learning_rate=1e-3,
            epochs=1,
            minibatch_size=64,
        ),
    )

    stats = loop.train_batch(
        learner_team=Team.VILLAGE,
        start_seed=1409,
        episodes=3,
    )

    assert stats.episodes == 3
    assert stats.wins + stats.losses + stats.draws == 3
    assert stats.update.decisions > 0
    assert stats.mean_decisions * stats.episodes == pytest.approx(
        stats.update.decisions
    )
    assert set(stats.opponent_policy_ids) == {
        f"werewolf:{general.policy_id}",
        f"fox:{general.policy_id}",
    }
    assert load_calls == [general.policy_id]
    assert stats.opponent_checkpoint_loads == 1
    assert learner.batch_sizes
    assert stats.inference_calls > 0
    assert stats.inference_observations > 0
    assert stats.max_pending_inference_requests > 17
    assert stats.max_inference_batch > 17
    assert stats.rollout_seconds > 0.0
    assert stats.learner_seconds > 0.0


def test_historical_opponent_cache_is_bounded_to_parallel_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pool = TorchPolicyPool(tmp_path / "bounded-pool")
    general = pool.add(_model(1411))
    original_load = pool.load
    load_calls: list[str] = []

    def counting_load(policy_id: str):
        load_calls.append(policy_id)
        return original_load(policy_id)

    monkeypatch.setattr(pool, "load", counting_load)
    loop = TorchHistoricalTrainingLoop(
        _specs(),
        _model(1413),
        pool,
        opponent_seed=1415,
        max_discussion_ticks=0,
        max_parallel_games=1,
        trainer_seed=1417,
        ppo_config=TorchPPOConfig(
            learning_rate=1e-3,
            epochs=1,
            minibatch_size=64,
        ),
    )

    stats = loop.train_batch(
        learner_team=Team.VILLAGE,
        start_seed=1419,
        episodes=2,
    )

    assert load_calls == [general.policy_id, general.policy_id]
    assert stats.opponent_checkpoint_loads == 2
    assert stats.episodes == 2


def test_historical_vectorized_limits_validate_before_training(tmp_path: Path):
    pool = TorchPolicyPool(tmp_path / "limits-pool")
    pool.add(_model(1421))

    with pytest.raises(ValueError, match="max_parallel_games must be positive"):
        TorchHistoricalTrainingLoop(
            _specs(),
            _model(1423),
            pool,
            max_parallel_games=0,
        )

    with pytest.raises(ValueError, match="max_inference_batch_size must be positive"):
        TorchHistoricalTrainingLoop(
            _specs(),
            _model(1425),
            pool,
            max_inference_batch_size=0,
        )
