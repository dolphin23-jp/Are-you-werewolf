from pathlib import Path

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.encoding import ObservationEncoder
from app.training.env import WerewolfTrainingEnv
from app.training.historical_train import HistoricalNumpyTrainingLoop
from app.training.numpy_policy import NumpyMLPPolicy
from app.training.numpy_trainer import PPOConfig
from app.training.policy_pool import NumpyPolicyPool


def _specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=False)
        for i in range(17)
    ]


def _initialized_model(seed: int) -> NumpyMLPPolicy:
    env = WerewolfTrainingEnv(_specs(), seed=seed)
    observation = ObservationEncoder().encode(env.observe("p0"))
    model = NumpyMLPPolicy(seed=seed, hidden_size=8)
    model.forward(observation)
    return model


def test_historical_self_play_trains_only_selected_faction(tmp_path: Path):
    pool = NumpyPolicyPool(tmp_path / "pool")
    pool.add(_initialized_model(201))
    learner = NumpyMLPPolicy(seed=202, hidden_size=8)
    loop = HistoricalNumpyTrainingLoop(
        _specs(),
        learner,
        pool,
        opponent_seed=203,
        max_discussion_ticks=2,
        ppo_config=PPOConfig(learning_rate=1e-3, epochs=1),
    )

    stats = loop.train_batch(
        learner_team=Team.FOX,
        start_seed=204,
        episodes=1,
    )

    assert stats.episodes == 1
    assert stats.wins + stats.losses + stats.draws == 1
    assert stats.update.decisions > 0
    assert stats.mean_decisions == stats.update.decisions
    assert len(stats.opponent_policy_ids) == 2
    assert learner.initialized is True
