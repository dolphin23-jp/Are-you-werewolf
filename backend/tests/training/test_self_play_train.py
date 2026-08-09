import numpy as np

from app.engine.game import PlayerSpec
from app.training.numpy_trainer import PPOConfig
from app.training.self_play_train import NumpySelfPlayTrainingLoop


def _specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=False)
        for i in range(17)
    ]


def test_numpy_self_play_batch_runs_episode_and_updates_model():
    loop = NumpySelfPlayTrainingLoop(
        _specs(),
        hidden_size=8,
        model_seed=31,
        max_discussion_ticks=2,
        ppo_config=PPOConfig(learning_rate=1e-3, epochs=1),
    )

    stats = loop.train_batch(start_seed=83, episodes=1)

    assert stats.episodes == 1
    assert stats.village_wins + stats.werewolf_wins + stats.fox_wins + stats.draws == 1
    assert stats.update.decisions > 0
    assert np.isfinite(stats.update.mean_policy_loss)
    assert np.isfinite(stats.update.mean_value_loss)
    assert loop.model.initialized is True
