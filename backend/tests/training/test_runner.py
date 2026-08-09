from app.engine.game import PlayerSpec
from app.training.runner import RandomEpisodeRunner


def _specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=(i == 0))
        for i in range(17)
    ]


def test_random_training_episodes_terminate_with_sparse_terminal_rewards():
    runner = RandomEpisodeRunner(_specs(), max_discussion_ticks=6)

    for seed in range(3):
        result = runner.run(seed)
        assert result.is_draw or result.winner is not None
        assert set(result.rewards.values()) <= {-1.0, 0.0, 1.0}
        assert result.trajectory.finalized is True
        assert result.trajectory.terminal_rewards == result.rewards
        assert result.trajectory.decisions
        assert all(
            decision.reward == result.rewards[decision.player_id]
            for decision in result.trajectory.decisions
        )
        if not result.is_draw:
            assert 1.0 in result.rewards.values()
            assert -1.0 in result.rewards.values()
