from app.engine.game import PlayerSpec
from app.training.learned_runner import LearnedEpisodeRunner
from app.training.trajectory import DecisionKind
from app.training.uniform_model import UniformPolicyModel


def _specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=(i == 0))
        for i in range(17)
    ]


def test_uniform_model_episode_terminates_and_records_policy_traces():
    runner = LearnedEpisodeRunner(
        _specs(),
        UniformPolicyModel(),
        max_discussion_ticks=4,
    )

    result = runner.run(seed=41)

    assert result.is_draw or result.winner is not None
    assert result.trajectory.finalized is True
    assert result.trajectory.decisions
    assert all(decision.policy_trace is not None for decision in result.trajectory.decisions)
    assert {decision.kind for decision in result.trajectory.decisions} >= {
        DecisionKind.TIMING,
        DecisionKind.VOTE,
    }
    assert all(
        decision.reward in {-1.0, 0.0, 1.0}
        for decision in result.trajectory.decisions
    )
