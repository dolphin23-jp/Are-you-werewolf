import pytest

from app.engine.game import PlayerSpec
from app.engine.roles import Team

torch = pytest.importorskip("torch")
torch_policy = pytest.importorskip("app.training.torch_policy")
torch_runner = pytest.importorskip("app.training.torch_runner")
torch_vectorized = pytest.importorskip("app.training.torch_vectorized")
TorchTransformerPolicy = torch_policy.TorchTransformerPolicy
TransformerPolicyConfig = torch_policy.TransformerPolicyConfig
TorchBatchedEpisodeRunner = torch_runner.TorchBatchedEpisodeRunner
TorchVectorizedEpisodeCollector = torch_vectorized.TorchVectorizedEpisodeCollector


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


def _trajectory_signature(result):
    return tuple(
        (
            decision.player_id,
            decision.kind,
            decision.speech_bundle,
            decision.target_id,
            decision.night_topic,
        )
        for decision in result.trajectory.decisions
    )


def test_mixed_model_vectorized_rollout_matches_single_game_grouped_runner():
    torch.manual_seed(1331)
    village = BatchCountingTransformer().eval()
    wolf = BatchCountingTransformer().eval()
    fox = BatchCountingTransformer().eval()
    team_models = {
        Team.VILLAGE: village,
        Team.WEREWOLF: wolf,
        Team.FOX: fox,
    }

    sequential = TorchBatchedEpisodeRunner(
        _specs(),
        village,
        team_models=team_models,
        max_discussion_ticks=1,
    ).run(1333)

    village.batch_sizes.clear()
    wolf.batch_sizes.clear()
    fox.batch_sizes.clear()
    vectorized = TorchVectorizedEpisodeCollector(
        _specs(),
        village,
        max_discussion_ticks=1,
    ).collect((1333,), team_models=(team_models,))[0]

    assert vectorized.winner == sequential.winner
    assert vectorized.is_draw == sequential.is_draw
    assert vectorized.days == sequential.days
    assert vectorized.semantic_event_count == sequential.semantic_event_count
    assert _trajectory_signature(vectorized) == _trajectory_signature(sequential)
    assert village.batch_sizes
    assert wolf.batch_sizes
    assert fox.batch_sizes


def test_vectorized_collector_rejects_misaligned_team_models():
    model = BatchCountingTransformer().eval()

    with pytest.raises(ValueError, match="team_models must align"):
        TorchVectorizedEpisodeCollector(_specs(), model).collect(
            (1343, 1345),
            team_models=({Team.VILLAGE: model},),
        )
