import pytest

from app.engine.game import PlayerSpec
from app.engine.roles import Team

torch = pytest.importorskip("torch")
torch_policy = pytest.importorskip("app.training.torch_policy")
torch_vectorized = pytest.importorskip("app.training.torch_vectorized")
TorchTransformerPolicy = torch_policy.TorchTransformerPolicy
TransformerPolicyConfig = torch_policy.TransformerPolicyConfig
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


def test_mixed_model_cross_game_batching_preserves_independent_game_semantics():
    torch.manual_seed(1331)
    village = BatchCountingTransformer().eval()
    wolf = BatchCountingTransformer().eval()
    fox = BatchCountingTransformer().eval()
    team_models = {
        Team.VILLAGE: village,
        Team.WEREWOLF: wolf,
        Team.FOX: fox,
    }

    isolated = TorchVectorizedEpisodeCollector(
        _specs(),
        village,
        max_discussion_ticks=1,
    ).collect((1333,), team_models=(team_models,))[0]

    village.batch_sizes.clear()
    wolf.batch_sizes.clear()
    fox.batch_sizes.clear()
    batched = TorchVectorizedEpisodeCollector(
        _specs(),
        village,
        max_discussion_ticks=1,
    ).collect(
        (1333, 1335),
        team_models=(team_models, team_models),
    )[0]

    assert batched.winner == isolated.winner
    assert batched.is_draw == isolated.is_draw
    assert batched.days == isolated.days
    assert batched.semantic_event_count == isolated.semantic_event_count
    assert _trajectory_signature(batched) == _trajectory_signature(isolated)
    assert village.batch_sizes
    assert wolf.batch_sizes
    assert fox.batch_sizes
    assert max(village.batch_sizes) > 17


def test_vectorized_collector_rejects_misaligned_team_models():
    model = BatchCountingTransformer().eval()

    with pytest.raises(ValueError, match="team_models must align"):
        TorchVectorizedEpisodeCollector(_specs(), model).collect(
            (1343, 1345),
            team_models=({Team.VILLAGE: model},),
        )
