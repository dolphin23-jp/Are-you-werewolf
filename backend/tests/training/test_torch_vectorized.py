import pytest

from app.engine.game import PlayerSpec

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


def test_vectorized_collector_finishes_multiple_independent_games():
    torch.manual_seed(1301)
    model = BatchCountingTransformer().eval()

    results = TorchVectorizedEpisodeCollector(
        _specs(),
        model,
        max_discussion_ticks=0,
    ).collect((1303, 1305, 1307))

    assert len(results) == 3
    assert all(result.trajectory.finalized for result in results)
    assert all(result.winner is not None or result.is_draw for result in results)
    assert len({result.trajectory.episode_id for result in results}) == 3
    assert model.batch_sizes
    assert max(model.batch_sizes) > 17
    assert max(model.batch_sizes) <= 51


def test_vectorized_collector_rejects_duplicate_seeds():
    model = BatchCountingTransformer().eval()

    with pytest.raises(ValueError, match="seeds must be unique"):
        TorchVectorizedEpisodeCollector(_specs(), model).collect((1311, 1311))
