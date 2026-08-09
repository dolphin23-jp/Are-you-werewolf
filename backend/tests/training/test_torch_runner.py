import pytest

from app.engine.game import PlayerSpec

torch = pytest.importorskip("torch")
torch_policy = pytest.importorskip("app.training.torch_policy")
torch_runner = pytest.importorskip("app.training.torch_runner")
TorchTransformerPolicy = torch_policy.TorchTransformerPolicy
TransformerPolicyConfig = torch_policy.TransformerPolicyConfig
TorchBatchedEpisodeRunner = torch_runner.TorchBatchedEpisodeRunner


class CountingTransformer(TorchTransformerPolicy):
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
        self.forward_batch_calls = 0

    def forward_batch(self, observations):
        self.forward_batch_calls += 1
        return super().forward_batch(observations)


def _specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=False)
        for i in range(17)
    ]


def test_batched_runner_completes_game_with_fewer_forwards_than_decisions():
    torch.manual_seed(801)
    model = CountingTransformer().eval()

    result = TorchBatchedEpisodeRunner(
        _specs(),
        model,
        max_discussion_ticks=1,
    ).run(803)

    assert result.trajectory.finalized is True
    assert result.winner is not None or result.is_draw
    assert len(result.trajectory.decisions) > 0
    assert model.forward_batch_calls > 0
    assert model.forward_batch_calls < len(result.trajectory.decisions)
