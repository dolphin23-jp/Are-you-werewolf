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


def test_vectorized_collector_caps_inference_microbatches_and_reports_shape():
    torch.manual_seed(1311)
    model = BatchCountingTransformer().eval()
    collector = TorchVectorizedEpisodeCollector(
        _specs(),
        model,
        max_discussion_ticks=0,
        max_inference_batch_size=7,
    )

    results = collector.collect((1313, 1315, 1317))
    stats = collector.inference_stats

    assert len(results) == 3
    assert model.batch_sizes
    assert max(model.batch_sizes) <= 7
    assert stats.max_inference_batch <= 7
    assert stats.max_pending_requests > 7
    assert stats.inference_calls == len(model.batch_sizes)
    assert stats.inference_observations == sum(model.batch_sizes)
    assert stats.mean_inference_batch == pytest.approx(
        sum(model.batch_sizes) / len(model.batch_sizes)
    )
    assert stats.microbatch_expansion > 1.0


def test_inference_microbatching_preserves_structured_rollout_semantics():
    torch.manual_seed(1321)
    unbounded_model = BatchCountingTransformer().eval()
    limited_model = BatchCountingTransformer().eval()
    limited_model.load_state_dict(unbounded_model.state_dict())

    unbounded = TorchVectorizedEpisodeCollector(
        _specs(),
        unbounded_model,
        max_discussion_ticks=1,
    ).collect((1323,))[0]
    limited = TorchVectorizedEpisodeCollector(
        _specs(),
        limited_model,
        max_discussion_ticks=1,
        max_inference_batch_size=5,
    ).collect((1323,))[0]

    assert limited.winner == unbounded.winner
    assert limited.is_draw == unbounded.is_draw
    assert limited.days == unbounded.days
    assert limited.semantic_event_count == unbounded.semantic_event_count
    assert _trajectory_signature(limited) == _trajectory_signature(unbounded)
    assert limited_model.batch_sizes
    assert max(limited_model.batch_sizes) <= 5


def test_vectorized_collector_rejects_duplicate_seeds():
    model = BatchCountingTransformer().eval()

    with pytest.raises(ValueError, match="seeds must be unique"):
        TorchVectorizedEpisodeCollector(_specs(), model).collect((1331, 1331))


def test_vectorized_collector_rejects_invalid_inference_batch_limit():
    model = BatchCountingTransformer().eval()

    with pytest.raises(ValueError, match="max_inference_batch_size must be positive"):
        TorchVectorizedEpisodeCollector(
            _specs(),
            model,
            max_inference_batch_size=0,
        )
