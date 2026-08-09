from dataclasses import fields

import pytest

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.encoding import ObservationEncoder
from app.training.env import WerewolfTrainingEnv

torch = pytest.importorskip("torch")
torch_policy = pytest.importorskip("app.training.torch_policy")
torch_vectorized = pytest.importorskip("app.training.torch_vectorized")
TorchTransformerPolicy = torch_policy.TorchTransformerPolicy
TransformerPolicyConfig = torch_policy.TransformerPolicyConfig
TorchVectorizedEpisodeCollector = torch_vectorized.TorchVectorizedEpisodeCollector
InferenceRequest = torch_vectorized._InferenceRequest


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


def _request(seed: int, player_id: str, model, slot_index: int):
    env = WerewolfTrainingEnv(_specs(), seed=seed)
    observation = env.observe(player_id)
    encoded = ObservationEncoder().encode(observation)
    return InferenceRequest(
        slot_index=slot_index,
        player_id=player_id,
        observation=observation,
        encoded=encoded,
        model=model,
    )


def _assert_logits_close(actual, expected) -> None:
    for field in fields(actual):
        left = getattr(actual, field.name)
        right = getattr(expected, field.name)
        if isinstance(left, tuple):
            assert left == pytest.approx(right, abs=1e-5, rel=1e-5)
        else:
            assert left == pytest.approx(right, abs=1e-5, rel=1e-5)


def test_mixed_model_inference_groups_by_identity_and_restores_request_order():
    torch.manual_seed(1331)
    village = BatchCountingTransformer().eval()
    wolf = BatchCountingTransformer().eval()
    collector = TorchVectorizedEpisodeCollector(_specs(), village)
    requests = [
        _request(1333, "p0", village, 0),
        _request(1335, "p1", wolf, 1),
        _request(1337, "p2", village, 2),
    ]

    expected = [
        request.model.forward(request.encoded)
        for request in requests
    ]
    village.batch_sizes.clear()
    wolf.batch_sizes.clear()

    prepared = collector._infer(requests)

    assert [item.request for item in prepared] == requests
    assert village.batch_sizes == [2]
    assert wolf.batch_sizes == [1]
    for item, direct in zip(prepared, expected, strict=True):
        _assert_logits_close(item.logits, direct)


def test_mixed_model_collector_completes_with_all_faction_models():
    torch.manual_seed(1341)
    village = BatchCountingTransformer().eval()
    wolf = BatchCountingTransformer().eval()
    fox = BatchCountingTransformer().eval()
    team_models = {
        Team.VILLAGE: village,
        Team.WEREWOLF: wolf,
        Team.FOX: fox,
    }

    result = TorchVectorizedEpisodeCollector(
        _specs(),
        village,
        max_discussion_ticks=1,
    ).collect((1343,), team_models=(team_models,))[0]

    assert result.trajectory.finalized
    assert result.winner is not None or result.is_draw
    assert village.batch_sizes
    assert wolf.batch_sizes
    assert fox.batch_sizes


def test_vectorized_collector_rejects_misaligned_team_models():
    model = BatchCountingTransformer().eval()

    with pytest.raises(ValueError, match="team_models must align"):
        TorchVectorizedEpisodeCollector(_specs(), model).collect(
            (1351, 1353),
            team_models=({Team.VILLAGE: model},),
        )
