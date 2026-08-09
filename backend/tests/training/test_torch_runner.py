import pytest

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.learned_runner import LearnedEpisodeRunner

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


def _decision_signature(decision):
    return (
        decision.player_id,
        decision.kind,
        decision.speech_bundle,
        decision.target_id,
        decision.night_topic,
    )


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


def test_batched_runner_matches_sequential_structured_actions_for_same_seed():
    torch.manual_seed(811)
    model = TorchTransformerPolicy(
        TransformerPolicyConfig(
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
    ).eval()

    sequential = LearnedEpisodeRunner(
        _specs(),
        model,
        max_discussion_ticks=1,
    ).run(813)
    batched = TorchBatchedEpisodeRunner(
        _specs(),
        model,
        max_discussion_ticks=1,
    ).run(813)

    assert batched.winner == sequential.winner
    assert batched.is_draw == sequential.is_draw
    assert batched.days == sequential.days
    assert batched.semantic_event_count == sequential.semantic_event_count
    assert [
        _decision_signature(decision) for decision in batched.trajectory.decisions
    ] == [
        _decision_signature(decision) for decision in sequential.trajectory.decisions
    ]


def test_mixed_team_models_batch_each_faction_instead_of_each_seat():
    torch.manual_seed(821)
    village = CountingTransformer().eval()
    werewolf = CountingTransformer().eval()
    fox = CountingTransformer().eval()

    result = TorchBatchedEpisodeRunner(
        _specs(),
        village,
        team_models={
            Team.VILLAGE: village,
            Team.WEREWOLF: werewolf,
            Team.FOX: fox,
        },
        max_discussion_ticks=1,
    ).run(823)

    total_calls = (
        village.forward_batch_calls
        + werewolf.forward_batch_calls
        + fox.forward_batch_calls
    )
    assert result.trajectory.finalized is True
    assert total_calls > 0
    assert total_calls < len(result.trajectory.decisions)
