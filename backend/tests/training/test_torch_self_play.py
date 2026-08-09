import pytest

from app.engine.game import PlayerSpec

torch = pytest.importorskip("torch")
torch_policy = pytest.importorskip("app.training.torch_policy")
torch_self_play = pytest.importorskip("app.training.torch_self_play")
torch_trainer = pytest.importorskip("app.training.torch_trainer")
TransformerPolicyConfig = torch_policy.TransformerPolicyConfig
TorchSelfPlayTrainingLoop = torch_self_play.TorchSelfPlayTrainingLoop
TorchPPOConfig = torch_trainer.TorchPPOConfig


def _specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=False)
        for i in range(17)
    ]


def test_transformer_self_play_runs_episode_and_updates_model():
    torch.manual_seed(701)
    loop = TorchSelfPlayTrainingLoop(
        _specs(),
        model_config=TransformerPolicyConfig(
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        ),
        ppo_config=TorchPPOConfig(
            learning_rate=1e-3,
            epochs=1,
            minibatch_size=64,
        ),
        max_discussion_ticks=0,
        trainer_seed=703,
    )

    stats = loop.train_batch(start_seed=705, episodes=1)

    assert stats.episodes == 1
    assert stats.village_wins + stats.werewolf_wins + stats.fox_wins + stats.draws == 1
    assert stats.update.decisions > 0
    assert torch.isfinite(torch.tensor(stats.update.mean_policy_loss))
    assert torch.isfinite(torch.tensor(stats.update.mean_value_loss))
    assert stats.update.gradient_norm > 0.0


def test_transformer_training_rejects_untracked_sampling_temperature():
    with pytest.raises(ValueError, match="requires temperature=1"):
        TorchSelfPlayTrainingLoop(
            _specs(),
            model_config=TransformerPolicyConfig(
                d_model=16,
                nhead=4,
                num_layers=1,
                dim_feedforward=32,
                dropout=0.0,
            ),
            temperature=0.8,
        )
