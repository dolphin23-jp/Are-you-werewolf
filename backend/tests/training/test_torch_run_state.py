from pathlib import Path

import numpy as np
import pytest

from app.engine.game import PlayerSpec

torch = pytest.importorskip("torch")
torch_policy = pytest.importorskip("app.training.torch_policy")
torch_run_state = pytest.importorskip("app.training.torch_run_state")
torch_self_play = pytest.importorskip("app.training.torch_self_play")
torch_trainer = pytest.importorskip("app.training.torch_trainer")
TransformerPolicyConfig = torch_policy.TransformerPolicyConfig
TorchRunProgress = torch_run_state.TorchRunProgress
load_torch_run_state = torch_run_state.load_torch_run_state
save_torch_run_state = torch_run_state.save_torch_run_state
TorchSelfPlayTrainingLoop = torch_self_play.TorchSelfPlayTrainingLoop
TorchPPOConfig = torch_trainer.TorchPPOConfig


def _specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=False)
        for i in range(17)
    ]


def _loop() -> TorchSelfPlayTrainingLoop:
    return TorchSelfPlayTrainingLoop(
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
        max_parallel_games=1,
        max_inference_batch_size=5,
        trainer_seed=1703,
    )


def _parameters(loop: TorchSelfPlayTrainingLoop):
    return tuple(parameter.detach().cpu().clone() for parameter in loop.model.parameters())


def test_run_state_restores_exact_next_batch_training(tmp_path: Path):
    torch.manual_seed(1701)
    original = _loop()
    original.train_batch(start_seed=1705, episodes=1)
    progress = TorchRunProgress(
        completed_episodes=1,
        batch_number=1,
        base_seed=1705,
        parent_policy_id="g000003",
        next_pool_generation=4,
    )
    path = tmp_path / "run-state.npz"
    save_torch_run_state(original, progress, path)

    restored, restored_progress = load_torch_run_state(path, _specs())

    assert restored_progress == progress
    assert restored.optimizer.config == original.optimizer.config
    assert restored.max_discussion_ticks == original.max_discussion_ticks
    assert restored.max_parallel_games == original.max_parallel_games
    assert restored.max_inference_batch_size == original.max_inference_batch_size
    assert all(
        torch.equal(left, right)
        for left, right in zip(_parameters(original), _parameters(restored), strict=True)
    )

    original_stats = original.train_batch(start_seed=1706, episodes=1)
    restored_stats = restored.train_batch(start_seed=1706, episodes=1)

    assert original_stats.update == restored_stats.update
    assert original_stats.inference_calls == restored_stats.inference_calls
    assert original_stats.max_inference_batch == restored_stats.max_inference_batch
    assert all(
        torch.equal(left, right)
        for left, right in zip(_parameters(original), _parameters(restored), strict=True)
    )


def test_run_state_archive_is_pickle_free(tmp_path: Path):
    torch.manual_seed(1707)
    loop = _loop()
    path = tmp_path / "safe-run-state.npz"
    save_torch_run_state(
        loop,
        TorchRunProgress(completed_episodes=0, batch_number=0, base_seed=1711),
        path,
    )

    with np.load(path, allow_pickle=False) as archive:
        assert "__metadata__" in archive.files
        assert "trainer_generator" in archive.files
        assert archive["__metadata__"].dtype == np.uint8
        assert archive["trainer_generator"].dtype == np.uint8
        assert all(archive[key].dtype != object for key in archive.files)
