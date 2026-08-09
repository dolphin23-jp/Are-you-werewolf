from pathlib import Path

import numpy as np
import pytest

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.meta_strategy import PopulationMetaStrategy, PopulationWeight

torch = pytest.importorskip("torch")
torch_historical = pytest.importorskip("app.training.torch_historical")
torch_historical_state = pytest.importorskip(
    "app.training.torch_historical_run_state"
)
torch_policy = pytest.importorskip("app.training.torch_policy")
torch_pool = pytest.importorskip("app.training.torch_pool")
torch_trainer = pytest.importorskip("app.training.torch_trainer")
TorchHistoricalTrainingLoop = torch_historical.TorchHistoricalTrainingLoop
TorchHistoricalRunProgress = torch_historical_state.TorchHistoricalRunProgress
load_torch_historical_run_state = (
    torch_historical_state.load_torch_historical_run_state
)
save_torch_historical_run_state = (
    torch_historical_state.save_torch_historical_run_state
)
TorchTransformerPolicy = torch_policy.TorchTransformerPolicy
TransformerPolicyConfig = torch_policy.TransformerPolicyConfig
TorchPolicyPool = torch_pool.TorchPolicyPool
TorchPPOConfig = torch_trainer.TorchPPOConfig


def _specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=False)
        for i in range(17)
    ]


def _model(seed: int) -> TorchTransformerPolicy:
    torch.manual_seed(seed)
    return TorchTransformerPolicy(
        TransformerPolicyConfig(
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
    ).eval()


def _strategy(policy_ids: tuple[str, str]) -> PopulationMetaStrategy:
    return PopulationMetaStrategy(
        by_team={
            team: (
                PopulationWeight(policy_ids[0], 0.65),
                PopulationWeight(policy_ids[1], 0.35),
            )
            for team in Team
        },
        temperature=0.75,
    )


def _parameters(loop: TorchHistoricalTrainingLoop):
    return tuple(parameter.detach().cpu().clone() for parameter in loop.model.parameters())


def test_historical_run_state_restores_exact_next_batch(tmp_path: Path):
    pool = TorchPolicyPool(tmp_path / "pool")
    first = pool.add(_model(1801))
    second = pool.add(_model(1803), parent_id=first.policy_id)
    strategy = _strategy((first.policy_id, second.policy_id))
    torch.manual_seed(1805)
    original = TorchHistoricalTrainingLoop(
        _specs(),
        _model(1807),
        pool,
        opponent_strategy=strategy,
        opponent_seed=1809,
        max_discussion_ticks=0,
        max_parallel_games=1,
        max_inference_batch_size=5,
        trainer_seed=1811,
        ppo_config=TorchPPOConfig(
            learning_rate=1e-3,
            epochs=1,
            minibatch_size=64,
        ),
    )
    original.train_batch(
        learner_team=Team.VILLAGE,
        start_seed=1813,
        episodes=1,
    )
    progress = TorchHistoricalRunProgress(
        completed_batches=1,
        base_seed=1813,
        episodes_per_batch=1,
        requested_teams=tuple(Team),
        parent_policy_id=second.policy_id,
        next_pool_generation=pool.next_generation,
    )
    path = tmp_path / "historical-run.npz"
    save_torch_historical_run_state(original, progress, path)

    restored, restored_progress = load_torch_historical_run_state(
        path,
        _specs(),
        pool,
    )

    assert restored_progress == progress
    assert restored_progress.next_learner_team is Team.WEREWOLF
    assert restored_progress.next_start_seed == 1814
    assert restored.opponent_strategy == strategy
    assert restored.optimizer.config == original.optimizer.config
    assert restored.max_discussion_ticks == original.max_discussion_ticks
    assert restored.max_parallel_games == original.max_parallel_games
    assert restored.max_inference_batch_size == original.max_inference_batch_size
    assert (
        restored.checkpoint_opponent_rng_state()
        == original.checkpoint_opponent_rng_state()
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(_parameters(original), _parameters(restored), strict=True)
    )

    original_stats = original.train_batch(
        learner_team=progress.next_learner_team,
        start_seed=progress.next_start_seed,
        episodes=progress.episodes_per_batch,
    )
    restored_stats = restored.train_batch(
        learner_team=restored_progress.next_learner_team,
        start_seed=restored_progress.next_start_seed,
        episodes=restored_progress.episodes_per_batch,
    )

    assert original_stats.opponent_policy_ids == restored_stats.opponent_policy_ids
    assert original_stats.update == restored_stats.update
    assert all(
        torch.equal(left, right)
        for left, right in zip(_parameters(original), _parameters(restored), strict=True)
    )


def test_historical_run_state_archive_is_pickle_free(tmp_path: Path):
    pool = TorchPolicyPool(tmp_path / "safe-pool")
    entry = pool.add(_model(1821))
    loop = TorchHistoricalTrainingLoop(
        _specs(),
        _model(1823),
        pool,
        opponent_seed=1825,
        max_discussion_ticks=0,
        trainer_seed=1827,
    )
    path = tmp_path / "safe-historical-run.npz"
    save_torch_historical_run_state(
        loop,
        TorchHistoricalRunProgress(
            completed_batches=0,
            base_seed=1829,
            episodes_per_batch=2,
            requested_teams=(Team.FOX,),
            parent_policy_id=entry.policy_id,
            next_pool_generation=pool.next_generation,
        ),
        path,
    )

    with np.load(path, allow_pickle=False) as archive:
        assert "__metadata__" in archive.files
        assert "trainer_generator" in archive.files
        assert archive["__metadata__"].dtype == np.uint8
        assert archive["trainer_generator"].dtype == np.uint8
        assert all(archive[key].dtype != object for key in archive.files)


def test_historical_run_progress_rotates_teams_and_seeds():
    progress = TorchHistoricalRunProgress(
        completed_batches=5,
        base_seed=1901,
        episodes_per_batch=4,
        requested_teams=(Team.VILLAGE, Team.FOX),
    )

    assert progress.next_learner_team is Team.FOX
    assert progress.next_start_seed == 1921
