from pathlib import Path

import numpy as np
import pytest

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.meta_strategy import PolicyWeight, PopulationMetaStrategy

torch = pytest.importorskip("torch")
torch_cycle = pytest.importorskip("app.training.torch_oracle_cycle")
torch_oracle_state = pytest.importorskip("app.training.torch_oracle_run_state")
torch_policy = pytest.importorskip("app.training.torch_policy")
torch_pool = pytest.importorskip("app.training.torch_pool")
torch_trainer = pytest.importorskip("app.training.torch_trainer")
TorchOracleRunProgress = torch_cycle.TorchOracleRunProgress
finalize_torch_oracle = torch_cycle.finalize_torch_oracle
start_torch_oracle_cycle = torch_cycle.start_torch_oracle_cycle
train_torch_oracle_subbatch = torch_cycle.train_torch_oracle_subbatch
validate_torch_oracle_pool_boundary = torch_cycle.validate_torch_oracle_pool_boundary
load_torch_oracle_run_state = torch_oracle_state.load_torch_oracle_run_state
save_torch_oracle_run_state = torch_oracle_state.save_torch_oracle_run_state
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


def _strategy(first: str, second: str) -> PopulationMetaStrategy:
    weights = (
        PolicyWeight(first, 0.75),
        PolicyWeight(second, 0.25),
    )
    return PopulationMetaStrategy(
        village=weights,
        werewolf=weights,
        fox=weights,
    )


def _parameters(model: TorchTransformerPolicy):
    return tuple(parameter.detach().cpu().clone() for parameter in model.parameters())


def _start(
    tmp_path: Path,
    *,
    episodes_per_oracle: int = 2,
    oracle_batch_size: int = 1,
):
    pool = TorchPolicyPool(tmp_path / "pool")
    first = pool.add(_model(2101))
    second = pool.add(_model(2103), parent_id=first.policy_id)
    strategy = _strategy(first.policy_id, second.policy_id)
    loop, progress = start_torch_oracle_cycle(
        _specs(),
        pool,
        strategy,
        teams=(Team.VILLAGE, Team.WEREWOLF),
        episodes_per_oracle=episodes_per_oracle,
        oracle_batch_size=oracle_batch_size,
        base_seed=2105,
        opponent_seed=2107,
        trainer_seed=2109,
        ppo_config=TorchPPOConfig(
            learning_rate=1e-3,
            epochs=1,
            minibatch_size=64,
        ),
        max_discussion_ticks=0,
        max_parallel_games=1,
        max_inference_batch_size=5,
    )
    return pool, first, strategy, loop, progress


def test_oracle_run_state_restores_exact_next_subbatch(tmp_path: Path):
    pool, _, strategy, original, progress = _start(tmp_path)
    _, progress = train_torch_oracle_subbatch(original, progress)
    path = tmp_path / "oracle-cycle.npz"
    save_torch_oracle_run_state(original, progress, path)

    restored, restored_progress = load_torch_oracle_run_state(
        path,
        _specs(),
        pool,
    )

    assert restored is not None
    assert restored_progress == progress
    assert restored.opponent_strategy == strategy
    assert (
        restored.checkpoint_opponent_rng_state()
        == original.checkpoint_opponent_rng_state()
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            _parameters(original.model),
            _parameters(restored.model),
            strict=True,
        )
    )

    original_stats, original_progress = train_torch_oracle_subbatch(
        original,
        progress,
    )
    restored_stats, resumed_progress = train_torch_oracle_subbatch(
        restored,
        restored_progress,
    )

    assert original_stats.opponent_policy_ids == restored_stats.opponent_policy_ids
    assert original_stats.update == restored_stats.update
    assert original_progress == resumed_progress
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            _parameters(original.model),
            _parameters(restored.model),
            strict=True,
        )
    )


def test_oracle_finalization_reuses_crash_generation_and_keeps_fixed_parent(
    tmp_path: Path,
):
    pool, first, strategy, loop, progress = _start(
        tmp_path,
        episodes_per_oracle=1,
    )
    _, progress = train_torch_oracle_subbatch(loop, progress)
    path = tmp_path / "pre-finalize.npz"
    save_torch_oracle_run_state(loop, progress, path)

    assert progress.active_parent_policy_id == first.policy_id
    precreated = pool.ensure_generation(
        loop.model,
        generation=progress.next_pool_generation,
        parent_id=progress.active_parent_policy_id,
        specialized_team=Team.VILLAGE,
    )

    restored, restored_progress = load_torch_oracle_run_state(
        path,
        _specs(),
        pool,
    )
    assert restored is not None
    validate_torch_oracle_pool_boundary(pool, restored_progress)

    next_loop, next_progress, reused = finalize_torch_oracle(
        _specs(),
        pool,
        restored,
        restored_progress,
    )

    assert reused == precreated
    assert next_loop is not None
    assert next_progress.active_team is Team.WEREWOLF
    assert next_progress.active_parent_policy_id == first.policy_id
    assert next_loop.opponent_strategy == strategy
    assert next_progress.completed_policy_ids == (precreated.policy_id,)
    assert pool.next_generation == next_progress.next_pool_generation
    validate_torch_oracle_pool_boundary(pool, next_progress)


def test_completed_oracle_run_state_is_pickle_free(tmp_path: Path):
    pool = TorchPolicyPool(tmp_path / "safe-pool")
    base = pool.add(_model(2201))
    specialist = pool.add(
        _model(2203),
        parent_id=base.policy_id,
        specialized_team=Team.VILLAGE,
    )
    progress = TorchOracleRunProgress(
        teams=(Team.VILLAGE,),
        team_index=1,
        completed_episodes=0,
        episodes_per_oracle=3,
        oracle_batch_size=2,
        base_seed=2205,
        opponent_seed=2207,
        trainer_seed=2209,
        next_pool_generation=pool.next_generation,
        active_parent_policy_id=None,
        completed_policy_ids=(specialist.policy_id,),
    )
    path = tmp_path / "complete-oracle.npz"
    save_torch_oracle_run_state(None, progress, path)

    with np.load(path, allow_pickle=False) as archive:
        assert "__metadata__" in archive.files
        assert "__historical_state__" not in archive.files
        assert archive["__metadata__"].dtype == np.uint8
        assert all(archive[key].dtype != object for key in archive.files)

    restored, restored_progress = load_torch_oracle_run_state(
        path,
        _specs(),
        pool,
    )
    assert restored is None
    assert restored_progress == progress
    validate_torch_oracle_pool_boundary(pool, restored_progress)


def test_oracle_progress_handles_partial_final_batch():
    progress = TorchOracleRunProgress(
        teams=(Team.FOX,),
        team_index=0,
        completed_episodes=4,
        episodes_per_oracle=5,
        oracle_batch_size=2,
        base_seed=2301,
        opponent_seed=2303,
        trainer_seed=2305,
        next_pool_generation=7,
        active_parent_policy_id="g000003",
        active_wins=2,
        active_losses=2,
    )

    assert progress.completed_batches == 2
    assert progress.next_batch_episodes == 1
    assert progress.next_start_seed == 2305


def test_oracle_progress_rejects_inconsistent_outcomes():
    with pytest.raises(ValueError, match="outcome counts must equal completed_episodes"):
        TorchOracleRunProgress(
            teams=(Team.VILLAGE,),
            team_index=0,
            completed_episodes=1,
            episodes_per_oracle=2,
            oracle_batch_size=1,
            base_seed=1,
            opponent_seed=2,
            trainer_seed=3,
            next_pool_generation=4,
            active_parent_policy_id="g000000",
        )
