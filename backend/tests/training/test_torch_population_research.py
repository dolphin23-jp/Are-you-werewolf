from pathlib import Path

import pytest

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.meta_strategy import PolicyWeight, PopulationMetaStrategy
from app.training.population_payoff import PolicyProfile

torch = pytest.importorskip("torch")
torch_policy = pytest.importorskip("app.training.torch_policy")
torch_pool = pytest.importorskip("app.training.torch_pool")
torch_research = pytest.importorskip("app.training.torch_population_research")
torch_trainer = pytest.importorskip("app.training.torch_trainer")
TorchTransformerPolicy = torch_policy.TorchTransformerPolicy
TransformerPolicyConfig = torch_policy.TransformerPolicyConfig
TorchPolicyPool = torch_pool.TorchPolicyPool
AdaptivePayoffPending = torch_research.AdaptivePayoffPending
TorchPopulationResearchConfig = torch_research.TorchPopulationResearchConfig
TorchPopulationResearchRun = torch_research.TorchPopulationResearchRun
TorchPopulationResearchState = torch_research.TorchPopulationResearchState
TorchPopulationRunPhase = torch_research.TorchPopulationRunPhase
load_torch_population_research_state = (
    torch_research.load_torch_population_research_state
)
save_torch_population_research_state = (
    torch_research.save_torch_population_research_state
)
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


def _config(*, extra_games: int = 0) -> TorchPopulationResearchConfig:
    return TorchPopulationResearchConfig(
        recent_policies=1,
        games_per_profile=1,
        extra_games=extra_games,
        uncertainty_prior=0.5,
        oracle_episodes=1,
        oracle_batch_size=1,
        evaluation_seed=2401,
        oracle_seed=2403,
        opponent_seed=2405,
        max_discussion_ticks=0,
        max_parallel_games=1,
        max_inference_batch_size=5,
        meta_temperature=0.25,
        meta_iterations=2,
        meta_damping=0.5,
        ppo_config=TorchPPOConfig(
            learning_rate=1e-3,
            epochs=1,
            minibatch_size=64,
        ),
    )


def test_population_research_default_recent_policies_is_five():
    assert TorchPopulationResearchConfig().recent_policies == 5


def test_population_research_updates_recent_policies_only_at_idle_boundary(
    tmp_path: Path,
):
    pool = TorchPolicyPool(tmp_path / "boundary-pool")
    pool.add(_model(2407))
    run_dir = tmp_path / "boundary-run"
    run = TorchPopulationResearchRun(_specs(), pool, run_dir)
    original = run.start(_config())

    updated = run.set_recent_policies(5)
    assert updated.completed_iterations == original.completed_iterations
    assert updated.phase is TorchPopulationRunPhase.IDLE
    assert updated.config.recent_policies == 5
    assert updated.config.games_per_profile == original.config.games_per_profile
    assert updated.config.ppo_config == original.config.ppo_config

    restored = TorchPopulationResearchRun(_specs(), pool, run_dir)
    restored_state = restored.resume()
    assert restored_state.config.recent_policies == 5

    with pytest.raises(ValueError, match="recent_policies must be positive"):
        restored.set_recent_policies(0)
    assert restored.state is not None
    assert restored.state.config.recent_policies == 5

    assert restored.step().kind == "iteration_started"
    with pytest.raises(ValueError, match="idle iteration boundary"):
        restored.set_recent_policies(3)
    assert restored.state is not None
    assert restored.state.config.recent_policies == 5


def test_population_research_state_roundtrips_active_strategy_and_pending(tmp_path: Path):
    profile = PolicyProfile("g000000", "g000001", "g000002")
    strategy = PopulationMetaStrategy(
        village=(PolicyWeight("g000000", 1.0),),
        werewolf=(PolicyWeight("g000001", 1.0),),
        fox=(PolicyWeight("g000002", 1.0),),
    )
    measure = TorchPopulationResearchState(
        config=_config(extra_games=2),
        completed_iterations=3,
        phase=TorchPopulationRunPhase.MEASURE,
        village_policy_ids=("g000000",),
        werewolf_policy_ids=("g000001",),
        fox_policy_ids=("g000002",),
        iteration_pool_generation=7,
        adaptive_games_completed=1,
        adaptive_pending=AdaptivePayoffPending(profile, 4, 99),
    )
    oracle = TorchPopulationResearchState(
        config=_config(extra_games=1),
        completed_iterations=2,
        phase=TorchPopulationRunPhase.ORACLE,
        village_policy_ids=("g000000",),
        werewolf_policy_ids=("g000001",),
        fox_policy_ids=("g000002",),
        iteration_pool_generation=6,
        adaptive_games_completed=1,
        meta_strategy=strategy,
    )

    measure_path = tmp_path / "measure.json"
    oracle_path = tmp_path / "oracle.json"
    save_torch_population_research_state(measure, measure_path)
    save_torch_population_research_state(oracle, oracle_path)

    assert load_torch_population_research_state(measure_path) == measure
    assert load_torch_population_research_state(oracle_path) == oracle


def test_population_research_recovers_payoff_commit_before_outer_state(tmp_path: Path):
    pool = TorchPolicyPool(tmp_path / "pool")
    base = pool.add(_model(2411))
    run_dir = tmp_path / "run"
    run = TorchPopulationResearchRun(_specs(), pool, run_dir)
    run.start(_config(extra_games=1))

    assert run.step().kind == "iteration_started"
    assert run.step().kind == "payoff_batch"
    selected = run.step()
    assert selected.kind == "adaptive_selected"
    state = run.state
    assert state is not None
    pending = state.adaptive_pending
    assert pending is not None

    # Simulate a process dying after the payoff-table atomic write but before
    # clearing the pending marker in the outer research state.
    before = run.table.get(pending.profile)
    assert before is not None
    assert before.games == pending.games_before
    run.table.record_result(
        pending.profile,
        winner=Team.VILLAGE,
        is_draw=False,
        days=3,
    )

    restored = TorchPopulationResearchRun(_specs(), pool, run_dir)
    restored_state = restored.resume()
    assert restored_state.adaptive_pending == pending
    event = restored.step()

    assert event.kind == "adaptive_completed"
    assert "recovered_table_commit=true" in event.message
    assert restored.state is not None
    assert restored.state.adaptive_games_completed == 1
    assert restored.state.adaptive_pending is None
    after = restored.table.get(pending.profile)
    assert after is not None
    assert after.games == pending.games_before + 1
    assert pool.next_generation == base.generation + 1


def test_population_research_end_to_end_resumes_after_measured_cube(tmp_path: Path):
    pool = TorchPolicyPool(tmp_path / "end-to-end-pool")
    base = pool.add(_model(2421))
    run_dir = tmp_path / "end-to-end-run"
    initial = TorchPopulationResearchRun(_specs(), pool, run_dir)
    initial.start(_config())

    assert initial.step().kind == "iteration_started"
    measured = initial.step()
    assert measured.kind == "payoff_batch"
    assert initial.state is not None
    assert initial.state.phase is TorchPopulationRunPhase.MEASURE
    assert pool.next_generation == 1

    # A fresh process sees the persisted payoff table, does not replay that game,
    # solves the frozen one-policy cube, and completes exactly three specialists.
    resumed = TorchPopulationResearchRun(_specs(), pool, run_dir)
    resumed.resume()
    final = resumed.run_until(1)

    assert final.completed_iterations == 1
    assert final.phase is TorchPopulationRunPhase.IDLE
    assert pool.next_generation == 4
    assert len(pool.entries) == 4
    assert pool.get("g000001").parent_id == base.policy_id
    assert pool.get("g000001").specialized_team is Team.VILLAGE
    assert pool.get("g000002").parent_id == base.policy_id
    assert pool.get("g000002").specialized_team is Team.WEREWOLF
    assert pool.get("g000003").parent_id == base.policy_id
    assert pool.get("g000003").specialized_team is Team.FOX
    record = resumed.table.get(
        PolicyProfile(base.policy_id, base.policy_id, base.policy_id)
    )
    assert record is not None
    assert record.games == 1
    assert (run_dir / "iteration-0001" / "meta.json").exists()
    assert (run_dir / "iteration-0001" / "oracle.run.npz").exists()
    assert (run_dir / "iteration-0001" / "summary.json").exists()


def test_population_research_freezes_iteration_policy_set(tmp_path: Path):
    pool = TorchPolicyPool(tmp_path / "freeze-pool")
    base = pool.add(_model(2431))
    run = TorchPopulationResearchRun(_specs(), pool, tmp_path / "freeze-run")
    run.start(_config())
    run.step()
    state = run.state
    assert state is not None
    assert state.village_policy_ids == (base.policy_id,)

    pool.add(
        _model(2433),
        parent_id=base.policy_id,
        specialized_team=Team.VILLAGE,
    )

    with pytest.raises(ValueError, match="pool changed during frozen payoff measurement"):
        run.step()
    assert run.state is not None
    assert run.state.village_policy_ids == (base.policy_id,)
