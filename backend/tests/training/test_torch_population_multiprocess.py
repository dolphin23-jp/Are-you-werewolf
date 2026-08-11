from pathlib import Path

import pytest

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.population_payoff import PolicyProfile, PopulationPayoffTable

torch = pytest.importorskip("torch")
torch_policy = pytest.importorskip("app.training.torch_policy")
torch_pool = pytest.importorskip("app.training.torch_pool")
torch_population = pytest.importorskip("app.training.torch_population")
torch_multiprocess = pytest.importorskip(
    "app.training.torch_population_multiprocess"
)
TorchTransformerPolicy = torch_policy.TorchTransformerPolicy
TransformerPolicyConfig = torch_policy.TransformerPolicyConfig
TorchPolicyPool = torch_pool.TorchPolicyPool
TorchProfileEvaluationRequest = torch_population.TorchProfileEvaluationRequest
evaluate_torch_policy_profiles = torch_population.evaluate_torch_policy_profiles
evaluate_torch_policy_profiles_multiprocess = (
    torch_multiprocess.evaluate_torch_policy_profiles_multiprocess
)


def _specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=False)
        for i in range(17)
    ]


def _zero_model(seed: int):
    torch.manual_seed(seed)
    model = TorchTransformerPolicy(
        TransformerPolicyConfig(
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
    ).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


def test_multiprocess_population_evaluation_matches_single_process(tmp_path: Path):
    pool = TorchPolicyPool(tmp_path / "pool")
    village = pool.add(_zero_model(1601), specialized_team=Team.VILLAGE)
    wolf = pool.add(_zero_model(1602), specialized_team=Team.WEREWOLF)
    fox = pool.add(_zero_model(1603), specialized_team=Team.FOX)
    profile = PolicyProfile(village.policy_id, wolf.policy_id, fox.policy_id)
    requests = (TorchProfileEvaluationRequest(profile, (1604, 1605)),)
    single_table = PopulationPayoffTable(tmp_path / "single.json")
    multiprocess_table = PopulationPayoffTable(tmp_path / "multiprocess.json")

    single_stats = evaluate_torch_policy_profiles(
        _specs(),
        pool,
        single_table,
        requests,
        max_discussion_ticks=0,
        max_parallel_games=2,
        max_inference_batch_size=32,
    )
    multiprocess_stats = evaluate_torch_policy_profiles_multiprocess(
        _specs(),
        pool,
        multiprocess_table,
        requests,
        worker_count=2,
        max_discussion_ticks=0,
        max_parallel_games=2,
        max_inference_batch_size=32,
        inference_coalesce_seconds=0.01,
    )

    assert multiprocess_table.get(profile) == single_table.get(profile)
    assert multiprocess_stats.games == single_stats.games == 2
    assert multiprocess_stats.rollout_chunks == single_stats.rollout_chunks == 1
    assert multiprocess_stats.checkpoint_loads == single_stats.checkpoint_loads == 3
    assert multiprocess_stats.inference_calls > 0
    assert 0 < multiprocess_stats.max_inference_batch <= 32


def test_multiprocess_population_evaluation_validates_worker_limits(tmp_path: Path):
    pool = TorchPolicyPool(tmp_path / "pool-limits")
    village = pool.add(_zero_model(1611), specialized_team=Team.VILLAGE)
    wolf = pool.add(_zero_model(1612), specialized_team=Team.WEREWOLF)
    fox = pool.add(_zero_model(1613), specialized_team=Team.FOX)
    request = TorchProfileEvaluationRequest(
        PolicyProfile(village.policy_id, wolf.policy_id, fox.policy_id),
        (1614,),
    )
    table = PopulationPayoffTable(tmp_path / "limits.json")

    with pytest.raises(ValueError, match="worker_count must be positive"):
        evaluate_torch_policy_profiles_multiprocess(
            _specs(),
            pool,
            table,
            (request,),
            worker_count=0,
        )
    with pytest.raises(ValueError, match="inference_coalesce_seconds"):
        evaluate_torch_policy_profiles_multiprocess(
            _specs(),
            pool,
            table,
            (request,),
            inference_coalesce_seconds=-0.1,
        )
