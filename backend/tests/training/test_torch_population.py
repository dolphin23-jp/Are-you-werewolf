from pathlib import Path

import pytest

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.population_payoff import PolicyProfile, PopulationPayoffTable

torch = pytest.importorskip("torch")
torch_policy = pytest.importorskip("app.training.torch_policy")
torch_pool = pytest.importorskip("app.training.torch_pool")
torch_population = pytest.importorskip("app.training.torch_population")
TorchTransformerPolicy = torch_policy.TorchTransformerPolicy
TransformerPolicyConfig = torch_policy.TransformerPolicyConfig
TorchPolicyPool = torch_pool.TorchPolicyPool
TorchProfileEvaluationRequest = torch_population.TorchProfileEvaluationRequest
evaluate_torch_policy_profile = torch_population.evaluate_torch_policy_profile
evaluate_torch_policy_profiles = torch_population.evaluate_torch_policy_profiles


def _specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=False)
        for i in range(17)
    ]


def _model(seed: int):
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


class _CountingTorchPolicyPool(TorchPolicyPool):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.load_calls: list[str] = []

    def load(self, policy_id: str):
        self.load_calls.append(policy_id)
        return super().load(policy_id)


def test_torch_population_profile_records_terminal_outcome(tmp_path: Path):
    pool = TorchPolicyPool(tmp_path / "pool")
    village = pool.add(_model(1101), specialized_team=Team.VILLAGE)
    wolf = pool.add(_model(1102), specialized_team=Team.WEREWOLF)
    fox = pool.add(_model(1103), specialized_team=Team.FOX)
    profile = PolicyProfile(village.policy_id, wolf.policy_id, fox.policy_id)
    table = PopulationPayoffTable(tmp_path / "payoffs.json")

    record = evaluate_torch_policy_profile(
        _specs(),
        pool,
        table,
        profile,
        seeds=(1104,),
        max_discussion_ticks=0,
    )

    assert record.games == 1
    assert record.village_wins + record.werewolf_wins + record.fox_wins + record.draws == 1
    assert table.has_complete_cube(
        (village.policy_id,),
        (wolf.policy_id,),
        (fox.policy_id,),
    )


def test_vectorized_population_evaluation_shares_models_across_profiles(
    tmp_path: Path,
):
    pool = _CountingTorchPolicyPool(tmp_path / "vectorized-pool")
    village = pool.add(_model(1111), specialized_team=Team.VILLAGE)
    wolf = pool.add(_model(1112), specialized_team=Team.WEREWOLF)
    first_fox = pool.add(_model(1113), specialized_team=Team.FOX)
    second_fox = pool.add(_model(1114), specialized_team=Team.FOX)
    first = PolicyProfile(village.policy_id, wolf.policy_id, first_fox.policy_id)
    second = PolicyProfile(village.policy_id, wolf.policy_id, second_fox.policy_id)
    table = PopulationPayoffTable(tmp_path / "vectorized-payoffs.json")

    stats = evaluate_torch_policy_profiles(
        _specs(),
        pool,
        table,
        (
            TorchProfileEvaluationRequest(first, (1115,)),
            TorchProfileEvaluationRequest(second, (1116,)),
        ),
        max_discussion_ticks=0,
        max_parallel_games=2,
    )

    assert table.get(first) is not None
    assert table.get(first).games == 1
    assert table.get(second) is not None
    assert table.get(second).games == 1
    assert stats.games == 2
    assert stats.rollout_chunks == 1
    assert stats.checkpoint_loads == 4
    assert len(pool.load_calls) == 4
    assert stats.inference_calls > 0
    assert stats.max_inference_batch > 17
    assert stats.mean_inference_batch > 0.0


def test_vectorized_population_evaluation_handles_duplicate_seeds_and_microbatches(
    tmp_path: Path,
):
    pool = TorchPolicyPool(tmp_path / "duplicate-seed-pool")
    village = pool.add(_model(1121), specialized_team=Team.VILLAGE)
    wolf = pool.add(_model(1122), specialized_team=Team.WEREWOLF)
    first_fox = pool.add(_model(1123), specialized_team=Team.FOX)
    second_fox = pool.add(_model(1124), specialized_team=Team.FOX)
    first = PolicyProfile(village.policy_id, wolf.policy_id, first_fox.policy_id)
    second = PolicyProfile(village.policy_id, wolf.policy_id, second_fox.policy_id)
    table = PopulationPayoffTable(tmp_path / "duplicate-seed-payoffs.json")

    stats = evaluate_torch_policy_profiles(
        _specs(),
        pool,
        table,
        (
            TorchProfileEvaluationRequest(first, (1125,)),
            TorchProfileEvaluationRequest(second, (1125,)),
        ),
        max_discussion_ticks=0,
        max_parallel_games=2,
        max_inference_batch_size=5,
    )

    assert table.get(first) is not None
    assert table.get(first).games == 1
    assert table.get(second) is not None
    assert table.get(second).games == 1
    assert stats.games == 2
    assert stats.rollout_chunks == 2
    assert 0 < stats.max_inference_batch <= 5
    assert stats.inference_calls > 0


def test_vectorized_population_evaluation_validates_runtime_limits(tmp_path: Path):
    pool = TorchPolicyPool(tmp_path / "limits-pool")
    village = pool.add(_model(1131), specialized_team=Team.VILLAGE)
    wolf = pool.add(_model(1132), specialized_team=Team.WEREWOLF)
    fox = pool.add(_model(1133), specialized_team=Team.FOX)
    request = TorchProfileEvaluationRequest(
        PolicyProfile(village.policy_id, wolf.policy_id, fox.policy_id),
        (1134,),
    )
    table = PopulationPayoffTable(tmp_path / "limits-payoffs.json")

    with pytest.raises(ValueError, match="max_parallel_games must be positive"):
        evaluate_torch_policy_profiles(
            _specs(),
            pool,
            table,
            (request,),
            max_parallel_games=0,
        )
    with pytest.raises(ValueError, match="max_inference_batch_size must be positive"):
        evaluate_torch_policy_profiles(
            _specs(),
            pool,
            table,
            (request,),
            max_inference_batch_size=0,
        )
