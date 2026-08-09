from pathlib import Path

import pytest

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.encoding import ObservationEncoder
from app.training.env import WerewolfTrainingEnv
from app.training.numpy_policy import NumpyMLPPolicy
from app.training.policy_pool import NumpyPolicyPool
from app.training.population_payoff import (
    PolicyProfile,
    PopulationPayoffResult,
    PopulationPayoffTable,
    evaluate_policy_profile,
)


def _specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=False)
        for i in range(17)
    ]


def _initialized_model(seed: int) -> NumpyMLPPolicy:
    env = WerewolfTrainingEnv(_specs(), seed=seed)
    observation = ObservationEncoder().encode(env.observe("p0"))
    model = NumpyMLPPolicy(seed=seed, hidden_size=8)
    model.forward(observation)
    return model


def test_population_payoff_persists_empirical_terminal_rewards(tmp_path: Path):
    path = tmp_path / "payoffs.json"
    table = PopulationPayoffTable(path)
    profile = PolicyProfile("g000000", "g000001", "g000002")

    table.record_result(profile, winner=Team.VILLAGE, is_draw=False, days=3)
    table.record_result(profile, winner=Team.FOX, is_draw=False, days=5)
    table.record_result(profile, winner=None, is_draw=True, days=4)

    restored = PopulationPayoffTable(path).get(profile)
    assert restored is not None
    assert restored.games == 3
    assert restored.village_wins == 1
    assert restored.werewolf_wins == 0
    assert restored.fox_wins == 1
    assert restored.draws == 1
    assert restored.mean_payoff(Team.VILLAGE) == 0.0
    assert restored.mean_payoff(Team.WEREWOLF) == -2 / 3
    assert restored.mean_payoff(Team.FOX) == 0.0
    assert restored.mean_days == 4.0


def test_population_payoff_batch_writes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    table = PopulationPayoffTable(tmp_path / "batch-payoffs.json")
    first = PolicyProfile("v0", "w0", "f0")
    second = PolicyProfile("v1", "w0", "f0")
    writes = 0
    original_write = table._write

    def counted_write() -> None:
        nonlocal writes
        writes += 1
        original_write()

    monkeypatch.setattr(table, "_write", counted_write)
    records = table.record_results(
        (
            PopulationPayoffResult(first, Team.VILLAGE, False, 3),
            PopulationPayoffResult(first, None, True, 4),
            PopulationPayoffResult(second, Team.FOX, False, 5),
        )
    )

    assert writes == 1
    assert [record.games for record in records] == [1, 2, 1]
    assert table.get(first) == records[1]
    assert table.get(second) == records[2]
    assert PopulationPayoffTable(table.path).get(first) == records[1]


def test_population_payoff_batch_validates_before_mutation(tmp_path: Path):
    table = PopulationPayoffTable(tmp_path / "invalid-batch.json")
    profile = PolicyProfile("v0", "w0", "f0")

    with pytest.raises(ValueError, match="draw cannot also have a winner"):
        table.record_results(
            (
                PopulationPayoffResult(profile, Team.VILLAGE, False, 2),
                PopulationPayoffResult(profile, Team.FOX, True, 3),
            )
        )

    assert table.get(profile) is None
    assert not table.path.exists()


def test_payoff_uncertainty_remains_positive_and_shrinks_with_evidence(tmp_path: Path):
    table = PopulationPayoffTable(tmp_path / "uncertainty.json")
    profile = PolicyProfile("v0", "w0", "f0")
    first = table.record_result(
        profile,
        winner=Team.VILLAGE,
        is_draw=False,
        days=3,
    )
    first_std = first.posterior_payoff_std(Team.VILLAGE)

    latest = first
    for _ in range(20):
        latest = table.record_result(
            profile,
            winner=Team.VILLAGE,
            is_draw=False,
            days=3,
        )

    assert first_std > 0.0
    assert latest.posterior_payoff_std(Team.VILLAGE) > 0.0
    assert latest.posterior_payoff_std(Team.VILLAGE) < first_std
    assert latest.posterior_payoff_mean(Team.VILLAGE) < 1.0
    assert latest.posterior_payoff_mean(Team.VILLAGE) > 0.0
    assert latest.max_posterior_payoff_std() >= latest.posterior_payoff_std(
        Team.VILLAGE
    )


def test_profile_evaluation_runs_three_saved_policies(tmp_path: Path):
    pool = NumpyPolicyPool(tmp_path / "pool")
    entries = [pool.add(_initialized_model(seed)) for seed in (301, 302, 303)]
    profile = PolicyProfile(*(entry.policy_id for entry in entries))
    table = PopulationPayoffTable(tmp_path / "payoffs.json")

    record = evaluate_policy_profile(
        _specs(),
        pool,
        table,
        profile,
        seeds=(304,),
        max_discussion_ticks=2,
    )

    assert record.games == 1
    assert record.village_wins + record.werewolf_wins + record.fox_wins + record.draws == 1
    assert table.has_complete_cube(
        (profile.village,),
        (profile.werewolf,),
        (profile.fox,),
    )
