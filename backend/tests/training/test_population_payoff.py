from pathlib import Path

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.encoding import ObservationEncoder
from app.training.env import WerewolfTrainingEnv
from app.training.numpy_policy import NumpyMLPPolicy
from app.training.policy_pool import NumpyPolicyPool
from app.training.population_payoff import (
    PolicyProfile,
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
