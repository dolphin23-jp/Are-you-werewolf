from pathlib import Path

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.encoding import ObservationEncoder
from app.training.env import WerewolfTrainingEnv
from app.training.numpy_policy import NumpyMLPPolicy
from app.training.numpy_trainer import PPOConfig
from app.training.policy_pool import NumpyPolicyPool
from app.training.population_iteration import run_population_iteration
from app.training.population_payoff import PolicyProfile, PopulationPayoffTable


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


def test_population_iteration_measures_solves_and_adds_three_oracles(tmp_path: Path):
    pool = NumpyPolicyPool(tmp_path / "pool")
    first = pool.add(_initialized_model(301))
    table = PopulationPayoffTable(tmp_path / "payoffs.json")

    stats = run_population_iteration(
        _specs(),
        pool,
        table,
        recent_policies=1,
        games_per_profile=1,
        extra_games=1,
        evaluation_seed=302,
        oracle_seed=303,
        opponent_seed=304,
        oracle_episodes=1,
        max_discussion_ticks=1,
        meta_iterations=2,
        ppo_config=PPOConfig(learning_rate=1e-3, epochs=1),
    )

    profile = PolicyProfile(first.policy_id, first.policy_id, first.policy_id)
    record = table.get(profile)
    assert record is not None
    assert record.games == 2
    assert stats.village_policy_ids == (first.policy_id,)
    assert stats.werewolf_policy_ids == (first.policy_id,)
    assert stats.fox_policy_ids == (first.policy_id,)
    assert stats.measured_profiles == 1
    assert stats.new_profile_games == 1
    assert stats.adaptive_games == 1
    for team in Team:
        weights = stats.meta_strategy.weights(team)
        assert len(weights) == 1
        assert weights[0].policy_id == first.policy_id
        assert weights[0].probability == 1.0

    assert len(stats.oracles) == 3
    assert tuple(oracle.team for oracle in stats.oracles) == tuple(Team)
    assert stats.oracles[0].update.decisions > 0
    assert len(pool.entries) == 4
    for oracle in stats.oracles:
        assert oracle.parent_policy_id == first.policy_id
        assert oracle.oracle_entry.parent_id == first.policy_id
        assert oracle.oracle_entry.specialized_team is oracle.team
