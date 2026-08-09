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


def test_population_iteration_measures_trains_and_saves_next_generation(tmp_path: Path):
    pool = NumpyPolicyPool(tmp_path / "pool")
    first = pool.add(_initialized_model(301))
    table = PopulationPayoffTable(tmp_path / "payoffs.json")

    stats = run_population_iteration(
        _specs(),
        pool,
        table,
        recent_policies=1,
        games_per_profile=1,
        evaluation_seed=302,
        training_seed=303,
        opponent_seed=304,
        episodes_per_team=1,
        max_discussion_ticks=1,
        meta_iterations=2,
        ppo_config=PPOConfig(learning_rate=1e-3, epochs=1),
    )

    profile = PolicyProfile(first.policy_id, first.policy_id, first.policy_id)
    record = table.get(profile)
    assert record is not None
    assert record.games == 1
    assert stats.measured_policy_ids == (first.policy_id,)
    assert stats.measured_profiles == 1
    assert stats.new_profile_games == 1
    assert len(stats.training) == 3
    assert tuple(item.learner_team for item in stats.training) == tuple(Team)
    assert stats.training[0].update.decisions > 0
    for team in Team:
        weights = stats.meta_strategy.weights(team)
        assert len(weights) == 1
        assert weights[0].policy_id == first.policy_id
        assert weights[0].probability == 1.0
    assert stats.saved_entry.policy_id == "g000001"
    assert stats.saved_entry.parent_id == first.policy_id
    assert pool.latest() == stats.saved_entry
