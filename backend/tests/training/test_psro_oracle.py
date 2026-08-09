from pathlib import Path

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.encoding import ObservationEncoder
from app.training.env import WerewolfTrainingEnv
from app.training.meta_strategy import PolicyWeight, PopulationMetaStrategy
from app.training.numpy_policy import NumpyMLPPolicy
from app.training.numpy_trainer import PPOConfig
from app.training.policy_pool import NumpyPolicyPool
from app.training.psro_oracle import dominant_policy_id, train_population_oracle


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


def test_oracle_clones_dominant_policy_and_adds_team_specialist(tmp_path: Path):
    pool = NumpyPolicyPool(tmp_path / "pool")
    base = pool.add(_initialized_model(301))
    alternative = pool.add(_initialized_model(302))
    strategy = PopulationMetaStrategy(
        village=(
            PolicyWeight(base.policy_id, 0.8),
            PolicyWeight(alternative.policy_id, 0.2),
        ),
        werewolf=(PolicyWeight(base.policy_id, 1.0),),
        fox=(PolicyWeight(base.policy_id, 1.0),),
    )

    assert dominant_policy_id(strategy, Team.VILLAGE) == base.policy_id
    stats = train_population_oracle(
        _specs(),
        pool,
        strategy,
        team=Team.VILLAGE,
        episodes=1,
        start_seed=303,
        opponent_seed=304,
        ppo_config=PPOConfig(learning_rate=1e-3, epochs=1),
        max_discussion_ticks=1,
    )

    assert stats.parent_policy_id == base.policy_id
    assert stats.oracle_entry.parent_id == base.policy_id
    assert stats.oracle_entry.specialized_team is Team.VILLAGE
    assert stats.update.decisions > 0
    assert stats.wins + stats.losses + stats.draws == 1
    assert stats.oracle_entry.policy_id in pool.policy_ids_for_team(
        Team.VILLAGE,
        include_general=False,
    )
    assert stats.oracle_entry.policy_id not in pool.policy_ids_for_team(
        Team.WEREWOLF,
        include_general=False,
    )
