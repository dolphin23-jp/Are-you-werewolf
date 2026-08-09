"""Run one empirical population-training iteration for the NumPy baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.numpy_trainer import PPOConfig
from app.training.policy_pool import NumpyPolicyPool
from app.training.population_iteration import run_population_iteration
from app.training.population_payoff import PopulationPayoffTable


def _player_specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=False)
        for i in range(17)
    ]


def _format_mix(stats) -> str:
    chunks = []
    for team in Team:
        weights = ",".join(
            f"{item.policy_id}:{item.probability:.3f}"
            for item in stats.meta_strategy.weights(team)
        )
        chunks.append(f"{team.value}=[{weights}]")
    return " ".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--payoff-table", type=Path, required=True)
    parser.add_argument("--recent-policies", type=int, default=3)
    parser.add_argument("--games-per-profile", type=int, default=3)
    parser.add_argument("--episodes-per-team", type=int, default=2)
    parser.add_argument("--evaluation-seed", type=int, default=30000)
    parser.add_argument("--training-seed", type=int, default=40000)
    parser.add_argument("--opponent-seed", type=int, default=1)
    parser.add_argument("--discussion-ticks", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--meta-temperature", type=float, default=0.25)
    parser.add_argument("--meta-iterations", type=int, default=100)
    parser.add_argument("--meta-damping", type=float, default=0.5)
    args = parser.parse_args()

    pool = NumpyPolicyPool(args.pool_dir)
    table = PopulationPayoffTable(args.payoff_table)
    stats = run_population_iteration(
        _player_specs(),
        pool,
        table,
        recent_policies=args.recent_policies,
        games_per_profile=args.games_per_profile,
        evaluation_seed=args.evaluation_seed,
        training_seed=args.training_seed,
        opponent_seed=args.opponent_seed,
        episodes_per_team=args.episodes_per_team,
        max_discussion_ticks=args.discussion_ticks,
        meta_temperature=args.meta_temperature,
        meta_iterations=args.meta_iterations,
        meta_damping=args.meta_damping,
        ppo_config=PPOConfig(
            learning_rate=args.learning_rate,
            epochs=args.ppo_epochs,
        ),
    )

    print(
        f"measured_policies={','.join(stats.measured_policy_ids)} "
        f"profiles={stats.measured_profiles} new_games={stats.new_profile_games}"
    )
    print(_format_mix(stats))
    for item in stats.training:
        print(
            f"team={item.learner_team.value} record={item.wins}-{item.losses}-{item.draws} "
            f"decisions={item.update.decisions} grad_norm={item.update.gradient_norm:.4f}"
        )
    print(
        f"restricted_max_deviation_gain={stats.diagnostics.max_deviation_gain:.4f} "
        f"saved={stats.saved_entry.policy_id} parent={stats.saved_entry.parent_id}"
    )


if __name__ == "__main__":
    main()
