"""Run PSRO-style empirical population iterations for the NumPy baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.numpy_trainer import PPOConfig
from app.training.policy_pool import NumpyPolicyPool
from app.training.population_iteration import PopulationIterationStats, run_population_iteration
from app.training.population_payoff import PopulationPayoffTable


def _player_specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=False)
        for i in range(17)
    ]


def _format_mix(stats: PopulationIterationStats) -> str:
    chunks = []
    for team in Team:
        weights = ",".join(
            f"{item.policy_id}:{item.probability:.3f}"
            for item in stats.meta_strategy.weights(team)
        )
        chunks.append(f"{team.value}=[{weights}]")
    return " ".join(chunks)


def _print_stats(iteration: int, stats: PopulationIterationStats) -> None:
    print(
        f"iteration={iteration} population village={','.join(stats.village_policy_ids)} "
        f"werewolf={','.join(stats.werewolf_policy_ids)} "
        f"fox={','.join(stats.fox_policy_ids)}"
    )
    print(
        f"iteration={iteration} profiles={stats.measured_profiles} "
        f"new_games={stats.new_profile_games} adaptive_games={stats.adaptive_games}"
    )
    print(f"iteration={iteration} {_format_mix(stats)}")
    for oracle in stats.oracles:
        print(
            f"iteration={iteration} oracle team={oracle.team.value} "
            f"parent={oracle.parent_policy_id} saved={oracle.oracle_entry.policy_id} "
            f"record={oracle.wins}-{oracle.losses}-{oracle.draws} "
            f"decisions={oracle.update.decisions} grad_norm={oracle.update.gradient_norm:.4f}"
        )
    print(
        f"iteration={iteration} "
        f"restricted_max_deviation_gain={stats.diagnostics.max_deviation_gain:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--payoff-table", type=Path, required=True)
    parser.add_argument("--meta-output", type=Path)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--recent-policies", type=int, default=3)
    parser.add_argument("--games-per-profile", type=int, default=3)
    parser.add_argument("--extra-games", type=int, default=0)
    parser.add_argument("--uncertainty-prior", type=float, default=0.5)
    parser.add_argument("--oracle-episodes", type=int, default=2)
    parser.add_argument("--evaluation-seed", type=int, default=30000)
    parser.add_argument("--oracle-seed", type=int, default=40000)
    parser.add_argument("--opponent-seed", type=int, default=1)
    parser.add_argument("--discussion-ticks", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--meta-temperature", type=float, default=0.25)
    parser.add_argument("--meta-iterations", type=int, default=100)
    parser.add_argument("--meta-damping", type=float, default=0.5)
    args = parser.parse_args()

    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    pool = NumpyPolicyPool(args.pool_dir)
    table = PopulationPayoffTable(args.payoff_table)
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        epochs=args.ppo_epochs,
    )
    for index in range(args.iterations):
        stats = run_population_iteration(
            _player_specs(),
            pool,
            table,
            recent_policies=args.recent_policies,
            games_per_profile=args.games_per_profile,
            extra_games=args.extra_games,
            uncertainty_prior=args.uncertainty_prior,
            evaluation_seed=args.evaluation_seed + index * 2_000_000_000,
            oracle_seed=args.oracle_seed + index * 1_000_000,
            opponent_seed=args.opponent_seed + index * 100,
            oracle_episodes=args.oracle_episodes,
            max_discussion_ticks=args.discussion_ticks,
            meta_temperature=args.meta_temperature,
            meta_iterations=args.meta_iterations,
            meta_damping=args.meta_damping,
            ppo_config=ppo_config,
        )
        if args.meta_output is not None:
            stats.meta_strategy.save(args.meta_output)
        _print_stats(index + 1, stats)


if __name__ == "__main__":
    main()
