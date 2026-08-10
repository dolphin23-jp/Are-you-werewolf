"""Measure empirical three-faction payoffs for Transformer generations."""

from __future__ import annotations

import argparse
import hashlib
import itertools
from dataclasses import dataclass
from pathlib import Path

import torch

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.population_payoff import PolicyProfile, PopulationPayoffTable, ProfilePayoff
from app.training.population_shards import select_profile_shard
from app.training.torch_pool import TorchPolicyPool
from app.training.torch_population import (
    TorchPopulationEvaluationStats,
    TorchProfileEvaluationRequest,
    evaluate_torch_policy_profiles,
)


@dataclass
class _EvaluationTotals:
    games: int = 0
    rollout_chunks: int = 0
    rollout_seconds: float = 0.0
    checkpoint_loads: int = 0
    inference_calls: int = 0
    inference_observations: int = 0
    max_pending_inference_requests: int = 0
    max_inference_batch: int = 0

    def add(self, stats: TorchPopulationEvaluationStats) -> None:
        self.games += stats.games
        self.rollout_chunks += stats.rollout_chunks
        self.rollout_seconds += stats.rollout_seconds
        self.checkpoint_loads += stats.checkpoint_loads
        self.inference_calls += stats.inference_calls
        self.inference_observations += stats.inference_observations
        self.max_pending_inference_requests = max(
            self.max_pending_inference_requests,
            stats.max_pending_inference_requests,
        )
        self.max_inference_batch = max(
            self.max_inference_batch,
            stats.max_inference_batch,
        )

    @property
    def games_per_second(self) -> float:
        if self.rollout_seconds <= 0:
            return 0.0
        return self.games / self.rollout_seconds

    @property
    def mean_inference_batch(self) -> float:
        if self.inference_calls == 0:
            return 0.0
        return self.inference_observations / self.inference_calls


def _player_specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=False)
        for i in range(17)
    ]


def _resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def _profile_seed_offset(profile: PolicyProfile) -> int:
    encoded = f"{profile.village}|{profile.werewolf}|{profile.fox}".encode()
    digest = hashlib.sha256(encoded).digest()
    return int.from_bytes(digest[:4], "big")


def _next_seed(base_seed: int, profile: PolicyProfile, games_before: int) -> int:
    return base_seed + _profile_seed_offset(profile) + games_before


def _require_record(
    table: PopulationPayoffTable,
    profile: PolicyProfile,
) -> ProfilePayoff:
    record = table.get(profile)
    if record is None:
        raise RuntimeError("profile measurement did not produce a record")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--last", type=int, default=3)
    parser.add_argument("--games-per-profile", type=int, default=3)
    parser.add_argument("--extra-games", type=int, default=0)
    parser.add_argument("--uncertainty-prior", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--discussion-ticks", type=int, default=8)
    parser.add_argument("--parallel-games", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.last <= 0:
        parser.error("--last must be positive")
    if args.games_per_profile <= 0:
        parser.error("--games-per-profile must be positive")
    if args.extra_games < 0:
        parser.error("--extra-games cannot be negative")
    if args.uncertainty_prior <= 0:
        parser.error("--uncertainty-prior must be positive")
    if args.parallel_games <= 0:
        parser.error("--parallel-games must be positive")
    if args.inference_batch_size is not None and args.inference_batch_size <= 0:
        parser.error("--inference-batch-size must be positive")
    if args.shard_count <= 0:
        parser.error("--shard-count must be positive")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        parser.error("--shard-index must satisfy 0 <= index < shard-count")
    if args.shard_count > 1 and args.extra_games:
        parser.error("adaptive --extra-games is not supported with sharded evaluation")
    try:
        device = _resolve_device(args.device)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    pool = TorchPolicyPool(args.pool_dir, device=device)
    if not pool.entries:
        parser.error("--pool-dir contains no policy generations")
    village_ids = pool.policy_ids_for_team(Team.VILLAGE, last=args.last)
    werewolf_ids = pool.policy_ids_for_team(Team.WEREWOLF, last=args.last)
    fox_ids = pool.policy_ids_for_team(Team.FOX, last=args.last)
    if not village_ids or not werewolf_ids or not fox_ids:
        parser.error("each faction must have at least one eligible policy")

    table = PopulationPayoffTable(args.table)
    all_profiles = tuple(
        PolicyProfile(village, werewolf, fox)
        for village, werewolf, fox in itertools.product(
            village_ids,
            werewolf_ids,
            fox_ids,
        )
    )
    profiles = select_profile_shard(
        all_profiles,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    if not profiles:
        parser.error("selected shard contains no profiles")
    totals = _EvaluationTotals()

    missing_requests: list[TorchProfileEvaluationRequest] = []
    for profile in profiles:
        existing = table.get(profile)
        existing_games = existing.games if existing is not None else 0
        missing = max(0, args.games_per_profile - existing_games)
        if not missing:
            continue
        seed_start = _next_seed(args.seed, profile, existing_games)
        missing_requests.append(
            TorchProfileEvaluationRequest(
                profile,
                tuple(range(seed_start, seed_start + missing)),
            )
        )

    if missing_requests:
        totals.add(
            evaluate_torch_policy_profiles(
                _player_specs(),
                pool,
                table,
                tuple(missing_requests),
                max_discussion_ticks=args.discussion_ticks,
                max_parallel_games=args.parallel_games,
                max_inference_batch_size=args.inference_batch_size,
            )
        )

    # Keep the existing adaptive semantics for uncertainty allocation: every
    # extra result is committed before choosing the next highest-uncertainty
    # profile. Only the initial payoff cube is co-scheduled across profiles.
    for _ in range(args.extra_games):
        selected = max(
            profiles,
            key=lambda profile: (
                _require_record(table, profile).max_posterior_payoff_std(
                    prior=args.uncertainty_prior
                ),
                -_require_record(table, profile).games,
                profile,
            ),
        )
        before = _require_record(table, selected)
        totals.add(
            evaluate_torch_policy_profiles(
                _player_specs(),
                pool,
                table,
                (
                    TorchProfileEvaluationRequest(
                        selected,
                        (_next_seed(args.seed, selected, before.games),),
                    ),
                ),
                max_discussion_ticks=args.discussion_ticks,
                max_parallel_games=args.parallel_games,
                max_inference_batch_size=args.inference_batch_size,
            )
        )

    for profile in profiles:
        record = _require_record(table, profile)
        print(
            f"profile={profile.village}/{profile.werewolf}/{profile.fox} "
            f"games={record.games} wins(v/w/f)="
            f"{record.village_wins}/{record.werewolf_wins}/{record.fox_wins} "
            f"draws={record.draws} payoffs(v/w/f)="
            f"{record.mean_payoff(Team.VILLAGE):.3f}/"
            f"{record.mean_payoff(Team.WEREWOLF):.3f}/"
            f"{record.mean_payoff(Team.FOX):.3f} "
            f"uncertainty={record.max_posterior_payoff_std(prior=args.uncertainty_prior):.4f}"
        )
    inference_limit = args.inference_batch_size or "unbounded"
    print(
        f"measured_profiles={len(profiles)}/{len(all_profiles)} new_games={totals.games} "
        f"extra_games={args.extra_games} parallel_games={args.parallel_games} "
        f"shard={args.shard_index}/{args.shard_count} "
        f"rollout_chunks={totals.rollout_chunks} "
        f"evaluation_s={totals.rollout_seconds:.3f} "
        f"games_s={totals.games_per_second:.2f} "
        f"checkpoint_loads={totals.checkpoint_loads} "
        f"inference_limit={inference_limit} "
        f"inference_mean_batch={totals.mean_inference_batch:.1f} "
        f"inference_max_batch={totals.max_inference_batch} "
        f"inference_max_pending={totals.max_pending_inference_requests} "
        f"village={','.join(village_ids)} werewolf={','.join(werewolf_ids)} "
        f"fox={','.join(fox_ids)} table={args.table} device={device}"
    )


if __name__ == "__main__":
    main()
