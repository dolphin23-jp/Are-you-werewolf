"""Audit dropped Transformer policies against a completed population iteration."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.meta_strategy import (
    PopulationMetaDiagnostics,
    PopulationMetaStrategy,
    diagnose_meta_strategy,
    solve_logit_response_mixture,
)
from app.training.population_payoff import PolicyProfile, PopulationPayoffTable
from app.training.retention_audit import (
    RetentionDiagnostics,
    build_retention_profiles,
    diagnose_retention,
)
from app.training.torch_pool import TorchPolicyPool
from app.training.torch_population import (
    TorchProfileEvaluationRequest,
    evaluate_torch_policy_profiles,
)
from app.training.torch_population_multiprocess import (
    evaluate_torch_policy_profiles_multiprocess,
)

_CONFIG_VERSION = 1
_REPORT_VERSION = 1


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


def _weights(strategy: PopulationMetaStrategy, team: Team) -> tuple[str, ...]:
    return tuple(item.policy_id for item in strategy.weights(team))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object in {path}")
    return raw


def _validate_challenger(
    pool: TorchPolicyPool,
    strategy: PopulationMetaStrategy,
    team: Team,
    policy_id: str,
) -> None:
    eligible = set(pool.policy_ids_for_team(team))
    if policy_id not in eligible:
        raise ValueError(f"{policy_id} is not eligible for {team.value}")
    if policy_id in _weights(strategy, team):
        raise ValueError(
            f"{team.value} challenger {policy_id} is already in iteration population"
        )
    entry = pool.get(policy_id)
    checkpoint = pool.root / entry.checkpoint
    if not checkpoint.is_file():
        raise ValueError(f"missing challenger checkpoint: {checkpoint}")


def _diagnostic_payload(diagnostics: RetentionDiagnostics) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for team in Team:
        item = diagnostics.for_team(team)
        raw = asdict(item)
        raw["team"] = team.value
        payload[team.value] = raw
    return payload


def _meta_diagnostic_payload(
    diagnostics: PopulationMetaDiagnostics,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for team in Team:
        item = diagnostics.for_team(team)
        raw = asdict(item)
        raw["team"] = team.value
        payload[team.value] = raw
    payload["max_deviation_gain"] = diagnostics.max_deviation_gain
    return payload


def _strategy_payload(strategy: PopulationMetaStrategy) -> dict[str, Any]:
    return {
        team.value: [asdict(item) for item in strategy.weights(team)]
        for team in Team
    }


def _render_text_report(payload: dict[str, Any]) -> str:
    lines = [
        "===== POPULATION RETENTION AUDIT =====",
        f"iteration={payload['iteration']}",
        f"profiles={payload['profile_count']} games_per_profile={payload['games_per_profile']} "
        f"total_games={payload['total_games']}",
    ]
    challengers = payload["challengers"]
    lines.append(
        "challengers="
        + " ".join(f"{team}={challengers[team]}" for team in ("village", "werewolf", "fox"))
    )
    for label in ("saved_meta", "fixed_meta"):
        lines.append("")
        lines.append(f"===== {label.upper()} CHALLENGER DIAGNOSTICS =====")
        diagnostics = payload["retention_diagnostics"][label]
        for team in ("village", "werewolf", "fox"):
            item = diagnostics[team]
            lines.append(
                f"{team}: mixture={item['mixture']['mean']:+.6f} "
                f"challenger={item['challenger_policy_id']} "
                f"challenger_payoff={item['challenger']['mean']:+.6f} "
                f"gain={item['challenger_gain_vs_mixture']:+.6f} "
                f"gain_ci95=[{item['challenger_gain_ci95_low']:+.6f},"
                f"{item['challenger_gain_ci95_high']:+.6f}] "
                f"best_current={item['best_current_policy_id']} "
                f"gain_vs_best={item['challenger_gain_vs_best_current']:+.6f}"
            )
    lines.append("")
    lines.append("===== FIXED-CUBE RESTRICTED DIAGNOSTICS =====")
    for label in ("saved_meta", "fixed_meta"):
        item = payload["restricted_diagnostics"][label]
        lines.append(
            f"{label}: max_deviation_gain={item['max_deviation_gain']:.6f}"
        )
    lines.append("===== END POPULATION RETENTION AUDIT =====")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--village-challenger", required=True)
    parser.add_argument("--werewolf-challenger", required=True)
    parser.add_argument("--fox-challenger", required=True)
    parser.add_argument("--games-per-profile", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17_001)
    parser.add_argument("--discussion-ticks", type=int)
    parser.add_argument("--parallel-games", type=int, default=16)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--request-batch-profiles", type=int, default=25)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="multiprocess rollout workers; zero keeps the single-process evaluator",
    )
    parser.add_argument("--inference-coalesce-ms", type=float, default=4.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.games_per_profile <= 1:
        parser.error("--games-per-profile must be greater than one")
    if args.parallel_games <= 0:
        parser.error("--parallel-games must be positive")
    if args.inference_batch_size <= 0:
        parser.error("--inference-batch-size must be positive")
    if args.request_batch_profiles <= 0:
        parser.error("--request-batch-profiles must be positive")
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    if args.inference_coalesce_ms < 0:
        parser.error("--inference-coalesce-ms cannot be negative")
    try:
        device = _resolve_device(args.device)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    state_path = args.run_dir / "population.run.json"
    if not state_path.is_file():
        parser.error(f"missing population run-state: {state_path}")
    state = _load_json(state_path)
    completed = int(state.get("completed_iterations", -1))
    phase = state.get("phase")
    if phase != "idle":
        parser.error(f"population run must be idle for a fixed audit; phase={phase!r}")
    iteration = completed if args.iteration is None else args.iteration
    if iteration <= 0 or iteration > completed:
        parser.error("--iteration must identify a completed population iteration")

    iteration_dir = args.run_dir / f"iteration-{iteration:04d}"
    meta_path = iteration_dir / "meta.json"
    summary_path = iteration_dir / "summary.json"
    if not meta_path.is_file() or not summary_path.is_file():
        parser.error(f"iteration {iteration} is missing meta.json or summary.json")
    saved_strategy = PopulationMetaStrategy.load(meta_path)

    config = state.get("config")
    if not isinstance(config, dict):
        parser.error("population run-state is missing config")
    discussion_ticks = (
        int(config["max_discussion_ticks"])
        if args.discussion_ticks is None
        else args.discussion_ticks
    )
    if discussion_ticks < 0:
        parser.error("--discussion-ticks cannot be negative")

    pool = TorchPolicyPool(args.pool_dir, device=device)
    challengers = {
        Team.VILLAGE: args.village_challenger,
        Team.WEREWOLF: args.werewolf_challenger,
        Team.FOX: args.fox_challenger,
    }
    try:
        for team in Team:
            _validate_challenger(pool, saved_strategy, team, challengers[team])
        profiles = build_retention_profiles(saved_strategy, challengers)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))

    output_dir = args.output_dir or (
        args.run_dir / f"retention-audit-iteration-{iteration:04d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "payoffs.json"
    audit_config_path = output_dir / "audit.config.json"
    audit_config = {
        "version": _CONFIG_VERSION,
        "iteration": iteration,
        "source_summary": str(summary_path),
        "current_population": {
            team.value: list(_weights(saved_strategy, team)) for team in Team
        },
        "challengers": {team.value: challengers[team] for team in Team},
        "games_per_profile": args.games_per_profile,
        "seed": args.seed,
        "discussion_ticks": discussion_ticks,
        "meta_temperature": float(config["meta_temperature"]),
        "meta_iterations": int(config["meta_iterations"]),
        "meta_damping": float(config["meta_damping"]),
    }
    if args.workers > 0:
        audit_config["evaluation_backend"] = "multiprocess"
        audit_config["workers"] = args.workers
        audit_config["inference_coalesce_ms"] = args.inference_coalesce_ms
    if audit_config_path.exists():
        if _load_json(audit_config_path) != audit_config:
            parser.error(
                "existing audit.config.json differs from requested controlled audit"
            )
    else:
        _atomic_json(audit_config_path, audit_config)

    table = PopulationPayoffTable(table_path)
    for profile in profiles:
        existing = table.get(profile)
        if existing is not None and existing.games > args.games_per_profile:
            parser.error(
                f"profile {profile} already has {existing.games} games; "
                f"controlled target is {args.games_per_profile}"
            )

    requests: list[TorchProfileEvaluationRequest] = []
    for profile in profiles:
        existing = table.get(profile)
        games_before = existing.games if existing is not None else 0
        missing = args.games_per_profile - games_before
        if missing <= 0:
            continue
        seed_start = _next_seed(args.seed, profile, games_before)
        requests.append(
            TorchProfileEvaluationRequest(
                profile,
                tuple(range(seed_start, seed_start + missing)),
            )
        )

    backend = "multiprocess" if args.workers > 0 else "single-process"
    print(
        f"evaluation_backend={backend} workers={args.workers} "
        f"coalesce_ms={args.inference_coalesce_ms:.3f}"
    )
    for start in range(0, len(requests), args.request_batch_profiles):
        batch = tuple(requests[start : start + args.request_batch_profiles])
        if args.workers > 0:
            stats = evaluate_torch_policy_profiles_multiprocess(
                _player_specs(),
                pool,
                table,
                batch,
                worker_count=args.workers,
                max_discussion_ticks=discussion_ticks,
                max_parallel_games=args.parallel_games,
                max_inference_batch_size=args.inference_batch_size,
                inference_coalesce_seconds=args.inference_coalesce_ms / 1000.0,
            )
        else:
            stats = evaluate_torch_policy_profiles(
                _player_specs(),
                pool,
                table,
                batch,
                max_discussion_ticks=discussion_ticks,
                max_parallel_games=args.parallel_games,
                max_inference_batch_size=args.inference_batch_size,
            )
        complete = sum(
            int(
                table.get(profile) is not None
                and table.get(profile).games == args.games_per_profile  # type: ignore[union-attr]
            )
            for profile in profiles
        )
        print(
            f"evaluation_progress={complete}/{len(profiles)} "
            f"new_games={stats.games} games_s={stats.games_per_second:.2f}"
        )

    records = []
    for profile in profiles:
        record = table.get(profile)
        if record is None or record.games != args.games_per_profile:
            parser.error(f"incomplete fixed-evaluation profile: {profile}")
        records.append(record)

    current = {team: _weights(saved_strategy, team) for team in Team}
    fixed_strategy = solve_logit_response_mixture(
        table,
        village=current[Team.VILLAGE],
        werewolf=current[Team.WEREWOLF],
        fox=current[Team.FOX],
        temperature=float(config["meta_temperature"]),
        iterations=int(config["meta_iterations"]),
        damping=float(config["meta_damping"]),
    )
    fixed_strategy.save(output_dir / "fixed-meta.json")

    saved_retention = diagnose_retention(table, saved_strategy, challengers)
    fixed_retention = diagnose_retention(table, fixed_strategy, challengers)
    saved_restricted = diagnose_meta_strategy(table, saved_strategy)
    fixed_restricted = diagnose_meta_strategy(table, fixed_strategy)

    payload = {
        "version": _REPORT_VERSION,
        "iteration": iteration,
        "source_summary": str(summary_path),
        "profile_count": len(profiles),
        "games_per_profile": args.games_per_profile,
        "total_games": sum(record.games for record in records),
        "seed": args.seed,
        "challengers": {team.value: challengers[team] for team in Team},
        "saved_meta_strategy": _strategy_payload(saved_strategy),
        "fixed_meta_strategy": _strategy_payload(fixed_strategy),
        "retention_diagnostics": {
            "saved_meta": _diagnostic_payload(saved_retention),
            "fixed_meta": _diagnostic_payload(fixed_retention),
        },
        "restricted_diagnostics": {
            "saved_meta": _meta_diagnostic_payload(saved_restricted),
            "fixed_meta": _meta_diagnostic_payload(fixed_restricted),
        },
    }
    report_json = output_dir / "report.json"
    report_text = output_dir / "report.txt"
    _atomic_json(report_json, payload)
    report_text.write_text(_render_text_report(payload), encoding="utf-8")
    print(report_text.read_text(encoding="utf-8"), end="")
    print(f"report_json={report_json}")
    print(f"report_text={report_text}")
    print(f"payoff_table={table_path} device={device}")


if __name__ == "__main__":
    main()
