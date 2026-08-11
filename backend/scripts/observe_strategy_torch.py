"""Run post-hoc strategy observations from frozen Transformer policies."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.strategy_observatory import (
    StrategyObservatoryRunner,
    render_strategy_transcript,
)
from app.training.strategy_snapshot import extract_strategy_snapshot
from app.training.torch_pool import TorchPolicyPool


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


def _git_commit() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def _validate_policy(pool: TorchPolicyPool, team: Team, policy_id: str) -> None:
    entry = pool.get(policy_id)
    if entry.specialized_team is not None and entry.specialized_team is not team:
        raise ValueError(
            f"{policy_id} specializes in {entry.specialized_team.value}, not {team.value}"
        )


def _run(
    *,
    pool_dir: Path,
    snapshot: dict[str, Any] | None,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if snapshot is not None:
        expected_commit = snapshot.get("source", {}).get("git_commit")
        actual_commit = _git_commit()
        if (
            isinstance(expected_commit, str)
            and expected_commit
            and actual_commit != expected_commit
            and not args.allow_code_mismatch
        ):
            raise ValueError(
                "snapshot/code commit mismatch: "
                f"snapshot={expected_commit} checkout={actual_commit}; "
                "checkout the snapshot commit or pass --allow-code-mismatch"
            )

    pool = TorchPolicyPool(pool_dir, device=device)
    policy_ids = {
        Team.VILLAGE: args.village_policy,
        Team.WEREWOLF: args.werewolf_policy,
        Team.FOX: args.fox_policy,
    }
    for team, policy_id in policy_ids.items():
        _validate_policy(pool, team, policy_id)
    models = {
        team: pool.load(policy_id).eval()
        for team, policy_id in policy_ids.items()
    }

    runner = StrategyObservatoryRunner(
        _player_specs(),
        models,
        policy_ids,
        max_loops=args.max_loops,
        max_discussion_ticks=args.max_discussion_ticks,
        temperature=args.temperature,
    )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "games.jsonl"
    transcript_dir = output_dir / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    if jsonl_path.exists() and not args.force:
        raise FileExistsError(
            f"output already exists: {jsonl_path}; use --force to replace it"
        )

    winners = {"village": 0, "werewolf": 0, "fox": 0, "draw": 0}
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for offset in range(args.games):
            seed = args.seed_start + offset
            game = runner.run(seed)
            jsonl.write(json.dumps(game, ensure_ascii=False, separators=(",", ":")) + "\n")
            transcript_path = transcript_dir / f"game-{seed}.txt"
            transcript_path.write_text(
                render_strategy_transcript(game),
                encoding="utf-8",
            )
            if game["is_draw"]:
                winners["draw"] += 1
            else:
                winner = game["winner"]
                if winner not in winners:
                    raise RuntimeError(f"unexpected winner: {winner!r}")
                winners[winner] += 1
            print(
                f"observed={offset + 1}/{args.games} seed={seed} "
                f"winner={game['winner']} draw={str(game['is_draw']).lower()} "
                f"days={game['days']} semantic_events={len(game['semantic_events'])}"
            )

    print("===== STRATEGY OBSERVATORY COMPLETE =====")
    print(
        f"profile=village={args.village_policy} werewolf={args.werewolf_policy} "
        f"fox={args.fox_policy}"
    )
    print(
        f"games={args.games} village_wins={winners['village']} "
        f"werewolf_wins={winners['werewolf']} fox_wins={winners['fox']} "
        f"draws={winners['draw']}"
    )
    print(f"device={device}")
    print(f"jsonl={jsonl_path}")
    print(f"transcripts={transcript_dir}")
    print("training_state_mutated=false")
    print("===== END STRATEGY OBSERVATORY =====")


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", type=Path)
    source.add_argument("--pool-dir", type=Path)
    parser.add_argument("--village-policy", required=True)
    parser.add_argument("--werewolf-policy", required=True)
    parser.add_argument("--fox-policy", required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=120_000)
    parser.add_argument("--max-loops", type=int, default=200)
    parser.add_argument("--max-discussion-ticks", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-code-mismatch", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.games <= 0:
        parser.error("--games must be positive")
    if args.max_loops <= 0:
        parser.error("--max-loops must be positive")
    if args.max_discussion_ticks < 0:
        parser.error("--max-discussion-ticks cannot be negative")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    try:
        device = _resolve_device(args.device)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    try:
        if args.snapshot is not None:
            with tempfile.TemporaryDirectory(prefix="werewolf-strategy-observe-") as temporary:
                root = Path(temporary)
                snapshot = extract_strategy_snapshot(args.snapshot, root)
                _run(
                    pool_dir=root / "pool",
                    snapshot=snapshot,
                    args=args,
                    device=device,
                )
        else:
            if args.pool_dir is None:
                raise RuntimeError("pool source was not resolved")
            _run(pool_dir=args.pool_dir, snapshot=None, args=args, device=device)
    except (FileExistsError, KeyError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
