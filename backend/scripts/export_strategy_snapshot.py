"""Export a portable read-only strategy-analysis snapshot."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from app.training.strategy_snapshot import export_strategy_snapshot


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--label")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        snapshot = export_strategy_snapshot(
            pool_dir=args.pool_dir,
            run_dir=args.run_dir,
            output_path=args.output,
            iteration=args.iteration,
            source_label=args.label,
            git_commit=_git_commit(),
            overwrite=args.force,
        )
    except (FileExistsError, OSError, ValueError) as exc:
        parser.error(str(exc))

    population = snapshot["population"]
    source = snapshot["source"]
    print("===== STRATEGY SNAPSHOT =====")
    print(
        f"mode={source['mode']} iteration={source['iteration']} "
        f"phase={source['run_phase_at_export']} git_commit={source['git_commit']}"
    )
    for team in ("village", "werewolf", "fox"):
        print(f"{team}=" + ",".join(population[team]))
    print(f"output={args.output}")
    print("source_run_state_mutated=false")
    print("snapshot_json=" + json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")))
    print("===== END STRATEGY SNAPSHOT =====")


if __name__ == "__main__":
    main()
