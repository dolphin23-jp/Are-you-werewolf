import json
from pathlib import Path

from app.training.strategy_snapshot import (
    export_strategy_snapshot,
    extract_strategy_snapshot,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_active_population_snapshot_is_subsetted_verified_and_read_only(tmp_path: Path):
    pool_dir = tmp_path / "pool"
    run_dir = tmp_path / "population"
    policy_ids = {
        "village": ["g000001", "g000004"],
        "werewolf": ["g000002", "g000005"],
        "fox": ["g000003", "g000006"],
    }
    entries = []
    for generation in range(1, 8):
        policy_id = f"g{generation:06d}"
        checkpoint = f"{policy_id}.npz"
        entries.append(
            {
                "policy_id": policy_id,
                "generation": generation,
                "checkpoint": checkpoint,
                "parent_id": None,
                "specialized_team": None,
            }
        )
        pool_dir.mkdir(parents=True, exist_ok=True)
        (pool_dir / checkpoint).write_bytes(f"checkpoint-{policy_id}".encode())
    _write_json(pool_dir / "manifest.json", {"version": 1, "entries": entries})
    state = {
        "completed_iterations": 11,
        "phase": "measure",
        "village_policy_ids": policy_ids["village"],
        "werewolf_policy_ids": policy_ids["werewolf"],
        "fox_policy_ids": policy_ids["fox"],
    }
    state_path = run_dir / "population.run.json"
    _write_json(state_path, state)
    state_before = state_path.read_bytes()

    archive = tmp_path / "strategy.tar.gz"
    snapshot = export_strategy_snapshot(
        pool_dir=pool_dir,
        run_dir=run_dir,
        output_path=archive,
        source_label="test-active",
        git_commit="abc123",
    )

    assert state_path.read_bytes() == state_before
    assert snapshot["source"]["mode"] == "active_frozen_population"
    assert snapshot["source"]["iteration"] == 12
    assert snapshot["source"]["git_commit"] == "abc123"
    assert snapshot["population"] == policy_ids
    assert set(snapshot["checkpoint_sha256"]) == {
        policy_id for team_ids in policy_ids.values() for policy_id in team_ids
    }

    extracted = tmp_path / "extracted"
    loaded = extract_strategy_snapshot(archive, extracted)
    assert loaded == snapshot
    subset_manifest = json.loads(
        (extracted / "pool" / "manifest.json").read_text(encoding="utf-8")
    )
    selected_ids = {entry["policy_id"] for entry in subset_manifest["entries"]}
    assert selected_ids == set(snapshot["checkpoint_sha256"])
    assert not (extracted / "pool" / "g000007.npz").exists()
