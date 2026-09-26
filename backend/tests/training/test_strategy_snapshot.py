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


def test_snapshot_refuses_checkpoint_paths_outside_the_pool(tmp_path: Path):
    """A manifest entry like `../outside.npz` made the exporter copy a file
    from outside the pool and write it outside its staging directory."""
    import pytest

    pool_dir = tmp_path / "pool"
    run_dir = tmp_path / "population"
    (tmp_path / "outside.npz").write_bytes(b"not a pool checkpoint")
    entries = [
        {
            "policy_id": "g000001",
            "generation": 1,
            "checkpoint": "../outside.npz",
            "parent_id": None,
            "specialized_team": None,
        }
    ]
    _write_json(pool_dir / "manifest.json", {"version": 1, "entries": entries})
    _write_json(
        run_dir / "population.run.json",
        {
            "completed_iterations": 1,
            "phase": "measure",
            "village_policy_ids": ["g000001"],
            "werewolf_policy_ids": ["g000001"],
            "fox_policy_ids": ["g000001"],
        },
    )

    with pytest.raises(ValueError, match="invalid checkpoint path"):
        export_strategy_snapshot(
            pool_dir=pool_dir,
            run_dir=run_dir,
            output_path=tmp_path / "strategy.tar.gz",
            source_label="test",
            git_commit="abc123",
        )


def test_pool_manifest_with_escaping_checkpoint_is_rejected(tmp_path: Path):
    import pytest

    from app.training.policy_pool import NumpyPolicyPool

    _write_json(
        tmp_path / "pool" / "manifest.json",
        {
            "version": 1,
            "entries": [
                {
                    "policy_id": "g000000",
                    "generation": 0,
                    "checkpoint": "/etc/passwd",
                    "parent_id": None,
                    "specialized_team": None,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="inside the pool"):
        NumpyPolicyPool(tmp_path / "pool")


def _exported_snapshot(tmp_path: Path, output_name: str = "strategy.tar.gz") -> Path:
    pool_dir = tmp_path / "pool"
    entries = []
    for generation in range(1, 4):
        policy_id = f"g{generation:06d}"
        entries.append(
            {
                "policy_id": policy_id,
                "generation": generation,
                "checkpoint": f"{policy_id}.npz",
                "parent_id": None,
                "specialized_team": None,
            }
        )
        pool_dir.mkdir(parents=True, exist_ok=True)
        (pool_dir / f"{policy_id}.npz").write_bytes(policy_id.encode())
    _write_json(pool_dir / "manifest.json", {"version": 1, "entries": entries})
    _write_json(
        tmp_path / "population" / "population.run.json",
        {
            "completed_iterations": 1,
            "phase": "measure",
            "village_policy_ids": ["g000001"],
            "werewolf_policy_ids": ["g000002"],
            "fox_policy_ids": ["g000003"],
        },
    )
    output = tmp_path / output_name
    export_strategy_snapshot(
        pool_dir=pool_dir,
        run_dir=tmp_path / "population",
        output_path=output,
        source_label="test",
        git_commit="abc123",
    )
    return output


def test_tgz_snapshot_is_compressed(tmp_path: Path):
    archive = _exported_snapshot(tmp_path, "strategy.tgz")

    assert archive.read_bytes()[:2] == b"\x1f\x8b"


def test_extraction_refuses_a_non_empty_destination(tmp_path: Path):
    import pytest

    archive = _exported_snapshot(tmp_path)
    destination = tmp_path / "existing"
    (destination / "pool").mkdir(parents=True)
    (destination / "pool" / "manifest.json").write_text("real pool", encoding="utf-8")

    with pytest.raises(ValueError, match="new or empty"):
        extract_strategy_snapshot(archive, destination)
    assert (destination / "pool" / "manifest.json").read_text(encoding="utf-8") == "real pool"


def test_tampered_snapshot_leaves_nothing_behind(tmp_path: Path):
    import io
    import tarfile

    import pytest

    archive = _exported_snapshot(tmp_path)
    tampered = tmp_path / "tampered.tar"
    with tarfile.open(archive, "r:*") as source, tarfile.open(tampered, "w") as target:
        for member in source.getmembers():
            data = source.extractfile(member) if member.isfile() else None
            if member.name == "pool/g000001.npz":
                payload = b"swapped weights"
                member.size = len(payload)
                target.addfile(member, io.BytesIO(payload))
            else:
                target.addfile(member, data)
    before = sorted(item.name for item in tmp_path.iterdir())

    with pytest.raises(ValueError, match="hash mismatch"):
        extract_strategy_snapshot(tampered, tmp_path / "extracted")
    assert sorted(item.name for item in tmp_path.iterdir()) == before
