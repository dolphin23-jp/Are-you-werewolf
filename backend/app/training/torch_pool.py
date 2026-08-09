"""Persistent immutable population for Transformer policy checkpoints."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from app.engine.roles import Team
from app.training.policy_pool import PolicyPoolEntry
from app.training.torch_checkpoint import load_torch_policy, save_torch_policy
from app.training.torch_policy import TorchTransformerPolicy

_MANIFEST_VERSION = 1


class TorchPolicyPool:
    """Store safe Transformer NPZ generations behind a JSON manifest."""

    def __init__(
        self,
        root: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        self.device = torch.device(device)
        self._entries = self._read_manifest()

    @property
    def entries(self) -> tuple[PolicyPoolEntry, ...]:
        return tuple(self._entries)

    @property
    def next_generation(self) -> int:
        if not self._entries:
            return 0
        return max(entry.generation for entry in self._entries) + 1

    def latest(self) -> PolicyPoolEntry | None:
        if not self._entries:
            return None
        return max(self._entries, key=lambda entry: entry.generation)

    def entries_for_team(
        self,
        team: Team,
        *,
        include_general: bool = True,
    ) -> tuple[PolicyPoolEntry, ...]:
        return tuple(
            entry
            for entry in self._entries
            if entry.specialized_team is team
            or (include_general and entry.specialized_team is None)
        )

    def policy_ids_for_team(
        self,
        team: Team,
        *,
        last: int | None = None,
        include_general: bool = True,
    ) -> tuple[str, ...]:
        entries = self.entries_for_team(team, include_general=include_general)
        if last is not None:
            if last <= 0:
                raise ValueError("last must be positive")
            entries = entries[-last:]
        return tuple(entry.policy_id for entry in entries)

    def add(
        self,
        model: TorchTransformerPolicy,
        *,
        generation: int | None = None,
        parent_id: str | None = None,
        specialized_team: Team | None = None,
    ) -> PolicyPoolEntry:
        generation = self.next_generation if generation is None else generation
        if generation < 0:
            raise ValueError("generation cannot be negative")
        policy_id = f"g{generation:06d}"
        if any(entry.policy_id == policy_id for entry in self._entries):
            raise ValueError(f"policy generation {generation} already exists")
        if parent_id is not None and not any(
            entry.policy_id == parent_id for entry in self._entries
        ):
            raise ValueError(f"unknown parent policy {parent_id}")

        self.root.mkdir(parents=True, exist_ok=True)
        checkpoint = f"{policy_id}.npz"
        save_torch_policy(model, self.root / checkpoint)
        entry = PolicyPoolEntry(
            policy_id=policy_id,
            generation=generation,
            checkpoint=checkpoint,
            parent_id=parent_id,
            specialized_team=specialized_team,
        )
        self._entries.append(entry)
        self._write_manifest()
        return entry

    def ensure_generation(
        self,
        model: TorchTransformerPolicy,
        *,
        generation: int,
        parent_id: str | None,
        specialized_team: Team | None,
    ) -> PolicyPoolEntry:
        """Create an expected generation or safely reuse its exact crash replay."""

        if generation < 0:
            raise ValueError("generation cannot be negative")
        policy_id = f"g{generation:06d}"
        try:
            existing = self.get(policy_id)
        except KeyError as exc:
            if self.next_generation != generation:
                raise ValueError(
                    "policy pool advanced beyond the expected generation boundary"
                ) from exc
            return self.add(
                model,
                generation=generation,
                parent_id=parent_id,
                specialized_team=specialized_team,
            )

        if existing.parent_id != parent_id:
            raise ValueError(
                f"existing {policy_id} has unexpected parent {existing.parent_id}"
            )
        if existing.specialized_team is not specialized_team:
            raise ValueError(
                f"existing {policy_id} has unexpected specialized faction"
            )
        persisted = self.load(policy_id)
        if not _same_model_state(model, persisted):
            raise ValueError(
                f"existing {policy_id} does not match replayed model tensors"
            )
        return existing

    def get(self, policy_id: str) -> PolicyPoolEntry:
        for entry in self._entries:
            if entry.policy_id == policy_id:
                return entry
        raise KeyError(policy_id)

    def load(self, policy_id: str) -> TorchTransformerPolicy:
        entry = self.get(policy_id)
        return load_torch_policy(
            self.root / entry.checkpoint,
            device=self.device,
        )

    def sample(self, rng: random.Random) -> PolicyPoolEntry:
        if not self._entries:
            raise ValueError("cannot sample an empty policy pool")
        return rng.choice(self._entries)

    def _read_manifest(self) -> list[PolicyPoolEntry]:
        if not self.manifest_path.exists():
            return []
        raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("version") != _MANIFEST_VERSION:
            raise ValueError("unsupported Transformer policy-pool manifest")
        entries = raw.get("entries")
        if not isinstance(entries, list):
            raise ValueError("invalid Transformer policy-pool entries")
        parsed = [_entry_from_json(item) for item in entries]
        return sorted(parsed, key=lambda entry: entry.generation)

    def _write_manifest(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _MANIFEST_VERSION,
            "entries": [asdict(entry) for entry in self._entries],
        }
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.manifest_path)


def _same_model_state(
    left: TorchTransformerPolicy,
    right: TorchTransformerPolicy,
) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    if left_state.keys() != right_state.keys():
        return False
    return all(torch.equal(left_state[name], right_state[name]) for name in left_state)


def _entry_from_json(item: Any) -> PolicyPoolEntry:
    if not isinstance(item, dict):
        raise ValueError("invalid Transformer policy-pool entry")
    policy_id = item.get("policy_id")
    generation = item.get("generation")
    checkpoint = item.get("checkpoint")
    parent_id = item.get("parent_id")
    raw_team = item.get("specialized_team")
    if not isinstance(policy_id, str):
        raise ValueError("policy_id must be a string")
    if not isinstance(generation, int):
        raise ValueError("generation must be an integer")
    if not isinstance(checkpoint, str):
        raise ValueError("checkpoint must be a string")
    if parent_id is not None and not isinstance(parent_id, str):
        raise ValueError("parent_id must be a string or null")
    if raw_team is None:
        specialized_team = None
    elif isinstance(raw_team, str):
        try:
            specialized_team = Team(raw_team)
        except ValueError as exc:
            raise ValueError("invalid specialized_team") from exc
    else:
        raise ValueError("specialized_team must be a string or null")
    return PolicyPoolEntry(
        policy_id=policy_id,
        generation=generation,
        checkpoint=checkpoint,
        parent_id=parent_id,
        specialized_team=specialized_team,
    )
