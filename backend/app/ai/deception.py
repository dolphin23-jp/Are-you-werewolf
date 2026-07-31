"""Deception engineering for the werewolf team.

Rather than letting the LLM freely improvise whether to lie turn-by-turn,
the wolf team (and independently the Madman) pre-commit once per game to a
named deception pattern via weighted random choice. `FakeClaimGuard` then
enforces consistency on fabricated claims so a faker can't self-contradict
across turns (this is the mechanism that keeps deception *coherent*
without scripting the actual bluff line-by-line).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from app.engine.roles import RoleName

WOLF_PATTERNS: list[tuple[str, str, float]] = [
    ("alpha", "偽占い1+潜伏2", 0.40),
    ("beta", "偽霊媒1+潜伏2", 0.20),
    ("gamma", "全潜伏", 0.25),
    ("delta", "偽占い1+偽霊媒1+潜伏1", 0.15),
]

MADMAN_STRATEGIES: list[tuple[str, RoleName | None, float]] = [
    ("fake_seer", RoleName.SEER, 0.4),
    ("fake_medium", RoleName.MEDIUM, 0.3),
    ("lurk", None, 0.3),
]


@dataclass(frozen=True)
class WolfDeceptionAssignment:
    pattern_name: str
    pattern_label: str
    fake_role_by_player: dict[str, RoleName]
    lurking_player_ids: list[str]


def _weighted_choice(options: list[tuple[Any, ...]], rng: random.Random) -> tuple[Any, ...]:
    weights = [w for *_rest, w in options]
    return rng.choices(options, weights=weights, k=1)[0]


def assign_wolf_deception(wolf_ids: list[str], seed: int | None = None) -> WolfDeceptionAssignment:
    rng = random.Random(seed)
    pattern_name, label, _weight = _weighted_choice(WOLF_PATTERNS, rng)
    ids = list(wolf_ids)
    rng.shuffle(ids)

    fake_role_by_player: dict[str, RoleName] = {}
    if pattern_name == "alpha" and ids:
        fake_role_by_player[ids[0]] = RoleName.SEER
    elif pattern_name == "beta" and ids:
        fake_role_by_player[ids[0]] = RoleName.MEDIUM
    elif pattern_name == "delta":
        if len(ids) > 0:
            fake_role_by_player[ids[0]] = RoleName.SEER
        if len(ids) > 1:
            fake_role_by_player[ids[1]] = RoleName.MEDIUM
    # "gamma" (all lurk) assigns no fake roles.

    lurking = [pid for pid in ids if pid not in fake_role_by_player]
    return WolfDeceptionAssignment(pattern_name, label, fake_role_by_player, lurking)


def assign_madman_strategy(seed: int | None = None) -> tuple[str, RoleName | None]:
    rng = random.Random(seed)
    name, fake_role, _weight = _weighted_choice(MADMAN_STRATEGIES, rng)
    return name, fake_role


@dataclass
class FakeClaimGuard:
    """Consistency enforcement for the werewolf team's fabricated claims."""

    wolf_team_ids: set[str]
    _claimed_targets_by_faker: dict[str, set[str]] = field(default_factory=dict)

    def can_claim_result(self, faker_id: str, target_id: str, claimed_is_werewolf: bool) -> bool:
        already_claimed = target_id in self._claimed_targets_by_faker.get(faker_id, set())
        if already_claimed:
            return False
        if claimed_is_werewolf and target_id in self.wolf_team_ids:
            # A fake-seer/fake-medium must never accidentally out a teammate.
            return False
        return True

    def register_claim(self, faker_id: str, target_id: str) -> None:
        self._claimed_targets_by_faker.setdefault(faker_id, set()).add(target_id)
