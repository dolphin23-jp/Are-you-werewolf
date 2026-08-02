"""Role definitions and the 17-player composition."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum


class Team(StrEnum):
    VILLAGE = "village"
    WEREWOLF = "werewolf"
    FOX = "fox"


class RoleName(StrEnum):
    VILLAGER = "villager"
    WEREWOLF = "werewolf"
    MADMAN = "madman"
    SEER = "seer"
    MEDIUM = "medium"
    HUNTER = "hunter"
    FOX = "fox"
    FREEMASON = "freemason"


@dataclass(frozen=True)
class RoleDefinition:
    name: RoleName
    team: Team
    count: int
    label_ja: str
    description_ja: str
    has_night_action: bool = False


ROLE_DEFINITIONS: dict[RoleName, RoleDefinition] = {
    RoleName.VILLAGER: RoleDefinition(
        RoleName.VILLAGER, Team.VILLAGE, 7, "村人", "議論と投票で人狼を追放する"
    ),
    RoleName.WEREWOLF: RoleDefinition(
        RoleName.WEREWOLF,
        Team.WEREWOLF,
        3,
        "人狼",
        "夜に仲間と相談し襲撃先を決定する",
        has_night_action=True,
    ),
    RoleName.MADMAN: RoleDefinition(
        RoleName.MADMAN, Team.WEREWOLF, 1, "狂人", "人狼陣営。特殊能力はないが人狼の勝利を助ける"
    ),
    RoleName.SEER: RoleDefinition(
        RoleName.SEER,
        Team.VILLAGE,
        1,
        "占い師",
        "毎夜1人を占い人狼か判定する",
        has_night_action=True,
    ),
    RoleName.MEDIUM: RoleDefinition(
        RoleName.MEDIUM, Team.VILLAGE, 1, "霊媒師", "処刑者の正体を自動で知る"
    ),
    RoleName.HUNTER: RoleDefinition(
        RoleName.HUNTER,
        Team.VILLAGE,
        1,
        "狩人",
        "2日目夜から毎夜1人を護衛する",
        has_night_action=True,
    ),
    RoleName.FOX: RoleDefinition(
        RoleName.FOX, Team.FOX, 1, "妖狐", "占われると呪殺される。生き残れば勝利する"
    ),
    RoleName.FREEMASON: RoleDefinition(
        RoleName.FREEMASON, Team.VILLAGE, 2, "共有者", "相方と確定白として協調する"
    ),
}

TOTAL_PLAYERS = sum(d.count for d in ROLE_DEFINITIONS.values())
assert TOTAL_PLAYERS == 17

# Roles that must never be the scripted Day-0 first victim. Public because it is
# a rule the table can reason from, not just an implementation detail of the
# assigner: everyone knows the first victim was neither wolf nor fox.
FIRST_VICTIM_EXCLUDED_ROLES = frozenset({RoleName.WEREWOLF, RoleName.FOX})


class RoleAssigner:
    """Assigns roles to a fixed player-id list using a seeded RNG."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def assign(self, player_ids: list[str]) -> dict[str, RoleName]:
        if len(player_ids) != TOTAL_PLAYERS:
            raise ValueError(f"expected {TOTAL_PLAYERS} players, got {len(player_ids)}")
        pool: list[RoleName] = []
        for definition in ROLE_DEFINITIONS.values():
            pool.extend([definition.name] * definition.count)
        shuffled_ids = list(player_ids)
        self._rng.shuffle(shuffled_ids)
        self._rng.shuffle(pool)
        return dict(zip(shuffled_ids, pool, strict=True))

    def pick_first_victim(self, assignment: dict[str, RoleName]) -> str:
        """Pick the Day-0 scripted victim, guaranteed not to be Wolf/Fox."""
        eligible = [
            pid for pid, role in assignment.items() if role not in FIRST_VICTIM_EXCLUDED_ROLES
        ]
        return self._rng.choice(eligible)


class AlphaWolfTracker:
    """Tracks which werewolf is the "alpha" (the only one allowed to submit
    the night attack), reassigning on the alpha's death."""

    def __init__(self, wolf_ids: list[str], seed: int | None = None) -> None:
        if not wolf_ids:
            raise ValueError("wolf_ids must not be empty")
        rng = random.Random(seed)
        self._alive_wolves = list(wolf_ids)
        self.alpha_id: str = rng.choice(wolf_ids)

    def on_wolf_death(self, dead_player_id: str, rng: random.Random | None = None) -> None:
        if dead_player_id in self._alive_wolves:
            self._alive_wolves.remove(dead_player_id)
        if dead_player_id == self.alpha_id and self._alive_wolves:
            rng = rng or random.Random()
            self.alpha_id = rng.choice(self._alive_wolves)
