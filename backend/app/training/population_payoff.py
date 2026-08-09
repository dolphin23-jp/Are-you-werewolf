"""Persistent empirical payoff table for three-faction policy populations.

A matchup profile is a triple of immutable policy generations: one controlling
Village, one Werewolf, and one Fox. Payoffs come only from completed games.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.learned_runner import LearnedEpisodeRunner
from app.training.policy_pool import NumpyPolicyPool

_TABLE_VERSION = 1


@dataclass(frozen=True, order=True)
class PolicyProfile:
    village: str
    werewolf: str
    fox: str

    def policy_id(self, team: Team) -> str:
        if team is Team.VILLAGE:
            return self.village
        if team is Team.WEREWOLF:
            return self.werewolf
        if team is Team.FOX:
            return self.fox
        raise ValueError(f"unsupported team {team}")


@dataclass(frozen=True)
class ProfilePayoff:
    profile: PolicyProfile
    games: int
    village_wins: int
    werewolf_wins: int
    fox_wins: int
    draws: int
    total_days: int

    def mean_payoff(self, team: Team) -> float:
        """Return empirical terminal reward mean for one faction."""
        wins = self.wins(team)
        losses = self.games - wins - self.draws
        if self.games == 0:
            raise ValueError("cannot score an empty payoff record")
        return (wins - losses) / self.games

    def wins(self, team: Team) -> int:
        if team is Team.VILLAGE:
            return self.village_wins
        if team is Team.WEREWOLF:
            return self.werewolf_wins
        if team is Team.FOX:
            return self.fox_wins
        raise ValueError(f"unsupported team {team}")

    @property
    def mean_days(self) -> float:
        if self.games == 0:
            return 0.0
        return self.total_days / self.games


class PopulationPayoffTable:
    """JSON-backed aggregate results for empirical normal-form profiles."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records = self._read()

    @property
    def records(self) -> tuple[ProfilePayoff, ...]:
        return tuple(sorted(self._records.values(), key=lambda record: record.profile))

    def get(self, profile: PolicyProfile) -> ProfilePayoff | None:
        return self._records.get(profile)

    def record_result(
        self,
        profile: PolicyProfile,
        *,
        winner: Team | None,
        is_draw: bool,
        days: int,
    ) -> ProfilePayoff:
        if days < 0:
            raise ValueError("days cannot be negative")
        if is_draw and winner is not None:
            raise ValueError("a draw cannot also have a winner")
        if not is_draw and winner is None:
            raise ValueError("a non-draw result requires a winner")

        current = self._records.get(
            profile,
            ProfilePayoff(profile, 0, 0, 0, 0, 0, 0),
        )
        updated = ProfilePayoff(
            profile=profile,
            games=current.games + 1,
            village_wins=current.village_wins + int(winner is Team.VILLAGE),
            werewolf_wins=current.werewolf_wins + int(winner is Team.WEREWOLF),
            fox_wins=current.fox_wins + int(winner is Team.FOX),
            draws=current.draws + int(is_draw),
            total_days=current.total_days + days,
        )
        self._records[profile] = updated
        self._write()
        return updated

    def policies(self, team: Team) -> tuple[str, ...]:
        values = {record.profile.policy_id(team) for record in self._records.values()}
        return tuple(sorted(values))

    def has_complete_cube(
        self,
        village: tuple[str, ...],
        werewolf: tuple[str, ...],
        fox: tuple[str, ...],
    ) -> bool:
        return all(
            PolicyProfile(village_id, werewolf_id, fox_id) in self._records
            for village_id in village
            for werewolf_id in werewolf
            for fox_id in fox
        )

    def _read(self) -> dict[PolicyProfile, ProfilePayoff]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("version") != _TABLE_VERSION:
            raise ValueError("unsupported population payoff table")
        records = raw.get("records")
        if not isinstance(records, list):
            raise ValueError("invalid population payoff records")
        parsed: dict[PolicyProfile, ProfilePayoff] = {}
        for item in records:
            record = _record_from_json(item)
            parsed[record.profile] = record
        return parsed

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _TABLE_VERSION,
            "records": [
                {
                    **asdict(record),
                    "profile": asdict(record.profile),
                }
                for record in self.records
            ],
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


def evaluate_policy_profile(
    player_specs: list[PlayerSpec],
    pool: NumpyPolicyPool,
    table: PopulationPayoffTable,
    profile: PolicyProfile,
    *,
    seeds: tuple[int, ...],
    max_discussion_ticks: int = 8,
    temperature: float = 1.0,
) -> ProfilePayoff:
    """Run missing games for one immutable three-policy profile."""
    if not seeds:
        raise ValueError("profile evaluation requires at least one seed")
    village_model = pool.load(profile.village)
    werewolf_model = pool.load(profile.werewolf)
    fox_model = pool.load(profile.fox)
    team_models = {
        Team.VILLAGE: village_model,
        Team.WEREWOLF: werewolf_model,
        Team.FOX: fox_model,
    }

    for seed in seeds:
        result = LearnedEpisodeRunner(
            player_specs,
            village_model,
            team_models=team_models,
            max_discussion_ticks=max_discussion_ticks,
            temperature=temperature,
        ).run(seed)
        table.record_result(
            profile,
            winner=result.winner,
            is_draw=result.is_draw,
            days=result.days,
        )

    record = table.get(profile)
    if record is None:
        raise RuntimeError("profile evaluation did not create a payoff record")
    return record


def _record_from_json(item: Any) -> ProfilePayoff:
    if not isinstance(item, dict):
        raise ValueError("invalid population payoff record")
    raw_profile = item.get("profile")
    if not isinstance(raw_profile, dict):
        raise ValueError("payoff record is missing its policy profile")
    try:
        profile = PolicyProfile(
            village=str(raw_profile["village"]),
            werewolf=str(raw_profile["werewolf"]),
            fox=str(raw_profile["fox"]),
        )
        return ProfilePayoff(
            profile=profile,
            games=int(item["games"]),
            village_wins=int(item["village_wins"]),
            werewolf_wins=int(item["werewolf_wins"]),
            fox_wins=int(item["fox_wins"]),
            draws=int(item["draws"]),
            total_days=int(item["total_days"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid population payoff record fields") from exc
