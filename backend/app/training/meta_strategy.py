"""Payoff-driven population mixtures for three-faction historical self-play.

This is an intermediate empirical meta-strategy, not a Nash/JPSRO solver. It
iterates independent logit responses against the other factions' current
mixtures using only measured terminal payoffs from a complete profile cube.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.engine.roles import Team
from app.training.population_payoff import PolicyProfile, PopulationPayoffTable

_STRATEGY_VERSION = 1


@dataclass(frozen=True)
class PolicyWeight:
    policy_id: str
    probability: float


@dataclass(frozen=True)
class PopulationMetaStrategy:
    village: tuple[PolicyWeight, ...]
    werewolf: tuple[PolicyWeight, ...]
    fox: tuple[PolicyWeight, ...]

    def weights(self, team: Team) -> tuple[PolicyWeight, ...]:
        if team is Team.VILLAGE:
            return self.village
        if team is Team.WEREWOLF:
            return self.werewolf
        if team is Team.FOX:
            return self.fox
        raise ValueError(f"unsupported team {team}")

    def sample(self, team: Team, rng: random.Random) -> str:
        weights = self.weights(team)
        if not weights:
            raise ValueError(f"meta-strategy has no policies for {team}")
        needle = rng.random()
        cumulative = 0.0
        for item in weights:
            cumulative += item.probability
            if needle <= cumulative:
                return item.policy_id
        return weights[-1].policy_id

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _STRATEGY_VERSION,
            "village": [_weight_json(item) for item in self.village],
            "werewolf": [_weight_json(item) for item in self.werewolf],
            "fox": [_weight_json(item) for item in self.fox],
        }
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> PopulationMetaStrategy:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("version") != _STRATEGY_VERSION:
            raise ValueError("unsupported population meta-strategy")
        return cls(
            village=_weights_from_json(raw.get("village")),
            werewolf=_weights_from_json(raw.get("werewolf")),
            fox=_weights_from_json(raw.get("fox")),
        )


def solve_logit_response_mixture(
    table: PopulationPayoffTable,
    *,
    village: tuple[str, ...] | None = None,
    werewolf: tuple[str, ...] | None = None,
    fox: tuple[str, ...] | None = None,
    temperature: float = 0.25,
    iterations: int = 100,
    damping: float = 0.5,
) -> PopulationMetaStrategy:
    """Iterate independent soft best responses on a complete empirical cube."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0 < damping <= 1:
        raise ValueError("damping must be in (0, 1]")

    village_ids = village or table.policies(Team.VILLAGE)
    werewolf_ids = werewolf or table.policies(Team.WEREWOLF)
    fox_ids = fox or table.policies(Team.FOX)
    if not village_ids or not werewolf_ids or not fox_ids:
        raise ValueError("meta-strategy requires at least one policy per faction")
    if not table.has_complete_cube(village_ids, werewolf_ids, fox_ids):
        raise ValueError("meta-strategy requires a complete measured payoff cube")

    mixtures = {
        Team.VILLAGE: _uniform(village_ids),
        Team.WEREWOLF: _uniform(werewolf_ids),
        Team.FOX: _uniform(fox_ids),
    }
    policies = {
        Team.VILLAGE: village_ids,
        Team.WEREWOLF: werewolf_ids,
        Team.FOX: fox_ids,
    }

    for _ in range(iterations):
        previous = {team: dict(weights) for team, weights in mixtures.items()}
        updated: dict[Team, dict[str, float]] = {}
        for team in Team:
            scores = {
                policy_id: _expected_payoff(
                    table,
                    team,
                    policy_id,
                    previous,
                    policies,
                )
                for policy_id in policies[team]
            }
            response = _softmax(scores, temperature)
            blended = {
                policy_id: (
                    (1.0 - damping) * previous[team][policy_id]
                    + damping * response[policy_id]
                )
                for policy_id in policies[team]
            }
            updated[team] = _normalize(blended)
        mixtures = updated

    return PopulationMetaStrategy(
        village=_as_weights(mixtures[Team.VILLAGE]),
        werewolf=_as_weights(mixtures[Team.WEREWOLF]),
        fox=_as_weights(mixtures[Team.FOX]),
    )


def _expected_payoff(
    table: PopulationPayoffTable,
    team: Team,
    policy_id: str,
    mixtures: dict[Team, dict[str, float]],
    policies: dict[Team, tuple[str, ...]],
) -> float:
    total = 0.0
    if team is Team.VILLAGE:
        for wolf_id in policies[Team.WEREWOLF]:
            for fox_id in policies[Team.FOX]:
                record = table.get(PolicyProfile(policy_id, wolf_id, fox_id))
                if record is None:
                    raise ValueError("missing village payoff profile")
                probability = (
                    mixtures[Team.WEREWOLF][wolf_id] * mixtures[Team.FOX][fox_id]
                )
                total += probability * record.mean_payoff(team)
        return total
    if team is Team.WEREWOLF:
        for village_id in policies[Team.VILLAGE]:
            for fox_id in policies[Team.FOX]:
                record = table.get(PolicyProfile(village_id, policy_id, fox_id))
                if record is None:
                    raise ValueError("missing werewolf payoff profile")
                probability = (
                    mixtures[Team.VILLAGE][village_id] * mixtures[Team.FOX][fox_id]
                )
                total += probability * record.mean_payoff(team)
        return total
    if team is Team.FOX:
        for village_id in policies[Team.VILLAGE]:
            for wolf_id in policies[Team.WEREWOLF]:
                record = table.get(PolicyProfile(village_id, wolf_id, policy_id))
                if record is None:
                    raise ValueError("missing fox payoff profile")
                probability = (
                    mixtures[Team.VILLAGE][village_id]
                    * mixtures[Team.WEREWOLF][wolf_id]
                )
                total += probability * record.mean_payoff(team)
        return total
    raise ValueError(f"unsupported team {team}")


def _uniform(policy_ids: tuple[str, ...]) -> dict[str, float]:
    probability = 1.0 / len(policy_ids)
    return {policy_id: probability for policy_id in policy_ids}


def _softmax(scores: dict[str, float], temperature: float) -> dict[str, float]:
    peak = max(scores.values())
    weights = {
        policy_id: math.exp((score - peak) / temperature)
        for policy_id, score in scores.items()
    }
    return _normalize(weights)


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0 or not math.isfinite(total):
        raise ValueError("meta-strategy weights must have finite positive mass")
    return {policy_id: weight / total for policy_id, weight in weights.items()}


def _as_weights(weights: dict[str, float]) -> tuple[PolicyWeight, ...]:
    return tuple(
        PolicyWeight(policy_id, probability)
        for policy_id, probability in sorted(weights.items())
    )


def _weight_json(weight: PolicyWeight) -> dict[str, Any]:
    return {"policy_id": weight.policy_id, "probability": weight.probability}


def _weights_from_json(raw: Any) -> tuple[PolicyWeight, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("invalid meta-strategy weight list")
    weights: list[PolicyWeight] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("invalid meta-strategy weight")
        policy_id = item.get("policy_id")
        probability = item.get("probability")
        if not isinstance(policy_id, str):
            raise ValueError("meta-strategy policy_id must be a string")
        if not isinstance(probability, (int, float)):
            raise ValueError("meta-strategy probability must be numeric")
        weights.append(PolicyWeight(policy_id, float(probability)))
    normalized = _normalize({item.policy_id: item.probability for item in weights})
    return _as_weights(normalized)
