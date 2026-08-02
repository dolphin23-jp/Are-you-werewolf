"""Behavioural assumptions, kept strictly out of the game's hard rules.

"The seer would have COed by now" and "this claimant is telling the truth" are
not rules of werewolf -- they are things a player chooses to suppose. Mixing
them into the hard constraints is how an AI ends up treating a lone claim as
proof, so they live here and only enter a deduction when a caller asks for them.

Four separate suppositions, because collapsing them loses real distinctions:

* `GenuineClaim(p, role)`   -- p really holds the role they claimed.
* `HonestResults(p)`        -- the verdicts p published are factually correct.
  Independent of the above: a madman's invented black can happen to land on a
  real wolf, and that world is perfectly consistent.
* `CompleteResultDisclosure(p)` -- p published every result they hold, so the
  count has to match the nights the role could have acted.
* `NoLatentClaim(role)`     -- nobody holds `role` while staying quiet about it.
  This is the toggle between "a lone seer CO is confirmed" and "it isn't".
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.solver.backend import (
    LabelledConstraint,
    RoleIs,
    RoleIsNot,
    Unsatisfiable,
)
from app.engine.roles import ROLE_DEFINITIONS, RoleName

MODULE_ID = "assumptions"

# Roles whose holder learns something every night, and therefore accumulates one
# result per night they were alive to act.
_NIGHTLY_RESULT_ROLES = (RoleName.SEER,)


@dataclass(frozen=True)
class GenuineClaim:
    player_id: str
    role: RoleName

    @property
    def key(self) -> str:
        return f"genuine:{self.player_id}:{self.role.value}"


@dataclass(frozen=True)
class HonestResults:
    player_id: str

    @property
    def key(self) -> str:
        return f"honest:{self.player_id}"


@dataclass(frozen=True)
class CompleteResultDisclosure:
    player_id: str

    @property
    def key(self) -> str:
        return f"complete:{self.player_id}"


@dataclass(frozen=True)
class NoLatentClaim:
    role: RoleName

    @property
    def key(self) -> str:
        return f"no_latent:{self.role.value}"


Assumption = GenuineClaim | HonestResults | CompleteResultDisclosure | NoLatentClaim


def expand(
    assumptions: Sequence[Assumption], observations: ObservationSet
) -> list[LabelledConstraint]:
    constraints: list[LabelledConstraint] = []
    for assumption in assumptions:
        constraints.extend(_expand_one(assumption, observations))
    return constraints


def signature(assumptions: Iterable[Assumption]) -> str:
    return ";".join(sorted(assumption.key for assumption in assumptions))


def _expand_one(
    assumption: Assumption, observations: ObservationSet
) -> list[LabelledConstraint]:
    match assumption:
        case GenuineClaim():
            return _genuine_claim(assumption, observations)
        case HonestResults():
            return _honest_results(assumption, observations)
        case CompleteResultDisclosure():
            return _complete_disclosure(assumption, observations)
        case NoLatentClaim():
            return _no_latent_claim(assumption, observations)
    raise TypeError(f"unknown assumption {assumption!r}")


def _genuine_claim(
    assumption: GenuineClaim, observations: ObservationSet
) -> list[LabelledConstraint]:
    label = ROLE_DEFINITIONS[assumption.role].label_ja
    return [
        LabelledConstraint(
            constraint_id=assumption.key,
            constraint=RoleIs(player_id=assumption.player_id, role=assumption.role),
            explanation=f"{assumption.player_id}の{label}COを真と仮定しています。",
            module_id=MODULE_ID,
        )
    ]


def _honest_results(
    assumption: HonestResults, observations: ObservationSet
) -> list[LabelledConstraint]:
    """Published verdicts are true statements about who is a wolf.

    Note what this does *not* say: that the claimant holds the role. A verdict
    can be honest by luck, and the pair of assumptions has to stay separable for
    "the madman's fake black happened to be right" to be expressible at all.
    """
    constraints: list[LabelledConstraint] = []
    for index, verdict in enumerate(observations.verdicts_by(assumption.player_id)):
        ability = "霊媒" if verdict.result_type == "medium" else "占い"
        colour = "黒" if verdict.is_werewolf else "白"
        constraint = (
            RoleIs(player_id=verdict.target_id, role=RoleName.WEREWOLF)
            if verdict.is_werewolf
            else RoleIsNot(player_id=verdict.target_id, role=RoleName.WEREWOLF)
        )
        constraints.append(
            LabelledConstraint(
                constraint_id=f"{assumption.key}:{index}",
                constraint=constraint,
                explanation=(
                    f"{verdict.day}日目の{assumption.player_id}の{ability}判定"
                    f"「{verdict.target_id}={colour}」が正しいと仮定しています。"
                ),
                module_id=MODULE_ID,
            )
        )
    return constraints


def _complete_disclosure(
    assumption: CompleteResultDisclosure, observations: ObservationSet
) -> list[LabelledConstraint]:
    """The claimant is holding nothing back, so the count must add up.

    A day-3 seer claim backed by one result is either hiding a result or is not
    the seer. Under this assumption the first option is ruled out, so the world
    where both hold is impossible -- and saying so is more useful than silently
    accepting a story with a missing night.
    """
    claimed = observations.claimed_role_of(assumption.player_id)
    if claimed not in _NIGHTLY_RESULT_ROLES:
        return []
    expected = _expected_result_count(assumption.player_id, observations)
    published = len(observations.verdicts_by(assumption.player_id))
    if published >= expected:
        return []
    label = ROLE_DEFINITIONS[claimed].label_ja
    return [
        LabelledConstraint(
            constraint_id=f"{assumption.key}:count",
            constraint=Unsatisfiable(
                reason=f"{assumption.player_id} published {published} of {expected} results"
            ),
            explanation=(
                f"{assumption.player_id}が真の{label}なら{expected}件の結果を持つはずですが、"
                f"公開されているのは{published}件です。"
            ),
            module_id=MODULE_ID,
        )
    ]


def _expected_result_count(player_id: str, observations: ObservationSet) -> int:
    """Nights the seer could have acted before today's discussion.

    Day 0's night happens before Day 1, so on day D a seer alive throughout has
    D results. A dead claimant stops accumulating, but the ones they had before
    dying still count.
    """
    if observations.day <= 0:
        return 0
    return observations.day


def _no_latent_claim(
    assumption: NoLatentClaim, observations: ObservationSet
) -> list[LabelledConstraint]:
    claimants = set(observations.claimants_of(assumption.role))
    label = ROLE_DEFINITIONS[assumption.role].label_ja
    return [
        LabelledConstraint(
            constraint_id=f"{assumption.key}:{player_id}",
            constraint=RoleIsNot(player_id=player_id, role=assumption.role),
            explanation=(
                f"潜伏{label}はいないと仮定しているため、"
                f"{label}COしていない{player_id}は{label}ではありません。"
            ),
            module_id=MODULE_ID,
        )
        for player_id in observations.player_ids
        if player_id not in claimants
    ]


def no_latent_constraints(
    role: RoleName, observations: ObservationSet
) -> list[LabelledConstraint]:
    """Exposed so a caller can ask whether a latent holder is *required*."""
    return _no_latent_claim(NoLatentClaim(role), observations)


__all__ = [
    "MODULE_ID",
    "Assumption",
    "CompleteResultDisclosure",
    "GenuineClaim",
    "HonestResults",
    "NoLatentClaim",
    "expand",
    "no_latent_constraints",
    "signature",
]
