"""Behavioural assumptions, kept strictly out of the game's hard rules.

"The seer would have COed by now" and "this claimant is telling the truth" are
not rules of werewolf -- they are things a player chooses to suppose. Mixing
them into the hard constraints is how an AI ends up treating a lone claim as
proof, so they live here and only enter a deduction when a caller asks for them.

Five separate suppositions, because collapsing them loses real distinctions:

* `GenuineClaim(p, role)`   -- p really holds the role they claimed.
* `HonestResults(p)`        -- the verdicts p published are factually correct.
  Independent of the above: a madman's invented black can happen to land on a
  real wolf, and that world is perfectly consistent.
* `CompleteResultDisclosure(p)` -- p published every result they hold, so the
  count has to match the nights the role could have acted.
* `NoLatentClaim(role)`     -- nobody holds `role` while staying quiet about it.
  This is the toggle between "a lone seer CO is confirmed" and "it isn't".
* `AccurateTimeline(p)`     -- the nights p says their results came from are the
  nights they came from. Separate again: a real seer may lie about *when* while
  telling the truth about *who*, so the timing is its own supposition.
* `NoVillageBluff()`        -- no village-side seat claims a role it does not
  hold, so a false claim (seer, medium, hunter, freemason) comes from a wolf,
  the madman or the fox. A table's habit, not a rule: a hunter may claim seer.
* `TruthfulIfGenuine()`     -- whoever really holds the seer or medium card
  publishes the colours it showed. Unlike `HonestResults(p)` it says nothing
  of a claimant who is not the holder, so a seat that knows a published colour
  is false can still suppose it: that claimant is then not the holder, where
  `HonestResults` would make the whole board impossible.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.solver.backend import (
    AnyOf,
    LabelledConstraint,
    RoleIs,
    RoleIsNot,
    RoleIsOneOf,
    Unsatisfiable,
)
from app.ai.reasoning.timeline import disclosable_result_nights, find_timeline_conflicts
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


@dataclass(frozen=True)
class AccurateTimeline:
    player_id: str

    @property
    def key(self) -> str:
        return f"timeline:{self.player_id}"


@dataclass(frozen=True)
class NoVillageBluff:
    @property
    def key(self) -> str:
        return "no_village_bluff"


@dataclass(frozen=True)
class TruthfulIfGenuine:
    @property
    def key(self) -> str:
        return "truthful_if_genuine"


Assumption = (
    GenuineClaim
    | HonestResults
    | CompleteResultDisclosure
    | NoLatentClaim
    | AccurateTimeline
    | NoVillageBluff
    | TruthfulIfGenuine
)

# The roles outside the village team: whoever bluffs a role, under
# `NoVillageBluff`, holds one of these.
NON_VILLAGE_ROLES = (RoleName.WEREWOLF, RoleName.MADMAN, RoleName.FOX)


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
        case AccurateTimeline():
            return _accurate_timeline(assumption, observations)
        case NoVillageBluff():
            return _no_village_bluff(assumption, observations)
        case TruthfulIfGenuine():
            return _truthful_if_genuine(assumption, observations)
    raise TypeError(f"unknown assumption {assumption!r}")


def _truthful_if_genuine(
    assumption: TruthfulIfGenuine, observations: ObservationSet
) -> list[LabelledConstraint]:
    """For every published verdict: its author is not the holder of that
    ability, or the colour is true."""
    constraints: list[LabelledConstraint] = []
    for index, verdict in enumerate(observations.verdicts):
        role = RoleName.MEDIUM if verdict.result_type == "medium" else RoleName.SEER
        colour = "黒" if verdict.is_werewolf else "白"
        true_colour = (
            RoleIs(player_id=verdict.target_id, role=RoleName.WEREWOLF)
            if verdict.is_werewolf
            else RoleIsNot(player_id=verdict.target_id, role=RoleName.WEREWOLF)
        )
        label = ROLE_DEFINITIONS[role].label_ja
        constraints.append(
            LabelledConstraint(
                constraint_id=f"{assumption.key}:{index}",
                constraint=AnyOf(
                    (RoleIsNot(player_id=verdict.claimant_id, role=role), true_colour)
                ),
                explanation=(
                    f"真の{label}は色を偽らないと仮定しているため、{verdict.claimant_id}が真の{label}なら"
                    f"「{verdict.target_id}={colour}」は正しい判定です。"
                ),
                module_id=MODULE_ID,
            )
        )
    return constraints


def _no_village_bluff(
    assumption: NoVillageBluff, observations: ObservationSet
) -> list[LabelledConstraint]:
    """Every standing claim is either true or made by a non-village seat.

    A villager CO is left alone: "I'm just a villager" is what a hiding
    hunter says, and calling that a bluff would make the supposition about
    something else.
    """
    constraints: list[LabelledConstraint] = []
    for claim in observations.claims:
        if claim.role is RoleName.VILLAGER:
            continue
        label = ROLE_DEFINITIONS[claim.role].label_ja
        constraints.append(
            LabelledConstraint(
                constraint_id=f"{assumption.key}:{claim.player_id}:{claim.role.value}",
                constraint=RoleIsOneOf(
                    player_id=claim.player_id, roles=(claim.role, *NON_VILLAGE_ROLES)
                ),
                explanation=(
                    f"村側は役職を騙らないと仮定しているため、{label}COした{claim.player_id}は"
                    f"{label}か、人狼・狂人・妖狐のどれかです。"
                ),
                module_id=MODULE_ID,
            )
        )
    return constraints


def _accurate_timeline(
    assumption: AccurateTimeline, observations: ObservationSet
) -> list[LabelledConstraint]:
    """Take the claimant's account of *when* at face value, then check it.

    Under this supposition a schedule the calendar rules out is not a schedule a
    holder of the claimed role could have produced, so the claimed role is
    excluded. Without a claimed role there is nothing to exclude -- an
    unattributed verdict conflicting with the calendar is odd, but it is the
    belief layer's business, not the solver's.
    """
    conflicts = find_timeline_conflicts(observations)
    constraints: list[LabelledConstraint] = []
    for conflict in conflicts:
        if conflict.claimant_id != assumption.player_id:
            continue
        if conflict.claimed_role is None:
            continue
        constraints.append(
            LabelledConstraint(
                constraint_id=f"{assumption.key}:{conflict.kind}:{conflict.night}",
                constraint=RoleIsNot(
                    player_id=conflict.claimant_id, role=conflict.claimed_role
                ),
                explanation=(
                    f"{conflict.explanation}"
                    f"{conflict.claimant_id}の申告した時系列が正確だと仮定すると、"
                    f"{ROLE_DEFINITIONS[conflict.claimed_role].label_ja}ではありません。"
                ),
                module_id=MODULE_ID,
            )
        )
    return constraints


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
    expected_nights = disclosable_result_nights(
        observations, assumption.player_id, observations.day
    )
    published_list = [v.source_night for v in observations.verdicts_by(assumption.player_id)]
    published_nights = frozenset(published_list)
    if published_nights == expected_nights and len(published_list) == len(published_nights):
        return []
    expected = len(expected_nights)
    published = len(published_list)
    label = ROLE_DEFINITIONS[claimed].label_ja
    return [
        LabelledConstraint(
            constraint_id=f"{assumption.key}:count",
            constraint=Unsatisfiable(
                reason=(f"{assumption.player_id} published nights {published_list}; "
                        f"expected {sorted(expected_nights)}")
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
    "AccurateTimeline",
    "Assumption",
    "CompleteResultDisclosure",
    "GenuineClaim",
    "NON_VILLAGE_ROLES",
    "HonestResults",
    "NoLatentClaim",
    "NoVillageBluff",
    "TruthfulIfGenuine",
    "expand",
    "no_latent_constraints",
    "signature",
]
