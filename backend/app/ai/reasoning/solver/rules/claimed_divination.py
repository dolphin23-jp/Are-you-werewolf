"""What a seer claim commits its owner to on a night the curse is visible.

A published divination says two things: a colour, and "on night N I looked at
this seat". The colour stays the claimant's own business here -- a real seer
may lie about it, and `HonestResults` remains a supposition a caller opts into.
This module reads only the second statement, and only through the curse:

    if X is the real seer, the seat X published for night N is the seat X
    divined on night N; so if that seat were the fox it died that night, and
    no *other* corpse of that night was the fox (nothing else cursed it, and
    the attack cannot kill a fox).

Each consequence is encoded as `X is not the seer, or <consequence>`, so the
rule says nothing about anyone unless X is real. Joined with what the corpses
make public (`DeathRuleModule`: two corpses in one night include the fox; a
night-0 corpse other than the first victim is the fox), it is what lets the
table conclude "X cannot be the real seer" from public information alone. The
first live Claude game turned on exactly that: two corpses after night 1, and a
wolf published a night-1 target who was alive. Nothing in the memo said he could
not be the seer, and the village hanged the villager who did.

Hard, and why that is honest. The premise -- a real seer's published target
and night are the ones they divined -- is a supposition about the claimant's
account, not a law of the engine; `docs/real_game_log_reasoning_plan.md` keeps
such suppositions explicit, and this module is where this one is made explicit.
It is taken as a rule because it is the one premise every table plays the curse
by (a real seer hiding their own curse throws away the strongest proof in the
game), because this engine's code publishes a seat's held results from its own
records when it obliges a seat to publish them, and because the owner decided so
after the first Claude game (analysis.md, change 1). Its reach is kept narrow to
match:

* Only verdicts whose night was stated. A verdict with no night is dated
  "the night before it was said" elsewhere, which is a guess -- a real seer
  publishing a held result late would be refuted by our arithmetic, not theirs.
* Only schedules the calendar allows. A claimant who was dead by that night, a
  target who was already dead, two different targets for one night: the
  premise is already void, and `timeline.py` reports those as what they are.
* Never the viewer's own claims. A seat knows its own card and, if it is the
  seer, its real divinations (which `FoxCurseRuleModule` reads directly). A
  real seer who misreported their own target must not make their own true world
  impossible; the rest of the table, reading only the account, may rule them out.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import Perspective
from app.ai.reasoning.solver.backend import AnyOf, RoleIsNot
from app.ai.reasoning.solver.builder import ConstraintBuilder
from app.ai.reasoning.solver.rules.base import BaseRuleModule
from app.ai.reasoning.timeline import SEER_RESULT, can_act_on_night, can_be_targeted_on_night
from app.engine.roles import RoleName


@dataclass(frozen=True)
class StatedDivination:
    """One published seer verdict the rule may read: its night was stated and
    the calendar allows it."""

    claimant_id: str
    night: int
    target_id: str


@dataclass(frozen=True)
class SeerClaimRefutation:
    """A claimant whose own account leaves the night's curse unexplained.

    On `night` the corpses (`corpse_ids`) include the cursed fox, so the real
    seer divined one of them; the claimant published `target_id` for that night,
    who is not among them. Kept as data, not prose, so the memo can word it and
    a test can assert on it.
    """

    claimant_id: str
    night: int
    target_id: str
    corpse_ids: tuple[str, ...]


def stated_divinations(
    observations: ObservationSet, *, viewer_id: str | None = None
) -> tuple[StatedDivination, ...]:
    """Published seer verdicts the premise can be applied to, oldest first.

    Restating the same verdict adds nothing; two different targets for one night
    void the claimant's account of that night, so neither is used.
    """
    seen: set[tuple[str, int, str]] = set()
    stated: list[StatedDivination] = []
    for verdict in sorted(observations.verdicts, key=lambda v: (v.day, v.claimant_id)):
        if verdict.result_type != SEER_RESULT or verdict.referenced_day is None:
            continue
        claimant, target, night = verdict.claimant_id, verdict.target_id, verdict.referenced_day
        if claimant == viewer_id or claimant == target or night < 0 or night >= verdict.day:
            continue
        if not can_act_on_night(observations, claimant, night):
            continue
        if not can_be_targeted_on_night(observations, target, night):
            continue
        key = (claimant, night, target)
        if key in seen:
            continue
        seen.add(key)
        stated.append(StatedDivination(claimant_id=claimant, night=night, target_id=target))
    per_night = Counter((item.claimant_id, item.night) for item in stated)
    return tuple(item for item in stated if per_night[(item.claimant_id, item.night)] == 1)


def curse_nights(observations: ObservationSet) -> dict[int, tuple[str, ...]]:
    """Nights whose corpses publicly include the cursed fox, with those corpses.

    Two corpses on a regular night (one attack, one curse), or any night death
    on night 0 (no attack that night; the first victim is not a night death).
    """
    nights: dict[int, tuple[str, ...]] = {}
    for night in observations.nights_with_deaths():
        corpses = tuple(death.player_id for death in observations.deaths_on(night))
        if night == 0 or len(corpses) >= 2:
            nights[night] = corpses
    return nights


def seer_claim_refutations(
    observations: ObservationSet, *, viewer_id: str | None = None
) -> tuple[SeerClaimRefutation, ...]:
    """The direct case of the rule, stated without a solver.

    The solver reaches every consequence of the rule, including chains across
    nights; this names the one-step case -- a curse night, and a published
    target who did not die on it -- so the memo can give the reason in words.
    """
    nights = curse_nights(observations)
    refutations: list[SeerClaimRefutation] = []
    for item in stated_divinations(observations, viewer_id=viewer_id):
        corpses = nights.get(item.night)
        if corpses is None or item.target_id in corpses:
            continue
        refutations.append(
            SeerClaimRefutation(
                claimant_id=item.claimant_id,
                night=item.night,
                target_id=item.target_id,
                corpse_ids=corpses,
            )
        )
    return tuple(refutations)


class ClaimedDivinationRuleModule(BaseRuleModule):
    """The curse rule, applied to each seer claimant's account on condition
    that the claimant is the real seer. See the module docstring."""

    module_id = "claimed_divination"

    def add_hard_constraints(
        self,
        builder: ConstraintBuilder,
        perspective: Perspective,
        observations: ObservationSet,
    ) -> None:
        for item in stated_divinations(observations, viewer_id=perspective.viewer_id):
            claimant, night, target = item.claimant_id, item.night, item.target_id
            not_real = RoleIsNot(player_id=claimant, role=RoleName.SEER)
            corpses = {death.player_id for death in observations.deaths_on(night)}
            prefix = f"{self.module_id}:{claimant}:{night}:{target}"
            if target not in corpses:
                self._record(
                    builder,
                    f"{prefix}:survived",
                    AnyOf((not_real, RoleIsNot(player_id=target, role=RoleName.FOX))),
                    (
                        f"{claimant}が真の占い師なら、公開どおり{night}日目夜に{target}を"
                        f"占っています。{target}はその夜に死んでいないので、"
                        f"その場合{target}は妖狐ではありません。"
                    ),
                )
            for corpse in sorted(corpses - {target}):
                self._record(
                    builder,
                    f"{prefix}:corpse:{corpse}",
                    AnyOf((not_real, RoleIsNot(player_id=corpse, role=RoleName.FOX))),
                    (
                        f"{claimant}が真の占い師なら{night}日目夜の占い先は{target}なので、"
                        f"同じ夜に死んだ{corpse}は呪殺ではありません。妖狐は襲撃では死なないため、"
                        f"その場合{corpse}は妖狐ではありません。"
                    ),
                )


__all__ = [
    "ClaimedDivinationRuleModule",
    "SeerClaimRefutation",
    "StatedDivination",
    "curse_nights",
    "seer_claim_refutations",
    "stated_divinations",
]
