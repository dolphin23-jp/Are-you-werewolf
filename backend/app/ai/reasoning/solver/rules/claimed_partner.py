"""What a freemason claimant's naming of a partner commits them to.

The two freemasons are told each other at the deal, and 17A has exactly two. A
claimant who names a partner says two things: "I hold the card", and "the other
card is this seat". This module reads the second, on condition of the first:

    if X is a real freemason, the seat X named as partner is the other one.

Encoded as `X is not a freemason, or the named seat is`, so it says nothing about
anyone unless X is real. Joined with the role count it is what lets the table
read a third freemason claim: when マコト and ホノカ have named each other, one
of them real makes both real, so the third claimant being real makes both fake
-- the pair stands or falls together (run #16, X24-2).

Like `ClaimedDivinationRuleModule` this is a premise about the claimant's
account, taken as a rule because every table plays it that way (a real
freemason naming somebody else throws away the one public proof the role has),
and kept narrow to match:

* Only the naming itself. Who confirmed whom is the engine's record
  (`partner_claims`); a confirmation is a naming in the other direction and
  enters as its own constraint.
* Never the viewer's own naming. A seat knows its own card and, if it is a
  freemason, its real partner (`AllyKnowledgeRuleModule`).

Its consequences are derived conclusions (`DERIVED_RULE_MODULE_IDS`), which
each seat notices at its own pace.
"""

from __future__ import annotations

from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import Perspective
from app.ai.reasoning.solver.backend import AnyOf, RoleIs, RoleIsNot
from app.ai.reasoning.solver.builder import ConstraintBuilder
from app.ai.reasoning.solver.rules.base import BaseRuleModule
from app.engine.roles import RoleName


class ClaimedPartnerRuleModule(BaseRuleModule):
    """A real freemason's named partner is the other freemason."""

    module_id = "claimed_partner"

    def add_hard_constraints(
        self,
        builder: ConstraintBuilder,
        perspective: Perspective,
        observations: ObservationSet,
    ) -> None:
        mason = RoleName.FREEMASON
        namings: list[tuple[str, str]] = []
        for claim in observations.partner_claims:
            namings.append((claim.claimant_id, claim.partner_id))
            # The engine keeps a confirmed pair as one record: the named seat
            # named the claimant back.
            if claim.confirmed:
                namings.append((claim.partner_id, claim.claimant_id))
        for claimant, partner in dict.fromkeys(namings):
            if claimant == perspective.viewer_id or claimant == partner:
                continue
            self._record(
                builder,
                f"{self.module_id}:{claimant}:{partner}",
                AnyOf(
                    (
                        RoleIsNot(player_id=claimant, role=mason),
                        RoleIs(player_id=partner, role=mason),
                    )
                ),
                (
                    f"{claimant}が真の共有者なら、相方と名指しした{partner}も共有者です"
                    "（共有者は互いを知っている）。"
                ),
            )


__all__ = ["ClaimedPartnerRuleModule"]
