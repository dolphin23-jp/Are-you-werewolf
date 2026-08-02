"""Night events: death, attack, guard, and the fox's curse.

There are exactly two ways to die overnight in this ruleset -- the wolves' one
attack, and the curse that kills a divined fox. Everything below is a
consequence of that, and every consequence is gated on knowing *which* it was.

The table cannot tell. A corpse in the morning is an attack victim or a cursed
fox, and the public death cause deliberately collapses the two. So the public
module deduces almost nothing, and the seats that took the night actions deduce
a great deal -- which is the point: night reasoning has to differ by viewpoint,
or the village is quietly playing with the wolves' information.

Each cause lives in its own module so a new one (a second killer role, a
different protection) is a new file rather than a rewrite.
"""

from __future__ import annotations

from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import Perspective
from app.ai.reasoning.solver.backend import AnyOf, RoleIs, RoleIsNot
from app.ai.reasoning.solver.builder import ConstraintBuilder
from app.ai.reasoning.solver.rules.base import BaseRuleModule
from app.engine.roles import RoleName


class DeathRuleModule(BaseRuleModule):
    """What the whole table can conclude from the corpses alone.

    Usually nothing: one night death is an attack or a curse and there is no way
    to tell. Two on the same night is the exception -- there is only one attack
    and only one curse per night, so one of the two was the fox. That is public,
    exact, and worth a great deal.
    """

    module_id = "death"

    def add_hard_constraints(
        self,
        builder: ConstraintBuilder,
        perspective: Perspective,
        observations: ObservationSet,
    ) -> None:
        for night in observations.nights_with_deaths():
            deaths = observations.deaths_on(night)
            if len(deaths) < 2:
                # One corpse could be either cause. Saying more here is exactly
                # the overclaim this module exists to avoid.
                continue
            self._record(
                builder,
                f"two_deaths_include_fox:{night}",
                AnyOf(
                    tuple(
                        RoleIs(player_id=death.player_id, role=RoleName.FOX)
                        for death in deaths
                    )
                ),
                (
                    f"{night}日目の夜に死体が{len(deaths)}体出ています。"
                    "襲撃は1件、呪殺は1件しか起きないため、"
                    f"{'、'.join(death.player_id for death in deaths)}のいずれかが妖狐です。"
                ),
            )


class AttackRuleModule(BaseRuleModule):
    """What the wolves can conclude, knowing where they struck.

    Knowing the attack target splits the morning's corpses: any death that was
    not their target had no other cause available but the curse, so that player
    was the fox. It is the cleanest deduction in the game and only the wolf team
    can make it.
    """

    module_id = "attack"

    def add_hard_constraints(
        self,
        builder: ConstraintBuilder,
        perspective: Perspective,
        observations: ObservationSet,
    ) -> None:
        attacks = perspective.known_night_actions(observations).attacks
        if not attacks:
            return
        for night, target_id in sorted(attacks.items()):
            for death in observations.deaths_on(night):
                if death.player_id == target_id:
                    # Their own kill -- or a curse that reached the same seat
                    # first. Either way it settles nothing about that role.
                    continue
                self._record(
                    builder,
                    f"unattacked_night_death_is_fox:{night}:{death.player_id}",
                    RoleIs(player_id=death.player_id, role=RoleName.FOX),
                    (
                        f"{night}日目の襲撃先は{target_id}なので、"
                        f"同じ夜に死んだ{death.player_id}は呪殺されたことになり、妖狐です。"
                    ),
                )


class GuardRuleModule(BaseRuleModule):
    """What the hunter can conclude, knowing who they covered.

    A guarded seat cannot be killed by the attack. If one dies anyway, the only
    remaining cause is the curse, and the curse only reaches the fox -- so the
    hunter has accidentally found it.
    """

    module_id = "guard"

    def add_hard_constraints(
        self,
        builder: ConstraintBuilder,
        perspective: Perspective,
        observations: ObservationSet,
    ) -> None:
        guards = perspective.known_night_actions(observations).guards
        if not guards:
            return
        for night, target_id in sorted(guards.items()):
            if not any(death.player_id == target_id for death in observations.deaths_on(night)):
                continue
            self._record(
                builder,
                f"guarded_but_died_is_fox:{night}:{target_id}",
                RoleIs(player_id=target_id, role=RoleName.FOX),
                (
                    f"{night}日目にあなたが護衛した{target_id}が死亡しています。"
                    "護衛は襲撃を防ぐため、残る死因は呪殺だけで、"
                    f"{target_id}は妖狐です。"
                ),
            )


class FoxCurseRuleModule(BaseRuleModule):
    """What the seer can conclude, knowing where they looked.

    A divined fox always dies that night. So the seat the seer looked at is not
    the fox if it survived, and any *other* seat that died that night is not the
    fox either -- nothing cursed it, and the attack cannot kill a fox.
    """

    module_id = "fox_curse"

    def add_hard_constraints(
        self,
        builder: ConstraintBuilder,
        perspective: Perspective,
        observations: ObservationSet,
    ) -> None:
        divines = perspective.known_night_actions(observations).divines
        if not divines:
            return
        for night, target_id in sorted(divines.items()):
            deaths = {death.player_id for death in observations.deaths_on(night)}
            if target_id not in deaths:
                self._record(
                    builder,
                    f"divined_and_survived_not_fox:{night}:{target_id}",
                    RoleIsNot(player_id=target_id, role=RoleName.FOX),
                    (
                        f"{night}日目に占った{target_id}は生き残っています。"
                        "妖狐は占われれば必ず呪殺されるため、妖狐ではありません。"
                    ),
                )
            for player_id in sorted(deaths - {target_id}):
                self._record(
                    builder,
                    f"uncursed_night_death_not_fox:{night}:{player_id}",
                    RoleIsNot(player_id=player_id, role=RoleName.FOX),
                    (
                        f"{night}日目に占ったのは{target_id}なので{player_id}は呪殺されておらず、"
                        "妖狐は襲撃では死なないため、妖狐ではありません。"
                    ),
                )
