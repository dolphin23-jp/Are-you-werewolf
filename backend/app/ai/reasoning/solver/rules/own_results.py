"""What the seer and medium know from their own ability, and nobody else does.

A true seer holding a black has the hardest fact in the game. Before this rule
the solver had no way to hear it: the divine *target* was observable through the
perspective, but not the verdict, so the seer could not treat their own result
as settled until they published it -- and after publishing, it was downgraded to
a claim like anyone else's.

White means "not a werewolf", not "villager". The fox divines white and dies of
it; the madman divines white and is on the wolf team. Encoding white as
village-confirmed would be the single most damaging mistake available here.
"""

from __future__ import annotations

from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import Perspective
from app.ai.reasoning.solver.backend import RoleIs, RoleIsNot
from app.ai.reasoning.solver.builder import ConstraintBuilder
from app.ai.reasoning.solver.rules.base import BaseRuleModule
from app.engine.roles import RoleName


class OwnAbilityResultRuleModule(BaseRuleModule):
    module_id = "own_results"

    def add_hard_constraints(
        self,
        builder: ConstraintBuilder,
        perspective: Perspective,
        observations: ObservationSet,
    ) -> None:
        for result in perspective.known_divine_results(observations):
            self._add(
                builder,
                kind="divine",
                night=result.night,
                target_id=result.target_id,
                is_werewolf=result.is_werewolf,
            )
        for medium in perspective.known_medium_results(observations):
            self._add(
                builder,
                kind="medium",
                night=medium.day,
                target_id=medium.target_id,
                is_werewolf=medium.is_werewolf,
            )

    def _add(
        self,
        builder: ConstraintBuilder,
        *,
        kind: str,
        night: int,
        target_id: str,
        is_werewolf: bool,
    ) -> None:
        ability = "占い" if kind == "divine" else "霊媒"
        if is_werewolf:
            self._record(
                builder,
                f"own_{kind}_black:{night}:{target_id}",
                RoleIs(player_id=target_id, role=RoleName.WEREWOLF),
                f"あなた自身の{night}日目の{ability}結果により、{target_id}は人狼です。",
            )
            return
        self._record(
            builder,
            f"own_{kind}_white:{night}:{target_id}",
            RoleIsNot(player_id=target_id, role=RoleName.WEREWOLF),
            (
                f"あなた自身の{night}日目の{ability}結果により、{target_id}は人狼ではありません"
                "(村人確定ではなく、妖狐・狂人の可能性は残ります)。"
            ),
        )
