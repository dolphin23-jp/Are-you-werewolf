"""The Day-0 victim is scripted, and the rules say who it cannot have been.

Public information: the whole table may reason from it, which is why the
excluded set is read from the engine rather than restated here.
"""

from __future__ import annotations

from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import Perspective
from app.ai.reasoning.solver.backend import RoleIsNot
from app.ai.reasoning.solver.builder import ConstraintBuilder
from app.ai.reasoning.solver.rules.base import BaseRuleModule
from app.engine.roles import FIRST_VICTIM_EXCLUDED_ROLES, ROLE_DEFINITIONS


class FirstVictimRuleModule(BaseRuleModule):
    module_id = "first_victim"

    def add_hard_constraints(
        self,
        builder: ConstraintBuilder,
        perspective: Perspective,
        observations: ObservationSet,
    ) -> None:
        victim_id = observations.first_victim_id
        if victim_id is None:
            return
        for role in sorted(FIRST_VICTIM_EXCLUDED_ROLES):
            self._record(
                builder,
                f"first_victim_not:{role.value}",
                RoleIsNot(player_id=victim_id, role=role),
                (
                    f"初日犠牲者{victim_id}は"
                    f"{ROLE_DEFINITIONS[role].label_ja}ではありません(ルール上選ばれません)。"
                ),
            )
