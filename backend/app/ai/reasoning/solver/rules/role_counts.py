"""The fixed 17-player composition -- the one thing every seat knows for certain."""

from __future__ import annotations

from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import Perspective
from app.ai.reasoning.solver.backend import RoleCountIs
from app.ai.reasoning.solver.builder import ConstraintBuilder
from app.ai.reasoning.solver.rules.base import BaseRuleModule
from app.engine.roles import ROLE_DEFINITIONS


class RoleCountRuleModule(BaseRuleModule):
    """村人7・人狼3・狂人1・占い師1・霊媒師1・狩人1・妖狐1・共有者2.

    Read from `ROLE_DEFINITIONS` rather than restated here: a composition change
    must not leave the solver reasoning about a game nobody is playing.
    """

    module_id = "role_counts"

    def add_hard_constraints(
        self,
        builder: ConstraintBuilder,
        perspective: Perspective,
        observations: ObservationSet,
    ) -> None:
        for role, definition in ROLE_DEFINITIONS.items():
            self._record(
                builder,
                f"role_count:{role.value}",
                RoleCountIs(role=role, count=definition.count),
                f"{definition.label_ja}はちょうど{definition.count}人です。",
            )
