"""A seat knows its own card. Everything else it must deduce."""

from __future__ import annotations

from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import Perspective
from app.ai.reasoning.solver.backend import RoleIs
from app.ai.reasoning.solver.builder import ConstraintBuilder
from app.ai.reasoning.solver.rules.base import BaseRuleModule


class SelfKnowledgeRuleModule(BaseRuleModule):
    module_id = "self_knowledge"

    def add_hard_constraints(
        self,
        builder: ConstraintBuilder,
        perspective: Perspective,
        observations: ObservationSet,
    ) -> None:
        viewer_id = perspective.viewer_id
        if viewer_id is None:
            return
        # Through the perspective, never `observations.true_roles`: this module
        # runs for the public view too, and it must learn nothing there.
        known = perspective.known_roles(observations)
        role = known.get(viewer_id)
        if role is None:
            return
        self._record(
            builder,
            f"self_role:{viewer_id}",
            RoleIs(player_id=viewer_id, role=role),
            f"あなた({viewer_id})自身の役職は{role.value}で確定しています。",
        )
