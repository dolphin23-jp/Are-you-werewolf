"""Roles that are dealt knowing each other: werewolves and freemasons.

Nothing else gets ally knowledge. The madman is on the werewolf team and still
does not know who the wolves are, and the fox knows nobody -- both facts are
enforced here by simply never adding a constraint for them, because the
perspective declines to report allies for those seats.
"""

from __future__ import annotations

from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import Perspective
from app.ai.reasoning.solver.backend import RoleIs
from app.ai.reasoning.solver.builder import ConstraintBuilder
from app.ai.reasoning.solver.rules.base import BaseRuleModule


class AllyKnowledgeRuleModule(BaseRuleModule):
    module_id = "allies"

    def add_hard_constraints(
        self,
        builder: ConstraintBuilder,
        perspective: Perspective,
        observations: ObservationSet,
    ) -> None:
        viewer_id = perspective.viewer_id
        known = perspective.known_roles(observations)
        for player_id, role in sorted(known.items()):
            if player_id == viewer_id:
                continue  # handled by self_knowledge
            self._record(
                builder,
                f"ally_role:{player_id}",
                RoleIs(player_id=player_id, role=role),
                f"{player_id}が{role.value}であることを、あなたは配役時から知っています。",
            )
