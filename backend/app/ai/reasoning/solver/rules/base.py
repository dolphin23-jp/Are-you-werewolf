"""The rule-module contract.

Every deduction rule is a separate module so later work (night actions, curses,
new roles) can be added or removed without touching the solver. A module only
sees the builder, the perspective and the observations -- and it must read role
knowledge *through* the perspective, never straight out of the observations.
"""

from __future__ import annotations

from typing import Protocol

from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import Perspective
from app.ai.reasoning.solver.builder import ConstraintBuilder


class RuleModule(Protocol):
    module_id: str

    def add_hard_constraints(
        self,
        builder: ConstraintBuilder,
        perspective: Perspective,
        observations: ObservationSet,
    ) -> None: ...

    def explain(self, constraint_id: str) -> str: ...


class BaseRuleModule:
    """Explanation bookkeeping shared by every rule module."""

    module_id = "base"

    def __init__(self) -> None:
        self._explanations: dict[str, str] = {}

    def _record(
        self,
        builder: ConstraintBuilder,
        constraint_id: str,
        constraint: object,
        explanation: str,
    ) -> None:
        self._explanations[constraint_id] = explanation
        builder.for_module(self.module_id).add(constraint_id, constraint, explanation)  # type: ignore[arg-type]

    def explain(self, constraint_id: str) -> str:
        return self._explanations.get(constraint_id, "")
