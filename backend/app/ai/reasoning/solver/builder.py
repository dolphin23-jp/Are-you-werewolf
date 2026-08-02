"""Collects the labelled constraints that describe one board from one viewpoint.

Rules never touch a solver object. They append to a builder, which keeps the
constraint IR, the id and the human explanation together -- so the same
declaration serves the encoder and the eventual "why is this impossible" answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.ai.reasoning.solver.backend import Constraint, LabelledConstraint


@dataclass
class ConstraintBuilder:
    module_id: str = ""
    constraints: list[LabelledConstraint] = field(default_factory=list)

    def for_module(self, module_id: str) -> ConstraintBuilder:
        """A view that stamps everything added through it with one module id."""
        return ConstraintBuilder(module_id=module_id, constraints=self.constraints)

    def add(self, constraint_id: str, constraint: Constraint, explanation: str) -> None:
        self.constraints.append(
            LabelledConstraint(
                constraint_id=constraint_id,
                constraint=constraint,
                explanation=explanation,
                module_id=self.module_id,
            )
        )

    def constraint_ids(self) -> tuple[str, ...]:
        return tuple(item.constraint_id for item in self.constraints)

    def signature(self) -> str:
        """Identifies this exact constraint set for caching."""
        return ";".join(sorted(self.constraint_ids()))
