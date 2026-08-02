"""Code-derived explanations for the constraints a deduction rests on.

The reason a world is impossible is a property of the rules, not a sentence a
model invents afterwards. Every constraint carries its own wording from the
module that added it; this registry just makes them addressable by id so an
UNSAT answer can be turned into "because the composition has exactly one seer".
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from app.ai.reasoning.solver.backend import ContradictionResult, LabelledConstraint


class ExplanationRegistry:
    def __init__(self, constraints: Iterable[LabelledConstraint] = ()) -> None:
        self._by_id: dict[str, LabelledConstraint] = {
            item.constraint_id: item for item in constraints
        }

    def add(self, constraint: LabelledConstraint) -> None:
        self._by_id[constraint.constraint_id] = constraint

    def explain(self, constraint_id: str) -> str:
        item = self._by_id.get(constraint_id)
        return item.explanation if item is not None else ""

    def module_of(self, constraint_id: str) -> str:
        item = self._by_id.get(constraint_id)
        return item.module_id if item is not None else ""

    def explain_all(self, constraint_ids: Sequence[str]) -> tuple[str, ...]:
        return tuple(
            explanation
            for explanation in (self.explain(cid) for cid in constraint_ids)
            if explanation
        )

    def describe(self, result: ContradictionResult) -> tuple[str, ...]:
        if not result.is_contradictory:
            return ()
        return self.explain_all(result.constraint_ids)
