"""z3 encoding of the role-assignment problem.

One integer variable per seat, holding a role index. That is enough for every
question at this stage and keeps the search tiny -- 17 variables over 8 values
with a fixed multiset of counts.

Determinism matters more here than raw speed. `is_possible` and friends are
naturally deterministic, but model *enumeration* is not: which satisfying world
a solver happens to find first is an implementation detail, and a village AI
whose "representative worlds" shuffle between runs cannot be debugged. So
enumeration walks the seats in a fixed order and pins each to the smallest
feasible role, yielding the lexicographically smallest remaining model every
time.
"""

from __future__ import annotations

from collections.abc import Sequence

import z3

from app.ai.reasoning.solver.backend import (
    Constraint,
    ContradictionResult,
    Hypothesis,
    LabelledConstraint,
    RoleCountIs,
    RoleIs,
    RoleIsNot,
    RoleIsOneOf,
    SolverStats,
    Unsatisfiable,
    WorldModel,
    render_constraint,
)
from app.engine.roles import RoleName

# Stable index order, so an encoding is reproducible across processes.
ROLE_ORDER: tuple[RoleName, ...] = tuple(RoleName)
_ROLE_INDEX: dict[RoleName, int] = {role: index for index, role in enumerate(ROLE_ORDER)}

# Enumeration is bounded by the caller, but never unbounded: a request for
# "all worlds" on an unconstrained board is millions of assignments.
DEFAULT_MODEL_LIMIT = 8
MAX_MODEL_LIMIT = 64


class Z3ConstraintBackend:
    """Answers possibility/necessity questions about one board and viewpoint."""

    def __init__(
        self,
        player_ids: Sequence[str],
        constraints: Sequence[LabelledConstraint],
        *,
        stats: SolverStats | None = None,
    ) -> None:
        self._player_ids = tuple(player_ids)
        self._constraints = tuple(constraints)
        self._stats = stats or SolverStats()
        self._vars = {pid: z3.Int(f"role_{pid}") for pid in self._player_ids}
        self._solver = z3.Solver()
        self._solver.set("random_seed", 0)
        for var in self._vars.values():
            self._solver.add(var >= 0, var < len(ROLE_ORDER))
        # Each constraint hides behind its own literal, and every check asserts
        # all of them. It costs nothing on a problem this size and it is what
        # makes `unsat_core` able to name the handful of rules in conflict --
        # otherwise an impossible story can only be reported as "impossible".
        self._literals: dict[str, z3.BoolRef] = {}
        for labelled in self._constraints:
            literal = self._literal(labelled.constraint_id)
            self._solver.add(z3.Implies(literal, self._encode(labelled.constraint)))
        self._base_literals = tuple(self._literals.values())
        self._by_literal = {
            str(self._literal(item.constraint_id)): item for item in self._constraints
        }
        # "Does this board admit any world at all" is asked by every `is_forced`
        # call and can never change: the constraint set is fixed at construction.
        self._base_satisfiable: bool | None = None

    @property
    def stats(self) -> SolverStats:
        return self._stats

    @property
    def constraints(self) -> tuple[LabelledConstraint, ...]:
        return self._constraints

    # -- queries --

    def is_possible(self, query: Hypothesis) -> bool:
        return self._check(self._encode_hypothesis(query))

    def is_forced(self, query: Hypothesis) -> bool:
        """True when the board admits worlds and every one of them satisfies the
        query. A board with no worlds at all forces nothing -- reporting
        otherwise turns a contradiction into false certainty."""
        if not self._satisfiable():
            return False
        return not self._check([z3.Not(z3.And(*self._encode_hypothesis(query)))])

    def implies(self, premise: Hypothesis, conclusion: Hypothesis) -> bool:
        """Material implication: no world satisfies the premise and denies the
        conclusion. An impossible premise implies everything, so callers that
        care should check `is_possible(premise)` first."""
        assertions = list(self._encode_hypothesis(premise))
        assertions.append(z3.Not(z3.And(*self._encode_hypothesis(conclusion))))
        return not self._check(assertions)

    def contradiction(self, *hypotheses: Hypothesis) -> ContradictionResult:
        """Whether the assumptions can hold, and if not, which rules collide.

        Extra hypotheses get their own tracking literals too, so a core can
        point at "this claim is genuine" rather than only at the fixed rules.
        """
        extra: list[tuple[z3.BoolRef, LabelledConstraint]] = []
        self._solver.push()
        try:
            for index, hypothesis in enumerate(hypotheses):
                for claim_index, claim in enumerate(hypothesis.claims):
                    labelled = LabelledConstraint(
                        constraint_id=f"hypothesis:{index}:{claim_index}",
                        constraint=claim,
                        explanation=hypothesis.label or render_constraint(claim),
                        module_id="hypothesis",
                    )
                    literal = z3.Bool(f"h!{index}!{claim_index}")
                    self._solver.add(z3.Implies(literal, self._encode(claim)))
                    extra.append((literal, labelled))
                for claim in hypothesis.excluded:
                    self._solver.add(z3.Not(self._encode(claim)))
            literals = [literal for literal, _ in extra]
            self._stats.checks += 1
            outcome = self._solver.check(*self._base_literals, *literals)
            if outcome == z3.sat:
                return ContradictionResult(is_contradictory=False)
            lookup = dict(self._by_literal)
            lookup.update({str(literal): labelled for literal, labelled in extra})
            core = [lookup[str(item)] for item in self._solver.unsat_core() if str(item) in lookup]
        finally:
            self._solver.pop()
        if not core:
            core = list(self._constraints)
        return ContradictionResult(
            is_contradictory=True,
            constraint_ids=tuple(item.constraint_id for item in core),
            explanations=tuple(item.explanation for item in core),
        )

    def representative_models(
        self, assumptions: Sequence[Hypothesis], limit: int = DEFAULT_MODEL_LIMIT
    ) -> list[WorldModel]:
        bounded = max(0, min(limit, MAX_MODEL_LIMIT))
        if bounded == 0:
            return []
        assertions: list[z3.BoolRef] = []
        for hypothesis in assumptions:
            assertions.extend(self._encode_hypothesis(hypothesis))

        found: list[WorldModel] = []
        self._solver.push()
        for assertion in assertions:
            self._solver.add(assertion)
        try:
            while len(found) < bounded:
                assignment = self._lexicographically_smallest_model()
                if assignment is None:
                    break
                found.append(
                    WorldModel(
                        roles=tuple(
                            (pid, ROLE_ORDER[assignment[pid]]) for pid in self._player_ids
                        )
                    )
                )
                self._stats.models_enumerated += 1
                self._solver.add(
                    z3.Not(
                        z3.And(
                            *[
                                self._vars[pid] == assignment[pid]
                                for pid in self._player_ids
                            ]
                        )
                    )
                )
        finally:
            self._solver.pop()
        return found

    # -- internals --

    def _literal(self, constraint_id: str) -> z3.BoolRef:
        if constraint_id not in self._literals:
            self._literals[constraint_id] = z3.Bool(f"c!{constraint_id}")
        return self._literals[constraint_id]

    def _check(self, assertions: Sequence[z3.BoolRef]) -> bool:
        self._stats.checks += 1
        # z3 is untyped, so the boundary between it and the rest of the codebase
        # is converted explicitly here rather than leaking Any upward.
        return bool(self._solver.check(*self._base_literals, *assertions) == z3.sat)

    def _satisfiable(self) -> bool:
        if self._base_satisfiable is None:
            self._base_satisfiable = self._check([])
        return self._base_satisfiable

    def _lexicographically_smallest_model(self) -> dict[str, int] | None:
        """Pin seats in order to their smallest feasible role.

        Costs one solver call per (seat, candidate role) pair, which on a
        17-seat board is trivial and buys reproducible output.
        """
        if not self._check([]):
            return None
        assignment: dict[str, int] = {}
        self._solver.push()
        try:
            for player_id in self._player_ids:
                var = self._vars[player_id]
                for index in range(len(ROLE_ORDER)):
                    if self._check([var == index]):
                        assignment[player_id] = index
                        self._solver.add(var == index)
                        break
                else:  # pragma: no cover - unreachable while the board is sat
                    return None
        finally:
            self._solver.pop()
        return assignment

    def _encode_hypothesis(self, hypothesis: Hypothesis) -> list[z3.BoolRef]:
        assertions = [self._encode(claim) for claim in hypothesis.claims]
        assertions += [z3.Not(self._encode(claim)) for claim in hypothesis.excluded]
        return assertions

    def _encode(self, constraint: Constraint) -> z3.BoolRef:
        match constraint:
            case RoleIs(player_id=pid, role=role):
                return self._var(pid) == _ROLE_INDEX[role]
            case RoleIsNot(player_id=pid, role=role):
                return self._var(pid) != _ROLE_INDEX[role]
            case RoleIsOneOf(player_id=pid, roles=roles):
                return z3.Or(*[self._var(pid) == _ROLE_INDEX[role] for role in roles])
            case Unsatisfiable():
                return z3.BoolVal(False)
            case RoleCountIs(role=role, count=count):
                index = _ROLE_INDEX[role]
                return (
                    z3.Sum([z3.If(self._var(pid) == index, 1, 0) for pid in self._player_ids])
                    == count
                )
        raise TypeError(f"unencodable constraint {constraint!r}")

    def _var(self, player_id: str) -> z3.ArithRef:
        try:
            return self._vars[player_id]
        except KeyError as exc:
            raise KeyError(f"unknown player {player_id}") from exc
