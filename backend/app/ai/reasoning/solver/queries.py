"""The object AI code actually holds: one board, one viewpoint, cached answers.

`RoleSolver` assembles the rule modules for a perspective, hands the resulting
constraints to a backend, and memoizes every question. It satisfies
`ConstraintBackend` itself, so a caller can be handed either one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar, cast

from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import Perspective
from app.ai.reasoning.solver import assumptions as assumption_rules
from app.ai.reasoning.solver.assumptions import Assumption
from app.ai.reasoning.solver.backend import (
    Certainty,
    ContradictionResult,
    Hypothesis,
    SolverStats,
    WorldModel,
    has_role,
)
from app.ai.reasoning.solver.builder import ConstraintBuilder
from app.ai.reasoning.solver.cache import QueryKey, SolverCache
from app.ai.reasoning.solver.explanations import ExplanationRegistry
from app.ai.reasoning.solver.rules import RuleModule, default_rule_modules
from app.ai.reasoning.solver.z3_backend import (
    DEFAULT_MODEL_LIMIT,
    ROLE_ORDER,
    Z3ConstraintBackend,
)
from app.engine.roles import RoleName

_T = TypeVar("_T")


class RoleSolver:
    def __init__(
        self,
        observations: ObservationSet,
        perspective: Perspective,
        *,
        rules: Sequence[RuleModule] | None = None,
        cache: SolverCache | None = None,
        assumptions: Sequence[Assumption] = (),
    ) -> None:
        self.observations = observations
        self.perspective = perspective
        self.assumptions = tuple(assumptions)
        self._modules = tuple(rules if rules is not None else default_rule_modules())
        builder = ConstraintBuilder()
        for module in self._modules:
            module.add_hard_constraints(builder, perspective, observations)
        # Behavioural suppositions ride on top of the hard rules and are never
        # folded into them: the same board with different assumptions is a
        # different solver, and the assumption signature says which.
        builder.constraints.extend(assumption_rules.expand(self.assumptions, observations))
        self._signature = builder.signature()
        self._stats = SolverStats(constraint_ids=list(builder.constraint_ids()))
        self._backend = Z3ConstraintBackend(
            observations.player_ids, builder.constraints, stats=self._stats
        )
        self.explanations = ExplanationRegistry(builder.constraints)
        self._cache = cache if cache is not None else SolverCache()

    @property
    def stats(self) -> SolverStats:
        return self._stats

    def assuming(self, *assumptions: Assumption) -> RoleSolver:
        """A solver for the same board under additional suppositions.

        A new instance rather than mutable state: "what follows if this claim is
        real" must not leak back into the unconditional view, and the two need
        separate cache identities.
        """
        return RoleSolver(
            self.observations,
            self.perspective,
            rules=self._modules,
            cache=self._cache,
            assumptions=self.assumptions + assumptions,
        )

    @property
    def cache(self) -> SolverCache:
        return self._cache

    # -- ConstraintBackend --

    def is_possible(self, query: Hypothesis) -> bool:
        return self._cached("is_possible", (query,), lambda: self._backend.is_possible(query))

    def is_forced(self, query: Hypothesis) -> bool:
        return self._cached("is_forced", (query,), lambda: self._backend.is_forced(query))

    def implies(self, premise: Hypothesis, conclusion: Hypothesis) -> bool:
        return self._cached(
            "implies",
            (premise, conclusion),
            lambda: self._backend.implies(premise, conclusion),
        )

    def representative_models(
        self, assumptions: Sequence[Hypothesis], limit: int = DEFAULT_MODEL_LIMIT
    ) -> list[WorldModel]:
        return self._cached(
            f"models:{limit}",
            tuple(assumptions),
            lambda: self._backend.representative_models(assumptions, limit),
        )

    def contradiction(self, *hypotheses: Hypothesis) -> ContradictionResult:
        return self._cached(
            "contradiction", hypotheses, lambda: self._backend.contradiction(*hypotheses)
        )

    # -- certainty --

    def assess(self, query: Hypothesis, given: Hypothesis | None = None) -> Certainty:
        """Distinguish impossible / possible / certain / certain-only-if.

        The fourth is the one play actually turns on: "X is a wolf if this seer
        is real" is not knowledge, but it is not idle speculation either.
        """
        if given is None or given.is_empty:
            if not self.is_possible(query):
                return Certainty.IMPOSSIBLE
            return Certainty.CERTAIN if self.is_forced(query) else Certainty.POSSIBLE
        if not self.is_possible(given):
            return Certainty.IMPOSSIBLE
        if self.is_forced(query):
            return Certainty.CERTAIN
        if not self.implies(given, query):
            return (
                Certainty.POSSIBLE
                if self.is_possible(query & given)
                else Certainty.IMPOSSIBLE
            )
        return Certainty.CONDITIONAL

    def required_latent_roles(self) -> tuple[RoleName, ...]:
        """Roles that must be held by someone who never claimed them.

        A bluffer's story can be internally consistent and still demand that,
        say, the real seer is sitting quiet -- which is exactly the cost of the
        story, and worth being able to state.
        """
        required = []
        for role in ROLE_ORDER:
            constraints = assumption_rules.no_latent_constraints(role, self.observations)
            if not constraints:
                continue
            hypothesis = Hypothesis(
                claims=tuple(item.constraint for item in constraints),
                label=f"no latent {role.value}",
            )
            if not self.is_possible(hypothesis):
                required.append(role)
        return tuple(required)

    def explain_contradiction(self, *hypotheses: Hypothesis) -> ContradictionResult:
        """`contradiction`, but always answering in this board's own wording."""
        result = self.contradiction(*hypotheses)
        if not result.is_contradictory:
            return result
        return ContradictionResult(
            is_contradictory=True,
            constraint_ids=result.constraint_ids,
            explanations=tuple(
                self.explanations.explain(cid) or explanation
                for cid, explanation in zip(
                    result.constraint_ids, result.explanations, strict=True
                )
            ),
        )

    # -- convenience --

    def possible_roles(self, player_id: str) -> tuple[RoleName, ...]:
        """Every role this seat could still hold from this viewpoint."""
        return tuple(
            role for role in ROLE_ORDER if self.is_possible(has_role(player_id, role))
        )

    def certain_role(self, player_id: str) -> RoleName | None:
        """The seat's role when only one remains possible, else None.

        Stops at the first role that fits and asks whether it is forced, rather
        than pricing all eight: an unsettled seat -- the common case -- costs two
        solver calls instead of a full sweep.
        """
        for role in ROLE_ORDER:
            if self.is_possible(has_role(player_id, role)):
                return role if self.is_forced(has_role(player_id, role)) else None
        return None

    # -- internals --

    def _cached(
        self, kind: str, hypotheses: Sequence[Hypothesis], compute: Callable[[], _T]
    ) -> _T:
        key = QueryKey(
            board_version=self.observations.board_version,
            perspective_id=(
                f"{self.perspective.perspective_id}"
                f"|{assumption_rules.signature(self.assumptions)}"
            ),
            constraint_signature=self._signature,
            query_kind=kind,
            assumptions=tuple(hypothesis.cache_key() for hypothesis in hypotheses),
        )
        hit = self._cache.get(key)
        if hit is not None:
            self._stats.cache_hits += 1
            return cast("_T", hit)
        value = compute()
        self._cache.put(key, value)
        return value


def build_solver(
    observations: ObservationSet,
    perspective: Perspective,
    *,
    rules: Sequence[RuleModule] | None = None,
    cache: SolverCache | None = None,
    assumptions: Sequence[Assumption] = (),
) -> RoleSolver:
    return RoleSolver(
        observations, perspective, rules=rules, cache=cache, assumptions=assumptions
    )
