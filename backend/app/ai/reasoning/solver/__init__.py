"""LLM-free role deduction over the fixed 17-player composition.

Answers three questions a werewolf player asks constantly and a language model
answers unreliably: is this arrangement possible, is that seat's role settled,
and does this assumption force that conclusion. No model is called anywhere
below this package, and the same board and viewpoint always give the same
answer.

Layout: `backend` holds the contract and the constraint IR, `z3_backend` the
only place that knows about z3, `rules/` one deduction rule per module,
`queries` the cached facade AI code holds, `explanations` the code-derived
reasons behind an impossibility.
"""

from app.ai.reasoning.solver.backend import (
    ConstraintBackend,
    ContradictionResult,
    Hypothesis,
    LabelledConstraint,
    RoleCountIs,
    RoleIs,
    RoleIsNot,
    RoleIsOneOf,
    SolverStats,
    WorldModel,
    all_of,
    has_role,
    not_role,
    role_count,
)
from app.ai.reasoning.solver.builder import ConstraintBuilder
from app.ai.reasoning.solver.cache import QueryKey, SolverCache
from app.ai.reasoning.solver.explanations import ExplanationRegistry
from app.ai.reasoning.solver.queries import RoleSolver, build_solver
from app.ai.reasoning.solver.rules import RuleModule, default_rule_modules
from app.ai.reasoning.solver.z3_backend import (
    DEFAULT_MODEL_LIMIT,
    MAX_MODEL_LIMIT,
    ROLE_ORDER,
    Z3ConstraintBackend,
)

__all__ = [
    "DEFAULT_MODEL_LIMIT",
    "MAX_MODEL_LIMIT",
    "ROLE_ORDER",
    "ConstraintBackend",
    "ConstraintBuilder",
    "ContradictionResult",
    "ExplanationRegistry",
    "Hypothesis",
    "LabelledConstraint",
    "QueryKey",
    "RoleCountIs",
    "RoleIs",
    "RoleIsNot",
    "RoleIsOneOf",
    "RoleSolver",
    "RuleModule",
    "SolverCache",
    "SolverStats",
    "WorldModel",
    "Z3ConstraintBackend",
    "all_of",
    "build_solver",
    "default_rule_modules",
    "has_role",
    "not_role",
    "role_count",
]
