"""The solver contract: constraints, hypotheses, and what may be asked of them.

Kept deliberately backend-agnostic. Rules emit a small declarative constraint
IR, and a `ConstraintBackend` translates it -- so the z3 dependency stays behind
one module and a later swap (weighted model counting, a hand-rolled propagator)
does not touch a single rule.

Four distinct questions, which game reasoning routinely conflates:

* `is_possible`  -- could the world be like this?
* `is_forced`    -- must it be?
* `implies`      -- does this premise settle that conclusion?
* `representative_models` -- show me a few worlds that fit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from app.engine.roles import RoleName

# -- constraint IR --------------------------------------------------------


@dataclass(frozen=True)
class RoleIs:
    player_id: str
    role: RoleName


@dataclass(frozen=True)
class RoleIsNot:
    player_id: str
    role: RoleName


@dataclass(frozen=True)
class RoleIsOneOf:
    player_id: str
    roles: tuple[RoleName, ...]


@dataclass(frozen=True)
class RoleCountIs:
    role: RoleName
    count: int


@dataclass(frozen=True)
class Unsatisfiable:
    """An assumption the observations already contradict.

    Some suppositions are not about who holds what -- "this claimant disclosed
    everything" is refuted by counting their published results, not by a role
    assignment. Expressing that as a constraint keeps the refusal inside the
    solver, so it shows up in the unsat core with its own explanation instead of
    being a special case the caller has to remember to check.
    """

    reason: str


Constraint = RoleIs | RoleIsNot | RoleIsOneOf | RoleCountIs | Unsatisfiable


@dataclass(frozen=True)
class LabelledConstraint:
    """A constraint plus why it exists.

    The id and explanation are what let an UNSAT answer say which rule made the
    world impossible instead of just reporting failure.
    """

    constraint_id: str
    constraint: Constraint
    explanation: str
    module_id: str = ""


# -- hypotheses -----------------------------------------------------------


@dataclass(frozen=True)
class Hypothesis:
    """A conjunction of claims, optionally with claims that must NOT hold."""

    claims: tuple[Constraint, ...] = ()
    excluded: tuple[Constraint, ...] = ()
    label: str = ""

    def __and__(self, other: Hypothesis) -> Hypothesis:
        return Hypothesis(
            claims=self.claims + other.claims,
            excluded=self.excluded + other.excluded,
            label=" & ".join(part for part in (self.label, other.label) if part),
        )

    @property
    def is_empty(self) -> bool:
        return not self.claims and not self.excluded

    def cache_key(self) -> str:
        parts = [render_constraint(claim) for claim in self.claims]
        parts += [f"!{render_constraint(claim)}" for claim in self.excluded]
        return ";".join(sorted(parts))


def has_role(player_id: str, role: RoleName, label: str = "") -> Hypothesis:
    return Hypothesis(
        claims=(RoleIs(player_id, role),), label=label or f"{player_id}={role.value}"
    )


def not_role(player_id: str, role: RoleName, label: str = "") -> Hypothesis:
    return Hypothesis(
        claims=(RoleIsNot(player_id, role),), label=label or f"{player_id}!={role.value}"
    )


def role_count(role: RoleName, count: int, label: str = "") -> Hypothesis:
    return Hypothesis(
        claims=(RoleCountIs(role, count),), label=label or f"#{role.value}={count}"
    )


def all_of(*hypotheses: Hypothesis) -> Hypothesis:
    combined = Hypothesis()
    for hypothesis in hypotheses:
        combined = combined & hypothesis
    return combined


def render_constraint(constraint: Constraint) -> str:
    """Short, stable text for a constraint. Doubles as its cache-key fragment."""
    match constraint:
        case RoleIs(player_id=pid, role=role):
            return f"{pid}={role.value}"
        case RoleIsNot(player_id=pid, role=role):
            return f"{pid}!={role.value}"
        case RoleIsOneOf(player_id=pid, roles=roles):
            return f"{pid}in{{{','.join(sorted(r.value for r in roles))}}}"
        case RoleCountIs(role=role, count=count):
            return f"#{role.value}={count}"
        case Unsatisfiable(reason=reason):
            return f"false({reason})"
    raise TypeError(f"unrenderable constraint {constraint!r}")


# -- results --------------------------------------------------------------


@dataclass(frozen=True)
class WorldModel:
    """One complete role assignment consistent with everything asserted."""

    roles: tuple[tuple[str, RoleName], ...]

    def role_of(self, player_id: str) -> RoleName:
        return dict(self.roles)[player_id]

    def players_with(self, role: RoleName) -> tuple[str, ...]:
        return tuple(pid for pid, assigned in self.roles if assigned == role)


@dataclass(frozen=True)
class ContradictionResult:
    """Why a set of assumptions cannot hold.

    `constraint_ids` is an unsat core: the subset of constraints that are
    actually in conflict, not every constraint in play. That is what makes the
    answer usable -- "占い師はちょうど1人 / p3のCOを真と仮定 / p7のCOを真と仮定"
    names the contradiction, while a dump of thirty constraints does not. The
    wording comes from the modules that declared them, so no model is inventing
    the reason.
    """

    is_contradictory: bool
    constraint_ids: tuple[str, ...] = ()
    explanations: tuple[str, ...] = ()


class Certainty(StrEnum):
    """How settled a question is -- four states, routinely collapsed into two.

    `CONDITIONAL` is the one that matters in play: "X is a wolf if this seer is
    real" is neither knowledge nor speculation, and a village that cannot say
    which of the two it has cannot decide what to do about it.
    """

    IMPOSSIBLE = "impossible"
    POSSIBLE = "possible"
    CERTAIN = "certain"
    CONDITIONAL = "conditional"


@dataclass
class SolverStats:
    checks: int = 0
    cache_hits: int = 0
    models_enumerated: int = 0
    constraint_ids: list[str] = field(default_factory=list)


class ConstraintBackend(Protocol):
    def is_possible(self, query: Hypothesis) -> bool: ...

    def is_forced(self, query: Hypothesis) -> bool: ...

    def implies(self, premise: Hypothesis, conclusion: Hypothesis) -> bool: ...

    def representative_models(
        self, assumptions: Sequence[Hypothesis], limit: int
    ) -> list[WorldModel]: ...
