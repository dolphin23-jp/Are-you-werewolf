"""Hard-constraint rule modules.

One deduction rule per module, assembled here. Adding a new death cause or a new
role means a new file and one line in `default_rule_modules` -- never a change to
the solver, which is the property that has to survive the next few stages.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.ai.reasoning.solver.rules.allies import AllyKnowledgeRuleModule
from app.ai.reasoning.solver.rules.base import BaseRuleModule, RuleModule
from app.ai.reasoning.solver.rules.first_victim import FirstVictimRuleModule
from app.ai.reasoning.solver.rules.night import (
    AttackRuleModule,
    DeathRuleModule,
    FoxCurseRuleModule,
    GuardRuleModule,
)
from app.ai.reasoning.solver.rules.role_counts import RoleCountRuleModule
from app.ai.reasoning.solver.rules.self_knowledge import SelfKnowledgeRuleModule


def default_rule_modules() -> Sequence[RuleModule]:
    """Fresh instances: modules accumulate their own explanation table."""
    return (
        RoleCountRuleModule(),
        SelfKnowledgeRuleModule(),
        AllyKnowledgeRuleModule(),
        FirstVictimRuleModule(),
        DeathRuleModule(),
        AttackRuleModule(),
        GuardRuleModule(),
        FoxCurseRuleModule(),
    )


__all__ = [
    "AllyKnowledgeRuleModule",
    "AttackRuleModule",
    "BaseRuleModule",
    "DeathRuleModule",
    "FirstVictimRuleModule",
    "FoxCurseRuleModule",
    "GuardRuleModule",
    "RoleCountRuleModule",
    "RuleModule",
    "SelfKnowledgeRuleModule",
    "default_rule_modules",
]
