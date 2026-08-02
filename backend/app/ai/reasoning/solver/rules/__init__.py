"""Hard-constraint rule modules.

`DEFAULT_RULE_MODULES` is the whole deduction ruleset for this stage: the fixed
composition, what a seat knows about itself, what it knows about its allies, and
the Day-0 victim. Later stages append modules here rather than editing the
solver.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.ai.reasoning.solver.rules.allies import AllyKnowledgeRuleModule
from app.ai.reasoning.solver.rules.base import BaseRuleModule, RuleModule
from app.ai.reasoning.solver.rules.first_victim import FirstVictimRuleModule
from app.ai.reasoning.solver.rules.role_counts import RoleCountRuleModule
from app.ai.reasoning.solver.rules.self_knowledge import SelfKnowledgeRuleModule


def default_rule_modules() -> Sequence[RuleModule]:
    """Fresh instances: modules accumulate their own explanation table."""
    return (
        RoleCountRuleModule(),
        SelfKnowledgeRuleModule(),
        AllyKnowledgeRuleModule(),
        FirstVictimRuleModule(),
    )


__all__ = [
    "AllyKnowledgeRuleModule",
    "BaseRuleModule",
    "FirstVictimRuleModule",
    "RoleCountRuleModule",
    "RuleModule",
    "SelfKnowledgeRuleModule",
    "default_rule_modules",
]
