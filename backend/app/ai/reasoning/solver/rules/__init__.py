"""Hard-constraint rule modules.

One deduction rule per module, assembled here. Adding a new death cause or a new
role means a new file and one line in `default_rule_modules` -- never a change to
the solver, which is the property that has to survive the next few stages.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.ai.reasoning.solver.rules.allies import AllyKnowledgeRuleModule
from app.ai.reasoning.solver.rules.base import BaseRuleModule, RuleModule
from app.ai.reasoning.solver.rules.claimed_divination import ClaimedDivinationRuleModule
from app.ai.reasoning.solver.rules.first_victim import FirstVictimRuleModule
from app.ai.reasoning.solver.rules.game_running import GameRunningRuleModule
from app.ai.reasoning.solver.rules.night import (
    AttackRuleModule,
    DeathRuleModule,
    FoxCurseRuleModule,
    GuardRuleModule,
)
from app.ai.reasoning.solver.rules.own_results import OwnAbilityResultRuleModule
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
        ClaimedDivinationRuleModule(),
        OwnAbilityResultRuleModule(),
        GameRunningRuleModule(),
    )


# What follows from the rules by inference rather than by reading the record:
# the curse applied to a claimant's own account, and the game still running.
# The board memo shows their consequences as derived conclusions, which each
# seat notices at its own pace (`app/ai/reasoning/skill.py`).
DERIVED_RULE_MODULE_IDS = frozenset(
    {ClaimedDivinationRuleModule.module_id, GameRunningRuleModule.module_id}
)


def plain_rule_modules() -> Sequence[RuleModule]:
    """The default rules without the derived ones: what the record alone settles."""
    return tuple(
        module
        for module in default_rule_modules()
        if module.module_id not in DERIVED_RULE_MODULE_IDS
    )


__all__ = [
    "AllyKnowledgeRuleModule",
    "AttackRuleModule",
    "BaseRuleModule",
    "ClaimedDivinationRuleModule",
    "DeathRuleModule",
    "FirstVictimRuleModule",
    "DERIVED_RULE_MODULE_IDS",
    "FoxCurseRuleModule",
    "GameRunningRuleModule",
    "GuardRuleModule",
    "OwnAbilityResultRuleModule",
    "RoleCountRuleModule",
    "RuleModule",
    "SelfKnowledgeRuleModule",
    "default_rule_modules",
    "plain_rule_modules",
]
