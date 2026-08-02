"""Public-fact foundation for AI reasoning.

`facts` projects the engine state into the only view the AI side may treat as
established; `validation` rejects or deterministically repairs AI output that
contradicts it; `summaries` renders the factual half of a day summary without
an LLM. Each rule lives in one small module so later reasoning work can add or
remove rules without touching a single large file.
"""

from app.ai.reasoning.facts import (
    MEDIUM_RESULT,
    RESULT_TYPES,
    SEER_RESULT,
    PublicCoFact,
    PublicExecutionFact,
    PublicFactLedger,
    PublicPlayerFact,
    PublicResultFact,
    PublicVoteFact,
    ResultVersion,
    RoleClaimState,
    SpeechEvent,
    mentions_player,
)
from app.ai.reasoning.summaries import (
    compose_day_summary,
    render_public_fact_summary,
    split_day_summary,
)
from app.ai.reasoning.validation import (
    MemoValidation,
    ResultValidation,
    TargetResolution,
    ValidationIssue,
    ValidationLog,
    VotePlanMismatch,
    detect_vote_plan_mismatch,
    resolve_target,
    validate_discussion_output,
    validate_public_result_claim,
    validate_public_result_claims,
    validate_reasoning_memo,
)

__all__ = [
    "MEDIUM_RESULT",
    "RESULT_TYPES",
    "SEER_RESULT",
    "MemoValidation",
    "PublicCoFact",
    "PublicExecutionFact",
    "PublicFactLedger",
    "PublicPlayerFact",
    "PublicResultFact",
    "PublicVoteFact",
    "ResultValidation",
    "ResultVersion",
    "RoleClaimState",
    "SpeechEvent",
    "TargetResolution",
    "ValidationIssue",
    "ValidationLog",
    "VotePlanMismatch",
    "compose_day_summary",
    "detect_vote_plan_mismatch",
    "mentions_player",
    "render_public_fact_summary",
    "resolve_target",
    "split_day_summary",
    "validate_discussion_output",
    "validate_public_result_claim",
    "validate_public_result_claims",
    "validate_reasoning_memo",
]
