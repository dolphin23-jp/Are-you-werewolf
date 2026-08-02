"""Per-seat beliefs with traceable provenance.

`state` holds what one AI believes and the evidence behind each part of it,
`corrections` checks claimed factual corrections against the ledger, and
`engine` derives evidence from public facts, applies corrections, and rebuilds
every score from whatever evidence is still standing.

The invariant the package exists for: no suspicion outlives the fact it rested
on. Retracting evidence necessarily moves the score, so "I accept I misremembered
your vote, but I still suspect you for that reason" cannot be represented.
"""

from app.ai.reasoning.belief.corrections import (
    CorrectionKind,
    CorrectionStatus,
    CorrectionVerdict,
    FactCorrection,
    claim_fact_id,
    execution_fact_id,
    parse_fact_corrections,
    verify,
    vote_fact_id,
)
from app.ai.reasoning.belief.engine import (
    CONTESTED_CLAIM_WEIGHT,
    PUBLISHED_BLACK_WEIGHT,
    PUBLISHED_WHITE_WEIGHT,
    TRUST_STEP,
    BeliefEngine,
    CorrectionOutcome,
)
from app.ai.reasoning.belief.state import (
    HARD_CONFIRMED_SCORE,
    HARD_EXCLUDED_SCORE,
    EvidenceRecord,
    PlayerBeliefState,
    RankedHypothesis,
    is_hard,
)

__all__ = [
    "CONTESTED_CLAIM_WEIGHT",
    "HARD_CONFIRMED_SCORE",
    "HARD_EXCLUDED_SCORE",
    "PUBLISHED_BLACK_WEIGHT",
    "PUBLISHED_WHITE_WEIGHT",
    "TRUST_STEP",
    "BeliefEngine",
    "CorrectionKind",
    "CorrectionOutcome",
    "CorrectionStatus",
    "CorrectionVerdict",
    "EvidenceRecord",
    "FactCorrection",
    "PlayerBeliefState",
    "RankedHypothesis",
    "claim_fact_id",
    "execution_fact_id",
    "is_hard",
    "parse_fact_corrections",
    "verify",
    "vote_fact_id",
]
