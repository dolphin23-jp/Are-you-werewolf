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
    MAJORITY_PRESSURE_WEIGHT,
    PUBLISHED_BLACK_WEIGHT,
    PUBLISHED_WHITE_WEIGHT,
    TRUST_STEP,
    BeliefEngine,
    CorrectionOutcome,
)
from app.ai.reasoning.belief.ranking import (
    RANK_LABELS_JA,
    HypothesisRank,
    RankedView,
    rank_hypotheses,
    summarise,
)
from app.ai.reasoning.belief.state import (
    HARD_CONFIRMED_SCORE,
    HARD_EXCLUDED_SCORE,
    EvidenceRecord,
    PlayerBeliefState,
    RankedHypothesis,
    is_hard,
)
from app.ai.reasoning.belief.story import (
    BETRAYAL_COST,
    DeceptionState,
    StoryStatus,
    deception_state_for,
    private_solver,
    refresh_story,
    story_is_possible,
    story_solver,
)
from app.ai.reasoning.belief.traits import (
    DEFAULT_PROFILE,
    TRAIT_PROFILES,
    CognitiveTraits,
    assign_traits,
    profile_name,
)

__all__ = [
    "BETRAYAL_COST",
    "BeliefEngine",
    "CONTESTED_CLAIM_WEIGHT",
    "CognitiveTraits",
    "CorrectionKind",
    "CorrectionOutcome",
    "CorrectionStatus",
    "CorrectionVerdict",
    "DEFAULT_PROFILE",
    "DeceptionState",
    "EvidenceRecord",
    "FactCorrection",
    "HARD_CONFIRMED_SCORE",
    "HARD_EXCLUDED_SCORE",
    "HypothesisRank",
    "MAJORITY_PRESSURE_WEIGHT",
    "PUBLISHED_BLACK_WEIGHT",
    "PUBLISHED_WHITE_WEIGHT",
    "PlayerBeliefState",
    "RANK_LABELS_JA",
    "RankedHypothesis",
    "RankedView",
    "StoryStatus",
    "TRAIT_PROFILES",
    "TRUST_STEP",
    "assign_traits",
    "claim_fact_id",
    "deception_state_for",
    "execution_fact_id",
    "is_hard",
    "parse_fact_corrections",
    "private_solver",
    "profile_name",
    "rank_hypotheses",
    "refresh_story",
    "story_is_possible",
    "story_solver",
    "summarise",
    "verify",
    "vote_fact_id",
]
