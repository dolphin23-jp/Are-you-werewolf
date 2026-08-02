"""What one AI currently believes, and why.

The "why" is the load-bearing part. A suspicion with no traceable origin cannot
be withdrawn when its origin turns out to be wrong -- which is how an AI ends up
conceding "you're right, I misremembered your vote" and then keeping the
conclusion that rested on it. Every score here is the sum of named, retractable
evidence, so retracting the evidence necessarily moves the score.

Scores are log-odds-ish accumulations, not probabilities. Nothing about this
board justifies reporting that someone is 73.4% likely to be a wolf, and
formatting a sum as a percentage would only invent that authority.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace

from app.ai.reasoning.solver.backend import Certainty

# Hard solver conclusions sit outside the soft scale on purpose. Personality
# scales soft weights (PR8), and no amount of scaling may reorder a seat the
# rules have already settled.
HARD_CONFIRMED_SCORE = 1e9
HARD_EXCLUDED_SCORE = -1e9

# Gap between the top two candidates at which a player is fully committed.
CONFIDENCE_SPREAD = 2.0


def is_hard(score: float) -> bool:
    return abs(score) >= HARD_CONFIRMED_SCORE


def tiebreak(salt: str, player_id: str) -> float:
    """A stable per-seat ordering for candidates the evidence cannot separate.

    Early on, most greys carry identical (zero) evidence. Breaking those ties by
    seat number makes every AI name the same person, which looks like consensus
    and is actually an artefact of sorting. Which grey you happen to distrust
    with no evidence is genuinely arbitrary, so it varies per seat -- seeded, so
    a game still replays exactly. Strictly a secondary key: it can never move a
    candidate the evidence has separated.
    """
    digest = hashlib.sha256(f"{salt}|{player_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") / 0xFFFFFFFF


@dataclass(frozen=True)
class EvidenceRecord:
    """One reason someone believes something, with what it rests on.

    `source_event_ids` is the retraction handle: when a fact turns out to be
    wrong, everything built on it is found through these and deactivated. A
    record is never deleted -- an inactive one is the audit trail showing which
    argument was withdrawn.
    """

    evidence_id: str
    subject_id: str | None
    category: str
    source_event_ids: tuple[str, ...]
    weight: float
    explanation: str
    active: bool = True

    def deactivated(self, reason: str = "") -> EvidenceRecord:
        detail = f"{self.explanation}（撤回: {reason}）" if reason else self.explanation
        return replace(self, active=False, explanation=detail)


@dataclass(frozen=True)
class RankedHypothesis:
    """A candidate reading of the board, with how settled it is.

    `certainty` comes from the solver and outranks `score`: a hypothesis the
    rules have excluded stays excluded however attractive the soft evidence.
    """

    hypothesis_id: str
    label: str
    score: float
    certainty: Certainty
    supporting_evidence_ids: tuple[str, ...] = ()


@dataclass
class PlayerBeliefState:
    """One seat's private picture of the board. Never shared between seats."""

    player_id: str
    perspective_id: str
    wolf_scores: dict[str, float] = field(default_factory=dict)
    fox_scores: dict[str, float] = field(default_factory=dict)
    claim_trust: dict[str, float] = field(default_factory=dict)
    source_trust: dict[str, float] = field(default_factory=dict)
    active_hypotheses: list[RankedHypothesis] = field(default_factory=list)
    # subject id -> the evidence ids currently supporting a view of that seat.
    evidence_links: dict[str, list[str]] = field(default_factory=dict)
    current_execution_target: str | None = None
    alternative_target: str | None = None
    confidence: float = 0.0

    def reasons_for(self, subject_id: str) -> tuple[str, ...]:
        return tuple(self.evidence_links.get(subject_id, ()))

    def ranked_suspects(self) -> tuple[tuple[str, float], ...]:
        """Most suspicious first. Ties break per-seat rather than by seat number
        -- see `tiebreak` for why that matters."""
        salt = f"{self.player_id}:{self.perspective_id}"
        return tuple(
            sorted(
                self.wolf_scores.items(),
                key=lambda item: (-item[1], tiebreak(salt, item[0]), item[0]),
            )
        )
