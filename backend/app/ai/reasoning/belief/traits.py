"""Cognitive traits: what makes two AIs disagree about the same evidence.

Tone alone is not a personality. Two players who phrase things differently and
then vote identically are one player with two voices, and a table of those is
what makes a game feel scripted. So traits act where decisions are actually
made -- on the weight each kind of evidence carries -- rather than on wording.

The hard line: traits scale *soft* evidence and nothing else. A seat the solver
has settled cannot be argued out of by a stubborn player or into by a conformist
one. Personality decides what you find persuasive, not what is true.

Every weight is a plain constant in one table, so the whole scale can be read,
reviewed and tuned in one place instead of being scattered through derivations.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

# Category -> how strongly each trait bears on it. Scales soft weight only.
EVIDENCE_CATEGORIES = (
    "published_black",
    "published_white",
    "contested_claim",
    "voted_for_cleared",
    "voted_for_wolf",
    "majority_pressure",
    "accusation",
    "misremembered_vote",
)


@dataclass(frozen=True)
class CognitiveTraits:
    """Nine dials, all in roughly 0.0-2.0 with 1.0 as unremarkable."""

    evidence_weight: float = 1.0
    conformity: float = 1.0
    stubbornness: float = 1.0
    authority_bias: float = 1.0
    contrarianism: float = 0.0
    emotional_susceptibility: float = 1.0
    risk_aversion: float = 1.0
    trust_sensitivity: float = 1.0
    novelty_preference: float = 1.0

    def scale_for(self, category: str) -> float:
        """How much this player is moved by one kind of evidence.

        Never negative: a contrarian is hard to sway by the majority, but is not
        pushed into automatically opposing it -- "everyone thinks X so X is
        false" is just conformity with the sign flipped.
        """
        if category == "majority_pressure":
            return max(0.0, self.conformity - self.contrarianism)
        if category in ("published_black", "published_white"):
            return self.evidence_weight * (0.5 + 0.5 * self.authority_bias)
        if category == "accusation":
            return self.emotional_susceptibility
        return self.evidence_weight

    @property
    def switching_cost(self) -> float:
        """How much better a new candidate must look before abandoning the old one."""
        return 0.25 * self.stubbornness

    @property
    def minority_review_band(self) -> float:
        """How far below the leader a hypothesis still gets a serious second look.

        Contrarians and the authority-sceptical keep more alternatives open --
        which is the useful half of contrarianism, unlike reflexive opposition.
        """
        return 0.5 + 0.5 * self.contrarianism + 0.25 * max(0.0, 2.0 - self.authority_bias)

    @property
    def commitment_threshold(self) -> float:
        """Confidence a risk-averse player wants before treating a lead as settled."""
        return min(0.95, 0.3 * self.risk_aversion)


# Named profiles rather than free-floating numbers, so a transcript can say
# "this was the sceptic" instead of listing nine floats.
TRAIT_PROFILES: dict[str, CognitiveTraits] = {
    "evidence_driven": CognitiveTraits(
        evidence_weight=1.6, conformity=0.6, stubbornness=0.7, authority_bias=1.0
    ),
    "conformist": CognitiveTraits(
        evidence_weight=0.8, conformity=1.8, stubbornness=0.6, authority_bias=1.5
    ),
    "sceptic": CognitiveTraits(
        evidence_weight=1.2,
        conformity=0.5,
        stubbornness=1.3,
        authority_bias=0.4,
        contrarianism=1.0,
    ),
    "stubborn": CognitiveTraits(
        evidence_weight=1.0, conformity=0.7, stubbornness=1.9, authority_bias=0.9
    ),
    "cautious": CognitiveTraits(
        evidence_weight=1.1,
        conformity=1.1,
        stubbornness=1.0,
        authority_bias=1.1,
        risk_aversion=1.8,
    ),
    "impulsive": CognitiveTraits(
        evidence_weight=0.9,
        conformity=1.2,
        stubbornness=0.4,
        emotional_susceptibility=1.7,
        risk_aversion=0.4,
        novelty_preference=1.6,
    ),
}

DEFAULT_PROFILE = "evidence_driven"


def assign_traits(
    player_ids: Sequence[str], seed: int | None = None
) -> dict[str, CognitiveTraits]:
    """Deterministic per-seed spread of profiles across the table.

    Shuffled rather than round-robin so the same seat is not always the sceptic,
    and seeded so a game can be replayed exactly.
    """
    names = sorted(TRAIT_PROFILES)
    rng = random.Random(f"{seed}:cognitive-traits")
    pool = [names[index % len(names)] for index in range(len(player_ids))]
    rng.shuffle(pool)
    return {
        player_id: TRAIT_PROFILES[profile]
        for player_id, profile in zip(sorted(player_ids), pool, strict=True)
    }


def profile_name(traits: CognitiveTraits) -> str:
    for name, profile in TRAIT_PROFILES.items():
        if profile == traits:
            return name
    return "custom"
