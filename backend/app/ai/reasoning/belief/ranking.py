"""Ranking hypotheses in words a player would actually use.

Not percentages. Nothing about a mid-game board justifies "73.4% likely", and
printing a number to three decimals manufactures an authority the reasoning does
not have. Five bands is what the evidence can actually support:

    本線 / 有力対抗 / 薄い / 論理上のみ / 不可能

The bottom band matters as much as the top. "Logically possible but nobody
argues it" is a real state, and collapsing it into "unlikely" is how a village
loses track of the line it should have checked.

Hard first: anything the solver has excluded is `IMPOSSIBLE` regardless of how
much soft evidence points at it, and anything the solver has settled is `MAIN`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from app.ai.reasoning.belief.state import RankedHypothesis
from app.ai.reasoning.belief.traits import CognitiveTraits
from app.ai.reasoning.solver.backend import Certainty

# A hypothesis with essentially no evidence behind it is "logically only".
ARGUED_FLOOR = 0.2


class HypothesisRank(StrEnum):
    MAIN = "main"
    STRONG_ALTERNATIVE = "strong_alternative"
    THIN = "thin"
    LOGICAL_ONLY = "logical_only"
    IMPOSSIBLE = "impossible"


RANK_LABELS_JA: dict[HypothesisRank, str] = {
    HypothesisRank.MAIN: "本線",
    HypothesisRank.STRONG_ALTERNATIVE: "有力対抗",
    HypothesisRank.THIN: "薄い",
    HypothesisRank.LOGICAL_ONLY: "論理上のみ",
    HypothesisRank.IMPOSSIBLE: "不可能",
}


@dataclass(frozen=True)
class RankedView:
    hypothesis: RankedHypothesis
    rank: HypothesisRank

    @property
    def label_ja(self) -> str:
        return RANK_LABELS_JA[self.rank]


def rank_hypotheses(
    hypotheses: Sequence[RankedHypothesis], traits: CognitiveTraits
) -> tuple[RankedView, ...]:
    """Band a scored list. Traits widen or narrow the second band only."""
    if not hypotheses:
        return ()
    live = [item for item in hypotheses if item.certainty is not Certainty.IMPOSSIBLE]
    excluded = [
        RankedView(hypothesis=item, rank=HypothesisRank.IMPOSSIBLE)
        for item in hypotheses
        if item.certainty is Certainty.IMPOSSIBLE
    ]
    if not live:
        return tuple(excluded)

    ordered = sorted(live, key=lambda item: (-item.score, item.hypothesis_id))
    top = ordered[0].score
    band = traits.minority_review_band
    views: list[RankedView] = []
    for index, item in enumerate(ordered):
        if item.certainty is Certainty.CERTAIN:
            rank = HypothesisRank.MAIN
        elif index == 0:
            rank = HypothesisRank.MAIN
        elif item.score >= top - band:
            rank = HypothesisRank.STRONG_ALTERNATIVE
        elif item.score > ARGUED_FLOOR:
            rank = HypothesisRank.THIN
        else:
            rank = HypothesisRank.LOGICAL_ONLY
        views.append(RankedView(hypothesis=item, rank=rank))
    return tuple(views + excluded)


def summarise(views: Sequence[RankedView]) -> str:
    """One line per band, for a prompt or a transcript. Never a percentage."""
    by_rank: dict[HypothesisRank, list[str]] = {}
    for view in views:
        by_rank.setdefault(view.rank, []).append(view.hypothesis.label)
    return " / ".join(
        f"{RANK_LABELS_JA[rank]}: {'、'.join(labels)}"
        for rank, labels in by_rank.items()
        if rank is not HypothesisRank.IMPOSSIBLE
    )
