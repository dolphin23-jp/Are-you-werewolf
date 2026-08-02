"""Factual corrections, checked against the ledger rather than taken on trust.

"I never voted for Daiki, I voted for Yui" is a claim about the record, and the
record can answer it. So it is matched in code -- no model is asked whether a
player is telling the truth about something the engine already knows.

Two things follow from a *confirmed* correction. The evidence built on the wrong
fact has to go, and the person who caught it has earned a little credibility. A
*refuted* one costs credibility instead. Both are why corrections are verified
before they are applied, rather than after.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from app.ai.reasoning.facts import PublicFactLedger, mentions_player
from app.engine.roles import RoleName


class CorrectionKind(StrEnum):
    VOTE_TARGET = "vote_target"
    EXECUTED = "executed"
    ALIVE = "alive"
    ROLE_CLAIM = "role_claim"


class CorrectionStatus(StrEnum):
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    UNVERIFIABLE = "unverifiable"


@dataclass(frozen=True)
class FactCorrection:
    """Someone says the table has a fact wrong."""

    kind: CorrectionKind
    source_player_id: str
    subject_id: str
    day: int | None = None
    asserted: str | None = None
    denied: str | None = None
    role: RoleName | None = None

    @property
    def correction_id(self) -> str:
        parts = [self.kind.value, self.subject_id, str(self.day), str(self.denied)]
        return ":".join(parts)


@dataclass(frozen=True)
class CorrectionVerdict:
    correction: FactCorrection
    status: CorrectionStatus
    detail: str
    # Source-fact ids the correction invalidates, matched against evidence.
    invalidated_source_ids: tuple[str, ...] = ()

    @property
    def is_confirmed(self) -> bool:
        return self.status is CorrectionStatus.CONFIRMED


def vote_fact_id(voter_id: str, day: int, round_number: int, target_id: str) -> str:
    """Identifies one ballot *as recorded by whoever cited it*.

    The target is part of the id on purpose: that is what makes a mistaken
    citation distinguishable from a correct one, and lets a correction retract
    exactly the evidence that got it wrong.
    """
    return f"vote:{voter_id}:{day}:{round_number}:{target_id}"


def claim_fact_id(player_id: str, role: RoleName) -> str:
    return f"claim:{player_id}:{role.value}"


def execution_fact_id(player_id: str, day: int) -> str:
    return f"execution:{player_id}:{day}"


# -- verification --


def verify(correction: FactCorrection, ledger: PublicFactLedger) -> CorrectionVerdict:
    match correction.kind:
        case CorrectionKind.VOTE_TARGET:
            return _verify_vote(correction, ledger)
        case CorrectionKind.EXECUTED:
            return _verify_executed(correction, ledger)
        case CorrectionKind.ALIVE:
            return _verify_alive(correction, ledger)
        case CorrectionKind.ROLE_CLAIM:
            return _verify_claim(correction, ledger)
    raise TypeError(f"unknown correction kind {correction.kind}")


def _verify_vote(
    correction: FactCorrection, ledger: PublicFactLedger
) -> CorrectionVerdict:
    day = correction.day if correction.day is not None else ledger.day
    recorded = ledger.vote_of(correction.subject_id, day)
    if recorded is None:
        return CorrectionVerdict(
            correction,
            CorrectionStatus.UNVERIFIABLE,
            f"{day}日目の{correction.subject_id}の投票記録がありません。",
        )
    if correction.asserted is not None and recorded.target_id != correction.asserted:
        return CorrectionVerdict(
            correction,
            CorrectionStatus.REFUTED,
            (
                f"{day}日目の{correction.subject_id}の投票先は記録上"
                f"{recorded.target_id}で、{correction.asserted}ではありません。"
            ),
        )
    if correction.denied is not None and recorded.target_id == correction.denied:
        return CorrectionVerdict(
            correction,
            CorrectionStatus.REFUTED,
            f"{day}日目の{correction.subject_id}は記録上{correction.denied}へ投票しています。",
        )
    invalidated: tuple[str, ...] = ()
    if correction.denied is not None:
        # Every round of that day, since the citation may name any of them.
        invalidated = tuple(
            vote_fact_id(correction.subject_id, day, round_number, correction.denied)
            for round_number in sorted(
                {vote.round for vote in ledger.votes_on(day)} | {recorded.round}
            )
        )
    return CorrectionVerdict(
        correction,
        CorrectionStatus.CONFIRMED,
        (
            f"{day}日目の{correction.subject_id}の投票先は記録上{recorded.target_id}です。"
        ),
        invalidated_source_ids=invalidated,
    )


def _verify_executed(
    correction: FactCorrection, ledger: PublicFactLedger
) -> CorrectionVerdict:
    executed_day = ledger.execution_day(correction.subject_id)
    if correction.denied is not None or executed_day is None:
        if executed_day is None:
            return CorrectionVerdict(
                correction,
                CorrectionStatus.CONFIRMED,
                f"{correction.subject_id}は処刑されていません。",
                invalidated_source_ids=tuple(
                    execution_fact_id(correction.subject_id, day)
                    for day in range(ledger.day + 1)
                ),
            )
        return CorrectionVerdict(
            correction,
            CorrectionStatus.REFUTED,
            f"{correction.subject_id}は{executed_day}日目に処刑されています。",
        )
    if correction.day is not None and correction.day != executed_day:
        return CorrectionVerdict(
            correction,
            CorrectionStatus.REFUTED,
            f"{correction.subject_id}の処刑は{executed_day}日目です。",
        )
    return CorrectionVerdict(
        correction,
        CorrectionStatus.CONFIRMED,
        f"{correction.subject_id}は{executed_day}日目に処刑されています。",
    )


def _verify_alive(
    correction: FactCorrection, ledger: PublicFactLedger
) -> CorrectionVerdict:
    if not ledger.is_known(correction.subject_id):
        return CorrectionVerdict(
            correction, CorrectionStatus.UNVERIFIABLE, "そのプレイヤーは存在しません。"
        )
    alive = ledger.is_alive(correction.subject_id)
    asserted_alive = correction.asserted != "dead"
    if alive == asserted_alive:
        return CorrectionVerdict(
            correction,
            CorrectionStatus.CONFIRMED,
            f"{correction.subject_id}は{'生存' if alive else '死亡'}しています。",
        )
    return CorrectionVerdict(
        correction,
        CorrectionStatus.REFUTED,
        f"{correction.subject_id}は記録上{'生存' if alive else '死亡'}しています。",
    )


def _verify_claim(
    correction: FactCorrection, ledger: PublicFactLedger
) -> CorrectionVerdict:
    standing = ledger.claimed_role_of(correction.subject_id)
    if correction.denied is not None:
        denied_role = _as_role(correction.denied)
        if standing is not None and standing is denied_role:
            return CorrectionVerdict(
                correction,
                CorrectionStatus.REFUTED,
                f"{correction.subject_id}は現在{standing.value}COしています。",
            )
        invalidated = (
            (claim_fact_id(correction.subject_id, denied_role),)
            if denied_role is not None
            else ()
        )
        return CorrectionVerdict(
            correction,
            CorrectionStatus.CONFIRMED,
            f"{correction.subject_id}のそのCOは記録にありません。",
            invalidated_source_ids=invalidated,
        )
    asserted_role = _as_role(correction.asserted)
    if asserted_role is not None and standing is asserted_role:
        return CorrectionVerdict(
            correction,
            CorrectionStatus.CONFIRMED,
            f"{correction.subject_id}は{asserted_role.value}COしています。",
        )
    return CorrectionVerdict(
        correction,
        CorrectionStatus.REFUTED,
        f"{correction.subject_id}の現在のCOは{standing.value if standing else 'なし'}です。",
    )


def _as_role(value: str | None) -> RoleName | None:
    if not value:
        return None
    try:
        return RoleName(value)
    except ValueError:
        return None


# -- parsing --

_VOTE_DENIAL_RE = re.compile(
    r"(?P<label>[^、，。！？!?\s]{1,20}?)(?:には|に|へ)は?投票(?:して(?:い)?ませんし?|して(?:い)?ない|しなかった)"
)
_VOTE_ASSERTION_RE = re.compile(
    r"(?P<label>[^、，。！？!?\s]{1,20}?)(?:へ|に)投票(?:しました|した|しています|している)"
)
_DAY_RE = re.compile(r"(?P<day>\d+)日目")


def parse_fact_corrections(
    text: str, ledger: PublicFactLedger, speaker_id: str
) -> list[FactCorrection]:
    """Read explicit corrections out of ordinary Japanese.

    Deliberately narrow: only forms whose subject the ledger can actually check.
    A sentence this misses becomes an ordinary opinion, which is the safe
    failure -- inventing a correction from a vague complaint is not.
    """
    day = _parse_day(text, ledger)
    denied = _first_named(_VOTE_DENIAL_RE, text, ledger, speaker_id)
    asserted = _first_named(_VOTE_ASSERTION_RE, text, ledger, speaker_id)
    if denied is None and asserted is None:
        return []
    subject = _correction_subject(text, ledger, speaker_id)
    return [
        FactCorrection(
            kind=CorrectionKind.VOTE_TARGET,
            source_player_id=speaker_id,
            subject_id=subject,
            day=day,
            asserted=asserted,
            denied=denied,
        )
    ]


def _parse_day(text: str, ledger: PublicFactLedger) -> int:
    match = _DAY_RE.search(text)
    return int(match.group("day")) if match else ledger.day


def _correction_subject(text: str, ledger: PublicFactLedger, speaker_id: str) -> str:
    """Whose record is being corrected. First person unless someone else is named
    as the voter, which the narrow grammar here does not yet cover."""
    return speaker_id


def _first_named(
    pattern: re.Pattern[str], text: str, ledger: PublicFactLedger, speaker_id: str
) -> str | None:
    for match in pattern.finditer(text):
        label = match.group("label")
        for player_id in ledger.known_player_ids():
            if player_id == speaker_id:
                continue
            if mentions_player(label, player_id, ledger.name_of(player_id)):
                return player_id
    return None
