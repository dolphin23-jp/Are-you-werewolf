"""Recompute release-quality facts from immutable transcript audit events."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from enum import StrEnum

from app.eval.transcript import DecisionAuditRecord, GameTranscript


class VoteChangeKind(StrEnum):
    NONE = "none"
    NEW_PUBLIC_EVIDENCE = "new_public_evidence"
    TARGET_BECAME_INVALID = "target_became_invalid"
    RUNOFF_RESTRICTION = "runoff_restriction"
    PUBLIC_RESULT_CHANGED = "public_result_changed"
    CORRECTION_ACCEPTED = "correction_accepted"
    STRATEGIC_TEAM_DECISION = "strategic_team_decision"
    UNEXPLAINED = "unexplained"


JUSTIFIED_CHANGES = frozenset(VoteChangeKind) - {VoteChangeKind.NONE, VoteChangeKind.UNEXPLAINED}


@dataclass(frozen=True)
class ReasoningQualityReport:
    decision_count: int = 0
    vote_count: int = 0
    dead_target_selection_count: int = 0
    invalid_action_target_count: int = 0
    public_fact_flip_count: int = 0
    vote_history_fabrication_count: int = 0
    private_evidence_exposed_count: int = 0
    team_private_evidence_exposed_count: int = 0
    displayed_target_mismatch_count: int = 0
    missing_required_result_count: int = 0
    duplicate_required_result_count: int = 0
    stale_evidence_used_in_brief_count: int = 0
    stale_evidence_cited_publicly_count: int = 0
    stale_evidence_attempt_count: int = 0
    stale_evidence_blocked_count: int = 0
    stale_evidence_publicly_emitted_count: int = 0
    justified_vote_change_count: int = 0
    unexplained_vote_change_count: int = 0
    wolf_ally_vote_count: int = 0
    planned_wolf_ally_vote_count: int = 0
    unplanned_wolf_ally_vote_count: int = 0
    illegal_timeline_claim_count: int = 0
    duplicate_night_action_count: int = 0
    correction_messages_heard: int = 0
    corrections_confirmed: int = 0
    corrections_refuted: int = 0
    corrections_unverifiable: int = 0
    confirmed_corrections_with_effect: int = 0
    seat_evidence_retractions: int = 0
    affected_seat_count: int = 0
    mean_belief_delta_per_affected_seat: float = 0.0
    median_belief_delta_per_affected_seat: float = 0.0

    @property
    def vote_change_rate(self) -> float:
        return (self.justified_vote_change_count + self.unexplained_vote_change_count) / max(
            self.vote_count, 1
        )

    @property
    def unexplained_vote_change_rate(self) -> float:
        return self.unexplained_vote_change_count / max(self.vote_count, 1)

    def to_dict(self) -> dict[str, int | float]:
        return {
            **asdict(self),
            "vote_change_rate": self.vote_change_rate,
            "unexplained_vote_change_rate": self.unexplained_vote_change_rate,
        }


class ReasoningTranscriptAnalyzer:
    def analyze(self, transcript: GameTranscript) -> ReasoningQualityReport:
        decisions = transcript.decision_audits
        publications = transcript.result_publication_audits
        corrections = {
            record.correction_id: record for record in transcript.correction_audits
        }.values()
        stale_attempts = sum(_stale(record) for record in decisions)
        stale_emitted = sum(_stale_emitted(record) for record in decisions)
        changes = [_change_kind(record) for record in decisions if record.vote_target is not None]
        deltas = [
            abs(delta)
            for correction in corrections
            for delta in correction.belief_delta_by_seat.values()
        ]
        public_results: dict[tuple[str, str, int | None], set[bool]] = {}
        for utterance in transcript.utterances:
            for result in utterance.public_results:
                key = (
                    utterance.player_id,
                    str(result.get("target_id", "")),
                    result.get("referenced_day"),
                )
                public_results.setdefault(key, set()).add(bool(result.get("is_werewolf")))
        night_keys = [
            (utterance.player_id, utterance.day, utterance.kind)
            for utterance in transcript.utterances
            if utterance.kind == "night_action"
        ]
        return ReasoningQualityReport(
            decision_count=len(decisions),
            vote_count=len(changes),
            dead_target_selection_count=sum(
                not record.target_alive for record in decisions if record.decision_target
            ),
            private_evidence_exposed_count=sum(
                len(set(record.public_evidence_ids) & set(record.private_evidence_ids))
                for record in decisions
            ),
            team_private_evidence_exposed_count=sum(
                len(set(record.public_evidence_ids) & set(record.team_private_evidence_ids))
                for record in decisions
            ),
            displayed_target_mismatch_count=sum(
                record.displayed_target != record.decision_target for record in decisions
            ),
            missing_required_result_count=sum(
                len(record.omitted_result_ids) for record in publications
            ),
            duplicate_required_result_count=sum(
                len(record.duplicate_result_ids) for record in publications
            ),
            public_fact_flip_count=sum(len(colours) > 1 for colours in public_results.values()),
            duplicate_night_action_count=len(night_keys) - len(set(night_keys)),
            stale_evidence_used_in_brief_count=stale_attempts,
            stale_evidence_cited_publicly_count=stale_emitted,
            stale_evidence_attempt_count=stale_attempts,
            stale_evidence_blocked_count=stale_attempts - stale_emitted,
            stale_evidence_publicly_emitted_count=stale_emitted,
            justified_vote_change_count=sum(kind in JUSTIFIED_CHANGES for kind in changes),
            unexplained_vote_change_count=changes.count(VoteChangeKind.UNEXPLAINED),
            wolf_ally_vote_count=sum(record.ally_vote for record in decisions),
            planned_wolf_ally_vote_count=sum(
                record.ally_vote and record.ally_vote_planned for record in decisions
            ),
            unplanned_wolf_ally_vote_count=sum(
                record.ally_vote and not record.ally_vote_planned for record in decisions
            ),
            correction_messages_heard=len(tuple(corrections)),
            corrections_confirmed=sum(record.verdict == "confirmed" for record in corrections),
            corrections_refuted=sum(record.verdict == "refuted" for record in corrections),
            corrections_unverifiable=sum(
                record.verdict == "unverifiable" for record in corrections
            ),
            confirmed_corrections_with_effect=sum(
                record.verdict == "confirmed" and bool(record.retracted_evidence_ids)
                for record in corrections
            ),
            seat_evidence_retractions=sum(
                len(record.retracted_evidence_ids) for record in corrections
            ),
            affected_seat_count=len(
                {seat for record in corrections for seat in record.affected_seat_ids}
            ),
            mean_belief_delta_per_affected_seat=statistics.fmean(deltas) if deltas else 0.0,
            median_belief_delta_per_affected_seat=statistics.median(deltas) if deltas else 0.0,
        )


def _change_kind(record: DecisionAuditRecord) -> VoteChangeKind:
    if record.vote_target == record.decision_target:
        return VoteChangeKind.NONE
    try:
        return VoteChangeKind(record.target_change_classification or VoteChangeKind.UNEXPLAINED)
    except ValueError:
        return VoteChangeKind.UNEXPLAINED


def _stale(record: DecisionAuditRecord) -> int:
    return int(bool(set(record.public_evidence_ids) - set(record.active_evidence_ids)))


def _stale_emitted(record: DecisionAuditRecord) -> int:
    stale = set(record.public_evidence_ids) - set(record.active_evidence_ids)
    return int(bool(stale & set(record.publicly_emitted_evidence_ids)))
