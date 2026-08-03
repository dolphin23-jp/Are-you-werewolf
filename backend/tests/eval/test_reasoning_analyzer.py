from app.eval.reasoning_analyzer import ReasoningTranscriptAnalyzer, VoteChangeKind
from app.eval.transcript import (
    CorrectionAuditRecord,
    DecisionAuditRecord,
    GameTranscript,
    ResultPublicationAuditRecord,
)


def test_analyzer_recomputes_visibility_publication_and_change_failures():
    transcript = GameTranscript(
        decision_audits=[
            DecisionAuditRecord(
                game_id="g", day=2, phase="voting", player_id="p1",
                decision_target="p2", displayed_target="p3", vote_target="p3",
                public_evidence_ids=("private:black", "stale"),
                private_evidence_ids=("private:black",), active_evidence_ids=("live",),
                publicly_emitted_evidence_ids=("stale",),
            )
        ],
        result_publication_audits=[
            ResultPublicationAuditRecord("p1", 2, ("seer:1:p2",), (), ("seer:1:p2",), ())
        ],
    )
    report = ReasoningTranscriptAnalyzer().analyze(transcript)
    assert report.private_evidence_exposed_count == 1
    assert report.displayed_target_mismatch_count == 1
    assert report.missing_required_result_count == 1
    assert report.stale_evidence_attempt_count == 1
    assert report.stale_evidence_publicly_emitted_count == 1
    assert report.unexplained_vote_change_count == 1


def test_justified_change_and_correction_are_counted_at_distinct_levels():
    transcript = GameTranscript(
        decision_audits=[DecisionAuditRecord(
            game_id="g", day=2, phase="voting", player_id="p1", decision_target="p2",
            displayed_target="p2", vote_target="p3",
            target_change_classification=VoteChangeKind.NEW_PUBLIC_EVIDENCE,
        )],
        correction_audits=[CorrectionAuditRecord(
            "c1", "m1", "p0", "confirmed", ("p1", "p2"), ("bad-vote",),
            {"p1": -1.0, "p2": -0.5},
        )],
    )
    report = ReasoningTranscriptAnalyzer().analyze(transcript)
    assert report.justified_vote_change_count == 1
    assert report.correction_messages_heard == 1
    assert report.confirmed_corrections_with_effect == 1
    assert report.affected_seat_count == 2
    assert report.seat_evidence_retractions == 1
