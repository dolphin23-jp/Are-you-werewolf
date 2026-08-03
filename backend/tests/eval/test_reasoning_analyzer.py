from app.eval.reasoning_analyzer import ReasoningTranscriptAnalyzer, VoteChangeKind
from app.eval.transcript import (
    CorrectionAuditRecord,
    DecisionAuditRecord,
    GameTranscript,
    NightActionAuditRecord,
    ResultPublicationAuditRecord,
    TranscriptRecorder,
    Utterance,
)


def test_analyzer_recomputes_visibility_publication_and_change_failures():
    transcript = GameTranscript(
        decision_audits=[
            DecisionAuditRecord(
                game_id="g",
                day=2,
                phase="voting",
                player_id="p1",
                decision_target="p2",
                displayed_target="p3",
                vote_target="p3",
                public_evidence_ids=("private:black", "stale"),
                private_evidence_ids=("private:black",),
                active_evidence_ids=("live",),
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
        decision_audits=[
            DecisionAuditRecord(
                game_id="g",
                day=2,
                phase="voting",
                player_id="p1",
                decision_target="p2",
                displayed_target="p2",
                vote_target="p3",
                target_change_classification=VoteChangeKind.NEW_PUBLIC_EVIDENCE,
            )
        ],
        correction_audits=[
            CorrectionAuditRecord(
                "c1",
                "m1",
                "p0",
                "confirmed",
                ("p1", "p2"),
                ("bad-vote",),
                {"p1": -1.0, "p2": -0.5},
            )
        ],
    )
    report = ReasoningTranscriptAnalyzer().analyze(transcript)
    assert report.justified_vote_change_count == 1
    assert report.correction_messages_heard == 1
    assert report.confirmed_corrections_with_effect == 1
    assert report.affected_seat_count == 2
    assert report.seat_evidence_retractions == 1


def test_transcript_round_trip_and_vote_attachment():
    recorder = TranscriptRecorder()
    recorder.record_decision_audit(
        DecisionAuditRecord("g", 2, "discussion", "p1", decision_target="p2")
    )
    recorder.update_latest_decision(
        "p1", 2, vote_target="p3", target_change_classification="unexplained"
    )
    restored = GameTranscript.from_dict(recorder.transcript.to_dict())
    assert restored.schema_version == 3
    assert restored.decision_audits[0].vote_target == "p3"


def test_analyzer_recomputes_fact_flips_and_duplicate_night_actions():
    base = dict(
        day=2,
        phase="night",
        player_id="p1",
        player_name="P1",
        role="seer",
        team="village",
        personality="calm",
        deception_role=None,
    )
    transcript = GameTranscript(
        utterances=[
            Utterance(
                kind="discussion",
                public_results=[{"target_id": "p2", "referenced_day": 1, "is_werewolf": False}],
                **base,
            ),
            Utterance(
                kind="discussion",
                public_results=[{"target_id": "p2", "referenced_day": 1, "is_werewolf": True}],
                **base,
            ),
        ],
        night_action_audits=[
            NightActionAuditRecord("a1", "p1", "p2", 2, "divine", True),
            NightActionAuditRecord("a2", "p1", "p3", 2, "divine", True),
        ],
    )
    report = ReasoningTranscriptAnalyzer().analyze(transcript)
    assert report.public_fact_flip_count == 1
    assert report.duplicate_night_action_count == 1


def test_analyzer_counts_rejected_or_illegal_night_target():
    transcript = GameTranscript(
        night_action_audits=[
            NightActionAuditRecord(
                "a1", "p1", "dead", 2, "guard", False,
                rejection_reason="GameError", target_was_legal=False,
            )
        ]
    )
    assert ReasoningTranscriptAnalyzer().analyze(transcript).invalid_action_target_count == 1


def test_analyzer_recomputes_future_and_duplicate_result_timeline():
    base = dict(
        day=2, phase="discussion", kind="discussion", player_id="p1",
        player_name="P1", role="seer", team="village", personality="calm",
        deception_role=None,
    )
    transcript = GameTranscript(
        utterances=[
            Utterance(
                **base,
                public_results=[
                    {"target_id": "p2", "referenced_day": 2, "is_werewolf": False},
                    {"target_id": "p3", "referenced_day": 2, "is_werewolf": True},
                ],
            )
        ],
        metrics={"inaccurate_vote_citations": 1},
    )
    report = ReasoningTranscriptAnalyzer().analyze(transcript)
    assert report.illegal_timeline_claim_count == 3  # two future claims plus duplicate night
    assert report.vote_history_fabrication_count == 1
