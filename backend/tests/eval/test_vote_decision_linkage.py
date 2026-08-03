from dataclasses import replace

from app.ai.reasoning.runtime import ReasoningRuntime, VoteDecisionSnapshot
from app.eval.reasoning_analyzer import ReasoningTranscriptAnalyzer, VoteChangeKind
from app.eval.transcript import DecisionAuditRecord, TranscriptRecorder


def _snapshot(
    *, results=(), black=(), corrections=(), alive=("p1", "p2", "p3"), eligible=("p2", "p3")
):
    return VoteDecisionSnapshot(results, black, corrections, alive, eligible)


def test_transcript_links_decision_displayed_and_accepted_vote_by_id():
    recorder = TranscriptRecorder()
    decision = DecisionAuditRecord(
        decision_id="decision-1",
        game_id="game",
        day=2,
        phase="discussion",
        player_id="p1",
        decision_target="p2",
        displayed_target="p2",
    )
    recorder.record_decision_audit(decision)

    index, open_decision = recorder.latest_open_decision(game_id="game", day=2, player_id="p1")
    recorder.replace_decision_audit(index, replace(open_decision, vote_target="p3"))

    recorded = recorder.transcript.decision_audits[0]
    assert (
        recorded.decision_id,
        recorded.decision_target,
        recorded.displayed_target,
        recorded.vote_target,
    ) == ("decision-1", "p2", "p2", "p3")


def test_new_public_black_classifies_changed_vote_as_public_evidence():
    before = _snapshot()
    after = _snapshot(results=("seer:p0:1:p3:True",), black=("seer:p0:1:p3:True",))
    kind, reason = ReasoningRuntime.classify_vote_change(
        before, after, decision_target="p2", vote_target="p3"
    )
    assert kind == VoteChangeKind.NEW_PUBLIC_EVIDENCE
    assert "公開黒" in reason


def test_change_without_public_reason_is_unexplained_in_transcript_report():
    kind, reason = ReasoningRuntime.classify_vote_change(
        _snapshot(), _snapshot(), decision_target="p2", vote_target="p3"
    )
    record = DecisionAuditRecord(
        decision_id="decision-2",
        game_id="game",
        day=2,
        phase="discussion",
        player_id="p1",
        decision_target="p2",
        displayed_target="p2",
        vote_target="p3",
        target_change_classification=kind,
        target_change_reason=reason,
    )
    recorder = TranscriptRecorder()
    recorder.record_decision_audit(record)
    report = ReasoningTranscriptAnalyzer().analyze(recorder.transcript)
    assert kind == VoteChangeKind.UNEXPLAINED
    assert report.unexplained_vote_change_count == 1
    assert reason
