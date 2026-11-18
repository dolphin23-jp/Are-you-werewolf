"""A runoff keeps the first round's ballot facts.

Each seat's ballot used to be written onto its latest audit of the day, so a
runoff ballot overwrote the first round's: a werewolf that voted its partner
unplanned in round 1 and the other tied seat in the runoff left no trace of
the ally vote, and a round-1 change nobody explained vanished the same way.
Every round's ballot now has its own record, and the analyzer counts what a
seat did once however many rounds it did it in.
"""

from __future__ import annotations

import asyncio

from app.ai.coordinator import AICoordinator
from app.ai.reasoning.runtime import ReasoningRuntime
from app.engine.phases import Phase
from app.engine.state import GameState
from app.eval.reasoning_analyzer import ReasoningTranscriptAnalyzer, VoteChangeKind
from app.eval.transcript import DecisionAuditRecord, GameTranscript, TranscriptRecorder
from tests.ai import first_claude_game as game1
from tests.ai.reasoning.solver import boards
from tests.ai.test_vote_audit import _PlanAndBallot, _v3_table, _wolf_pair_board


def _runoff(state: GameState, *tied: str) -> None:
    state.phase = Phase.RUNOFF
    state.vote_round += 1
    state.runoff_candidates = list(tied)
    state.pending_votes.clear()


def _wolf_table() -> tuple[GameState, str, str, _PlanAndBallot, TranscriptRecorder, AICoordinator]:
    state, voter, ally = _wolf_pair_board()
    ai_ids = [pid for pid in boards.PLAYER_IDS if pid != "p0"]
    provider = _PlanAndBallot(ally)
    recorder = TranscriptRecorder()
    coordinator = AICoordinator(
        state,
        ai_ids,
        provider,
        seed=1,
        reasoning=ReasoningRuntime(state, ai_ids, seed=1),
        model_decides=True,
        recorder=recorder,
    )
    return state, voter, ally, provider, recorder, coordinator


def _ballots(recorder: TranscriptRecorder, player_id: str) -> list[DecisionAuditRecord]:
    return [
        audit
        for audit in recorder.transcript.decision_audits
        if audit.player_id == player_id and audit.vote_target is not None
    ]


def test_a_runoff_ballot_does_not_erase_a_first_round_unplanned_ally_vote():
    """The review's runoff.py: round 1 on the partner, the runoff on the other tied seat."""
    state, voter, ally, provider, recorder, coordinator = _wolf_table()
    box = game1.BallotBox(state)
    asyncio.run(coordinator._cast_vote(box, state, voter))
    first = ReasoningTranscriptAnalyzer().analyze(recorder.transcript)
    assert (first.unplanned_wolf_ally_vote_count, first.vote_count) == (1, 1)

    _runoff(state, ally, "p5")
    provider.vote = "p5"
    asyncio.run(coordinator._cast_vote(box, state, voter))

    report = ReasoningTranscriptAnalyzer().analyze(recorder.transcript)
    assert report.unplanned_wolf_ally_vote_count == 1
    assert report.wolf_ally_vote_count == 1
    # Two ballots, two records, each with its round.
    assert report.vote_count == 2
    ballots = _ballots(recorder, voter)
    assert [(b.vote_round, b.vote_target, b.ally_vote, b.ally_vote_planned) for b in ballots] == [
        (1, ally, True, False),
        (2, "p5", False, False),
    ]
    assert ballots[1].vote_only is True
    assert ballots[1].votable_ids_at_decision == (ally, "p5")


def test_the_same_unplanned_ally_vote_in_two_rounds_is_counted_once():
    state, voter, ally, provider, recorder, coordinator = _wolf_table()
    box = game1.BallotBox(state)
    asyncio.run(coordinator._cast_vote(box, state, voter))
    _runoff(state, ally, "p5")
    asyncio.run(coordinator._cast_vote(box, state, voter))

    assert len(_ballots(recorder, voter)) == 2
    report = ReasoningTranscriptAnalyzer().analyze(recorder.transcript)
    assert report.vote_count == 2
    assert report.wolf_ally_vote_count == 1
    assert report.unplanned_wolf_ally_vote_count == 1
    assert report.planned_wolf_ally_vote_count == 0


def test_a_runoff_ballot_does_not_erase_a_first_round_unexplained_change():
    controller, state, coordinator, provider, recorder, speaker, plan, other = _v3_table()
    asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))
    provider.vote = other
    state.phase = Phase.VOTING
    asyncio.run(coordinator._cast_vote(controller, state, speaker))
    first = ReasoningTranscriptAnalyzer().analyze(recorder.transcript)
    assert first.unexplained_vote_change_count == 1

    # The runoff is between the seat's plan and where it went; it goes back.
    _runoff(state, plan, other)
    provider.vote = plan
    asyncio.run(coordinator._cast_vote(controller, state, speaker))

    ballots = _ballots(recorder, speaker)
    assert [(b.vote_round, b.vote_target, b.target_change_classification) for b in ballots] == [
        (1, other, VoteChangeKind.UNEXPLAINED),
        (2, plan, VoteChangeKind.NONE),
    ]
    # The turn's record keeps the turn; the runoff ballot is its own record.
    assert ballots[0].vote_only is False
    assert ballots[1].vote_only is True
    report = ReasoningTranscriptAnalyzer().analyze(recorder.transcript)
    assert report.unexplained_vote_change_count == 1
    assert report.vote_count == 2
    assert report.decision_count == 1


def test_the_same_unexplained_change_in_two_rounds_is_counted_once():
    controller, state, coordinator, provider, recorder, speaker, plan, other = _v3_table()
    asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))
    provider.vote = other
    state.phase = Phase.VOTING
    asyncio.run(coordinator._cast_vote(controller, state, speaker))
    _runoff(state, plan, other)
    asyncio.run(coordinator._cast_vote(controller, state, speaker))

    ballots = _ballots(recorder, speaker)
    assert [b.target_change_classification for b in ballots] == [VoteChangeKind.UNEXPLAINED] * 2
    report = ReasoningTranscriptAnalyzer().analyze(recorder.transcript)
    assert report.unexplained_vote_change_count == 1


def test_a_runoff_ballot_is_classified_against_what_the_seat_had_seen_when_it_spoke():
    """The runoff record carries the turn's snapshot, so a third round reads the same one."""
    controller, state, coordinator, provider, recorder, speaker, plan, other = _v3_table()
    asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))
    turn = recorder.latest_decision(speaker, state.day)
    assert turn is not None
    state.phase = Phase.VOTING
    asyncio.run(coordinator._cast_vote(controller, state, speaker))
    _runoff(state, plan, other)
    asyncio.run(coordinator._cast_vote(controller, state, speaker))
    _runoff(state, plan, other)
    asyncio.run(coordinator._cast_vote(controller, state, speaker))

    ballots = _ballots(recorder, speaker)
    assert [b.vote_round for b in ballots] == [1, 2, 3]
    for ballot in ballots[1:]:
        assert ballot.public_result_ids_at_decision == turn.public_result_ids_at_decision
        assert ballot.correction_ids_at_decision == turn.correction_ids_at_decision
        assert ballot.board_version == turn.board_version


def test_a_ballot_recast_in_the_same_round_replaces_that_rounds_record():
    state, voter, ally, provider, recorder, coordinator = _wolf_table()
    box = game1.BallotBox(state)
    asyncio.run(coordinator._cast_vote(box, state, voter))
    provider.vote = "p5"
    asyncio.run(coordinator._cast_vote(box, state, voter))

    ballots = _ballots(recorder, voter)
    assert [(b.vote_round, b.vote_target) for b in ballots] == [(1, "p5")]
    assert ReasoningTranscriptAnalyzer().analyze(recorder.transcript).wolf_ally_vote_count == 0


def test_the_analyzer_counts_each_rounds_record_and_dedupes_a_repeat():
    """Hand-built: two seats, two rounds. What one seat did twice counts once."""

    def ballot(
        player: str, round_number: int, target: str, **fields: object
    ) -> DecisionAuditRecord:
        fields.setdefault("target_change_classification", "none")
        return DecisionAuditRecord(
            game_id="g",
            day=2,
            phase="voting" if round_number == 1 else "runoff",
            player_id=player,
            vote_target=target,
            vote_round=round_number,
            vote_only=True,
            **fields,  # type: ignore[arg-type]
        )

    transcript = GameTranscript(
        decision_audits=[
            DecisionAuditRecord(
                game_id="g", day=2, phase="discussion", player_id="p1",
                decision_target="p2", displayed_target="p2", vote_target="p3",
                target_change_classification="unexplained", vote_round=1,
            ),
            ballot("p1", 2, "p3", target_change_classification="unexplained"),
            ballot("p4", 1, "p5", ally_vote=True, ally_vote_planned=False),
            ballot("p4", 2, "p6"),
            ballot("p7", 1, "p5", ally_vote=True, ally_vote_planned=True),
            ballot("p7", 2, "p5", ally_vote=True, ally_vote_planned=True),
        ]
    )

    report = ReasoningTranscriptAnalyzer().analyze(transcript)

    assert report.vote_count == 6
    assert report.decision_count == 1
    assert report.unexplained_vote_change_count == 1
    assert report.wolf_ally_vote_count == 2
    assert report.unplanned_wolf_ally_vote_count == 1
    assert report.planned_wolf_ally_vote_count == 1


def test_an_exported_transcript_without_vote_rounds_still_loads():
    data = GameTranscript(decision_audits=[DecisionAuditRecord("g", 1, "voting", "p1")]).to_dict()
    del data["decision_audits"][0]["vote_round"]

    restored = GameTranscript.from_dict(data)

    assert restored.decision_audits[0].vote_round == 0
