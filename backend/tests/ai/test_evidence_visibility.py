"""Private certainty may drive a decision; it may not be said out loud.

`EvidenceVisibility` existed with a filter that read it, but nothing ever set
it: every record defaulted to PUBLIC_ARGUMENT, so the filter let everything
through. Vote evidence is built from the seat's *private* solver, which is how
a seer with an unpublished black produced "p2 voted for a confirmed wolf", and
how a wolf -- knowing exactly which three seats are wolves -- produced "p9 is
confirmed not a wolf" as a reason it could give the table.

The line these tests hold: the seat may *use* what it privately knows; its
speech may only argue from what the table could reach too.
"""

from __future__ import annotations

from app.ai.reasoning.belief import BeliefEngine, EvidenceVisibility
from app.ai.reasoning.belief.utility import RoleCertainty
from app.ai.reasoning.facts import PublicFactLedger
from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import PlayerPrivatePerspective
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.reasoning.solver import build_solver
from app.ai.reasoning.solver.backend import Certainty
from app.engine.roles import RoleName
from app.engine.state import VoteRecord
from tests.ai.reasoning.solver import boards

AI_IDS = [f"p{i}" for i in range(1, 17)]
# On this deal p9 and p10 are the other two wolves; p2, p5, p7 are villagers.
VILLAGERS = ("p2", "p5", "p7")


def _board():  # type: ignore[no-untyped-def]
    """Day 2. The seer (p4) privately knows p1 is a wolf; p2 then voted for p1.

    The ballots are cast on day 2, after the night-1 look: judging a day-1
    ballot by a night-1 result would be hindsight, which the engine refuses.
    """
    state = boards.deal({"p1": RoleName.WEREWOLF, "p4": RoleName.SEER}, day=2)
    boards.divine(state, "p4", "p1", night=1)
    state.vote_records.append(VoteRecord(voter_id="p2", target_id="p1", day=2, round=1))
    state.vote_records.append(VoteRecord(voter_id="p3", target_id="p5", day=2, round=1))
    return state


def _runtime(state):  # type: ignore[no-untyped-def]
    runtime = ReasoningRuntime(state, AI_IDS, seed=1)
    runtime.refresh(state)
    return runtime


def _public_text(runtime, seat: str, state) -> str:  # type: ignore[no-untyped-def]
    belief = runtime.seats[seat].belief
    return (
        "".join(r.explanation for r in belief.public_argument_evidence_for(None))
        + runtime.discussion_decision(state, seat).render_brief()
    )


def test_the_seers_unpublished_black_never_reaches_another_seats_brief():
    state = _board()
    runtime = _runtime(state)

    for seat in VILLAGERS:
        assert "確定" not in _public_text(runtime, seat, state)


def test_the_seers_own_brief_does_not_say_confirmed_before_publishing():
    state = _board()
    runtime = _runtime(state)

    assert "確定" not in _public_text(runtime, "p4", state)


def test_a_wolf_does_not_argue_from_knowing_the_wolf_roster():
    state = _board()
    runtime = _runtime(state)

    text = _public_text(runtime, "p9", state)
    assert "確定" not in text
    wolf_view = runtime.seats["p9"].belief.active_evidence()
    assert any(r.visibility is EvidenceVisibility.TEAM_PRIVATE for r in wolf_view)


def test_private_certainty_is_still_used_internally():
    """Hiding it from speech must not mean discarding it."""
    state = _board()
    runtime = _runtime(state)
    seer = runtime.seats["p4"].belief

    assert seer.state.private_role_certainties["p1"] is RoleCertainty.CONFIRMED
    private = [
        r
        for r in seer.active_evidence()
        if r.visibility is EvidenceVisibility.PRIVATE_REASONING
    ]
    assert private and private[0].subject_id == "p2"
    # It moves the seat's own view of p2, but not what the table could argue.
    assert seer.state.own_suspicion_scores.get("p2", 0.0) != 0.0
    assert seer.state.public_suspicion_scores.get("p2", 0.0) == 0.0


def test_a_ballot_is_not_judged_with_what_was_learned_afterwards():
    """p2 voted for p1 on day 1; the seer only learned p1 was a wolf that night."""
    state = boards.deal({"p1": RoleName.WEREWOLF, "p4": RoleName.SEER}, day=2)
    boards.divine(state, "p4", "p1", night=1)
    state.vote_records.append(VoteRecord(voter_id="p2", target_id="p1", day=1, round=1))
    runtime = _runtime(state)

    categories = {record.category for record in runtime.seats["p4"].belief.active_evidence()}
    assert "voted_for_wolf" not in categories


def test_a_conclusion_the_table_reaches_too_stays_public():
    """Visibility is about the *conclusion*, not about who computed it."""
    state = _board()
    observations = ObservationSet.from_state(state)
    engine = BeliefEngine("p4", PlayerPrivatePerspective("p4"))

    engine.observe(
        PublicFactLedger(state),
        build_solver(observations, PlayerPrivatePerspective("p4")),
        observations,
        public_certainties={"p1": Certainty.CERTAIN},
    )

    vote = next(r for r in engine.active_evidence() if r.category == "voted_for_wolf")
    assert vote.visibility is EvidenceVisibility.PUBLIC_ARGUMENT


def test_with_no_public_view_supplied_the_safe_default_is_private():
    state = _board()
    observations = ObservationSet.from_state(state)
    engine = BeliefEngine("p4", PlayerPrivatePerspective("p4"))

    engine.observe(
        PublicFactLedger(state),
        build_solver(observations, PlayerPrivatePerspective("p4")),
        observations,
    )

    vote = next(r for r in engine.active_evidence() if r.category == "voted_for_wolf")
    assert vote.visibility is not EvidenceVisibility.PUBLIC_ARGUMENT


def test_a_published_result_is_a_public_reason():
    state = _board()
    boards.claim(state, "p4", RoleName.SEER, day=2)
    boards.verdict(state, "p4", "seer", "p1", True, day=2, referenced_day=1)
    runtime = _runtime(state)

    for seat in ("p4", *VILLAGERS):
        published = [
            r
            for r in runtime.seats[seat].belief.public_argument_evidence_for("p1")
            if r.category == "published_black"
        ]
        assert published, f"{seat} cannot argue from a result the table heard"


def test_a_vote_reason_never_falls_back_to_a_private_one():
    state = _board()
    runtime = _runtime(state)
    candidates = [pid for pid in state.alive_ids() if pid != "p4"]

    _, reason = runtime.vote_decision("p4", candidates)

    assert reason
    assert "確定" not in reason
