"""Three scenarios the roadmap names, played out rather than described.

Aggregate numbers over mock games say a layer runs. They do not say it reaches
the right answer, because a mock provider never produces the situations that
distinguish good reasoning from a plausible average. These three do, and each
asserts a specific outcome rather than a trend:

1. **ダイキ処刑** -- somebody misquotes a ballot, the table builds on the quote,
   the person named produces the record, and the reason has to go with it.
2. **人狼仲間投票** -- a wolf's private certainty about its partners is real and
   must survive, while never becoming a reason to nominate them.
3. **能力者私的知識** -- the seer's own black is settled for the seer before it
   is published, and unavailable to everybody else while it stays unpublished.

Every one runs the production objects (`ReasoningRuntime`, `BeliefEngine`,
`AICoordinator`), not a rehearsal of the logic in test code.
"""

from __future__ import annotations

import asyncio

from app.ai.coordinator import AICoordinator
from app.ai.provider.base import Message, SchemaT
from app.ai.reasoning.belief.utility import RoleCertainty
from app.ai.reasoning.facts import PublicFactLedger
from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import PlayerPrivatePerspective
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.reasoning.solver import build_solver, has_role
from app.ai.reasoning.solver.backend import Certainty
from app.ai.schemas import (
    DiscussionOutput,
    MorningIntentOutput,
    NightActionOutput,
    SummaryOutput,
    VoteOutput,
)
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.state import VoteRecord
from tests.ai.reasoning.solver import boards

ALL_SEATS = [f"p{i}" for i in range(17)]
AI_SEATS = [f"p{i}" for i in range(1, 17)]
# `boards.deal` completes the composition deterministically, so on the seer
# boards below p9 and p10 are the other two wolves and p16 is a freemason. A
# "does anyone else know?" check has to ask a seat that really knows nothing,
# or it fails on a seat that legitimately does.
PLAIN_VILLAGERS = ("p2", "p5", "p7")


class QuietProvider:
    """Says nothing of substance. The scenarios are about the engine."""

    def __init__(self, message: str = "様子を見ます。") -> None:
        self.message = message

    async def generate_structured(
        self, *, system: str, messages: list[Message], response_schema: type[SchemaT], **kwargs
    ):  # type: ignore[no-untyped-def]
        del system, messages, kwargs
        if response_schema is MorningIntentOutput:
            return MorningIntentOutput()
        if response_schema is DiscussionOutput:
            return DiscussionOutput(public_message=self.message)
        if response_schema is VoteOutput:
            return VoteOutput(vote_target="p2")
        if response_schema is NightActionOutput:
            return NightActionOutput(target="p2")
        return SummaryOutput(summary="要約")


# -- 1. ダイキ処刑: a misquoted ballot, and what happens when it is disproved --


def _daiki_board():  # type: ignore[no-untyped-def]
    """Day 2. Daiki (p12) was executed on day 1; p0 voted for p7, not for p12."""
    state = boards.deal({"p1": RoleName.WEREWOLF, "p4": RoleName.SEER}, day=2)
    boards.execute(state, "p12", day=1)
    state.vote_records.append(
        VoteRecord(voter_id="p0", target_id="p7", day=1, round=1)
    )
    state.vote_records.append(
        VoteRecord(voter_id="p3", target_id="p12", day=1, round=1)
    )
    return state


def test_a_misquoted_ballot_becomes_a_reason_the_table_holds():
    state = _daiki_board()
    runtime = ReasoningRuntime(state, AI_SEATS, seed=7)
    runtime.refresh(state)

    runtime.record_public_speech(
        state, "p5", "1日目、p0はp12へ投票しました。処刑を押していました。", "m1"
    )

    listener = runtime.seats["p9"].belief
    cited = [
        record
        for record in listener.active_evidence()
        if record.category == "misremembered_vote"
    ]
    assert cited, "a quote the record contradicts has to land somewhere"
    assert cited[0].subject_id == "p0"
    assert listener.state.public_suspicion_scores.get("p0", 0.0) > 0


def test_producing_the_record_takes_the_reason_away():
    """The whole point of provenance: the score cannot survive its premise."""
    state = _daiki_board()
    runtime = ReasoningRuntime(state, AI_SEATS, seed=7)
    runtime.refresh(state)
    runtime.record_public_speech(state, "p5", "1日目、p0はp12へ投票しました。", "m1")
    before = runtime.seats["p9"].belief.state.public_suspicion_scores.get("p0", 0.0)

    runtime.apply_human_message(
        state,
        "p0",
        "1日目、私はp12には投票していません。p7へ投票しました。",
        "m2",
    )

    after = runtime.seats["p9"].belief.state.public_suspicion_scores.get("p0", 0.0)
    assert before > 0
    assert after < before
    assert not [
        record
        for record in runtime.seats["p9"].belief.active_evidence()
        if record.category == "misremembered_vote"
    ]


def test_a_correction_the_record_refutes_costs_the_speaker_instead():
    state = _daiki_board()
    runtime = ReasoningRuntime(state, AI_SEATS, seed=7)
    runtime.refresh(state)

    # p0 really did vote for p7, so denying it is refuted by the record.
    runtime.apply_human_message(
        state, "p0", "1日目、私はp7には投票していません。p12へ投票しました。", "m1"
    )

    trust = runtime.seats["p9"].belief.state.source_trust.get("p0", 0.0)
    assert trust < 0


def test_the_correction_costs_one_pass_over_the_table_not_one_per_seat():
    state = _daiki_board()
    runtime = ReasoningRuntime(state, AI_SEATS, seed=7)
    runtime.refresh(state)
    provider = QuietProvider()
    AICoordinator(state, AI_SEATS, provider, seed=7, reasoning=runtime)

    runtime.apply_human_message(
        state, "p0", "1日目、私はp12には投票していません。p7へ投票しました。", "m1"
    )

    metrics = runtime.metrics()
    assert metrics["human_messages_considered"] == 1
    assert metrics["corrections_heard"] == 1
    # Sixteen seats, zero model calls: the parse happened once, in code.
    assert metrics["seats_considering_each_human_message"] == len(AI_SEATS)


# -- 2. 人狼仲間投票: knowing is not wanting --


def _wolf_board():  # type: ignore[no-untyped-def]
    return boards.deal(
        {
            "p1": RoleName.WEREWOLF,
            "p2": RoleName.WEREWOLF,
            "p3": RoleName.WEREWOLF,
            "p8": RoleName.MADMAN,
            "p6": RoleName.FOX,
        },
        day=1,
    )


def test_a_wolf_keeps_knowing_its_partners():
    """Not solved by throwing the knowledge away. That was never the bug."""
    state = _wolf_board()
    runtime = ReasoningRuntime(state, AI_SEATS, seed=7)
    runtime.refresh(state)

    certainties = runtime.seats["p1"].belief.state.private_role_certainties

    assert certainties["p2"] is RoleCertainty.CONFIRMED
    assert certainties["p3"] is RoleCertainty.CONFIRMED


def test_a_wolf_does_not_nominate_the_partners_it_knows_about():
    state = _wolf_board()
    runtime = ReasoningRuntime(state, AI_SEATS, seed=7)
    runtime.refresh(state)

    for wolf, partners in (
        ("p1", {"p2", "p3"}),
        ("p2", {"p1", "p3"}),
        ("p3", {"p1", "p2"}),
    ):
        target = runtime.seats[wolf].belief.state.current_execution_target
        assert target not in partners, f"{wolf} nominated a partner"
        ballot, _ = runtime.vote_decision(
            wolf, [pid for pid in state.alive_ids() if pid != wolf]
        )
        assert ballot not in partners, f"{wolf} voted for a partner"


def test_the_madman_is_on_the_team_and_still_does_not_know_where_it_is():
    state = _wolf_board()
    runtime = ReasoningRuntime(state, AI_SEATS, seed=7)
    runtime.refresh(state)

    certainties = runtime.seats["p8"].belief.state.private_role_certainties

    assert all(
        certainties[wolf] is not RoleCertainty.CONFIRMED for wolf in ("p1", "p2", "p3")
    )


def test_a_villager_cannot_see_the_wolves():
    state = _wolf_board()
    runtime = ReasoningRuntime(state, AI_SEATS, seed=7)
    runtime.refresh(state)

    certainties = runtime.seats["p9"].belief.state.private_role_certainties

    assert all(
        certainties[wolf] is not RoleCertainty.CONFIRMED for wolf in ("p1", "p2", "p3")
    )


# -- 3. 能力者私的知識: the seer's own result, before and after publication --


def _seer_board(day: int = 2):  # type: ignore[no-untyped-def]
    state = boards.deal({"p1": RoleName.WEREWOLF, "p4": RoleName.SEER}, day=day)
    boards.divine(state, "p4", "p1", night=1)  # a true black, unpublished
    return state


def test_the_seer_treats_their_own_unpublished_black_as_settled():
    state = _seer_board()
    runtime = ReasoningRuntime(state, AI_SEATS, seed=7)
    runtime.refresh(state)

    assert (
        runtime.seats["p4"].belief.state.private_role_certainties["p1"]
        is RoleCertainty.CONFIRMED
    )


def test_nobody_else_can_reach_the_unpublished_result():
    state = _seer_board()
    observations = ObservationSet.from_state(state)
    runtime = ReasoningRuntime(state, AI_SEATS, seed=7)
    runtime.refresh(state)

    for viewer in PLAIN_VILLAGERS:
        assert (
            runtime.seats[viewer].belief.state.private_role_certainties["p1"]
            is not RoleCertainty.CONFIRMED
        )
        solver = build_solver(observations, PlayerPrivatePerspective(viewer))
        assert solver.assess(has_role("p1", RoleName.WEREWOLF)) is not Certainty.CERTAIN


def test_publishing_does_not_cost_the_seer_their_own_certainty():
    """The original failure ran the other way: publishing downgraded the seer's
    own result to a claim like anybody else's."""
    state = _seer_board()
    boards.claim(state, "p4", RoleName.SEER, day=1)
    boards.verdict(state, "p4", "seer", "p1", True, day=2, referenced_day=1)
    runtime = ReasoningRuntime(state, AI_SEATS, seed=7)
    runtime.refresh(state)

    assert (
        runtime.seats["p4"].belief.state.private_role_certainties["p1"]
        is RoleCertainty.CONFIRMED
    )


def test_a_published_black_is_a_reason_for_others_not_a_certainty():
    state = _seer_board()
    boards.claim(state, "p4", RoleName.SEER, day=1)
    boards.verdict(state, "p4", "seer", "p1", True, day=2, referenced_day=1)
    runtime = ReasoningRuntime(state, AI_SEATS, seed=7)
    runtime.refresh(state)

    listener = runtime.seats["p2"].belief
    assert listener.state.public_suspicion_scores["p1"] > 0
    # Suspicion, not knowledge: a lone seer CO is not a confirmation.
    assert listener.state.private_role_certainties["p1"] is not RoleCertainty.CONFIRMED


def test_the_seer_never_looks_at_the_same_seat_twice():
    state = _seer_board(day=3)
    runtime = ReasoningRuntime(state, AI_SEATS, seed=7)
    runtime.refresh(state)

    target, _ = runtime.night_target("p4", "divine", list(state.alive_ids()))

    assert target != "p1"


def test_the_medium_holds_their_own_result_the_same_way():
    state = boards.deal({"p1": RoleName.WEREWOLF, "p5": RoleName.MEDIUM}, day=2)
    boards.execute(state, "p1", day=1)
    ledger = PublicFactLedger(state)
    assert ledger.was_executed("p1")
    from app.engine.state import MediumRecord

    state.medium_records.append(
        MediumRecord(medium_id="p5", target_id="p1", day=1, is_werewolf=True)
    )
    runtime = ReasoningRuntime(state, AI_SEATS, seed=7)
    runtime.refresh(state)

    assert (
        runtime.seats["p5"].belief.state.private_role_certainties["p1"]
        is RoleCertainty.CONFIRMED
    )
    assert (
        runtime.seats["p2"].belief.state.private_role_certainties["p1"]
        is not RoleCertainty.CONFIRMED
    )


# -- the whole thing still runs a game --


def test_the_scenarios_do_not_break_an_ordinary_round():
    state = _daiki_board()
    state.phase = Phase.DISCUSSION
    runtime = ReasoningRuntime(state, AI_SEATS, seed=7)
    coordinator = AICoordinator(
        state, AI_SEATS, QuietProvider(), seed=7, reasoning=runtime
    )

    round_state = asyncio.run(coordinator._start_discussion_round(state))

    assert round_state.order
    assert all(state.players[pid].alive for pid in round_state.order)
