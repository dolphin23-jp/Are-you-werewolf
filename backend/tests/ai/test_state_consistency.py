"""Regression scenarios for state corruption in AI output.

Each of these reproduces something an AI actually did that no human player
could: putting a corpse on the execution block, reading back a white medium
result as black, or publishing a verdict about a player it never named. They go
through the real coordinator path, because the requirement is that the board is
reconciled *before* anything is persisted -- not that a validator exists.
"""

from __future__ import annotations

import asyncio

from app.ai.coordinator import AICoordinator
from app.ai.reasoning.summaries import FACTS_HEADING, OPINION_HEADING
from app.ai.schemas import (
    DiscussionOutput,
    MorningIntentOutput,
    NightActionOutput,
    PublicResultClaim,
    ReasoningMemo,
    SummaryOutput,
    VoteOutput,
)
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.state import DeathCause, DeathRecord
from tests.conftest import make_controller


class ScriptedProvider:
    """Answers each contract with a canned object, so a test can hand the
    coordinator exactly the malformed output it needs to defend against."""

    def __init__(
        self,
        discussion: DiscussionOutput | None = None,
        vote: VoteOutput | None = None,
        night_action: NightActionOutput | None = None,
        summary: str = "",
    ) -> None:
        self._discussion = discussion
        self._vote = vote
        self._night_action = night_action
        self._summary = summary

    async def generate_structured(self, *, response_schema, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        if response_schema is MorningIntentOutput:
            return MorningIntentOutput()
        if response_schema is DiscussionOutput:
            return self._discussion
        if response_schema is VoteOutput:
            return self._vote
        if response_schema is NightActionOutput:
            return self._night_action
        if response_schema is SummaryOutput:
            return SummaryOutput(summary=self._summary)
        return None


def _execute(controller, player_id: str, day: int) -> None:  # type: ignore[no-untyped-def]
    player = controller.state.players[player_id]
    player.alive = False
    player.death_cause = DeathCause.EXECUTED
    player.death_day = day
    controller.state.death_records.append(
        DeathRecord(player_id=player_id, cause=DeathCause.EXECUTED, day=day)
    )


def test_yesterdays_corpse_is_not_persisted_as_todays_execution_candidate():
    controller = make_controller(seed=4)
    controller.state.phase = Phase.DISCUSSION
    controller.state.day = 2
    _execute(controller, "p0", day=1)
    output = DiscussionOutput(
        public_message="今日はPlayer0(p0)を吊りたいです。",
        reasoning_memo=ReasoningMemo(execution_target="p0", suspects=["p0", "p5"]),
    )
    coordinator = AICoordinator(
        controller.state, ["p1"], ScriptedProvider(discussion=output), seed=1
    )

    asyncio.run(coordinator._speak(controller, controller.state, "p1", "initial_view"))

    stored = coordinator._context.get_reasoning_memo("p1")
    assert stored["execution_target"] == "p5"
    assert controller.state.players[stored["execution_target"]].alive
    assert coordinator.validation.by_code("memo_execution_target_dead")


def test_a_published_white_medium_result_is_not_re_registered_as_black():
    controller = make_controller(seed=4)
    controller.state.phase = Phase.DISCUSSION
    controller.state.day = 3
    _execute(controller, "p4", day=1)
    controller.co("p1", RoleName.MEDIUM.value)
    controller.public_result("p1", "medium", "p4", False)
    output = DiscussionOutput(
        public_message="霊媒結果を訂正します。Player4(p4)は人狼でした。",
        public_results=[PublicResultClaim(result_type="medium", target_id="p4", is_werewolf=True)],
    )
    coordinator = AICoordinator(
        controller.state, ["p1"], ScriptedProvider(discussion=output), seed=1
    )

    asyncio.run(coordinator._speak(controller, controller.state, "p1", "initial_view"))

    verdicts = {
        claim.is_werewolf
        for claim in controller.state.public_result_claims
        if claim.claimant_id == "p1" and claim.target_id == "p4"
    }
    assert verdicts == {False}
    assert coordinator.validation.by_code("result_polarity_conflict")


def test_a_medium_result_about_a_living_player_never_reaches_the_board():
    controller = make_controller(seed=4)
    controller.state.phase = Phase.DISCUSSION
    controller.state.day = 2
    controller.co("p1", RoleName.MEDIUM.value)
    output = DiscussionOutput(
        public_message="霊媒結果です。Player7(p7)は人狼でした。",
        public_results=[PublicResultClaim(result_type="medium", target_id="p7", is_werewolf=True)],
    )
    coordinator = AICoordinator(
        controller.state, ["p1"], ScriptedProvider(discussion=output), seed=1
    )

    asyncio.run(coordinator._speak(controller, controller.state, "p1", "initial_view"))

    assert controller.state.public_result_claims == []
    assert coordinator.validation.by_code("result_medium_target_not_executed")


def test_a_result_about_p1_is_not_attached_to_a_sentence_naming_only_p11():
    controller = make_controller(seed=4)
    controller.state.phase = Phase.DISCUSSION
    controller.state.day = 1
    output = DiscussionOutput(
        public_message="占いCO。Player11(p11)は人狼でした。",
        public_results=[PublicResultClaim(result_type="seer", target_id="p1", is_werewolf=True)],
    )
    coordinator = AICoordinator(
        controller.state, ["p2"], ScriptedProvider(discussion=output), seed=1
    )

    asyncio.run(coordinator._speak(controller, controller.state, "p2", "initial_view"))

    published = {
        (claim.target_id, claim.is_werewolf) for claim in controller.state.public_result_claims
    }
    assert published == {("p11", True)}


def test_the_ballot_the_engine_holds_is_what_gets_compared_to_the_stated_plan():
    controller = make_controller(seed=4)
    controller.state.phase = Phase.VOTING
    controller.state.day = 2
    provider = ScriptedProvider(vote=VoteOutput(vote_target="p7", reason="灰の中で最も怪しい"))
    coordinator = AICoordinator(controller.state, ["p1"], provider, seed=1)
    coordinator._context.set_reasoning_memo("p1", {"execution_target": "p5", "suspects": ["p5"]})

    asyncio.run(coordinator._cast_vote(controller, controller.state, "p1"))

    assert controller.state.pending_votes["p1"] == "p7"
    mismatch = coordinator.validation.vote_plan_mismatches[0]
    assert (mismatch.voter_id, mismatch.stated_target, mismatch.actual_target) == ("p1", "p5", "p7")
    assert mismatch.day == 2


def test_voting_the_stated_plan_records_no_mismatch():
    controller = make_controller(seed=4)
    controller.state.phase = Phase.VOTING
    controller.state.day = 2
    provider = ScriptedProvider(vote=VoteOutput(vote_target="p5", reason="宣言どおり"))
    coordinator = AICoordinator(controller.state, ["p1"], provider, seed=1)
    coordinator._context.set_reasoning_memo("p1", {"execution_target": "p5"})

    asyncio.run(coordinator._cast_vote(controller, controller.state, "p1"))

    assert coordinator.validation.vote_plan_mismatches == []


def test_an_invalid_vote_target_resolves_to_the_players_own_stated_belief():
    controller = make_controller(seed=4)
    controller.state.phase = Phase.VOTING
    controller.state.day = 2
    provider = ScriptedProvider(vote=VoteOutput(vote_target="p999", reason="存在しない相手"))
    coordinator = AICoordinator(controller.state, ["p1"], provider, seed=1)
    coordinator._context.set_reasoning_memo("p1", {"execution_target": "p6", "suspects": ["p6"]})

    asyncio.run(coordinator._cast_vote(controller, controller.state, "p1"))

    # Deterministic and traceable to something this player argued, rather than
    # a random seat nobody can explain after the game.
    assert controller.state.pending_votes["p1"] == "p6"


def test_a_night_victim_is_never_submitted_as_tonights_divine_target():
    controller = make_controller(seed=4)
    controller.state.phase = Phase.NIGHT
    controller.state.day = 2
    controller.state.first_victim_id = "p0"
    controller.state.players["p1"].role = RoleName.SEER
    dead = controller.state.players["p6"]
    dead.alive = False
    dead.death_cause = DeathCause.ATTACKED
    dead.death_day = 1
    controller.state.death_records.append(
        DeathRecord(player_id="p6", cause=DeathCause.ATTACKED, day=1)
    )
    provider = ScriptedProvider(night_action=NightActionOutput(target="p6", reason="昨日の死体"))
    coordinator = AICoordinator(controller.state, ["p1"], provider, seed=1)
    coordinator._context.set_reasoning_memo("p1", {"execution_target": "p9", "suspects": ["p9"]})

    asyncio.run(coordinator._cast_divine(controller, controller.state, "p1"))

    assert controller.state.pending_divine == ("p1", "p9")


def test_the_day_summary_states_facts_from_the_ledger_and_labels_the_rest():
    controller = make_controller(seed=4)
    controller.state.phase = Phase.VOTE_RESULT
    controller.state.day = 2
    _execute(controller, "p4", day=2)
    provider = ScriptedProvider(summary="Player4が怪しいという意見が多数でした。")
    coordinator = AICoordinator(controller.state, ["p1"], provider, seed=1)

    asyncio.run(coordinator._generate_day_summary(controller, controller.state))

    summary = coordinator.day_summary_text(2)
    facts, opinion = summary.split(f"\n{OPINION_HEADING}\n")
    assert facts.startswith(f"{FACTS_HEADING}2日目")
    assert "- 処刑結果: Player4(p4)" in facts
    assert opinion == "Player4が怪しいという意見が多数でした。"
    # The factual half is generated from the ledger, so it repeats exactly.
    asyncio.run(coordinator._generate_day_summary(controller, controller.state))
    assert coordinator.day_summary_text(2) == summary
