from __future__ import annotations

import asyncio

from app.ai.personalities import PERSONALITIES
from app.ai.player_agent import AIPlayerAgent
from app.ai.reasoning.facts import PublicFactLedger
from app.ai.reasoning.summaries import render_public_fact_summary
from app.ai.reasoning.validation import (
    detect_vote_intent_mismatch,
    sanitize_discussion_output,
    valid_action_targets,
    validate_public_result,
)
from app.ai.schemas import DiscussionOutput, PublicResultClaim, ReasoningMemo
from app.engine.roles import RoleName
from app.engine.state import (
    CoDeclaration,
    DeathCause,
    DeathRecord,
    GameState,
    PlayerState,
    VoteRecord,
)
from app.engine.state import (
    PublicResultClaim as StatePublicResultClaim,
)


def _state() -> GameState:
    return GameState(
        session_id="facts",
        day=3,
        players={
            "p0": PlayerState("p0", "Zero", RoleName.VILLAGER),
            "p1": PlayerState("p1", "One", RoleName.MEDIUM),
            "p2": PlayerState("p2", "Two", RoleName.WEREWOLF),
            "p11": PlayerState("p11", "Eleven", RoleName.SEER),
        },
    )


def _kill(state: GameState, player_id: str, cause: DeathCause, day: int) -> None:
    state.players[player_id].alive = False
    state.players[player_id].death_cause = cause
    state.players[player_id].death_day = day
    state.death_records.append(DeathRecord(player_id, cause, day))


def test_dead_players_are_excluded_from_later_execution_and_night_targets():
    state = _state()
    _kill(state, "p0", DeathCause.EXECUTED, 2)
    _kill(state, "p2", DeathCause.ATTACKED, 2)

    output = DiscussionOutput(
        public_message="test",
        reasoning_memo=ReasoningMemo(
            execution_target="p0", suspects=["p0", "p11"], fox_candidates=["p2", "p11"]
        ),
    )
    sanitize_discussion_output(state, "p1", output)

    assert output.reasoning_memo.execution_target is None
    assert valid_action_targets(state, "p1") == ["p11"]
    assert output.reasoning_memo.suspects == ["p0", "p11"]
    assert output.reasoning_memo.fox_candidates == ["p2", "p11"]


def test_exact_player_ids_do_not_confuse_p1_and_p11():
    state = _state()
    _kill(state, "p1", DeathCause.EXECUTED, 2)
    ledger = PublicFactLedger.from_state(state)

    assert "p1" in ledger.dead_ids
    assert "p11" in ledger.alive_ids
    assert valid_action_targets(state, "p0") == ["p11", "p2"]


def test_medium_result_for_non_executed_player_is_rejected():
    state = _state()
    state.co_declarations.append(CoDeclaration("p1", RoleName.MEDIUM, 1))
    claim = PublicResultClaim(result_type="medium", target_id="p2", is_werewolf=False)

    assert validate_public_result(state, "p1", claim) is False


def test_existing_white_result_cannot_be_reversed_to_black():
    state = _state()
    state.co_declarations.append(CoDeclaration("p1", RoleName.MEDIUM, 1))
    _kill(state, "p0", DeathCause.EXECUTED, 2)
    state.public_result_claims.append(StatePublicResultClaim("p1", "medium", "p0", False, 3))

    white = PublicResultClaim(result_type="medium", target_id="p0", is_werewolf=False)
    black = PublicResultClaim(result_type="medium", target_id="p0", is_werewolf=True)
    assert validate_public_result(state, "p1", white) is True
    assert validate_public_result(state, "p1", black) is False


def test_vote_ledger_uses_actual_target_and_detects_spoken_mismatch():
    state = _state()
    state.vote_records.append(VoteRecord("p0", "p11", day=2, round=1))
    ledger = PublicFactLedger.from_state(state)

    assert ledger.vote_target("p0", 2) == "p11"
    mismatch = detect_vote_intent_mismatch("p0", 2, "p2", "p11")
    assert mismatch is not None
    assert mismatch.intended_target_id == "p2"
    assert mismatch.actual_target_id == "p11"


def test_public_summary_is_deterministic_and_does_not_leak_private_roles():
    state = _state()
    state.co_declarations.append(CoDeclaration("p1", RoleName.MEDIUM, 1))
    first = render_public_fact_summary(PublicFactLedger.from_state(state))
    second = render_public_fact_summary(PublicFactLedger.from_state(state))

    assert first == second
    assert "mediumCO" in first
    assert "werewolf" not in first
    assert "seer" not in first


def test_final_day_candidates_contain_only_the_other_survivor():
    state = _state()
    _kill(state, "p0", DeathCause.EXECUTED, 1)
    _kill(state, "p2", DeathCause.ATTACKED, 2)
    assert valid_action_targets(state, "p1") == ["p11"]


class _InvalidTargetProvider:
    async def generate_structured(self, *, response_schema, **kwargs):
        return response_schema(target="dead", reason="invalid")


def test_invalid_night_target_is_repaired_deterministically():
    agent = AIPlayerAgent(_InvalidTargetProvider(), PERSONALITIES[0])
    result = asyncio.run(agent.generate_night_action("system", [], ["p11", "p2"]))
    assert result.target == "p11"
