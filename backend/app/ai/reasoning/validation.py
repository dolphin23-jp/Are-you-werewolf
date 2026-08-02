"""Deterministic validation of structured AI output before persistence."""

from __future__ import annotations

from dataclasses import dataclass

from app.ai.reasoning.facts import PublicFactLedger
from app.ai.schemas import DiscussionOutput, PublicResultClaim
from app.engine.roles import RoleName
from app.engine.state import GameState


@dataclass(frozen=True)
class VoteIntentMismatch:
    player_id: str
    day: int
    intended_target_id: str
    actual_target_id: str


def valid_action_targets(state: GameState, actor_id: str) -> list[str]:
    """Return stable, exact-ID targets; substring matching is never used."""
    return sorted(pid for pid in state.alive_ids() if pid != actor_id)


def sanitize_discussion_output(
    state: GameState, actor_id: str, output: DiscussionOutput
) -> DiscussionOutput:
    valid_players = set(state.players) - {actor_id}
    valid_targets = set(valid_action_targets(state, actor_id))
    memo = output.reasoning_memo
    memo.suspects = _unique_valid(memo.suspects, valid_players)
    memo.trusted = _unique_valid(memo.trusted, valid_players)
    memo.fox_candidates = _unique_valid(memo.fox_candidates, valid_players)
    if memo.trusted_seer not in valid_players:
        memo.trusted_seer = None
    if memo.execution_target not in valid_targets:
        memo.execution_target = None
    if output.alternative_execution_target not in valid_targets:
        output.alternative_execution_target = None
    return output


def validate_public_result(state: GameState, claimant_id: str, claim: PublicResultClaim) -> bool:
    if claim.result_type not in {"seer", "medium"} or claim.target_id not in state.players:
        return False
    expected_role = RoleName.SEER if claim.result_type == "seer" else RoleName.MEDIUM
    if not any(
        item.player_id == claimant_id and item.claimed_role == expected_role
        for item in state.co_declarations
    ):
        return False
    if claim.result_type == "medium" and not any(
        death.player_id == claim.target_id and death.cause.value == "executed"
        for death in state.death_records
    ):
        return False
    ledger = PublicFactLedger.from_state(state)
    prior = [
        item
        for item in ledger.results
        if item.claimant_id == claimant_id
        and item.result_type == claim.result_type
        and item.target_id == claim.target_id
    ]
    return not prior or all(item.is_werewolf == claim.is_werewolf for item in prior)


def detect_vote_intent_mismatch(
    player_id: str, day: int, intended_target_id: str | None, actual_target_id: str
) -> VoteIntentMismatch | None:
    if intended_target_id is None or intended_target_id == actual_target_id:
        return None
    return VoteIntentMismatch(player_id, day, intended_target_id, actual_target_id)


def _unique_valid(values: list[str], valid: set[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value in valid))
