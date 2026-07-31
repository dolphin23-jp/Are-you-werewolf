"""Vote collection, tally, and runoff handling."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from app.engine.state import DeathCause, DeathRecord, GameState, VoteRecord

DEFAULT_MAX_VOTE_ROUNDS = 4


@dataclass
class VoteTallyResult:
    executed_player_id: str | None = None
    tied_player_ids: list[str] | None = None
    is_draw: bool = False


class VoteManager:
    def __init__(self, max_vote_rounds: int = DEFAULT_MAX_VOTE_ROUNDS) -> None:
        self.max_vote_rounds = max_vote_rounds

    def record_vote(self, state: GameState, voter_id: str, target_id: str) -> None:
        voter = state.players[voter_id]
        target = state.players[target_id]
        if not voter.alive:
            raise ValueError(f"voter {voter_id} is not alive")
        if not target.alive:
            raise ValueError(f"target {target_id} is not alive")
        if voter_id == target_id:
            raise ValueError("cannot vote for self")
        if state.runoff_candidates and target_id not in state.runoff_candidates:
            raise ValueError(
                f"runoff is limited to {state.runoff_candidates}; {target_id} is not a candidate"
            )
        state.pending_votes[voter_id] = target_id

    def all_votes_in(self, state: GameState) -> bool:
        return len(state.pending_votes) >= len(state.alive_ids())

    def tally(self, state: GameState) -> VoteTallyResult:
        for voter_id, target_id in state.pending_votes.items():
            state.vote_records.append(
                VoteRecord(
                    voter_id=voter_id, target_id=target_id, day=state.day, round=state.vote_round
                )
            )

        counts = Counter(state.pending_votes.values())
        if not counts:
            state.pending_votes = {}
            return VoteTallyResult(is_draw=True)

        max_count = max(counts.values())
        top = [pid for pid, c in counts.items() if c == max_count]
        state.pending_votes = {}

        if len(top) == 1:
            executed_id = top[0]
            player = state.players[executed_id]
            player.alive = False
            player.death_cause = DeathCause.EXECUTED
            player.death_day = state.day
            state.death_records.append(
                DeathRecord(player_id=executed_id, cause=DeathCause.EXECUTED, day=state.day)
            )
            state.runoff_candidates = []
            return VoteTallyResult(executed_player_id=executed_id)

        if state.vote_round >= self.max_vote_rounds:
            state.runoff_candidates = []
            return VoteTallyResult(is_draw=True)

        # Narrow the next round to the tied players. Skipping this is what
        # made runoffs re-open the whole field and stall into a false draw.
        state.runoff_candidates = sorted(top)
        state.vote_round += 1
        return VoteTallyResult(tied_player_ids=top)
