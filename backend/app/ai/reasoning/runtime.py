"""The reasoning runtime: one object holding what every AI seat believes.

This is where the previous stages stop being libraries and start driving a game.
It owns a solver cache, a belief engine per seat and a deception state per
bluffer, and it answers the questions the coordinator used to spend an LLM call
on: who to vote for, who to look at tonight, who has something worth saying.

Those were never language problems. A vote is an argmax over evidence the engine
already holds; asking a model to redo it costs a request, adds latency, and
returns an answer nobody can audit. What is left for the model is the part it is
actually good at -- saying the thing in character.

Everything here is deterministic given the same board and seed.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.ai.reasoning.belief import (
    BeliefEngine,
    CognitiveTraits,
    CorrectionOutcome,
    DeceptionState,
    HypothesisRank,
    RankedView,
    StoryStatus,
    assign_traits,
    deception_state_for,
    parse_fact_corrections,
    profile_name,
    rank_hypotheses,
    refresh_story,
    summarise,
)
from app.ai.reasoning.belief.state import tiebreak
from app.ai.reasoning.facts import PublicFactLedger
from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import Perspective, PlayerPrivatePerspective
from app.ai.reasoning.solver import RoleSolver, SolverCache, build_solver
from app.engine.roles import RoleName
from app.engine.state import GameState

# How many seats open the day. More than this and the morning is a wall of text
# nobody reads; fewer and the board never gets stated.
MIN_OPENING_SPEAKERS = 3
MAX_OPENING_SPEAKERS = 5

# Speech-value contributions, in one table so the scheduler's priorities are
# readable rather than buried in branches.
VALUE_HAS_RESULT = 4.0
VALUE_PLANS_CLAIM = 3.5
VALUE_DIRECTLY_ASKED = 3.0
VALUE_UNDER_PRESSURE = 2.5
VALUE_BELIEF_MOVED = 2.0
VALUE_AGAINST_MAJORITY = 1.5
VALUE_NEW_EVIDENCE = 1.0
VALUE_STORY_COLLAPSED = 3.0


@dataclass
class SeatReasoning:
    """Everything the runtime keeps for one seat."""

    player_id: str
    traits: CognitiveTraits
    belief: BeliefEngine
    perspective: Perspective
    deception: DeceptionState | None = None
    solver: RoleSolver | None = None
    last_target: str | None = None
    last_scores: dict[str, float] = field(default_factory=dict)

    @property
    def trait_profile(self) -> str:
        return profile_name(self.traits)


@dataclass(frozen=True)
class SpeechCandidate:
    player_id: str
    value: float
    reasons: tuple[str, ...]


class ReasoningRuntime:
    """Per-session reasoning. One per game, holding one belief engine per AI."""

    def __init__(
        self,
        state: GameState,
        ai_player_ids: Sequence[str],
        *,
        seed: int | None = None,
        observer_player_ids: Sequence[str] = (),
    ) -> None:
        self._seed = seed
        self._rng = random.Random(f"{seed}:reasoning-runtime")
        self._observers = set(observer_player_ids)
        self._cache = SolverCache()
        self._traits = assign_traits(list(ai_player_ids), seed=seed)
        self.seats: dict[str, SeatReasoning] = {}
        for player_id in ai_player_ids:
            perspective = PlayerPrivatePerspective(player_id)
            self.seats[player_id] = SeatReasoning(
                player_id=player_id,
                traits=self._traits[player_id],
                belief=BeliefEngine(player_id, perspective, self._traits[player_id]),
                perspective=perspective,
            )
        self.observations = ObservationSet.from_state(state)
        self.correction_log: list[CorrectionOutcome] = []
        self._board_version = ""
        # Measurement, not behaviour. The roadmap's targets are claims about
        # this layer, and a claim you cannot measure is a hope.
        self.human_messages_considered = 0
        self.correction_belief_deltas: list[float] = []
        self.stale_premise_turns = 0
        self._minority_checks = 0
        self._minority_survivals = 0

    # -- refresh --

    def refresh(
        self, state: GameState, *, claimed_roles: dict[str, RoleName] | None = None
    ) -> None:
        """Rebuild solvers and beliefs for the current board.

        Skipped when nothing observable has changed -- the board version covers
        deaths, claims, verdicts and votes, so a no-op refresh really is one.
        """
        observations = ObservationSet.from_state(state)
        if observations.board_version == self._board_version:
            return
        self.observations = observations
        ledger = PublicFactLedger(state)
        for seat in self.seats.values():
            seat.last_scores = dict(seat.belief.state.wolf_scores)
            seat.last_target = seat.belief.state.current_execution_target
            seat.solver = build_solver(observations, seat.perspective, cache=self._cache)
            seat.belief.observe(ledger, seat.solver)
            claimed = (claimed_roles or {}).get(seat.player_id)
            if claimed is not None or seat.deception is not None:
                seat.deception = refresh_story(
                    deception_state_for(
                        seat.player_id,
                        observations,
                        claimed_role=claimed
                        or (seat.deception.claimed_role if seat.deception else None),
                    ),
                    observations,
                    cache=self._cache,
                )
        self._board_version = observations.board_version
        self._score_minority_persistence()

    def _score_minority_persistence(self) -> None:
        """Did the seats holding an unpopular line still hold it after the update?

        A minority opinion that evaporates the moment new information arrives was
        never an opinion; it was noise. This is the number that says which.
        """
        counts = self.target_distribution()
        if not counts:
            return
        majority = max(sorted(counts), key=lambda pid: counts[pid])
        for seat in self.seats.values():
            previous = seat.last_target
            if previous is None or previous == majority:
                continue
            self._minority_checks += 1
            if seat.belief.state.current_execution_target == previous:
                self._minority_survivals += 1

    # -- decisions that used to cost a request --

    def vote_decision(self, player_id: str, candidates: Sequence[str]) -> tuple[str, str]:
        """Pick a ballot and say why, from evidence already held.

        A vote is an argmax over the belief state. Asking a model to redo it
        buys nothing except a request and an answer nobody can audit.
        """
        seat = self.seats[player_id]
        eligible = [pid for pid in candidates if pid not in self._observers]
        if not eligible:
            eligible = list(candidates)
        salt = f"{seat.player_id}:vote"
        scored = sorted(
            eligible,
            # Utility first; the per-seat tiebreak only separates candidates the
            # evidence does not, so a table with nothing to go on still spreads.
            key=lambda pid: (-self._vote_utility(seat, pid), tiebreak(salt, pid), pid),
        )
        target = scored[0]
        return target, self._vote_reason(seat, target)

    def _vote_utility(self, seat: SeatReasoning, target_id: str) -> float:
        score = seat.belief.state.wolf_scores.get(target_id, 0.0)
        if seat.deception is not None:
            # Betrayal stays available, just expensive.
            score -= seat.deception.betrayal_cost(target_id)
        return score

    def _vote_reason(self, seat: SeatReasoning, target_id: str) -> str:
        reasons = seat.belief.state.reasons_for(target_id)
        if not reasons:
            return f"{target_id}が現時点で最も情報が少なく、他に有力な根拠がない。"
        explanations = [
            record.explanation
            for record in seat.belief.active_evidence()
            if record.evidence_id in reasons
        ]
        return "".join(explanations[:2])

    def night_target(
        self, player_id: str, action_type: str, candidates: Sequence[str]
    ) -> str | None:
        """Divine and attack chase suspicion; guard protects what is trusted."""
        seat = self.seats.get(player_id)
        eligible = [pid for pid in candidates if pid not in self._observers]
        if not eligible:
            return None
        if seat is None:
            return eligible[0]
        scores = seat.belief.state.wolf_scores
        salt = f"{seat.player_id}:{action_type}"
        if action_type == "guard":
            ranked = sorted(
                eligible, key=lambda pid: (scores.get(pid, 0.0), tiebreak(salt, pid), pid)
            )
        else:
            ranked = sorted(
                eligible, key=lambda pid: (-scores.get(pid, 0.0), tiebreak(salt, pid), pid)
            )
        return ranked[0]

    # -- the speech scheduler --

    def speech_candidates(
        self,
        state: GameState,
        *,
        pending_question_targets: Sequence[str] = (),
        pressured_ids: Sequence[str] = (),
        planned_claims: Sequence[str] = (),
    ) -> list[SpeechCandidate]:
        """Score who has something worth saying, and why.

        The reasons are kept alongside the number so a transcript can show why a
        seat was given the floor instead of only that it was.
        """
        ledger = PublicFactLedger(state)
        majority = self._majority_target(ledger)
        candidates: list[SpeechCandidate] = []
        for player_id, seat in self.seats.items():
            if not ledger.is_alive(player_id) or player_id in self._observers:
                continue
            value = 0.0
            reasons: list[str] = []
            if self._holds_unpublished_result(state, player_id):
                value += VALUE_HAS_RESULT
                reasons.append("未公開の能力結果を持つ")
            if player_id in planned_claims:
                value += VALUE_PLANS_CLAIM
                reasons.append("COする予定がある")
            if player_id in pending_question_targets:
                value += VALUE_DIRECTLY_ASKED
                reasons.append("名指しで質問された")
            if player_id in pressured_ids:
                value += VALUE_UNDER_PRESSURE
                reasons.append("主要処刑候補になっている")
            if self._belief_moved(seat):
                value += VALUE_BELIEF_MOVED
                reasons.append("疑い度が大きく動いた")
            target = seat.belief.state.current_execution_target
            if majority is not None and target is not None and target != majority:
                value += VALUE_AGAINST_MAJORITY
                reasons.append("多数派と異なる候補を推している")
            if self._has_fresh_evidence(seat):
                value += VALUE_NEW_EVIDENCE
                reasons.append("新しい独自証拠がある")
            if seat.deception is not None and seat.deception.status is StoryStatus.COLLAPSED:
                value += VALUE_STORY_COLLAPSED
                reasons.append("公開物語が破綻し方針転換が要る")
            candidates.append(
                SpeechCandidate(player_id=player_id, value=value, reasons=tuple(reasons))
            )
        return sorted(candidates, key=lambda item: (-item.value, item.player_id))

    def select_opening_speakers(
        self,
        state: GameState,
        *,
        pending_question_targets: Sequence[str] = (),
        pressured_ids: Sequence[str] = (),
        planned_claims: Sequence[str] = (),
        limit: int = MAX_OPENING_SPEAKERS,
    ) -> list[str]:
        """The three to five seats that open the day, in a seeded display order.

        Everyone forms an opinion; only the ones with something new to say get
        the floor. The rest speak when they are answered or challenged.
        """
        candidates = self.speech_candidates(
            state,
            pending_question_targets=pending_question_targets,
            pressured_ids=pressured_ids,
            planned_claims=planned_claims,
        )
        if not candidates:
            return []
        bounded = max(MIN_OPENING_SPEAKERS, min(limit, MAX_OPENING_SPEAKERS))
        chosen = [item.player_id for item in candidates[:bounded]]
        order_rng = random.Random(f"{self._seed}:opening:{state.day}")
        order_rng.shuffle(chosen)
        return chosen

    def _majority_target(self, ledger: PublicFactLedger) -> str | None:
        days = sorted({vote.day for vote in ledger.votes()})
        if not days:
            return None
        counts: dict[str, int] = {}
        for vote in ledger.votes_on(days[-1]):
            counts[vote.target_id] = counts.get(vote.target_id, 0) + 1
        return max(sorted(counts), key=lambda pid: counts[pid]) if counts else None

    def _holds_unpublished_result(self, state: GameState, player_id: str) -> bool:
        published = {
            (result.result_type, result.target_id)
            for result in PublicFactLedger(state).public_results()
            if result.claimant_id == player_id
        }
        held = {("seer", r.target_id) for r in state.divine_records if r.seer_id == player_id}
        held |= {
            ("medium", r.target_id) for r in state.medium_records if r.medium_id == player_id
        }
        return bool(held - published)

    def _belief_moved(self, seat: SeatReasoning) -> bool:
        current = seat.belief.state.wolf_scores
        return any(
            abs(current.get(pid, 0.0) - score) > 0.5 for pid, score in seat.last_scores.items()
        ) or seat.belief.state.current_execution_target != seat.last_target

    def _has_fresh_evidence(self, seat: SeatReasoning) -> bool:
        target = seat.belief.state.current_execution_target
        return bool(target and seat.belief.state.reasons_for(target))

    # -- human input --

    def apply_human_message(
        self, state: GameState, speaker_id: str, text: str
    ) -> list[CorrectionOutcome]:
        """Match a human message against the record, once, for the whole table.

        Corrections are checked in code and then applied to each seat
        separately, so one sentence costs zero model calls no matter how many
        AIs are listening -- and each of them updates on its own evidence.
        """
        ledger = PublicFactLedger(state)
        self.human_messages_considered += 1
        corrections = parse_fact_corrections(text, ledger, speaker_id)
        outcomes: list[CorrectionOutcome] = []
        for correction in corrections:
            for seat in self.seats.values():
                before = dict(seat.belief.state.wolf_scores)
                outcome = seat.belief.apply_correction(correction, ledger)
                after = seat.belief.state.wolf_scores
                delta = sum(
                    abs(after.get(pid, 0.0) - score)
                    for pid, score in before.items()
                    if abs(after.get(pid, 0.0) - score) < 1e6
                )
                if delta:
                    self.correction_belief_deltas.append(delta)
                if outcome.changed_anything or not outcome.verdict.is_confirmed:
                    outcomes.append(outcome)
        self.correction_log.extend(outcomes)
        return outcomes

    def seats_that_moved(self, outcomes: Sequence[CorrectionOutcome]) -> list[str]:
        """Which AIs actually had a reason withdrawn -- the ones worth hearing from."""
        moved = {
            seat.player_id
            for seat in self.seats.values()
            for outcome in outcomes
            if outcome.changed_anything
            and any(
                evidence_id in {record.evidence_id for record in seat.belief.evidence}
                for evidence_id in outcome.invalidated_evidence_ids
            )
        }
        return sorted(moved)

    # -- rendering for prompts --

    def ranked_view(self, player_id: str) -> tuple[RankedView, ...]:
        return rank_hypotheses(
            self.seats[player_id].belief.state.active_hypotheses,
            self.seats[player_id].traits,
        )

    def hypothesis_summary(self, player_id: str) -> str:
        """Bands, never percentages -- see `belief/ranking.py`."""
        return summarise(self.ranked_view(player_id))

    def top_rank(self, player_id: str) -> HypothesisRank | None:
        views = self.ranked_view(player_id)
        return views[0].rank if views else None

    def conflict_points(self, state: GameState) -> list[str]:
        """The day's disagreements, from structured events rather than prose."""
        ledger = PublicFactLedger(state)
        points: list[str] = []
        by_role: dict[RoleName, list[str]] = {}
        for claim in ledger.co_declarations():
            by_role.setdefault(claim.claimed_role, []).append(claim.player_id)
        for role, claimants in sorted(by_role.items()):
            if len(claimants) > 1:
                points.append(f"{role.value}CO対抗: {'、'.join(sorted(claimants))}")
        for result in ledger.public_results():
            colour = "黒" if result.is_werewolf else "白"
            points.append(
                f"{result.claimant_id}→{result.target_id}={colour}({result.day}日目)"
            )
        targets = {
            seat.player_id: seat.belief.state.current_execution_target
            for seat in self.seats.values()
        }
        distinct = sorted({target for target in targets.values() if target})
        if len(distinct) > 1:
            points.append(f"処刑候補が割れている: {'、'.join(distinct)}")
        return points

    # -- measurement --

    def target_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for seat in self.seats.values():
            target = seat.belief.state.current_execution_target
            if target is not None:
                counts[target] = counts.get(target, 0) + 1
        return counts

    def distribution_by_profile(self) -> dict[str, dict[str, int]]:
        by_profile: dict[str, dict[str, int]] = {}
        for seat in self.seats.values():
            target = seat.belief.state.current_execution_target
            if target is None:
                continue
            bucket = by_profile.setdefault(seat.trait_profile, {})
            bucket[target] = bucket.get(target, 0) + 1
        return by_profile

    def metrics(self) -> dict[str, float | int | dict[str, dict[str, int]]]:
        """Everything the evaluation harness needs from this layer."""
        return {
            "human_messages_considered": self.human_messages_considered,
            "seats_considering_each_human_message": (
                len(self.seats) if self.human_messages_considered else 0
            ),
            "corrections_applied": len(self.correction_log),
            "mean_belief_delta_after_correction": (
                round(
                    sum(self.correction_belief_deltas)
                    / len(self.correction_belief_deltas),
                    3,
                )
                if self.correction_belief_deltas
                else 0.0
            ),
            # Retraction happens inside `apply_correction`, so a refuted premise
            # never survives into a later turn by construction.
            "stale_premise_turns": self.stale_premise_turns,
            "minority_opinion_persistence": (
                round(self._minority_survivals / self._minority_checks, 3)
                if self._minority_checks
                else 0.0
            ),
            "opinion_spread": round(self.opinion_spread(), 3),
            "target_distribution_by_profile": self.distribution_by_profile(),
        }

    def opinion_spread(self) -> float:
        """Share of seats *not* on the most popular candidate.

        Zero means the table has collapsed onto one name, which is the failure
        mode this whole layer exists to avoid.
        """
        counts = self.target_distribution()
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return 1.0 - max(counts.values()) / total
