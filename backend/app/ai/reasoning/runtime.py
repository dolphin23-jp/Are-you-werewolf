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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from app.ai.metrics import MetricsCollector
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
    profile_name,
    rank_hypotheses,
    refresh_story,
    summarise,
)
from app.ai.reasoning.belief.state import EvidenceRecord, tiebreak
from app.ai.reasoning.board_memo import fake_claim_floor, render_board_memo, role_label
from app.ai.reasoning.citations import VoteCitation, parse_vote_citations
from app.ai.reasoning.dialogue import (
    ArgumentEvent,
    BeliefChange,
    ConclusionType,
    DiscussionDecision,
    RequiredPublicResult,
    SpeechGoal,
    parse_argument,
)
from app.ai.reasoning.facts import PublicFactLedger
from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import (
    ClaimedStoryPerspective,
    CommonPublicPerspective,
    Perspective,
    PlayerPrivatePerspective,
)
from app.ai.reasoning.rendering import declared_candidates
from app.ai.reasoning.solver import (
    AccurateTimeline,
    RoleSolver,
    SolverCache,
    build_solver,
    has_role,
)
from app.ai.reasoning.timeline import disclosable_result_nights
from app.engine.roles import RoleName
from app.engine.state import ChatChannel, GameState

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

# What someone else's account of a ballot is worth before traits and trust. Low:
# a quote is weaker than the record, and this one is already known to disagree
# with it. It exists so that correcting it has something to retract.
CITED_VOTE_WEIGHT = 0.5

# How much a human's structured argument is worth as soft evidence, before the
# listener's traits and their trust in the speaker scale it.
_ARGUMENT_WEIGHTS: dict[ConclusionType, tuple[float, str]] = {
    ConclusionType.ACCUSATION: (0.9, "accusation"),
    ConclusionType.DEFENCE: (-0.9, "accusation"),
    ConclusionType.STRATEGIC_CLAIM: (0.4, "accusation"),
    ConclusionType.CLOSED_WORLD_CHALLENGE: (0.5, "accusation"),
}


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


@dataclass(frozen=True)
class PublicMark:
    """Something the whole table already holds against a seat.

    Either a black published on it, or a role claim the public record proves
    it cannot hold. Both are public: any seat could cite them, so a werewolf
    voting a partner marked this way is distancing itself from a lost cause,
    not betraying a teammate nobody suspected. The id is stable so an audit
    can point at the mark it relied on.
    """

    evidence_id: str
    source_message_id: str
    explanation: str


class ReasoningRuntime:
    """Per-session reasoning. One per game, holding one belief engine per AI."""

    def __init__(
        self,
        state: GameState,
        ai_player_ids: Sequence[str],
        *,
        seed: int | None = None,
        observer_player_ids: Sequence[str] = (),
        metrics: MetricsCollector | None = None,
    ) -> None:
        self._seed = seed
        self._rng = random.Random(f"{seed}:reasoning-runtime")
        self._observers = set(observer_player_ids)
        self._cache = SolverCache()
        self._metrics_collector = metrics
        self._reported_solver_queries = 0
        self._reported_solver_hits = 0
        self._reported_solver_seconds = 0.0
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
        # What each seat last told the table, and on which day. A vote is
        # compared against this, never against another internal candidate --
        # and only against what was said that day: yesterday's candidate was
        # argued on a board that no longer exists, so reading it back as today's
        # plan reports a change of mind nobody made.
        self.stated_targets: dict[str, tuple[int, str | None]] = {}
        self.argument_log: list[ArgumentEvent] = []
        self.vote_citations: list[VoteCitation] = []
        self.reassessment_queue: list[str] = []
        self.correction_belief_deltas: list[float] = []
        # Heard vs applied are different numbers and only one of them is a
        # quality claim: a correction that corrected nothing corrected nothing.
        self.corrections_heard = 0
        self.stale_premise_turns = 0
        # Seat-by-message evaluations that actually happened, not seats times
        # messages. See `_apply_argument`.
        self._seat_evaluations = 0
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
        bluffs = self._bluffed_roles(ledger, observations)
        # What the table can settle with no seat's private knowledge. Computed
        # once per board and shared, so each seat can tell a conclusion anyone
        # could argue from one only it can reach.
        public_solver = build_solver(observations, CommonPublicPerspective(), cache=self._cache)
        public_certainties = {
            player_id: public_solver.assess(has_role(player_id, RoleName.WEREWOLF))
            for player_id in observations.player_ids
        }
        for seat in self.seats.values():
            seat.last_scores = dict(seat.belief.state.public_suspicion_scores)
            seat.last_target = seat.belief.state.current_execution_target
            seat.solver = build_solver(observations, seat.perspective, cache=self._cache)
            seat.belief.observe(
                ledger, seat.solver, observations, public_certainties=public_certainties
            )
            claimed = (claimed_roles or {}).get(seat.player_id) or bluffs.get(
                seat.player_id
            )
            if claimed is not None or seat.deception is not None:
                seat.deception = refresh_story(
                    deception_state_for(
                        seat.player_id,
                        observations,
                        claimed_role=claimed
                        or (seat.deception.claimed_role if seat.deception else None),
                    ),
                    observations,
                    # The bluffer's own account, taken at its stated timing. A
                    # story with a night that cannot have happened is a story
                    # that will not survive the table noticing, and the bluffer
                    # wants to know that before they are asked.
                    assumptions=(AccurateTimeline(seat.player_id),),
                    cache=self._cache,
                )
        self._board_version = observations.board_version
        self._score_minority_persistence()

    def _bluffed_roles(
        self, ledger: PublicFactLedger, observations: ObservationSet
    ) -> dict[str, RoleName]:
        """Seats whose standing CO is not the card they are holding.

        This is what finally connects `ClaimedStoryPerspective` to a real game:
        before, a public story existed only when a caller passed `claimed_roles`
        by hand, and nothing in production ever did, so wolves fake-claiming seer
        reasoned about their bluff with their real card in view.

        No leak: each seat is compared against *its own* knowledge -- the card it
        was dealt and the CO it made. Nobody learns anything about anybody else.
        """
        bluffs: dict[str, RoleName] = {}
        for player_id in self.seats:
            claimed = ledger.claimed_role_of(player_id)
            if claimed is None:
                continue
            own = observations.seat_knowledge(player_id).self_role
            if claimed != own:
                bluffs[player_id] = claimed
        return bluffs

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
        # The *same* salt the belief state uses to pick `current_execution_target`.
        # A different one here would let the stated candidate and the ballot
        # diverge purely from tie-breaking, which reads as a change of mind that
        # nobody made.
        salt = f"{seat.player_id}:execution"
        scored = sorted(
            eligible,
            # Utility first; the per-seat tiebreak only separates candidates the
            # evidence does not, so a table with nothing to go on still spreads.
            key=lambda pid: (-self._vote_utility(seat, pid), tiebreak(salt, pid), pid),
        )
        target = scored[0]
        return target, self._vote_reason(seat, target)

    def _vote_utility(self, seat: SeatReasoning, target_id: str) -> float:
        """Faction-aware execution utility, not raw suspicion.

        A werewolf knows its partners are wolves and wants them alive; a
        villager knows nothing and wants the suspicious one gone. Reading both
        off one number is what made wolves vote for their own team.
        """
        score = float(seat.belief.state.execution_utility_scores.get(target_id, 0.0))
        if seat.deception is not None:
            score -= seat.deception.betrayal_cost(target_id)
        return score

    def _vote_reason(self, seat: SeatReasoning, target_id: str) -> str:
        reasons = seat.belief.state.reasons_for(target_id)
        if not reasons:
            return f"{target_id}が現時点で最も情報が少なく、他に有力な根拠がない。"
        explanations = [
            record.explanation
            for record in seat.belief.public_argument_evidence_for(target_id)
            if record.evidence_id in reasons
        ]
        if not explanations:
            # Every reason is private. Say so neutrally rather than returning an
            # empty reason, and never reach for the private one to fill the gap.
            return f"{target_id}への疑いが現時点で最も強い。"
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
        # One utility per action. Reusing suspicion for all three is what made
        # the seer re-divine the same seat and the wolves bite their own cover.
        scores = {
            "divine": seat.belief.state.divine_utility_scores,
            "guard": seat.belief.state.guard_utility_scores,
            "attack": seat.belief.state.attack_utility_scores,
        }.get(action_type, seat.belief.state.execution_utility_scores)
        salt = f"{seat.player_id}:{action_type}"
        ranked = [
            pid
            for pid in sorted(
                eligible,
                key=lambda pid: (-scores.get(pid, 0.0), tiebreak(salt, pid), pid),
            )
            if scores.get(pid, 0.0) != float("-inf")
        ]
        return ranked[0] if ranked else None

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
            if self.holds_unpublished_result(state, player_id):
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

    def holds_unpublished_result(self, state: GameState, player_id: str) -> bool:
        """Whether this seat is sitting on a result the table has not heard.

        Public because the coordinator uses it to guarantee a duty speaker is
        never dropped by the value ranking -- a seer who cannot get a word in
        is a lost game, not a quiet morning.
        """
        return bool(self.unpublished_results(state, player_id))

    def required_claim_speakers(
        self,
        state: GameState,
        *,
        planned_fake_roles: Mapping[str, RoleName] | None = None,
        freemason_leader: str | None = None,
    ) -> tuple[str, ...]:
        """Seats with a claim to make, change or correct -- never dropped from the day.

        A claim that has been planned but never gets a turn is worse than one
        never planned: the freemason plan existed and the leader simply was not
        picked, so nobody COed. Each source below is a claim this seat *must*
        say out loud, and the value ranking is not allowed to outvote it.

        Sources, all read from state the seat legitimately has:

        * the freemason leader, until the freemason CO is made;
        * a seer or medium holding a result without the CO it needs;
        * a planned fake claimant, until that claim is actually made;
        * a standing seer or medium claimant, true or fake, with this
          morning's result still to announce (see `owes_morning_result`);
        * a bluffer whose story has collapsed and must retract or slide;
        * a seer or medium whose published colour contradicts its own result.
        """
        ledger = PublicFactLedger(state)
        observations = ObservationSet.from_state(state)
        required: list[str] = []

        def alive_ai(player_id: str) -> bool:
            return player_id in self.seats and state.players[player_id].alive

        if freemason_leader and alive_ai(freemason_leader):
            if ledger.claimed_role_of(freemason_leader) is not RoleName.FREEMASON:
                required.append(freemason_leader)
        for player_id, seat in self.seats.items():
            if not alive_ai(player_id):
                continue
            standing = ledger.claimed_role_of(player_id)
            unpublished = self.unpublished_results(state, player_id)
            if unpublished and self._claim_needed_for(state, player_id, unpublished):
                required.append(player_id)
            planned = (planned_fake_roles or {}).get(player_id)
            if planned is not None and standing is not planned:
                required.append(player_id)
            if self.owes_morning_result(state, player_id):
                required.append(player_id)
            if seat.deception is not None and seat.deception.status is StoryStatus.COLLAPSED:
                required.append(player_id)
            if self._published_colour_contradicts_own(ledger, observations, seat):
                required.append(player_id)
        return tuple(dict.fromkeys(required))

    def owes_morning_result(self, state: GameState, player_id: str) -> bool:
        """A living seer or medium claimant, true or fake, with today's result unsaid.

        The table expects every standing seer to report last night's look and
        every standing medium to report yesterday's execution, first thing in
        the morning. A real holder is already a duty speaker through its held
        result; a fake one used to be one only until its CO, after which its
        results arrived late or never and nobody noticed (the first Claude
        game's madman seer reported at line 10 on day 2, at line 6 on day 4
        after three requests, and not at all on day 3). The claimed role sets
        the schedule, so the rule is the same whatever the seat really holds.

        Read from public facts only -- the standing claim, the published
        results and the public calendar -- plus the seat's own held results.

        The published nights are read from the observations, not the raw
        ledger. The ledger dates a verdict repeated today without a night as a
        result for last night; the projection dates it by the statement it
        repeats, as the board memo does. Saying yesterday's white again is not
        this morning's result, so it must not cancel the duty while the memo
        still shows last night as 未公開.
        """
        if player_id not in self.seats or not state.players[player_id].alive:
            return False
        ledger = PublicFactLedger(state)
        claimed = ledger.claimed_role_of(player_id)
        if claimed not in (RoleName.SEER, RoleName.MEDIUM):
            return False
        if self.holds_unpublished_result(state, player_id):
            return True
        last_night = state.day - 1
        if last_night < 0:
            return False
        observations = ObservationSet.from_state(state)
        published = observations.verdicts_by(player_id)
        if claimed is RoleName.SEER:
            if last_night not in disclosable_result_nights(observations, player_id, state.day):
                return False
            return not any(
                r.result_type == "seer" and r.source_night == last_night for r in published
            )
        executed = {
            pid for pid in ledger.executed_ids() if ledger.execution_day(pid) == last_night
        }
        return bool(executed) and not any(
            r.result_type == "medium" and r.target_id in executed for r in published
        )

    def impossible_claim_reasons(self, state: GameState, player_id: str) -> tuple[str, ...]:
        """Why the public record proves this seat does not hold the role it claims.

        Empty when it might. The test is the claimant's own public story taken
        at its word -- the claimed card, and the looks it published as the
        looks it took -- run through the same hard rules every seat uses. If no
        world fits, nobody needs private knowledge to call the claim fake:
        a seer CO whose night-1 look survived a two-body night (one body a
        cursed fox, so the real seer looked at one of the dead) is out. That
        was ダイキ in the first Claude game, provable from day 2 on.
        """
        claimed = PublicFactLedger(state).claimed_role_of(player_id)
        if claimed is None:
            return ()
        self.refresh(state)
        solver = build_solver(
            self.observations, ClaimedStoryPerspective(player_id, claimed), cache=self._cache
        )
        result = solver.explain_contradiction()
        return tuple(result.explanations) if result.is_contradictory else ()

    def public_marks(self, state: GameState, player_id: str) -> tuple[PublicMark, ...]:
        """Everything public that already marks this seat: blacks, a disproved claim."""
        ledger = PublicFactLedger(state)
        marks = [
            PublicMark(
                evidence_id=f"black:{result.claimant_id}:{result.result_type}:{player_id}",
                source_message_id=result.source_message_id,
                explanation=(
                    f"{result.claimant_id}の{'霊媒' if result.result_type == 'medium' else '占い'}"
                    f"結果で{player_id}は黒"
                ),
            )
            for result in ledger.public_results()
            if result.target_id == player_id and result.is_werewolf
        ]
        reasons = self.impossible_claim_reasons(state, player_id)
        if reasons:
            claim = next(
                (c for c in ledger.co_declarations() if c.player_id == player_id), None
            )
            role = claim.claimed_role.value if claim is not None else "unknown"
            marks.append(
                PublicMark(
                    evidence_id=f"impossible_claim:{player_id}:{role}",
                    source_message_id=claim.source_message_id if claim is not None else "",
                    explanation="".join(reasons),
                )
            )
        return tuple(marks)

    def _published_colour_contradicts_own(
        self, ledger: PublicFactLedger, observations: ObservationSet, seat: SeatReasoning
    ) -> bool:
        own = {
            ("seer", r.target_id): r.is_werewolf
            for r in seat.perspective.known_divine_results(observations)
        }
        own |= {
            ("medium", r.target_id): r.is_werewolf
            for r in seat.perspective.known_medium_results(observations)
        }
        return any(
            own.get((result.result_type, result.target_id), result.is_werewolf)
            != result.is_werewolf
            for result in ledger.public_results()
            if result.claimant_id == seat.player_id
        )

    def unpublished_results(
        self, state: GameState, player_id: str
    ) -> tuple[RequiredPublicResult, ...]:
        """This seat's own results that it has not yet published, oldest first.

        Read through the seat's perspective, which is the only legal route to a
        private result -- so a seat can never be told to publish someone else's.
        One method feeds both the speech goal and the requirement, so "you have
        something to publish" and "here is what to publish" cannot disagree.

        "Published" is keyed on the target, not the night: a result already out
        under a free-text claim with no stated night would otherwise look
        unpublished forever and be repeated every turn.
        """
        seat = self.seats.get(player_id)
        if seat is None:
            return ()
        observations = ObservationSet.from_state(state)
        published = {
            (result.result_type, result.target_id)
            for result in PublicFactLedger(state).public_results()
            if result.claimant_id == player_id
        }
        held = [
            RequiredPublicResult("seer", r.target_id, r.is_werewolf, r.night)
            for r in seat.perspective.known_divine_results(observations)
        ] + [
            RequiredPublicResult("medium", r.target_id, r.is_werewolf, r.day)
            for r in seat.perspective.known_medium_results(observations)
        ]
        return tuple(
            sorted(
                (item for item in held if (item.result_type, item.target_id) not in published),
                key=lambda item: (item.referenced_day, item.result_type, item.target_id),
            )
        )

    def _belief_moved(self, seat: SeatReasoning) -> bool:
        current = seat.belief.state.public_suspicion_scores
        return any(
            abs(current.get(pid, 0.0) - score) > 0.5 for pid, score in seat.last_scores.items()
        ) or seat.belief.state.current_execution_target != seat.last_target

    def _has_fresh_evidence(self, seat: SeatReasoning) -> bool:
        target = seat.belief.state.current_execution_target
        return bool(target and seat.belief.state.reasons_for(target))

    def _count_stale_premises(self, seat: SeatReasoning, target: str | None) -> None:
        """Is this turn about to argue from a reason that has been withdrawn?

        Measured, not asserted. The obvious way to report this number is to
        return zero and note that retraction happens inside `apply_correction`,
        so it cannot happen -- but a metric that cannot move is not evidence
        that the thing it names does not occur. This checks the reasons the
        seat is *actually* about to cite against the records behind them.
        """
        if target is None:
            return
        active = {
            record.evidence_id for record in seat.belief.active_evidence()
        }
        if any(
            evidence_id not in active
            for evidence_id in seat.belief.state.reasons_for(target)
        ):
            self.stale_premise_turns += 1

    # -- what a turn is allowed to say --

    def discussion_decision(
        self,
        state: GameState,
        player_id: str,
        *,
        pending_question: bool = False,
        under_pressure: bool = False,
    ) -> DiscussionDecision:
        """Fix this turn's conclusions before any wording exists.

        The model receives this and renders it. Letting it re-derive the target
        is what allowed a day of arguing for one name and a ballot for another.
        """
        self.refresh(state)
        seat = self.seats[player_id]
        belief = seat.belief.state
        target = belief.current_execution_target
        self._count_stale_premises(seat, target)
        supporting = tuple(
            record.explanation
            for record in seat.belief.public_argument_evidence_for(target)
            if target is not None and record.evidence_id in belief.reasons_for(target)
        )[:3]
        counter = tuple(
            record.explanation
            for record in seat.belief.public_argument_evidence_for(target)
            if record.subject_id == target and record.weight < 0
        )[:2]
        rank = self.top_rank(player_id)
        speech_goal = self._speech_goal(
            state,
            seat,
            pending_question=pending_question,
            under_pressure=under_pressure,
        )
        required: tuple[RequiredPublicResult, ...] = ()
        required_claim: RoleName | None = None
        if speech_goal is SpeechGoal.PUBLISH_RESULT:
            required = self.unpublished_results(state, player_id)
            required_claim = self._claim_needed_for(state, player_id, required)
        return DiscussionDecision(
            speaker_id=player_id,
            execution_target=target,
            alternative_target=belief.alternative_target,
            target_confidence_band=rank.value if rank is not None else "unranked",
            supporting_evidence=supporting,
            counter_evidence=counter,
            belief_changes=self._belief_changes(seat),
            strongest_countercase=self._countercase(seat, target),
            public_story_status=(
                seat.deception.status.value if seat.deception is not None else None
            ),
            speech_goal=speech_goal,
            required_public_results=required,
            required_claim_role=required_claim,
        )

    def _claim_needed_for(
        self,
        state: GameState,
        player_id: str,
        results: Sequence[RequiredPublicResult],
    ) -> RoleName | None:
        """The CO a result needs behind it, if the seat has not made it.

        A verdict with no claim is incoherent to the table, and worse, the result
        validator drops a result whose role contradicts a standing claim -- so a
        seer who had slid elsewhere would lose the result without a trace. The
        role asked for is the one the result itself proves the seat holds.
        """
        if not results:
            return None
        role = RoleName.SEER if results[0].result_type == "seer" else RoleName.MEDIUM
        if PublicFactLedger(state).claimed_role_of(player_id) is role:
            return None
        return role

    def _belief_changes(self, seat: SeatReasoning) -> tuple[BeliefChange, ...]:
        current = seat.belief.state.public_suspicion_scores
        changes = []
        for subject, before in sorted(seat.last_scores.items()):
            after = current.get(subject, 0.0)
            if abs(after - before) <= 0.5:
                continue
            reasons = seat.belief.state.reasons_for(subject)
            # This text goes into the brief, and from the brief into speech. A
            # change driven by something only this seat knows -- a private black,
            # the wolf roster -- is voiced without its reason, not with it.
            explanation = next(
                (
                    record.explanation
                    for record in seat.belief.public_argument_evidence_for(subject)
                    if record.evidence_id in reasons
                ),
                "根拠が更新された",
            )
            changes.append(
                BeliefChange(
                    subject_id=subject, before=before, after=after, reason=explanation
                )
            )
        return tuple(changes[:3])

    def _countercase(self, seat: SeatReasoning, target: str | None) -> str:
        if target is None:
            return ""
        alternative = seat.belief.state.alternative_target
        if alternative is None:
            return f"{target}を疑う独立した根拠がまだ薄い。"
        return f"{alternative}の方が疑わしいという見方も残っている。"

    def _speech_goal(
        self,
        state: GameState,
        seat: SeatReasoning,
        *,
        pending_question: bool,
        under_pressure: bool,
    ) -> SpeechGoal:
        if seat.deception is not None and seat.deception.status is StoryStatus.COLLAPSED:
            return SpeechGoal.RECOVER_STORY
        if self.holds_unpublished_result(state, seat.player_id):
            return SpeechGoal.PUBLISH_RESULT
        if pending_question:
            return SpeechGoal.ANSWER_QUESTION
        if under_pressure:
            return SpeechGoal.DEFEND
        if self._belief_moved(seat):
            return SpeechGoal.REASSESS
        if seat.belief.state.current_execution_target is not None:
            return SpeechGoal.PRESS_CANDIDATE
        return SpeechGoal.OBSERVE

    def board_memo(self, state: GameState, player_id: str) -> str:
        """The board as this seat's notes, for the engine that lets the model decide.

        Public facts and public logic come from the ledger and a solver with no
        private knowledge; the final section is the seat's own card, results and
        allies, read through its perspective. The seat's private solver adds
        only hard conclusions. Nothing here is a conclusion about whom to
        suspect -- that is what the model is being asked for.
        """
        self.refresh(state)
        seat = self.seats[player_id]
        public_solver = build_solver(
            self.observations, CommonPublicPerspective(), cache=self._cache
        )
        return render_board_memo(
            state,
            player_id,
            observations=self.observations,
            public_solver=public_solver,
            seat_solver=seat.solver,
        )

    def record_stated_target(self, player_id: str, target: str | None, day: int) -> None:
        """Remember what a seat told the table on `day`, so the ballot is checked
        against the statement rather than against another internal number."""
        self.stated_targets[player_id] = (day, target)

    def stated_target(self, player_id: str, day: int | None = None) -> str | None:
        """The seat's last stated candidate; with `day`, only one stated that day.

        Every caller that compares or shows the statement passes the day. The
        undated form is the last thing the seat said, whenever it said it.
        """
        entry = self.stated_targets.get(player_id)
        if entry is None:
            return None
        stated_day, target = entry
        if day is not None and stated_day != day:
            return None
        return target

    def stated_targets_on(self, day: int) -> dict[str, str]:
        """Every seat's candidate as stated on `day`, seats that named nobody left out."""
        return {
            player_id: target
            for player_id, (stated_day, target) in self.stated_targets.items()
            if stated_day == day and target is not None
        }

    # -- human input --

    def apply_human_message(
        self,
        state: GameState,
        speaker_id: str,
        text: str,
        source_message_id: str = "",
    ) -> list[CorrectionOutcome]:
        """Match a human message against the record, once, for the whole table.

        Corrections are checked in code and then applied to each seat
        separately, so one sentence costs zero model calls no matter how many
        AIs are listening -- and each of them updates on its own evidence.
        """
        ledger = PublicFactLedger(state)
        self.human_messages_considered += 1
        argument = parse_argument(text, ledger, speaker_id, source_message_id)
        if argument is not None:
            self.argument_log.append(argument)
            self._seat_evaluations += self._apply_argument(argument, ledger)
        corrections = list(argument.factual_claims) if argument is not None else []
        self.corrections_heard += len(corrections)
        outcomes: list[CorrectionOutcome] = []
        for correction in corrections:
            for seat in self.seats.values():
                self._seat_evaluations += 1
                before = dict(seat.belief.state.public_suspicion_scores)
                outcome = seat.belief.apply_correction(correction, ledger)
                after = seat.belief.state.public_suspicion_scores
                delta = sum(
                    abs(after.get(pid, 0.0) - score)
                    for pid, score in before.items()
                    if abs(after.get(pid, 0.0) - score) < 1e6
                )
                if delta:
                    self.correction_belief_deltas.append(delta)
                outcome = replace(
                    outcome,
                    seat_id=seat.player_id,
                    correction_id=correction.correction_id,
                    belief_delta=delta,
                )
                outcomes.append(outcome)
        self.correction_log.extend(outcomes)
        return outcomes

    def _apply_argument(self, argument: ArgumentEvent, ledger: PublicFactLedger) -> int:
        """Let every seat weigh the same argument on its own terms.

        Facts are corrected identically for everyone -- the record is the record.
        A *strategic* claim is advice, and how persuasive it is depends on who
        said it and who is listening, so it enters as ordinary soft evidence and
        each seat's traits and trust do the rest.

        Returns how many seats actually took the argument into account, which is
        the number the harness reports. Assuming every seat did -- because the
        message existed -- would make the metric unable to notice the case it is
        there to catch.
        """
        subject = argument.conclusion_target_id
        if subject is None or argument.conclusion_type is ConclusionType.FACT_CORRECTION:
            return 0
        weight, category = _ARGUMENT_WEIGHTS.get(
            argument.conclusion_type, (0.0, "accusation")
        )
        if weight == 0.0:
            return 0
        evaluated = 0
        for seat in self.seats.values():
            if seat.player_id == argument.speaker_id:
                continue
            trust = seat.belief.state.source_trust.get(argument.speaker_id, 0.0)
            scaled = weight * argument.rhetorical_strength * (1.0 + 0.5 * trust)
            seat.belief.add_evidence(
                EvidenceRecord(
                    evidence_id=f"{argument.argument_id}:{subject}",
                    subject_id=subject,
                    category=category,
                    source_event_ids=(
                        argument.source_message_id or argument.argument_id,
                    ),
                    weight=scaled,
                    explanation=(
                        f"{argument.speaker_id}の主張"
                        f"({argument.conclusion_type.value}): {argument.premises[0].text}"
                        if argument.premises
                        else f"{argument.speaker_id}の主張"
                    ),
                )
            )
            seat.belief.recompute(ledger)
            evaluated += 1
        return evaluated

    def record_public_speech(
        self, state: GameState, speaker_id: str, text: str, source_message_id: str = ""
    ) -> tuple[VoteCitation, ...]:
        """Absorb one public message's account of who voted for whom.

        Only *inaccurate* citations become evidence, keyed on the ballot as
        cited. Repeating the record correctly says nothing about anybody, and
        recording it would put a reason in every seat that no correction could
        ever be about.

        The listener does not silently check the record and dismiss the quote --
        taking a confident misquote at face value is on the list of errors an AI
        is allowed to make. What is not allowed is keeping it after someone
        produces the ballot.
        """
        ledger = PublicFactLedger(state)
        citations = parse_vote_citations(text, ledger, speaker_id, source_message_id)
        wrong = [
            citation
            for citation in citations
            if citation.is_verifiable and not citation.is_accurate
        ]
        for citation in wrong:
            for seat in self.seats.values():
                if seat.player_id == speaker_id:
                    continue
                seat.belief.add_evidence(
                    EvidenceRecord(
                        evidence_id=f"cited:{citation.fact_id}:{speaker_id}",
                        subject_id=citation.voter_id,
                        category="misremembered_vote",
                        source_event_ids=(citation.fact_id,),
                        weight=CITED_VOTE_WEIGHT,
                        explanation=(
                            f"{speaker_id}によれば、{citation.day}日目に"
                            f"{citation.voter_id}は{citation.cited_target_id}へ"
                            "投票したとのこと。"
                        ),
                    )
                )
                seat.belief.recompute(ledger)
        self.vote_citations.extend(citations)
        return citations

    def queue_reassessment_speakers(self, player_ids: Sequence[str]) -> None:
        """Seats whose reasons were withdrawn owe the table an explanation."""
        for player_id in player_ids:
            if player_id in self.seats and player_id not in self.reassessment_queue:
                self.reassessment_queue.append(player_id)

    def take_reassessment_speakers(self) -> list[str]:
        queued, self.reassessment_queue = list(self.reassessment_queue), []
        return queued

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

    def conflict_points(
        self, state: GameState, *, stated_targets_are_public: bool = True
    ) -> list[str]:
        """The day's disagreements, from structured public events rather than prose.

        Every seat reads the day summary this feeds, so nothing here may come
        from what a seat privately believes. The split in execution candidates
        used to be read off each seat's belief engine, which weighs what only
        that seat knows -- a seer's unpublished black, a wolf's own team -- and
        was handed to the whole table as if someone had said it. Only
        candidates a seat named in public today count now -- and named as a
        candidate, in a sentence that puts them forward for execution. A seat
        that only mentioned its private target, to clear or question them, has
        not told the table it wants them hanged.

        Under v3 even that is not enough: the stated target is the model's
        private plan note, and a keyword match cannot tell "X を吊りたい" from
        "X は吊るべきではない". So the split is reported only when stated
        targets are themselves public -- v2, where code writes the candidate
        into the message -- and v3 seats read the disagreement from the log.
        """
        ledger = PublicFactLedger(state)
        points: list[str] = []
        by_role: dict[RoleName, list[str]] = {}
        for claim in ledger.co_declarations():
            by_role.setdefault(claim.claimed_role, []).append(claim.player_id)
        for role, claimants in sorted(by_role.items()):
            # The same rule the board memo states: a contest is more claimants
            # than the role has seats, so a freemason pair is not one.
            if fake_claim_floor(role, len(claimants)):
                points.append(f"{role_label(role)}CO対抗: {'、'.join(sorted(claimants))}")
        # Dated through the projection, as the board memo is: a verdict said
        # again today without a night keeps the day it was first published.
        for result in ObservationSet.from_state(state).verdicts:
            colour = "黒" if result.is_werewolf else "白"
            points.append(
                f"{result.claimant_id}→{result.target_id}={colour}({result.day}日目)"
            )
        # Under v3 a stated target is the model's structured answer, which its
        # message need not name. Only a candidate the seat declared out loud
        # today is public; a bare mention is no declaration, and a seat with no
        # declaring sentence is left out rather than guessed at.
        if not stated_targets_are_public:
            return points
        named_today: dict[str, set[str]] = {}
        for message in state.chat_log:
            if message.day == state.day and message.channel is ChatChannel.PUBLIC:
                named_today.setdefault(message.author_id, set()).update(
                    declared_candidates(
                        message.content, ledger, speaker_id=message.author_id
                    )
                )
        backers: dict[str, list[str]] = {}
        for speaker_id, target_id in sorted(self.stated_targets_on(state.day).items()):
            # A candidate who has since died -- usually the one just executed --
            # is no longer a choice anyone can disagree about.
            if ledger.is_alive(target_id) and target_id in named_today.get(speaker_id, ()):
                backers.setdefault(target_id, []).append(speaker_id)
        if len(backers) > 1:
            ranked = sorted(backers.items(), key=lambda item: (-len(item[1]), item[0]))
            points.append(
                "吊り先の意見が割れている: "
                + "、".join(f"{target}←{'・'.join(seats)}" for target, seats in ranked)
            )
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
            "solver_query_count": self._cache.hits + self._cache.misses,
            "solver_cache_hit_rate": self._cache.hits
            / max(self._cache.hits + self._cache.misses, 1),
            "time_spent_in_solver": round(self._cache.query_seconds, 6),
            "human_messages_considered": self.human_messages_considered,
            "seats_considering_each_human_message": (
                round(self._seat_evaluations / self.human_messages_considered, 2)
                if self.human_messages_considered
                else 0.0
            ),
            "corrections_heard": self.corrections_heard,
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
            "stale_premise_turns": self.stale_premise_turns,
            "vote_citations_heard": len(self.vote_citations),
            "inaccurate_vote_citations": sum(
                1
                for citation in self.vote_citations
                if citation.is_verifiable and not citation.is_accurate
            ),
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

    def flush_metrics(self) -> None:
        """Publish solver deltas once into the shared operational collector."""
        if self._metrics_collector is None:
            return
        queries = self._cache.hits + self._cache.misses
        with self._metrics_collector._lock:
            self._metrics_collector.solver_query_count += (
                queries - self._reported_solver_queries
            )
            self._metrics_collector.solver_cache_hits += (
                self._cache.hits - self._reported_solver_hits
            )
            self._metrics_collector.time_spent_in_solver += (
                self._cache.query_seconds - self._reported_solver_seconds
            )
        self._reported_solver_queries = queries
        self._reported_solver_hits = self._cache.hits
        self._reported_solver_seconds = self._cache.query_seconds
