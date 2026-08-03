"""Per-seat belief updates, derived from public facts and the solver.

One engine per AI, never shared: seventeen players who all reach the same
conclusion from the same evidence are one player wearing seventeen hats. Each
engine holds its own evidence set, its own trust graph and its own conclusion.

Two layers, kept apart:

* The solver's verdicts are *hard*. A seat the rules have excluded stays
  excluded, and nothing downstream -- weighting, personality (PR8), a persuasive
  argument -- may reorder it.
* Everything else is soft evidence with a weight, and every point of a
  suspicion score traces back to a named `EvidenceRecord`. Retract the record
  and the score moves; that is what makes "I accept I misremembered your vote,
  but I still think you're the wolf for the same reason" impossible to express.

No LLM is called here, and nothing reads a free-text memo. The inputs are the
ledger, the solver and explicit corrections.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.ai.reasoning.belief.corrections import (
    CorrectionStatus,
    CorrectionVerdict,
    FactCorrection,
    claim_fact_id,
    verify,
    vote_fact_id,
)
from app.ai.reasoning.belief.ranking import RankedView, rank_hypotheses
from app.ai.reasoning.belief.state import (
    CONFIDENCE_SPREAD,
    HARD_EXCLUDED_SCORE,
    EvidenceRecord,
    PlayerBeliefState,
    RankedHypothesis,
    is_hard,
)
from app.ai.reasoning.belief.traits import CognitiveTraits
from app.ai.reasoning.belief.utility import (
    RoleCertainty,
    UtilityInputs,
    attack_utility,
    divine_utility,
    execution_utility,
    guard_utility,
)
from app.ai.reasoning.facts import MEDIUM_RESULT, PublicFactLedger
from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import Perspective
from app.ai.reasoning.solver.backend import Certainty, has_role
from app.ai.reasoning.solver.queries import RoleSolver
from app.engine.roles import RoleName

# Default soft weights. Collected here rather than scattered through the
# derivation so the whole scale can be read -- and later tuned -- in one place.
PUBLISHED_BLACK_WEIGHT = 1.2
PUBLISHED_WHITE_WEIGHT = -1.0
CONTESTED_CLAIM_WEIGHT = 0.6
VOTED_FOR_CLEARED_WEIGHT = 0.4
VOTED_FOR_WOLF_WEIGHT = -0.3
MAJORITY_PRESSURE_WEIGHT = 0.8
TRUST_STEP = 0.5

_WOLF_CATEGORIES = frozenset(
    {
        "published_black",
        "published_white",
        "contested_claim",
        "voted_for_cleared",
        "voted_for_wolf",
        "misremembered_vote",
        "majority_pressure",
        "accusation",
    }
)


@dataclass(frozen=True)
class CorrectionOutcome:
    """What a correction actually did to one seat's beliefs."""

    verdict: CorrectionVerdict
    invalidated_evidence_ids: tuple[str, ...] = ()
    changed_subjects: tuple[str, ...] = ()

    @property
    def changed_anything(self) -> bool:
        return bool(self.invalidated_evidence_ids)


class BeliefEngine:
    """One AI seat's beliefs. Construct one per player; never share them."""

    def __init__(
        self,
        player_id: str,
        perspective: Perspective,
        traits: CognitiveTraits | None = None,
    ) -> None:
        self.state = PlayerBeliefState(
            player_id=player_id, perspective_id=perspective.perspective_id
        )
        self._perspective = perspective
        # Traits scale soft evidence only. Nothing here can move a hard verdict.
        self.traits = traits or CognitiveTraits()
        self._evidence: dict[str, EvidenceRecord] = {}
        self._hard: dict[str, Certainty] = {}
        self._ranked: tuple[RankedView, ...] = ()
        # Filled from the perspective, never from the observations directly.
        self._self_role: RoleName | None = None
        self._ally_ids: frozenset[str] = frozenset()
        self._already_divined: frozenset[str] = frozenset()
        self._claimed_roles: dict[str, RoleName] = {}
        self._alive_ids: frozenset[str] = frozenset()

    @property
    def ranked_views(self) -> tuple[RankedView, ...]:
        """Hypotheses in bands (本線 / 有力対抗 / …), never as percentages."""
        return self._ranked

    # -- evidence --

    @property
    def evidence(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._evidence.values())

    def active_evidence(self) -> tuple[EvidenceRecord, ...]:
        return tuple(record for record in self._evidence.values() if record.active)

    def add_evidence(self, record: EvidenceRecord) -> EvidenceRecord:
        """Register a reason. Re-adding an id keeps the first, so re-observing
        the same public fact does not double its weight."""
        return self._evidence.setdefault(record.evidence_id, record)

    def invalidate_source_facts(
        self, source_ids: Sequence[str], reason: str = ""
    ) -> tuple[str, ...]:
        """Retract every reason that rests on any of these facts."""
        retracted: list[str] = []
        for evidence_id, record in self._evidence.items():
            if not record.active:
                continue
            if any(source in record.source_event_ids for source in source_ids):
                self._evidence[evidence_id] = record.deactivated(reason)
                retracted.append(evidence_id)
        return tuple(retracted)

    # -- observation --

    def observe(
        self,
        ledger: PublicFactLedger,
        solver: RoleSolver | None = None,
        observations: ObservationSet | None = None,
    ) -> None:
        """Derive evidence from the public record and recompute.

        Only public facts, this seat's own private knowledge and solver verdicts
        are read. Free-text memos are not re-parsed on every update -- they are
        opinion, and opinion is not a source of evidence.
        """
        if observations is not None:
            self._absorb_private_knowledge(observations)
        self._derive_verdict_evidence(ledger)
        self._derive_claim_evidence(ledger)
        self._derive_majority_pressure(ledger)
        if solver is not None:
            self._derive_solver_facts(ledger, solver)
            self._derive_vote_evidence(ledger)
        self.recompute(ledger)

    def _absorb_private_knowledge(self, observations: ObservationSet) -> None:
        """Read this seat's own cards and ability results, through the perspective.

        The perspective is the gate: asking it rather than the observations is
        what stops a villager's engine from quietly picking up the seer's
        unpublished black.
        """
        known = self._perspective.known_roles(observations)
        self._self_role = known.get(self.state.player_id)
        self._ally_ids = frozenset(
            pid for pid in known if pid != self.state.player_id
        )
        self._already_divined = frozenset(
            result.target_id
            for result in self._perspective.known_divine_results(observations)
        )
        self._alive_ids = frozenset(
            pid for pid, alive in observations.alive.items() if alive
        )

    def _derive_verdict_evidence(self, ledger: PublicFactLedger) -> None:
        for result in ledger.public_results():
            ability = "霊媒" if result.result_type == MEDIUM_RESULT else "占い"
            colour = "黒" if result.is_werewolf else "白"
            self.add_evidence(
                EvidenceRecord(
                    evidence_id=(
                        f"verdict:{result.claimant_id}:{result.result_type}:"
                        f"{result.target_id}:{colour}"
                    ),
                    subject_id=result.target_id,
                    category="published_black" if result.is_werewolf else "published_white",
                    source_event_ids=(result.source_message_id or result.target_id,),
                    weight=(
                        PUBLISHED_BLACK_WEIGHT
                        if result.is_werewolf
                        else PUBLISHED_WHITE_WEIGHT
                    ),
                    explanation=(
                        f"{result.day}日目、{result.claimant_id}の{ability}判定で"
                        f"{result.target_id}は{colour}と公開された。"
                    ),
                )
            )

    def _derive_claim_evidence(self, ledger: PublicFactLedger) -> None:
        by_role: dict[RoleName, list[str]] = {}
        for claim in ledger.co_declarations():
            by_role.setdefault(claim.claimed_role, []).append(claim.player_id)
        for role, claimants in by_role.items():
            if len(claimants) < 2:
                continue
            for player_id in claimants:
                self.add_evidence(
                    EvidenceRecord(
                        evidence_id=f"contested:{role.value}:{player_id}",
                        subject_id=player_id,
                        category="contested_claim",
                        source_event_ids=(claim_fact_id(player_id, role),),
                        weight=CONTESTED_CLAIM_WEIGHT,
                        explanation=(
                            f"{role.value}COが{len(claimants)}人おり、"
                            f"{player_id}はそのうちの1人。少なくとも1人は偽。"
                        ),
                    )
                )

    def _derive_majority_pressure(self, ledger: PublicFactLedger) -> None:
        """Where the table's votes actually landed.

        Only a conformist is moved much by this, and a sceptic barely at all --
        which is the difference between a table that converges because it agrees
        and one that converges because everyone is watching everyone else.
        """
        days = sorted({vote.day for vote in ledger.votes()})
        if not days:
            return
        latest = days[-1]
        counts: dict[str, int] = {}
        for vote in ledger.votes_on(latest):
            counts[vote.target_id] = counts.get(vote.target_id, 0) + 1
        if not counts:
            return
        leader = max(sorted(counts), key=lambda pid: counts[pid])
        self.add_evidence(
            EvidenceRecord(
                evidence_id=f"majority:{latest}:{leader}",
                subject_id=leader,
                category="majority_pressure",
                source_event_ids=tuple(
                    vote_fact_id(vote.voter_id, vote.day, vote.round, vote.target_id)
                    for vote in ledger.votes_on(latest)
                    if vote.target_id == leader
                ),
                weight=MAJORITY_PRESSURE_WEIGHT,
                explanation=(
                    f"{latest}日目の投票は{leader}へ{counts[leader]}票集まった。"
                ),
            )
        )

    def _derive_vote_evidence(self, ledger: PublicFactLedger) -> None:
        """Who someone voted for, read against what the solver has since settled."""
        for vote in ledger.votes():
            certainty = self._hard.get(vote.target_id)
            if certainty is Certainty.IMPOSSIBLE:
                self.add_evidence(
                    EvidenceRecord(
                        evidence_id=f"vote_cleared:{vote.voter_id}:{vote.day}:{vote.round}",
                        subject_id=vote.voter_id,
                        category="voted_for_cleared",
                        source_event_ids=(
                            vote_fact_id(
                                vote.voter_id, vote.day, vote.round, vote.target_id
                            ),
                        ),
                        weight=VOTED_FOR_CLEARED_WEIGHT,
                        explanation=(
                            f"{vote.day}日目、{vote.voter_id}は人狼ではないと確定した"
                            f"{vote.target_id}へ投票した。"
                        ),
                    )
                )
            elif certainty is Certainty.CERTAIN:
                self.add_evidence(
                    EvidenceRecord(
                        evidence_id=f"vote_wolf:{vote.voter_id}:{vote.day}:{vote.round}",
                        subject_id=vote.voter_id,
                        category="voted_for_wolf",
                        source_event_ids=(
                            vote_fact_id(
                                vote.voter_id, vote.day, vote.round, vote.target_id
                            ),
                        ),
                        weight=VOTED_FOR_WOLF_WEIGHT,
                        explanation=(
                            f"{vote.day}日目、{vote.voter_id}は人狼確定の"
                            f"{vote.target_id}へ投票した。"
                        ),
                    )
                )

    def _derive_solver_facts(self, ledger: PublicFactLedger, solver: RoleSolver) -> None:
        """Hard verdicts, recorded outside the soft scale."""
        self._hard = {}
        for player_id in ledger.known_player_ids():
            self._hard[player_id] = solver.assess(has_role(player_id, RoleName.WEREWOLF))

    # -- corrections --

    def apply_correction(
        self, correction: FactCorrection, ledger: PublicFactLedger
    ) -> CorrectionOutcome:
        """Check a claimed correction against the record, then act on it.

        A confirmed correction retracts the evidence that rested on the wrong
        fact and credits whoever caught it. A refuted one costs them credit. An
        unverifiable one does neither -- silence is better than guessing.
        """
        verdict = verify(correction, ledger)
        if verdict.status is CorrectionStatus.REFUTED:
            self._adjust_trust(
                correction.source_player_id, -TRUST_STEP * self.traits.trust_sensitivity
            )
            self.recompute(ledger)
            return CorrectionOutcome(verdict=verdict)
        if verdict.status is CorrectionStatus.UNVERIFIABLE:
            return CorrectionOutcome(verdict=verdict)

        retracted = self.invalidate_source_facts(
            verdict.invalidated_source_ids, reason=verdict.detail
        )
        self._adjust_trust(
            correction.source_player_id, TRUST_STEP * self.traits.trust_sensitivity
        )
        subjects = tuple(
            sorted(
                {
                    subject
                    for evidence_id in retracted
                    if (subject := self._evidence[evidence_id].subject_id) is not None
                }
            )
        )
        self.recompute(ledger)
        return CorrectionOutcome(
            verdict=verdict, invalidated_evidence_ids=retracted, changed_subjects=subjects
        )

    def _adjust_trust(self, player_id: str, delta: float) -> None:
        self.state.source_trust[player_id] = (
            self.state.source_trust.get(player_id, 0.0) + delta
        )

    # -- recomputation --

    def recompute_public(self, state: object) -> None:
        """Convenience for callers holding a `GameState` rather than a ledger."""
        self.recompute(PublicFactLedger(state))  # type: ignore[arg-type]

    def recompute(self, ledger: PublicFactLedger) -> None:
        """Rebuild every score from the evidence that is still active.

        Rebuilt, not adjusted: an incremental update can leave a score carrying
        the ghost of a retracted argument, and the whole point here is that it
        cannot.
        """
        wolf_scores: dict[str, float] = {}
        fox_scores: dict[str, float] = {}
        links: dict[str, list[str]] = {}
        for record in sorted(self.active_evidence(), key=lambda item: item.evidence_id):
            if record.subject_id is None:
                continue
            weight = record.weight * self.traits.scale_for(record.category)
            if record.category in _WOLF_CATEGORIES:
                wolf_scores[record.subject_id] = (
                    wolf_scores.get(record.subject_id, 0.0) + weight
                )
            elif record.category.startswith("fox"):
                fox_scores[record.subject_id] = (
                    fox_scores.get(record.subject_id, 0.0) + weight
                )
            links.setdefault(record.subject_id, []).append(record.evidence_id)

        # Hard verdicts stay out of the public suspicion scale entirely. They
        # are knowledge, not an argument, and each faction wants a different
        # thing done about them.
        certainties: dict[str, RoleCertainty] = {}
        for player_id in ledger.known_player_ids():
            wolf_scores.setdefault(player_id, 0.0)
            certainty = self._hard.get(player_id)
            if certainty is Certainty.CERTAIN:
                certainties[player_id] = RoleCertainty.CONFIRMED
            elif certainty is Certainty.IMPOSSIBLE:
                certainties[player_id] = RoleCertainty.EXCLUDED
            else:
                certainties[player_id] = RoleCertainty.UNKNOWN

        self.state.public_suspicion_scores = wolf_scores
        self.state.private_role_certainties = certainties
        self.state.fox_scores = fox_scores
        self.state.evidence_links = links
        self.state.claim_trust = self._claim_trust(ledger)
        self._claimed_roles = {
            claim.player_id: claim.claimed_role for claim in ledger.co_declarations()
        }
        if not self._alive_ids:
            self._alive_ids = frozenset(ledger.alive_ids())
        self._recompute_utilities(ledger)
        self._rank_hypotheses(ledger)
        self._choose_targets(ledger)

    def _utility_inputs(self, ledger: PublicFactLedger) -> UtilityInputs:
        return UtilityInputs(
            actor_id=self.state.player_id,
            actor_role=self._self_role,
            ally_ids=self._ally_ids,
            wolf_certainty=self.state.private_role_certainties,
            public_suspicion=self.state.public_suspicion_scores,
            fox_suspicion=self.state.fox_scores,
            claim_trust=self.state.claim_trust,
            claimed_roles=self._claimed_roles,
            alive_ids=self._alive_ids,
            already_divined=self._already_divined,
        )

    def _recompute_utilities(self, ledger: PublicFactLedger) -> None:
        inputs = self._utility_inputs(ledger)
        players = ledger.known_player_ids()
        self.state.execution_utility_scores = {
            pid: execution_utility(inputs, pid) for pid in players
        }
        self.state.divine_utility_scores = {
            pid: divine_utility(inputs, pid) for pid in players
        }
        self.state.guard_utility_scores = {
            pid: guard_utility(inputs, pid) for pid in players
        }
        self.state.attack_utility_scores = {
            pid: attack_utility(inputs, pid) for pid in players
        }

    def _claim_trust(self, ledger: PublicFactLedger) -> dict[str, float]:
        """How much each standing role claim is worth believing, from hard facts
        plus how much the table trusts the claimant as a source."""
        trust: dict[str, float] = {}
        for claim in ledger.co_declarations():
            base = self.state.source_trust.get(claim.player_id, 0.0)
            contested = sum(
                1
                for other in ledger.co_declarations()
                if other.claimed_role is claim.claimed_role
            )
            trust[claim.player_id] = base - CONTESTED_CLAIM_WEIGHT * (contested - 1)
        return trust

    def _rank_hypotheses(self, ledger: PublicFactLedger) -> None:
        """Ranked by what could be argued, not by what this seat wants done."""
        ranked: list[RankedHypothesis] = []
        for player_id, score in self.state.ranked_suspects():
            certainty = self._hard.get(player_id, Certainty.POSSIBLE)
            if certainty is Certainty.IMPOSSIBLE:
                continue
            ranked.append(
                RankedHypothesis(
                    hypothesis_id=f"wolf:{player_id}",
                    label=f"{player_id}が人狼",
                    score=score,
                    certainty=certainty,
                    supporting_evidence_ids=self.state.reasons_for(player_id),
                )
            )
        self.state.active_hypotheses = ranked
        self._ranked = rank_hypotheses(ranked, self.traits)

    def _choose_targets(self, ledger: PublicFactLedger) -> None:
        """Pick today's execution candidate. Dead seats fall out on their own --
        they are simply not in the eligible set any more."""
        eligible = [
            (player_id, score)
            for player_id, score in self.state.ranked_targets()
            if ledger.is_alive(player_id)
            and player_id != self.state.player_id
            and score > HARD_EXCLUDED_SCORE
        ]
        if not eligible:
            self.state.current_execution_target = None
            self.state.alternative_target = None
            self.state.confidence = 0.0
            return
        # Changing your mind should cost something, or a player flips on every
        # new scrap. How much it costs is what stubbornness means in practice.
        incumbent = self.state.current_execution_target
        if incumbent is not None and incumbent != eligible[0][0]:
            incumbent_score = next(
                (score for pid, score in eligible if pid == incumbent), None
            )
            if (
                incumbent_score is not None
                and eligible[0][1] - incumbent_score < self.traits.switching_cost
            ):
                eligible = [(incumbent, incumbent_score)] + [
                    item for item in eligible if item[0] != incumbent
                ]
        self.state.current_execution_target = eligible[0][0]
        self.state.alternative_target = eligible[1][0] if len(eligible) > 1 else None
        top = eligible[0][1]
        runner_up = eligible[1][1] if len(eligible) > 1 else 0.0
        if is_hard(top):
            self.state.confidence = 1.0
        else:
            raw = max(0.0, min(1.0, (top - runner_up) / CONFIDENCE_SPREAD))
            # A cautious player wants a wider gap before calling it settled.
            self.state.confidence = (
                0.0 if raw < self.traits.commitment_threshold else raw
            )
