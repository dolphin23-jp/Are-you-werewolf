"""AICoordinator: orchestrates all AI players through each game phase with
a phase-appropriate concurrency policy:

  - discussion: strictly sequential (each AI reads prior messages before
    speaking, for coherence)
  - voting: parallel (independent judgments), looped across runoff rounds
  - night: independent actions (divine/guard/wolf-chat/freemason-chat) run
    in parallel, then the alpha werewolf's attack decision runs afterward,
    fed the wolf-chat transcript

CO (role-claim) detection is always based on pattern-matching the AI's own
generated public chat text -- never a scripted side-channel -- so bluffing
stays emergent rather than scripted.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import Counter
from dataclasses import replace

from app.ai.co_detection import detect_claimed_role
from app.ai.context import ContextBuilder, DaySummaryManager
from app.ai.deception import FakeClaimGuard, assign_madman_strategy, assign_wolf_deception
from app.ai.personalities import assign_personalities, discussion_length_range
from app.ai.player_agent import AIPlayerAgent, truncate_at_sentence
from app.ai.provider.base import LLMProvider, Message
from app.ai.reasoning import (
    PublicFactLedger,
    ValidationLog,
    compose_day_summary,
    detect_vote_plan_mismatch,
    render_public_fact_summary,
    validate_discussion_output,
    validate_public_result_claim,
)
from app.ai.reasoning.claims import (
    SpeechEventDraft,
    build_claim_drafts,
    ensure_fact_sentences,
    register_claim_drafts,
)
from app.ai.reasoning.runtime import ReasoningRuntime, SeatReasoning
from app.ai.schemas import (
    DiscussionOutput,
    MorningIntentOutput,
    NightActionOutput,
    VoteOutput,
)
from app.ai.schemas import PublicResultClaim as SchemaPublicResultClaim
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.speech_events import SpeechEventType
from app.engine.state import GameState, PendingQuestion
from app.eval.transcript import TranscriptRecorder, Utterance
from app.sessions.models import DiscussionRoundState


def _night_utility(seat: SeatReasoning, action_type: str) -> dict[str, float]:
    """The utility map that decided this night action, for the recorded reason."""
    state = seat.belief.state
    maps: dict[str, dict[str, float]] = {
        "divine": state.divine_utility_scores,
        "guard": state.guard_utility_scores,
        "attack": state.attack_utility_scores,
    }
    return maps.get(action_type, state.execution_utility_scores)


# How many times the round may re-queue pressured execution candidates that have
# not rebutted yet. Bounded so the speaker selection always terminates.
_MAX_MAJOR_TARGET_SWEEPS = 2

# Consecutive speakers that may yield no output before the segment gives up.
_MAX_CONSECUTIVE_SPEECH_FAILURES = 3

# Cap for a message that only agrees with an existing point, matching the
# "60文字以内の短い同意" instruction in DISCUSSION_OUTPUT_INSTRUCTION.
_REACTION_MAX_CHARS = 60

# Private-channel messages are capped at 100 chars by `generate_wolf_chat`; the
# evaluation harness needs the same number to flag overruns.
_PRIVATE_CHAT_MAX_CHARS = 100


class AICoordinator:
    def __init__(
        self,
        state: GameState,
        ai_player_ids: list[str],
        provider: LLMProvider,
        seed: int | None = None,
        recorder: TranscriptRecorder | None = None,
        observer_player_ids: set[str] | None = None,
        max_discussion_followups: int = 4,
        discussion_segment_size: int = 4,
        pacing_scale: float = 0.0,
        reasoning: ReasoningRuntime | None = None,
    ) -> None:
        self._ai_player_ids = list(ai_player_ids)
        self._observer_player_ids = observer_player_ids or set()
        self._max_discussion_followups = max(0, max_discussion_followups)
        self._segment_size = max(1, discussion_segment_size)
        self._pacing_scale = max(0.0, pacing_scale)
        self._rng = random.Random(seed)
        self._pending_questions = state.pending_questions
        # Pending questions disappear when answered, but their topic must remain in
        # the day's ledger or another AI immediately asks the same thing again.
        self._asked_question_topics: set[tuple[int, str, str]] = {
            (question.day, target_id, question.topic)
            for target_id, questions in state.pending_questions.items()
            for question in questions
            if question.topic
        }
        self._forced_partner_confirmations: set[str] = set()
        self._freemason_public_plan: tuple[str, str, bool] | None = None
        self._freemason_death_announced: set[str] = set()
        freemasons = [
            player.player_id
            for player in state.players_by_role(RoleName.FREEMASON)
            if player.player_id in self._ai_player_ids
        ]
        if len(freemasons) == 2:
            plan_rng = random.Random(f"{seed}:freemason-public-plan")
            leader, partner = plan_rng.sample(freemasons, 2)
            self._freemason_public_plan = (leader, partner, plan_rng.random() < 0.5)
        self._metrics = getattr(provider, "_metrics", None)
        # Present only in v2. When it is, votes, night actions and the speaking
        # order are decided in code and the model is asked for wording alone.
        self.reasoning = reasoning
        # Every state-consistency repair and every say-one-thing-vote-another
        # discrepancy lands here, so a game can be audited after the fact
        # without the AI layer having to be re-run.
        self.validation = ValidationLog()

        self._personalities = assign_personalities(self._ai_player_ids, seed=seed)

        wolf_ids = [p.player_id for p in state.players_by_role(RoleName.WEREWOLF)]
        madman = state.players_by_role(RoleName.MADMAN)
        wolf_deception = assign_wolf_deception(wolf_ids, seed=seed)
        madman_strategy_name, madman_fake_role = assign_madman_strategy(seed=seed)
        fake_claim_guard = FakeClaimGuard(wolf_team_ids=set(wolf_ids))
        self._wolf_deception = wolf_deception
        self._madman_fake_role_by_player = {
            player.player_id: madman_fake_role
            for player in madman
            if madman_fake_role is not None
        }

        self._day_summaries = DaySummaryManager()
        self._context = ContextBuilder(
            personalities=self._personalities,
            day_summaries=self._day_summaries,
            wolf_deception=wolf_deception,
            madman_fake_role=madman_fake_role if madman else None,
            fake_claim_guard=fake_claim_guard,
            observer_player_ids=self._observer_player_ids,
        )

        self._agents: dict[str, AIPlayerAgent] = {
            pid: AIPlayerAgent(provider, self._personalities[pid]) for pid in self._ai_player_ids
        }

        self._recorder = recorder
        if recorder is not None:
            recorder.set_roster(
                names={pid: p.name for pid, p in state.players.items()},
                roles={pid: p.role.value for pid, p in state.players.items()},
                teams={pid: p.team.value for pid, p in state.players.items()},
                personalities={pid: p.name for pid, p in self._personalities.items()},
                deception={
                    "wolf_pattern": wolf_deception.pattern_name,
                    "wolf_pattern_label": wolf_deception.pattern_label,
                    "fake_role_by_player": {
                        **{
                            pid: role.value
                            for pid, role in wolf_deception.fake_role_by_player.items()
                        },
                        **{
                            pid: role.value
                            for pid, role in self._madman_fake_role_by_player.items()
                        },
                    },
                    "lurking_player_ids": list(wolf_deception.lurking_player_ids),
                    "madman_strategy": madman_strategy_name if madman else None,
                },
                seed=seed,
                provider=type(provider).__name__,
            )

    def _deception_role(self, player_id: str) -> str | None:
        if player_id in self._wolf_deception.fake_role_by_player:
            return f"fake_{self._wolf_deception.fake_role_by_player[player_id].value}"
        if player_id in self._wolf_deception.lurking_player_ids:
            return "lurker"
        if player_id in self._madman_fake_role_by_player:
            return f"fake_{self._madman_fake_role_by_player[player_id].value}"
        return None

    def _record(
        self,
        state: GameState,
        player_id: str,
        kind: str,
        *,
        text: str = "",
        target: str | None = None,
        reasoning_memo: object = None,
        used_fallback: bool = False,
        public_claim_role: str | None = None,
        public_results: list[dict[str, object]] | None = None,
        directed_question_targets: list[str] | None = None,
        ready_to_vote: bool | None = None,
        effective_length_limit: int | None = None,
        key_point: str = "",
        agrees_with: list[str] | None = None,
        decision_evidence: str = "",
        countercase: str = "",
        alternative_target: str | None = None,
    ) -> None:
        if self._recorder is None:
            return
        player = state.players[player_id]
        memo = reasoning_memo.model_dump() if reasoning_memo is not None else None  # type: ignore[attr-defined]
        self._recorder.record(
            Utterance(
                day=state.day,
                phase=state.phase.value,
                kind=kind,
                player_id=player_id,
                player_name=player.name,
                role=player.role.value,
                team=player.team.value,
                personality=self._personalities[player_id].name,
                deception_role=self._deception_role(player_id),
                text=text,
                target=target,
                reasoning_memo=memo,
                public_claim_role=public_claim_role,
                public_results=public_results or [],
                directed_question_targets=directed_question_targets or [],
                ready_to_vote=ready_to_vote,
                used_fallback=used_fallback,
                effective_length_limit=effective_length_limit,
                key_point=key_point,
                agrees_with=agrees_with or [],
                decision_evidence=decision_evidence,
                countercase=countercase,
                alternative_target=alternative_target,
            )
        )

    def _find_ai_with_role(self, state: GameState, role: RoleName) -> str | None:
        for pid in self._ai_player_ids:
            player = state.players[pid]
            if player.alive and player.role == role:
                return pid
        return None

    # -- discussion --

    async def run_discussion_round(self, session: object) -> None:
        """Compatibility wrapper: advance segments until the round completes.

        Offline/evaluation callers use this method and must never pause for a
        human. Interactive routes call :meth:`advance_discussion` directly.
        """
        while session.controller.state.phase == Phase.DISCUSSION:  # type: ignore[attr-defined]
            await self.advance_discussion(session, allow_human_pause=False)
            round_state = getattr(session, "discussion_round", None)
            if round_state is None or round_state.complete:
                return

    async def advance_discussion(
        self, session: object, *, allow_human_pause: bool = True
    ) -> None:
        """Run at most one discussion segment while holding the session lock."""
        async with session.discussion_lock:  # type: ignore[attr-defined]
            controller = session.controller  # type: ignore[attr-defined]
            state = controller.state
            if state.phase != Phase.DISCUSSION:
                return
            round_state = getattr(session, "discussion_round", None)
            if round_state is None or round_state.day != state.day:
                round_state = await self._start_discussion_round(state)
                session.discussion_round = round_state  # type: ignore[attr-defined]

            human_id = getattr(session, "human_id", "")
            human = state.players.get(human_id)
            can_pause = bool(
                allow_human_pause
                and human
                and human.alive
                and human_id not in self._observer_player_ids
            )
            if (
                can_pause
                and getattr(session, "discussion_paused", False)
                and not getattr(session, "discussion_step_budget", None)
            ):
                self._set_awaiting(round_state, True)
                return
            if round_state.awaiting_human and can_pause:
                return
            self._set_awaiting(round_state, False)
            if round_state.stage == "immediate" and round_state.immediate_count == 0:
                round_state.stage = "initial_view"
                if can_pause:
                    self._set_awaiting(round_state, True)
                    return

            spoken = 0
            # A speaker that never produces output must not be retried forever:
            # `_speak` returns None both when the phase left DISCUSSION mid-segment
            # (without awaiting, so a retry loop would never yield to the event
            # loop) and when generation fails outright.
            consecutive_failures = 0
            while spoken < self._segment_size and not round_state.complete:
                if state.phase != Phase.DISCUSSION:
                    return
                pid, stage = self._next_discussion_speaker(state, round_state)
                if pid is None:
                    round_state.complete = True
                    break
                output = await self._speak(controller, state, pid, stage)
                if output is None:
                    consecutive_failures += 1
                    if consecutive_failures >= _MAX_CONSECUTIVE_SPEECH_FAILURES:
                        break
                    continue
                consecutive_failures = 0
                round_state.outputs.append((pid, output))
                for target_id in tuple(self._forced_partner_confirmations):
                    self._round_queue_reply(state, round_state, target_id)
                    self._forced_partner_confirmations.discard(target_id)
                round_state.speech_counts[pid] = round_state.speech_counts.get(pid, 0) + 1
                spoken += 1
                step_budget = getattr(session, "discussion_step_budget", None)
                if step_budget is not None:
                    session.discussion_step_budget = max(0, step_budget - 1)  # type: ignore[attr-defined]
                if getattr(session, "discussion_pause_requested", False):
                    session.discussion_pause_requested = False  # type: ignore[attr-defined]
                    session.discussion_paused = True  # type: ignore[attr-defined]
                if getattr(session, "discussion_step_budget", None) == 0:
                    session.discussion_paused = True  # type: ignore[attr-defined]
                if can_pause and getattr(session, "discussion_paused", False):
                    self._set_awaiting(round_state, True)
                    break
                if stage == "consensus_summary":
                    round_state.complete = True
                    break
                for question in output.directed_questions:
                    self._round_queue_reply(state, round_state, question.target_id)
                if output.needs_another_statement and self._rng.random() < min(
                    0.95, self._personalities[pid].talkativeness / 1.5
                ):
                    self._round_queue_reply(state, round_state, pid)

                # The first segment contains only morning-immediate speakers.
                if (
                    round_state.stage == "immediate"
                    and round_state.cursor >= round_state.immediate_count
                ):
                    round_state.stage = "initial_view"
                    if can_pause:
                        self._set_awaiting(round_state, True)
                        break
            if (
                can_pause
                and round_state.stage != "immediate"
                and not round_state.complete
                and spoken >= self._segment_size
            ):
                self._set_awaiting(round_state, True)

    async def _start_discussion_round(self, state: GameState) -> DiscussionRoundState:
        alive = [pid for pid in self._ai_player_ids if state.players[pid].alive]
        if self.reasoning is not None:
            return self._coded_discussion_round(state, alive)
        shuffled = list(alive)
        self._rng.shuffle(shuffled)
        intents = await asyncio.gather(*(self._morning_intent(state, pid) for pid in shuffled))
        priority = {"immediate": 0, "after_results": 1, "normal": 2, "hold": 3}
        intent_by_id = dict(zip(shuffled, intents, strict=True))
        order = sorted(
            shuffled,
            key=lambda pid: (
                priority.get(intent_by_id[pid].timing, 2),
                0 if intent_by_id[pid].intent == "publish_result" else 1,
            ),
        )
        immediate_count = sum(intent_by_id[pid].timing == "immediate" for pid in order)
        return DiscussionRoundState(
            day=state.day,
            order=order,
            stage="immediate",
            immediate_count=immediate_count,
            max_total=max(len(alive) + self._max_discussion_followups, int(len(alive) * 2.5)),
        )

    def _coded_discussion_round(
        self, state: GameState, alive: list[str]
    ) -> DiscussionRoundState:
        """Open the day with the seats that have something to say.

        Everyone forms an opinion -- the belief engine updates all of them -- but
        a morning where sixteen AIs each state a view is a wall of text nobody
        reads. Duty speakers (a held result, a planned CO, a direct question)
        are never dropped; the rest speak when they are answered or challenged.
        """
        assert self.reasoning is not None
        self.reasoning.refresh(state)
        duty = [
            pid
            for pid in alive
            if self.reasoning.holds_unpublished_result(state, pid)
            or self._pending_questions.get(pid)
        ]
        chosen = self.reasoning.select_opening_speakers(
            state,
            pending_question_targets=[
                pid for pid in alive if self._pending_questions.get(pid)
            ],
        )
        order = list(dict.fromkeys(duty + chosen))
        return DiscussionRoundState(
            day=state.day,
            order=order,
            stage="immediate",
            immediate_count=len(duty),
            max_total=max(len(alive) + self._max_discussion_followups, len(order) * 3),
        )

    def _next_discussion_speaker(
        self, state: GameState, round_state: DiscussionRoundState
    ) -> tuple[str | None, str]:
        if self.reasoning is not None:
            for player_id in self.reasoning.take_reassessment_speakers():
                self._round_queue_reply(state, round_state, player_id)
        for target_id in sorted(self._forced_partner_confirmations):
            self._forced_partner_confirmations.discard(target_id)
            if target_id not in self._agents or not state.players[target_id].alive:
                continue
            try:
                pending_index = round_state.order.index(target_id, round_state.cursor)
            except ValueError:
                pass
            else:
                round_state.order.pop(pending_index)
            return target_id, "freemason_confirmation"
        if round_state.cursor < len(round_state.order):
            pid = round_state.order[round_state.cursor]
            round_state.cursor += 1
            return pid, "immediate" if round_state.stage == "immediate" else "initial_view"
        if not round_state.major_targets_ready:
            # Restricted to seats this coordinator speaks for: the pressure list
            # drives speaking limits and the rebuttal sweep, neither of which can
            # apply to the human.
            latest_targets = {
                speaker: target
                for speaker, output in round_state.outputs
                if isinstance((target := output.reasoning_memo.execution_target), str)
                and target in self._agents
                and state.players[target].alive
            }
            pressure: Counter[str] = Counter(latest_targets.values())
            round_state.major_targets = [target for target, _count in pressure.most_common(2)]
            round_state.major_targets_ready = True
            for target in round_state.major_targets:
                if round_state.speech_counts.get(target, 0) < 2:
                    self._round_queue_reply(state, round_state, target)
        # Every branch below must change round state before returning. A branch that
        # returns the same speaker without advancing anything spins forever when
        # `_speak` yields nothing -- and the phase-changed path of `_speak` returns
        # without awaiting, so such a spin would never release the event loop.
        while True:
            while round_state.reply_queue and len(round_state.outputs) < round_state.max_total:
                pid = round_state.reply_queue.pop(0)
                round_state.queued.discard(pid)
                limit = max(1, round(2.5 * self._personalities[pid].talkativeness))
                if pid in round_state.major_targets:
                    limit = max(limit, 4)
                if state.players[pid].alive and round_state.speech_counts.get(pid, 0) < limit:
                    if (
                        pid in round_state.major_targets
                        and round_state.speech_counts.get(pid, 0) < 2
                    ):
                        return pid, "rebuttal_or_reassessment"
                    return (
                        pid,
                        "reaction" if self._rng.random() < 0.35 else "rebuttal_or_reassessment",
                    )
            # The queue is empty, but the most-pressured execution candidates still
            # owe the table a rebuttal. Re-queue them rather than returning them
            # directly, so the pop above stays the single source of progress.
            if round_state.major_target_sweeps >= _MAX_MAJOR_TARGET_SWEEPS:
                break
            owed = [
                target
                for target in round_state.major_targets
                if state.players[target].alive
                and round_state.speech_counts.get(target, 0) < 2
            ]
            if not owed or len(round_state.outputs) >= round_state.max_total:
                break
            round_state.major_target_sweeps += 1
            for target in owed:
                self._round_queue_reply(state, round_state, target)
        if (
            not round_state.minority_review_done
            and len(round_state.outputs) < round_state.max_total
        ):
            latest_public_targets = {
                speaker: target
                for speaker, output in round_state.outputs
                if isinstance((target := output.reasoning_memo.execution_target), str)
                and target in state.players
                and state.players[target].alive
            }
            public_pressure: Counter[str] = Counter(latest_public_targets.values())
            round_state.minority_review_done = True
            if public_pressure:
                top_target, top_count = public_pressure.most_common(1)[0]
                stated_count = sum(public_pressure.values())
                if stated_count >= 4 and top_count / stated_count >= 0.6:
                    reviewers = [
                        pid
                        for pid in self._agents
                        if state.players[pid].alive and pid != top_target
                    ]
                    if reviewers:
                        reviewer = min(
                            reviewers,
                            key=lambda pid: (round_state.speech_counts.get(pid, 0), pid),
                        )
                        return reviewer, f"minority_review:{top_target}"
        if not round_state.summary_done and len(round_state.outputs) < round_state.max_total:
            leader = next(
                (
                    claim.player_id
                    for claim in state.co_declarations
                    if claim.claimed_role == RoleName.FREEMASON
                    and claim.player_id in self._agents
                    and state.players[claim.player_id].alive
                ),
                None,
            )
            if leader is None:
                leader = next(
                    (
                        pid
                        for pid, output in reversed(round_state.outputs)
                        if output.ready_to_vote and state.players[pid].alive
                    ),
                    round_state.order[0] if round_state.order else None,
                )
            round_state.summary_done = True
            if leader is not None:
                return leader, "consensus_summary"
        return None, "consensus_summary"

    def _round_queue_reply(
        self, state: GameState, round_state: DiscussionRoundState, target_id: str
    ) -> None:
        # Only seats this coordinator can speak for. AIs routinely question the
        # human or name them as an execution target, and scheduling that seat used
        # to reach `self._personalities[human_id]` and raise KeyError -- which,
        # inside the fire-and-forget discussion task, silently killed the round.
        if (
            target_id in self._agents
            and state.players[target_id].alive
            and target_id not in round_state.queued
        ):
            round_state.reply_queue.append(target_id)
            round_state.queued.add(target_id)

    def resume_after_human(
        self,
        session: object,
        reply_to: str | None = None,
        references: list[str] | None = None,
        *,
        release_wait: bool = True,
    ) -> None:
        """Release the human pause and prioritize the referenced AI/questioner."""
        round_state = getattr(session, "discussion_round", None)
        if round_state is None:
            return
        if release_wait:
            self._set_awaiting(round_state, False)
        for message_id in dict.fromkeys([reply_to, *(references or [])]):
            if not message_id:
                continue
            message = next(
                (m for m in session.controller.state.chat_log if m.message_id == message_id),  # type: ignore[attr-defined]
                None,
            )
            if message and message.author_id in self._agents:
                self._round_queue_reply(session.controller.state, round_state, message.author_id)  # type: ignore[attr-defined]

    @staticmethod
    def _question_topic(question: str) -> str:
        if any(word in question for word in ("処刑", "吊り", "狼候補", "怪しい")):
            return "execution_candidate"
        if "狐" in question:
            return "fox_candidate"
        if any(word in question for word in ("CO理由", "占い理由", "選んだ理由")):
            return "claim_reason"
        if any(word in question for word in ("時系列", "順番", "先に")):
            return "timeline"
        return ""

    @staticmethod
    def _set_awaiting(round_state: DiscussionRoundState, awaiting: bool) -> None:
        round_state.awaiting_human = awaiting
        round_state.awaiting_since = time.time() if awaiting else None

    async def _morning_intent(self, state: GameState, player_id: str) -> MorningIntentOutput:
        if self.reasoning is not None:
            output = self._coded_morning_intent(state, player_id)
        else:
            system, messages = self._context.build_morning_intent_context(state, player_id)
            output = await self._agents[player_id].generate_morning_intent(system, messages)
        player = state.players[player_id]
        if self._freemason_public_plan is not None and player.role == RoleName.FREEMASON:
            leader, partner, full_reveal = self._freemason_public_plan
            leader_alive = state.players[leader].alive
            partner_alive = state.players[partner].alive
            already_claimed = any(
                claim.player_id == player_id for claim in state.co_declarations
            )
            if player_id == leader and not already_claimed:
                output.timing = "after_results"
                output.intent = "claim"
                output.public_claim_role = RoleName.FREEMASON.value
            elif player_id == partner and not already_claimed:
                under_black = any(
                    claim.target_id == player_id and claim.is_werewolf
                    for claim in state.public_result_claims
                )
                if leader_alive and not under_black:
                    output.timing = "hold"
                    output.intent = "normal"
                    output.public_claim_role = None
                else:
                    output.timing = "immediate"
                    output.intent = "claim"
                    output.public_claim_role = RoleName.FREEMASON.value
            elif player_id == leader and already_claimed and not partner_alive and not full_reveal:
                output.timing = "immediate"
                output.intent = "lead"
        has_divine = any(r.seer_id == player_id for r in state.divine_records)
        has_medium = any(r.medium_id == player_id for r in state.medium_records)
        if (player.role == RoleName.SEER and has_divine) or (
            player.role == RoleName.MEDIUM and has_medium
        ):
            output.timing = "immediate"
            output.intent = "publish_result"
            output.public_claim_role = player.role.value
        return output

    def _coded_morning_intent(self, state: GameState, player_id: str) -> MorningIntentOutput:
        """Who speaks first is a scheduling question, not a language one.

        Holding an unpublished result outranks everything; the rest follows the
        speech-value ranking the runtime already computes. One request per AI
        per morning disappears.
        """
        assert self.reasoning is not None
        player = state.players[player_id]
        has_result = any(r.seer_id == player_id for r in state.divine_records) or any(
            r.medium_id == player_id for r in state.medium_records
        )
        if has_result and player.role in (RoleName.SEER, RoleName.MEDIUM):
            return MorningIntentOutput(
                timing="immediate",
                intent="publish_result",
                public_claim_role=player.role.value,
                priority_reason="未公開の能力結果を持つ",
            )
        candidates = {
            item.player_id: item for item in self.reasoning.speech_candidates(state)
        }
        candidate = candidates.get(player_id)
        if candidate is None or candidate.value <= 0:
            return MorningIntentOutput(timing="hold", intent="normal")
        return MorningIntentOutput(
            timing="normal" if candidate.value < 2.5 else "after_results",
            intent="lead" if candidate.value >= 2.5 else "normal",
            priority_reason="、".join(candidate.reasons),
        )

    async def _speak(
        self, controller: object, state: GameState, player_id: str, stage: str
    ) -> DiscussionOutput | None:
        if state.phase != Phase.DISCUSSION:
            return None
        system, messages = self._context.build_discussion_context(state, player_id, stage)
        decision = None
        if self.reasoning is not None:
            decision = self.reasoning.discussion_decision(
                state,
                player_id,
                pending_question=bool(self._pending_questions.get(player_id)),
                under_pressure=stage.startswith("rebuttal")
                or stage.startswith("minority_review"),
            )
            messages = [
                *messages[:-1],
                Message(role="user", content=decision.render_brief()),
                messages[-1],
            ]
        freemason_opening = self._freemason_opening(state, player_id)
        freemason_must_hide = self._freemason_must_hide(state, player_id)
        if freemason_opening is not None:
            system += (
                "\n\n【共有公開計画】この発言では指定された共有CO文を最初に"
                "そのまま述べてください。"
            )
        elif freemason_must_hide:
            system += (
                "\n\n【共有公開計画】あなたは潜伏側です。共有CO、相方の名前、共有者だと"
                "推測できる表現を公開発言に絶対に含めないでください。"
            )
        controller.set_typing(player_id, True)  # type: ignore[attr-defined]
        try:
            output = await self._agents[player_id].generate_discussion(system, messages)
        finally:
            controller.set_typing(player_id, False)  # type: ignore[attr-defined]
        if self._metrics is not None:
            self._metrics.record_discussion_result(skipped=output is None)
        if output is None:
            return None
        if freemason_opening is not None:
            output.public_message = freemason_opening
            output.public_claim_role = RoleName.FREEMASON.value
            output.contains_co_claim = True
            if "死亡" in freemason_opening:
                self._freemason_death_announced.add(player_id)
        elif freemason_must_hide and detect_claimed_role(output.public_message) is not None:
            output.public_message = "現時点ではCOしません。既出の判定と灰の発言を比較します。"
            output.public_claim_role = None
            output.contains_co_claim = False
        valid_reassessments = [
            item
            for item in output.reassessments
            if item.player_id in state.players and item.player_id != player_id
        ]
        output.reassessments = valid_reassessments
        if any(
            item.changed_mind
            and item.player_id == output.reasoning_memo.execution_target
            and not item.remaining_reason.strip()
            for item in valid_reassessments
        ):
            output.reasoning_memo.execution_target = None
        # Nothing below may read an unvalidated target. A dead execution
        # candidate, a flipped verdict or a medium result about someone who was
        # never executed is state corruption, not a human-style misread, and it
        # is repaired deterministically here before it reaches the engine.
        self._validate_output(state, player_id, output)
        if decision is not None:
            # The model rendered the turn; it does not get to revise the
            # conclusion. Two reasoning systems running side by side is what
            # let an AI argue one name all day and then vote for another.
            output.reasoning_memo.execution_target = decision.execution_target
            output.alternative_execution_target = decision.alternative_target
            assert self.reasoning is not None
            self.reasoning.record_stated_target(player_id, decision.execution_target)
        pending_relation = next(
            (
                claim
                for claim in state.freemason_partner_claims
                if claim.partner_id == player_id
                and not claim.confirmed
                and state.players[claim.claimant_id].role == RoleName.FREEMASON
                and state.players[player_id].role == RoleName.FREEMASON
            ),
            None,
        )
        if pending_relation is not None:
            claimant = state.players[pending_relation.claimant_id]
            output.public_message = (
                f"共有者CO。{claimant.name}({pending_relation.claimant_id})の相方は"
                f"私{state.players[player_id].name}({player_id})で間違いありません。"
            )
            output.public_claim_role = RoleName.FREEMASON.value
            source_question = next(
                (
                    question
                    for question in self._pending_questions.get(player_id, [])
                    if question.asker == pending_relation.claimant_id
                ),
                None,
            )
            if source_question is not None:
                output.reply_to = source_question.source_message_id
        if output.agrees_with and not output.key_point.strip():
            # Agreement with no new argument is a reaction, not an analysis. Cut at a
            # sentence boundary so the shortened line still reads as finished Japanese.
            output.public_message = truncate_at_sentence(
                output.public_message, _REACTION_MAX_CHARS
            )
        self._context.set_reasoning_memo(player_id, output.reasoning_memo.model_dump())
        self._record(
            state,
            player_id,
            "discussion",
            text=output.public_message,
            reasoning_memo=output.reasoning_memo,
            used_fallback=self._is_fallback(player_id, output.public_message),
            public_claim_role=output.public_claim_role,
            public_results=[item.model_dump() for item in output.public_results],
            directed_question_targets=[item.target_id for item in output.directed_questions],
            ready_to_vote=output.ready_to_vote,
            effective_length_limit=discussion_length_range(
                self._personalities[player_id].verbosity
            )[1],
            key_point=output.key_point,
            agrees_with=output.agrees_with,
        )
        # The claims this turn publishes are decided before it is spoken, so the
        # message can be made to state them. Dropping a declared verdict because
        # the prose forgot to name its target is how a result silently vanishes.
        drafts = build_claim_drafts(output, PublicFactLedger(state), speaker_id=player_id)
        output.public_message = ensure_fact_sentences(
            output.public_message,
            drafts,
            PublicFactLedger(state),
            speaker_id=player_id,
        )
        try:
            message_id = controller.chat(  # type: ignore[attr-defined]
                player_id,
                output.public_message,
                "public",
                output.reply_to,
                output.quote,
            )
        except Exception:
            return None
        if self._pacing_scale:
            base = 0.35 + len(output.public_message) * 0.012
            delay = min(3.0, base) * self._pacing_scale * self._rng.uniform(0.8, 1.2)
            await asyncio.sleep(delay)
        for question in output.directed_questions:
            if question.target_id not in state.players or not question.question.strip():
                continue
            proposed_topic = question.topic.strip()
            topic = (
                proposed_topic
                if proposed_topic
                in {"execution_candidate", "fox_candidate", "claim_reason", "timeline"}
                else self._question_topic(question.question)
            )
            existing = self._pending_questions.setdefault(question.target_id, [])
            if topic:
                topic_key = (state.day, question.target_id, topic)
                if topic_key in self._asked_question_topics or any(
                    item.day == state.day and item.topic == topic for item in existing
                ):
                    continue
                self._asked_question_topics.add(topic_key)
            existing.append(
                PendingQuestion(
                    asker=player_id,
                    target=question.target_id,
                    question=question.question.strip(),
                    source_message_id=message_id,
                    day=state.day,
                    topic=topic,
                )
            )
        self._context.record_key_point(state.day, message_id, player_id, output.key_point)
        self._register_claim_drafts(controller, player_id, drafts, message_id)
        return output

    def _validate_output(
        self, state: GameState, player_id: str, output: DiscussionOutput
    ) -> None:
        """Reconcile one discussion turn with the public fact ledger."""
        previous = self._context.get_reasoning_memo(player_id) or {}
        previous_target = previous.get("execution_target")
        _output, issues = validate_discussion_output(
            output,
            PublicFactLedger(state),
            speaker_id=player_id,
            previous_execution_target=previous_target if isinstance(previous_target, str) else None,
            excluded_target_ids=self._observer_player_ids,
        )
        self.validation.extend(issues)

    def _coded_night_target(
        self, player_id: str, action_type: str, candidates: list[str]
    ) -> NightActionOutput | None:
        """Night targeting from beliefs, when v2 is on.

        Chasing the top suspect (divine, attack) or covering the most trusted
        seat (guard) is a ranking the belief engine already computes. Spending a
        request to re-derive it produced a worse answer, not a better one.
        """
        if self.reasoning is None:
            return None
        target = self.reasoning.night_target(player_id, action_type, candidates)
        if target is None:
            return None
        seat = self.reasoning.seats.get(player_id)
        reason = ""
        if seat is not None:
            score = _night_utility(seat, action_type).get(target, 0.0)
            reason = f"{action_type}の評価値が最も高い対象は{target}({score:+.1f})。"
        return NightActionOutput(target=target, reason=reason)

    def _belief_targets(self, player_id: str) -> list[str]:
        """This player's own stated beliefs, best first. Used to repair an
        invalid target without reaching for a random seat."""
        memo = self._context.get_reasoning_memo(player_id) or {}
        preferred: list[str] = []
        for value in (memo.get("execution_target"), *(memo.get("suspects") or [])):
            if isinstance(value, str) and value and value not in preferred:
                preferred.append(value)
        return preferred

    def _freemason_must_hide(self, state: GameState, player_id: str) -> bool:
        if self._freemason_public_plan is None:
            return False
        leader, partner, full_reveal = self._freemason_public_plan
        named_for_confirmation = any(
            claim.partner_id == player_id for claim in state.freemason_partner_claims
        )
        partner_under_black = any(
            claim.target_id == player_id and claim.is_werewolf
            for claim in state.public_result_claims
        )
        return (
            player_id == partner
            and state.players[leader].alive
            and not partner_under_black
            and (not full_reveal or not named_for_confirmation)
            and not any(claim.player_id == player_id for claim in state.co_declarations)
        )

    def _freemason_opening(self, state: GameState, player_id: str) -> str | None:
        """Return the public line required by the AI-only shared-role plan."""
        if self._freemason_public_plan is None:
            return None
        leader, partner, full_reveal = self._freemason_public_plan
        if any(claim.player_id == player_id for claim in state.co_declarations):
            if (
                player_id == leader
                and not state.players[partner].alive
                and not full_reveal
                and player_id not in self._freemason_death_announced
            ):
                return "共有者として報告します。相方は死亡しました。"
            return None
        if player_id == leader:
            if full_reveal:
                partner_player = state.players[partner]
                return f"共有者CO。相方は{partner_player.name}({partner})です。"
            return "共有者CO。相方は生存しています。"
        partner_under_black = any(
            claim.target_id == player_id and claim.is_werewolf
            for claim in state.public_result_claims
        )
        if player_id == partner and (
            not state.players[leader].alive or partner_under_black
        ):
            leader_player = state.players[leader]
            status = "死亡した" if not state.players[leader].alive else ""
            return f"共有者CO。相方は{status}{leader_player.name}({leader})です。"
        return None

    def _is_fallback(self, player_id: str, text: str) -> bool:
        """The agent substitutes a personality-specific canned line when the
        model fails entirely. Those lines are distinctive enough that an
        exact match is a reliable signal, and it keeps the agent's return
        types unchanged."""
        return text.strip() == self._personalities[player_id].get_fallback_message()

    def note_human_message(
        self, state: GameState, speaker_id: str, text: str, source_message_id: str = ""
    ) -> list[str]:
        """Handle one human message for the whole table, once.

        Factual corrections are matched in code and applied to each seat's own
        evidence, so a sentence costs zero model calls however many AIs are
        listening. Returns the seats that actually had a reason withdrawn --
        the ones with something to say back.
        """
        if self.reasoning is None:
            return []
        self.reasoning.refresh(state)
        outcomes = self.reasoning.apply_human_message(
            state, speaker_id, text, source_message_id
        )
        moved = self.reasoning.seats_that_moved(outcomes)
        # A seat whose reason was withdrawn owes the table an explanation, so it
        # is queued to speak. Dropping this list was why a correction landed and
        # nobody said anything about it.
        self.reasoning.queue_reassessment_speakers(moved)
        return moved

    def register_public_claim(
        self,
        controller: object,
        player_id: str,
        output: DiscussionOutput,
        message_id: str = "",
    ) -> None:
        """Public entry point: derive this turn's claims and record them.

        Used for the human path and by callers that already hold a finished
        message; `_speak` splits the two halves so it can fix the prose first.
        """
        drafts = build_claim_drafts(
            output, PublicFactLedger(controller.state), speaker_id=player_id  # type: ignore[attr-defined]
        )
        self._register_claim_drafts(controller, player_id, drafts, message_id)

    def _register_claim_drafts(
        self,
        controller: object,
        player_id: str,
        drafts: list[SpeechEventDraft],
        message_id: str = "",
    ) -> None:
        state = controller.state  # type: ignore[attr-defined]
        checked: list[SpeechEventDraft] = []
        for draft in drafts:
            if draft.event_type is not SpeechEventType.ABILITY_RESULT:
                checked.append(draft)
                continue
            # Published verdicts clear the same consistency bar however they were
            # derived: a declared field and a regex are equally unaware of whether
            # the target was ever executed.
            validation = validate_public_result_claim(
                SchemaPublicResultClaim(
                    result_type=draft.role.value if draft.role else "",
                    target_id=draft.target_id or "",
                    is_werewolf=bool(draft.result_is_werewolf),
                ),
                PublicFactLedger(state),
                claimant_id=player_id,
            )
            self.validation.extend(validation.issues)
            if validation.claim is not None:
                checked.append(
                    replace(draft, result_is_werewolf=validation.claim.is_werewolf)
                )
        register_claim_drafts(controller, player_id, checked, message_id)
        for draft in checked:
            if draft.event_type is SpeechEventType.PARTNER_CLAIM and draft.target_id:
                self._prompt_partner_confirmation(state, player_id, draft.target_id, message_id)

    def _prompt_partner_confirmation(
        self, state: GameState, player_id: str, partner_id: str, message_id: str
    ) -> None:
        relation = next(
            (
                claim
                for claim in state.freemason_partner_claims
                if claim.claimant_id == player_id and claim.partner_id == partner_id
            ),
            None,
        )
        if relation is None or relation.confirmed:
            return
        already_asked = any(
            question.asker == player_id and question.topic == "freemason_confirmation"
            for question in self._pending_questions.get(partner_id, [])
        )
        if already_asked:
            return
        self._pending_questions.setdefault(partner_id, []).append(
            PendingQuestion(
                asker=player_id,
                target=partner_id,
                question=(
                    "共有相方として指名されました。本人ならこの発言で確認共有COし、"
                    "相方でなければ明確に否定してください。"
                ),
                source_message_id=message_id,
                day=state.day,
                topic="freemason_confirmation",
            )
        )
        if partner_id in self._agents:
            self._forced_partner_confirmations.add(partner_id)

    # -- voting (loops across runoff rounds) --

    async def generate_all_votes(self, session: object) -> None:
        controller = session.controller  # type: ignore[attr-defined]
        state = controller.state
        human_id = session.human_id  # type: ignore[attr-defined]

        while state.phase in (Phase.VOTING, Phase.RUNOFF):
            human = state.players[human_id]
            observer = human_id in self._observer_player_ids
            waiting_for_human = (
                human.is_human
                and human.alive
                and not observer
                and human_id not in state.pending_votes
            )
            if waiting_for_human:
                return  # wait for the human's vote this round

            alive_ai = [pid for pid in self._ai_player_ids if state.players[pid].alive]
            await asyncio.gather(*(self._cast_vote(controller, state, pid) for pid in alive_ai))

            try:
                controller.resolve_votes()
            except Exception:
                return

        if state.phase == Phase.VOTE_RESULT:
            await self._generate_day_summary(controller, state)

    async def _cast_vote(self, controller: object, state: GameState, player_id: str) -> None:
        # In a runoff this is narrowed to the tied players, so the AI is not
        # offered choices the engine would reject.
        candidates = state.votable_ids(player_id)
        candidates = [pid for pid in candidates if pid not in self._observer_player_ids]
        if not candidates:
            return
        stated_target = (self._context.get_reasoning_memo(player_id) or {}).get(
            "execution_target"
        )
        if self.reasoning is not None:
            # A vote is an argmax over evidence the engine already holds, so it
            # costs no request and the reason is the evidence itself.
            self.reasoning.refresh(state)
            target, reason = self.reasoning.vote_decision(player_id, candidates)
            # Compared against what this seat actually told the table, not
            # against another internal number -- otherwise the check is the
            # runtime marking its own homework.
            stated_target = self.reasoning.stated_target(player_id)
            output = VoteOutput(
                vote_target=target,
                reason=reason,
                decisive_evidence=reason,
                alternative_target=self.reasoning.seats[
                    player_id
                ].belief.state.alternative_target,
            )
        else:
            system, messages = self._context.build_vote_context(state, player_id, candidates)
            output = await self._agents[player_id].generate_vote(
                system, messages, candidates, preferred_targets=self._belief_targets(player_id)
            )
        self._record(
            state,
            player_id,
            "vote",
            text=output.reason,
            target=output.vote_target,
            decision_evidence=output.decisive_evidence,
            countercase=output.countercase,
            alternative_target=output.alternative_target,
        )
        try:
            controller.vote(player_id, output.vote_target)  # type: ignore[attr-defined]
        except Exception:
            return
        # Voting against your own stated plan is legal play, not corruption, so
        # this is recorded rather than corrected -- but silently losing it means
        # nobody can tell a change of mind from an AI that lost track of itself.
        # The ballot the engine actually holds, not the model's own answer:
        # `controller.vote` may have rejected it, and `vote_records` is only
        # written at tally time.
        mismatch = detect_vote_plan_mismatch(
            PublicFactLedger(state),
            voter_id=player_id,
            stated_target=stated_target if isinstance(stated_target, str) else None,
            day=state.day,
            round_number=state.vote_round,
            actual_target=state.pending_votes.get(player_id),
        )
        if mismatch is not None:
            self.validation.vote_plan_mismatches.append(mismatch)
            self._record(
                state,
                player_id,
                "vote_change",
                text=(
                    f"公開発言では{mismatch.stated_target}を第一候補としたが、"
                    f"最終判断で{mismatch.actual_target}へ変更した。{output.reason}"
                ),
                target=mismatch.actual_target,
                alternative_target=mismatch.stated_target,
            )

    async def _generate_day_summary(self, controller: object, state: GameState) -> None:
        # The factual half is rendered from the ledger, not asked for: who died,
        # who COed, which verdicts were published and who voted for whom are
        # already known exactly, and a model restating them can only lose them.
        facts = render_public_fact_summary(PublicFactLedger(state), state.day)
        alive_ai = [pid for pid in self._ai_player_ids if state.players[pid].alive]
        if not alive_ai:
            self._day_summaries.set_summary(state.day, "", facts)
            self._day_summaries.compress_if_needed()
            return
        narrator_id = alive_ai[0]
        if self.reasoning is not None:
            # The disagreements are already structured events; narrating them
            # back through a model adds a request and a chance to lose one.
            self.reasoning.refresh(state)
            commentary = "。".join(self.reasoning.conflict_points(state))
            self._record(state, narrator_id, "summary", text=commentary)
            self._day_summaries.set_summary(state.day, commentary, facts)
            self._day_summaries.compress_if_needed()
            return
        system, messages = self._context.build_summary_context(state, narrator_id)
        output = await self._agents[narrator_id].generate_summary(system, messages)
        self._record(state, narrator_id, "summary", text=output.summary)
        self._day_summaries.set_summary(state.day, output.summary, facts)
        self._day_summaries.compress_if_needed()

    def day_summary_text(self, day: int) -> str:
        """The stored summary for a day: deterministic facts plus the labelled
        commentary the narrator generated on top of them."""
        return compose_day_summary(
            self._day_summaries.facts.get(day, ""), self._day_summaries.summaries.get(day, "")
        )

    # -- night --

    async def run_night_phase(self, session: object) -> None:
        controller = session.controller  # type: ignore[attr-defined]
        state = controller.state

        if state.phase != Phase.NIGHT:
            return

        if state.day == 0:
            if state.pending_divine is None:
                seer_id = self._find_ai_with_role(state, RoleName.SEER)
                if seer_id is not None:
                    await self._cast_divine(controller, state, seer_id)
            try:
                controller.resolve_night()
            except Exception:
                pass
            return

        tasks = []
        if state.pending_divine is None:
            seer_id = self._find_ai_with_role(state, RoleName.SEER)
            if seer_id is not None:
                tasks.append(self._cast_divine(controller, state, seer_id))
        if state.pending_guard is None:
            hunter_id = self._find_ai_with_role(state, RoleName.HUNTER)
            if hunter_id is not None:
                tasks.append(self._cast_guard(controller, state, hunter_id))
        tasks.append(self._run_wolf_chat_round(controller, state))
        tasks.append(self._run_freemason_chat_round(controller, state))
        if tasks:
            await asyncio.gather(*tasks)

        if state.pending_attack is None:
            alpha_id = controller.alpha_wolf_id
            if alpha_id in self._agents and state.players[alpha_id].alive:
                await self._cast_attack(controller, state, alpha_id)

        try:
            controller.resolve_night()
        except Exception:
            pass

    async def _cast_divine(self, controller: object, state: GameState, seer_id: str) -> None:
        candidates = [
            pid
            for pid in state.alive_ids()
            if pid != seer_id and pid != state.first_victim_id
        ]
        candidates = [pid for pid in candidates if pid not in self._observer_player_ids]
        if not candidates:
            return
        system, messages = self._context.build_night_action_context(
            state, seer_id, "divine", candidates
        )
        coded = self._coded_night_target(seer_id, "divine", candidates)
        if coded is not None:
            output = coded
        else:
            output = await self._agents[seer_id].generate_night_action(
                system, messages, candidates, preferred_targets=self._belief_targets(seer_id)
            )
        self._record(state, seer_id, "night_action", text=output.reason, target=output.target)
        try:
            controller.submit_night_action(seer_id, "divine", output.target)  # type: ignore[attr-defined]
        except Exception:
            pass

    async def _cast_guard(self, controller: object, state: GameState, hunter_id: str) -> None:
        candidates = [pid for pid in state.alive_ids() if pid != hunter_id]
        candidates = [pid for pid in candidates if pid not in self._observer_player_ids]
        if not candidates:
            return
        system, messages = self._context.build_night_action_context(
            state, hunter_id, "guard", candidates
        )
        # The hunter protects who they trust, so their fallback preference is the
        # trusted list rather than the suspect list every other action uses.
        memo = self._context.get_reasoning_memo(hunter_id) or {}
        trusted = [
            value
            for value in (memo.get("trusted_seer"), *(memo.get("trusted") or []))
            if isinstance(value, str) and value
        ]
        coded = self._coded_night_target(hunter_id, "guard", candidates)
        if coded is not None:
            output = coded
        else:
            output = await self._agents[hunter_id].generate_night_action(
                system, messages, candidates, preferred_targets=trusted
            )
        self._record(state, hunter_id, "night_action", text=output.reason, target=output.target)
        try:
            controller.submit_night_action(hunter_id, "guard", output.target)  # type: ignore[attr-defined]
        except Exception:
            pass

    async def _cast_attack(self, controller: object, state: GameState, alpha_id: str) -> None:
        candidates = [
            pid
            for pid in state.alive_ids()
            if state.players[pid].role != RoleName.WEREWOLF
            and pid not in self._observer_player_ids
        ]
        if not candidates:
            return
        system, messages = self._context.build_night_action_context(
            state, alpha_id, "attack", candidates
        )
        coded = self._coded_night_target(alpha_id, "attack", candidates)
        if coded is not None:
            output = coded
        else:
            output = await self._agents[alpha_id].generate_night_action(
                system, messages, candidates, preferred_targets=self._belief_targets(alpha_id)
            )
        self._record(state, alpha_id, "night_action", text=output.reason, target=output.target)
        try:
            controller.submit_night_action(alpha_id, "attack", output.target)  # type: ignore[attr-defined]
        except Exception:
            pass

    async def _run_wolf_chat_round(self, controller: object, state: GameState) -> None:
        wolf_ids = [
            pid
            for pid in self._ai_player_ids
            if state.players[pid].alive and state.players[pid].role == RoleName.WEREWOLF
        ]
        if self.reasoning is not None and wolf_ids:
            # One plan for the team, not one soliloquy each. Wolves share every
            # fact they have, so N requests bought N restatements of it.
            await self._run_team_plan(controller, state, wolf_ids, "wolf")
            return
        for pid in wolf_ids:
            system, messages = self._context.build_wolf_chat_context(state, pid)
            output = await self._agents[pid].generate_wolf_chat(system, messages)
            self._record(
                state,
                pid,
                "wolf_chat",
                text=output.message,
                used_fallback=self._is_fallback(pid, output.message),
                effective_length_limit=_PRIVATE_CHAT_MAX_CHARS,
            )
            try:
                controller.chat(pid, output.message, "wolf")  # type: ignore[attr-defined]
            except Exception:
                pass

    async def _run_freemason_chat_round(self, controller: object, state: GameState) -> None:
        mason_ids = [
            pid
            for pid in self._ai_player_ids
            if state.players[pid].alive and state.players[pid].role == RoleName.FREEMASON
        ]
        if self.reasoning is not None and mason_ids:
            await self._run_team_plan(controller, state, mason_ids, "freemason")
            return
        for pid in mason_ids:
            system, messages = self._context.build_freemason_chat_context(state, pid)
            output = await self._agents[pid].generate_wolf_chat(system, messages)
            self._record(
                state,
                pid,
                "freemason_chat",
                text=output.message,
                used_fallback=self._is_fallback(pid, output.message),
                effective_length_limit=_PRIVATE_CHAT_MAX_CHARS,
            )
            try:
                controller.chat(pid, output.message, "freemason")  # type: ignore[attr-defined]
            except Exception:
                pass

    async def _run_team_plan(
        self, controller: object, state: GameState, member_ids: list[str], channel: str
    ) -> None:
        """One request for the whole team's night plan."""
        assert self.reasoning is not None
        speaker = member_ids[0]
        builder = (
            self._context.build_freemason_chat_context
            if channel == "freemason"
            else self._context.build_wolf_chat_context
        )
        system, messages = builder(state, speaker)
        output = await self._agents[speaker].generate_wolf_chat(system, messages)
        self._record(
            state,
            speaker,
            f"{channel}_chat",
            text=output.message,
            used_fallback=self._is_fallback(speaker, output.message),
            effective_length_limit=_PRIVATE_CHAT_MAX_CHARS,
        )
        try:
            controller.chat(speaker, output.message, channel)  # type: ignore[attr-defined]
        except Exception:
            return
        # The rest of the team acknowledges in code: their agreement carries no
        # information the plan does not already state.
        for pid in member_ids[1:]:
            target = self.reasoning.seats[pid].belief.state.current_execution_target
            note = f"了解。自分の第一候補は{target}。" if target else "了解。"
            try:
                controller.chat(pid, note, channel)  # type: ignore[attr-defined]
            except Exception:
                continue

    async def respond_to_private_chat(self, session: object, channel: str) -> None:
        """Let living AI teammates answer a human private message immediately."""
        controller = session.controller  # type: ignore[attr-defined]
        state = controller.state
        role = RoleName.FREEMASON if channel == "freemason" else RoleName.WEREWOLF
        responders = [
            pid
            for pid in self._ai_player_ids
            if state.players[pid].alive and state.players[pid].role == role
        ]
        for pid in responders:
            builder = (
                self._context.build_freemason_chat_context
                if channel == "freemason"
                else self._context.build_wolf_chat_context
            )
            system, messages = builder(state, pid)
            controller.set_typing(pid, True, channel)
            try:
                output = await self._agents[pid].generate_wolf_chat(system, messages)
            finally:
                controller.set_typing(pid, False, channel)
            try:
                controller.chat(pid, output.message, channel)
            except Exception:
                pass
