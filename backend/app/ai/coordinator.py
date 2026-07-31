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

from app.ai.co_detection import detect_claimed_role
from app.ai.context import ContextBuilder, DaySummaryManager
from app.ai.deception import FakeClaimGuard, assign_madman_strategy, assign_wolf_deception
from app.ai.personalities import assign_personalities
from app.ai.player_agent import AIPlayerAgent
from app.ai.provider.base import LLMProvider
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.state import GameState
from app.eval.transcript import TranscriptRecorder, Utterance


class AICoordinator:
    def __init__(
        self,
        state: GameState,
        ai_player_ids: list[str],
        provider: LLMProvider,
        seed: int | None = None,
        recorder: TranscriptRecorder | None = None,
    ) -> None:
        self._ai_player_ids = list(ai_player_ids)

        self._personalities = assign_personalities(self._ai_player_ids, seed=seed)

        wolf_ids = [p.player_id for p in state.players_by_role(RoleName.WEREWOLF)]
        madman = state.players_by_role(RoleName.MADMAN)
        wolf_deception = assign_wolf_deception(wolf_ids, seed=seed)
        madman_strategy_name, madman_fake_role = assign_madman_strategy(seed=seed)
        fake_claim_guard = FakeClaimGuard(wolf_team_ids=set(wolf_ids))
        self._wolf_deception = wolf_deception

        self._day_summaries = DaySummaryManager()
        self._context = ContextBuilder(
            personalities=self._personalities,
            day_summaries=self._day_summaries,
            wolf_deception=wolf_deception,
            madman_fake_role=madman_fake_role if madman else None,
            fake_claim_guard=fake_claim_guard,
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
                        pid: role.value for pid, role in wolf_deception.fake_role_by_player.items()
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
                used_fallback=used_fallback,
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
        async with session.discussion_lock:  # type: ignore[attr-defined]
            controller = session.controller  # type: ignore[attr-defined]
            state = controller.state
            for pid in self._ai_player_ids:
                player = state.players.get(pid)
                if player is None or not player.alive:
                    continue
                if state.phase != Phase.DISCUSSION:
                    return
                system, messages = self._context.build_discussion_context(state, pid)
                output = await self._agents[pid].generate_discussion(system, messages)
                self._record(
                    state,
                    pid,
                    "discussion",
                    text=output.public_message,
                    reasoning_memo=output.reasoning_memo,
                    used_fallback=self._is_fallback(pid, output.public_message),
                )
                try:
                    controller.chat(pid, output.public_message, "public")
                except Exception:
                    continue
                self._maybe_register_co(controller, pid, output.public_message)

    def _is_fallback(self, player_id: str, text: str) -> bool:
        """The agent substitutes a personality-specific canned line when the
        model fails entirely. Those lines are distinctive enough that an
        exact match is a reliable signal, and it keeps the agent's return
        types unchanged."""
        return text.strip() == self._personalities[player_id].get_fallback_message()

    def _maybe_register_co(self, controller: object, player_id: str, message: str) -> None:
        state = controller.state  # type: ignore[attr-defined]
        already = any(c.player_id == player_id for c in state.co_declarations)
        if already:
            return
        other_names = [p.name for pid, p in state.players.items() if pid != player_id]
        role = detect_claimed_role(message, other_names)
        if role is None:
            return
        try:
            controller.co(player_id, role.value)  # type: ignore[attr-defined]
        except Exception:
            pass

    # -- voting (loops across runoff rounds) --

    async def generate_all_votes(self, session: object) -> None:
        controller = session.controller  # type: ignore[attr-defined]
        state = controller.state
        human_id = session.human_id  # type: ignore[attr-defined]

        while state.phase in (Phase.VOTING, Phase.RUNOFF):
            human = state.players[human_id]
            if human.alive and human_id not in state.pending_votes:
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
        if not candidates:
            return
        system, messages = self._context.build_vote_context(state, player_id, candidates)
        output = await self._agents[player_id].generate_vote(system, messages, candidates)
        self._record(state, player_id, "vote", text=output.reason, target=output.vote_target)
        try:
            controller.vote(player_id, output.vote_target)  # type: ignore[attr-defined]
        except Exception:
            pass

    async def _generate_day_summary(self, controller: object, state: GameState) -> None:
        alive_ai = [pid for pid in self._ai_player_ids if state.players[pid].alive]
        if not alive_ai:
            return
        narrator_id = alive_ai[0]
        system, messages = self._context.build_summary_context(state, narrator_id)
        output = await self._agents[narrator_id].generate_summary(system, messages)
        self._record(state, narrator_id, "summary", text=output.summary)
        self._day_summaries.set_summary(state.day, output.summary)
        self._day_summaries.compress_if_needed()

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
        candidates = [pid for pid in state.alive_ids() if pid != seer_id]
        if not candidates:
            return
        system, messages = self._context.build_night_action_context(
            state, seer_id, "divine", candidates
        )
        output = await self._agents[seer_id].generate_night_action(system, messages, candidates)
        self._record(state, seer_id, "night_action", text=output.reason, target=output.target)
        try:
            controller.submit_night_action(seer_id, "divine", output.target)  # type: ignore[attr-defined]
        except Exception:
            pass

    async def _cast_guard(self, controller: object, state: GameState, hunter_id: str) -> None:
        candidates = [pid for pid in state.alive_ids() if pid != hunter_id]
        if not candidates:
            return
        system, messages = self._context.build_night_action_context(
            state, hunter_id, "guard", candidates
        )
        output = await self._agents[hunter_id].generate_night_action(system, messages, candidates)
        self._record(state, hunter_id, "night_action", text=output.reason, target=output.target)
        try:
            controller.submit_night_action(hunter_id, "guard", output.target)  # type: ignore[attr-defined]
        except Exception:
            pass

    async def _cast_attack(self, controller: object, state: GameState, alpha_id: str) -> None:
        candidates = [
            pid for pid in state.alive_ids() if state.players[pid].role != RoleName.WEREWOLF
        ]
        if not candidates:
            return
        system, messages = self._context.build_night_action_context(
            state, alpha_id, "attack", candidates
        )
        output = await self._agents[alpha_id].generate_night_action(system, messages, candidates)
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
        for pid in wolf_ids:
            system, messages = self._context.build_wolf_chat_context(state, pid)
            output = await self._agents[pid].generate_wolf_chat(system, messages)
            self._record(
                state,
                pid,
                "wolf_chat",
                text=output.message,
                used_fallback=self._is_fallback(pid, output.message),
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
        for pid in mason_ids:
            system, messages = self._context.build_freemason_chat_context(state, pid)
            output = await self._agents[pid].generate_wolf_chat(system, messages)
            self._record(
                state,
                pid,
                "freemason_chat",
                text=output.message,
                used_fallback=self._is_fallback(pid, output.message),
            )
            try:
                controller.chat(pid, output.message, "freemason")  # type: ignore[attr-defined]
            except Exception:
                pass
