"""Thin Phase-0 self-play adapter over the production GameController.

This module intentionally does not contain a learning algorithm. It gives
random, neural, and human controllers the same information/action protocol and
keeps the production rule engine as the authority for legal game transitions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.engine.game import GameController, GameError, PlayerSpec
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.training.actions import (
    ActionType,
    Channel,
    ResultValue,
    SemanticAction,
    SpeechBundle,
    TimedSemanticEvent,
    Topic,
)
from app.training.observation import ObservationBuilder, PolicyObservation
from app.training.scheduler import (
    EventDrivenDiscussionScheduler,
    ScheduledSpeaker,
    SpeakIntent,
)


class WerewolfTrainingEnv:
    """Human-compatible self-play shell around one 17-seat game."""

    def __init__(
        self,
        player_specs: Sequence[PlayerSpec],
        *,
        seed: int | None = None,
        forced_roles: Mapping[str, RoleName] | None = None,
    ) -> None:
        if len(player_specs) != 17:
            raise ValueError("WerewolfTrainingEnv requires exactly 17 seats")
        self._player_specs = tuple(player_specs)
        self._forced_roles = dict(forced_roles or {})
        self._seed = seed
        self._observation_builder = ObservationBuilder()
        self.scheduler = EventDrivenDiscussionScheduler(seed=seed)
        self.semantic_events: list[TimedSemanticEvent] = []
        self._next_semantic_event_number = 1
        self.controller = self._new_controller(seed)
        self.controller.start_game()

    def _new_controller(self, seed: int | None) -> GameController:
        return GameController(
            session_id="training",
            player_specs=list(self._player_specs),
            seed=seed,
            forced_roles=self._forced_roles,
        )

    def reset(self, *, seed: int | None = None) -> dict[str, PolicyObservation]:
        if seed is not None:
            self._seed = seed
        self.controller = self._new_controller(self._seed)
        self.controller.start_game()
        self.scheduler = EventDrivenDiscussionScheduler(seed=self._seed)
        self.semantic_events = []
        self._next_semantic_event_number = 1
        return self.observe_alive()

    def observe(self, player_id: str) -> PolicyObservation:
        return self._observation_builder.build(
            self.controller,
            player_id,
            discussion_tick=self.scheduler.discussion_tick,
            semantic_events=self.semantic_events,
        )

    def observe_alive(self) -> dict[str, PolicyObservation]:
        return {
            player_id: self.observe(player_id)
            for player_id in self.controller.state.alive_ids()
        }

    def select_next_speaker(
        self, intents: Mapping[str, SpeakIntent]
    ) -> ScheduledSpeaker | None:
        return self.scheduler.select_next(intents)

    def emit_speech(self, actor_id: str, bundle: SpeechBundle) -> tuple[TimedSemanticEvent, ...]:
        """Commit one public turn; callers must replan everyone afterwards."""
        state = self.controller.state
        if state.phase is not Phase.DISCUSSION:
            raise GameError(f"cannot speak during phase {state.phase}")
        player = state.players.get(actor_id)
        if player is None or not player.alive:
            raise GameError(f"player {actor_id} is not alive")
        if any(atom.channel is not Channel.PUBLIC for atom in bundle.atoms):
            raise GameError("public discussion bundles may only contain public atoms")

        tick = self.scheduler.discussion_tick
        committed: list[TimedSemanticEvent] = []
        for atom in bundle.atoms:
            event_id = f"t{self._next_semantic_event_number}"
            self._apply_public_board_effect(actor_id, atom, source_event_id=event_id)
            event = TimedSemanticEvent(
                event_id=event_id,
                actor_id=actor_id,
                day=state.day,
                discussion_tick=tick,
                action=atom,
            )
            self.semantic_events.append(event)
            committed.append(event)
            self._next_semantic_event_number += 1

        # A three-atom bundle is one chat turn, so logical time advances once.
        self.scheduler.record_emitted_event()
        return tuple(committed)

    def vote(self, voter_id: str, target_id: str) -> None:
        """Votes remain hidden in GameState.pending_votes until resolution."""
        self.controller.vote(voter_id, target_id)

    def night_action(self, player_id: str, topic: Topic, target_id: str) -> None:
        """Night submissions remain hidden until the production resolver runs."""
        action_type = {
            Topic.DIVINE: "divine",
            Topic.GUARD: "guard",
            Topic.ATTACK: "attack",
        }.get(topic)
        if action_type is None:
            raise GameError(f"{topic} is not a night execution action")
        self.controller.submit_night_action(player_id, action_type, target_id)

    def is_terminal(self) -> bool:
        return self.controller.state.phase is Phase.GAME_OVER

    def rewards(self) -> dict[str, float]:
        """Sparse terminal team reward; no hand-authored intermediate doctrine."""
        state = self.controller.state
        if state.phase is not Phase.GAME_OVER or state.is_draw or state.winner is None:
            return {player_id: 0.0 for player_id in state.players}
        return {
            player_id: (1.0 if player.team is state.winner else -1.0)
            for player_id, player in state.players.items()
        }

    def _apply_public_board_effect(
        self, actor_id: str, action: SemanticAction, *, source_event_id: str
    ) -> None:
        """Mirror only board-defining semantics into the production event log.

        Suspicion, proposals, questions and reactions remain in the richer
        training semantic log. CO/result/partner claims also update the
        production compatibility views so the existing UI/engine can coexist.
        """
        if action.action_type is ActionType.CLAIM:
            if action.topic is Topic.PARTNER and action.target_id is not None:
                self.controller.claim_freemason_partner(
                    actor_id, action.target_id, source_message_id=source_event_id
                )
            elif action.role is not None:
                self.controller.co(
                    actor_id, action.role.value, source_message_id=source_event_id
                )
            return

        if action.action_type is ActionType.RETRACT and action.topic is Topic.ROLE:
            self.controller.retract_co(actor_id, source_message_id=source_event_id)
            return

        if action.action_type not in (ActionType.REPORT, ActionType.CORRECT):
            return
        result_type = {
            Topic.SEER_RESULT: "seer",
            Topic.MEDIUM_RESULT: "medium",
        }.get(action.topic)
        if result_type is None or action.target_id is None:
            return
        if action.result not in (ResultValue.WHITE, ResultValue.BLACK):
            return
        is_werewolf = action.result is ResultValue.BLACK
        if action.action_type is ActionType.CORRECT:
            self.controller.correct_public_result(
                actor_id,
                result_type,
                action.target_id,
                is_werewolf,
                source_message_id=source_event_id,
                referenced_day=action.referenced_day,
            )
        else:
            self.controller.public_result(
                actor_id,
                result_type,
                action.target_id,
                is_werewolf,
                source_message_id=source_event_id,
                referenced_day=action.referenced_day,
            )
