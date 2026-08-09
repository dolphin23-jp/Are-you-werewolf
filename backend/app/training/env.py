"""Thin Phase-0 self-play adapter over the production GameController.

This module intentionally does not contain a learning algorithm. It gives
random, neural, and human controllers the same information/action protocol and
keeps the production rule engine as the authority for legal game transitions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.engine.game import GameController, GameError, NightActionType, PlayerSpec
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.state import PlayerState
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

    def emit_speech(
        self, actor_id: str, bundle: SpeechBundle
    ) -> tuple[TimedSemanticEvent, ...]:
        """Commit one public turn; callers must replan everyone afterwards."""
        state = self.controller.state
        if state.phase is not Phase.DISCUSSION:
            raise GameError(f"cannot speak during phase {state.phase}")
        self._require_alive(actor_id)
        if any(atom.channel is not Channel.PUBLIC for atom in bundle.atoms):
            raise GameError("public discussion bundles may only contain public atoms")

        committed = self._commit_bundle(
            actor_id,
            bundle,
            logical_tick=self.scheduler.discussion_tick,
            mirror_public_board=True,
        )
        self.scheduler.record_emitted_event()
        return committed

    def emit_private_speech(
        self, actor_id: str, bundle: SpeechBundle
    ) -> tuple[TimedSemanticEvent, ...]:
        """Commit a night-only wolf/freemason semantic message."""
        state = self.controller.state
        if state.phase is not Phase.NIGHT:
            raise GameError("private planning is only available at night")
        player = self._require_alive(actor_id)
        channels = {atom.channel for atom in bundle.atoms}
        if len(channels) != 1 or Channel.PUBLIC in channels:
            raise GameError("a private bundle must use one private channel")
        channel = next(iter(channels))
        if channel is Channel.WOLF and player.role is not RoleName.WEREWOLF:
            raise GameError("only werewolves may use the wolf semantic channel")
        if channel is Channel.FREEMASON and player.role is not RoleName.FREEMASON:
            raise GameError("only freemasons may use the freemason semantic channel")

        return self._commit_bundle(
            actor_id,
            bundle,
            logical_tick=-1,
            mirror_public_board=False,
        )

    def vote(self, voter_id: str, target_id: str) -> None:
        """Votes remain hidden in GameState.pending_votes until resolution."""
        self.controller.vote(voter_id, target_id)

    def night_action(self, player_id: str, topic: Topic, target_id: str) -> None:
        """Night submissions remain hidden until the production resolver runs."""
        action_types: dict[Topic, NightActionType] = {
            Topic.DIVINE: "divine",
            Topic.GUARD: "guard",
            Topic.ATTACK: "attack",
        }
        action_type = action_types.get(topic)
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

    def _require_alive(self, player_id: str) -> PlayerState:
        player = self.controller.state.players.get(player_id)
        if player is None or not player.alive:
            raise GameError(f"player {player_id} is not alive")
        return player

    def _commit_bundle(
        self,
        actor_id: str,
        bundle: SpeechBundle,
        *,
        logical_tick: int,
        mirror_public_board: bool,
    ) -> tuple[TimedSemanticEvent, ...]:
        state = self.controller.state
        committed: list[TimedSemanticEvent] = []
        for atom in bundle.atoms:
            event_id = f"t{self._next_semantic_event_number}"
            if mirror_public_board:
                self._apply_public_board_effect(actor_id, atom, source_event_id=event_id)
            event = TimedSemanticEvent(
                event_id=event_id,
                actor_id=actor_id,
                day=state.day,
                discussion_tick=logical_tick,
                action=atom,
            )
            self.semantic_events.append(event)
            committed.append(event)
            self._next_semantic_event_number += 1
        return tuple(committed)

    def _apply_public_board_effect(
        self, actor_id: str, action: SemanticAction, *, source_event_id: str
    ) -> None:
        """Mirror only board-defining semantics into the production event log."""
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

        if action.action_type is ActionType.RETRACT:
            referenced = self._semantic_event(action.reference_event_id)
            if referenced is None or referenced.actor_id != actor_id:
                return
            previous = referenced.action
            if previous.action_type is ActionType.CLAIM and previous.role is not None:
                self.controller.retract_co(actor_id, source_message_id=source_event_id)
                return
            if (
                previous.action_type is ActionType.REPORT
                and previous.topic in (Topic.SEER_RESULT, Topic.MEDIUM_RESULT)
                and previous.target_id is not None
            ):
                retracted_result_type = (
                    "seer" if previous.topic is Topic.SEER_RESULT else "medium"
                )
                self.controller.retract_public_result(
                    actor_id,
                    retracted_result_type,
                    previous.target_id,
                    source_message_id=source_event_id,
                    referenced_day=previous.referenced_day,
                )
            return

        if action.action_type not in (ActionType.REPORT, ActionType.CORRECT):
            return
        if action.topic is None:
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

    def _semantic_event(self, event_id: str | None) -> TimedSemanticEvent | None:
        if event_id is None:
            return None
        return next(
            (event for event in reversed(self.semantic_events) if event.event_id == event_id),
            None,
        )
