"""GameController: the single façade every caller (human via API, or AI via
the coordinator) drives the game through. The engine has zero knowledge of
LLMs/prompts; this class only knows game rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from app.engine.events import EventBus, GameEvent, GameEventType
from app.engine.night_resolver import NightResolver
from app.engine.phases import Phase, PhaseEvent, next_phase
from app.engine.roles import ROLE_DEFINITIONS, AlphaWolfTracker, RoleAssigner, RoleName
from app.engine.speech_events import (
    MEDIUM_RESULT,
    RESULT_EVENT_TYPES,
    RESULT_TYPES,
    SEER_RESULT,
    SpeechEvent,
    SpeechEventType,
    active_result,
    current_role_claim,
    result_role,
)
from app.engine.state import (
    ChatChannel,
    ChatMessage,
    GameState,
    PlayerState,
)
from app.engine.victory import VictoryChecker
from app.engine.vote import VoteManager

NightActionType = Literal["divine", "guard", "attack"]

_RESULT_TYPES: dict[RoleName, str] = {
    RoleName.SEER: SEER_RESULT,
    RoleName.MEDIUM: MEDIUM_RESULT,
}


@dataclass(frozen=True)
class PlayerSpec:
    player_id: str
    name: str
    is_human: bool = False


class GameError(ValueError):
    """Raised for illegal actions; safe to surface to API callers as 4xx."""


class GameController:
    def __init__(
        self,
        session_id: str,
        player_specs: list[PlayerSpec],
        seed: int | None = None,
        forced_roles: dict[str, RoleName] | None = None,
        inactive_player_ids: set[str] | None = None,
    ) -> None:
        if len(player_specs) != 17:
            raise GameError(f"expected 17 players, got {len(player_specs)}")

        self._assigner = RoleAssigner(seed=seed)
        assignment = self._assigner.assign([p.player_id for p in player_specs])
        # Evaluation can reserve a non-participating human seat as an ordinary
        # villager without changing the overall 17-player composition. Partial
        # overrides are implemented as deterministic swaps, never by adding or
        # removing a role.
        for player_id, role in (forced_roles or {}).items():
            if player_id not in assignment:
                raise GameError(f"cannot force role for unknown player {player_id}")
            if assignment[player_id] == role:
                continue
            swap_id = next(
                (
                    pid
                    for pid, assigned in assignment.items()
                    if assigned == role and pid != player_id
                ),
                None,
            )
            if swap_id is None:
                raise GameError(f"role {role} is not present in the assignment")
            assignment[player_id], assignment[swap_id] = assignment[swap_id], assignment[player_id]

        inactive_ids = inactive_player_ids or set()
        unknown_inactive = inactive_ids - set(assignment)
        if unknown_inactive:
            raise GameError(f"unknown inactive players: {sorted(unknown_inactive)}")

        players = {
            spec.player_id: PlayerState(
                player_id=spec.player_id,
                name=spec.name,
                role=assignment[spec.player_id],
                is_human=spec.is_human,
                alive=spec.player_id not in inactive_ids,
            )
            for spec in player_specs
        }
        self.state = GameState(session_id=session_id, players=players)

        wolf_ids = [pid for pid, role in assignment.items() if role == RoleName.WEREWOLF]
        self._alpha_tracker = AlphaWolfTracker(wolf_ids, seed=seed)

        self._night_resolver = NightResolver()
        self._vote_manager = VoteManager()
        self._victory_checker = VictoryChecker()
        self.events = EventBus()


    # -- read-only views --

    def get_player_view(self, player_id: str) -> dict[str, Any]:
        view = self.state.get_player_view(player_id)
        player = self.state.players.get(player_id)
        # Only revealed to werewolves themselves -- who the alpha is would
        # otherwise leak a player's team to non-wolf viewers.
        if player is not None and player.role == RoleName.WEREWOLF:
            view["is_alpha_wolf"] = player_id == self._alpha_tracker.alpha_id
        return view

    def get_debug_view(self) -> dict[str, Any]:
        return self.state.get_debug_view()

    @property
    def alpha_wolf_id(self) -> str:
        return self._alpha_tracker.alpha_id

    # -- phase transitions --

    def start_game(self) -> None:
        self._transition(PhaseEvent.START_GAME)
        active_assignment = {
            pid: player.role for pid, player in self.state.players.items() if player.alive
        }
        self.state.first_victim_id = self._assigner.pick_first_victim(active_assignment)

    def resolve_night(self) -> None:
        if self.state.phase != Phase.NIGHT:
            raise GameError(f"cannot resolve night during phase {self.state.phase}")

        if self.state.day == 0:
            if self.state.first_victim_id is None:
                raise GameError("first victim not selected; call start_game() first")
            result = self._night_resolver.resolve_day_zero(self.state, self.state.first_victim_id)
        else:
            result = self._night_resolver.resolve_night(self.state)

        for death in result.deaths:
            player = self.state.players[death.player_id]
            if player.role == RoleName.WEREWOLF:
                self._alpha_tracker.on_wolf_death(death.player_id)
            self.events.publish(
                GameEvent(
                    GameEventType.PLAYER_DIED, {"player_id": death.player_id, "cause": death.cause}
                )
            )

        if result.divine_result is not None:
            self.events.publish(
                GameEvent(
                    GameEventType.DIVINE_RESULT,
                    {
                        "target_id": result.divine_result.target_id,
                        "is_werewolf": result.divine_result.is_werewolf,
                    },
                    recipient_id=result.divine_result.seer_id,
                )
            )
        for medium_result in result.medium_results or []:
            self.events.publish(
                GameEvent(
                    GameEventType.MEDIUM_RESULT,
                    {
                        "target_id": medium_result.target_id,
                        "is_werewolf": medium_result.is_werewolf,
                    },
                    recipient_id=medium_result.medium_id,
                )
            )

        if not self._check_victory():
            self._transition(PhaseEvent.RESOLVE_NIGHT)

    def start_discussion(self) -> None:
        self._transition(PhaseEvent.START_DISCUSSION)
        self.state.day += 1

    def end_discussion(self) -> None:
        self.state.vote_round = 1
        self.state.runoff_candidates = []
        self._transition(PhaseEvent.END_DISCUSSION)

    def resolve_votes(self) -> None:
        if self.state.phase not in (Phase.VOTING, Phase.RUNOFF):
            raise GameError(f"cannot resolve votes during phase {self.state.phase}")

        result = self._vote_manager.tally(self.state)

        if result.executed_player_id is not None:
            player = self.state.players[result.executed_player_id]
            if player.role == RoleName.WEREWOLF:
                self._alpha_tracker.on_wolf_death(result.executed_player_id)
            self.events.publish(
                GameEvent(
                    GameEventType.VOTE_RESULT,
                    {"executed_player_id": result.executed_player_id, "is_draw": False},
                )
            )
            if not self._check_victory():
                self._transition(PhaseEvent.VOTE_RESOLVED)
            return

        if result.is_draw:
            # Vote deadlock past max_vote_rounds ends the whole game as a
            # draw, not just this day's execution.
            self.state.is_draw = True
            self.state.victory_reason = "投票が上限ラウンドに達したため引き分け"
            self.state.phase = Phase.GAME_OVER
            self.events.publish(
                GameEvent(GameEventType.VOTE_RESULT, {"executed_player_id": None, "is_draw": True})
            )
            self.events.publish(
                GameEvent(
                    GameEventType.VICTORY, {"winner": None, "reason": self.state.victory_reason}
                )
            )
            return

        # tie -> runoff
        self.events.publish(
            GameEvent(
                GameEventType.VOTE_RESULT,
                {
                    "executed_player_id": None,
                    "is_draw": False,
                    "tied_player_ids": result.tied_player_ids,
                },
            )
        )
        self._transition(PhaseEvent.VOTE_TIE)

    def start_night(self) -> None:
        self._transition(PhaseEvent.START_NIGHT)

    # -- player actions --

    def chat(
        self,
        author_id: str,
        content: str,
        channel: str = "public",
        reply_to: str | None = None,
        quote: str | None = None,
        references: list[str] | None = None,
    ) -> str:
        player = self._require_alive(author_id)
        chat_channel = ChatChannel(channel)
        if chat_channel == ChatChannel.WOLF and player.role != RoleName.WEREWOLF:
            raise GameError("only werewolves may use the wolf channel")
        if chat_channel == ChatChannel.FREEMASON and player.role != RoleName.FREEMASON:
            raise GameError("only freemasons may use the freemason channel")

        referenced = next(
            (
                message
                for message in self.state.chat_log
                if message.message_id == reply_to and message.channel == chat_channel
            ),
            None,
        )
        if referenced is None:
            reply_to = None
            quote = None
        valid_references = [
            message_id
            for message_id in dict.fromkeys(references or [])
            if message_id != reply_to
            and any(
                message.message_id == message_id and message.channel == chat_channel
                for message in self.state.chat_log
            )
        ][:10]
        message_id = f"m{self.state.next_message_number}"
        self.state.next_message_number += 1
        message = ChatMessage(
            message_id=message_id,
            author_id=author_id,
            content=content,
            channel=chat_channel,
            day=self.state.day,
            reply_to=reply_to,
            quote=quote,
            references=valid_references,
        )
        self.state.chat_log.append(message)
        answered_ids = {message_id for message_id in [reply_to, *valid_references] if message_id}
        if answered_ids:
            pending = self.state.pending_questions.get(author_id, [])
            self.state.pending_questions[author_id] = [
                question for question in pending if question.source_message_id not in answered_ids
            ]
        self.events.publish(
            GameEvent(
                GameEventType.CHAT_MESSAGE,
                {
                    "message_id": message_id,
                    "author_id": author_id,
                    "content": content,
                    "channel": chat_channel,
                    "day": self.state.day,
                    "reply_to": reply_to,
                    "quote": quote,
                    "references": valid_references,
                },
            )
        )
        return message_id

    def vote(self, voter_id: str, target_id: str) -> None:
        if self.state.phase not in (Phase.VOTING, Phase.RUNOFF):
            raise GameError(f"cannot vote during phase {self.state.phase}")
        try:
            self._vote_manager.record_vote(self.state, voter_id, target_id)
        except GameError:
            raise
        except ValueError as exc:
            # The vote manager lives below GameError in the import graph, so it
            # signals illegal votes with plain ValueError. Translate here so the
            # API answers 400 instead of leaking a 500.
            raise GameError(str(exc)) from exc

    def submit_night_action(
        self, player_id: str, action_type: NightActionType, target_id: str
    ) -> None:
        if self.state.phase != Phase.NIGHT:
            raise GameError(f"cannot submit night action during phase {self.state.phase}")
        player = self._require_alive(player_id)
        self._require_alive(target_id)

        if self.state.day == 0 and action_type != "divine":
            raise GameError("only the seer may act on the Day-0 night")

        if action_type == "divine":
            if player.role != RoleName.SEER:
                raise GameError("only the seer may divine")
            if target_id == self.state.first_victim_id:
                raise GameError("the first victim cannot be divined")
            self.state.pending_divine = (player_id, target_id)
        elif action_type == "guard":
            if player.role != RoleName.HUNTER:
                raise GameError("only the hunter may guard")
            if target_id == player_id:
                raise GameError("the hunter cannot guard themselves")
            self.state.pending_guard = (player_id, target_id)
        elif action_type == "attack":
            if player.role != RoleName.WEREWOLF:
                raise GameError("only werewolves may attack")
            if player_id != self._alpha_tracker.alpha_id:
                raise GameError("only the alpha werewolf may submit the attack")
            self.state.pending_attack = (player_id, target_id)
        else:
            raise GameError(f"unknown action_type {action_type}")

    def record_speech_event(
        self,
        actor_id: str,
        event_type: SpeechEventType,
        *,
        source_message_id: str = "",
        target_id: str | None = None,
        role: RoleName | None = None,
        result_is_werewolf: bool | None = None,
        referenced_day: int | None = None,
        confidence: float = 1.0,
    ) -> SpeechEvent | None:
        """The single write path for every public claim.

        `co`, `public_result` and `claim_freemason_partner` are thin wrappers
        over this, so there is no second place that can record a claim and drift
        from the log. Returns None when the event adds nothing (a same-day
        restatement of an identical verdict, a partner claim already made).
        """
        self._require_alive(actor_id)
        if event_type in (SpeechEventType.ROLE_CLAIM, SpeechEventType.ROLE_SWITCH):
            if role is None or role not in ROLE_DEFINITIONS:
                raise GameError(f"unknown claimed role {role}")
        if event_type in RESULT_EVENT_TYPES:
            if role not in (RoleName.SEER, RoleName.MEDIUM):
                raise GameError("a published result must name the seer or medium ability")
            if target_id not in self.state.players:
                raise GameError("invalid public result claim")
            if self._is_redundant_result(
                actor_id, event_type, role, target_id, is_werewolf=result_is_werewolf
            ):
                return None
        if event_type is SpeechEventType.PARTNER_CLAIM:
            if target_id not in self.state.players or target_id == actor_id:
                raise GameError("invalid freemason partner")
            if any(
                claim.claimant_id == actor_id and claim.partner_id == target_id
                for claim in self.state.freemason_partner_claims
            ):
                return None

        role_before = current_role_claim(self.state.speech_events, actor_id)
        event = self.state.append_speech_event(
            actor_id,
            event_type,
            source_message_id=source_message_id,
            target_id=target_id,
            role=role,
            result_is_werewolf=result_is_werewolf,
            referenced_day=referenced_day,
            confidence=confidence,
        )
        role_after = current_role_claim(self.state.speech_events, actor_id)
        # Only an actual change in the standing claim is news. A reaffirmation
        # and an unpromoted low-confidence guess both leave the board as it was.
        if role_after is not None and (role_before is None or role_before.role != role_after.role):
            self.events.publish(
                GameEvent(
                    GameEventType.CO_DECLARED,
                    {"player_id": actor_id, "claimed_role": role_after.role},
                )
            )
        return event

    def _is_redundant_result(
        self,
        actor_id: str,
        event_type: SpeechEventType,
        role: RoleName,
        target_id: str,
        *,
        is_werewolf: bool | None,
    ) -> bool:
        """Saying the same verdict twice in one day is emphasis, not a new
        version. A correction or a retraction always counts."""
        if event_type is not SpeechEventType.ABILITY_RESULT:
            return False
        existing = active_result(
            self.state.speech_events, actor_id, _RESULT_TYPES[role], target_id
        )
        return (
            existing is not None
            and existing.day == self.state.day
            and existing.is_werewolf == bool(is_werewolf)
        )

    def co(
        self,
        player_id: str,
        claimed_role: str,
        source_message_id: str = "",
        confidence: float = 1.0,
    ) -> None:
        try:
            role = RoleName(claimed_role)
        except ValueError as exc:
            raise GameError(f"unknown role {claimed_role}") from exc
        self.record_speech_event(
            player_id,
            SpeechEventType.ROLE_CLAIM,
            role=role,
            source_message_id=source_message_id,
            confidence=confidence,
        )

    def retract_co(self, player_id: str, source_message_id: str = "") -> None:
        self.record_speech_event(
            player_id, SpeechEventType.ROLE_RETRACTION, source_message_id=source_message_id
        )

    def claim_freemason_partner(
        self, claimant_id: str, partner_id: str, source_message_id: str = ""
    ) -> None:
        """Record only the public relationship claim, never the secret role truth."""
        self.record_speech_event(
            claimant_id,
            SpeechEventType.PARTNER_CLAIM,
            target_id=partner_id,
            source_message_id=source_message_id,
        )

    def public_result(
        self,
        claimant_id: str,
        result_type: str,
        target_id: str,
        is_werewolf: bool,
        source_message_id: str = "",
        confidence: float = 1.0,
        referenced_day: int | None = None,
    ) -> None:
        self._record_result_event(
            claimant_id,
            SpeechEventType.ABILITY_RESULT,
            result_type,
            target_id,
            is_werewolf=is_werewolf,
            source_message_id=source_message_id,
            referenced_day=referenced_day,
            confidence=confidence,
        )

    def correct_public_result(
        self,
        claimant_id: str,
        result_type: str,
        target_id: str,
        is_werewolf: bool,
        source_message_id: str = "",
        referenced_day: int | None = None,
    ) -> None:
        """Publish a verdict that replaces an earlier one. The superseded
        version stays in the log so "what did they say first" is answerable."""
        self._record_result_event(
            claimant_id,
            SpeechEventType.RESULT_CORRECTION,
            result_type,
            target_id,
            is_werewolf=is_werewolf,
            source_message_id=source_message_id,
            referenced_day=referenced_day,
        )

    def retract_public_result(
        self,
        claimant_id: str,
        result_type: str,
        target_id: str,
        source_message_id: str = "",
        referenced_day: int | None = None,
    ) -> None:
        self._record_result_event(
            claimant_id,
            SpeechEventType.RESULT_RETRACTION,
            result_type,
            target_id,
            is_werewolf=None,
            source_message_id=source_message_id,
            referenced_day=referenced_day,
        )

    def _record_result_event(
        self,
        claimant_id: str,
        event_type: SpeechEventType,
        result_type: str,
        target_id: str,
        *,
        is_werewolf: bool | None,
        source_message_id: str = "",
        referenced_day: int | None = None,
        confidence: float = 1.0,
    ) -> None:
        if result_type not in RESULT_TYPES:
            raise GameError("invalid public result claim")
        self.record_speech_event(
            claimant_id,
            event_type,
            role=result_role(result_type),
            target_id=target_id,
            result_is_werewolf=is_werewolf,
            source_message_id=source_message_id,
            referenced_day=referenced_day,
            confidence=confidence,
        )

    def set_typing(self, player_id: str, typing: bool, channel: str = "public") -> None:
        if typing:
            self.state.typing_channels[player_id] = channel
        else:
            self.state.typing_channels.pop(player_id, None)
        self.events.publish(
            GameEvent(
                GameEventType.TYPING_CHANGED,
                {"player_id": player_id, "typing": typing, "channel": channel},
            )
        )

    # -- internals --

    def _require_alive(self, player_id: str) -> PlayerState:
        player = self.state.players.get(player_id)
        if player is None:
            raise GameError(f"unknown player {player_id}")
        if not player.alive:
            raise GameError(f"player {player_id} is not alive")
        return player

    def _transition(self, event: PhaseEvent) -> None:
        new_phase = next_phase(self.state.phase, event)
        if new_phase is None:
            raise GameError(f"illegal transition {event} from {self.state.phase}")
        self.state.phase = new_phase
        self.events.publish(
            GameEvent(GameEventType.PHASE_CHANGED, {"phase": new_phase, "day": self.state.day})
        )

    def _check_victory(self) -> bool:
        result = self._victory_checker.check(self.state)
        if result is None:
            return False
        self.state.winner = result.winner
        self.state.victory_reason = result.reason
        self.state.phase = Phase.GAME_OVER
        self.events.publish(
            GameEvent(GameEventType.VICTORY, {"winner": result.winner, "reason": result.reason})
        )
        return True
