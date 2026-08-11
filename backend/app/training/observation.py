"""Information-safe observations for self-play policies.

The policy receives only public facts plus information the viewing seat is
legitimately entitled to know. Attack success and the reason for a no-death
night are deliberately absent.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from app.engine.game import GameController
from app.engine.roles import RoleName
from app.engine.state import DeathCause
from app.training.actions import Channel, TimedSemanticEvent


@dataclass(frozen=True)
class PublicPlayerObservation:
    player_id: str
    alive: bool
    death_day: int | None
    death_kind: str | None
    current_claim: RoleName | None


@dataclass(frozen=True)
class PublicClaimEventObservation:
    event_id: str
    actor_id: str
    day: int
    ordinal: int
    event_type: str
    target_id: str | None
    role: RoleName | None
    result_is_werewolf: bool | None
    referenced_day: int | None


@dataclass(frozen=True)
class SemanticEventObservation:
    event_id: str
    actor_id: str
    day: int
    discussion_tick: int
    channel: str
    action_type: str
    topic: str | None
    target_id: str | None
    secondary_target_id: str | None
    role: RoleName | None
    result: str | None
    quantity: int | None
    referenced_day: int | None
    scope: str | None
    stance: str | None


@dataclass(frozen=True)
class VoteObservation:
    voter_id: str
    target_id: str
    day: int
    round: int


@dataclass(frozen=True)
class DawnObservation:
    night_day: int
    dead_player_ids: tuple[str, ...]
    no_death: bool


@dataclass(frozen=True)
class PrivateResultObservation:
    target_id: str
    day: int
    is_werewolf: bool


@dataclass(frozen=True)
class PrivateTargetObservation:
    target_id: str
    day: int


@dataclass(frozen=True)
class PrivateObservation:
    role: RoleName
    allies: tuple[str, ...]
    is_alpha_wolf: bool
    divine_results: tuple[PrivateResultObservation, ...]
    medium_results: tuple[PrivateResultObservation, ...]
    guard_history: tuple[PrivateTargetObservation, ...]
    attack_history: tuple[PrivateTargetObservation, ...]
    semantic_events: tuple[SemanticEventObservation, ...]


@dataclass(frozen=True)
class PolicyObservation:
    viewer_id: str
    day: int
    phase: str
    discussion_tick: int
    vote_round: int
    first_victim_id: str | None
    players: tuple[PublicPlayerObservation, ...]
    claim_events: tuple[PublicClaimEventObservation, ...]
    semantic_events: tuple[SemanticEventObservation, ...]
    votes: tuple[VoteObservation, ...]
    dawns: tuple[DawnObservation, ...]
    private: PrivateObservation


class ObservationBuilder:
    def build(
        self,
        controller: GameController,
        viewer_id: str,
        *,
        discussion_tick: int = 0,
        semantic_events: Iterable[TimedSemanticEvent] = (),
    ) -> PolicyObservation:
        """Build one information-safe observation."""
        return self.build_many(
            controller,
            (viewer_id,),
            discussion_tick=discussion_tick,
            semantic_events=semantic_events,
        )[viewer_id]

    def build_many(
        self,
        controller: GameController,
        viewer_ids: Sequence[str],
        *,
        discussion_tick: int = 0,
        semantic_events: Iterable[TimedSemanticEvent] = (),
    ) -> dict[str, PolicyObservation]:
        """Build same-state observations while constructing public data once.

        All returned observations retain viewer-specific private information.
        Only immutable public tuples are shared between viewers, so the policy
        information boundary is identical to repeated :meth:`build` calls.
        """
        if not viewer_ids:
            return {}
        if len(set(viewer_ids)) != len(viewer_ids):
            raise ValueError("viewer_ids must be unique")

        state = controller.state
        viewers = {viewer_id: state.players[viewer_id] for viewer_id in viewer_ids}
        current_claims = {
            claim.player_id: claim.claimed_role for claim in state.co_declarations
        }
        all_semantic_events = tuple(semantic_events)

        players = tuple(
            PublicPlayerObservation(
                player_id=player.player_id,
                alive=player.alive,
                death_day=player.death_day,
                death_kind=_public_death_kind(player.death_cause),
                current_claim=current_claims.get(player.player_id),
            )
            for player in state.players.values()
        )

        claim_events = tuple(
            PublicClaimEventObservation(
                event_id=event.event_id,
                actor_id=event.actor_id,
                day=event.day,
                ordinal=index,
                event_type=event.event_type.value,
                target_id=event.target_id,
                role=event.role,
                result_is_werewolf=event.result_is_werewolf,
                referenced_day=event.referenced_day,
            )
            for index, event in enumerate(state.speech_events)
        )

        public_semantic_events = tuple(
            _semantic_observation(event)
            for event in all_semantic_events
            if event.action.channel is Channel.PUBLIC
        )
        private_semantic_events_by_role = {
            role: tuple(
                _semantic_observation(event)
                for event in all_semantic_events
                if _private_event_visible(role, event.action.channel)
            )
            for role in {viewer.role for viewer in viewers.values()}
        }

        votes = tuple(
            VoteObservation(
                voter_id=vote.voter_id,
                target_id=vote.target_id,
                day=vote.day,
                round=vote.round,
            )
            for vote in state.vote_records
        )
        dawns = _dawn_history(controller)
        werewolf_ids = tuple(
            player.player_id
            for player in state.players.values()
            if player.role is RoleName.WEREWOLF
        )
        freemason_ids = tuple(
            player.player_id
            for player in state.players.values()
            if player.role is RoleName.FREEMASON
        )
        wolf_attack_history = tuple(
            PrivateTargetObservation(record.target_id, record.day)
            for record in state.attack_records
        )

        result: dict[str, PolicyObservation] = {}
        for viewer_id, viewer in viewers.items():
            private = PrivateObservation(
                role=viewer.role,
                allies=_allies_from_role_members(
                    viewer.role,
                    viewer_id,
                    werewolf_ids=werewolf_ids,
                    freemason_ids=freemason_ids,
                ),
                is_alpha_wolf=(
                    viewer.role is RoleName.WEREWOLF
                    and viewer_id == controller.alpha_wolf_id
                ),
                divine_results=tuple(
                    PrivateResultObservation(record.target_id, record.day, record.is_werewolf)
                    for record in state.divine_records
                    if record.seer_id == viewer_id
                ),
                medium_results=tuple(
                    PrivateResultObservation(record.target_id, record.day, record.is_werewolf)
                    for record in state.medium_records
                    if record.medium_id == viewer_id
                ),
                guard_history=tuple(
                    PrivateTargetObservation(record.target_id, record.day)
                    for record in state.guard_records
                    if viewer.role is RoleName.HUNTER and record.hunter_id == viewer_id
                ),
                attack_history=(
                    wolf_attack_history if viewer.role is RoleName.WEREWOLF else ()
                ),
                semantic_events=private_semantic_events_by_role[viewer.role],
            )
            result[viewer_id] = PolicyObservation(
                viewer_id=viewer_id,
                day=state.day,
                phase=state.phase.value,
                discussion_tick=discussion_tick,
                vote_round=state.vote_round,
                first_victim_id=state.first_victim_id,
                players=players,
                claim_events=claim_events,
                semantic_events=public_semantic_events,
                votes=votes,
                dawns=dawns,
                private=private,
            )
        return result


def _semantic_observation(event: TimedSemanticEvent) -> SemanticEventObservation:
    action = event.action
    return SemanticEventObservation(
        event_id=event.event_id,
        actor_id=event.actor_id,
        day=event.day,
        discussion_tick=event.discussion_tick,
        channel=action.channel.value,
        action_type=action.action_type.value,
        topic=action.topic.value if action.topic is not None else None,
        target_id=action.target_id,
        secondary_target_id=action.secondary_target_id,
        role=action.role,
        result=action.result.value if action.result is not None else None,
        quantity=action.quantity,
        referenced_day=action.referenced_day,
        scope=action.scope.value if action.scope is not None else None,
        stance=action.stance.value if action.stance is not None else None,
    )


def _private_event_visible(role: RoleName, channel: Channel) -> bool:
    if channel is Channel.WOLF:
        return role is RoleName.WEREWOLF
    if channel is Channel.FREEMASON:
        return role is RoleName.FREEMASON
    return False


def _allies_from_role_members(
    role: RoleName,
    viewer_id: str,
    *,
    werewolf_ids: tuple[str, ...],
    freemason_ids: tuple[str, ...],
) -> tuple[str, ...]:
    if role is RoleName.WEREWOLF:
        return tuple(player_id for player_id in werewolf_ids if player_id != viewer_id)
    if role is RoleName.FREEMASON:
        return tuple(player_id for player_id in freemason_ids if player_id != viewer_id)
    return ()


def _allies(controller: GameController, viewer_id: str) -> tuple[str, ...]:
    state = controller.state
    viewer = state.players[viewer_id]
    werewolf_ids = tuple(
        player.player_id
        for player in state.players.values()
        if player.role is RoleName.WEREWOLF
    )
    freemason_ids = tuple(
        player.player_id
        for player in state.players.values()
        if player.role is RoleName.FREEMASON
    )
    return _allies_from_role_members(
        viewer.role,
        viewer_id,
        werewolf_ids=werewolf_ids,
        freemason_ids=freemason_ids,
    )


def _public_death_kind(cause: DeathCause | None) -> str | None:
    if cause is None:
        return None
    if cause is DeathCause.EXECUTED:
        return "executed"
    if cause is DeathCause.FIRST_VICTIM:
        return "first_victim"
    return "night_death"


def _dawn_history(controller: GameController) -> tuple[DawnObservation, ...]:
    """Expose who died overnight, never why an attack failed or a fox died."""
    state = controller.state
    resolved_regular_nights = sorted({record.day for record in state.attack_records})
    dawns: list[DawnObservation] = []
    for night_day in resolved_regular_nights:
        dead = tuple(
            record.player_id
            for record in state.death_records
            if record.day == night_day
            and record.cause in (DeathCause.ATTACKED, DeathCause.CURSED)
        )
        dawns.append(
            DawnObservation(
                night_day=night_day,
                dead_player_ids=dead,
                no_death=not dead,
            )
        )
    return tuple(dawns)
