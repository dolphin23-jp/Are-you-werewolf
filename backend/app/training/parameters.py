"""Action-conditioned semantic parameter masks.

The masks are built from a policy observation, never from the true world. They
rule out malformed speech while deliberately preserving deception, unusual
claims, self-sacrifice proposals, late COs, and contradictory public stories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import local

from app.engine.roles import RoleName
from app.training.actions import (
    ActionType,
    ResultValue,
    Scope,
    Stance,
    TimingBucket,
    Topic,
)
from app.training.observation import PolicyObservation, SemanticEventObservation


@dataclass(frozen=True)
class SemanticParameterMask:
    topics: tuple[Topic, ...] = ()
    target_ids: tuple[str, ...] = ()
    roles: tuple[RoleName, ...] = ()
    results: tuple[ResultValue, ...] = ()
    quantities: tuple[int, ...] = ()
    referenced_days: tuple[int, ...] = ()
    scopes: tuple[Scope, ...] = ()
    stances: tuple[Stance, ...] = ()
    reference_event_ids: tuple[str, ...] = ()
    timings: tuple[TimingBucket, ...] = (
        TimingBucket.IMMEDIATE,
        TimingBucket.EARLY,
        TimingBucket.NORMAL,
        TimingBucket.LATE,
        TimingBucket.HOLD,
    )


@dataclass
class _SemanticMaskCacheState:
    """Bounded per-thread cache for repeated queries on one immutable observation."""

    observation: PolicyObservation | None = None
    values: dict[tuple[ActionType, Topic | None], SemanticParameterMask] = field(
        default_factory=dict
    )


class _SemanticMaskCache(local):
    def __init__(self) -> None:
        self.state = _SemanticMaskCacheState()


_SEMANTIC_MASK_CACHE = _SemanticMaskCache()
_TOPIC_INSENSITIVE_ACTIONS = frozenset(
    {
        ActionType.PASS,
        ActionType.EVALUATE,
        ActionType.DECLARE,
        ActionType.QUESTION,
        ActionType.REACT,
        ActionType.RETRACT,
    }
)


def _cache_topic(action_type: ActionType, topic: Topic | None) -> Topic | None:
    if action_type in _TOPIC_INSENSITIVE_ACTIONS:
        return None
    if action_type is ActionType.PROPOSE and topic is not Topic.EXECUTION:
        return None
    return topic


def semantic_parameter_mask(
    observation: PolicyObservation,
    action_type: ActionType,
    *,
    topic: Topic | None = None,
) -> SemanticParameterMask:
    """Return legal semantic parameters using only information in ``observation``.

    ``PolicyObservation`` is frozen, and speech sampling asks for the same masks
    repeatedly while materializing one decision. Cache only the current
    observation per thread so duplicate legality queries avoid rebuilding the
    same player/event tuples without retaining rollout history.
    """

    state = _SEMANTIC_MASK_CACHE.state
    if state.observation is not observation:
        state.observation = observation
        state.values.clear()

    normalized_topic = _cache_topic(action_type, topic)
    key = (action_type, normalized_topic)
    cached = state.values.get(key)
    if cached is not None:
        return cached

    mask = _build_semantic_parameter_mask(
        observation,
        action_type,
        topic=normalized_topic,
    )
    state.values[key] = mask
    return mask


def _build_semantic_parameter_mask(
    observation: PolicyObservation,
    action_type: ActionType,
    *,
    topic: Topic | None = None,
) -> SemanticParameterMask:
    alive = tuple(player.player_id for player in observation.players if player.alive)
    others_alive = tuple(player_id for player_id in alive if player_id != observation.viewer_id)
    all_players = tuple(player.player_id for player in observation.players)
    other_players = tuple(
        player_id for player_id in all_players if player_id != observation.viewer_id
    )
    past_days = tuple(range(max(0, observation.day)))

    if action_type is ActionType.PASS:
        return SemanticParameterMask()

    if action_type is ActionType.CLAIM:
        claim_topics = (Topic.ROLE, Topic.PARTNER, Topic.WOLF_COUNT)
        if topic is Topic.ROLE:
            return SemanticParameterMask(topics=claim_topics, roles=tuple(RoleName))
        if topic is Topic.PARTNER:
            return SemanticParameterMask(topics=claim_topics, target_ids=other_players)
        if topic is Topic.WOLF_COUNT:
            return SemanticParameterMask(topics=claim_topics, quantities=(1, 2, 3))
        return SemanticParameterMask(topics=claim_topics)

    if action_type is ActionType.REPORT:
        seer_targets = _seer_claim_targets(observation, past_days)
        medium_targets = _executed_player_ids(observation)
        report_topics = tuple(
            candidate
            for candidate, available in (
                (Topic.SEER_RESULT, bool(seer_targets and past_days)),
                (Topic.MEDIUM_RESULT, bool(medium_targets and past_days)),
                (Topic.GUARD, bool(all_players and past_days)),
                (Topic.ATTACK, bool(all_players and past_days)),
            )
            if available
        )
        if topic is Topic.SEER_RESULT and topic in report_topics:
            return SemanticParameterMask(
                topics=report_topics,
                target_ids=seer_targets,
                results=(ResultValue.WHITE, ResultValue.BLACK),
                referenced_days=past_days,
            )
        if topic is Topic.MEDIUM_RESULT and topic in report_topics:
            return SemanticParameterMask(
                topics=report_topics,
                target_ids=medium_targets,
                results=(ResultValue.WHITE, ResultValue.BLACK),
                referenced_days=past_days,
            )
        if topic in (Topic.GUARD, Topic.ATTACK) and topic in report_topics:
            return SemanticParameterMask(
                topics=report_topics,
                target_ids=all_players,
                referenced_days=past_days,
            )
        return SemanticParameterMask(topics=report_topics)

    if action_type is ActionType.EVALUATE:
        return SemanticParameterMask(
            topics=(
                Topic.WOLF,
                Topic.FOX,
                Topic.MADMAN,
                Topic.SEER_AUTHENTICITY,
                Topic.MEDIUM_AUTHENTICITY,
                Topic.HUNTER_AUTHENTICITY,
                Topic.FREEMASON_AUTHENTICITY,
            ),
            target_ids=other_players,
            stances=(Stance.TRUST, Stance.SUSPECT, Stance.NEUTRAL),
        )

    if action_type is ActionType.DECLARE:
        return SemanticParameterMask(
            topics=(Topic.VOTE, Topic.DIVINE, Topic.GUARD, Topic.ATTACK),
            target_ids=others_alive,
        )

    if action_type is ActionType.PROPOSE:
        scopes = (
            Scope.NONE,
            Scope.SELF,
            Scope.ALL,
            Scope.ALIVE,
            Scope.UNCLAIMED,
            Scope.SEER_CLAIMANTS,
            Scope.MEDIUM_CLAIMANTS,
            Scope.HUNTER_CLAIMANTS,
            Scope.FREEMASON_CLAIMANTS,
            Scope.FREE_CHOICE,
        )
        if topic is Topic.EXECUTION:
            # Self is intentionally legal: this is the primitive behind a pillar
            # or wolf-pillar strategy, without teaching either doctrine.
            return SemanticParameterMask(
                topics=(Topic.EXECUTION, Topic.DIVINE, Topic.GUARD, Topic.CO_REQUEST),
                target_ids=alive,
                scopes=scopes,
            )
        return SemanticParameterMask(
            topics=(Topic.EXECUTION, Topic.DIVINE, Topic.GUARD, Topic.CO_REQUEST),
            target_ids=others_alive,
            scopes=scopes,
        )

    if action_type is ActionType.QUESTION:
        return SemanticParameterMask(
            topics=(
                Topic.GENERAL,
                Topic.ROLE,
                Topic.CO_INTENTION,
                Topic.SEER_RESULT,
                Topic.MEDIUM_RESULT,
                Topic.VOTE_REASON,
                Topic.WOLF,
                Topic.FOX,
            ),
            target_ids=others_alive,
        )

    if action_type is ActionType.REACT:
        return SemanticParameterMask(
            stances=(Stance.SUPPORT, Stance.OPPOSE, Stance.NEUTRAL),
            reference_event_ids=tuple(event.event_id for event in observation.semantic_events),
        )

    if action_type is ActionType.RETRACT:
        own_events = _own_retractable_events(observation)
        return SemanticParameterMask(
            reference_event_ids=tuple(event.event_id for event in own_events)
        )

    if action_type is ActionType.CORRECT:
        correction_topics = tuple(
            candidate
            for candidate in (Topic.SEER_RESULT, Topic.MEDIUM_RESULT)
            if past_days and _own_reports_for_topic(observation, candidate)
        )
        if topic not in correction_topics:
            return SemanticParameterMask(topics=correction_topics)
        own_reports = _own_reports_for_topic(observation, topic)
        return SemanticParameterMask(
            topics=correction_topics,
            target_ids=all_players,
            results=(ResultValue.WHITE, ResultValue.BLACK),
            referenced_days=past_days,
            reference_event_ids=tuple(event.event_id for event in own_reports),
        )

    return SemanticParameterMask()


def _executed_player_ids(observation: PolicyObservation) -> tuple[str, ...]:
    return tuple(
        player.player_id
        for player in observation.players
        if player.death_kind == "executed"
    )


def _seer_claim_targets(
    observation: PolicyObservation, past_days: tuple[int, ...]
) -> tuple[str, ...]:
    if not past_days:
        return ()
    # A fake report can lie about the verdict or whether a divination occurred,
    # but its target should at least be a seat that could have been alive on one
    # of the claimed past nights. True role information is never consulted.
    return tuple(
        player.player_id
        for player in observation.players
        if player.player_id != observation.viewer_id
        and player.player_id != observation.first_victim_id
        and (player.death_day is None or player.death_day > 0)
    )


def _own_reports_for_topic(
    observation: PolicyObservation,
    topic: Topic,
) -> tuple[SemanticEventObservation, ...]:
    return tuple(
        event
        for event in observation.semantic_events
        if event.actor_id == observation.viewer_id
        and event.action_type == ActionType.REPORT.value
        and event.topic == topic.value
    )


def _own_retractable_events(
    observation: PolicyObservation,
) -> tuple[SemanticEventObservation, ...]:
    retractable = {ActionType.CLAIM.value, ActionType.REPORT.value, ActionType.DECLARE.value}
    return tuple(
        event
        for event in observation.semantic_events
        if event.actor_id == observation.viewer_id and event.action_type in retractable
    )
