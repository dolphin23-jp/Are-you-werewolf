"""Deterministic fixed-shape encoding for policy observations.

This module is deliberately ML-framework agnostic. It converts an
information-safe :class:`PolicyObservation` into integer feature arrays that
can later be wrapped by PyTorch/JAX/NumPy without coupling the production game
engine to a learning library.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.engine.roles import RoleName
from app.training.actions import ActionType, Channel, ResultValue, Scope, Stance, Topic
from app.training.observation import PolicyObservation, SemanticEventObservation

MAX_SEATS = 17
MAX_SEMANTIC_EVENTS = 128
MAX_VOTE_EVENTS = 128
MAX_DAWN_EVENTS = 16


class PhaseCode(StrEnum):
    LOBBY = "lobby"
    NIGHT = "night"
    DAWN = "dawn"
    DISCUSSION = "discussion"
    VOTING = "voting"
    RUNOFF = "runoff"
    VOTE_RESULT = "vote_result"
    GAME_OVER = "game_over"


_ROLE_INDEX = {value.value: index + 1 for index, value in enumerate(RoleName)}
_PHASE_INDEX = {value.value: index + 1 for index, value in enumerate(PhaseCode)}
_ACTION_INDEX = {value.value: index + 1 for index, value in enumerate(ActionType)}
_TOPIC_INDEX = {value.value: index + 1 for index, value in enumerate(Topic)}
_CHANNEL_INDEX = {value.value: index + 1 for index, value in enumerate(Channel)}
_RESULT_INDEX = {value.value: index + 1 for index, value in enumerate(ResultValue)}
_SCOPE_INDEX = {value.value: index + 1 for index, value in enumerate(Scope)}
_STANCE_INDEX = {value.value: index + 1 for index, value in enumerate(Stance)}
_DEATH_INDEX = {"executed": 1, "first_victim": 2, "night_death": 3}


@dataclass(frozen=True)
class EncodedPolicyObservation:
    """Fixed-shape integer features plus explicit validity masks.

    Padding is always zero. Real categorical values start at one, which keeps
    zero available for both padding and "not present" fields.
    """

    global_features: tuple[int, ...]
    player_tokens: tuple[tuple[int, ...], ...]
    semantic_tokens: tuple[tuple[int, ...], ...]
    semantic_mask: tuple[int, ...]
    vote_tokens: tuple[tuple[int, ...], ...]
    vote_mask: tuple[int, ...]
    dawn_tokens: tuple[tuple[int, ...], ...]
    dawn_mask: tuple[int, ...]


class ObservationEncoder:
    def encode(self, observation: PolicyObservation) -> EncodedPolicyObservation:
        seat_ids = tuple(player.player_id for player in observation.players)
        if len(seat_ids) != MAX_SEATS:
            raise ValueError(f"expected {MAX_SEATS} seats, got {len(seat_ids)}")
        seat_index = {player_id: index + 1 for index, player_id in enumerate(seat_ids)}

        global_features = (
            observation.day,
            _PHASE_INDEX.get(observation.phase, 0),
            observation.discussion_tick,
            observation.vote_round,
            seat_index[observation.viewer_id],
            seat_index.get(observation.first_victim_id or "", 0),
            _ROLE_INDEX[observation.private.role.value],
            int(observation.private.is_alpha_wolf),
        )

        allies = set(observation.private.allies)
        divine = {result.target_id: result for result in observation.private.divine_results}
        medium = {result.target_id: result for result in observation.private.medium_results}
        guard_counts = _target_counts(observation.private.guard_history)
        attack_counts = _target_counts(observation.private.attack_history)

        player_tokens = tuple(
            (
                seat_index[player.player_id],
                int(player.player_id == observation.viewer_id),
                int(player.alive),
                (player.death_day + 1) if player.death_day is not None else 0,
                _DEATH_INDEX.get(player.death_kind or "", 0),
                _ROLE_INDEX.get(player.current_claim.value, 0)
                if player.current_claim is not None
                else 0,
                int(player.player_id in allies),
                _private_result_code(divine.get(player.player_id)),
                _private_result_code(medium.get(player.player_id)),
                guard_counts.get(player.player_id, 0),
                attack_counts.get(player.player_id, 0),
            )
            for player in observation.players
        )

        semantic_tokens, semantic_mask = _encode_semantics(
            observation.semantic_events,
            seat_index,
        )
        vote_tokens, vote_mask = _encode_votes(observation, seat_index)
        dawn_tokens, dawn_mask = _encode_dawns(observation, seat_index)

        return EncodedPolicyObservation(
            global_features=global_features,
            player_tokens=player_tokens,
            semantic_tokens=semantic_tokens,
            semantic_mask=semantic_mask,
            vote_tokens=vote_tokens,
            vote_mask=vote_mask,
            dawn_tokens=dawn_tokens,
            dawn_mask=dawn_mask,
        )


def _target_counts(records: tuple[object, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        target_id = getattr(record, "target_id")
        counts[target_id] = counts.get(target_id, 0) + 1
    return counts


def _private_result_code(result: object | None) -> int:
    if result is None:
        return 0
    return 2 if bool(getattr(result, "is_werewolf")) else 1


def _encode_semantics(
    events: tuple[SemanticEventObservation, ...],
    seat_index: dict[str, int],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    selected = events[-MAX_SEMANTIC_EVENTS:]
    tokens = [
        (
            event.day,
            event.discussion_tick + 1,
            seat_index.get(event.actor_id, 0),
            _CHANNEL_INDEX.get(event.channel, 0),
            _ACTION_INDEX.get(event.action_type, 0),
            _TOPIC_INDEX.get(event.topic or "", 0),
            seat_index.get(event.target_id or "", 0),
            seat_index.get(event.secondary_target_id or "", 0),
            _ROLE_INDEX.get(event.role.value, 0) if event.role is not None else 0,
            _RESULT_INDEX.get(event.result or "", 0),
            (event.quantity + 1) if event.quantity is not None else 0,
            (event.referenced_day + 1) if event.referenced_day is not None else 0,
            _SCOPE_INDEX.get(event.scope or "", 0),
            _STANCE_INDEX.get(event.stance or "", 0),
        )
        for event in selected
    ]
    return _pad(tokens, MAX_SEMANTIC_EVENTS, width=14)


def _encode_votes(
    observation: PolicyObservation,
    seat_index: dict[str, int],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    selected = observation.votes[-MAX_VOTE_EVENTS:]
    tokens = [
        (
            vote.day,
            vote.round,
            seat_index.get(vote.voter_id, 0),
            seat_index.get(vote.target_id, 0),
        )
        for vote in selected
    ]
    return _pad(tokens, MAX_VOTE_EVENTS, width=4)


def _encode_dawns(
    observation: PolicyObservation,
    seat_index: dict[str, int],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    selected = observation.dawns[-MAX_DAWN_EVENTS:]
    tokens = []
    for dawn in selected:
        dead = tuple(seat_index.get(player_id, 0) for player_id in dawn.dead_player_ids[:2])
        dead = dead + (0,) * (2 - len(dead))
        tokens.append((dawn.night_day, int(dawn.no_death), dead[0], dead[1]))
    return _pad(tokens, MAX_DAWN_EVENTS, width=4)


def _pad(
    tokens: list[tuple[int, ...]],
    length: int,
    *,
    width: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    if any(len(token) != width for token in tokens):
        raise ValueError("encoded token width mismatch")
    mask = [1] * len(tokens)
    while len(tokens) < length:
        tokens.append((0,) * width)
        mask.append(0)
    return tuple(tokens), tuple(mask)
