"""Structured public-speech events: the single record of what was publicly claimed.

A flat `list[CoDeclaration]` cannot answer the questions a real game raises.
Who said this, and in which message? Was the seer claim retracted, or slid to
freemason? Is this verdict the original one or the correction? Was the "CO" a
claim at all, or a guess a matcher made about an ambiguous human sentence?

So claims are recorded as append-only events carrying their source message, and
everything else -- the current CO, the CO history, the live verdicts and their
superseded versions -- is *derived* from that log. There is one write path and
one source of truth; `co_declarations` and `public_result_claims` survive only
as compatibility views over these derivations.

Confidence is what keeps a hedged human sentence from becoming a binding claim.
An event below `CLAIM_CONFIDENCE_THRESHOLD` is still recorded -- it is evidence
that something claim-shaped was said -- but no derivation will promote it into
the board's CO composition.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from app.engine.roles import RoleName

# A matcher's guess about free text must clear this before it counts as a claim.
CLAIM_CONFIDENCE_THRESHOLD = 0.8

SEER_RESULT = "seer"
MEDIUM_RESULT = "medium"
RESULT_TYPES = (SEER_RESULT, MEDIUM_RESULT)

_RESULT_ROLES: dict[RoleName, str] = {
    RoleName.SEER: SEER_RESULT,
    RoleName.MEDIUM: MEDIUM_RESULT,
}
_RESULT_ROLE_BY_TYPE: dict[str, RoleName] = {value: key for key, value in _RESULT_ROLES.items()}


class SpeechEventType(StrEnum):
    ROLE_CLAIM = "role_claim"
    ROLE_RETRACTION = "role_retraction"
    ROLE_SWITCH = "role_switch"
    ABILITY_RESULT = "ability_result"
    RESULT_RETRACTION = "result_retraction"
    RESULT_CORRECTION = "result_correction"
    PARTNER_CLAIM = "partner_claim"
    FACT_CORRECTION = "fact_correction"
    ACCUSATION = "accusation"
    DEFENSE = "defense"
    VOTE_INTENTION = "vote_intention"
    QUESTION = "question"
    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"


ROLE_EVENT_TYPES = (
    SpeechEventType.ROLE_CLAIM,
    SpeechEventType.ROLE_SWITCH,
    SpeechEventType.ROLE_RETRACTION,
)
RESULT_EVENT_TYPES = (
    SpeechEventType.ABILITY_RESULT,
    SpeechEventType.RESULT_CORRECTION,
    SpeechEventType.RESULT_RETRACTION,
)


@dataclass(frozen=True)
class SpeechEvent:
    event_id: str
    source_message_id: str
    actor_id: str
    event_type: SpeechEventType
    day: int
    target_id: str | None = None
    role: RoleName | None = None
    result_is_werewolf: bool | None = None
    referenced_day: int | None = None
    confidence: float = 1.0

    @property
    def is_binding(self) -> bool:
        """Whether this event may change the board's public claim state."""
        return self.confidence >= CLAIM_CONFIDENCE_THRESHOLD

    @property
    def result_type(self) -> str | None:
        """Seer and medium results are the same event type but answer different
        questions, so the ability used is part of every result's identity."""
        return _RESULT_ROLES.get(self.role) if self.role is not None else None


def result_role(result_type: str) -> RoleName:
    return _RESULT_ROLE_BY_TYPE[result_type]


# -- role claims --


class RoleClaimStatus(StrEnum):
    NONE = "none"
    CLAIMED = "claimed"
    RETRACTED = "retracted"
    SWITCHED = "switched"


@dataclass(frozen=True)
class RoleClaimState:
    """One transition in a player's public claim history."""

    player_id: str
    role: RoleName | None
    status: RoleClaimStatus
    day: int
    source_message_id: str
    event_id: str
    previous_role: RoleName | None = None
    reaffirmed: bool = False

    @property
    def is_active(self) -> bool:
        return self.role is not None and self.status in (
            RoleClaimStatus.CLAIMED,
            RoleClaimStatus.SWITCHED,
        )


def role_claim_history(
    events: Sequence[SpeechEvent], player_id: str | None = None
) -> tuple[RoleClaimState, ...]:
    """Every binding claim transition, in the order it was spoken."""
    current: dict[str, RoleClaimState] = {}
    # Survives a retraction: "used to claim seer" stays explanatory even after
    # the seer claim is gone.
    last_role: dict[str, RoleName | None] = {}
    history: list[RoleClaimState] = []
    for event in events:
        if event.event_type not in ROLE_EVENT_TYPES or not event.is_binding:
            continue
        actor = event.actor_id
        previous_role = last_role.get(actor)
        active_role = current[actor].role if actor in current else None
        if event.event_type is SpeechEventType.ROLE_RETRACTION:
            status, role, reaffirmed = RoleClaimStatus.RETRACTED, None, False
        elif event.event_type is SpeechEventType.ROLE_SWITCH or (
            active_role is not None and event.role != active_role
        ):
            status, role, reaffirmed = RoleClaimStatus.SWITCHED, event.role, False
        else:
            status, role = RoleClaimStatus.CLAIMED, event.role
            reaffirmed = active_role is not None and active_role == event.role
        state = RoleClaimState(
            player_id=actor,
            role=role,
            status=status,
            day=event.day,
            source_message_id=event.source_message_id,
            event_id=event.event_id,
            previous_role=previous_role,
            reaffirmed=reaffirmed,
        )
        current[actor] = state
        if role is not None:
            last_role[actor] = role
        history.append(state)
    if player_id is not None:
        return tuple(state for state in history if state.player_id == player_id)
    return tuple(history)


def current_role_claim(
    events: Sequence[SpeechEvent], player_id: str
) -> RoleClaimState | None:
    """The claim standing right now, or None once retracted or never made."""
    history = role_claim_history(events, player_id)
    if not history:
        return None
    latest = history[-1]
    return latest if latest.is_active else None


def current_role_claims(events: Sequence[SpeechEvent]) -> tuple[RoleClaimState, ...]:
    """Active claims only, ordered by when the standing claim was made -- a
    player who slid moves to the position of their new claim, not their old one."""
    latest: dict[str, RoleClaimState] = {}
    for state in role_claim_history(events):
        latest[state.player_id] = state
    return tuple(state for state in latest.values() if state.is_active)


# -- ability results --


class ResultStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"


@dataclass(frozen=True)
class ResultVersion:
    claimant_id: str
    result_type: str
    target_id: str
    is_werewolf: bool
    day: int
    version: int
    status: ResultStatus
    source_message_id: str
    event_id: str
    corrected: bool = False
    referenced_day: int | None = None

    @property
    def is_active(self) -> bool:
        return self.status is ResultStatus.ACTIVE


def result_versions(
    events: Sequence[SpeechEvent],
    *,
    claimant_id: str | None = None,
    result_type: str | None = None,
    target_id: str | None = None,
) -> tuple[ResultVersion, ...]:
    """Every version of every published verdict, newest last within each subject.

    Versioning is what makes a correction legible: the original verdict stays in
    the record as SUPERSEDED rather than being overwritten, so "what did they
    say before they corrected it" remains answerable.
    """
    groups: dict[tuple[str, str, str], list[ResultVersion]] = {}
    for event in events:
        if event.event_type not in RESULT_EVENT_TYPES or not event.is_binding:
            continue
        event_result_type = event.result_type
        if event_result_type is None or event.target_id is None:
            continue
        key = (event.actor_id, event_result_type, event.target_id)
        versions = groups.setdefault(key, [])
        superseded = ResultStatus.RETRACTED
        if event.event_type is not SpeechEventType.RESULT_RETRACTION:
            superseded = ResultStatus.SUPERSEDED
        groups[key] = [
            replace(version, status=superseded) if version.is_active else version
            for version in versions
        ]
        if event.event_type is SpeechEventType.RESULT_RETRACTION:
            continue
        groups[key].append(
            ResultVersion(
                claimant_id=event.actor_id,
                result_type=event_result_type,
                target_id=event.target_id,
                is_werewolf=bool(event.result_is_werewolf),
                day=event.day,
                version=len(groups[key]) + 1,
                status=ResultStatus.ACTIVE,
                source_message_id=event.source_message_id,
                event_id=event.event_id,
                corrected=event.event_type is SpeechEventType.RESULT_CORRECTION,
                referenced_day=event.referenced_day,
            )
        )
    return tuple(
        version
        for key, versions in groups.items()
        for version in versions
        if (claimant_id is None or key[0] == claimant_id)
        and (result_type is None or key[1] == result_type)
        and (target_id is None or key[2] == target_id)
    )


def active_results(events: Sequence[SpeechEvent]) -> tuple[ResultVersion, ...]:
    return tuple(version for version in result_versions(events) if version.is_active)


def active_result(
    events: Sequence[SpeechEvent], claimant_id: str, result_type: str, target_id: str
) -> ResultVersion | None:
    return next(
        (
            version
            for version in result_versions(
                events,
                claimant_id=claimant_id,
                result_type=result_type,
                target_id=target_id,
            )
            if version.is_active
        ),
        None,
    )


# -- freemason partner claims --


@dataclass(frozen=True)
class PartnerClaimState:
    claimant_id: str
    partner_id: str
    day: int
    confirmed: bool
    source_message_id: str
    event_id: str


def partner_claims(events: Sequence[SpeechEvent]) -> tuple[PartnerClaimState, ...]:
    """One record per pair. A pair is confirmed once both sides have named each
    other, which is the only public evidence a shared-role pair can offer."""
    claimed: set[tuple[str, str]] = set()
    records: list[PartnerClaimState] = []
    for event in events:
        if event.event_type is not SpeechEventType.PARTNER_CLAIM or not event.is_binding:
            continue
        if event.target_id is None:
            continue
        pair = (event.actor_id, event.target_id)
        reverse = (event.target_id, event.actor_id)
        if pair in claimed:
            continue
        claimed.add(pair)
        if reverse in claimed:
            records = [
                replace(record, confirmed=True)
                if (record.claimant_id, record.partner_id) == reverse
                else record
                for record in records
            ]
            continue
        records.append(
            PartnerClaimState(
                claimant_id=event.actor_id,
                partner_id=event.target_id,
                day=event.day,
                confirmed=False,
                source_message_id=event.source_message_id,
                event_id=event.event_id,
            )
        )
    return tuple(records)


# -- message-level lookups --


def events_for_message(
    events: Sequence[SpeechEvent], message_id: str
) -> tuple[SpeechEvent, ...]:
    """A single sentence can carry a CO and two verdicts; all of them point back
    to the same message."""
    return tuple(event for event in events if event.source_message_id == message_id)


def unpromoted_events(events: Iterable[SpeechEvent]) -> tuple[SpeechEvent, ...]:
    """Claim-shaped events too uncertain to bind. Kept for auditing why an
    ambiguous sentence did not become a CO."""
    return tuple(event for event in events if not event.is_binding)
