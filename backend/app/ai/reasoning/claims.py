"""Turning what a player published into structured speech events.

Two sources, deliberately unequal:

* **An AI's structured output is a declaration.** When the model fills in
  `public_claim_role` or `public_results`, that is the claim -- registering it
  must not depend on a regex also finding it in the prose. Instead, the prose is
  brought into line: `ensure_fact_sentences` writes the canonical fact sentence
  into the message when it is missing, so the visible text and the ledger agree.
* **Free text is a matcher's reading.** It fills the gaps the structured output
  left empty (models do omit the field after writing an unambiguous CO) and
  carries the human path entirely. A quoted or hedged sentence still produces an
  event, but at a confidence too low to be promoted into a binding claim.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.ai.co_detection import (
    AMBIGUOUS_CLAIM_CONFIDENCE,
    SPOKEN_CLAIM_CONFIDENCE,
    detect_claimed_role_with_confidence,
    detect_freemason_partner,
)
from app.ai.public_speech import detect_public_results
from app.ai.reasoning.facts import MEDIUM_RESULT, SEER_RESULT, PublicFactLedger, mentions_player
from app.ai.schemas import DiscussionOutput
from app.engine.game import GameError
from app.engine.roles import RoleName
from app.engine.speech_events import SpeechEvent, SpeechEventType, result_role

logger = logging.getLogger(__name__)

# A field the model filled in is a declaration, not an inference about text.
DECLARED_CONFIDENCE = 1.0

_CLAIM_LABELS: dict[RoleName, str] = {
    RoleName.SEER: "占い",
    RoleName.MEDIUM: "霊媒",
    RoleName.HUNTER: "狩人",
    RoleName.FREEMASON: "共有",
}
_RESULT_ROLE_BY_TYPE: dict[str, RoleName] = {
    SEER_RESULT: RoleName.SEER,
    MEDIUM_RESULT: RoleName.MEDIUM,
}
# A retraction and a slide belong to the same slot as the claim they replace:
# a message that both retracts a CO and reads as one in prose must not produce
# two competing role events.
_ROLE_CATEGORY = (
    SpeechEventType.ROLE_CLAIM,
    SpeechEventType.ROLE_RETRACTION,
    SpeechEventType.ROLE_SWITCH,
)
_RESULT_CATEGORY = (
    SpeechEventType.ABILITY_RESULT,
    SpeechEventType.RESULT_RETRACTION,
    SpeechEventType.RESULT_CORRECTION,
)


@dataclass(frozen=True)
class SpeechEventDraft:
    """A speech event that has not been given an id or a source message yet."""

    event_type: SpeechEventType
    target_id: str | None = None
    role: RoleName | None = None
    result_is_werewolf: bool | None = None
    referenced_day: int | None = None
    confidence: float = DECLARED_CONFIDENCE
    # Canonical wording for this claim, used to keep the public message honest.
    fact_sentence: str = ""


def build_claim_drafts(
    output: DiscussionOutput,
    ledger: PublicFactLedger,
    *,
    speaker_id: str,
) -> list[SpeechEventDraft]:
    """Structured declarations first; free text only fills what they left out."""
    structured = _structured_drafts(output, ledger, speaker_id=speaker_id)
    claimed_categories = {_category(draft) for draft in structured}
    spoken = _spoken_drafts(output.public_message, ledger, speaker_id=speaker_id)
    return structured + [
        draft for draft in spoken if _category(draft) not in claimed_categories
    ]


def _category(draft: SpeechEventDraft) -> str:
    if draft.event_type in _ROLE_CATEGORY:
        return "role"
    if draft.event_type in _RESULT_CATEGORY:
        return "result"
    return "partner"


def _structured_drafts(
    output: DiscussionOutput, ledger: PublicFactLedger, *, speaker_id: str
) -> list[SpeechEventDraft]:
    # Changes to standing claims come first: a message that both withdraws the
    # old verdict and states the new one has to be read in that order, or the
    # correction lands before there is anything to correct.
    drafts = _change_drafts(output, ledger, speaker_id=speaker_id)
    role = _parse_role(output.public_claim_role)
    if role is not None:
        drafts.append(
            SpeechEventDraft(
                event_type=SpeechEventType.ROLE_CLAIM,
                role=role,
                fact_sentence=render_role_claim_sentence(role),
            )
        )
    for result in output.public_results:
        result_type = result.result_type if result.result_type in _RESULT_ROLE_BY_TYPE else None
        if result_type is None or not ledger.is_known(result.target_id):
            continue
        if result.target_id == speaker_id:
            continue
        drafts.append(
            SpeechEventDraft(
                event_type=SpeechEventType.ABILITY_RESULT,
                role=result_role(result_type),
                target_id=result.target_id,
                result_is_werewolf=result.is_werewolf,
                referenced_day=result.referenced_day,
                fact_sentence=render_result_sentence(
                    ledger, result_type, result.target_id, result.is_werewolf
                ),
            )
        )
    return drafts


def _change_drafts(
    output: DiscussionOutput, ledger: PublicFactLedger, *, speaker_id: str
) -> list[SpeechEventDraft]:
    """Retractions, slides and corrections, from declared fields only.

    Deliberately not inferred from prose here: withdrawing a CO is a move the
    table has to be able to point at, and a matcher's reading of "I might have
    been wrong" is not that. The human path gets a narrow free-text route in
    `_spoken_change_drafts`, gated on there being something to withdraw.
    """
    drafts: list[SpeechEventDraft] = []
    action = output.claim_action
    if action is not None:
        standing = ledger.claimed_role_of(speaker_id)
        if action.action == "retract" and standing is not None:
            drafts.append(
                SpeechEventDraft(
                    event_type=SpeechEventType.ROLE_RETRACTION,
                    role=standing,
                    fact_sentence=render_retraction_sentence(standing),
                )
            )
        elif action.action == "switch":
            new_role = _parse_role(action.role)
            if new_role is not None and new_role != standing:
                drafts.append(
                    SpeechEventDraft(
                        event_type=SpeechEventType.ROLE_SWITCH,
                        role=new_role,
                        fact_sentence=render_switch_sentence(standing, new_role),
                    )
                )
    for change in output.result_actions:
        if change.result_type not in _RESULT_ROLE_BY_TYPE:
            continue
        existing = ledger.find_result(speaker_id, change.result_type, change.target_id)
        if existing is None:
            # Nothing published to withdraw or amend. Recording the event anyway
            # would invent a history the table never heard.
            continue
        if change.action == "retract":
            drafts.append(
                SpeechEventDraft(
                    event_type=SpeechEventType.RESULT_RETRACTION,
                    role=result_role(change.result_type),
                    target_id=change.target_id,
                    result_is_werewolf=existing.is_werewolf,
                    referenced_day=change.referenced_day or existing.referenced_day,
                    fact_sentence=render_result_retraction_sentence(
                        ledger, change.result_type, change.target_id
                    ),
                )
            )
        elif change.action == "correct" and change.is_werewolf is not None:
            drafts.append(
                SpeechEventDraft(
                    event_type=SpeechEventType.RESULT_CORRECTION,
                    role=result_role(change.result_type),
                    target_id=change.target_id,
                    result_is_werewolf=change.is_werewolf,
                    referenced_day=change.referenced_day or existing.referenced_day,
                    fact_sentence=render_result_correction_sentence(
                        ledger, change.result_type, change.target_id, change.is_werewolf
                    ),
                )
            )
    return drafts


# "I withdraw my CO" is a specific sentence, not a mood. The pattern is narrow
# on purpose: a hedge ("maybe I was wrong about that") must not delete a claim
# the table is still reasoning from.
_ROLE_RETRACTION_RE = re.compile(
    r"(?:CO|ＣＯ|カミングアウト)(?:を|は)?(?:取り消|撤回|取り下げ)"
)
_RESULT_RETRACTION_RE = re.compile(
    r"(?:占い|霊媒)?結果(?:を|は)?(?:取り消|撤回|取り下げ)"
)
_CORRECTION_RE = re.compile(r"訂正")


def _names_target(
    message: str, draft: SpeechEventDraft, ledger: PublicFactLedger
) -> bool:
    if draft.target_id is None:
        return True
    return mentions_player(message, draft.target_id, ledger.name_of(draft.target_id))


def _spoken_change_drafts(
    text: str, ledger: PublicFactLedger, *, speaker_id: str
) -> list[SpeechEventDraft]:
    """The human path to withdrawing something already published.

    Gated on there actually being a standing claim: with nothing to withdraw the
    sentence is just talk, and recording a retraction of nothing would put an
    event in the log that never happened at the table.
    """
    standing = ledger.claimed_role_of(speaker_id)
    if standing is not None and _ROLE_RETRACTION_RE.search(text):
        return [
            SpeechEventDraft(
                event_type=SpeechEventType.ROLE_RETRACTION,
                role=standing,
                confidence=SPOKEN_CLAIM_CONFIDENCE,
            )
        ]
    if not _RESULT_RETRACTION_RE.search(text):
        return []
    live = [
        result for result in ledger.public_results() if result.claimant_id == speaker_id
    ]
    named = [
        result
        for result in live
        if mentions_player(text, result.target_id, ledger.name_of(result.target_id))
    ]
    # Only an unambiguous target: withdrawing "the result" when three are
    # standing is a question for the speaker, not a guess for the parser.
    subjects = named or (live if len(live) == 1 else [])
    return [
        SpeechEventDraft(
            event_type=SpeechEventType.RESULT_RETRACTION,
            role=result_role(result.result_type),
            target_id=result.target_id,
            result_is_werewolf=result.is_werewolf,
            referenced_day=result.referenced_day,
            confidence=SPOKEN_CLAIM_CONFIDENCE,
        )
        for result in subjects
    ]


def _spoken_drafts(
    text: str, ledger: PublicFactLedger, *, speaker_id: str
) -> list[SpeechEventDraft]:
    changes = _spoken_change_drafts(text, ledger, speaker_id=speaker_id)
    if changes:
        return changes
    others = {
        pid: ledger.name_of(pid) for pid in ledger.known_player_ids() if pid != speaker_id
    }
    role, confidence = detect_claimed_role_with_confidence(text, list(others.values()))
    drafts: list[SpeechEventDraft] = []
    if role is not None:
        drafts.append(
            SpeechEventDraft(
                event_type=SpeechEventType.ROLE_CLAIM, role=role, confidence=confidence
            )
        )
    # A result needs a role to belong to: the one just claimed, or the one this
    # player is already publicly committed to.
    binding_role = role if confidence >= SPOKEN_CLAIM_CONFIDENCE else None
    effective_role = binding_role or ledger.claimed_role_of(speaker_id)
    for detected in detect_public_results(
        text,
        effective_role,
        others,
        role_claimed_in_message=binding_role in (RoleName.SEER, RoleName.MEDIUM),
    ):
        drafts.append(
            SpeechEventDraft(
                event_type=SpeechEventType.ABILITY_RESULT,
                role=result_role(detected.result_type),
                target_id=detected.target_id,
                result_is_werewolf=detected.is_werewolf,
                confidence=SPOKEN_CLAIM_CONFIDENCE,
            )
        )
    if effective_role == RoleName.FREEMASON:
        partner_id = detect_freemason_partner(text, others)
        if partner_id is not None:
            drafts.append(
                SpeechEventDraft(
                    event_type=SpeechEventType.PARTNER_CLAIM,
                    target_id=partner_id,
                    confidence=(
                        confidence if role is not None else SPOKEN_CLAIM_CONFIDENCE
                    ),
                )
            )
    return drafts


def _parse_role(value: str | None) -> RoleName | None:
    if not value:
        return None
    try:
        return RoleName(value)
    except ValueError:
        return None


# -- canonical fact sentences --


def render_role_claim_sentence(role: RoleName) -> str:
    return f"{_CLAIM_LABELS.get(role, role.value)}CO。"


def render_result_sentence(
    ledger: PublicFactLedger, result_type: str, target_id: str, is_werewolf: bool
) -> str:
    ability = "霊媒" if result_type == MEDIUM_RESULT else "占い"
    verdict = "黒(人狼)" if is_werewolf else "白(人狼ではない)"
    return f"{ability}結果、{ledger.label_of(target_id)}は{verdict}です。"


def render_retraction_sentence(role: RoleName) -> str:
    return f"先ほどの{_CLAIM_LABELS.get(role, role.value)}COを撤回します。"


def render_switch_sentence(previous: RoleName | None, new_role: RoleName) -> str:
    label = _CLAIM_LABELS.get(new_role, new_role.value)
    if previous is None:
        return f"{label}COします。"
    previous_label = _CLAIM_LABELS.get(previous, previous.value)
    return f"{previous_label}COを取り下げ、{label}COに変更します。"


def render_result_retraction_sentence(
    ledger: PublicFactLedger, result_type: str, target_id: str
) -> str:
    ability = "霊媒" if result_type == MEDIUM_RESULT else "占い"
    return f"{ledger.label_of(target_id)}への{ability}結果を撤回します。"


def render_result_correction_sentence(
    ledger: PublicFactLedger, result_type: str, target_id: str, is_werewolf: bool
) -> str:
    ability = "霊媒" if result_type == MEDIUM_RESULT else "占い"
    verdict = "黒(人狼)" if is_werewolf else "白(人狼ではない)"
    return f"訂正します。{ledger.label_of(target_id)}への{ability}結果は{verdict}です。"


def ensure_fact_sentences(
    message: str, drafts: Sequence[SpeechEventDraft], ledger: PublicFactLedger, *, speaker_id: str
) -> str:
    """Prepend the canonical wording for any declared claim the prose omits.

    Without this a structured verdict either goes unpublished (silently losing
    the result) or is published while the visible message never says it, which
    leaves the table arguing about a claim nobody made.
    """
    missing = [
        draft.fact_sentence
        for draft in drafts
        if draft.fact_sentence and not _states(message, draft, ledger, speaker_id=speaker_id)
    ]
    if not missing:
        return message
    return "".join(missing) + message


def _states(
    message: str,
    draft: SpeechEventDraft,
    ledger: PublicFactLedger,
    *,
    speaker_id: str,
) -> bool:
    if draft.event_type is SpeechEventType.ROLE_RETRACTION:
        return bool(_ROLE_RETRACTION_RE.search(message))
    if draft.event_type is SpeechEventType.RESULT_RETRACTION:
        return bool(_RESULT_RETRACTION_RE.search(message)) and _names_target(
            message, draft, ledger
        )
    if draft.event_type is SpeechEventType.RESULT_CORRECTION:
        return bool(_CORRECTION_RE.search(message)) and _names_target(
            message, draft, ledger
        )
    if draft.event_type is SpeechEventType.ROLE_SWITCH:
        spoken, confidence = detect_claimed_role_with_confidence(message)
        return spoken == draft.role and confidence >= SPOKEN_CLAIM_CONFIDENCE
    if draft.event_type is SpeechEventType.ROLE_CLAIM:
        spoken, confidence = detect_claimed_role_with_confidence(message)
        return spoken == draft.role and confidence >= SPOKEN_CLAIM_CONFIDENCE
    if draft.event_type is SpeechEventType.ABILITY_RESULT and draft.target_id is not None:
        others = {
            pid: ledger.name_of(pid) for pid in ledger.known_player_ids() if pid != speaker_id
        }
        published = detect_public_results(
            message,
            draft.role,
            others,
            role_claimed_in_message=True,
        )
        return any(
            result.target_id == draft.target_id
            and result.is_werewolf == draft.result_is_werewolf
            for result in published
        )
    if draft.target_id is not None:
        return mentions_player(message, draft.target_id, ledger.name_of(draft.target_id))
    return True


# -- registration --


def register_claim_drafts(
    controller: object,
    actor_id: str,
    drafts: Sequence[SpeechEventDraft],
    source_message_id: str = "",
) -> list[SpeechEvent]:
    """Write drafts through the engine's single speech-event write path.

    Only `GameError` is tolerated, and only because the engine raising it means
    the move was illegal -- a claim in the wrong phase, an unknown target. A
    bare `except Exception` here used to hide real bugs behind a game that kept
    running, which is the worst of both: the claim silently vanishes and nothing
    says so.
    """
    recorded: list[SpeechEvent] = []
    for draft in drafts:
        try:
            event = controller.record_speech_event(  # type: ignore[attr-defined]
                actor_id,
                draft.event_type,
                source_message_id=source_message_id,
                target_id=draft.target_id,
                role=draft.role,
                result_is_werewolf=draft.result_is_werewolf,
                referenced_day=draft.referenced_day,
                confidence=draft.confidence,
            )
        except GameError:
            logger.info(
                "speech event rejected by engine: actor=%s type=%s target=%s",
                actor_id,
                draft.event_type.value,
                draft.target_id,
            )
            continue
        if event is not None:
            recorded.append(event)
    return recorded


__all__ = [
    "AMBIGUOUS_CLAIM_CONFIDENCE",
    "DECLARED_CONFIDENCE",
    "SPOKEN_CLAIM_CONFIDENCE",
    "SpeechEventDraft",
    "build_claim_drafts",
    "ensure_fact_sentences",
    "register_claim_drafts",
    "render_result_correction_sentence",
    "render_result_retraction_sentence",
    "render_result_sentence",
    "render_retraction_sentence",
    "render_role_claim_sentence",
    "render_switch_sentence",
]
