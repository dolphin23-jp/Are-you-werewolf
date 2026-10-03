"""Interpret machine-relevant facts from public speech.

The spoken text remains authoritative for both humans and AIs. Structured LLM
fields can supplement it, but common compact forms must not lose a CO result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.engine.roles import RoleName


@dataclass(frozen=True)
class DetectedPublicResult:
    result_type: str
    target_id: str
    is_werewolf: bool


_TARGET_TRAILER_RE = re.compile(r"^(?:\(p\d+\))?(?:さん|君|ちゃん|氏)?\s*")
_WHITE_RE = re.compile(
    r"^(?:は|が|=|＝|を占って[、，,]?)?\s*(?:人狼ではな(?:い|かった)|白|○)"
)
_BLACK_RE = re.compile(
    r"^(?:は|が|=|＝|を占って[、，,]?)?\s*(?:人狼(?:でした|だった|です|だ)?|黒|●)"
)
_SPECULATION_RE = re.compile(r"(?:と思|に見え|かもしれ|可能性)")
_HONORIFIC = r"(?:さん|君|ちゃん|氏)?"
# "アオイの占い結果", "(p15)の霊判定": the verdict that follows is that player's.
_RESULT_NOUN = r"(?:真|偽)?(?:占い|霊媒|霊能|霊)(?:師)?(?:結果|判定)"
_FIRST_PERSON_RE = re.compile(r"(?:私|僕|俺|自分|わたし|あたし|ワイ|うち)")


def detect_public_result(
    text: str,
    effective_role: RoleName | None,
    candidates: dict[str, str],
    *,
    role_claimed_in_message: bool,
) -> DetectedPublicResult | None:
    """The first explicit seer/medium result in the text, or None."""
    results = detect_public_results(
        text, effective_role, candidates, role_claimed_in_message=role_claimed_in_message
    )
    return results[0] if results else None


def detect_public_results(
    text: str,
    effective_role: RoleName | None,
    candidates: dict[str, str],
    *,
    role_claimed_in_message: bool,
) -> list[DetectedPublicResult]:
    """Every explicit seer/medium result in the text, excluding speculation.

    One message routinely carries two verdicts ("Aは白、Bは黒でした"), and
    stopping at the first silently drops half of what was published.
    """
    if effective_role not in (RoleName.SEER, RoleName.MEDIUM):
        return []
    capability_word = "占" if effective_role == RoleName.SEER else "霊"
    if not role_claimed_in_message and capability_word not in text and "結果" not in text:
        return []
    found: list[DetectedPublicResult] = []
    owners = "|".join(
        re.escape(label)
        for label in sorted(
            (
                label
                for target_id, name in candidates.items()
                for label in (name, f"({target_id})")
            ),
            key=len,
            reverse=True,
        )
    )
    possessive_re = re.compile(rf"(?:{owners}){_HONORIFIC}(?:の|→)\s*$")
    owned_result_re = re.compile(rf"(?:{owners}){_HONORIFIC}の{_RESULT_NOUN}")
    for sentence in re.split(r"[。！？!?\n]", text):
        for target_id, name in candidates.items():
            # The trailing-digit guard is what keeps "Player11(p11)は白" from
            # being read as a verdict about Player1.
            matches = [
                (match.start(), label)
                for label in (name, f"({target_id})")
                for match in [re.search(rf"{re.escape(label)}(?![0-9])", sentence)]
                if match is not None
            ]
            if not matches:
                continue
            start, matched_label = min(matches)
            if _is_someone_elses_verdict(
                sentence[:start], possessive_re=possessive_re, owned_result_re=owned_result_re
            ):
                continue
            result_text = sentence[start + len(matched_label) : start + len(matched_label) + 50]
            result_text = _TARGET_TRAILER_RE.sub("", result_text, count=1)
            if _SPECULATION_RE.search(result_text):
                continue
            if _WHITE_RE.match(result_text) is not None:
                found.append(DetectedPublicResult(effective_role.value, target_id, False))
            elif _BLACK_RE.match(result_text) is not None:
                found.append(DetectedPublicResult(effective_role.value, target_id, True))
    return found


def _is_someone_elses_verdict(
    before: str, *, possessive_re: re.Pattern[str], owned_result_re: re.Pattern[str]
) -> bool:
    """Whether the verdict after `before` is quoted from another player.

    "アオイのソウタ白は…" talks about a result; it is not the speaker's own. A seer
    discussing a rival's divination is the commonest thing said at a table, and
    reading it as a second result of their own put a false "two results in one
    night" on the public record -- which then discredited the one seer who was
    telling the truth.
    """
    if possessive_re.search(before):
        return True
    owned = owned_result_re.search(before)
    # "アカリの霊結果とは別に、私の占い結果はソウタ白": a first-person marker after
    # the quoted result hands the verdict back to the speaker.
    return owned is not None and _FIRST_PERSON_RE.search(before[owned.end() :]) is None
