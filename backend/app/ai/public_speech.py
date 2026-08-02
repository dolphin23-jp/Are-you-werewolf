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


def detect_public_result(
    text: str,
    effective_role: RoleName | None,
    candidates: dict[str, str],
    *,
    role_claimed_in_message: bool,
) -> DetectedPublicResult | None:
    """Detect one explicit seer/medium result, excluding ordinary speculation."""
    if effective_role not in (RoleName.SEER, RoleName.MEDIUM):
        return None
    capability_word = "占" if effective_role == RoleName.SEER else "霊"
    if not role_claimed_in_message and capability_word not in text and "結果" not in text:
        return None
    for sentence in re.split(r"[。！？!?\n]", text):
        for target_id, name in candidates.items():
            positions = [
                index
                for index in (sentence.find(name), sentence.find(f"({target_id})"))
                if index >= 0
            ]
            if not positions:
                continue
            start = min(positions)
            matched_label = name if sentence.startswith(name, start) else f"({target_id})"
            result_text = sentence[start + len(matched_label) : start + len(matched_label) + 50]
            result_text = _TARGET_TRAILER_RE.sub("", result_text, count=1)
            if _SPECULATION_RE.search(result_text):
                continue
            white = _WHITE_RE.match(result_text)
            if white is not None:
                return DetectedPublicResult(effective_role.value, target_id, False)
            black = _BLACK_RE.match(result_text)
            if black is not None:
                return DetectedPublicResult(effective_role.value, target_id, True)
    return None
