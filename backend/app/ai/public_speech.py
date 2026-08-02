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


_WHITE_RE = re.compile(r"(?:人狼ではな(?:い|かった)|白|○)")
_BLACK_RE = re.compile(r"(?:人狼(?:でした|だった|です|だ)?|黒|●)")
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
            result_text = sentence[start : start + 50]
            if _SPECULATION_RE.search(result_text):
                continue
            white = _WHITE_RE.search(result_text)
            if white is not None:
                return DetectedPublicResult(effective_role.value, target_id, False)
            black = _BLACK_RE.search(result_text)
            if black is not None:
                return DetectedPublicResult(effective_role.value, target_id, True)
    return None
