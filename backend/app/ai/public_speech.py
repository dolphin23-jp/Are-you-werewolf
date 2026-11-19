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
# Right after the colour, these make it an impression ("黒っぽい", "白寄り",
# "黒目"), a side ("人狼側"), a negation ("黒ではない") or a hypothetical
# ("黒なら"), not a result. Reading "Xは黒っぽい" as a published black put a
# binding verdict nobody gave on the record. The list follows claims.py's
# `_VERDICT_TAIL_REJECT_RE`, minus the bare "い" and "確" that a short
# verdict can legitimately be followed by here; "ら" and "としたら" close the
# gap after a full verdict ("人狼だったら"). The colour group is atomic, so a
# rejected "人狼だった" cannot fall back to "人狼だ" + "ったら" and pass.
_NOT_A_VERDICT_TAIL = (
    r"(?![」』\"”)）]*(?:っぽ|ぽい|寄り|より|目|め|く見|視|要素|印象|すぎ|過ぎ|塗|置き|位置"
    r"|側|陣営|候補|では|じゃ|なら|だったら|だとしたら|として|としたら|とすれば|ら|かも))"
)
# A bare colour followed by a particle is a verdict being talked about, not one
# being given: 「リク黒が外れたのはダイキさんの問題」 (run 15: ダイキ's black on
# リク, registered as the speaker's own), 「ミオ白を出した」, 「ソウタ白は信用でき
# ない」, 「ハルト黒と私の霊結果が一致」. A verdict the speaker gives ends its
# clause (「ソウタ白。」「ソウタ白、ケント黒」) or goes on as a predicate
# (「白でした」「白と出た」). Only the bare forms are checked: after 「人狼でした」
# the clause has already ended as a verdict. The speaker's own possessive keeps
# the verdict theirs whatever follows (「私のカイト黒を覆す根拠にはならない」).
_TALKED_ABOUT = r"(?!が|を|の|は|も|に|へ|から|まで|って|と(?!出))"
_VERDICT_LEAD = r"^(?:は|が|=|＝|を占って[、，,]?)?\s*"


def _verdict_re(white: bool, *, talked_about_ok: bool) -> re.Pattern[str]:
    after_bare = "" if talked_about_ok else _TALKED_ABOUT
    colour = (
        rf"(?>人狼ではな(?:い|かった)|(?:白|○){after_bare})"
        if white
        else rf"(?>人狼(?:でした|だった|です|だ)|(?:人狼|黒|●){after_bare})"
    )
    return re.compile(_VERDICT_LEAD + colour + _NOT_A_VERDICT_TAIL)


_WHITE_RE = _verdict_re(True, talked_about_ok=False)
_BLACK_RE = _verdict_re(False, talked_about_ok=False)
_OWN_WHITE_RE = _verdict_re(True, talked_about_ok=True)
_OWN_BLACK_RE = _verdict_re(False, talked_about_ok=True)
_SPECULATION_RE = re.compile(r"(?:と思|に見え|かもしれ|可能性)")
_HONORIFIC = r"(?:さん|君|ちゃん|氏)?"
# "アオイの占い結果", "(p15)の霊判定", "ケントの結果": the verdict that follows is
# that player's.
_RESULT_NOUN = r"(?:(?:真|偽)?(?:占い|霊媒|霊能|霊)(?:師)?)?(?:結果|判定)"
_FIRST_PERSON_RE = re.compile(r"(?:私|僕|俺|自分|わたし|あたし|ワイ|うち)")
# 「私の」 right before the target: the verdict is the speaker's own.
_OWN_POSSESSIVE_RE = re.compile(rf"{_FIRST_PERSON_RE.pattern}の\s*$")


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
            own = _OWN_POSSESSIVE_RE.search(sentence[:start]) is not None
            white_re, black_re = (_OWN_WHITE_RE, _OWN_BLACK_RE) if own else (_WHITE_RE, _BLACK_RE)
            if white_re.match(result_text) is not None:
                found.append(DetectedPublicResult(effective_role.value, target_id, False))
            elif black_re.match(result_text) is not None:
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
