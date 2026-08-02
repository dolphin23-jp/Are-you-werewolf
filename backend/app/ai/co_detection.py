"""Detecting a role claim (CO) from what a player actually said.

CO is deliberately emergent: the engine only learns of a claim once it is
spoken in character, never through a side channel. That makes this matcher
load-bearing -- a false positive injects a phantom claim into the CO
composition that every other AI then reasons about.

The naive pattern (role word, then anything within a few characters, then a
claim marker) matched all of these, none of which is a claim:

    占い師のCOを待ってから動いた方がいいと思います   (waiting for someone else's CO)
    占い師は誰ですか                                  (asking who it is)
    あなたは占い師ですか？                            (asking someone else)

So the claim marker must follow the role word *immediately*: any
intervening particle (の / は / が) means the sentence is about somebody
else's role, not the speaker's own. Question and quotative forms are
excluded too, and a sentence naming another player before the role word is
treated as reporting rather than claiming.
"""

from __future__ import annotations

import re

from app.engine.roles import RoleName

_ROLE_WORDS: dict[RoleName, str] = {
    RoleName.SEER: "(?:占い師|占い)",
    RoleName.MEDIUM: "(?:霊媒師|霊媒)",
    RoleName.HUNTER: "狩人",
    RoleName.FREEMASON: "(?:共有者|共有)",
}

# Markers that only make sense when applied to oneself, anchored directly
# to the role word.
_CLAIM_SUFFIX = (
    r"(?:"
    r"[\s、，,]{0,2}(?:CO|ＣＯ|ｃｏ|co|カミングアウト)(?!の|を|が|は|者|数)"
    r"|です(?![かね？?])"
    r"|でした"
    r"|だ(?![とがけろうねよ])"
    r"|をやってい"
    r"|を担当"
    r")"
)

CO_PATTERNS: dict[RoleName, re.Pattern[str]] = {
    role: re.compile(rf"{word}{_CLAIM_SUFFIX}") for role, word in _ROLE_WORDS.items()
}

_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?\n]")
_PARTNER_CONFIRMATION_RE = re.compile(
    r"相方(?:は|が)[、，,\s]*私(?:[^。！？!?\n]{0,30})"
    r"(?:です|だ|で間違い(?:ありません|ない)|で合っています|で合って(?:い)?ます)"
)


def detect_claimed_role(text: str, other_player_names: list[str] | None = None) -> RoleName | None:
    """Return the role this text claims for the speaker, or None.

    `other_player_names` suppresses third-person reports such as
    「ハルトは占い師です」 -- a name appearing before the role word in the
    same sentence means the speaker is talking about someone else.
    """
    names = [n for n in (other_player_names or []) if n]

    # A shared partner confirmation necessarily mentions the first claimant before
    # describing oneself (「ユイの共有CO、相方は私…」).  It is nevertheless an
    # explicit self-claim, so handle this established form before the generic
    # third-person-report guard below.
    if _PARTNER_CONFIRMATION_RE.search(text):
        return RoleName.FREEMASON

    for sentence in _SENTENCE_SPLIT_RE.split(text):
        if not sentence.strip():
            continue
        for role, pattern in CO_PATTERNS.items():
            match = pattern.search(sentence)
            if match is None:
                continue
            if _mentions_other_before(sentence, names, match.start()):
                continue
            return role
    return None


def detect_freemason_partner(
    text: str, candidates: dict[str, str]
) -> str | None:
    """Return the player id publicly named as the speaker's shared partner.

    Handles both an initial reveal (「相方はツムギ(p11)」) and the standard
    confirmation form (「ユイの共有CO、相方は私」).  A bare role discussion
    without the word 相方 is deliberately ignored.
    """
    if "相方" not in text:
        return None
    confirmation = _PARTNER_CONFIRMATION_RE.search(text)
    for player_id, name in candidates.items():
        labels = (name, player_id, f"({player_id})")
        if confirmation is not None:
            prefix = text[: confirmation.start()]
            if any(label in prefix for label in labels):
                return player_id
            continue
        partner_index = text.find("相方")
        suffix = text[partner_index : partner_index + 50]
        if any(label in suffix for label in labels):
            return player_id
    return None


def _mentions_other_before(sentence: str, names: list[str], role_index: int) -> bool:
    for name in names:
        index = sentence.find(name)
        if 0 <= index < role_index:
            return True
    return False
