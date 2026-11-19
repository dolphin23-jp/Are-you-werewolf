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

That was still a reading of "role word + CO", not of a declaration. Run 16's
カイト, a wolf told to lurk, said 「共有CO二人は了解」 -- he acknowledged the
two freemason claims -- and the matcher registered a third freemason CO for
him; the table spent the game on a claim nobody made. So a sentence counts
only in the shapes a speaker declares their own role in
(`_declares_with_marker`, `_declares_with_copula`): the CO marker closes the
clause or is followed by a declaring verb (「占い師CO。」「霊媒COします」
「共有CO、相方は…」), or the role is the predicate of a sentence about the
speaker (「占い師です」「俺は本物の共有だ」). The marker followed by anything
else -- a count (「CO二人」), an acknowledgement (「CO了解」), a noun or a
particle that makes the CO a thing talked about (「CO相手」「COで」「CO後」) --
is talk about claims, and is not read as one.
"""

from __future__ import annotations

import re

from app.ai.reasoning.facts import mentions_player
from app.engine.roles import RoleName
from app.engine.speech_events import CLAIM_CONFIDENCE_THRESHOLD

# An unambiguous matcher hit on free text is a claim; a quoted or hedged one is
# only evidence that something claim-shaped was said.
SPOKEN_CLAIM_CONFIDENCE = 0.9
AMBIGUOUS_CLAIM_CONFIDENCE = 0.4

_ROLE_WORDS: dict[RoleName, str] = {
    RoleName.SEER: "(?:占い師|占い)",
    RoleName.MEDIUM: "(?:霊媒師|霊媒)",
    RoleName.HUNTER: "狩人",
    RoleName.FREEMASON: "(?:共有者|共有)",
}

# What may close a declaring clause: the end of the sentence, a comma, a space,
# decoration, or a closing quote (a quote is then judged by `_REPORTED_SPEECH_RE`).
_CLAUSE_END = r"(?=$|[、，,\s　〜~ー…・！!♪☆」』)）])"

# The role word with the CO marker right after it (a space at most).
_WITH_MARKER: dict[RoleName, re.Pattern[str]] = {
    role: re.compile(rf"{word}[ 　]{{0,2}}(?:CO|ＣＯ|ｃｏ|co|Co|カミングアウト)")
    for role, word in _ROLE_WORDS.items()
}
# What may follow the marker in a declaration: nothing, a clause boundary, or
# a declaring verb or copula said by the speaker of themselves. Not -- among
# what the old reader let through -- a count (「CO二人」), a noun (「CO相手」
# 「CO後」「CO対抗」), a particle (「COで」「COと」), a request (「COして」), a
# quotation (「COするって」) or a condition (「COするなら」).
_MARKER_TAIL = re.compile(
    r"(?:"
    rf"[〜~ー…♪☆！!\s　]*$"
    r"|[、，,\s　」』]"
    rf"|し(?:ます|ました|まーす|とく|ておく|ておきます|ときます)(?:ね|よ)?{_CLAUSE_END}"
    rf"|する(?:ね|よ|ぞ|ぜ|わ)?{_CLAUSE_END}"
    rf"|です(?:よ)?{_CLAUSE_END}"
    rf"|でした{_CLAUSE_END}"
    rf"|だ(?:よ|ぜ|ぞ)?{_CLAUSE_END}"
    r"|を?(?:継続|続け)"
    # A slide, as `claims.render_switch_sentence` writes it: 「…、占いCOに変更します」.
    r"|(?:に|へ)(?:変更|切り替え|変え)"
    r")"
)
# After 「CO、」: an acknowledgement of claims already made, or a count of them,
# is talk about others' COs (「共有CO、了解」「共有CO、二人とも確認」).
_ACKNOWLEDGED = re.compile(
    r"了解|承知|把握|ありがと|確認しま|確認した|受け取|おけ$|オッケ|おっけ|OK|ＯＫ"
)
_COUNTED = re.compile(
    r"^[、，,\s　]*(?:[0-9０-９一二三四五六七八九十]+(?:人|名)|二人|ふたり|両方|両者|全員)"
)

# The role as the predicate of the speaker's own sentence: 「占い師です」,
# 「霊媒師だ、COしておく」, 「占い師でした」, 「占い師をやっています」.
_WITH_COPULA: dict[RoleName, re.Pattern[str]] = {
    role: re.compile(
        rf"{word}(?:"
        rf"(?:です|でした|だ)(?:よ|ぜ|ぞ)?{_CLAUSE_END}"
        rf"|をやって(?:います|いる|ます|る){_CLAUSE_END}"
        rf"|を担当(?:しています|してる|します|する){_CLAUSE_END}"
        r")"
    )
    for role, word in _ROLE_WORDS.items()
}
_FIRST_PERSON = r"(?:私|わたし|わたくし|俺|おれ|オレ|僕|ぼく|ボク|自分|あたし|うち|ウチ|ワイ)"
_GENUINE = r"(?:本物の|真の|本当の|ほんとの)"
# The speaker as the subject right before the role word: 「俺は本物の共有だ」,
# 「私も占い師CO」. A seat named earlier in the sentence is then who is being
# spoken to (「ホノカ、俺は本物の共有だ」), not whose role it is.
_FIRST_PERSON_SUBJECT = re.compile(
    rf"{_FIRST_PERSON}(?:は|が|も|こそ)[、，,\s　]*(?:実は)?{_GENUINE}?$"
)
# What may stand before a copula-form claim in its clause: nothing, the
# speaker as subject, or a word that only colours the claim. A hedge or a
# condition is let through so that the sentence is recorded as ambiguous
# (`_HEDGE_RE`), not dropped: 「たぶん占い師です」 was said, and is not a claim.
_COPULA_PREFIX = re.compile(
    r"^[\s　…‥・ー〜~「『]*"
    r"(?:(?:実は|じつは|本当は|ほんとは|やっぱり|やっぱ|やはり|一応|改めて|ちなみに|正直|"
    r"はい|うん|えっと|えーと|まあ|じゃあ|では|それでは|なので|だから|つまり|"
    r"たぶん|多分|おそらく|もし|仮に)[、，,\s　]*)?"
    rf"(?:{_FIRST_PERSON}(?:は|が|も|こそ)[、，,\s　]*)?"
    r"(?:実は|じつは)?"
    rf"{_GENUINE}?$"
)
# Someone else as the subject of the clause (「あいつは共有CO」, 「対抗は占い師だ」).
_OTHER_SUBJECT = re.compile(
    r"(?:彼|彼女|あなた|あんた|君|きみ|お前|おまえ|あいつ|こいつ|そいつ|その人|この人|あの人"
    r"|誰|どっち|どちら|片方|もう片方|もう一人|相手|対抗|ほか|他|二人|全員)(?:は|が|も|こそ)"
)

# A clause that ends by marking a subject right before the role word, and the
# few topics that only set the scene (「今日は占い師COします」).
_SUBJECT_END = re.compile(r"(?:は|が|も|こそ)$")
_ADVERBIAL_TOPIC = re.compile(
    r"(?:今日|本日|ここ|ここで|まず|最初|それで|で|実|じつ|本当|ほんと)(?:は|も)$"
)

_SENTENCE_RE = re.compile(r"[^。！？!?\n]+[。！？!?]*")
_CLAUSE_BOUNDARY_RE = re.compile(r"[、，,\s　]")

# Forms that make a claim-shaped sentence something other than a claim: relaying
# what someone else said, or hedging one's own. These still produce an event --
# something claim-shaped was said and that is worth recording -- but at a
# confidence too low to be promoted into the board's CO composition.
_REPORTED_SPEECH_RE = re.compile(
    r"(?:と(?:言|いって|言って|主張|自称)|とのこと|そうです|らしい|だそう|"
    r"と述べ|と宣言|って(?:言|い[うっ]|主張|話)|「[^」]*」)"
)
_HEDGE_RE = re.compile(
    r"(?:たぶん|多分|おそらく|かもしれ|かも$|ような気|気がし|仮に|もし|"
    r"だとしたら|であれば|ということにし|かどうか|でしょうか)"
)
_PARTNER_CONFIRMATION_RE = re.compile(
    r"相方(?:は|が)[、，,\s]*私(?:[^。！？!?\n]{0,30})"
    r"(?:です|だ|で間違い(?:ありません|ない)|で合っています|で合って(?:い)?ます)"
)
_PARTNER_REVEAL_RE = re.compile(
    r"相方(?:は|が|[:：])[、，,\s]*(?P<label>[^、，。！？!?\n]{1,30})"
)


def detect_claimed_role(text: str, other_player_names: list[str] | None = None) -> RoleName | None:
    """Return the role this text unambiguously claims for the speaker, or None.

    `other_player_names` suppresses third-person reports such as
    「ハルトは占い師です」 -- a name appearing before the role word in the
    same sentence means the speaker is talking about someone else.
    """
    role, confidence = detect_claimed_role_with_confidence(text, other_player_names)
    return role if confidence >= CLAIM_CONFIDENCE_THRESHOLD else None


def detect_claimed_role_with_confidence(
    text: str, other_player_names: list[str] | None = None
) -> tuple[RoleName | None, float]:
    """Same match, plus how much of a claim it actually is.

    A quoted or hedged sentence is reported at `AMBIGUOUS_CLAIM_CONFIDENCE`
    rather than dropped, so the log can show that something claim-shaped was
    said without a phantom CO entering the board analysis. A sentence that
    only talks about claims (「共有CO二人は了解」) is not claim-shaped at all.
    """
    names = [n for n in (other_player_names or []) if n]

    # A shared partner confirmation necessarily mentions the first claimant before
    # describing oneself (「ユイの共有CO、相方は私…」).  It is nevertheless an
    # explicit self-claim, so handle this established form before the generic
    # third-person-report guard below.
    if _PARTNER_CONFIRMATION_RE.search(text):
        return RoleName.FREEMASON, SPOKEN_CLAIM_CONFIDENCE

    for match in _SENTENCE_RE.finditer(text):
        whole = match.group(0)
        sentence = whole.rstrip("。！？!?")
        if not sentence.strip() or any(mark in whole for mark in "？?"):
            # A question asks; it does not declare (「占い師CO？」).
            continue
        found = _declared_role(sentence, names)
        if found is None:
            continue
        if _REPORTED_SPEECH_RE.search(sentence) or _HEDGE_RE.search(sentence):
            return found, AMBIGUOUS_CLAIM_CONFIDENCE
        return found, SPOKEN_CLAIM_CONFIDENCE
    return None, 0.0


def _declared_role(sentence: str, names: list[str]) -> RoleName | None:
    """The first role the sentence declares for its speaker, by position, or None."""
    candidates: list[tuple[int, RoleName]] = []
    for role in _ROLE_WORDS:
        for found in _WITH_MARKER[role].finditer(sentence):
            if _declares_with_marker(sentence, found.start(), found.end(), names):
                candidates.append((found.start(), role))
        for found in _WITH_COPULA[role].finditer(sentence):
            if _declares_with_copula(sentence, found.start(), names):
                candidates.append((found.start(), role))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def _declares_with_marker(sentence: str, start: int, end: int, names: list[str]) -> bool:
    """「占い師CO。」「対抗で霊媒COします」「共有CO、相方は…」 -- not 「共有CO二人は了解」."""
    before = sentence[:start]
    if not _speaker_owns(before, names):
        return False
    # 「マコトの共有CO」, 「二人の共有CO」: a CO that belongs to someone.
    if before.endswith("の") and not _FIRST_PERSON_SUBJECT.search(before):
        return False
    tail = sentence[end:]
    if any(tail.startswith(name) for name in names):
        # The compact human form, 「占いCOアカリ白」.
        return True
    if _MARKER_TAIL.match(tail) is None:
        return False
    return not (_ACKNOWLEDGED.search(tail) or _COUNTED.match(tail))


def _declares_with_copula(sentence: str, start: int, names: list[str]) -> bool:
    """「占い師です」「霊媒師だ、COしておく」「俺は本物の共有だ」 -- not 「共有だから」."""
    before = sentence[:start]
    if not _speaker_owns(before, names):
        return False
    clause = _CLAUSE_BOUNDARY_RE.split(before)[-1] if before else ""
    if _COPULA_PREFIX.match(clause):
        return True
    # 「…、俺は 本物の共有だ」: the subject split from the role by a space.
    return bool(_FIRST_PERSON_SUBJECT.search(before))


def _speaker_owns(before: str, names: list[str]) -> bool:
    """Whether the role word that follows `before` is the speaker's, not someone else's.

    A seat named before the role word makes the sentence a report about that
    seat (「ハルトは占い師です」), unless the speaker stands as the subject
    right before the role (「ホノカ、俺は本物の共有だ」). Another subject in the
    same clause (「あいつは共有CO」) is someone else's claim too.
    """
    if _FIRST_PERSON_SUBJECT.search(before):
        return True
    if any(name in before for name in names):
        return False
    clause = _CLAUSE_BOUNDARY_RE.split(before)[-1] if before else ""
    if _SUBJECT_END.search(clause) and not _ADVERBIAL_TOPIC.search(clause):
        # 「あっちが占いCO」: whoever is marked as the subject, it is not the speaker.
        return False
    return _OTHER_SUBJECT.search(clause) is None


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
    if confirmation is not None:
        prefix = text[: confirmation.start()]
        confirmation_matches = [
            (max(prefix.rfind(name), prefix.rfind(f"({player_id})")), player_id)
            for player_id, name in candidates.items()
            if _mentions_player(prefix, player_id, name)
        ]
        return max(confirmation_matches)[1] if confirmation_matches else None
    reveal = _PARTNER_REVEAL_RE.search(text)
    if reveal is None:
        return None
    label = reveal.group("label")
    reveal_matches = [
        player_id
        for player_id, name in candidates.items()
        if _mentions_player(label, player_id, name)
    ]
    if len(reveal_matches) == 1:
        return reveal_matches[0]
    return None


def _mentions_player(text: str, player_id: str, name: str) -> bool:
    """Match p1 without accidentally treating the p1 prefix in p11 as a hit."""
    return mentions_player(text, player_id, name)
