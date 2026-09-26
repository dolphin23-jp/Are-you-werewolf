"""Detecting a role claim (CO) from what a player actually said.

CO is deliberately emergent: the engine only learns of a claim once it is
spoken in character, never through a side channel. That makes this matcher
load-bearing -- a false positive injects a phantom claim into the CO
composition that every other AI then reasons about, and a miss leaves a real
claim off the board.

A claim is recognised only in a few forms, each with an allow-list for what
may follow it (not a deny-list of what went wrong before):

    CO marker ending the clause:  占い師CO / 占いCOします / 霊媒CO、まだ結果はありません
    copula, not a question:       占い師です / 霊能者です / 狩人やってます / 占い師だよ
    as / continuative (speaker):  占い師として、… / 私は占い師で、…
    role as topic, speaker holds: 占い師は私です
    one's own result announced:   占い結果、ハルトは白 / 今朝の占い結果です。… / 私の霊媒結果
    freemason via the partner:    私の相方は… / 相方はツムギです / ユイは相方として…

Everything else that mentions a role is talk about it: 「霊媒CO後」「共有CO対抗
だったハルト」「占いCOが2人」「占い師COした人」「占いCOしてください」
「偽占い師です」「占い師ですか」「もし私が占い師なら」「私は占い師ではない」.
Another seat named before the role word, a nominalised subject (「噛まれたのは」),
a compound (「情報共有」) or an object particle (「意見を共有」) makes it someone
else's. Quoted or hedged claims are recorded at a low confidence that never
enters the board.

The labelled corpus in tests/ai/fixtures/co_corpus*.py measures this: every case
there, including live-game messages and cases written after the matcher, must be
read exactly.
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

# Every name a table uses for the four claimable roles. Longest first, so
# 「占い師」 is read as one word rather than 「占い」 plus 「師」.
_ROLE_ALIASES: dict[RoleName, tuple[str, ...]] = {
    RoleName.SEER: ("占い師", "占い"),
    RoleName.MEDIUM: ("霊媒師", "霊能者", "霊媒", "霊能"),
    RoleName.HUNTER: ("狩人", "騎士"),
    RoleName.FREEMASON: ("共有者", "共有"),
}
_ROLE_BY_WORD = {word: role for role, words in _ROLE_ALIASES.items() for word in words}
_ROLE_WORD_RE = re.compile(
    "|".join(re.escape(word) for word in sorted(_ROLE_BY_WORD, key=len, reverse=True))
)
_RESULT_ROLES = (RoleName.SEER, RoleName.MEDIUM)

# Kept for callers that want one pattern per role (and for the docs above).
CO_PATTERNS: dict[RoleName, re.Pattern[str]] = {
    role: re.compile("|".join(re.escape(word) for word in words))
    for role, words in _ROLE_ALIASES.items()
}

_SENTENCE_RE = re.compile(r"[^。！？!?\n]+[。！？!?]?")
_CLAUSE_BREAK = "、，,"
_SPACE = r"[\s　]*"

# -- what may follow each form of claim --------------------------------------
#
# Allow-lists, not deny-lists. The old matcher excluded what it had seen go
# wrong (の / を / が after CO), so every new continuation -- 「霊媒CO後」
# 「共有CO対抗だった」「占いCO遅れ」「COした人」 -- was a fresh false positive.
# Here a CO marker only counts when it ends the clause or is conjugated as the
# speaker's own act.
_CO_MARK_RE = re.compile(_SPACE + r"(?:CO|ＣＯ|Co|co|ｃｏ|カミングアウト)")
_AFTER_CO_RE = re.compile(
    r"^" + _SPACE + r"(?:"
    r"$|[、，,。!！…〜~ー」』”]"
    r"|します(?!か)|しました(?!か)|しておきます|しとく|させてください|させて(?:もら|いただ)"
    r"|です(?![かね]|よね|[？?])|でした(?!か|っけ|[？?])|だ(?:よ(?!ね)|ぞ|ぜ)?(?=[、，,。!！]|$)"
    r"|する(?:ね|よ|わ)?(?=[、，,。!！]|$)|し(?=[、，,])"
    r"|(?:を)?した(?:私|僕|俺|自分)"
    # Chat style: 「占いCO　ユイ白」 -- a space, then a verdict.
    r"|[\s\u3000]+(?=[^\s\u3000、。]{1,12}?(?:白|黒))"
    r")"
)
_AFTER_COPULA_RE = re.compile(
    r"^(?:"
    r"です(?![かね]|よね|[？?])|でした(?!か|っけ|[？?])"
    r"|だ(?:よ(?!ね)|ぞ|ぜ|わ)?(?=[、，,。!！\s]|$)"
    r"|(?:を)?やって(?:い|ま|る)|を担当|担当です"
    r")"
)
# 「私は占い師で、…」 -- a continuative copula, only with the speaker as subject.
_AFTER_CONTINUATIVE_RE = re.compile(r"^で(?=[、，,])")
# 「占い師は私です」 -- the role as topic, the speaker as its holder.
_AFTER_ROLE_IS_ME_RE = re.compile(
    r"^(?:は|が)" + _SPACE + r"[、，,]?" + _SPACE
    + r"(?:私|僕|俺|自分|わたし|ぼく|おれ)(?:です(?![かね]|よね|[？?])|だ(?=[、。!！]|$)|でした)"
)
_AFTER_AS_RE = re.compile(r"^として(?!の)")
_RESULT_ANNOUNCED_RE = re.compile(
    r"^結果(?:[、，,:：→]|です|を(?:出|公開|報告|伝|発表|言))"
)
_VERDICT_RE = re.compile(r"白|黒|人狼|人間|村人|狼|シロ|クロ")
# After a name: 「ハルトは人狼ではない」「ユイ白」「ソウタさんが黒」.
_NAME_VERDICT_RE = re.compile(
    r"^(?:さん|くん|ちゃん)?(?:は|が)?[^。！？!?]{0,4}?(?:白|黒|人狼|人間|村人|狼|シロ|クロ)"
)

# -- who the sentence is about -------------------------------------------------

_SELF_WORD = r"(?:私|僕|俺|自分|わたし|ぼく|おれ|うち)"
# The role word must start a word: 「偽占い師」「情報共有」「対抗占い」 are
# about something else. 真 (「真占い師です」) and a pronoun directly attached
# (「私占い師です」) are the only fused prefixes that still mean the speaker.
_COMPOUND_PREFIX_RE = re.compile(r"[一-鿿゠-ヿA-Za-z0-9]$")
_ALLOWED_PREFIX_RE = re.compile(r"(?:真|本物の|" + _SELF_WORD + r")$")
# に only as a case particle, not in adverbs (「先に」「最初に」「正直に」「特に」).
_OBJECT_PARTICLE_BEFORE_RE = re.compile(
    r"(?:を|と|(?<![先次初直特更既共仮実])に)" + _SPACE + r"$"
)
# A subject other than the speaker right before the role word:
# 「噛まれたのは占い師でした」「これは共有です」「ハルトが占い師です」.
_TOPIC_BEFORE_RE = re.compile(r"(?:は|が)" + _SPACE + r"$")
_SELF_OR_ADVERB_TOPIC_RE = re.compile(
    r"(?:" + _SELF_WORD + r"|実|本当|実際|今|まず|では|それで|じつ)(?:は|が)" + _SPACE + r"$"
)
# 「では、」「それでは」 opening the clause: a conjunction, not a topic.
_CONJUNCTION_HEAD_RE = re.compile(r"(?:^|[、，,\s])(?:それ)?では" + _SPACE + r"$")
_OTHER_ADDRESS_RE = re.compile(r"^(?:さん|くん|君|ちゃん|様)?(?:へ|に)?[、，,\s]")
_SELF_HEAD_RE = re.compile(
    r"^(?:[\s　]*(?:えっと|えー|はい|まず|では|じゃあ|改めて|実は|やっぱり|本当は)[、，,\s]*)*$"
)
_SELF_SUBJECT_HEAD_RE = re.compile(
    r"(?:^|[、，,\s])" + _SELF_WORD + r"(?:は|が|も)?" + _SPACE + r"$"
)
_TIME_HEAD_RE = re.compile(
    r"(?:今朝|昨夜|昨晩|昨日|本日|今日|初日|\d+日目|[一二三四五六七八九]日目)の" + _SPACE + r"$"
)
_SELF_POSSESSIVE_RE = re.compile(
    _SELF_WORD + r"(?P<between>[^、，,。！？!?]{0,24}?)の" + _SPACE + r"$"
)
# After 「私の占いCO」/「私の霊媒結果」: the act itself or a particle, never a
# noun that makes it something else (「私の占いCO予想」).
_AFTER_POSSESSIVE_RE = re.compile(
    r"^" + _SPACE + r"(?:直後|後|前|時|の(?:時|際|直後|後|タイミング)"
    r"|では|で|は|が|を|に|から|より|も|と|[、，,。:：→]|$)"
)

# -- freemason partner forms ---------------------------------------------------

_MY_PARTNER_RE = re.compile(
    _SELF_WORD + r"の相方(?:は|が|[:：])|(?:は|が)" + _SPACE + _SELF_WORD + r"の相方"
)
_PARTNER_IS_RE = re.compile(
    r"(?:^|[、，,])" + _SPACE
    + r"相方(?:は|が|[:：])" + _SPACE
    + r"(?!" + _SELF_WORD + r"|誰|だれ|い(?:ま|る|な)|未|不明|まだ)"
    + r"(?P<label>[^、，,。！？!?]{1,30}?)(?:さん)?(?:です|で|[。、，,]|$)"
)
_PARTNER_AS_RE = re.compile(r"は相方(?:として|です|で(?=[、，,]))")

# -- what makes a claim-shaped sentence something else -------------------------

# Relaying what someone else said. These still produce an event -- something
# claim-shaped was said and that is worth recording -- but at a confidence too
# low to be promoted into the board's CO composition.
_REPORTED_SPEECH_RE = re.compile(
    r"(?:と(?:言|いって|言って|主張|自称|名乗)|とのこと|そうです|らしい|だそう|"
    r"と述べ|と宣言)"
)
_HEDGE_RE = re.compile(
    r"(?:たぶん|多分|おそらく|かもしれ|かも$|ような気|気がし|仮に|もし|"
    r"だとしたら|だったら|であれば|なら(?:ば)?(?=[、，,。]|$)|ということにし|かどうか|でしょうか)"
)
_QUESTION_END_RE = re.compile(r"(?:[？?]|(?:か|かな|かね|の)[。]?)$")
_PARTNER_CONFIRMATION_RE = re.compile(
    r"相方(?:は|が)[、，,\s]*私(?:[^。！？!?\n]{0,30})"
    r"(?:です|だ|で間違い(?:ありません|ない)|で合っています|で合って(?:い)?ます)"
)
_PARTNER_REVEAL_RE = re.compile(
    r"相方(?:は|が|[:：])[、，,\s]*(?P<label>[^、，。！？!?\n]{1,30})"
)

_BINDING, _AMBIGUOUS = "binding", "ambiguous"


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
    said without a phantom CO entering the board analysis. Anything that is
    about someone else, a request, a question or a denial is (None, 0.0).
    """
    names = [n for n in (other_player_names or []) if n]

    # A shared partner confirmation necessarily mentions the first claimant before
    # describing oneself (「ユイの共有CO、相方は私…」).  It is nevertheless an
    # explicit self-claim, so handle this established form before the generic
    # third-person-report guard below.
    confirmation = _PARTNER_CONFIRMATION_RE.search(text)
    if confirmation is not None:
        sentence = _sentence_around(text, confirmation.start())
        # Quoting or supposing a confirmation is not making one.
        if _REPORTED_SPEECH_RE.search(sentence) or _HEDGE_RE.search(sentence):
            return RoleName.FREEMASON, AMBIGUOUS_CLAIM_CONFIDENCE
        return RoleName.FREEMASON, SPOKEN_CLAIM_CONFIDENCE

    ambiguous: RoleName | None = None
    for sentence_match in _SENTENCE_RE.finditer(text):
        sentence = sentence_match.group(0)
        rest_of_text = text[sentence_match.end() :]
        for role, strength in _sentence_claims(sentence, rest_of_text, names):
            if strength == _BINDING:
                return role, SPOKEN_CLAIM_CONFIDENCE
            if ambiguous is None:
                ambiguous = role
    if ambiguous is not None:
        return ambiguous, AMBIGUOUS_CLAIM_CONFIDENCE
    return None, 0.0


def _sentence_claims(
    sentence: str, rest_of_text: str, names: list[str]
) -> list[tuple[RoleName, str]]:
    """Every self-claim in one sentence, in order, with how binding each is."""
    found: list[tuple[RoleName, str]] = []
    is_question = bool(_QUESTION_END_RE.search(sentence.strip()))
    for match in _ROLE_WORD_RE.finditer(sentence):
        role = _ROLE_BY_WORD[match.group(0)]
        head = sentence[: match.start()]
        tail = sentence[match.end() :]
        form = _claim_form(role, head, tail, sentence[match.end() :] + rest_of_text, names)
        if form is None:
            continue
        if form != "possessive" and _about_someone_else(head, names):
            continue
        strength = _strength(sentence, match.start(), match.end(), is_question)
        if strength is not None:
            found.append((role, strength))
    if "相方" in sentence:
        partner = _partner_claim(sentence, names, is_question)
        if partner is not None:
            found.append((RoleName.FREEMASON, partner))
    return found


def _claim_form(
    role: RoleName, head: str, tail: str, following: str, names: list[str]
) -> str | None:
    """Which kind of self-claim the role word starts, if any."""
    if _COMPOUND_PREFIX_RE.search(head) and not _ALLOWED_PREFIX_RE.search(head):
        return None
    # 「意見を共有です」「ハルトと共有」: the word is a verb or someone else's.
    if _OBJECT_PARTICLE_BEFORE_RE.search(head):
        return None
    clause_head = _clause_head(head)
    self_possessive = _self_possessive(clause_head)
    co = _CO_MARK_RE.match(tail)
    if co is not None:
        if self_possessive and _AFTER_POSSESSIVE_RE.match(tail[co.end() :]):
            return "possessive"  # 「私の共有CO直後に…」
        after = tail[co.end() :]
        if _AFTER_CO_RE.match(after) or _names_then_verdict(after, names):
            return "co"
        return None
    if _AFTER_COPULA_RE.match(tail):
        return "copula"
    if _AFTER_CONTINUATIVE_RE.match(tail) and _SELF_SUBJECT_HEAD_RE.search(clause_head):
        return "copula"
    if _AFTER_ROLE_IS_ME_RE.match(tail):
        return "role_is_me"
    if _AFTER_AS_RE.match(tail) and (
        _SELF_HEAD_RE.match(clause_head) or _SELF_SUBJECT_HEAD_RE.search(clause_head)
    ):
        return "as"
    if role in _RESULT_ROLES and tail.startswith("結果"):
        if self_possessive and _AFTER_POSSESSIVE_RE.match(tail[len("結果") :]):
            return "possessive"  # 「私の霊媒結果では…」
        announced = _RESULT_ANNOUNCED_RE.match(tail) is not None
        if (
            announced
            and (_SELF_HEAD_RE.match(clause_head) or _TIME_HEAD_RE.search(clause_head))
            and _VERDICT_RE.search(following)
        ):
            return "result"
    return None


def _names_then_verdict(after: str, names: list[str]) -> bool:
    """「占いCOハルトは人狼ではない」: written without a break, but the next word
    is a seat's name and a verdict follows it -- the claimant's first result."""
    stripped = after.lstrip(" \u3000")
    for name in names:
        if stripped.startswith(name):
            remainder = stripped[len(name) :]
            return _NAME_VERDICT_RE.match(remainder) is not None
    return False


def _clause_head(head: str) -> str:
    cut = max(head.rfind(mark) for mark in _CLAUSE_BREAK)
    return head[cut + 1 :] if cut >= 0 else head


def _self_possessive(clause_head: str) -> bool:
    match = _SELF_POSSESSIVE_RE.search(clause_head)
    if match is None:
        return False
    # 「私が疑うハルトの占い結果」 -- someone else's result after all.
    between = match.group("between")
    # The speaker's own label may sit in between: 「私Player14(p14)の霊媒結果」.
    return "の" not in between or bool(re.fullmatch(r"[A-Za-z]+\d+\(p\d+\)", between))


def _about_someone_else(head: str, names: list[str]) -> bool:
    """The role word belongs to another subject in this sentence."""
    if _mentions_other_before(head, names):
        return True
    clause_head = _clause_head(head)
    if (
        _TOPIC_BEFORE_RE.search(clause_head)
        and not _SELF_OR_ADVERB_TOPIC_RE.search(clause_head)
        and not _CONJUNCTION_HEAD_RE.search(clause_head)
    ):
        return True
    return False


def _strength(sentence: str, start: int, end: int, is_question: bool) -> str | None:
    """Binding, merely claim-shaped, or not a claim at all."""
    if is_question and not _has_clause_break(sentence[end:]):
        return None
    if _inside_quote(sentence, start):
        return _AMBIGUOUS
    clause = _clause_around(sentence, start, end)
    if _HEDGE_RE.search(clause) or _REPORTED_SPEECH_RE.search(sentence[end:]):
        return _AMBIGUOUS
    return _BINDING


def _partner_claim(sentence: str, names: list[str], is_question: bool) -> str | None:
    """「私の相方は…」「相方はツムギです」「ユイは相方として…」: freemason, said as one."""
    if is_question:
        return None
    for pattern in (_MY_PARTNER_RE, _PARTNER_IS_RE):
        match = pattern.search(sentence)
        if match is None:
            continue
        before = sentence[: match.start() + 1]
        if pattern is _PARTNER_IS_RE and re.search(r"の" + _SPACE + r"$", before):
            continue
        return _strength(sentence, match.start(), match.end(), False)
    match = _PARTNER_AS_RE.search(sentence)
    if match is not None:
        clause_head = _clause_head(sentence[: match.start()])
        mentioned = [name for name in names if name in clause_head]
        # Exactly one seat named, and nobody else's 相方: the speaker's own partner.
        if len(mentioned) == 1 and "の相方" not in clause_head and "と" not in clause_head:
            return _strength(sentence, match.start(), match.end(), False)
    return None


def _has_clause_break(text: str) -> bool:
    return any(mark in text for mark in _CLAUSE_BREAK)


def _clause_around(sentence: str, start: int, end: int) -> str:
    before = max(sentence.rfind(mark, 0, start) for mark in _CLAUSE_BREAK)
    afters = [i for i in (sentence.find(mark, end) for mark in _CLAUSE_BREAK) if i != -1]
    return sentence[before + 1 : min(afters) if afters else len(sentence)]


def _inside_quote(sentence: str, index: int) -> bool:
    before = sentence[:index]
    return any(before.count(open_) > before.count(close) for open_, close in ("「」", "『』", "“”"))


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


def _sentence_around(text: str, index: int) -> str:
    start = max((text.rfind(mark, 0, index) for mark in "。！？!?\n"), default=-1) + 1
    ends = [end for end in (text.find(mark, index) for mark in "。！？!?\n") if end != -1]
    return text[start : min(ends) if ends else len(text)]


def _mentions_other_before(head: str, names: list[str]) -> bool:
    """Another seat is named before the role word -- unless only addressed
    (「ハルトさんへ、占い師COです」)."""
    for name in names:
        for found in re.finditer(re.escape(name), head):
            after = head[found.end() :]
            # 「Player1」 inside 「Player12」 is a different seat.
            if after[:1].isdigit():
                continue
            if _OTHER_ADDRESS_RE.match(after) and ("へ" in after[:4] or "、" in after[:4]):
                continue
            return True
    return False
