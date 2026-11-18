"""AICoordinator: orchestrates all AI players through each game phase with
a phase-appropriate concurrency policy:

  - discussion: strictly sequential (each AI reads prior messages before
    speaking, for coherence)
  - voting: parallel (independent judgments), looped across runoff rounds
  - night: independent actions (divine/guard/wolf-chat/freemason-chat) run
    in parallel, then the alpha werewolf's attack decision runs afterward,
    fed the wolf-chat transcript

CO (role-claim) detection is always based on pattern-matching the AI's own
generated public chat text -- never a scripted side-channel -- so bluffing
stays emergent rather than scripted.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from app.ai.co_detection import detect_claimed_role
from app.ai.context import ContextBuilder, DaySummaryManager
from app.ai.deception import FakeClaimGuard, assign_madman_strategy, assign_wolf_deception
from app.ai.personalities import (
    assign_personalities,
    chat_stage_max_chars,
    discussion_length_range,
)
from app.ai.player_agent import AIPlayerAgent, truncate_at_sentence
from app.ai.provider.base import LLMProvider, Message
from app.ai.reasoning import (
    PublicFactLedger,
    ValidationLog,
    compose_day_summary,
    detect_vote_plan_mismatch,
    render_public_fact_summary,
    validate_discussion_output,
    validate_public_result_claim,
)
from app.ai.reasoning.belief import CorrectionOutcome, EvidenceVisibility, StoryStatus
from app.ai.reasoning.board_memo import namify
from app.ai.reasoning.claims import (
    SpeechEventDraft,
    build_claim_drafts,
    ensure_fact_sentences,
    register_claim_drafts,
)
from app.ai.reasoning.dialogue import DiscussionDecision
from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.rendering import (
    displayed_execution_target,
    enforce_execution_target,
)
from app.ai.reasoning.runtime import ReasoningRuntime, SeatReasoning
from app.ai.schemas import (
    DirectedQuestion,
    DiscussionOutput,
    MorningIntentOutput,
    NightActionOutput,
    PublicResultClaim,
    VoteOutput,
)
from app.ai.schemas import PublicResultClaim as SchemaPublicResultClaim
from app.engine.game import GameError
from app.engine.phases import Phase
from app.engine.roles import ROLE_DEFINITIONS, RoleName
from app.engine.speech_events import SpeechEventType
from app.engine.state import (
    ChatChannel,
    FreemasonPartnerClaim,
    GameState,
    PendingQuestion,
)
from app.eval.reasoning_analyzer import (
    BallotFacts,
    PublishedResult,
    SpokenLine,
    classify_vote_change,
    declares_grey_random,
    named_seat_ids,
    sentence_naming,
)
from app.eval.transcript import (
    CorrectionAuditRecord,
    DecisionAuditRecord,
    NightActionAuditRecord,
    ResultPublicationAuditRecord,
    TranscriptRecorder,
    Utterance,
    WolfAllyVotePlan,
)
from app.sessions.models import DiscussionRoundState


def _night_utility(seat: SeatReasoning, action_type: str) -> dict[str, float]:
    """The utility map that decided this night action, for the recorded reason."""
    state = seat.belief.state
    maps: dict[str, dict[str, float]] = {
        "divine": state.divine_utility_scores,
        "guard": state.guard_utility_scores,
        "attack": state.attack_utility_scores,
    }
    return maps.get(action_type, state.execution_utility_scores)


def _normalized_result_key(result_id: str) -> tuple[str, str]:
    """Drop the day from a `type:day:target` id, leaving `(type, target)`.

    The publication audit answers one question -- was the result disclosed at
    all -- and it has to answer it across two differently-shaped sources: the
    requirement carries the day from the engine's own record, while the ledger
    carries whatever day survived the claim it registered. A model that names
    the right result without pinning a day leaves those two ids disagreeing on
    a component the question does not turn on, and the set difference then
    reports an omission for a result that was published in plain sight (seed 11
    day 5: required `medium:None:p13` against published `medium:4:p13`).

    A claimant holds at most one result per (type, target), so ignoring the day
    cannot merge two distinct results. Whether the day itself is right is a
    separate question, and the timeline-conflict check already asks it.
    """
    result_type, _, remainder = result_id.partition(":")
    _, _, target_id = remainder.partition(":")
    return result_type, target_id


def _omitted_result_ids(
    required_ids: tuple[str, ...], published_ids: tuple[str, ...]
) -> tuple[str, ...]:
    """Required results with no matching publication, compared day-agnostically."""
    published = {_normalized_result_key(item) for item in published_ids}
    return tuple(
        sorted(item for item in required_ids if _normalized_result_key(item) not in published)
    )


# How many times the round may re-queue pressured execution candidates that have
# not rebutted yet. Bounded so the speaker selection always terminates.
_MAX_MAJOR_TARGET_SWEEPS = 2

# Consecutive speakers that may yield no output before the segment gives up.
_MAX_CONSECUTIVE_SPEECH_FAILURES = 3

# Cap for a message that only agrees with an existing point, matching the
# "60文字以内の短い同意" instruction in DISCUSSION_OUTPUT_INSTRUCTION.
_REACTION_MAX_CHARS = 60

# Private-channel messages are capped at 100 chars by `generate_wolf_chat`; the
# evaluation harness needs the same number to flag overruns.
_PRIVATE_CHAT_MAX_CHARS = 100

# The closing turn after a summary that put questions to people: whatever
# answers came are in, and the summariser names today's candidate. A stage of
# its own in the context builder (its text, its 60-character cap): it used to
# be built as a second `consensus_summary`, whose text invites one last
# question -- one the round, closing on this turn, would never let anyone answer.
_FINAL_DESIGNATION_STAGE = "final_designation"

# A seat named as a freemason's partner, confirming (or denying) it. Its own
# stage so the turn is told what it is for, whichever way it was scheduled.
_CONFIRMATION_STAGE = "freemason_confirmation"

# From day 2, the short turn each living AI seat with no line yet gets before
# the summary. Its text asks for nothing but what the seat wants to say: the
# turn is the opportunity, what is said in it is the seat's own.
_OPEN_FLOOR_STAGE = "open_floor"

# A seat the closing summary put a question to, answering it. It used to be
# the 30-character reaction for any seat not under pressure, too short for an
# answer the day is about to be decided on.
_SUMMARY_ANSWER_STAGE = "summary_answer"

# The stage a day summary written in code is recorded under.
_CODE_SUMMARY_STAGE = "code"

# A seat asked something by name during the day, answering it. Its own 60-character
# stage: run 14 answered named questions in the 30-character reaction slot, and
# 17 of 20 such lines overran it.
_QUESTION_ANSWER_STAGE = "question_answer"

# How many answers a seat is owed outside the follow-up budget in one day. Each
# is one per batch of questions -- every question put to the seat before it
# answers -- so two seats asking each other back and forth still end.
_MAX_OWED_ANSWERS_PER_SEAT = 2

# The facilitator's second try when the summary named nobody: a candidate,
# then that seat's defence, then the final designation. Once a day.
_PROVISIONAL_STAGE = "provisional_designation"

# The facilitator's turn after the seats it asked by name have answered:
# the floor back to act on the answer. At most `_MAX_FACILITATOR_FOLLOWUPS`
# a day.
_FACILITATOR_FOLLOWUP_STAGE = "facilitator_followup"
_MAX_FACILITATOR_FOLLOWUPS = 2

# The closing lines whose follow-ups the scheduler opens (answers, a defence,
# the designation) instead of queueing ordinary replies.
_SUMMARY_STAGES = ("consensus_summary", _PROVISIONAL_STAGE)

# A wolf-chat sentence that puts a partner up for the ballot, read by
# `_wolf_chat_ballot_plan`. A planned vote on a partner passes the release
# gate, and the two ways of misreading the chat are not equally bad: a false
# "plan" hides exactly the betrayal the gate exists to catch and passes a
# wrong game silently, while a missed plan fails a correct game with the wolf
# chat right beside the failure. So the rule only ever says yes, and only to a
# positive shape: the partner is the object of a ballot word -- only an
# honorific, particles and at most a word of when stand between them -- and
# that word ends the way the team's own proposal or decision ends. A question,
# a report, a prohibition, a refusal, a guess or another seat's ballot is no
# plan because it does not end that way, not because a list of refusals
# happened to hold it.
_SENTENCE_RE = re.compile(r"(?P<body>[^。！？!?\n]+)(?P<end>[。！？!?]*)")
# A sentence that asks rather than proposes, even without ？: 吊ろうか, 一択かな,
# and one that checks rather than decides: 吊りでいいよね, 吊ってもいいよな.
_QUESTION_RE = re.compile(r"(?:か(?:な|ね|も|しら)?|よ[ねな])[ぁぃぅぇぉっー〜～]*$")
_HONORIFIC = r"(?:さん|くん|ちゃん|君)?"
_FINAL_PARTICLES = r"[よねなぜぞわ〜～ー]*"
# How a proposal closes: the sentence ends, or a comma follows that does not go
# on to quote it, take it back or hedge it. Right after the comma: a quoting
# particle, a but or a turn-around (吊ろう、って村が言ってる / 吊ろう、が村の
# 総意 / 吊ろう、けど迷う / 吊ろう、いやミオを吊ろう); anywhere after it: a
# doubt, a guess, a condition or a later day (吊ろう、ただし明日次第 / 吊ろう、
# 多分 / 吊ろう、もし黒が出たら / 吊ってもいい、最終日に).
_AFTER_THE_COMMA_UNDOES = (
    r"\s*(?:と|って|なんて|とか|という|みたい|じゃな|ではな|どう|[がはをの]|も(?!う)"
    r"|けど|けれど|だけど|でも|ただ|しかし|とはいえ|か|いや|やっぱ|やはり)"
    r"|.*?(?:迷|かも|次第|多分|たぶん|おそらく|恐らく|きっと|保留|冗談|嘘|うそ|なんてね"
    r"|たら|なら|もし|仮に|場合"
    r"|最終|最後|明後日|あさって|以降|後で|あとで|いずれ|そのうち|いつか)"
)
_CLOSE = _FINAL_PARTICLES + rf"(?:$|(?=[、,，](?!{_AFTER_THE_COMMA_UNDOES})))"
# A plain form decides only when it ends the sentence, and never with な (吊るな
# forbids it), か (asks it) or よね (checks it): 明日はダイキに入れる。
_DECLARED = r"(?:[ねぜぞ]|よ|わ[よね]?)?[〜～ー]*$"
# What a て-form asks for: 吊ってくれ, 入れてほしい, 切ってもいい, 入れておいて.
_TE_REQUEST = (
    r"(?:くれ|ください|下さい|ほしい|欲しい|おいて"
    r"|も?(?:いい|良い|構わない|かまわない|大丈夫))(?:です)?"
)
# Doing it ahead, which the speaker decides or proposes rather than asks:
# 入れておく, 入れておこう.
_TE_AHEAD = r"(?:おこう|おく|おきます|おきましょう)"
# A て-form may hand the ballot on to the verb that closes the sentence, when
# that verb proposes or asks (切って信用を取ってくれ, 入れて白く見せよう,
# 切って信用を取れ). The words in between must not undo it: a negation, a past
# or ongoing act, a report, a condition, a question, a pretence, a quotation
# (吊って欲しいと村に思わせよう), someone else acting, a relative clause, a
# taunt (俺に入れて負けろ).
_UNDOES = (
    r"な(?:い|く|かっ|けれ)|ません|ず|回避|阻止|防|控え|論外|ありえ|早い|必要|負け|後悔"
    r"|た|て[るいく]|そう|流れ|声|ふり|振り|フリ|つもり|もら|させ|られ"
    r"|ば|なら|ても|でも|けど|けれど|のに|から|ので|か|べき|どう"
    r"|[がはと]|人|奴|やつ|者|方"
)
# The verb that closes such a chain, when it asks: 取ってくれ, 見せろ, 信用を取れ
# -- but not a dare (吊ってみろ), an imperative straight after the て-form.
_CHAIN_ASKS = (
    r"(?:[てで](?:くれ|ください|下さい|ほしい|欲しい)"
    r"|(?<![てで])[いきぎじちにびみりえけげせぜてでねべめれ]ろ|せよ|を[一-龥]{1,2}[えけげせてねべめれ])"
)
# ... or when it proposes: 取ろう, 見せよう.
_CHAIN_PROPOSES = r"(?:(?<!だ)ろう|(?<![るうくすつぬむぶぐ])よう|[こごぼもとお]う|ましょう)"


def _verb_moods(
    volitional: str, te: str, imperative: str, plain: str, negative: str, polite: str
) -> dict[str, re.Pattern[str]]:
    """The endings of one conjugation class that propose, ask or decide a ballot.

    Each argument is how the class spells that form after the verb's stem: 吊-ろ(う),
    吊-って, 吊-れ, 吊-る, 吊-ら(ないと), 吊-り(ましょう).
    """
    must = (
        rf"{negative}(?:ないと|なきゃ|なくちゃ|なければ|なくては)"
        r"(?:いけ(?:ない|ません)|ダメ|だめ|駄目|まずい|なら(?:ない|ん))(?:だ|です)?"
    )
    chain = rf"{te}(?:(?!{_UNDOES})[^、,，]){{0,14}}?"
    request = (
        rf"{te}{_TE_REQUEST}{_CLOSE}|{te}[よねな]*$|(?:{imperative}){_CLOSE}"
        rf"|{chain}{_CHAIN_ASKS}{_CLOSE}"
    )
    return {
        # the team proposes it: 吊ろう, 投票しましょう, 切るしかない, 吊らないといけない
        "propose": re.compile(
            rf"(?:(?:{volitional}う?|{polite}ましょう)(?:と思う|と思います)?"
            rf"|{plain}しか(?:ない|ありません|ねえ)(?:です)?|{must}){_CLOSE}"
            rf"|{negative}(?:ないと|なきゃ|なくちゃ){_DECLARED}"
        ),
        # someone is asked to, or the speaker sees to it: 吊ってくれ, 入れて。,
        # 投票しろ, 切って信用を取れ, 入れておこう, 入れて白く見せよう
        "ask": re.compile(
            rf"{request}|{te}{_TE_AHEAD}{_CLOSE}|{chain}{_CHAIN_PROPOSES}{_CLOSE}"
        ),
        # strictly a request, which is all a pronoun is read with: 俺を切ってくれ,
        # 俺に投票して, 俺を吊れ, お前を切っていいな -- not 俺に入れておく
        "request": re.compile(request),
        # the speaker decides it and the sentence ends there: 明日はダイキに入れる。
        "decide": re.compile(rf"(?:{plain}|{polite}ます){_DECLARED}"),
    }


_WHEN = r"(?:明日は?|今日は?|もう|さっさと|早めに|潔く|素直に)?"
_OBJECT = rf"(?:を|は|も)?{_WHEN}"
_TO = rf"(?:に|への?)[はも]?{_WHEN}"
_SORRY = rf"には(?:悪い|すまない|すまん|申し訳ない)(?:けど|が)[、,，]?{_WHEN}"
# The words that put the seat just before them up for the ballot, by how they
# conjugate, each with the particles (and at most a word of when) that make
# that seat its object: ダイキを吊る, ダイキを明日吊る, ダイキに投票, ダイキ吊り,
# ダイキには悪いけど切る. ダイキはミオに投票 (ダイキ votes), ダイキ、ミオを
# 吊ろう (a vocative), 仕切って and 切り札 match none of them. Bare に乗る is
# left out: in wolf chat it backs the partner's line. The longer phrase comes
# first where a shorter one starts the same way.
_BALLOT_WORDS = {
    "godan_r": rf"の?吊りに乗|への票に乗|(?:{_OBJECT}|{_SORRY})(?:見切|見限|吊|切)",
    "godan_s": rf"(?:{_OBJECT}|{_SORRY})落と",
    "ichidan": (
        rf"(?:{_OBJECT}|{_SORRY})(?:切り捨て|見捨て|捨て)"
        rf"|(?:{_TO}|{_SORRY})(?:一?票を?)?(?:入れ|投じ)"
        rf"|(?:{_TO}|で)(?:一?票|吊り)を?(?:合わせ|揃え|集め|まとめ|固め)"
    ),
    "suru": (
        rf"(?:{_OBJECT}|{_SORRY})(?:身内切り|処刑|犠牲に|吊り(?:候補|先|対象)に)"
        rf"|(?:{_TO}|{_SORRY})投票"
    ),
    "noun": rf"の?(?:身内切り|吊り|切り|処刑)|{_TO}(?:投票|一票)",
}
_MOODS: dict[str, dict[str, re.Pattern[str]]] = {
    "godan_r": _verb_moods("ろ", "って", "れ", "る", "ら", "り"),  # 吊る 切る 乗る
    "godan_s": _verb_moods("そ", "して", "せ", "す", "さ", "し"),  # 落とす
    "ichidan": _verb_moods("よ", "て", "ろ|よ", "る", "", ""),  # 入れる 見捨てる
    "suru": _verb_moods("しよ", "して", "しろ|せよ", "する", "し", "し"),  # 投票する
    "noun": {
        # ダイキ吊りでいこう, ダイキの処刑にしよう, もうダイキ吊りしかない
        "propose": re.compile(
            r"(?:で(?:行こう|いこう|行きましょう|いきましょう)|に(?:しよう|しよ|しましょう)"
            rf"|しか(?:ない|ありません|ねえ)(?:です)?)(?:と思う|と思います)?{_CLOSE}"
        ),
        # ダイキ吊りで決まり, ダイキ切り一択, ダイキに一票。, ダイキ吊りで。
        "decide": re.compile(
            r"(?:(?:で(?:決まり|決定|確定)|一択)(?:だ|です)?|で(?:いい|お願い(?:します)?)?"
            rf"|に(?:決めた|決定|する|します)|お願い(?:します)?)?{_FINAL_PARTICLES}$"
        ),
    },
}
# The partner by name may be proposed, asked for or decided. A pronoun only
# stands for the partner with a request: 俺/私 in the partner's own line
# (俺を切って, 俺に投票して -- not 俺に入れておく, where it votes itself), and
# お前 when the partner alone was addressed (お前を切っていいな -- not お前に
# 投票する). A seat marked by は/も counts only with a proposal (ダイキは切ろう),
# since ダイキは吊って asks ダイキ to vote.
_ANY_MOOD: tuple[str, ...] = ("propose", "ask", "decide")
_REQUESTING: tuple[str, ...] = ("request",)
# The partner's own ballot on another seat (俺はミオに入れる, ミオに投票する,
# 入れておく) is no counter-proposal -- it cannot vote itself; only its
# proposal or request (ミオを吊ろう, ミオに入れてくれ) replaces a plan.
_COUNTER_PROPOSING: tuple[str, ...] = ("propose", "request")
# Who お前 means when the partner alone was addressed, and who 俺 means when
# the partner wrote the line (自分で is "by oneself", not "me").
_SECOND_PERSON = r"(?:お前|おまえ|あなた|あんた|貴方|きみ|(?<![ァ-ヶー一-龥A-Za-z0-9])君)"
_FIRST_PERSON = r"(?:私|わたし|あたし|俺|おれ|オレ|僕|ぼく|ボク|自分(?!で))(?!たち|達|ら)"
# Before the partner, anywhere in the sentence:
# - a subject other than the team: the village or a crowd (村はダイキを吊り
#   たがってる, みんなダイキに投票する -- not みんなでダイキに投票しよう), a
#   third person (あいつはダイキに入れる, あいつ、ダイキに入れる), a role or a
#   rival (占い師はダイキに投票する, 対抗はダイキに入れる), the one addressed
#   (お前はダイキに投票して), and -- added per board -- another seat that is
#   anything but an object, a comparison or a vocative (ミオたちは, ミオだけ,
#   ミオ曰く, ミオの予想だと; see `_SEAT_NOT_A_SUBJECT`);
# - a guess (多分), a reading of the table (票読みだと, 今の流れだと, どうやら),
#   a show for the village (表向きは), an earlier day (昨日) or a quotation;
# - a condition nobody can check yet (黒が出ればダイキに入れよう, 最悪ダイキ
#   を切ろう, 状況次第でダイキを吊ろう, 様子を見てダイキを吊ろう, 決選では),
#   or some later day rather than this ballot (最終的には, いずれ, 最終日には).
_NOT_THE_TEAM = (
    r"村人?(?:は|が|の|も|目線|視点|側|的|だと|では)"
    r"|(?:みんな|皆|全員|誰か|他の人)(?!さん[で、,，]|[で、,，])"
    r"|(?:あいつ|こいつ|そいつ|どいつ|彼女?|奴|やつ|ヤツ)(?:ら|たち|達)?[、,，]"
    r"|(?:あいつ|こいつ|そいつ|どいつ|彼女?|奴|やつ|ヤツ|相方|相手|向こう"
    r"|他の(?:狼|人狼|仲間|奴|やつ)|もう(?:一人|ひとり)"
    r"|占い師?|霊媒師?|霊能者?|共有者?|狩人|騎士|狂人|妖狐|狐|背徳者?|猫又|グレー"
    r"|対抗|過半数|半数|大半|多数"
    rf"|{_SECOND_PERSON}|[人者陣勢組側派])(?:ら|たち|達)?(?:さん)?[はがも]"
    r"|多分|たぶん|おそらく|恐らく|きっと|どうせ|このまま|もし|仮に"
    r"|票読み|予想|予測|見立て|流れ(?:だと|では|的|から|上|で|なら)|どうやら|残念ながら"
    r"|表向き|建前"
    r"|たら|だら|なら|[えけげせてねべめれ]ば|あらば|最悪|場合|次第|によ(?:って|ると|れば|り)"
    r"|(?:時|とき|際)(?:は|に)|万が?一|いざ|あわよくば|保険|決[選戦]"
    r"|見て|読んで|見極め|見計ら|伺って|うかがって|窺って"
    r"|最終(?:的|日|局面)|最後(?:は|に|の)|いずれ|そのうち|いつか|後で|あとで|後々"
    r"|ゆくゆく|明後日|あさって|終盤"
    r"|昨日|前日|さっき|前回|[「『“\"]"
)
# What may follow another seat that stands before the partner without making
# it the one who votes or the one who says so: an object (ミオを噛んで, ミオに
# 黒を出して), a bite or a ballot of its own (ミオ噛み、ミオ吊りは無理だから), a
# comparison or replacement (ミオより, ミオじゃなくて, ミオの代わりに), or a
# vocative (ミオ、ダイキに入れる). Anything else -- が/は/も, たち, だけ, まで,
# って, 曰く, 目線, の予想, によると -- is no plan of the team's.
_SEAT_NOT_A_SUBJECT = (
    r"(?:さん|くん|ちゃん|君)?"
    r"(?:を|に(?![はもよと])|へ|噛|吊|より|じゃ|では?なく|でなく|の代わり|[、,，])"
)
# Groups addressed at the head of a sentence, beside player names.
_GROUP_VOCATIVE = r"(?:みんな|皆|みなさん|二人とも|ふたりとも|お前ら|おまえら)"
# A plan stops being one when a later line unsettles it: it names the
# partner's ballot to doubt or refuse it (俺を切るなよ, ダイキは吊らない,
# 俺を切る？), keeps the partner (ダイキは残そう, ダイキを守ろう), opens on a
# turn-around or an objection (いや、俺はまだ戦える / やっぱやめた / 今のなし /
# 反対), proposes another seat instead (やっぱりミオを吊ろう), or -- written
# by the partner -- objects (嫌だ, 勘弁してくれ, それは困る).
_REFUSAL_RE = re.compile(
    r"な(?:い|く|かっ|けれ|よ|$)|ません|ず|やめ|止め|反対|ダメ|だめ|駄目|避け|待|早い"
    r"|控え|回避|阻止|防|守|論外|ありえ"
)
_KEEP = rf"(?:を|は|も)?{_WHEN}(?:残|守|庇|生か|助け)"
# What may come before the first word of a sentence: an ellipsis, a dash, an
# opening bracket (……なんてね, （嘘）).
_LEAD = r"[\s…‥・ー〜～.．、]*[（(「『【\[]?\s*"
_RETRACTION_RE = re.compile(
    _LEAD + r"(?:いや|やっぱ|やはり|待って|待て|冗談|嘘|うそ|撤回|前言撤回|取り消|今のなし"
    r"|なんてね|な[ー〜～]*んて|なんちゃって|ジョーク|やめ|反対|嫌|やだ|勘弁|断"
    r"|ダメ|だめ|駄目|無理|と言|という|って言)"
)
# What the partner says against a plan to vote it, anywhere in the line: 嫌だ,
# 困る, 勘弁して, ふざけるな, 断る, 待ってくれ, それはない. Its acceptance
# (仕方ない, 了解) and its other news (ミオは占い師じゃない) leave the plan.
_OBJECTION_RE = re.compile(
    r"嫌|やだ|困|勘弁|ふざけ|断[るりわ]|反対|待って|待て|無理|やめ|止め|ダメ|だめ|駄目"
    r"|ありえ|論外|納得(?:でき|いか)|それは(?:ない|無い|ねえ|なし|ナシ)"
)
# A sentence of the same message that only continues the plan's sentence, to
# quote or refuse it: …吊りで。って村が言ってる / …投票。は避けたい /
# …入れる。わけない / …投票する。と見せかけて / …吊ろう！と村人が叫んでる --
# hands it to the village: …吊る。それが村の考えだ -- or laughs it off: …吊ろう。
# 笑 / www. A sentence of its own never opens on a particle (とりあえず,
# とにかく and ところで are not one).
_CONTINUES_RE = re.compile(
    _LEAD + r"(?:って|なんて|と(?!りあえず|にかく|ころで|もかく|もあれ|っとと|りま|ても)"
    r"|ん?な?わけ|みたい|らしい|っぽい|を|は(?![んっい])|が(?![んっ])"
    r"|それ(?:が|は|って)(?:村|みんな|皆|全員)|笑|草|[wｗ]+$)"
)


def _unsettles_ballot_on(sentence: str, seat: str, asked: bool) -> bool:
    """Whether `sentence` doubts, refuses or keeps the seat `seat` matches from a ballot."""
    if re.search(rf"{seat}{_HONORIFIC}{_KEEP}", sentence):
        return True
    return any(
        asked or _REFUSAL_RE.search(sentence, match.end()) is not None
        for words in _BALLOT_WORDS.values()
        for match in re.finditer(rf"{seat}{_HONORIFIC}(?:{words})", sentence)
    )


def _proposes_ballot_on(
    sentence: str, seat: str, moods: tuple[str, ...], not_the_team: re.Pattern[str]
) -> bool:
    """Whether `sentence` proposes or decides a ballot on the seat `seat` matches."""
    for kind, words in _BALLOT_WORDS.items():
        for match in re.finditer(rf"(?P<seat>{seat}{_HONORIFIC})(?:{words})", sentence):
            if not_the_team.search(sentence, 0, match.start()):
                continue
            marker = sentence[match.end("seat") : match.end("seat") + 1]
            allowed = [m for m in moods if marker not in ("は", "も") or m == "propose"]
            rest = sentence[match.end() :]
            if any(m in _MOODS[kind] and _MOODS[kind][m].match(rest) for m in allowed):
                return True
    return False


def _seat_pattern(player_id: str, name: str) -> str:
    """A seat by id or name, matched whole (p1 is not p11), as `mentions_player` does."""
    names = "|".join(re.escape(item) for item in (name, player_id) if item)
    return rf"(?<![0-9A-Za-z])(?:{names})(?![0-9])"


def _leading_vocatives(
    sentence: str, seats: dict[str, re.Pattern[str]], group: re.Pattern[str]
) -> list[str | None]:
    """Who a sentence addresses at its head ("ダイキ、ハルト、…"), in order.

    A seat is its id; a group ("みんな、") is None, so it never reads as one
    partner addressed alone.
    """
    called: list[str | None] = []
    position = len(sentence) - len(sentence.lstrip())
    while True:
        match = next(
            (
                (pid, found)
                for pid, pattern in seats.items()
                if (found := pattern.match(sentence, position)) is not None
            ),
            None,
        )
        if match is not None:
            called.append(match[0])
            position = match[1].end()
            continue
        found_group = group.match(sentence, position)
        if found_group is None:
            return called
        called.append(None)
        position = found_group.end()


@dataclass(eq=False)
class _RoundBook:
    """What the speaker scheduler tracks about one round, beside `DiscussionRoundState`.

    The round state is the session's record, shared with the API layer; these
    only decide who speaks next, so they live with the scheduler. A book is
    bound to its round by identity -- a new day's round starts a new book.
    """

    round_state: DiscussionRoundState
    # Blacks and standing COs when the pressured candidates were last ranked.
    # A new one changes who the table is about to hang, so the ranking is redone.
    accusation_signature: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    # Today's blacks already looked at, as (claimant, result type, target).
    accusations_seen: set[tuple[str, str, str]] = field(default_factory=set)
    # Seats blacked today that have not spoken since: each is owed one turn.
    owed_replies: list[str] = field(default_factory=list)
    roll_call_done: bool = False
    summary_speaker: str | None = None
    # Seats the summary put a question to, each owed one answer before the close.
    summary_answers: list[str] = field(default_factory=list)
    # The summary put a question to the living human, and the round may wait
    # for them: the table stops right after it so they have the floor first.
    summary_asked_human: bool = False
    designation_pending: bool = False
    # The seat the closing line put up for the hang, owed a defence turn.
    defence_target: str | None = None
    # A summary that named nobody: the unheard seats speak, then the
    # facilitator gives a provisional designation. Once a day.
    provisional_used: bool = False
    unheard_after_summary: bool = False
    provisional_pending: bool = False
    # The seats the facilitator's latest line asked by name, still to answer;
    # the follow-ups it has had today; whether one is queued now.
    facilitator_waiting: set[str] = field(default_factory=set)
    facilitator_followups: int = 0
    facilitator_due: bool = False
    # Where the summary's line sits in the chat log: who has spoken since.
    summary_position: int | None = None
    # The final designation named a seat unheard since the summary, which was
    # given its defence and the facilitator one more designation. Once a day.
    redesignation_used: bool = False
    # Seats whose place in the opening order is their partner confirmation:
    # named during the morning, they close the morning block with it.
    confirmation_turns: set[str] = field(default_factory=set)
    # The day's closing line has been given; only a partner confirmation the
    # closing line itself asked for may still follow.
    closed: bool = False
    # The seats offered a silent-seat turn today, drawn once and walked once.
    unheard_plan: list[str] | None = None
    # Answers given outside the budget to seats asked by name, per seat.
    owed_answers_given: dict[str, int] = field(default_factory=dict)


def _unwrap_provider(provider: object) -> object:
    """Follow `.inner` through provider wrappers to the provider that calls the API."""
    seen: set[int] = set()
    while hasattr(provider, "inner") and id(provider) not in seen:
        seen.add(id(provider))
        provider = provider.inner
    return provider


class AICoordinator:
    def __init__(
        self,
        state: GameState,
        ai_player_ids: list[str],
        provider: LLMProvider,
        seed: int | None = None,
        recorder: TranscriptRecorder | None = None,
        observer_player_ids: set[str] | None = None,
        max_discussion_followups: int = 4,
        discussion_segment_size: int = 4,
        pacing_scale: float = 0.0,
        reasoning: ReasoningRuntime | None = None,
        model_decides: bool = False,
    ) -> None:
        self._ai_player_ids = list(ai_player_ids)
        self._observer_player_ids = observer_player_ids or set()
        self._max_discussion_followups = max(0, max_discussion_followups)
        self._segment_size = max(1, discussion_segment_size)
        self._pacing_scale = max(0.0, pacing_scale)
        self._seed = seed
        self._rng = random.Random(seed)
        self._round_book: _RoundBook | None = None
        self._pending_questions = state.pending_questions
        # Pending questions disappear when answered, but their topic must remain in
        # the day's ledger or another AI immediately asks the same thing again.
        self._asked_question_topics: set[tuple[int, str, str]] = {
            (question.day, target_id, question.topic)
            for target_id, questions in state.pending_questions.items()
            for question in questions
            if question.topic
        }
        # Seats named as a freemason's partner, owed their confirmation turn, in
        # the order they were named (never sorted: a seat id says nothing about
        # who should speak first).
        self._forced_partner_confirmations: list[str] = []
        # The part of the last spoken line the model wrote itself (empty for a
        # line code wrote). The closing turns read their candidate from it: a
        # plan-note hang target the line never names is not a designation.
        self._last_model_text = ""
        self._freemason_public_plan: tuple[str, str, bool] | None = None
        self._freemason_death_announced: set[str] = set()
        freemasons = [
            player.player_id
            for player in state.players_by_role(RoleName.FREEMASON)
            if player.player_id in self._ai_player_ids
        ]
        if len(freemasons) == 2:
            plan_rng = random.Random(f"{seed}:freemason-public-plan")
            leader, partner = plan_rng.sample(freemasons, 2)
            self._freemason_public_plan = (leader, partner, plan_rng.random() < 0.5)
        self._metrics = getattr(provider, "_metrics", None)
        # Present in v2 and v3. In v2 votes, night actions and the speaking
        # order are decided in code and the model is asked for wording alone.
        self.reasoning = reasoning
        # v3: the runtime still supplies facts, hard logic, validation and the
        # speaking order, but whom to suspect, how to vote and what to do at
        # night are the model's call, made from a board memo instead of a
        # brief. `model_decides` without a runtime is just legacy.
        self.model_decides = bool(model_decides and reasoning is not None)
        self._engine = (
            "v3" if self.model_decides else ("v2" if reasoning is not None else "legacy")
        )
        # Every state-consistency repair and every say-one-thing-vote-another
        # discrepancy lands here, so a game can be audited after the fact
        # without the AI layer having to be re-run.
        self.validation = ValidationLog()

        self._personalities = assign_personalities(self._ai_player_ids, seed=seed)

        wolf_ids = [p.player_id for p in state.players_by_role(RoleName.WEREWOLF)]
        madman = state.players_by_role(RoleName.MADMAN)
        wolf_deception = assign_wolf_deception(wolf_ids, seed=seed)
        madman_strategy_name, madman_fake_role = assign_madman_strategy(seed=seed)
        fake_claim_guard = FakeClaimGuard(wolf_team_ids=set(wolf_ids))
        self._wolf_deception = wolf_deception
        self._madman_fake_role_by_player = {
            player.player_id: madman_fake_role for player in madman if madman_fake_role is not None
        }

        self._day_summaries = DaySummaryManager()
        self._context = ContextBuilder(
            personalities=self._personalities,
            day_summaries=self._day_summaries,
            wolf_deception=wolf_deception,
            madman_fake_role=madman_fake_role if madman else None,
            fake_claim_guard=fake_claim_guard,
            observer_player_ids=self._observer_player_ids,
            engine=self._engine,
        )

        # v3 seats speak through the compact chat wire schema: the full
        # `DiscussionOutput` exceeds Anthropic's structured-output limits, so
        # sending it costs one rejected request per game and then runs without
        # grammar enforcement (schema embedded in the prompt, parsed leniently).
        # v2 and legacy still send it; see approach-reset §11. Agents hand back a
        # `DiscussionOutput` either way, so nothing below changes.
        self._agents: dict[str, AIPlayerAgent] = {
            pid: AIPlayerAgent(
                provider,
                self._personalities[pid],
                chat_contract=self.model_decides,
                metrics=self._metrics,
                seat_id=pid,
            )
            for pid in self._ai_player_ids
        }

        self._recorder = recorder
        if recorder is not None:
            transcript = getattr(recorder, "transcript", None)
            if transcript is not None:
                transcript.game_id = state.session_id
            recorder.set_roster(
                names={pid: p.name for pid, p in state.players.items()},
                roles={pid: p.role.value for pid, p in state.players.items()},
                teams={pid: p.team.value for pid, p in state.players.items()},
                personalities={pid: p.name for pid, p in self._personalities.items()},
                deception={
                    "wolf_pattern": wolf_deception.pattern_name,
                    "wolf_pattern_label": wolf_deception.pattern_label,
                    "fake_role_by_player": {
                        **{
                            pid: role.value
                            for pid, role in wolf_deception.fake_role_by_player.items()
                        },
                        **{
                            pid: role.value
                            for pid, role in self._madman_fake_role_by_player.items()
                        },
                    },
                    "lurking_player_ids": list(wolf_deception.lurking_player_ids),
                    "madman_strategy": madman_strategy_name if madman else None,
                },
                seed=seed,
                # Wrappers (the session/evaluation budget, prompt capture) keep
                # the real provider as `inner`; the transcript names the model
                # vendor, not the wrapper.
                provider=type(_unwrap_provider(provider)).__name__,
            )

    @property
    def _decides_in_code(self) -> bool:
        """v2: the belief engine fixes targets, votes and night actions."""
        return self.reasoning is not None and not self.model_decides

    def _record_event(self, name: str) -> None:
        """Count a named event on the provider's metrics collector, if it keeps any.

        Looked up rather than assumed: a collector without `record_event`, or a
        provider with no collector at all, leaves the event uncounted instead of
        turning a bookkeeping gap into a crash in the middle of a game.
        """
        record = getattr(self._metrics, "record_event", None)
        if callable(record):
            record(name)

    def _memo_for(self, state: GameState, player_id: str) -> str | None:
        """The seat's board memo under v3; None for the engines that do not use one."""
        if not self.model_decides or self.reasoning is None:
            return None
        return self.reasoning.board_memo(state, player_id)

    def _required_lines(
        self, state: GameState, player_id: str, decision: DiscussionDecision
    ) -> list[str]:
        """Code-decided obligations for this turn, in plain words for the memo prompt."""
        lines: list[str] = []
        if decision.required_claim_role is not None:
            lines.append(f"{ROLE_DEFINITIONS[decision.required_claim_role].label_ja}CO")
        for item in decision.required_public_results:
            ability = "霊媒" if item.result_type == "medium" else "占い"
            when = f"{item.referenced_day}日目{'処刑' if item.result_type == 'medium' else '夜'}"
            colour = "黒" if item.is_werewolf else "白"
            name = state.players[item.target_id].name
            lines.append(f"{ability}結果を公開する: {name}={colour}（{when}のもの）")
        owed = self._owed_claimed_result_line(state, player_id, decision)
        if owed:
            lines.append(owed)
        return lines

    def _owed_claimed_result_line(
        self, state: GameState, player_id: str, decision: DiscussionDecision
    ) -> str:
        """The morning result a standing claim still owes, when no real one stands behind it.

        A real holder's result is already in the decision, verdict and all. A
        fake claimant has no result for code to hand over -- whom it "looked
        at" and what it "saw" are its own call -- but its claim still owes the
        table this morning's report, and the duty turn it now gets is for
        saying it. Phrased as the claim's obligation, so dropping the claim
        stays open; a story that has already collapsed is not asked to go on.
        """
        if self.reasoning is None or decision.required_public_results:
            return ""
        if not self.reasoning.owes_morning_result(state, player_id):
            return ""
        seat = self.reasoning.seats.get(player_id)
        if (
            seat is not None
            and seat.deception is not None
            and seat.deception.status is StoryStatus.COLLAPSED
        ):
            return ""
        ledger = PublicFactLedger(state)
        yesterday = state.day - 1
        if ledger.claimed_role_of(player_id) is RoleName.MEDIUM:
            executed = "、".join(
                ledger.name_of(pid)
                for pid in ledger.executed_ids()
                if ledger.execution_day(pid) == yesterday
            )
            return f"霊媒COを続けるなら、{yesterday}日目に処刑された{executed}の霊媒結果を出す"
        return f"占いCOを続けるなら、{yesterday}日目夜の占い結果(誰を占って白か黒か)を出す"

    def _deception_role(self, player_id: str) -> str | None:
        if player_id in self._wolf_deception.fake_role_by_player:
            return f"fake_{self._wolf_deception.fake_role_by_player[player_id].value}"
        if player_id in self._wolf_deception.lurking_player_ids:
            return "lurker"
        if player_id in self._madman_fake_role_by_player:
            return f"fake_{self._madman_fake_role_by_player[player_id].value}"
        return None

    def _record(
        self,
        state: GameState,
        player_id: str,
        kind: str,
        *,
        text: str = "",
        target: str | None = None,
        reasoning_memo: object = None,
        used_fallback: bool = False,
        public_claim_role: str | None = None,
        public_results: list[dict[str, object]] | None = None,
        directed_question_targets: list[str] | None = None,
        ready_to_vote: bool | None = None,
        effective_length_limit: int | None = None,
        model_text_chars: int | None = None,
        key_point: str = "",
        agrees_with: list[str] | None = None,
        decision_evidence: str = "",
        countercase: str = "",
        alternative_target: str | None = None,
        reply_to: str | None = None,
        stage: str | None = None,
    ) -> None:
        if self._recorder is None:
            return
        player = state.players[player_id]
        memo = reasoning_memo.model_dump() if reasoning_memo is not None else None  # type: ignore[attr-defined]
        self._recorder.record(
            Utterance(
                day=state.day,
                phase=state.phase.value,
                kind=kind,
                player_id=player_id,
                player_name=player.name,
                role=player.role.value,
                team=player.team.value,
                personality=self._personalities[player_id].name,
                deception_role=self._deception_role(player_id),
                text=text,
                target=target,
                reasoning_memo=memo,
                public_claim_role=public_claim_role,
                public_results=public_results or [],
                directed_question_targets=directed_question_targets or [],
                ready_to_vote=ready_to_vote,
                used_fallback=used_fallback,
                effective_length_limit=effective_length_limit,
                model_text_chars=model_text_chars,
                key_point=key_point,
                agrees_with=agrees_with or [],
                decision_evidence=decision_evidence,
                countercase=countercase,
                alternative_target=alternative_target,
                reply_to=reply_to,
                stage=stage,
            )
        )

    def _find_ai_with_role(self, state: GameState, role: RoleName) -> str | None:
        for pid in self._ai_player_ids:
            player = state.players[pid]
            if player.alive and player.role == role:
                return pid
        return None

    # -- discussion --

    async def run_discussion_round(self, session: object) -> None:
        """Compatibility wrapper: advance segments until the round completes.

        Offline/evaluation callers use this method and must never pause for a
        human. Interactive routes call :meth:`advance_discussion` directly.
        """
        while session.controller.state.phase == Phase.DISCUSSION:  # type: ignore[attr-defined]
            await self.advance_discussion(session, allow_human_pause=False)
            round_state = getattr(session, "discussion_round", None)
            if round_state is None or round_state.complete:
                return

    async def advance_discussion(self, session: object, *, allow_human_pause: bool = True) -> None:
        """Run at most one discussion segment while holding the session lock."""
        async with session.discussion_lock:  # type: ignore[attr-defined]
            controller = session.controller  # type: ignore[attr-defined]
            state = controller.state
            round_state = getattr(session, "discussion_round", None)
            if state.phase != Phase.DISCUSSION:
                if round_state is not None and round_state.day == state.day:
                    # DISCUSSION opens once a day, so today's round is over, not
                    # paused. Left open, it reads as a morning still to drain to
                    # any caller that loops until the round settles.
                    round_state.complete = True
                return
            if round_state is None or round_state.day != state.day:
                round_state = await self._start_discussion_round(state)
                session.discussion_round = round_state  # type: ignore[attr-defined]

            human_id = getattr(session, "human_id", "")
            human = state.players.get(human_id)
            can_pause = bool(
                allow_human_pause
                and human
                and human.alive
                and human_id not in self._observer_player_ids
            )
            if (
                can_pause
                and getattr(session, "discussion_paused", False)
                and not getattr(session, "discussion_step_budget", None)
            ):
                self._set_awaiting(round_state, True)
                return
            if round_state.awaiting_human and can_pause:
                return
            self._set_awaiting(round_state, False)
            if self._close_immediate_stage(round_state) and can_pause:
                self._set_awaiting(round_state, True)
                return

            spoken = 0
            # A speaker that never produces output must not be retried forever:
            # `_speak` returns None both when the phase left DISCUSSION mid-segment
            # (without awaiting, so a retry loop would never yield to the event
            # loop) and when generation fails outright.
            consecutive_failures = 0
            stalled = False
            while spoken < self._segment_size and not round_state.complete:
                if state.phase != Phase.DISCUSSION or round_state.day != state.day:
                    # The table moved on while a turn was generating. The phase
                    # check alone is not enough: a vote, a night and a new
                    # morning can all pass inside one slow request, DISCUSSION is
                    # open again, and this segment would go on speaking in
                    # yesterday's order ahead of the new morning's duty speakers.
                    # Closing the round also keeps a caller looping until it
                    # settles from spinning on one that can no longer move.
                    round_state.complete = True
                    return
                pid, stage = self._next_discussion_speaker(state, round_state)
                if pid is None:
                    round_state.complete = True
                    break
                output = await self._speak(controller, state, pid, stage)
                if output is None:
                    consecutive_failures += 1
                    # A failed turn still used its slot: the last morning duty
                    # speaker failing ends the morning segment just the same.
                    finished_immediate = self._close_immediate_stage(round_state)
                    if consecutive_failures >= _MAX_CONSECUTIVE_SPEECH_FAILURES:
                        stalled = True
                        break
                    if finished_immediate and can_pause:
                        self._set_awaiting(round_state, True)
                        break
                    continue
                consecutive_failures = 0
                round_state.outputs.append((pid, output))
                # A partner named during the morning closes the morning block
                # with its confirmation; one named later takes the next turn
                # (`_take_partner_confirmation`). Placed before the morning is
                # closed below, so a naming by the last duty speaker still
                # lands inside the block. These used to be moved into the
                # reply queue right here, so the forced turn never ran and
                # day 1's confirmation came thirteen lines late.
                self._place_morning_confirmations(state, round_state, self._book_for(round_state))
                round_state.speech_counts[pid] = round_state.speech_counts.get(pid, 0) + 1
                spoken += 1
                # Closed before the pause checks below, so a pause that lands on
                # the last morning speaker is also the morning's pause -- not one
                # more wait sprung on the human right after they resume.
                finished_immediate = self._close_immediate_stage(round_state)
                step_budget = getattr(session, "discussion_step_budget", None)
                if step_budget is not None:
                    session.discussion_step_budget = max(0, step_budget - 1)  # type: ignore[attr-defined]
                if getattr(session, "discussion_pause_requested", False):
                    session.discussion_pause_requested = False  # type: ignore[attr-defined]
                    session.discussion_paused = True  # type: ignore[attr-defined]
                if getattr(session, "discussion_step_budget", None) == 0:
                    session.discussion_paused = True  # type: ignore[attr-defined]
                # What the closing turns open or close is settled before any
                # pause is honoured. A step or a pause that landed on the
                # summary used to stop the loop before its follow-ups were
                # queued, and the next segment found nothing left to say.
                waits_for_human = False
                if stage in _SUMMARY_STAGES:
                    # A summary that asks the candidate for a last word used
                    # to close the round before the question was queued, so
                    # the answer never came (five of six days in the first
                    # Claude game). Its questions are answered first now.
                    if not self._open_summary_followups(
                        state,
                        round_state,
                        pid,
                        output,
                        waiting_human=human_id if can_pause else None,
                        provisional=stage == _PROVISIONAL_STAGE,
                    ) and self._close_round(state, round_state):
                        break
                    # A question to the human is theirs to answer before any
                    # AI answer or the designation: the floor goes to them now.
                    waits_for_human = self._book_for(round_state).summary_asked_human
                elif stage == _FINAL_DESIGNATION_STAGE:
                    if any(
                        question.target_id in state.players
                        and question.target_id != pid
                        and question.question.strip()
                        for question in output.directed_questions
                    ):
                        # Told not to ask, it asked anyway. The round closes
                        # here regardless -- one designation, never a new
                        # exchange -- so the miss is counted, not looped on.
                        self._record_event("designation_question_unanswered")
                    if not self._open_redesignation(
                        state, round_state, pid, output
                    ) and self._close_round(state, round_state):
                        break
                else:
                    # Settled before the pause check too: a step pauses after
                    # every turn, so queued here after it, no question asked
                    # in step mode ever got its answer turn that day.
                    self._queue_turn_replies(state, round_state, pid, output)
                if can_pause and (waits_for_human or getattr(session, "discussion_paused", False)):
                    self._set_awaiting(round_state, True)
                    break
                # The first segment contains only morning-immediate speakers.
                if finished_immediate and can_pause:
                    self._set_awaiting(round_state, True)
                    break
            if stalled:
                # Nobody could be made to speak. Without a wait the client has no
                # countdown to run, so nothing would ask for the next segment and
                # the table would sit silent until the human typed something.
                # Waiting hands control back: the countdown's auto-pass retries.
                self._record_event("discussion_stalled")
                if can_pause:
                    self._set_awaiting(round_state, True)
            elif (
                can_pause
                and round_state.stage != "immediate"
                and not round_state.complete
                and spoken >= self._segment_size
            ):
                self._set_awaiting(round_state, True)

    @staticmethod
    def _close_immediate_stage(round_state: DiscussionRoundState) -> bool:
        """Leave the morning-immediate stage once its speakers have had their turn.

        Returns True only on the call that makes the transition. Bounded by the
        order's length as well as `immediate_count`: a count that ever outran
        the order -- it once counted a seat with two duties twice -- would
        otherwise pin the stage at "immediate", and a living human would never
        be given the floor.
        """
        if round_state.stage != "immediate":
            return False
        if round_state.cursor < min(round_state.immediate_count, len(round_state.order)):
            return False
        round_state.stage = "initial_view"
        return True

    async def _start_discussion_round(self, state: GameState) -> DiscussionRoundState:
        alive = [pid for pid in self._ai_player_ids if state.players[pid].alive]
        if self.reasoning is not None:
            return self._coded_discussion_round(state, alive)
        shuffled = list(alive)
        self._rng.shuffle(shuffled)
        intents = await asyncio.gather(*(self._morning_intent(state, pid) for pid in shuffled))
        priority = {"immediate": 0, "after_results": 1, "normal": 2, "hold": 3}
        intent_by_id = dict(zip(shuffled, intents, strict=True))
        order = sorted(
            shuffled,
            key=lambda pid: (
                priority.get(intent_by_id[pid].timing, 2),
                0 if intent_by_id[pid].intent == "publish_result" else 1,
            ),
        )
        immediate_count = sum(intent_by_id[pid].timing == "immediate" for pid in order)
        return DiscussionRoundState(
            day=state.day,
            order=order,
            stage="immediate",
            immediate_count=immediate_count,
            max_total=max(len(alive) + self._max_discussion_followups, int(len(alive) * 2.5)),
        )

    def _coded_discussion_round(self, state: GameState, alive: list[str]) -> DiscussionRoundState:
        """Open the day with the seats that have something to say.

        Everyone forms an opinion -- the belief engine updates all of them -- but
        a morning where sixteen AIs each state a view is a wall of text nobody
        reads. Duty speakers (a held result, a planned CO, a direct question)
        are never dropped, and up to three seats without a CO open beside
        them; the rest speak when answered or challenged, in day 1's roll
        call, or in the short turn every silent seat gets from day 2 on.
        """
        assert self.reasoning is not None
        self.reasoning.refresh(state)
        holders = [pid for pid in alive if self.reasoning.holds_unpublished_result(state, pid)]
        answering = [pid for pid in alive if self._open_questions(pid)]
        # Every seat with a claim to make, change or correct -- the freemason
        # leader, a result holder without its CO, a planned fake claimant, a
        # bluffer whose story collapsed. Only the freemason leader used to be
        # here, so a planned seer counter-CO could simply never get a turn.
        planned_fake_roles = {
            **self._wolf_deception.fake_role_by_player,
            **self._madman_fake_role_by_player,
        }
        planned_claims = [
            pid
            for pid in self.reasoning.required_claim_speakers(
                state,
                planned_fake_roles=planned_fake_roles,
                freemason_leader=(
                    self._freemason_public_plan[0]
                    if self._freemason_public_plan is not None
                    else None
                ),
            )
            if pid in alive
        ]
        # One turn per seat, however many duties it holds. A result holder
        # without its CO is also a required claimant, and counting it twice made
        # `immediate_count` outrun the speakers actually in the order.
        #
        # The results and the COs come in an order drawn per game and day.
        # They used to come result holders first, planned claimants after: the
        # real seer opened day 1 in all three games of run 14, and on day 2 the
        # real medium reported before the fakes -- an order that says who is
        # real. The seats answering yesterday's questions follow, and a
        # freemason's own report (the hidden partner's takeover CO, the
        # leader's word that the partner died) closes the block, where the
        # table expects it, as a named partner's confirmation does.
        order_rng = random.Random(f"{self._seed}:morning:{state.day}")
        reports = [pid for pid in alive if self._owes_freemason_report(state, pid)]
        speakers = [
            pid for pid in dict.fromkeys([*holders, *planned_claims]) if pid not in reports
        ]
        order_rng.shuffle(speakers)
        answering = [pid for pid in answering if pid not in speakers and pid not in reports]
        order_rng.shuffle(answering)
        duty = [*speakers, *answering, *reports]
        # The duty speakers, then up to three seats without a CO. The three
        # used to share a cap of five with the duty speakers, so a morning
        # with three results and two planned COs heard no gray seat at all.
        chosen = self.reasoning.select_opening_speakers(
            state,
            pending_question_targets=[pid for pid in alive if self._open_questions(pid)],
            planned_claims=planned_claims,
            exclude=[*duty, *self._observer_player_ids],
        )
        order = list(dict.fromkeys(duty + chosen))
        return DiscussionRoundState(
            day=state.day,
            order=order,
            stage="immediate",
            immediate_count=len(duty),
            max_total=max(len(alive) + self._max_discussion_followups, len(order) * 3),
        )

    def _next_discussion_speaker(
        self, state: GameState, round_state: DiscussionRoundState
    ) -> tuple[str | None, str]:
        if self.reasoning is not None:
            reassessing = self.reasoning.take_reassessment_speakers()
            # Handed over sorted by seat id. Who answers a correction first is
            # drawn per game instead, so p10 never outranks p2 by spelling.
            random.Random(
                f"{self._seed}:reassessment:{state.day}:{len(round_state.outputs)}"
            ).shuffle(reassessing)
            for player_id in reassessing:
                self._round_queue_reply(state, round_state, player_id)
        book = self._book_for(round_state)
        confirming = self._take_partner_confirmation(state, round_state, book)
        if confirming is not None:
            return confirming, _CONFIRMATION_STAGE
        if book.closed:
            return None, "consensus_summary"
        if round_state.summary_done:
            # The wrap-up has been given. Only what it opened is left: its
            # addressees' answers, anyone it left freshly accused, the close.
            return self._closing_speaker(state, round_state, book)
        if round_state.cursor >= len(round_state.order):
            self._extend_roll_call(state, round_state, book)
        if round_state.cursor < len(round_state.order):
            index = round_state.cursor
            pid = round_state.order[index]
            round_state.cursor += 1
            if pid in book.confirmation_turns:
                book.confirmation_turns.discard(pid)
                return pid, _CONFIRMATION_STAGE
            morning = round_state.stage == "immediate" and index < round_state.immediate_count
            return pid, "immediate" if morning else "initial_view"
        self._refresh_major_targets(state, round_state, book)
        accused = self._take_owed_reply(state, round_state, book)
        if accused is not None:
            return accused, "rebuttal_or_reassessment"
        # Every branch below must change round state before returning. A branch that
        # returns the same speaker without advancing anything spins forever when
        # `_speak` yields nothing -- and the phase-changed path of `_speak` returns
        # without awaiting, so such a spin would never release the event loop.
        while True:
            while round_state.reply_queue and len(round_state.outputs) < round_state.max_total:
                pid = round_state.reply_queue.pop(0)
                round_state.queued.discard(pid)
                if book.facilitator_due and pid == self._facilitator(state):
                    book.facilitator_due = False
                    return pid, _FACILITATOR_FOLLOWUP_STAGE
                if not self._may_schedule(state, pid):
                    continue
                # No per-seat cap. It used to be the persona's talkativeness
                # (round(2.5 x talkativeness) replies), and a quiet villager
                # asked five times by name reached it, answered only after the
                # summary, and was hanged 13-1 for his silence (run 14, seed
                # 2, day 2). A persona sets tone and amount in words; whether
                # to say more is the seat's own call (`needs_another_statement`),
                # and the round's total keeps the day bounded.
                if pid in round_state.major_targets and round_state.speech_counts.get(pid, 0) < 2:
                    return pid, "rebuttal_or_reassessment"
                if self._asked_today(state, pid):
                    return pid, _QUESTION_ANSWER_STAGE
                return (
                    pid,
                    "reaction" if self._rng.random() < 0.35 else "rebuttal_or_reassessment",
                )
            # The queue is empty, but the most-pressured execution candidates still
            # owe the table a rebuttal. Re-queue them rather than returning them
            # directly, so the pop above stays the single source of progress.
            if round_state.major_target_sweeps >= _MAX_MAJOR_TARGET_SWEEPS:
                break
            owed = [
                target
                for target in round_state.major_targets
                if state.players[target].alive and round_state.speech_counts.get(target, 0) < 2
            ]
            if not owed or len(round_state.outputs) >= round_state.max_total:
                break
            round_state.major_target_sweeps += 1
            for target in owed:
                self._round_queue_reply(state, round_state, target)
        # A seat asked something by name today answers before the close, even
        # once the replies have used the budget: one turn for every question
        # put to it before it speaks, at most `_MAX_OWED_ANSWERS_PER_SEAT` such
        # turns a day.
        answering = self._take_owed_answer(state, round_state, book)
        if answering is not None:
            return answering, _QUESTION_ANSWER_STAGE
        if state.day >= 2:
            # Before the close, every seat that has not spoken today speaks once.
            unheard = self._take_unheard_seat(state, round_state, book)
            if unheard is not None:
                return unheard, _OPEN_FLOOR_STAGE
        # No seat is appointed to argue against the majority any more (the
        # "minority review" turn): when the talk piles onto one person, the
        # pushback should come from whoever at the table feels it, as it does
        # between people. Every seat has had a turn by now to be that voice.
        # The summary is the day's closing line: it opens the designation, the
        # defence and the close, so a day the replies used up still gets it --
        # given outside the budget and counted, as an owed answer is. Gating it
        # on the budget closed a sixth of the days with nobody named at all.
        #
        # Only a facilitator gives it: a living freemason claimant. With none,
        # the day has no summary and no designation -- the floor has had its
        # turns, and the vote follows. The fallback that used to appoint one
        # (the last seat ready to vote, else the first in the day's order) made
        # a wolf's fake claimant the summariser in all three games of run 14,
        # and its lines reached every ballot as 【進行役の方針】. Who leads
        # after that is for the table to settle in talk; code does not pick.
        if not round_state.summary_done:
            round_state.summary_done = True
            leader = self._facilitator(state)
            if leader is not None:
                if len(round_state.outputs) >= round_state.max_total:
                    self._grant_extra_turn(round_state)
                book.summary_speaker = leader
                return leader, "consensus_summary"
            self._record_event("summary_without_speaker")
        return None, "consensus_summary"

    def _may_schedule(self, state: GameState, player_id: str) -> bool:
        """A living seat this coordinator speaks for: never the human, never an observer."""
        return (
            player_id in self._agents
            and player_id not in self._observer_player_ids
            and player_id in state.players
            and state.players[player_id].alive
        )

    @staticmethod
    def _in_morning_block(round_state: DiscussionRoundState) -> bool:
        """The morning's duty speakers are still speaking (or the last just spoke)."""
        return (
            round_state.stage == "immediate"
            and round_state.immediate_count > 0
            and round_state.cursor <= min(round_state.immediate_count, len(round_state.order))
        )

    def _place_morning_confirmations(
        self, state: GameState, round_state: DiscussionRoundState, book: _RoundBook
    ) -> None:
        """A partner named during the morning confirms at the end of the morning block.

        Day 2's results arrive together in the block, so the confirmation goes
        after them, not between them: at the block's end, which it then
        closes. A partner still due to speak in the block confirms on that
        turn instead of taking a second one. Outside the morning this does
        nothing; `_take_partner_confirmation` gives the next turn instead.
        """
        if not self._in_morning_block(round_state):
            return
        named, self._forced_partner_confirmations = self._forced_partner_confirmations, []
        for target in named:
            if not self._may_schedule(state, target):
                continue
            book.confirmation_turns.add(target)
            if target in round_state.order[round_state.cursor : round_state.immediate_count]:
                continue
            try:
                later = round_state.order.index(target, round_state.immediate_count)
            except ValueError:
                # A turn the day did not have: counted, so progress keeps its total.
                self._grant_extra_turn(round_state)
            else:
                round_state.order.pop(later)
            round_state.order.insert(round_state.immediate_count, target)
            round_state.immediate_count += 1

    def _take_partner_confirmation(
        self, state: GameState, round_state: DiscussionRoundState, book: _RoundBook
    ) -> str | None:
        """The named partner, if one is owed its confirmation: it takes the next turn.

        Whatever line named it -- an opening view, a reply, the summary, the
        final designation -- the table hears the partner next, before the
        summary's answers or the vote. A confirmation is one fixed sentence
        and settles a public fact the vote should not start without, so it
        is never deferred to the next day. A seat still waiting in the order
        gives up that later place for it.
        """
        self._place_morning_confirmations(state, round_state, book)
        while self._forced_partner_confirmations:
            target = self._forced_partner_confirmations.pop(0)
            if not self._may_schedule(state, target):
                continue
            try:
                pending_index = round_state.order.index(target, round_state.cursor)
            except ValueError:
                self._grant_extra_turn(round_state)
            else:
                round_state.order.pop(pending_index)
                if pending_index < round_state.immediate_count:
                    # A duty speaker taking its turn early. Left counted, the
                    # morning stage would wait for one more seat than it holds.
                    round_state.immediate_count -= 1
            return target
        return None

    def _close_round(self, state: GameState, round_state: DiscussionRoundState) -> bool:
        """End the day after its closing line. False while a partner confirmation is owed.

        The closing line (the summary or the final designation) may itself
        name a freemason's partner; its confirmation is the one turn that can
        still follow it, and the round closes after that.
        """
        book = self._book_for(round_state)
        book.closed = True
        if any(self._may_schedule(state, pid) for pid in self._forced_partner_confirmations):
            return False
        round_state.complete = True
        return True

    def _book_for(self, round_state: DiscussionRoundState) -> _RoundBook:
        """The scheduler's notes on this round, started fresh for a new one."""
        book = self._round_book
        if book is None or book.round_state is not round_state:
            book = _RoundBook(round_state=round_state)
            self._round_book = book
        return book

    def _extend_roll_call(
        self, state: GameState, round_state: DiscussionRoundState, book: _RoundBook
    ) -> None:
        """Day 1: every AI seat that has had no turn yet gets one first-impression turn.

        The opening is the duty speakers plus up to three seats, and the
        rest were to speak when answered or challenged. On day 1 nobody has a
        result or a reputation to be challenged on, so seven gray seats in the
        first Claude game never spoke at all -- and the day's votes went to
        "the one who stayed quiet", a silence the scheduler had imposed.
        Appended once, after the opening order, before any rebuttal sweep; the
        follow-up budget grows by the same count so the extra turns do not eat
        the replies the day would otherwise have had.

        Every seat counts, not only grays: on day 1 a seat that has not spoken
        holds no CO, since a CO is made by speaking.
        """
        if book.roll_call_done or state.day != 1:
            return
        book.roll_call_done = True
        spoken_today = {
            message.author_id
            for message in state.chat_log
            if message.day == state.day and message.channel is ChatChannel.PUBLIC
        }
        had_turn = (
            set(round_state.order)
            | set(round_state.speech_counts)
            | set(round_state.reply_queue)
            | spoken_today
        )
        silent = [
            pid
            for pid in self._ai_player_ids
            if pid not in had_turn
            and state.players[pid].alive
            and pid not in self._observer_player_ids
        ]
        if not silent:
            return
        random.Random(f"{self._seed}:roll-call:{state.day}").shuffle(silent)
        round_state.order.extend(silent)
        round_state.max_total += len(silent)

    def _take_unheard_seat(
        self, state: GameState, round_state: DiscussionRoundState, book: _RoundBook
    ) -> str | None:
        """The next living AI seat with no line today, for one short turn.

        Day 1 has its roll call; later days had nothing like it, and in seed 1
        37 of 83 ballots on days 2-5 came from seats that had no turn that
        day. Each seat is offered once a day, in an order drawn per game and
        day, and only if it still has no line when its turn comes -- a seat
        that spoke meanwhile (answering a question, say) is passed over. A
        seat whose turn fails is not offered again. So this ends after at
        most one turn per living seat. Each turn is outside the follow-up
        budget and counted in the round's total, so the progress counter
        never reads past it. The human and observers are never offered one.
        """
        if book.unheard_plan is None:
            plan = [pid for pid in self._ai_player_ids if self._may_schedule(state, pid)]
            random.Random(f"{self._seed}:unheard:{state.day}").shuffle(plan)
            book.unheard_plan = plan
        spoken_today = {
            message.author_id
            for message in state.chat_log
            if message.day == state.day and message.channel is ChatChannel.PUBLIC
        }
        while book.unheard_plan:
            pid = book.unheard_plan.pop(0)
            if (
                not self._may_schedule(state, pid)
                or pid in spoken_today
                or round_state.speech_counts.get(pid, 0)
            ):
                continue
            if pid in round_state.queued:
                # One turn, not this one and a queued reply right after it.
                round_state.queued.discard(pid)
                round_state.reply_queue.remove(pid)
            self._grant_extra_turn(round_state)
            return pid
        return None

    @staticmethod
    def _accusation_signature(state: GameState) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The published blacks and standing COs: what re-points the table's pressure."""
        ledger = PublicFactLedger(state)
        blacks = tuple(
            sorted(
                f"{r.claimant_id}:{r.result_type}:{r.target_id}"
                for r in ledger.public_results()
                if r.is_werewolf
            )
        )
        claims = tuple(
            sorted(f"{c.player_id}:{c.claimed_role.value}" for c in ledger.co_declarations())
        )
        return blacks, claims

    def _pressured_targets(
        self, state: GameState, round_state: DiscussionRoundState
    ) -> list[str]:
        """The two seats today's latest plan notes most want hanged.

        Restricted to seats this coordinator speaks for: the pressure list
        drives speaking limits and the rebuttal sweep, neither of which can
        apply to the human.
        """
        latest_targets = {
            speaker: target
            for speaker, output in round_state.outputs
            if isinstance((target := output.reasoning_memo.execution_target), str)
            and target in self._agents
            and state.players[target].alive
        }
        pressure: Counter[str] = Counter(latest_targets.values())
        return [target for target, _count in pressure.most_common(2)]

    def _refresh_major_targets(
        self, state: GameState, round_state: DiscussionRoundState, book: _RoundBook
    ) -> None:
        """Rank the pressured candidates once, and again whenever a black or a CO lands.

        It used to be ranked exactly once, when the opening order ran out.
        A black published after that never reached the list: on day 4 of the
        first Claude game the madman's black on ツムギ came after the ranking,
        カイト (the only name in the plan notes until then) was queued twice,
        and the summary closed the day with ツムギ never heard. A re-rank
        keeps what was already on the list and adds what is new; a seat that
        joins it with fewer than two turns is queued for its rebuttal, as on
        the first ranking.

        Blacks are also handled on their own, every time (see
        `_collect_owed_replies`): the accused is owed an answer whatever the
        plan notes say about them.
        """
        signature = self._accusation_signature(state)
        previous_signature = book.accusation_signature
        book.accusation_signature = signature
        # A round that arrives already ranked, with no record of the board it
        # was ranked on, takes the board as it is now for its baseline.
        if not round_state.major_targets_ready or previous_signature not in (None, signature):
            first = not round_state.major_targets_ready
            known = list(round_state.major_targets)
            pressured = self._pressured_targets(state, round_state)
            round_state.major_targets = list(dict.fromkeys([*known, *pressured]))
            round_state.major_targets_ready = True
            for target in pressured:
                if (first or target not in known) and round_state.speech_counts.get(
                    target, 0
                ) < 2:
                    self._round_queue_reply(state, round_state, target)
        self._collect_owed_replies(state, round_state, book)

    def _collect_owed_replies(
        self, state: GameState, round_state: DiscussionRoundState, book: _RoundBook
    ) -> None:
        """Owe one turn to every AI seat blacked today that has not spoken since.

        A black is the strongest accusation the table hears; a human player
        given one always answers it -- a counter-CO, a defence -- before the
        vote. Each black is looked at once, so a seat that answered (or whose
        owed turn was spent, even on a failed generation) is not called back
        for the same accusation.

        Today's blacks are read as the projection dates them: a black said
        again today without naming a night is the one published on an earlier
        day, and its target was owed its answer then. Only a black first
        stated today -- or restated for a night of its own, which is a new
        statement -- opens the floor.
        """
        positions = {message.message_id: index for index, message in enumerate(state.chat_log)}
        for result in ObservationSet.from_state(state).verdicts:
            if not result.is_werewolf or result.day != state.day:
                continue
            key = (result.claimant_id, result.result_type, result.target_id)
            if key in book.accusations_seen:
                continue
            book.accusations_seen.add(key)
            target = result.target_id
            if target not in self._agents or not state.players[target].alive:
                continue
            since = positions.get(result.source_message_id, -1)
            answered = any(
                message.author_id == target
                and message.channel is ChatChannel.PUBLIC
                and message.day == state.day
                for message in state.chat_log[since + 1 :]
            )
            if answered:
                continue
            if target not in round_state.major_targets:
                round_state.major_targets.append(target)
            if target not in book.owed_replies:
                book.owed_replies.append(target)

    @staticmethod
    def _grant_extra_turn(round_state: DiscussionRoundState) -> None:
        """Count a turn given outside the follow-up budget in the round's total.

        The total is what the human's progress counter reads against. An owed
        answer or a closing turn is handed out whatever the budget says, so
        leaving it uncounted let the counter read past its total (31/29), and
        an owed answer used to eat one of the replies the day would otherwise
        have had -- the roll call grows the total for the same reason.
        """
        round_state.max_total += 1

    def _take_owed_reply(
        self, state: GameState, round_state: DiscussionRoundState, book: _RoundBook
    ) -> str | None:
        """The next accused seat owed its answer, taken off every queue it was on."""
        while book.owed_replies:
            pid = book.owed_replies.pop(0)
            if pid in self._agents and state.players[pid].alive:
                # One turn answers the accusation and whatever else had queued
                # the seat; two back-to-back turns would only repeat it.
                if pid in round_state.queued:
                    round_state.queued.discard(pid)
                    round_state.reply_queue.remove(pid)
                self._grant_extra_turn(round_state)
                return pid
        return None

    def _asked_today(self, state: GameState, player_id: str) -> bool:
        """The seat has a question put to it by name today that it has not had a turn to answer."""
        return any(question.day == state.day for question in self._open_questions(player_id))

    def _take_owed_answer(
        self, state: GameState, round_state: DiscussionRoundState, book: _RoundBook
    ) -> str | None:
        """The next AI seat asked by name today and not yet given a turn to answer.

        The reply queue answers questions while the budget lasts; this is for
        the rest, so a seat is never hanged for a silence the scheduler
        imposed (run 14, seed 2: マコト was asked five times and answered only
        after the summary). The seat asked first answers first. Every
        question put to a seat before it speaks is answered in one turn (its
        prompt shows them all), and a seat gets at most
        `_MAX_OWED_ANSWERS_PER_SEAT` such turns a day, so questions passed
        back and forth cannot hold the day open. Each turn is outside the
        follow-up budget and counted in the round's total. The human and
        observers are never scheduled.
        """
        asked: list[tuple[int, str]] = []
        positions = {message.message_id: index for index, message in enumerate(state.chat_log)}
        for pid in self._ai_player_ids:
            if not self._may_schedule(state, pid):
                continue
            if book.owed_answers_given.get(pid, 0) >= _MAX_OWED_ANSWERS_PER_SEAT:
                continue
            open_today = [q for q in self._open_questions(pid) if q.day == state.day]
            if open_today:
                first = min(positions.get(q.source_message_id, len(positions)) for q in open_today)
                asked.append((first, pid))
        if not asked:
            return None
        _first, pid = min(asked)
        book.owed_answers_given[pid] = book.owed_answers_given.get(pid, 0) + 1
        if pid in round_state.queued:
            round_state.queued.discard(pid)
            round_state.reply_queue.remove(pid)
        self._grant_extra_turn(round_state)
        return pid

    def _open_summary_followups(
        self,
        state: GameState,
        round_state: DiscussionRoundState,
        speaker_id: str,
        output: DiscussionOutput,
        *,
        waiting_human: str | None = None,
        provisional: bool = False,
    ) -> bool:
        """Queue what a closing line opened. False when there is nothing to wait for.

        Each AI seat the summary put a question to answers once, in the order
        asked, and the summariser then closes with a short designation. A
        black the summary itself published is owed its answer the same way.
        The human is never scheduled. `waiting_human` is the living human the
        round may wait for (None when it may not): a question to them is
        noted, so the round gives them the floor before the AI answers and
        the designation, and the designation follows their turn even when
        no AI seat was asked.

        A summary that names neither a candidate nor anyone in its questions
        no longer ends the day (seed 1, day 2: 「皆さんは気になるグレーの方は
        いますか…？」 asked nobody by name, and the vote began). The seats
        still unheard speak, then the facilitator gives a provisional
        designation (`provisional`): its candidate defends, then the final
        designation. Once a day -- a provisional designation that names
        nobody either closes the day.

        The candidate a closing line names (its plan note's hang target) is
        owed a defence turn whether or not it was asked anything: a seat
        first named in the closing lines used to go to the vote unheard,
        though the facilitator doctrine says to hear the candidate before
        the final designation. Asked and named, it still answers once. A
        living human named this way is given the floor, as when asked.

        A closing line that declares a grey-random day (「今日はグレランで」)
        names nobody on purpose: the seats it asked still answer, but no
        candidate defends -- a plan-note lean is not a designation then --
        and there is no final designation. The vote follows. A line that
        says グレラン and designates in the same breath (「グレランで、仮指定
        はダイキ」) is a designation; one that only names a seat -- to hear its
        answer, or the summariser's own lean -- is not: see
        `reasoning_analyzer.declares_grey_random`.

        The candidate is the plan note's hang target only where the line
        itself names that seat (`_named_closing_target`): a lean the table
        never heard used to hand an unnamed seat a defence turn.
        """
        book = self._book_for(round_state)
        book.summary_speaker = speaker_id
        if not provisional:
            book.summary_position = len(state.chat_log)
        named = dict.fromkeys(
            question.target_id
            for question in output.directed_questions
            if question.target_id in state.players
            and question.target_id != speaker_id
            and state.players[question.target_id].alive
            and question.question.strip()
        )
        target = self._named_closing_target(
            state, speaker_id, output.reasoning_memo.execution_target
        )
        if declares_grey_random(output.public_message):
            book.defence_target = None
            book.summary_answers = [pid for pid in named if self._may_schedule(state, pid)]
            book.summary_asked_human = waiting_human is not None and waiting_human in named
            self._collect_owed_replies(state, round_state, book)
            book.designation_pending = False
            return bool(book.summary_answers or book.owed_replies or book.summary_asked_human)
        if target is None and not named and not provisional and not book.provisional_used:
            book.provisional_used = True
            book.unheard_after_summary = True
            book.provisional_pending = True
            book.summary_answers = []
            book.summary_asked_human = False
            self._collect_owed_replies(state, round_state, book)
            return True
        # The candidate defends first, then the others asked answer.
        defence = target if target is not None and self._may_schedule(state, target) else None
        book.defence_target = defence
        answers = [defence] if defence is not None else []
        answers += [pid for pid in named if self._may_schedule(state, pid)]
        book.summary_answers = list(dict.fromkeys(answers))
        book.summary_asked_human = waiting_human is not None and (
            waiting_human in named or waiting_human == target
        )
        self._collect_owed_replies(state, round_state, book)
        book.designation_pending = bool(
            book.summary_answers or book.owed_replies or book.summary_asked_human
        )
        return book.designation_pending

    def _named_closing_target(
        self, state: GameState, speaker_id: str, target: object
    ) -> str | None:
        """The closing line's candidate: its plan note's hang target, if the line names it.

        The note is the speaker's private lean; the table only hears the line.
        A target the model's own words do not name (whole, as the vote prompt
        reads names) -- or a line code wrote, which has no model words -- is
        no candidate, so nobody is scheduled to defend against an accusation
        nobody made.
        """
        if not (
            isinstance(target, str)
            and target != speaker_id
            and target in state.players
            and state.players[target].alive
        ):
            return None
        names = {pid: player.name for pid, player in state.players.items()}
        if target not in named_seat_ids(self._last_model_text, names):
            return None
        return target

    def _open_redesignation(
        self,
        state: GameState,
        round_state: DiscussionRoundState,
        speaker_id: str,
        output: DiscussionOutput,
    ) -> bool:
        """The safety net: a final designation on a seat that never spoke after the summary.

        Seed 1's designations could name a seat first raised in the closing
        lines, and that seat went to the vote without a word. If the final
        designation's candidate is a living AI seat that has had no line
        since the summary, it gets one defence turn and the facilitator one
        more designation. Once a day, so the second designation closes the
        day whomever it names. The human is never scheduled for it.
        """
        book = self._book_for(round_state)
        if book.redesignation_used:
            return False
        target = self._named_closing_target(
            state, speaker_id, output.reasoning_memo.execution_target
        )
        if target is None or not self._may_schedule(state, target):
            return False
        since = book.summary_position if book.summary_position is not None else 0
        if any(
            message.author_id == target
            and message.channel is ChatChannel.PUBLIC
            and message.day == state.day
            for message in state.chat_log[since:]
        ):
            return False
        book.redesignation_used = True
        book.summary_speaker = speaker_id
        book.defence_target = target
        book.summary_answers = [target]
        book.designation_pending = True
        return True

    def _closing_speaker(
        self, state: GameState, round_state: DiscussionRoundState, book: _RoundBook
    ) -> tuple[str | None, str]:
        """After the summary: its addressees, anyone still owed an answer, then the close.

        Bounded on purpose -- each addressee once, each accusation once, each
        unheard seat once, one provisional designation, one designation --
        and every pick removes itself before it is returned, so a seat whose
        generation fails cannot hold the close open.
        """
        self._collect_owed_replies(state, round_state, book)
        while book.summary_answers or book.owed_replies:
            pending = book.summary_answers or book.owed_replies
            pid = pending.pop(0)
            if pid in book.owed_replies:
                book.owed_replies.remove(pid)
            if pid in self._agents and state.players[pid].alive:
                defends = pid in round_state.major_targets or pid == book.defence_target
                self._grant_extra_turn(round_state)
                return pid, "rebuttal_or_reassessment" if defends else _SUMMARY_ANSWER_STAGE
        if book.unheard_after_summary:
            unheard = self._take_unheard_seat(state, round_state, book)
            if unheard is not None:
                return unheard, _OPEN_FLOOR_STAGE
            book.unheard_after_summary = False
        if book.provisional_pending:
            book.provisional_pending = False
            speaker = book.summary_speaker
            if speaker is not None and self._may_schedule(state, speaker):
                self._grant_extra_turn(round_state)
                return speaker, _PROVISIONAL_STAGE
        if book.designation_pending:
            book.designation_pending = False
            speaker = book.summary_speaker
            if speaker is not None and speaker in self._agents and state.players[speaker].alive:
                self._grant_extra_turn(round_state)
                return speaker, _FINAL_DESIGNATION_STAGE
        return None, "consensus_summary"

    def _queue_turn_replies(
        self,
        state: GameState,
        round_state: DiscussionRoundState,
        speaker_id: str,
        output: DiscussionOutput,
    ) -> None:
        """What an ordinary turn opens: an answer from each seat it asked, its own next word.

        And the facilitator's next word: once every seat its latest line asked
        by name has spoken, it is queued at the head of the replies to act on
        the answers (seed 1, m48 and m52: asked, answered, and no turn left
        to do anything with it). At most twice a day, before the summary.
        """
        for question in output.directed_questions:
            self._round_queue_reply(state, round_state, question.target_id)
        # The seat says whether it has more to say; its persona does not get
        # to overrule it (it used to, at odds of talkativeness / 1.5).
        if output.needs_another_statement:
            self._round_queue_reply(state, round_state, speaker_id)
        facilitator = self._facilitator(state)
        if facilitator is None or round_state.summary_done:
            return
        book = self._book_for(round_state)
        if speaker_id == facilitator:
            book.facilitator_waiting = {
                question.target_id
                for question in output.directed_questions
                if question.target_id != speaker_id
                and question.question.strip()
                and self._may_schedule(state, question.target_id)
            }
            return
        if speaker_id not in book.facilitator_waiting:
            return
        book.facilitator_waiting.discard(speaker_id)
        if book.facilitator_waiting or book.facilitator_followups >= _MAX_FACILITATOR_FOLLOWUPS:
            return
        book.facilitator_followups += 1
        book.facilitator_due = True
        if facilitator in round_state.queued:
            round_state.reply_queue.remove(facilitator)
        else:
            round_state.queued.add(facilitator)
            self._grant_extra_turn(round_state)
        round_state.reply_queue.insert(0, facilitator)

    def _facilitator(self, state: GameState) -> str | None:
        """The living AI seat that leads the day: the first standing freemason CO.

        The only seat the summary and the designations fall to. With none,
        the day has no facilitator (owner's decision, 2026-11-18): a seat
        ready to vote, a result holder or a medium is not appointed in its
        place. One may still come to lead because the table asks it to in
        talk; that is the table's to settle, not the scheduler's.
        """
        return next(
            (
                claim.player_id
                for claim in state.co_declarations
                if claim.claimed_role == RoleName.FREEMASON
                and claim.player_id in self._agents
                and state.players[claim.player_id].alive
            ),
            None,
        )

    def _round_queue_reply(
        self, state: GameState, round_state: DiscussionRoundState, target_id: str
    ) -> None:
        # Only seats this coordinator can speak for. AIs routinely question the
        # human or name them as an execution target, and scheduling that seat used
        # to reach `self._personalities[human_id]` and raise KeyError -- which,
        # inside the fire-and-forget discussion task, silently killed the round.
        if (
            target_id in self._agents
            and state.players[target_id].alive
            and target_id not in round_state.queued
        ):
            round_state.reply_queue.append(target_id)
            round_state.queued.add(target_id)

    def resume_after_human(
        self,
        session: object,
        reply_to: str | None = None,
        references: list[str] | None = None,
        *,
        release_wait: bool = True,
    ) -> None:
        """Release the human pause and prioritize the referenced AI/questioner."""
        round_state = getattr(session, "discussion_round", None)
        if round_state is None:
            return
        if release_wait:
            self._set_awaiting(round_state, False)
        for message_id in dict.fromkeys([reply_to, *(references or [])]):
            if not message_id:
                continue
            message = next(
                (m for m in session.controller.state.chat_log if m.message_id == message_id),  # type: ignore[attr-defined]
                None,
            )
            if message and message.author_id in self._agents:
                self._round_queue_reply(session.controller.state, round_state, message.author_id)  # type: ignore[attr-defined]

    def _open_questions(self, player_id: str) -> list[PendingQuestion]:
        """Questions this seat has not yet had its turn to answer."""
        return [q for q in self._pending_questions.get(player_id, []) if not q.served]

    @staticmethod
    def _question_topic(question: str) -> str:
        if any(word in question for word in ("処刑", "吊り", "狼候補", "怪しい")):
            return "execution_candidate"
        if "狐" in question:
            return "fox_candidate"
        if any(word in question for word in ("CO理由", "占い理由", "選んだ理由")):
            return "claim_reason"
        if any(word in question for word in ("時系列", "順番", "先に")):
            return "timeline"
        return ""

    @staticmethod
    def _set_awaiting(round_state: DiscussionRoundState, awaiting: bool) -> None:
        round_state.awaiting_human = awaiting
        round_state.awaiting_since = time.time() if awaiting else None

    async def _morning_intent(self, state: GameState, player_id: str) -> MorningIntentOutput:
        if self.reasoning is not None:
            output = self._coded_morning_intent(state, player_id)
        else:
            system, messages = self._context.build_morning_intent_context(state, player_id)
            output = await self._agents[player_id].generate_morning_intent(system, messages)
        player = state.players[player_id]
        if self._freemason_public_plan is not None and player.role == RoleName.FREEMASON:
            leader, partner, full_reveal = self._freemason_public_plan
            leader_alive = state.players[leader].alive
            partner_alive = state.players[partner].alive
            already_claimed = any(claim.player_id == player_id for claim in state.co_declarations)
            if player_id == leader and not already_claimed:
                output.timing = "after_results"
                output.intent = "claim"
                output.public_claim_role = RoleName.FREEMASON.value
            elif player_id == partner and not already_claimed:
                under_black = any(
                    claim.target_id == player_id and claim.is_werewolf
                    for claim in state.public_result_claims
                )
                if leader_alive and not under_black:
                    output.timing = "hold"
                    output.intent = "normal"
                    output.public_claim_role = None
                else:
                    output.timing = "immediate"
                    output.intent = "claim"
                    output.public_claim_role = RoleName.FREEMASON.value
            elif player_id == leader and already_claimed and not partner_alive and not full_reveal:
                output.timing = "immediate"
                output.intent = "lead"
        has_divine = any(r.seer_id == player_id for r in state.divine_records)
        has_medium = any(r.medium_id == player_id for r in state.medium_records)
        if (player.role == RoleName.SEER and has_divine) or (
            player.role == RoleName.MEDIUM and has_medium
        ):
            output.timing = "immediate"
            output.intent = "publish_result"
            output.public_claim_role = player.role.value
        return output

    def _coded_morning_intent(self, state: GameState, player_id: str) -> MorningIntentOutput:
        """Who speaks first is a scheduling question, not a language one.

        Holding an unpublished result outranks everything; the rest follows the
        speech-value ranking the runtime already computes. One request per AI
        per morning disappears.
        """
        assert self.reasoning is not None
        player = state.players[player_id]
        has_result = any(r.seer_id == player_id for r in state.divine_records) or any(
            r.medium_id == player_id for r in state.medium_records
        )
        if has_result and player.role in (RoleName.SEER, RoleName.MEDIUM):
            return MorningIntentOutput(
                timing="immediate",
                intent="publish_result",
                public_claim_role=player.role.value,
                priority_reason="未公開の能力結果を持つ",
            )
        candidates = {item.player_id: item for item in self.reasoning.speech_candidates(state)}
        candidate = candidates.get(player_id)
        if candidate is None or candidate.value <= 0:
            return MorningIntentOutput(timing="hold", intent="normal")
        return MorningIntentOutput(
            timing="normal" if candidate.value < 2.5 else "after_results",
            intent="lead" if candidate.value >= 2.5 else "normal",
            priority_reason="、".join(candidate.reasons),
        )

    async def _speak(
        self, controller: object, state: GameState, player_id: str, stage: str
    ) -> DiscussionOutput | None:
        if state.phase != Phase.DISCUSSION:
            return None
        # What this turn's prompt will put to the seat. Only these are served by
        # it: a question asked while the model is generating has not been seen.
        shown_questions = self._open_questions(player_id)
        decision = None
        if self.reasoning is not None:
            decision = self.reasoning.discussion_decision(
                state,
                player_id,
                pending_question=bool(self._open_questions(player_id)),
                under_pressure=stage.startswith("rebuttal"),
            )
        freemason_opening = self._freemason_opening(state, player_id)
        freemason_must_hide = self._freemason_must_hide(state, player_id)
        pending_relation = self._pending_partner_relation(state, player_id)
        # Lines code writes whole: a freemason CO (the plan's opening, the
        # hidden partner's reveal) and a named partner's confirmation. The
        # model's text for them was replaced word for word, so it is not asked
        # for one -- two requests a game that bought nothing. The death report
        # of an already-claimed leader is left on the old path: it lands on
        # whatever turn the leader has next, which may be the summary.
        code_writes_line = pending_relation is not None or (
            freemason_opening is not None
            and not any(claim.player_id == player_id for claim in state.co_declarations)
        )
        # The turn belongs to this phase of this day. Generation can take long
        # enough for the table to move on, and the engine's `chat` checks only
        # that the speaker is alive -- so a line written for the discussion
        # would otherwise land in the middle of the vote.
        epoch = (state.phase, state.day)
        brief_turn = False
        output: DiscussionOutput | None
        if code_writes_line:
            output = DiscussionOutput(public_message="")
        else:
            if self.model_decides and self.reasoning is not None and decision is not None:
                # v3: the board as a memo, plus what this turn must say. No brief --
                # the conclusions are what the model is being asked for.
                system, messages = self._context.build_discussion_context(
                    state,
                    player_id,
                    stage,
                    board_memo=self.reasoning.board_memo(state, player_id),
                    required_lines=self._required_lines(state, player_id, decision),
                )
            else:
                system, messages = self._context.build_discussion_context(
                    state, player_id, stage
                )
                if decision is not None:
                    messages = [
                        *messages[:-1],
                        Message(role="user", content=decision.render_brief()),
                        messages[-1],
                    ]
            if freemason_opening is not None:
                system += (
                    "\n\n【共有公開計画】この発言では指定された共有CO文を最初にそのまま述べてください。"
                )
            elif freemason_must_hide:
                system += (
                    "\n\n【共有公開計画】あなたは潜伏側です。共有CO、相方の名前、共有者だと"
                    "推測できる表現を公開発言に絶対に含めないでください。"
                )
            controller.set_typing(player_id, True)  # type: ignore[attr-defined]
            agent = self._agents[player_id]
            try:
                output = await agent.generate_discussion(system, messages)
            finally:
                controller.set_typing(player_id, False)  # type: ignore[attr-defined]
            # The brief fallback asks for the sentence alone, so like a turn written
            # in code it carries no notes: an empty memo and no target, by omission.
            brief_turn = output is not None and agent.last_discussion_was_brief
            if self._metrics is not None:
                self._metrics.record_discussion_result(skipped=output is None)
        owes_coded_line = decision is not None and self._owes_coded_line(
            state, player_id, decision, freemason_opening
        )
        if (state.phase, state.day) != epoch or not state.players[player_id].alive:
            # Dropped unseen: nothing is posted, and nothing is recorded as said.
            if output is not None or owes_coded_line:
                self._record_event("speech_discarded_after_phase_end")
            return None
        posted_by_code = False
        if output is None:
            if not owes_coded_line:
                return None
            # Generation failed, but what this turn owes the table was never the
            # model's to write: the results, the CO behind them and the
            # freemason lines are fixed in code below and given their canonical
            # sentences by `ensure_fact_sentences`. Going silent here is how a
            # seer whose request failed lost its result for the whole day.
            output = DiscussionOutput(public_message="")
            posted_by_code = True
        notes_missing = posted_by_code or brief_turn or code_writes_line
        # Set when code replaces the whole line, so its length is not the model's.
        line_replaced_by_code = False
        model_requested_target = output.reasoning_memo.execution_target
        if freemason_opening is not None:
            output.public_message = freemason_opening
            output.public_claim_role = RoleName.FREEMASON.value
            output.contains_co_claim = True
            if "死亡" in freemason_opening:
                self._freemason_death_announced.add(player_id)
        elif freemason_must_hide:
            # Every route to a CO is closed, not only the prose. A claim set in
            # the structured fields alone is drafted into a ROLE_CLAIM and given
            # its canonical sentence below, so checking the sentence let the
            # code meant to keep the partner hidden be the thing that outed it.
            if detect_claimed_role(output.public_message) is not None:
                output.public_message = "現時点ではCOしません。既出の判定と灰の発言を比較します。"
                line_replaced_by_code = True
            output.public_claim_role = None
            output.contains_co_claim = False
            output.claim_action = None
        valid_reassessments = [
            item
            for item in output.reassessments
            if item.player_id in state.players and item.player_id != player_id
        ]
        output.reassessments = valid_reassessments
        if any(
            item.changed_mind
            and item.player_id == output.reasoning_memo.execution_target
            and not item.remaining_reason.strip()
            for item in valid_reassessments
        ):
            output.reasoning_memo.execution_target = None
        # Nothing below may read an unvalidated target. A dead execution
        # candidate, a flipped verdict or a medium result about someone who was
        # never executed is state corruption, not a human-style misread, and it
        # is repaired deterministically here before it reaches the engine.
        self._validate_output(state, player_id, output)
        enforced_results: tuple[PublicResultClaim, ...] = ()
        if decision is not None:
            if self._decides_in_code:
                # v2: the model rendered the turn; it does not get to revise the
                # conclusion. Two reasoning systems running side by side is what
                # let an AI argue one name all day and then vote for another.
                # v3 keeps the model's own (validated) target instead, and the
                # vote is checked against what the seat said.
                output.reasoning_memo.execution_target = decision.execution_target
                output.alternative_execution_target = decision.alternative_target
            if decision.required_public_results:
                # The decision already says which results go out and at which
                # night. The model's own `public_results` is replaced, not merged:
                # a model that forgot the field used to make a result vanish,
                # and one that invented an extra would publish a result the
                # seat never held.
                required = [
                    PublicResultClaim(
                        result_type=item.result_type,
                        target_id=item.target_id,
                        is_werewolf=item.is_werewolf,
                        referenced_day=item.referenced_day,
                    )
                    for item in decision.required_public_results
                ]
                output.public_results = required
                if decision.required_claim_role is not None and output.claim_action is None:
                    # A result needs a claim behind it. Without one the table
                    # reads a verdict from nobody in particular, and the
                    # validator drops a result that contradicts a standing claim.
                    output.public_claim_role = decision.required_claim_role.value
                    output.contains_co_claim = True
                # Only what the engine actually demanded is a requirement. A
                # result the model volunteers on some later turn is speech, not
                # an obligation, and must not be audited as one.
                enforced_results = tuple(required)
        if pending_relation is not None:
            claimant_label = self._spoken_label(state, pending_relation.claimant_id)
            output.public_message = (
                f"共有者CO。{claimant_label}の相方は"
                f"私{self._spoken_label(state, player_id)}で間違いありません。"
            )
            output.public_claim_role = RoleName.FREEMASON.value
            asked = [
                question
                for question in self._pending_questions.get(player_id, [])
                if question.asker == pending_relation.claimant_id
            ]
            # The line that named this seat, not some earlier question from the
            # same claimant.
            source_question = next(
                (q for q in reversed(asked) if q.topic == "freemason_confirmation"),
                asked[0] if asked else None,
            )
            if source_question is not None:
                output.reply_to = source_question.source_message_id
        elif stage == _CONFIRMATION_STAGE and output.reply_to is None:
            # A naming the code cannot confirm for the seat (one side is no
            # freemason): the model answers, and its answer still points at
            # the line that named it.
            naming = next(
                (
                    question
                    for question in reversed(self._pending_questions.get(player_id, []))
                    if question.topic == "freemason_confirmation"
                ),
                None,
            )
            if naming is not None:
                output.reply_to = naming.source_message_id
        if output.agrees_with and not output.key_point.strip():
            # Agreement with no new argument is a reaction, not an analysis. Cut at a
            # sentence boundary so the shortened line still reads as finished Japanese.
            output.public_message = truncate_at_sentence(output.public_message, _REACTION_MAX_CHARS)
        if not notes_missing:
            # A turn written in code, or answered through the brief contract,
            # carries no notes. Storing its empty memo would wipe the last thing
            # the model actually thought.
            self._context.set_reasoning_memo(
                player_id, output.reasoning_memo.model_dump(), day=state.day, stage=stage
            )
        # The claims this turn publishes are decided before it is spoken, so the
        # message can be made to state them. Dropping a declared verdict because
        # the prose forgot to name its target is how a result silently vanishes.
        drafts = build_claim_drafts(
            output,
            PublicFactLedger(state),
            speaker_id=player_id,
            names_only=self.model_decides,
        )
        # What the model itself wrote, measured before code puts the canonical
        # CO and result sentences in front of it: the stage cap and the register
        # are the model's to answer for, the prefix is not. A turn code wrote
        # whole -- a failed generation's obligations, the freemason lines, the
        # personality's canned fallback -- has no model part at all.
        model_text_chars = (
            0
            if posted_by_code
            or line_replaced_by_code
            or freemason_opening is not None
            or pending_relation is not None
            or self._is_fallback(player_id, output.public_message)
            else len(output.public_message)
        )
        model_text = output.public_message if model_text_chars else ""
        self._last_model_text = model_text
        output.public_message = ensure_fact_sentences(
            output.public_message,
            drafts,
            PublicFactLedger(state),
            speaker_id=player_id,
        )
        if decision is not None and self._decides_in_code:
            # Last change to the text, on purpose. Anything that ran after it --
            # the freemason confirmation overwrite did -- could drop the decided
            # candidate and leave the table hearing a different one. v3 injects
            # no sentence: a human never opens every line with a candidate.
            output.public_message = enforce_execution_target(
                output.public_message,
                decision.execution_target,
                PublicFactLedger(state),
                speaker_id=player_id,
            )
        if not output.public_message.strip():
            # Nothing survived to be said -- a code-written turn whose only
            # result validation rejected, say. An empty bubble is not a turn.
            return None
        # Only questions the line puts aloud are recorded, wait for an answer
        # or book a reply: the turn returns this list to the round.
        output.directed_questions = self._questions_said_aloud(
            state, output.directed_questions, model_text
        )
        self._record(
            state,
            player_id,
            "discussion",
            text=output.public_message,
            reasoning_memo=output.reasoning_memo,
            used_fallback=posted_by_code or self._is_fallback(player_id, output.public_message),
            public_claim_role=output.public_claim_role,
            public_results=[item.model_dump() for item in output.public_results],
            directed_question_targets=[item.target_id for item in output.directed_questions],
            ready_to_vote=output.ready_to_vote,
            # v3: the cap this turn's prompt named for its stage, so an overrun
            # counts what the model was told, not the hard cut at 160.
            effective_length_limit=(
                chat_stage_max_chars(stage)
                if self.model_decides
                else discussion_length_range(self._personalities[player_id].verbosity)[1]
            ),
            model_text_chars=model_text_chars,
            stage=stage,
            key_point=output.key_point,
            agrees_with=output.agrees_with,
            reply_to=output.reply_to,
        )
        try:
            message_id = controller.chat(  # type: ignore[attr-defined]
                player_id,
                output.public_message,
                "public",
                output.reply_to,
                output.quote,
            )
        except GameError:
            # The engine refused the message (the speaker died mid-round, say).
            # Nothing was displayed, so nothing is recorded as said.
            return None
        if posted_by_code:
            self._record_event("obligation_posted_by_code")
        for shown in shown_questions:
            shown.served = True
        displayed_target: str | None = None
        decided_target = decision.execution_target if decision is not None else None
        if decision is not None and self.reasoning is not None:
            if self.model_decides:
                # No canonical sentence to read back under v3: the model's
                # validated structured target is what the seat committed to,
                # and the ballot is checked against it.
                displayed_target = output.reasoning_memo.execution_target
                decided_target = displayed_target
            else:
                # Read back from the string the table saw, not copied from the
                # decision: that is the only way the two can be caught disagreeing.
                displayed_target = displayed_execution_target(
                    output.public_message, PublicFactLedger(state)
                )
            if not (notes_missing and displayed_target is None):
                # A v3 turn written in code or through the brief contract names
                # no candidate because nobody asked the model for one, not
                # because the seat dropped its earlier one; recording None would
                # erase what it did say today.
                self.reasoning.record_stated_target(player_id, displayed_target, state.day)
        required_ids: tuple[str, ...] = ()
        if self._recorder is not None and decision is not None and self.reasoning is not None:
            belief = self.reasoning.seats[player_id].belief
            public_records = belief.public_argument_evidence_for(decided_target)
            private_records = tuple(
                record
                for record in belief.active_evidence()
                if record.visibility is EvidenceVisibility.PRIVATE_REASONING
            )
            team_records = tuple(
                record
                for record in belief.active_evidence()
                if record.visibility is EvidenceVisibility.TEAM_PRIVATE
            )
            required_ids = tuple(
                f"{item.result_type}:{item.referenced_day}:{item.target_id}"
                for item in enforced_results
            )
            self._recorder.record_decision_audit(
                DecisionAuditRecord(
                    game_id=state.session_id,
                    day=state.day,
                    phase=state.phase.value,
                    player_id=player_id,
                    decision_target=decided_target,
                    displayed_target=displayed_target,
                    public_evidence_ids=tuple(r.evidence_id for r in public_records),
                    private_evidence_ids=tuple(r.evidence_id for r in private_records),
                    team_private_evidence_ids=tuple(r.evidence_id for r in team_records),
                    required_public_result_ids=required_ids,
                    published_result_ids=(),
                    model_requested_target=model_requested_target,
                    model_target_was_overridden=(model_requested_target != decided_target),
                    active_evidence_ids=tuple(r.evidence_id for r in belief.active_evidence()),
                    publicly_emitted_evidence_ids=tuple(r.evidence_id for r in public_records),
                    attempted_public_evidence_ids=(
                        belief.state.reasons_for(decided_target) if decided_target else ()
                    ),
                    brief_public_evidence_ids=tuple(r.evidence_id for r in public_records),
                    emitted_public_evidence_ids=tuple(r.evidence_id for r in public_records),
                    board_version=self.reasoning.observations.board_version,
                    public_result_ids_at_decision=self._public_result_ids(state),
                    correction_ids_at_decision=tuple(
                        sorted(
                            {
                                item.correction_id
                                for item in self._recorder.transcript.correction_audits
                            }
                        )
                    ),
                    votable_ids_at_decision=tuple(state.votable_ids(player_id)),
                    target_alive=(
                        decided_target is None or state.players[decided_target].alive
                    ),
                )
            )
        if self._pacing_scale:
            base = 0.35 + len(output.public_message) * 0.012
            delay = min(3.0, base) * self._pacing_scale * self._rng.uniform(0.8, 1.2)
            await asyncio.sleep(delay)
        names = {pid: player.name for pid, player in state.players.items()}
        for question in output.directed_questions:
            # What the target is shown is what the asker said to it: the
            # sentence of its line that names the target, not the structured
            # field's paraphrase (43 of 44 in seed 1 differed from the line).
            said_aloud = sentence_naming(model_text, question.target_id, names)
            if question.target_id not in state.players or not said_aloud:
                continue
            proposed_topic = question.topic.strip()
            topic = (
                proposed_topic
                if proposed_topic
                in {"execution_candidate", "fox_candidate", "claim_reason", "timeline"}
                else self._question_topic(question.question.strip() or said_aloud)
            )
            existing = self._pending_questions.setdefault(question.target_id, [])
            if topic:
                topic_key = (state.day, question.target_id, topic)
                if topic_key in self._asked_question_topics or any(
                    item.day == state.day and item.topic == topic for item in existing
                ):
                    continue
                self._asked_question_topics.add(topic_key)
            existing.append(
                PendingQuestion(
                    asker=player_id,
                    target=question.target_id,
                    question=said_aloud,
                    source_message_id=message_id,
                    day=state.day,
                    topic=topic,
                )
            )
            # The engine drops a pending question once it is answered; the
            # day's record keeps every one, for tomorrow's 問答 block.
            self._context.record_question(existing[-1])
        self._context.record_key_point(
            state.day, message_id, player_id, output.key_point, stage=stage
        )
        self._register_claim_drafts(controller, player_id, drafts, message_id)
        if self._recorder is not None and decision is not None and self.reasoning is not None:
            # What the seat had seen when it spoke includes what it said: the
            # turn's audit is recorded before its own CO and results are
            # registered, so the snapshot is retaken now.
            self._recorder.update_latest_decision(
                player_id, state.day, public_result_ids_at_decision=self._public_result_ids(state)
            )
        if self._recorder is not None and required_ids:
            published_list = [
                f"{result.result_type}:{result.referenced_day}:{result.target_id}"
                for result in PublicFactLedger(state).public_results()
                if result.claimant_id == player_id and result.source_message_id == message_id
            ]
            published_ids = tuple(dict.fromkeys(published_list))
            self._recorder.update_latest_decision(
                player_id, state.day, published_result_ids=published_ids
            )
            self._recorder.record_result_publication_audit(
                ResultPublicationAuditRecord(
                    player_id=player_id,
                    day=state.day,
                    required_result_ids=required_ids,
                    published_result_ids=published_ids,
                    omitted_result_ids=_omitted_result_ids(required_ids, published_ids),
                    duplicate_result_ids=tuple(
                        sorted({item for item in published_list if published_list.count(item) > 1})
                    ),
                )
            )
        if self.reasoning is not None:
            self.reasoning.record_public_speech(state, player_id, output.public_message, message_id)
        return output

    def _questions_said_aloud(
        self, state: GameState, questions: Sequence[DirectedQuestion], model_text: str
    ) -> list[DirectedQuestion]:
        """The structured questions the line itself puts to their targets.

        A question in the structured field is only the model's note that it
        asked someone. Where the text it wrote does not name that seat (whole,
        as the vote prompt reads names), the table heard no question, and
        booking the seat a reply or showing it "what you were asked" invents
        one. A line code wrote -- `model_text` is then empty -- asks nothing of
        the model's, whatever names the code put in it. Each drop is counted.
        """
        if not model_text:
            for _question in questions:
                self._record_event("directed_question_on_code_line")
            return []
        named = set(
            named_seat_ids(model_text, {pid: player.name for pid, player in state.players.items()})
        )
        kept: list[DirectedQuestion] = []
        for question in questions:
            if question.target_id in named:
                kept.append(question)
            else:
                self._record_event("directed_question_target_unnamed")
        return kept

    def _owes_coded_line(
        self,
        state: GameState,
        player_id: str,
        decision: DiscussionDecision,
        freemason_opening: str | None,
    ) -> bool:
        """Whether this turn carries a line code fixes, whatever the model writes.

        Those are the turn's obligations -- a result to publish, the CO behind
        it, the freemason plan's opening or a partner confirmation -- and a
        failed generation must not cancel them, because none of them ever
        depended on the model's wording.
        """
        return bool(
            decision.required_public_results
            or decision.required_claim_role is not None
            or freemason_opening is not None
            or self._pending_partner_relation(state, player_id) is not None
        )

    @staticmethod
    def _pending_partner_relation(
        state: GameState, player_id: str
    ) -> FreemasonPartnerClaim | None:
        """A freemason's naming of this seat as partner, still awaiting its confirmation."""
        return next(
            (
                claim
                for claim in state.freemason_partner_claims
                if claim.partner_id == player_id
                and not claim.confirmed
                and state.players[claim.claimant_id].role == RoleName.FREEMASON
                and state.players[player_id].role == RoleName.FREEMASON
            ),
            None,
        )

    def _validate_output(self, state: GameState, player_id: str, output: DiscussionOutput) -> None:
        """Reconcile one discussion turn with the public fact ledger."""
        previous = self._context.get_reasoning_memo(player_id) or {}
        previous_target = previous.get("execution_target")
        _output, issues = validate_discussion_output(
            output,
            PublicFactLedger(state),
            speaker_id=player_id,
            previous_execution_target=previous_target if isinstance(previous_target, str) else None,
            excluded_target_ids=self._observer_player_ids,
        )
        self.validation.extend(issues)

    def _coded_night_target(
        self, player_id: str, action_type: str, candidates: list[str]
    ) -> NightActionOutput | None:
        """Night targeting from beliefs, when v2 is on.

        Chasing the top suspect (divine, attack) or covering the most trusted
        seat (guard) is a ranking the belief engine already computes. Spending a
        request to re-derive it produced a worse answer, not a better one.
        """
        if not self._decides_in_code or self.reasoning is None:
            return None
        target = self.reasoning.night_target(player_id, action_type, candidates)
        if target is None:
            return None
        seat = self.reasoning.seats.get(player_id)
        reason = ""
        if seat is not None:
            score = _night_utility(seat, action_type).get(target, 0.0)
            reason = f"{action_type}の評価値が最も高い対象は{target}({score:+.1f})。"
        return NightActionOutput(target=target, reason=reason)

    def _belief_targets(self, player_id: str) -> list[str]:
        """This player's own stated beliefs, best first. Used to repair an
        invalid target without reaching for a random seat."""
        memo = self._context.get_reasoning_memo(player_id) or {}
        preferred: list[str] = []
        for value in (memo.get("execution_target"), *(memo.get("suspects") or [])):
            if isinstance(value, str) and value and value not in preferred:
                preferred.append(value)
        return preferred

    def _freemason_must_hide(self, state: GameState, player_id: str) -> bool:
        if self._freemason_public_plan is None:
            return False
        leader, partner, full_reveal = self._freemason_public_plan
        named_for_confirmation = any(
            claim.partner_id == player_id for claim in state.freemason_partner_claims
        )
        partner_under_black = any(
            claim.target_id == player_id and claim.is_werewolf
            for claim in state.public_result_claims
        )
        return (
            player_id == partner
            and state.players[leader].alive
            and not partner_under_black
            and (not full_reveal or not named_for_confirmation)
            and not any(claim.player_id == player_id for claim in state.co_declarations)
        )

    def _freemason_opening(self, state: GameState, player_id: str) -> str | None:
        """Return the public line required by the AI-only shared-role plan."""
        if self._freemason_public_plan is None:
            return None
        leader, partner, full_reveal = self._freemason_public_plan
        if any(claim.player_id == player_id for claim in state.co_declarations):
            if (
                player_id == leader
                and not state.players[partner].alive
                and not full_reveal
                and player_id not in self._freemason_death_announced
            ):
                return "共有者として報告します。相方は死亡しました。"
            return None
        if player_id == leader:
            if full_reveal:
                return f"共有者CO。相方は{self._spoken_label(state, partner)}です。"
            return "共有者CO。相方は生存しています。"
        partner_under_black = any(
            claim.target_id == player_id and claim.is_werewolf
            for claim in state.public_result_claims
        )
        if player_id == partner and (not state.players[leader].alive or partner_under_black):
            status = "死亡した" if not state.players[leader].alive else ""
            return f"共有者CO。相方は{status}{self._spoken_label(state, leader)}です。"
        return None

    def _owes_freemason_report(self, state: GameState, player_id: str) -> bool:
        """A freemason of the AI-only plan whose next line is a report code writes.

        The hidden partner's takeover CO once the leader is dead (or the
        partner is blacked), and the claimed leader's word that the hidden
        partner died. Not the leader's first CO: that is a planned claim.
        """
        if self._freemason_public_plan is None:
            return False
        leader, _partner, _full_reveal = self._freemason_public_plan
        opening = self._freemason_opening(state, player_id)
        if opening is None:
            return False
        claimed = any(claim.player_id == player_id for claim in state.co_declarations)
        return claimed or player_id != leader

    def _spoken_label(self, state: GameState, player_id: str) -> str:
        """A player as the coded lines name them: `名前(pN)` for legacy and v2,
        the bare name under v3, where the table never sees ids."""
        player = state.players[player_id]
        return player.name if self.model_decides else f"{player.name}({player_id})"

    def _is_fallback(self, player_id: str, text: str) -> bool:
        """The agent substitutes a personality-specific canned line when the
        model fails entirely. Those lines are distinctive enough that an
        exact match is a reliable signal, and it keeps the agent's return
        types unchanged."""
        return text.strip() == self._personalities[player_id].get_fallback_message()

    def note_human_message(
        self, state: GameState, speaker_id: str, text: str, source_message_id: str = ""
    ) -> list[str]:
        """Handle one human message for the whole table, once.

        Factual corrections are matched in code and applied to each seat's own
        evidence, so a sentence costs zero model calls however many AIs are
        listening. Returns the seats that actually had a reason withdrawn --
        the ones with something to say back.
        """
        if self.reasoning is None:
            return []
        self.reasoning.refresh(state)
        outcomes = self.reasoning.apply_human_message(state, speaker_id, text, source_message_id)
        moved = self.reasoning.seats_that_moved(outcomes)
        if self._recorder is not None and outcomes:
            by_correction: dict[str, list[CorrectionOutcome]] = {}
            for outcome in outcomes:
                by_correction.setdefault(outcome.verdict.correction.correction_id, []).append(
                    outcome
                )
            for correction_id, grouped in by_correction.items():
                affected = tuple(
                    sorted({outcome.seat_id for outcome in grouped if outcome.changed_anything})
                )
                first = grouped[0]
                self._recorder.record_correction_audit(
                    CorrectionAuditRecord(
                        correction_id=correction_id,
                        source_message_id=source_message_id,
                        speaker_id=speaker_id,
                        verdict=first.verdict.status.value,
                        affected_seat_ids=affected,
                        retracted_evidence_ids=tuple(
                            sorted(
                                {
                                    item
                                    for outcome in grouped
                                    for item in outcome.invalidated_evidence_ids
                                }
                            )
                        ),
                        belief_delta_by_seat={
                            outcome.seat_id: outcome.belief_delta
                            for outcome in grouped
                            if outcome.seat_id in affected
                        },
                    )
                )
        # A seat whose reason was withdrawn owes the table an explanation, so it
        # is queued to speak. Dropping this list was why a correction landed and
        # nobody said anything about it.
        self.reasoning.queue_reassessment_speakers(moved)
        return moved

    def register_public_claim(
        self,
        controller: object,
        player_id: str,
        output: DiscussionOutput,
        message_id: str = "",
    ) -> None:
        """Public entry point: derive this turn's claims and record them.

        Used for the human path and by callers that already hold a finished
        message; `_speak` splits the two halves so it can fix the prose first.
        """
        state = controller.state  # type: ignore[attr-defined]
        drafts = build_claim_drafts(output, PublicFactLedger(state), speaker_id=player_id)
        self._register_claim_drafts(controller, player_id, drafts, message_id)
        if self.reasoning is not None:
            # What this message said about other people's ballots. A quote the
            # record contradicts becomes a reason the listeners hold -- and the
            # thing a later correction is able to take away from them.
            self.reasoning.record_public_speech(state, player_id, output.public_message, message_id)

    def _register_claim_drafts(
        self,
        controller: object,
        player_id: str,
        drafts: list[SpeechEventDraft],
        message_id: str = "",
    ) -> None:
        state = controller.state  # type: ignore[attr-defined]
        checked: list[SpeechEventDraft] = []
        for draft in drafts:
            if draft.event_type not in (
                SpeechEventType.ABILITY_RESULT,
                # A correction states a verdict too, so it clears the same bar.
                # Letting it through unchecked would make "correcting" a result
                # the way to publish one the validator would have rejected.
                SpeechEventType.RESULT_CORRECTION,
            ):
                checked.append(draft)
                continue
            # Published verdicts clear the same consistency bar however they were
            # derived: a declared field and a regex are equally unaware of whether
            # the target was ever executed.
            validation = validate_public_result_claim(
                SchemaPublicResultClaim(
                    result_type=draft.role.value if draft.role else "",
                    target_id=draft.target_id or "",
                    is_werewolf=bool(draft.result_is_werewolf),
                ),
                PublicFactLedger(state),
                claimant_id=player_id,
                is_correction=(draft.event_type is SpeechEventType.RESULT_CORRECTION),
            )
            self.validation.extend(validation.issues)
            if validation.claim is not None:
                checked.append(replace(draft, result_is_werewolf=validation.claim.is_werewolf))
        register_claim_drafts(controller, player_id, checked, message_id)
        for draft in checked:
            if draft.event_type is SpeechEventType.PARTNER_CLAIM and draft.target_id:
                self._prompt_partner_confirmation(state, player_id, draft.target_id, message_id)

    def _prompt_partner_confirmation(
        self, state: GameState, player_id: str, partner_id: str, message_id: str
    ) -> None:
        relation = next(
            (
                claim
                for claim in state.freemason_partner_claims
                if claim.claimant_id == player_id and claim.partner_id == partner_id
            ),
            None,
        )
        if relation is None or relation.confirmed:
            return
        already_asked = any(
            question.asker == player_id and question.topic == "freemason_confirmation"
            for question in self._pending_questions.get(partner_id, [])
        )
        if already_asked:
            return
        self._pending_questions.setdefault(partner_id, []).append(
            PendingQuestion(
                asker=player_id,
                target=partner_id,
                question=(
                    "共有相方として指名されました。本人ならこの発言で確認共有COし、"
                    "相方でなければ明確に否定してください。"
                ),
                source_message_id=message_id,
                day=state.day,
                topic="freemason_confirmation",
            )
        )
        if partner_id in self._agents and partner_id not in self._forced_partner_confirmations:
            self._forced_partner_confirmations.append(partner_id)

    # -- voting (loops across runoff rounds) --

    async def generate_all_votes(self, session: object) -> None:
        controller = session.controller  # type: ignore[attr-defined]
        state = controller.state
        human_id = session.human_id  # type: ignore[attr-defined]

        while state.phase in (Phase.VOTING, Phase.RUNOFF):
            human = state.players[human_id]
            observer = human_id in self._observer_player_ids
            waiting_for_human = (
                human.is_human
                and human.alive
                and not observer
                and human_id not in state.pending_votes
            )
            if waiting_for_human:
                return  # wait for the human's vote this round

            alive_ai = [pid for pid in self._ai_player_ids if state.players[pid].alive]
            await asyncio.gather(*(self._cast_vote(controller, state, pid) for pid in alive_ai))

            try:
                controller.resolve_votes()
            except Exception:
                return

        if state.phase == Phase.VOTE_RESULT:
            await self._generate_day_summary(controller, state)

    async def _cast_vote(self, controller: object, state: GameState, player_id: str) -> None:
        # In a runoff this is narrowed to the tied players, so the AI is not
        # offered choices the engine would reject.
        candidates = state.votable_ids(player_id)
        candidates = [pid for pid in candidates if pid not in self._observer_player_ids]
        if not candidates:
            return
        stated_target = (self._context.get_reasoning_memo(player_id) or {}).get("execution_target")
        if self._decides_in_code and self.reasoning is not None:
            # A vote is an argmax over evidence the engine already holds, so it
            # costs no request and the reason is the evidence itself.
            self.reasoning.refresh(state)
            target, reason = self.reasoning.vote_decision(player_id, candidates)
            # Compared against what this seat actually told the table, not
            # against another internal number -- otherwise the check is the
            # runtime marking its own homework. Today's statement only.
            stated_target = self.reasoning.stated_target(player_id, state.day)
            output = VoteOutput(
                vote_target=target,
                reason=reason,
                decisive_evidence=reason,
                alternative_target=self.reasoning.seats[player_id].belief.state.alternative_target,
            )
        else:
            if self.reasoning is not None:
                # v3: checked against what the seat told the table, like v2 --
                # and, like v2, against what it told the table today.
                stated_target = self.reasoning.stated_target(player_id, state.day)
            system, messages = self._context.build_vote_context(
                state,
                player_id,
                candidates,
                board_memo=self._memo_for(state, player_id),
                # A candidate who has since died is not one the ballot can
                # follow; reminding the seat of it only invites a vote for a
                # dead player. The comparison below still sees it, and files
                # the change as the target having become invalid.
                stated_target=(
                    stated_target
                    if isinstance(stated_target, str)
                    and stated_target in state.players
                    and state.players[stated_target].alive
                    else None
                ),
            )
            output = await self._agents[player_id].generate_vote(
                system, messages, candidates, preferred_targets=self._belief_targets(player_id)
            )
        self._record(
            state,
            player_id,
            "vote",
            text=output.reason,
            target=output.vote_target,
            decision_evidence=output.decisive_evidence,
            countercase=output.countercase,
            alternative_target=output.alternative_target,
        )
        try:
            controller.vote(player_id, output.vote_target)  # type: ignore[attr-defined]
        except Exception:
            return
        actual_target = state.pending_votes.get(player_id)
        if self._recorder is not None and hasattr(
            self._recorder, "update_latest_decision"
        ):
            # Classified before a ballot-only audit can exist: the comparison
            # is with what the seat had seen when it last spoke. Each round's
            # ballot lands on its own record (`_ensure_ballot_audit`), so a
            # runoff never overwrites the first round's.
            change_kind, change_reason = self._classify_vote_change(
                state,
                player_id,
                stated_target=stated_target if isinstance(stated_target, str) else None,
                actual_target=actual_target,
                candidates=candidates,
                reason=output.reason,
            )
            self._ensure_ballot_audit(state, player_id, candidates)
            ally_vote, ally_vote_planned = self._audit_ally_vote(state, player_id, actual_target)
            self._recorder.update_latest_decision(
                player_id,
                state.day,
                vote_target=actual_target,
                vote_round=state.vote_round,
                target_change_classification=change_kind,
                target_change_reason=change_reason,
                ally_vote=ally_vote,
                ally_vote_planned=ally_vote_planned,
            )
        # Voting against your own stated plan is legal play, not corruption, so
        # this is recorded rather than corrected -- but silently losing it means
        # nobody can tell a change of mind from an AI that lost track of itself.
        # The ballot the engine actually holds, not the model's own answer:
        # `controller.vote` may have rejected it, and `vote_records` is only
        # written at tally time.
        mismatch = detect_vote_plan_mismatch(
            PublicFactLedger(state),
            voter_id=player_id,
            stated_target=stated_target if isinstance(stated_target, str) else None,
            day=state.day,
            round_number=state.vote_round,
            actual_target=state.pending_votes.get(player_id),
        )
        if mismatch is not None:
            self.validation.vote_plan_mismatches.append(mismatch)
            # Where the baseline came from. Under v2 it is read back from the
            # sentence the table saw; under v3 (and legacy) it is the seat's
            # own plan note, which nobody else ever saw -- calling it a public
            # statement made the record claim something no one said.
            baseline = "公開発言では" if self._decides_in_code else "メモでは"
            self._record(
                state,
                player_id,
                "vote_change",
                text=(
                    f"{baseline}{mismatch.stated_target}を第一候補としたが、"
                    f"最終判断で{mismatch.actual_target}へ変更した。{output.reason}"
                ),
                target=mismatch.actual_target,
                alternative_target=mismatch.stated_target,
            )

    @staticmethod
    def _public_result_ids(state: GameState) -> tuple[str, ...]:
        """Every published verdict, in the form decision audits record them."""
        return tuple(
            sorted(
                f"{r.claimant_id}:{r.result_type}:{r.referenced_day}:"
                f"{r.target_id}:{r.is_werewolf}"
                for r in PublicFactLedger(state).public_results()
            )
        )

    # A result that can be news at the vote: a verdict stated, or a stated one
    # corrected. A retraction takes a verdict away and tells nobody anything new.
    _NEW_RESULT_EVENTS = (SpeechEventType.ABILITY_RESULT, SpeechEventType.RESULT_CORRECTION)

    def _classify_vote_change(
        self,
        state: GameState,
        player_id: str,
        *,
        stated_target: str | None,
        actual_target: str | None,
        candidates: Sequence[str],
        reason: str,
    ) -> tuple[str, str]:
        """Why the ballot left the seat's stated candidate, if it did.

        The judgement is `reasoning_analyzer.classify_vote_change`, the same
        function the offline audit runs on a committed transcript; this only
        hands it the day as the table saw it. What the seat heard is read by
        position in the day's log -- the lines after its own last one -- not
        by comparing a snapshot of the published results: the snapshot was
        taken before the turn's own results were registered, so a seer's own
        verdicts read as news to it at the vote.
        """
        corrections: tuple[str, ...] = ()
        recorder = self._recorder
        if recorder is not None and hasattr(recorder, "transcript"):
            corrections = tuple(
                item.source_message_id for item in recorder.transcript.correction_audits
            )
        facts = BallotFacts(
            voter_id=player_id,
            basis=stated_target,
            vote_target=actual_target,
            reason=reason,
            names={pid: player.name for pid, player in state.players.items()},
            day_lines=tuple(
                SpokenLine(message.message_id, message.author_id, message.content)
                for message in state.chat_log
                if message.day == state.day and message.channel is ChatChannel.PUBLIC
            ),
            results=tuple(
                PublishedResult(event.actor_id, event.target_id, event.source_message_id)
                for event in state.speech_events
                if event.event_type in self._NEW_RESULT_EVENTS
                and event.is_binding
                and event.day == state.day
                and event.target_id is not None
            ),
            correction_message_ids=corrections,
            runoff=state.phase is Phase.RUNOFF,
            candidates=tuple(candidates),
            alive_ids=frozenset(state.alive_ids()),
            closing_message_ids=tuple(
                message_id for _label, message_id in self._context.direction_lines(state.day)
            ),
            grey_ids=self._grey_ids(state),
        )
        kind, why = classify_vote_change(facts)
        return kind.value, why

    @staticmethod
    def _grey_ids(state: GameState) -> frozenset[str]:
        """The grey now: alive, no claim, no seer claimant's verdict (the board memo's grey)."""
        claimed = {claim.player_id for claim in state.co_declarations}
        judged = {
            event.target_id
            for event in state.speech_events
            if event.event_type is SpeechEventType.ABILITY_RESULT
            and event.role is RoleName.SEER
            and event.is_binding
            and event.target_id is not None
        }
        return frozenset(seat for seat in state.alive_ids() if seat not in claimed | judged)

    def _ensure_ballot_audit(
        self, state: GameState, player_id: str, candidates: Sequence[str]
    ) -> None:
        """Open a ballot-only audit for a ballot no audit of today can hold.

        That is a seat with no discussion turn today, or a runoff ballot from
        a seat whose latest audit already holds an earlier round's.

        Audits used to start at a discussion turn, so a seat that never spoke
        cast a ballot nobody checked -- 22 of the 69 in the first Claude game,
        a werewolf's vote on its partner among them, so the gate's zero for
        ally votes was a zero over the seats that happened to speak. And a
        runoff ballot used to be written over the first round's, so an
        unplanned ally vote or an unexplained change in round 1 left the gate
        when the seat voted differently in the runoff. A runoff record keeps
        the snapshot of the audit it follows -- what the seat had seen when it
        last spoke -- so every round is classified against the same thing.
        v2 and v3 only, like the turn audits: legacy keeps no audit at all.
        """
        recorder = self._recorder
        if recorder is None or self.reasoning is None or not hasattr(recorder, "latest_decision"):
            return
        latest = recorder.latest_decision(player_id, state.day)
        if latest is not None and (
            latest.vote_target is None or latest.vote_round == state.vote_round
        ):
            return
        if latest is None:
            board_version = self.reasoning.observations.board_version
            public_result_ids = self._public_result_ids(state)
            correction_ids = tuple(
                sorted({item.correction_id for item in recorder.transcript.correction_audits})
            )
        else:
            board_version = latest.board_version
            public_result_ids = latest.public_result_ids_at_decision
            correction_ids = latest.correction_ids_at_decision
        recorder.record_decision_audit(
            DecisionAuditRecord(
                game_id=state.session_id,
                day=state.day,
                phase=state.phase.value,
                player_id=player_id,
                board_version=board_version,
                public_result_ids_at_decision=public_result_ids,
                correction_ids_at_decision=correction_ids,
                votable_ids_at_decision=tuple(candidates),
                vote_only=True,
            )
        )

    def _audit_ally_vote(
        self, state: GameState, player_id: str, actual_target: str | None
    ) -> tuple[bool, bool]:
        """Whether this ballot is a werewolf's vote on its partner, and whether it was planned.

        A plan someone recorded beforehand counts first. Otherwise one is
        recognised from what existed before the ballot (see
        `_derive_wolf_ally_vote_plan`) and written down, so the transcript
        says why the gate let it through.
        """
        if actual_target is None or actual_target not in state.players:
            return False, False
        if not (
            state.players[player_id].role is RoleName.WEREWOLF
            and state.players[actual_target].role is RoleName.WEREWOLF
        ):
            return False, False
        recorder = self._recorder
        if recorder is None:
            return True, False
        if (
            hasattr(recorder, "matching_wolf_ally_vote_plan")
            and recorder.matching_wolf_ally_vote_plan(
                player_id, actual_target, state.day, state.vote_round
            )
            is not None
        ):
            return True, True
        if self.reasoning is None:
            return True, False
        plan = self._derive_wolf_ally_vote_plan(state, player_id, actual_target)
        if plan is None:
            return True, False
        recorder.record_wolf_ally_vote_plan(plan)
        return True, True

    def _derive_wolf_ally_vote_plan(
        self, state: GameState, voter_id: str, ally_id: str
    ) -> WolfAllyVotePlan | None:
        """Recognise a vote on a partner as a decision already made, or return None.

        Two grounds, both fixed before the ballot was cast:

        * the wolf chat since the last night proposed voting or cutting this
          partner -- the team decided it together;
        * the table already marks the partner: a black on them, or a claim the
          public record proves impossible (`ReasoningRuntime.public_marks`).
          Voting a partner everyone can see is lost is how a wolf distances
          itself, not a betrayal -- the first Claude game's day-3 vote by
          カイト on ダイキ, a seer claim dead since day 2, is the case.

        Anything else is a wolf hanging a teammate nobody had given up on, and
        stays unplanned: a hard failure at the release gate.
        """
        assert self.reasoning is not None
        marks = self.reasoning.public_marks(state, ally_id)
        evidence_ids = tuple(mark.evidence_id for mark in marks)
        plan_id = f"{state.session_id}:{state.day}:{state.vote_round}:{voter_id}:{ally_id}"
        agreed = self._wolf_chat_ballot_plan(state, ally_id)
        if agreed is not None:
            message_id, sentence = agreed
            return WolfAllyVotePlan(
                plan_id=plan_id,
                voter_id=voter_id,
                target_id=ally_id,
                day=state.day,
                round_number=state.vote_round,
                public_evidence_ids=evidence_ids,
                story_reason=sentence,
                strategy_reason="人狼チャットで仲間への投票を決めていた",
                source_event_id=message_id,
                basis="wolf_chat",
            )
        if not marks:
            return None
        return WolfAllyVotePlan(
            plan_id=plan_id,
            voter_id=voter_id,
            target_id=ally_id,
            day=state.day,
            round_number=state.vote_round,
            public_evidence_ids=evidence_ids,
            story_reason=marks[0].explanation,
            strategy_reason="公開情報で既に狼か偽と見られている仲間から距離を取る",
            source_event_id=marks[0].source_message_id or marks[0].evidence_id,
            basis="public_mark",
        )

    @staticmethod
    def _wolf_chat_ballot_plan(state: GameState, ally_id: str) -> tuple[str, str] | None:
        """The wolf-chat sentence, since last night, that proposed voting this partner.

        When in doubt, it is not a plan. A false plan passes the very
        betrayal the release gate exists to catch, silently; a missed one
        fails a correct game with the wolf chat beside the failure. So a
        sentence counts only when the partner is the object of a ballot word
        that ends as the team's own proposal or decision (see the comment
        above `_SENTENCE_RE`): 吊ろう / 投票しよう, 入れてくれ / 切っていい /
        吊れ, 切るしかない / 吊らないといけない, ダイキ吊りで決まり / 一択,
        or a plain form closing the sentence (明日はダイキに入れる。). A
        question or a check (吊りでいいよね), a report (吊りたがってる, 流れ,
        入れてた), a prohibition (切るなよ), a negation (吊らなくていい), a
        dare (吊ってみろ), a guess, a quotation, a condition nobody can check
        yet (最悪, 次第, 〜たら, 様子を見て) or a later day (いずれ, 最終日には),
        a hedge after the comma (吊ろう、けど迷う), or another subject or
        source standing before the partner -- another seat (ミオたちは, ミオ
        曰く), the village, a crowd, a third person, a role -- is no plan.

        The partner is that object by name; by pronoun only with a request:
        as お前 when it alone was addressed at the head of that sentence or
        an earlier one of the same message ("ダイキ、お前を切っていいな"), or
        as 俺/私 in a line it wrote itself ("明日は俺を切って信用を取れ").
        "ダイキ、明日はお前に投票する", a plea ("俺を切るなよ") or a report
        ("村は俺を吊りたがってる") is no plan.

        A plan holds until a later line doubts or refuses the partner's
        ballot ("俺を切るのはまだ早い", "やっぱりダイキは吊らない"), keeps the
        partner ("いや、ダイキは残そう"), opens on a turn-around ("やっぱ
        やめた", "今のなし"), proposes another seat ("やっぱりミオを吊ろう";
        not the partner naming its own vote, "俺はミオに入れる"), or is the
        partner objecting ("嫌だ", "それは困る"); or until the same
        message takes it back, laughs it off or goes on to quote it ("…吊ろう。
        冗談だよ", "…吊ろう。笑", "…吊りで。って村が言ってる"): a team still
        arguing has not decided. A sentence that proposes the partner and
        another seat at once is no plan either. The latest plan still
        standing is returned.
        """
        ally = _seat_pattern(ally_id, state.players[ally_id].name)
        # A name that ends its sentence is called too: "ダイキ、ミオ。お前を…"
        # calls both, so お前 is neither.
        vocatives = {
            pid: re.compile(
                _seat_pattern(pid, player.name) + _HONORIFIC + r"\s*(?:[、,，]\s*|$)"
            )
            for pid, player in state.players.items()
        }
        group = re.compile(_GROUP_VOCATIVE + r"\s*(?:[、,，]\s*|$)")
        other_seats = "|".join(
            _seat_pattern(pid, player.name)
            for pid, player in state.players.items()
            if pid != ally_id
        )
        every_seat = "|".join(
            _seat_pattern(pid, player.name) for pid, player in state.players.items()
        )
        not_the_team = re.compile(
            rf"(?:{other_seats})(?!{_SEAT_NOT_A_SUBJECT})|{_NOT_THE_TEAM}"
        )
        # A ballot proposed on another seat replaces the plan; who proposes it
        # is read the same way, the partner included (ダイキはミオに投票して
        # tells the partner where its own vote goes, and replaces nothing).
        not_the_team_for_another = re.compile(
            rf"(?:{every_seat})(?!{_SEAT_NOT_A_SUBJECT})|{_NOT_THE_TEAM}"
        )
        not_the_team_for_partners_own = re.compile(
            rf"{_FIRST_PERSON}[はがも]|{not_the_team_for_another.pattern}"
        )
        plan: tuple[str, str] | None = None
        for message in state.chat_log:
            if message.channel is not ChatChannel.WOLF or message.day not in (
                state.day - 1,
                state.day,
            ):
                continue
            addressee: str | None = None
            planned_in_this_message = False
            by_partner = message.author_id == ally_id
            for sentence in _SENTENCE_RE.finditer(message.content):
                body = sentence["body"].rstrip()
                called = _leading_vocatives(body, vocatives, group)
                if called:
                    addressee = called[0] if len(called) == 1 else None
                seats = [(ally, _ANY_MOOD)]
                if addressee == ally_id:
                    seats.append((_SECOND_PERSON, _REQUESTING))
                if by_partner:
                    seats.append((_FIRST_PERSON, _REQUESTING))
                asked = bool(set(sentence["end"]) & {"？", "?"} or _QUESTION_RE.search(body))
                on_partner = not asked and any(
                    _proposes_ballot_on(body, seat, moods, not_the_team) for seat, moods in seats
                )
                on_another = not asked and _proposes_ballot_on(
                    body,
                    f"(?:{other_seats})",
                    _COUNTER_PROPOSING if by_partner else _ANY_MOOD,
                    not_the_team_for_partners_own if by_partner else not_the_team_for_another,
                )
                if on_partner and not on_another:
                    plan = (message.message_id, body.strip())
                    planned_in_this_message = True
                elif plan is not None and (
                    on_another
                    or _RETRACTION_RE.match(body)
                    or (planned_in_this_message and _CONTINUES_RE.match(body))
                    or (by_partner and _OBJECTION_RE.search(body))
                    or any(_unsettles_ballot_on(body, seat, asked) for seat, _ in seats)
                ):
                    plan = None
                    planned_in_this_message = False
        return plan

    async def _generate_day_summary(self, controller: object, state: GameState) -> None:
        # The factual half is rendered from the ledger, not asked for: who died,
        # who COed, which verdicts were published and who voted for whom are
        # already known exactly, and a model restating them can only lose them.
        # v3 reads the board in its memo, so its past day is the day itself:
        # execution, votes, the night before and the lines quoted as said.
        facts = (
            self._context.day_record(state)
            if self.model_decides
            else render_public_fact_summary(PublicFactLedger(state), state.day)
        )
        if self.model_decides:
            self._day_summaries.set_exchanges(state.day, self._context.question_exchanges(state))
        alive_ai = [pid for pid in self._ai_player_ids if state.players[pid].alive]
        if not alive_ai:
            self._day_summaries.set_summary(state.day, "", facts)
            self._day_summaries.compress_if_needed()
            return
        narrator_id = alive_ai[0]
        if self.reasoning is not None:
            # The disagreements are already structured events; narrating them
            # back through a model adds a request and a chance to lose one.
            self.reasoning.refresh(state)
            commentary = "。".join(
                self.reasoning.conflict_points(
                    state, stated_targets_are_public=not self.model_decides
                )
            )
            if self.model_decides:
                commentary = namify(commentary, PublicFactLedger(state))
            # Written in code: the seat is only where the record files it, and
            # the stage says so, so no transcript names it as the author.
            self._record(state, narrator_id, "summary", text=commentary, stage=_CODE_SUMMARY_STAGE)
            self._day_summaries.set_summary(state.day, commentary, facts)
            self._day_summaries.compress_if_needed()
            return
        system, messages = self._context.build_summary_context(state, narrator_id)
        output = await self._agents[narrator_id].generate_summary(system, messages)
        self._record(state, narrator_id, "summary", text=output.summary)
        self._day_summaries.set_summary(state.day, output.summary, facts)
        self._day_summaries.compress_if_needed()

    def day_summary_text(self, day: int) -> str:
        """The stored summary for a day: deterministic facts plus the labelled
        commentary the narrator generated on top of them."""
        return compose_day_summary(
            self._day_summaries.facts.get(day, ""), self._day_summaries.summaries.get(day, "")
        )

    # -- night --

    async def run_night_phase(self, session: object) -> None:
        controller = session.controller  # type: ignore[attr-defined]
        state = controller.state

        if state.phase != Phase.NIGHT:
            return

        if state.day == 0:
            if state.pending_divine is None:
                seer_id = self._find_ai_with_role(state, RoleName.SEER)
                if seer_id is not None:
                    await self._cast_divine(controller, state, seer_id)
            try:
                controller.resolve_night()
            except Exception:
                pass
            return

        tasks = []
        if state.pending_divine is None:
            seer_id = self._find_ai_with_role(state, RoleName.SEER)
            if seer_id is not None:
                tasks.append(self._cast_divine(controller, state, seer_id))
        if state.pending_guard is None:
            hunter_id = self._find_ai_with_role(state, RoleName.HUNTER)
            if hunter_id is not None:
                tasks.append(self._cast_guard(controller, state, hunter_id))
        tasks.append(self._run_wolf_chat_round(controller, state))
        tasks.append(self._run_freemason_chat_round(controller, state))
        if tasks:
            await asyncio.gather(*tasks)

        if state.pending_attack is None:
            alpha_id = controller.alpha_wolf_id
            if alpha_id in self._agents and state.players[alpha_id].alive:
                await self._cast_attack(controller, state, alpha_id)
                if state.pending_attack is None:
                    self._attack_anyone(controller, state, alpha_id)

        try:
            controller.resolve_night()
        except Exception:
            # The engine refuses a night with a living wolf and no bite (wolves
            # bite a non-wolf every night; the board memo's night lines rest on
            # it). An AI alpha always has a bite by now; a human alpha has not
            # acted yet and the orchestrator waits for it. Counted, not silent:
            # a night that cannot resolve is a stalled game, not a quiet one.
            self._record_event("night_not_resolved")

    async def _cast_divine(self, controller: object, state: GameState, seer_id: str) -> None:
        candidates = [
            pid for pid in state.alive_ids() if pid != seer_id and pid != state.first_victim_id
        ]
        candidates = [pid for pid in candidates if pid not in self._observer_player_ids]
        if not candidates:
            return
        system, messages = self._context.build_night_action_context(
            state, seer_id, "divine", candidates, board_memo=self._memo_for(state, seer_id)
        )
        coded = self._coded_night_target(seer_id, "divine", candidates)
        if coded is not None:
            output = coded
        else:
            output = await self._agents[seer_id].generate_night_action(
                system, messages, candidates, preferred_targets=self._belief_targets(seer_id)
            )
        self._record(state, seer_id, "night_action", text=output.reason, target=output.target)
        self._submit_audited_night_action(
            controller, state, seer_id, "divine", output.target, candidates
        )

    async def _cast_guard(self, controller: object, state: GameState, hunter_id: str) -> None:
        candidates = [pid for pid in state.alive_ids() if pid != hunter_id]
        candidates = [pid for pid in candidates if pid not in self._observer_player_ids]
        if not candidates:
            return
        system, messages = self._context.build_night_action_context(
            state, hunter_id, "guard", candidates, board_memo=self._memo_for(state, hunter_id)
        )
        # The hunter protects who they trust, so their fallback preference is the
        # trusted list rather than the suspect list every other action uses.
        memo = self._context.get_reasoning_memo(hunter_id) or {}
        trusted = [
            value
            for value in (memo.get("trusted_seer"), *(memo.get("trusted") or []))
            if isinstance(value, str) and value
        ]
        coded = self._coded_night_target(hunter_id, "guard", candidates)
        if coded is not None:
            output = coded
        else:
            output = await self._agents[hunter_id].generate_night_action(
                system, messages, candidates, preferred_targets=trusted
            )
        self._record(state, hunter_id, "night_action", text=output.reason, target=output.target)
        self._submit_audited_night_action(
            controller, state, hunter_id, "guard", output.target, candidates
        )

    async def _cast_attack(self, controller: object, state: GameState, alpha_id: str) -> None:
        candidates = [
            pid
            for pid in state.alive_ids()
            if state.players[pid].role != RoleName.WEREWOLF and pid not in self._observer_player_ids
        ]
        if not candidates:
            return
        system, messages = self._context.build_night_action_context(
            state, alpha_id, "attack", candidates, board_memo=self._memo_for(state, alpha_id)
        )
        coded = self._coded_night_target(alpha_id, "attack", candidates)
        if coded is not None:
            output = coded
        else:
            output = await self._agents[alpha_id].generate_night_action(
                system, messages, candidates, preferred_targets=self._belief_targets(alpha_id)
            )
        self._record(state, alpha_id, "night_action", text=output.reason, target=output.target)
        self._submit_audited_night_action(
            controller, state, alpha_id, "attack", output.target, candidates
        )

    def _attack_anyone(self, controller: object, state: GameState, alpha_id: str) -> None:
        """The bite the night must have when the AI alpha's own went nowhere.

        `_cast_attack` offers only non-wolf seats that are not observers; it
        submits nothing when every non-wolf is an observer, and a submission
        can be refused. The engine then would not resolve the night, so code
        bites the first living non-wolf -- an observer only as the last resort.
        Counted once a night as its own event, `night_code_bite`: a bite the
        wolves never chose is not one of the ordinary code-filled actions.
        """
        living = [pid for pid in state.alive_ids() if state.players[pid].role != RoleName.WEREWOLF]
        playing = [pid for pid in living if pid not in self._observer_player_ids]
        if living:
            self._record_event("night_code_bite")
        for target in [*playing, *[pid for pid in living if pid not in playing]]:
            self._submit_audited_night_action(controller, state, alpha_id, "attack", target, living)
            if state.pending_attack is not None:
                return

    def _submit_audited_night_action(
        self,
        controller: object,
        state: GameState,
        actor_id: str,
        action_type: str,
        target_id: str,
        candidates: list[str],
    ) -> None:
        accepted = False
        rejection = ""
        try:
            controller.submit_night_action(actor_id, action_type, target_id)  # type: ignore[attr-defined]
            accepted = True
        except Exception as exc:
            rejection = type(exc).__name__
        if self._recorder is not None and hasattr(
            self._recorder, "record_night_action_audit"
        ):
            self._recorder.record_night_action_audit(
                NightActionAuditRecord(
                    action_id=f"{state.session_id}:{state.day}:{action_type}:{actor_id}",
                    actor_id=actor_id,
                    target_id=target_id,
                    night=state.day,
                    action_type=action_type,
                    accepted=accepted,
                    rejection_reason=rejection,
                    actor_alive=state.players[actor_id].alive,
                    target_was_legal=target_id in candidates,
                )
            )

    async def _run_wolf_chat_round(self, controller: object, state: GameState) -> None:
        wolf_ids = [
            pid
            for pid in self._ai_player_ids
            if state.players[pid].alive and state.players[pid].role == RoleName.WEREWOLF
        ]
        if not self._channel_has_company(state, RoleName.WEREWOLF):
            return
        if self._decides_in_code and wolf_ids:
            # v2: one plan for the team, not one soliloquy each. Wolves share
            # every fact they have, so N requests bought N restatements of it.
            # v3 lets each wolf talk: the chat is where the attack gets decided.
            await self._run_team_plan(controller, state, wolf_ids, "wolf")
            return
        for pid in wolf_ids:
            system, messages = self._context.build_wolf_chat_context(
                state, pid, board_memo=self._memo_for(state, pid)
            )
            output = await self._agents[pid].generate_wolf_chat(system, messages)
            self._record(
                state,
                pid,
                "wolf_chat",
                text=output.message,
                used_fallback=self._is_fallback(pid, output.message),
                effective_length_limit=_PRIVATE_CHAT_MAX_CHARS,
            )
            try:
                controller.chat(pid, output.message, "wolf")  # type: ignore[attr-defined]
            except Exception:
                pass

    async def _run_freemason_chat_round(self, controller: object, state: GameState) -> None:
        mason_ids = [
            pid
            for pid in self._ai_player_ids
            if state.players[pid].alive and state.players[pid].role == RoleName.FREEMASON
        ]
        if not self._channel_has_company(state, RoleName.FREEMASON):
            return
        if self._decides_in_code and mason_ids:
            await self._run_team_plan(controller, state, mason_ids, "freemason")
            return
        for pid in mason_ids:
            system, messages = self._context.build_freemason_chat_context(
                state, pid, board_memo=self._memo_for(state, pid)
            )
            output = await self._agents[pid].generate_wolf_chat(system, messages)
            self._record(
                state,
                pid,
                "freemason_chat",
                text=output.message,
                used_fallback=self._is_fallback(pid, output.message),
                effective_length_limit=_PRIVATE_CHAT_MAX_CHARS,
            )
            try:
                controller.chat(pid, output.message, "freemason")  # type: ignore[attr-defined]
            except Exception:
                pass

    @staticmethod
    def _channel_has_company(state: GameState, role: RoleName) -> bool:
        """A private channel with at least two living members, the human counted.

        A lone survivor has nobody to talk to: the last wolf and the freemason
        whose partner died were each asked for a line to a dead teammate once
        a night in all three games of run 14.
        """
        return sum(1 for player in state.players_by_role(role) if player.alive) >= 2

    async def _run_team_plan(
        self, controller: object, state: GameState, member_ids: list[str], channel: str
    ) -> None:
        """One request for the whole team's night plan."""
        assert self.reasoning is not None
        speaker = member_ids[0]
        builder = (
            self._context.build_freemason_chat_context
            if channel == "freemason"
            else self._context.build_wolf_chat_context
        )
        system, messages = builder(state, speaker)
        output = await self._agents[speaker].generate_wolf_chat(system, messages)
        self._record(
            state,
            speaker,
            f"{channel}_chat",
            text=output.message,
            used_fallback=self._is_fallback(speaker, output.message),
            effective_length_limit=_PRIVATE_CHAT_MAX_CHARS,
        )
        try:
            controller.chat(speaker, output.message, channel)  # type: ignore[attr-defined]
        except Exception:
            return
        # The rest of the team acknowledges in code: their agreement carries no
        # information the plan does not already state.
        for pid in member_ids[1:]:
            target = self.reasoning.seats[pid].belief.state.current_execution_target
            note = f"了解。自分の第一候補は{target}。" if target else "了解。"
            try:
                controller.chat(pid, note, channel)  # type: ignore[attr-defined]
            except Exception:
                continue

    async def respond_to_private_chat(self, session: object, channel: str) -> None:
        """Let living AI teammates answer a human private message immediately."""
        controller = session.controller  # type: ignore[attr-defined]
        state = controller.state
        role = RoleName.FREEMASON if channel == "freemason" else RoleName.WEREWOLF
        responders = [
            pid
            for pid in self._ai_player_ids
            if state.players[pid].alive and state.players[pid].role == role
        ]
        for pid in responders:
            builder = (
                self._context.build_freemason_chat_context
                if channel == "freemason"
                else self._context.build_wolf_chat_context
            )
            system, messages = builder(state, pid, board_memo=self._memo_for(state, pid))
            controller.set_typing(pid, True, channel)
            try:
                output = await self._agents[pid].generate_wolf_chat(system, messages)
            finally:
                controller.set_typing(pid, False, channel)
            try:
                controller.chat(pid, output.message, channel)
            except Exception:
                pass
