"""How the table talks on day 1 and reads its grey seats.

The owner's review of runs 14 and 15 found the day-1 discussion spent on the
night-0 divination reasons, day-1 ballots cast because a seat's line was
"thin", no grey ever placed as trusted from what it said, hang targets put on
seats that had not spoken yet, and the genuine seer choosing whom to divine by
who stood out. This module counts each of those per run, pooled over its
games, by fixed rules read from the record with no model:

1. the day-1 public model lines that discuss the night-0 divination reasons;
2. the grounds the day-1 ballots give, by type;
3. grey seats trusted from speech -- in the private memo and in public lines;
4. memo hang targets on seats that had not spoken yet that day;
5. the genuine seer's reasons for its divination targets;
6. by skill tier: content-based ballot grounds, grey trust from speech, and
   whether an evaluation of one seat's line matched that seat's camp.

Every rule is a pattern over sentences or a fact from the record, so each
count lists the sentences it matched and a reader can check it. The samples
are a handful of games: a cell is an example, not a rate. Kept out of the
release gate.

"Grey" here is the usual 灰: a living seat that has not claimed a role and
has no seer result (genuine or claimed) published on it before the line in
question. A seat the speaker knows privately -- itself, its werewolf allies,
its freemason partner, the seats it divined as the genuine seer -- is never
read as grey trust "from speech".
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.eval.reasoning_analyzer import mention_spans, split_sentences
from app.eval.transcript import GameTranscript, Utterance
from app.eval.transcript_render import team_label

SKILL_TIERS = ("expert", "average", "novice")
SKILL_LABELS = {"expert": "熟練", "average": "普通", "novice": "初心者"}

# How many matched sentences each table lists under it.
EXAMPLES = 12


# -- the facts every count reads: positions, claims, results, deaths --


class _Table:
    """One game's public record, indexed for "what did the table know at this line"."""

    def __init__(self, transcript: GameTranscript) -> None:
        self.transcript = transcript
        state = transcript.final_state or {}
        self.chat: list[dict[str, Any]] = list(state.get("chat_log", []) or [])
        message_index = {str(m.get("message_id")): i for i, m in enumerate(self.chat)}
        # Each public line by its seat and day, in chat order.
        self.public_lines: dict[tuple[str, int], list[int]] = {}
        free: dict[tuple[str, int, str], list[int]] = {}
        for i, message in enumerate(self.chat):
            if message.get("channel") != "public":
                continue
            seat, day = str(message.get("author_id")), int(message.get("day", -1))
            self.public_lines.setdefault((seat, day), []).append(i)
            free.setdefault((seat, day, str(message.get("content") or "")), []).append(i)
        # The chat position of each discussion utterance (its public line).
        self.position: dict[int, int] = {}
        for k, utterance in enumerate(transcript.utterances):
            if utterance.kind != "discussion":
                continue
            slots = free.get((utterance.player_id, utterance.day, utterance.text))
            if slots:
                self.position[k] = slots.pop(0)
        # The first claim of each seat, by chat position.
        self.claimed_at: dict[str, int] = {}
        for claim in state.get("co_declarations", []) or []:
            seat = str(claim.get("player_id"))
            at = message_index.get(str(claim.get("source_message_id")), len(self.chat))
            self.claimed_at[seat] = min(at, self.claimed_at.get(seat, at))
        # The first seer result (genuine or claimed) published on each seat.
        self.resulted_at: dict[str, int] = {}
        for k, utterance in enumerate(transcript.utterances):
            published = self.position.get(k)
            if published is None:
                continue
            for result in utterance.public_results:
                if result.get("result_type") != "seer":
                    continue
                seat = str(result.get("target_id"))
                self.resulted_at[seat] = min(published, self.resulted_at.get(seat, published))
        # The day each seat died: executed on day d (after that day's talk),
        # attacked on night d, the first victim on night 0.
        self.death_day = {
            str(death.get("player_id")): int(death.get("day", 0))
            for death in state.get("death_records", []) or []
        }
        self.divined = [
            (str(record.get("seer_id")), str(record.get("target_id")), int(record.get("day", 0)))
            for record in state.get("divine_records", []) or []
        ]

    def alive(self, seat: str, day: int) -> bool:
        died = self.death_day.get(seat)
        return died is None or died >= day

    def grey(self, seat: str, day: int, at: int) -> bool:
        """A living seat with no claim and no seer result before chat position `at`."""
        return (
            seat in self.transcript.teams
            and self.alive(seat, day)
            and self.claimed_at.get(seat, len(self.chat)) >= at
            and self.resulted_at.get(seat, len(self.chat)) >= at
        )

    def claimed_by(self, seat: str, day: int) -> bool:
        """Whether `seat` had claimed a role in a line of `day` or earlier."""
        at = self.claimed_at.get(seat)
        return at is not None and at < len(self.chat) and int(self.chat[at].get("day", 0)) <= day

    def known_to(self, speaker: str, day: int) -> set[str]:
        """Seats `speaker` knows by its role, not from the talk."""
        roles = self.transcript.roles
        role = roles.get(speaker)
        known = {speaker}
        if role in ("werewolf", "freemason", "fox"):
            known |= {seat for seat, other in roles.items() if other == role}
        if role == "seer":
            known |= {
                target for seer, target, night in self.divined if seer == speaker and night < day
            }
        return known

    def spoke_before(self, seat: str, day: int, at: int) -> bool:
        return any(i < at for i in self.public_lines.get((seat, day), []))

    def first_line(self, seat: str, day: int) -> int | None:
        lines = self.public_lines.get((seat, day))
        return lines[0] if lines else None

    def model_lines(self) -> Iterable[tuple[int, Utterance, int]]:
        """Each public discussion line the model wrote: (utterance index, it, chat position)."""
        for k, utterance in enumerate(self.transcript.utterances):
            if utterance.kind != "discussion" or utterance.model_text_chars == 0:
                continue
            at = self.position.get(k)
            if at is not None:
                yield k, utterance, at


def _name(transcript: GameTranscript, seat: str) -> str:
    return transcript.names.get(seat, seat)


def _who(transcript: GameTranscript, seat: str) -> str:
    return f"{_name(transcript, seat)}（{team_label(transcript.teams.get(seat, '?'))}）"


Example = tuple[int | None, int, str, str]  # seed, day, who, text


# -- 1. the night-0 divination reasons on day 1 --
#
# A sentence discusses why a seer chose its night-0 target when it says the
# choice was random (「ランダム」「無作為」), or names a reason (理由／なぜ／
# どうして) beside a divination or a choice (占／選／白先): 「なぜマコトを
# 占ったのか」, 「占い先の理由」, 「白先の選び方」. A sentence about tonight's or
# tomorrow's target (今夜／明日／次の) is about the next divination, not
# night 0, and does not count.

_RANDOM_CHOICE = re.compile(r"ランダム|無作為")
_REASON_WORD = re.compile(r"理由|なぜ|何故|どうして")
_DIVINATION_WORD = re.compile(r"占|選|白先")
_NEXT_DIVINATION = re.compile(r"今夜|今晩|明日|次の|次は")


def discusses_divination_reason(sentence: str) -> bool:
    if _NEXT_DIVINATION.search(sentence):
        return False
    if _RANDOM_CHOICE.search(sentence):
        return True
    return bool(_REASON_WORD.search(sentence) and _DIVINATION_WORD.search(sentence))


@dataclass
class DivinationTalk:
    lines: int = 0
    matched: int = 0
    examples: list[Example] = field(default_factory=list)


def divination_reason_talk(
    transcripts: Iterable[GameTranscript], *, examples: int = EXAMPLES
) -> DivinationTalk:
    """Day-1 public model lines, and those with a sentence on the night-0 reasons."""
    found = DivinationTalk()
    for transcript in transcripts:
        for _k, utterance, _at in _Table(transcript).model_lines():
            if utterance.day != 1:
                continue
            found.lines += 1
            sentence = next(
                (s for s in split_sentences(utterance.text) if discusses_divination_reason(s)),
                None,
            )
            if sentence is None:
                continue
            found.matched += 1
            if len(found.examples) < examples:
                found.examples.append(
                    (transcript.seed, 1, _who(transcript, utterance.player_id), sentence)
                )
    return found


# -- 2. the grounds of the day-1 ballots --
#
# Each ballot's line is read for every type it shows (one ballot can show
# several); a ballot that shows none is "other":
#
# * result -- a seer or medium result already out: 「リク黒」「黒が外れた」,
#   「白を出した」, 「霊結果と食い違う」 -- not one denied or awaited
#   (「黒判定がなく」「黒材料はない」「今夜の占い結果を見て」「霊結果で見る」);
# * content -- a reading of what the seat said or did: a camp's viewpoint
#   (「村視点」「狼っぽい」「狼なら」), a contradiction (矛盾／食い違い／辻褄／
#   筋が通らない／不自然／噛み合わない／説明がつかない／前と違う), an answer
#   read as late, made up or general (後付け／後出し／一般論／答えになって
#   いない), reasons that keep moving (「理由が毎回動く」「疑いを散らした」),
#   a ballot read against the line or the table (「投票と発言が」, 流れ任せ,
#   単独票, 「2日連続」, 票を合わせる・寄せる, 見直しがない), protecting or
#   steering (庇う／擁護／誘導／囲い);
# * designation -- following a designation or another seat's word
#   (指定／指示／「方針に従」「〜の言う通り」「〜の指摘どおり」) -- the
#   speaker's own earlier word (「さっき言った通り」「予定どおり」) is not one;
# * thin -- the seat's view or suspicion is not visible, its line is thin or
#   short, its answer weak: 「考えが見えない」「疑い先を出していない」「発言が
#   薄い」「中身がない」「様子見」「質問ばかり」「同調だけ」「答えが弱い」.
#
# The public vote line and the ballot's `decision_evidence` (the grounds the
# model gave beside it) are read separately.

_RESULT_GROUND = re.compile(
    r"(?:(?<!真っ)黒(?!要素|っぽ|寄り|目|く|い|幕|材料|判定(?:は|が|も)?な|探し)"
    r"|白(?:判定|結果|を出|が出|出し|だった|をもら|を貰)"
    r"|霊(?:媒)?(?:の)?結果|占い結果)"
    r"(?![^。、！？]{0,4}?(?:を見て|を待|次第|で見|が出(?:れば|たら)))"
    r"|呪殺"
)
_CONTENT_GROUND = re.compile(
    r"(?:村|狼|人狼|狐|妖狐)(?:視点|目線|っぽ|らし|寄り)|(?:狼|人狼|狐|妖狐)(?:なら|だったら|側の動き|の動き)"
    r"|矛盾|食い違|噛み合わ|辻褄|ちぐはぐ|筋が通らな|不自然|一貫してな|(?:前|昨日|さっき)と(?:違|逆)"
    r"|説明が?つ(?:か|きま)|答えになって|腑に落ち|後付け|後出し|一般論"
    r"|(?:理由|意見|主張|疑い先)(?:が|も|は)?[^。！？、]{0,6}?(?:動(?:い|く|き)|変わ(?!らな|らず|りませ|って(?:い)?な)|ぶれ|ブレ|広が|広げ)"
    r"|疑いを散ら"
    r"|見直(?:し(?:が)?な|され(?:な|ず))|本命を(?:出さ|言わ)|本命が出な"
    r"|(?:投票|票)(?:先)?と(?:発言|意見|主張)|(?:発言|意見|主張)と(?:投票|票)"
    r"|(?:疑って|疑い先に)[^。！？]{0,12}のに[^。！？]{0,12}(?:投票|票)"
    r"|流れ任せ|単独票|(?:[0-9０-９二三]日|2回|二回)連続|(?:票|投票)(?:を|が)?(?:合わせ|寄せ|乗)|に寄(?:って|せ)"
    r"|庇|かば|擁護|誘導|印象操作|身内切|囲(?:い|っ|う)"
)
_DESIGNATION_GROUND = re.compile(
    r"指定(?!(?:は|を|も)?(?:なし|無し|しな|しま?せ|せず))|指示|方針に(?:従|沿)|に従(?:って|う|い)"
    r"|の(?:言う|言った|指摘)(?:通り|どおり)"
)
_THIN_GROUND = re.compile(
    r"(?:答え|理由|根拠|説明|反論|疑い先)(?:が|は|も)?(?:まだ)?(?:弱|浅)|"
    r"見えな|見えませ|見えず|見えて(?:こ)?な|見えにく|薄[いくめさか]|中身(?:が|の)?(?:な|無)|"
    r"(?:意見|見立て|考え|推理|疑い先|疑い|怪しい人|怪しいか|見方|見解|視点|指摘|軸)[^。！？]{0,10}"
    r"(?:な(?:い|く|かっ)|無(?:い|く)|出て(?:い)?(?:な|ません)|出てこな|出さ|出して(?:い)?(?:な|ません)"
    r"|示さ|示して(?:い)?(?:な|ません)|挙げ(?:て(?:い)?(?:な|ません)|ず)|一つも)"
    r"|誰も疑(?:わ|って(?:い)?(?:な|ません))"
    r"|発言(?:量)?が少な|寡黙|様子見|材料(?:が|は)?(?:少な|薄|な|無)|同調|流(?:し|す)だけ"
    r"|(?:質問|聞いて|聞く|振る|回す|煽る|催促)(?:ばかり|だけ)|乗っ(?:た|かった)だけ|なぞ|踏み込"
)

VOTE_GROUND_TYPES = ("content", "result", "designation", "thin", "other")
VOTE_GROUND_LABELS = {
    "content": "発言・行動の中身を読んだ（視点・矛盾・票と発言のずれ・庇い）",
    "result": "結果（占い・霊媒）",
    "designation": "指定・方針・誰かの言う通り",
    "thin": "考え・疑い先が見えない／薄い／発言が少ない",
    "other": "どれにも当たらない",
}


def vote_ground_types(text: str) -> frozenset[str]:
    found = {
        kind
        for kind, pattern in (
            ("content", _CONTENT_GROUND),
            ("result", _RESULT_GROUND),
            ("designation", _DESIGNATION_GROUND),
            ("thin", _THIN_GROUND),
        )
        if pattern.search(text)
    }
    return frozenset(found or {"other"})


@dataclass
class VoteGrounds:
    ballots: int = 0
    line: Counter[str] = field(default_factory=Counter)
    evidence: Counter[str] = field(default_factory=Counter)
    # Ballots whose line was "thin" and nothing else.
    thin_only: int = 0
    examples: dict[str, list[Example]] = field(default_factory=dict)


def _ballots(transcript: GameTranscript) -> Iterable[Utterance]:
    """The day's first ballot of each seat, with its line."""
    seen: set[tuple[str, int]] = set()
    for utterance in transcript.utterances:
        if utterance.kind != "vote" or (utterance.player_id, utterance.day) in seen:
            continue
        seen.add((utterance.player_id, utterance.day))
        yield utterance


def day1_vote_grounds(transcripts: Iterable[GameTranscript], *, examples: int = 4) -> VoteGrounds:
    found = VoteGrounds()
    for transcript in transcripts:
        for ballot in _ballots(transcript):
            if ballot.day != 1:
                continue
            found.ballots += 1
            kinds = vote_ground_types(ballot.text)
            found.line.update(kinds)
            found.evidence.update(vote_ground_types(ballot.decision_evidence))
            found.thin_only += kinds == {"thin"}
            for kind in kinds:
                listed = found.examples.setdefault(kind, [])
                if len(listed) < examples:
                    listed.append(
                        (transcript.seed, 1, _who(transcript, ballot.player_id), ballot.text)
                    )
    return found


# -- 3. grey seats trusted from speech --
#
# In the memo: a seat in `trusted` that was grey at the line and that the
# speaker does not know by its role. In public: a sentence that names a grey
# seat (the nearest name before the words) and places it as trusted or as not
# today's hang: 白い／白っぽい／白寄り／白目, 村っぽい／村寄り／村目／村らしい,
# 信用できる／信頼, 「吊り位置ではない」「吊らない」「吊るのは早い／怖い／違う」,
# 怪しくない／疑っていない, 残したい／後回し. Each (speaker, day, seat) is
# counted once.

_TRUST_WORDS = re.compile(
    r"(?<![面黒])白(?:い|く(?:見え|感じ|思)|っぽ|寄り|め|目|視)"
    r"|村(?:っぽ|寄り|目|らし|側(?:に見|っぽ|らし))|村人(?:っぽ|らし)"
    r"|信用(?:でき|して|する|が置|を置)|信頼"
    r"|(?:吊り|処刑)(?:位置|先|候補|どころ)(?:じゃ|では|には)な|(?:は|を)吊らな(?:い|くて|いで)"
    r"|吊る(?:の|ところ|位置|必要)(?:は|が)?(?:怖|早|違|弱|急|な)"
    r"|怪しく(?:な|は)|疑(?:って|わ)(?:ない|いない|なくて)|残し(?:たい|ておき)|後回し"
)


def trust_mention(sentence: str, names: Mapping[str, str]) -> list[tuple[str, int]]:
    """The seats a sentence places as trusted: (seat, end of the trust words)."""
    spans = mention_spans(sentence, names)
    placed: list[tuple[str, int]] = []
    for match in _TRUST_WORDS.finditer(sentence):
        before = [seat for _s, end, seat in spans if end <= match.start()]
        if before:
            placed.append((before[-1], match.end()))
    return placed


@dataclass
class GreyTrust:
    """Distinct (speaker, day, seat) grey trusts, by the speaker's and the seat's camp."""

    memo: Counter[tuple[str, str]] = field(default_factory=Counter)
    public: Counter[tuple[str, str]] = field(default_factory=Counter)
    memo_lines: int = 0
    memo_lines_with: int = 0
    memo_examples: list[Example] = field(default_factory=list)
    public_examples: list[Example] = field(default_factory=list)
    # By the speaker's skill tier, for section 6.
    memo_by_tier: Counter[str] = field(default_factory=Counter)
    public_by_tier: Counter[str] = field(default_factory=Counter)


def grey_trust(transcripts: Iterable[GameTranscript], *, examples: int = EXAMPLES) -> GreyTrust:
    found = GreyTrust()
    for transcript in transcripts:
        table = _Table(transcript)
        memo_seen: set[tuple[str, int, str]] = set()
        public_seen: set[tuple[str, int, str]] = set()
        for _k, utterance, at in table.model_lines():
            speaker, day = utterance.player_id, utterance.day
            camp = transcript.teams.get(speaker, "?")
            tier = transcript.skills.get(speaker)
            known = table.known_to(speaker, day)
            memo = utterance.reasoning_memo or {}
            if memo:
                found.memo_lines += 1
                trusted = [
                    str(seat)
                    for seat in memo.get("trusted") or []
                    if str(seat) not in known and table.grey(str(seat), day, at)
                ]
                found.memo_lines_with += bool(trusted)
                for seat in trusted:
                    if (speaker, day, seat) in memo_seen:
                        continue
                    memo_seen.add((speaker, day, seat))
                    found.memo[(camp, transcript.teams.get(seat, "?"))] += 1
                    if tier:
                        found.memo_by_tier[tier] += 1
                    if len(found.memo_examples) < examples:
                        found.memo_examples.append(
                            (
                                transcript.seed,
                                day,
                                _who(transcript, speaker),
                                f"trusted に {_who(transcript, seat)}",
                            )
                        )
            for sentence in split_sentences(utterance.text):
                for seat, _end in trust_mention(sentence, transcript.names):
                    if seat in known or not table.grey(seat, day, at):
                        continue
                    if (speaker, day, seat) in public_seen:
                        continue
                    public_seen.add((speaker, day, seat))
                    found.public[(camp, transcript.teams.get(seat, "?"))] += 1
                    if tier:
                        found.public_by_tier[tier] += 1
                    if len(found.public_examples) < examples:
                        found.public_examples.append(
                            (transcript.seed, day, _who(transcript, speaker), sentence)
                        )
    return found


# -- 4. memo hang targets on seats that had not spoken yet that day --


@dataclass
class UnspokenTargets:
    """Memo hang targets and suspects, and those on a seat with no line yet that day.

    Only targets on another living seat count. `*_day1` repeats each count
    for day 1, when no seat has spoken before the day's first line.
    """

    targets: int = 0
    targets_unspoken: int = 0
    targets_day1: int = 0
    targets_unspoken_day1: int = 0
    suspects: int = 0
    suspects_unspoken: int = 0
    suspects_day1: int = 0
    suspects_unspoken_day1: int = 0
    examples: list[Example] = field(default_factory=list)


def unspoken_targets(
    transcripts: Iterable[GameTranscript], *, examples: int = EXAMPLES
) -> UnspokenTargets:
    found = UnspokenTargets()
    for transcript in transcripts:
        table = _Table(transcript)
        for _k, utterance, at in table.model_lines():
            memo = utterance.reasoning_memo or {}
            speaker, day = utterance.player_id, utterance.day
            first = day == 1
            others = {
                seat for seat in transcript.teams if seat != speaker and table.alive(seat, day)
            }
            target = memo.get("execution_target")
            if target and str(target) in others:
                target = str(target)
                unspoken = not table.spoke_before(target, day, at)
                found.targets += 1
                found.targets_unspoken += unspoken
                found.targets_day1 += first
                found.targets_unspoken_day1 += first and unspoken
                if unspoken and len(found.examples) < examples:
                    later = table.first_line(target, day)
                    when = (
                        f"その日の初発言は {later - at} 行あと"
                        if later is not None
                        else "その日は発言なし"
                    )
                    found.examples.append(
                        (
                            transcript.seed,
                            day,
                            _who(transcript, speaker),
                            f"吊り先 {_who(transcript, target)}（{when}）",
                        )
                    )
            for seat in memo.get("suspects") or []:
                seat = str(seat)
                if seat not in others:
                    continue
                unspoken = not table.spoke_before(seat, day, at)
                found.suspects += 1
                found.suspects_unspoken += unspoken
                found.suspects_day1 += first
                found.suspects_unspoken_day1 += first and unspoken
    return found


# -- 5. the genuine seer's divination targets --
#
# Read in this order, the first that fits:
#
# * claimant -- the target had claimed a role before the night;
# * strong seat -- the seat is one the table will not hang or that leads it
#   (吊られにくい／吊れない／信用が高い／発言力／まとめ役／確定白に);
# * suspicion -- the seat stood out or drew suspicion (疑／怪し／気になる／
#   目立／票が割れ／投票の動き／様子見／黒っぽい／材料になる);
# * random -- no information (無作為／ランダム／情報がない／適当);
# * other.

_STRONG_SEAT = re.compile(
    r"吊(?:られ|り)?(?:にく|づら)|吊れな|信用(?:が|を)?(?:高|集|得)|発言力|まとめ役|進行役"
    r"|確定白|白(?:を)?確定|(?:村|白)として(?:残|確定)|残(?:りそう|る位置)"
)
_SUSPICION_PICK = re.compile(
    r"疑|怪し|気にな|目立|票が割れ|投票の動き|様子見|黒(?:っぽ|寄り|目|探し)|材料になる"
)
_RANDOM_PICK = re.compile(r"無作為|ランダム|情報が(?:何も)?な|材料が(?:何も)?な|適当")

SEER_PICK_TYPES = ("strong", "suspicion", "claimant", "random", "other")
SEER_PICK_LABELS = {
    "strong": "吊られにくい・強い位置",
    "suspicion": "疑い・目立ち（発言・投票）",
    "claimant": "CO済みの席（対抗など）",
    "random": "無作為・材料なし",
    "other": "その他",
}


def seer_pick_type(text: str, *, target_claimed: bool) -> str:
    if target_claimed:
        return "claimant"
    if _STRONG_SEAT.search(text):
        return "strong"
    if _SUSPICION_PICK.search(text):
        return "suspicion"
    if _RANDOM_PICK.search(text):
        return "random"
    return "other"


@dataclass
class SeerPicks:
    kinds: Counter[str] = field(default_factory=Counter)
    rows: list[tuple[int | None, int, str, str, str]] = field(default_factory=list)


def seer_picks(transcripts: Iterable[GameTranscript]) -> SeerPicks:
    """Each genuine seer's night action: (seed, night, target, type, its stated reason)."""
    found = SeerPicks()
    for transcript in transcripts:
        table = _Table(transcript)
        for utterance in transcript.utterances:
            if (
                utterance.kind != "night_action"
                or transcript.roles.get(utterance.player_id) != "seer"
            ):
                continue
            target = str(utterance.target or "")
            kind = seer_pick_type(
                utterance.text, target_claimed=table.claimed_by(target, utterance.day)
            )
            found.kinds[kind] += 1
            found.rows.append(
                (transcript.seed, utterance.day, _who(transcript, target), kind, utterance.text)
            )
    return found


# -- 6. by skill tier --
#
# An evaluation of one seat's line is a sentence that names the seat and, at
# most a few characters on, agrees with its line or doubts it:
#
# * agreeing -- 「Xの質問に同意」「Xに賛成」「Xの言う通り」「Xの指摘は一理ある」
#   「Xの答えは筋が通っている」「Xの疑問はもっともだ」「的を射ている」「納得」;
# * doubting -- 「Xの発言が引っかかる」「Xの理由は不自然」「違和感」「矛盾」
#   「筋が通らない」「おかしい」「怪しい」.
#
# The name must be followed by の and a word for a line (発言・質問・意見・
# 指摘・疑問・反論・答え・回答・見方・理由・主張・整理・提案・話) or by
# 「の言う通り」, or -- for agreeing only -- by 「に同意／に賛成」. A
# doubting word with なら after it (「レンさんが不自然なら」) is a condition,
# not an evaluation. The evaluation was right when the seat's camp matches
# its sign: an agreed-with seat on the village side, a doubted one on the
# werewolf or fox side. Only village-side speakers count -- the other camps
# know or pretend.

_HON = r"(?:さん|くん|君|ちゃん|氏|様)?"
_LINE_NOUN = (
    r"(?:最後の|昨日の|今日の|さっきの)?"
    r"(?:発言|質問|意見|指摘|疑問|反論|答え|回答|見方|理由|主張|整理|提案|話|考え)"
)
_AGREE = re.compile(
    r"同意|賛成|一理|筋が通(?!らな)|もっとも|的を射|納得|その通り|鋭い|いい(?:質問|指摘)"
)
_DOUBT = re.compile(r"引っかか|違和感|不自然|矛盾|筋が通らな|おかし|怪し")
_ABOUT_LINE = re.compile(_HON + r"\s*の\s*" + _LINE_NOUN + r"(?:[はがもにを、]|って)?")
_SAID_SO = re.compile(_HON + r"\s*の\s*言(?:う|った)通り")
_AGREED_WITH = re.compile(_HON + r"\s*に\s*(?:同意|賛成)")


def line_evaluations(sentence: str, names: Mapping[str, str]) -> list[tuple[str, int]]:
    """The seats whose line a sentence evaluates: (seat, +1 agreeing / -1 doubting)."""
    found: dict[str, int] = {}
    spans = mention_spans(sentence, names)
    for index, (_start, end, seat) in enumerate(spans):
        stop = spans[index + 1][0] if index + 1 < len(spans) else len(sentence)
        if _SAID_SO.match(sentence, end) or _AGREED_WITH.match(sentence, end):
            found.setdefault(seat, 1)
            continue
        about = _ABOUT_LINE.match(sentence, end)
        if not about:
            continue
        rest = sentence[about.end() : min(stop, about.end() + 16)]
        doubt = _DOUBT.search(rest)
        agree = _AGREE.search(rest)
        if doubt and not rest[doubt.end() :].lstrip("だ").startswith("なら"):
            found.setdefault(seat, -1)
        elif agree:
            found.setdefault(seat, 1)
    return list(found.items())


@dataclass
class TierRow:
    seats: int = 0
    ballots: int = 0
    content_ballots: int = 0
    ballots_day1: int = 0
    content_ballots_day1: int = 0
    evaluations: int = 0
    evaluations_right: int = 0
    agreeing: int = 0
    agreeing_right: int = 0
    doubting: int = 0
    doubting_right: int = 0


@dataclass
class ByTier:
    games_with_skills: int = 0
    games_without: int = 0
    rows: dict[str, TierRow] = field(default_factory=dict)
    memo_trust: Counter[str] = field(default_factory=Counter)
    public_trust: Counter[str] = field(default_factory=Counter)
    examples: list[Example] = field(default_factory=list)


def by_tier(transcripts: Sequence[GameTranscript], *, examples: int = EXAMPLES) -> ByTier:
    found = ByTier(rows={tier: TierRow() for tier in SKILL_TIERS})
    rated = [t for t in transcripts if t.skills]
    found.games_with_skills = len(rated)
    found.games_without = len(transcripts) - len(rated)
    trust = grey_trust(rated)
    found.memo_trust, found.public_trust = trust.memo_by_tier, trust.public_by_tier
    for transcript in rated:
        table = _Table(transcript)
        for tier in transcript.skills.values():
            if tier in found.rows:
                found.rows[tier].seats += 1
        for ballot in _ballots(transcript):
            row = found.rows.get(transcript.skills.get(ballot.player_id, ""))
            if row is None:
                continue
            content = "content" in vote_ground_types(ballot.text)
            row.ballots += 1
            row.content_ballots += content
            if ballot.day == 1:
                row.ballots_day1 += 1
                row.content_ballots_day1 += content
        seen: set[tuple[str, int, str]] = set()
        for _k, utterance, _at in table.model_lines():
            speaker, day = utterance.player_id, utterance.day
            row = found.rows.get(transcript.skills.get(speaker, ""))
            if row is None or transcript.teams.get(speaker) != "village":
                continue
            known = table.known_to(speaker, day)
            for sentence in split_sentences(utterance.text):
                for seat, sign in line_evaluations(sentence, transcript.names):
                    if seat in known or (speaker, day, seat) in seen:
                        continue
                    seen.add((speaker, day, seat))
                    right = (transcript.teams.get(seat) == "village") == (sign > 0)
                    row.evaluations += 1
                    row.evaluations_right += right
                    if sign > 0:
                        row.agreeing += 1
                        row.agreeing_right += right
                    else:
                        row.doubting += 1
                        row.doubting_right += right
                    if len(found.examples) < examples:
                        verdict = "当たり" if right else "外れ"
                        found.examples.append(
                            (
                                transcript.seed,
                                day,
                                f"{_who(transcript, speaker)}・"
                                f"{SKILL_LABELS[transcript.skills[speaker]]}",
                                f"{sentence} → {_who(transcript, seat)}を"
                                f"{'肯定' if sign > 0 else '疑い'}（{verdict}）",
                            )
                        )
    return found


# -- the report section --

GREY_READING_HEADING = "1日目の議論と灰の読み（参考・合否に数えない）"


def render_grey_reading(transcripts: list[GameTranscript]) -> str:
    """The six tables, pooled over the run's games, or "" with no recorded game."""
    games = [t for t in transcripts if (t.final_state or {}).get("chat_log")]
    if not games:
        return ""
    out = [
        f"この回の{len(games)}試合を合わせた件数です。どれも決まった規則で記録から数えた"
        "おおよその値で（`app/eval/grey_reading.py`）、当てはまった文を下に引くので確かめ"
        "られます。試合数が少ないので一つの欄は傾向ではなく例として読んでください。"
        "合否には数えません。「灰」は、その発言の時点で生きていて、COしておらず、"
        "占い結果（本物・騙りとも）が出ていない席です。発言者が役職で知っている席"
        "（自分・人狼の仲間・共有の相方・本物の占い師が占った先）は灰の読みに数えません。",
        "",
        *_render_divination_talk(divination_reason_talk(games)),
        "",
        *_render_vote_grounds(day1_vote_grounds(games)),
        "",
        *_render_grey_trust(grey_trust(games)),
        "",
        *_render_unspoken(unspoken_targets(games)),
        "",
        *_render_seer_picks(seer_picks(games)),
        "",
        *_render_by_tier(by_tier(games)),
    ]
    return "\n".join(out)


def _share(part: int, whole: int) -> str:
    return f"{part}/{whole}（{part / whole:.0%}）" if whole else "0/0"


def _examples(heading: str, examples: Sequence[Example]) -> list[str]:
    if not examples:
        return []
    return [
        "",
        heading,
        "",
        *(
            f"- seed {seed if seed is not None else '—'} {day}日目 {who}: 「{text}」"
            for seed, day, who, text in examples
        ),
    ]


def _render_divination_talk(found: DivinationTalk) -> list[str]:
    return [
        "### 1日目の発言のうち、0日目夜の占い理由を話した発言",
        "",
        "1日目にモデルが書いた公開発言（コードが書いた定型のCO・共有の発言は除く）のうち、"
        "どれかの文が占い先の選び方に触れたものです。「ランダム」「無作為」か、"
        "理由・なぜ・どうしてと「占／選／白先」が同じ文にあるもの。今夜・明日・次の占い先の"
        "話は数えません。",
        "",
        f"- 占い理由に触れた発言: {_share(found.matched, found.lines)}",
        *_examples(f"当てはまった文（先頭から最大{EXAMPLES}件）:", found.examples),
    ]


def _render_vote_grounds(found: VoteGrounds) -> list[str]:
    out = [
        "### 1日目の投票理由の型",
        "",
        "1日目の各席の最初の投票について、投票の発言と、モデルが投票に添えた決め手"
        "（`decision_evidence`）を別々に読みます。一票がいくつもの型に当たることがあり、"
        "どれにも当たらない票は「どれにも当たらない」です。",
        "",
        "| 型 | 投票の発言 | 決め手 |",
        "|---|---|---|",
    ]
    for kind in VOTE_GROUND_TYPES:
        out.append(
            f"| {VOTE_GROUND_LABELS[kind]} | {_share(found.line[kind], found.ballots)} | "
            f"{_share(found.evidence[kind], found.ballots)} |"
        )
    out += [
        "",
        f"- 投票の発言が「見えない・薄い」だけの票: {_share(found.thin_only, found.ballots)}",
    ]
    for kind in VOTE_GROUND_TYPES:
        out += _examples(f"「{VOTE_GROUND_LABELS[kind]}」の例:", found.examples.get(kind, []))
    return out


def _render_grey_trust(found: GreyTrust) -> list[str]:
    out = [
        "### 発言から灰を信用した数（結果ではなく）",
        "",
        "非公開メモの `trusted` に置いた灰と、公開発言で灰を白い・村っぽい・信用できる・"
        "今日の吊り位置ではない（「吊らない」「吊るのは早い」「怪しくない」「残したい」など）"
        "と置いた文を、発言者・日・相手の組ごとに1回数えます。公開発言は、その言葉より前で"
        "いちばん近い名前を相手とします。",
        "",
        "- メモに発言からの灰の信用があった発言: "
        f"{_share(found.memo_lines_with, found.memo_lines)}",
        "",
        "| 発言者の陣営 | メモ：相手が村人陣営 | 人狼陣営 | 妖狐陣営 | "
        "公開発言：相手が村人陣営 | 人狼陣営 | 妖狐陣営 |",
        "|---|---|---|---|---|---|---|",
    ]
    for camp in ("village", "werewolf", "fox"):
        cells = [str(found.memo[(camp, other)]) for other in ("village", "werewolf", "fox")]
        cells += [str(found.public[(camp, other)]) for other in ("village", "werewolf", "fox")]
        out.append(f"| {team_label(camp)} | {' | '.join(cells)} |")
    out += _examples("メモの例:", found.memo_examples)
    out += _examples("公開発言の例:", found.public_examples)
    if not found.public_examples:
        out += ["", "公開発言で灰を信用した文はありません。"]
    return out


def _render_unspoken(found: UnspokenTargets) -> list[str]:
    return [
        "### まだその日に発言していない席への吊り先",
        "",
        "モデルが書いた公開発言ごとに、そのときの非公開メモの吊り先（`execution_target`）と"
        "疑い先（`suspects`）が、その日まだ一度も公開発言をしていない席だった数です"
        "（自分と死んだ席は除く）。1日目は、その日の最初の発言の前には誰も話していません。",
        "",
        "| | 全日 | うち1日目 |",
        "|---|---|---|",
        f"| 吊り先が未発言の席 | {_share(found.targets_unspoken, found.targets)} | "
        f"{_share(found.targets_unspoken_day1, found.targets_day1)} |",
        f"| 疑い先が未発言の席 | {_share(found.suspects_unspoken, found.suspects)} | "
        f"{_share(found.suspects_unspoken_day1, found.suspects_day1)} |",
        *_examples(f"吊り先の例（先頭から最大{EXAMPLES}件）:", found.examples),
    ]


def _render_seer_picks(found: SeerPicks) -> list[str]:
    out = [
        "### 本物の占い師の占い先の理由",
        "",
        "本物の占い師の夜行動に添えた理由を、先に当たった型で数えます。相手がその夜までに"
        "COしていれば「CO済みの席」、次に「吊られにくい／吊れない／信用が高い／発言力／"
        "まとめ役」なら強い位置、「疑／怪し／気になる／目立／票が割れ／投票の動き／様子見」"
        "なら疑い・目立ち、「無作為／ランダム／情報がない」なら材料なし、残りはその他です。",
        "",
        "| 型 | 件数 |",
        "|---|---|",
    ]
    out += [f"| {SEER_PICK_LABELS[kind]} | {found.kinds[kind]} |" for kind in SEER_PICK_TYPES]
    if found.rows:
        out += ["", "各夜:", ""]
        out += [
            f"- seed {seed if seed is not None else '—'} {night}日目夜 → {who}"
            f"（{SEER_PICK_LABELS[kind]}）: 「{text}」"
            for seed, night, who, kind, text in found.rows
        ]
    return out


def _render_by_tier(found: ByTier) -> list[str]:
    out = [
        "### 熟練度ごと",
        "",
        "席の熟練度（熟練・普通・初心者）ごとに、投票の発言が中身を読んだ型（上の"
        "「発言・行動の中身」）だった票、発言からの灰の信用（上と同じ数え方）、"
        "ほかの席の一つの発言への評価（「Xの質問に同意」「Xの言う通り」「Xの指摘は一理ある」"
        "「Xの答えは筋が通っている」、「Xの発言が引っかかる」「Xの理由は不自然」など、"
        "名前のすぐ後に「の＋発言・質問・指摘・答え…」か「の言う通り」「に同意／賛成」が"
        "続くもの）が、その席の本当の陣営と合っていたか（肯定した相手が村人陣営、"
        "疑った相手が人狼・妖狐陣営なら当たり）です。評価は村人陣営の発言者だけを数えます"
        "（ほかの陣営は答えを知っているか、演じているため）。",
        "",
    ]
    if not found.games_with_skills:
        return [*out, "記録なし（この回の記録には席の熟練度がありません）。"]
    if found.games_without:
        out += [f"熟練度の記録がない{found.games_without}試合は数えていません。", ""]
    out += [
        "| 熟練度 | 席 | 中身を読んだ票（全日） | うち1日目 | 灰の信用：メモ | "
        "灰の信用：公開発言 | 発言への評価の当たり | 肯定の当たり | 疑いの当たり |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for tier in SKILL_TIERS:
        row = found.rows[tier]
        out.append(
            f"| {SKILL_LABELS[tier]} | {row.seats} | {_share(row.content_ballots, row.ballots)} | "
            f"{_share(row.content_ballots_day1, row.ballots_day1)} | {found.memo_trust[tier]} | "
            f"{found.public_trust[tier]} | {_share(row.evaluations_right, row.evaluations)} | "
            f"{_share(row.agreeing_right, row.agreeing)} | "
            f"{_share(row.doubting_right, row.doubting)} |"
        )
    out += _examples("発言への評価の例:", found.examples)
    return out
