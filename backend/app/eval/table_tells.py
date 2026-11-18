"""What a human who plays the AI table again and again could learn to read.

The owner's review of run 14 (X7) asked for the regularities a repeat player
would pick up from outcomes alone, counted per run and pooled over its games:

1. the order of the day-1 claims of each role, genuine against fake, by the
   fake claimant's camp -- if the genuine seer always claims first, the
   table has a tell;
2. seats that publicly conceded a point, and how often they were executed
   that day or the next, village side against wolf side -- run 13's seed 3
   hanged three villagers who admitted a weakness;
3. executions by personality and camp -- a personality the table hangs
   whatever its role is one a player learns to ignore.

Plain counts, read from the record's final state with no model: facts the
table saw (claims, lines, executions) beside the roles it did not. Samples
are a handful of games, so a cell is an example, not a rate. Kept out of the
release gate.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from app.eval.reasoning_analyzer import mention_spans, split_sentences
from app.eval.transcript import GameTranscript
from app.eval.transcript_render import role_label, team_label

CAMPS = ("village", "werewolf", "fox")

# -- 1. the order of the day-1 claims --


@dataclass
class ClaimOrder:
    """Day-1 claims of each role by their place in the day's claim order.

    `genuine[(role, place)]` counts claimants who hold the role;
    `fake[(role, place, camp)]` the ones who do not, by their own camp.
    """

    genuine: Counter[tuple[str, int]] = field(default_factory=Counter)
    fake: Counter[tuple[str, int, str]] = field(default_factory=Counter)

    def roles(self) -> list[str]:
        found = {role for role, _place in self.genuine} | {role for role, _p, _c in self.fake}
        return sorted(found)

    def places(self, role: str) -> list[int]:
        found = {place for r, place in self.genuine if r == role}
        found |= {place for r, place, _camp in self.fake if r == role}
        return sorted(found)


def claim_order(transcripts: Iterable[GameTranscript]) -> ClaimOrder:
    """Each seat's first day-1 claim of a role, placed by the line it was said in.

    Place 1 is the first seat of the day to claim that role. A seat that
    claims the same role again later in the day is counted once.
    """
    order = ClaimOrder()
    for transcript in transcripts:
        state = transcript.final_state or {}
        position = {
            str(message.get("message_id")): index
            for index, message in enumerate(state.get("chat_log", []) or [])
        }
        claims = sorted(
            (
                claim
                for claim in state.get("co_declarations", []) or []
                if int(claim.get("day", -1)) == 1 and claim.get("claimed_role")
            ),
            key=lambda claim: position.get(str(claim.get("source_message_id")), len(position)),
        )
        seen: set[tuple[str, str]] = set()
        places: Counter[str] = Counter()
        for claim in claims:
            seat, role = str(claim.get("player_id")), str(claim.get("claimed_role"))
            if (seat, role) in seen:
                continue
            seen.add((seat, role))
            places[role] += 1
            if transcript.roles.get(seat) == role:
                order.genuine[(role, places[role])] += 1
            else:
                order.fake[(role, places[role], transcript.teams.get(seat, "?"))] += 1
    return order


# -- 2. conceding a point in public --
#
# A sentence concedes when the speaker grants something about its own play,
# read from three forms and nothing else:
#
# * 「認める」「認めます」 in the present (「そこは認める」, 「理由が薄いのは
#   認めます」) -- not 「認めた」 (mostly about someone else: 「流れ任せは
#   認めたので」), 「認めてくれて」, 「認めるなら」 or a negative;
# * withdrawing: 「そこは引く」「ここは引きます」, 「引き下がる」, 「撤回する」;
# * the speaker's own thing called weak: 「自分の理由は弱かった」, 「自分でも
#   弱かったと思う」 (a first-person word, then 弱い／弱かった).
#
# And no other seat stands as the sentence's subject (a name followed by
# は／が／も): 「ユイは根拠が弱いと認める」 is about ユイ. A name called at the
# head (「マコトさん、理由が薄いのは認めます」) is not a subject.

_CONCEDES = re.compile(
    r"認め(?:る|ます)(?!な|べき|ん|ませ)"
    r"|(?:そこ|ここ)は引(?:く|きます)|引き下が|撤回(?:する|します)"
    r"|(?:自分|私|俺|僕|わたし|あたし|オレ|ボク)(?:の[^。、！？]{0,12}?|でも)(?:は|が|も)?弱(?:い|かった)"
)
_HONORIFIC_SUBJECT = re.compile(r"(?:さん|くん|君|ちゃん|氏|様)?\s*[はがも]")


def concedes(sentence: str, names: Mapping[str, str]) -> bool:
    """Whether a sentence concedes a point of the speaker's own (see the rule above)."""
    if not _CONCEDES.search(sentence):
        return False
    return not any(
        _HONORIFIC_SUBJECT.match(sentence, end)
        for _start, end, _seat in mention_spans(sentence, names)
    )


@dataclass
class Concessions:
    """Seat-days by camp: those that conceded, and the seat-days that spoke without one.

    Each counter is keyed by camp; `*_executed_same` / `*_executed_next` count
    the seat-days whose seat was executed that day or the next.
    """

    conceded: Counter[str] = field(default_factory=Counter)
    conceded_executed_same: Counter[str] = field(default_factory=Counter)
    conceded_executed_next: Counter[str] = field(default_factory=Counter)
    other: Counter[str] = field(default_factory=Counter)
    other_executed_same: Counter[str] = field(default_factory=Counter)
    other_executed_next: Counter[str] = field(default_factory=Counter)
    # A few of the conceding sentences, for a reader to check the rule.
    examples: list[tuple[int | None, int, str, str]] = field(default_factory=list)


def concessions(transcripts: Iterable[GameTranscript], *, examples: int = 20) -> Concessions:
    """Every seat-day with a public line, split by whether a line conceded."""
    found = Concessions()
    for transcript in transcripts:
        state = transcript.final_state or {}
        executed = {
            str(death.get("player_id")): int(death.get("day", 0))
            for death in state.get("death_records", []) or []
            if death.get("cause") == "executed"
        }
        spoke: dict[tuple[str, int], str | None] = {}
        for message in state.get("chat_log", []) or []:
            if message.get("channel") != "public":
                continue
            seat, day = str(message.get("author_id")), int(message.get("day", -1))
            if seat not in transcript.teams:
                continue
            key = (seat, day)
            spoke.setdefault(key, None)
            if spoke[key] is None:
                spoke[key] = next(
                    (
                        sentence
                        for sentence in split_sentences(str(message.get("content") or ""))
                        if concedes(sentence, transcript.names)
                    ),
                    None,
                )
        for (seat, day), sentence in spoke.items():
            camp = transcript.teams[seat]
            hanged = executed.get(seat)
            if sentence is not None:
                found.conceded[camp] += 1
                found.conceded_executed_same[camp] += hanged == day
                found.conceded_executed_next[camp] += hanged == day + 1
                if len(found.examples) < examples:
                    name = transcript.names.get(seat, seat)
                    found.examples.append(
                        (transcript.seed, day, f"{name}（{team_label(camp)}）", sentence)
                    )
            else:
                found.other[camp] += 1
                found.other_executed_same[camp] += hanged == day
                found.other_executed_next[camp] += hanged == day + 1
    return found


# -- 3. executions by personality --


@dataclass
class PersonalityExecutions:
    """`seated[(personality, camp)]` seats played, `executed[...]` of them hanged."""

    seated: Counter[tuple[str, str]] = field(default_factory=Counter)
    executed: Counter[tuple[str, str]] = field(default_factory=Counter)


def personality_executions(transcripts: Iterable[GameTranscript]) -> PersonalityExecutions:
    found = PersonalityExecutions()
    for transcript in transcripts:
        state = transcript.final_state or {}
        hanged = {
            str(death.get("player_id"))
            for death in state.get("death_records", []) or []
            if death.get("cause") == "executed"
        }
        for seat, personality in transcript.personalities.items():
            camp = transcript.teams.get(seat, "?")
            found.seated[(personality, camp)] += 1
            found.executed[(personality, camp)] += seat in hanged
    return found


# -- the report section --

TABLE_TELLS_HEADING = "常連の人間が読み取れる傾向（参考・合否に数えない）"


def render_table_tells(transcripts: list[GameTranscript]) -> str:
    """The three tables, pooled over the run's games, or "" with no recorded game."""
    games = [t for t in transcripts if (t.final_state or {}).get("chat_log")]
    if not games:
        return ""
    out = [
        f"この回の{len(games)}試合を合わせた、そのままの件数です。試合数が少ないので、"
        "一つの欄は傾向ではなく例として読んでください。卓に見えた事実（CO・発言・処刑）"
        "と、卓に見えない役職を並べただけで、合否には数えません。",
        "",
        *_render_claim_order(claim_order(games)),
        "",
        *_render_concessions(concessions(games)),
        "",
        *_render_personalities(personality_executions(games)),
    ]
    return "\n".join(out)


def _render_claim_order(order: ClaimOrder) -> list[str]:
    out = [
        "### 1日目のCOの順番（役職ごと、本物と騙り）",
        "",
        "その役職を1日目に名乗った席を、名乗った発言の順に1番目・2番目…と数えます"
        "（同じ日の再表明は1回）。騙りは、名乗った席の陣営で分けます。",
        "",
    ]
    roles = order.roles()
    if not roles:
        return [*out, "(1日目のCOはありません)"]
    out += [
        "| 役職 | 順番 | 本物 | 騙り（人狼陣営） | 騙り（妖狐陣営） | 騙り（村人陣営） |",
        "|---|---|---|---|---|---|",
    ]
    for role in roles:
        for place in order.places(role):
            out.append(
                f"| {role_label(role)} | {place}番目 | {order.genuine[(role, place)]} | "
                f"{order.fake[(role, place, 'werewolf')]} | {order.fake[(role, place, 'fox')]} | "
                f"{order.fake[(role, place, 'village')]} |"
            )
    return out


def _render_concessions(found: Concessions) -> list[str]:
    out = [
        "### 公開の場で非を認めた席と、その日・翌日の処刑",
        "",
        "席と日の組ごとに数えます。その日の公開発言のどれかの文が、自分の非を認める"
        "形（「認める／認めます」、「そこは引く」「撤回する」、「自分の理由は弱かった」の"
        "ような一人称の「弱い」）で、ほかの席が主語（名前＋は・が・も）でないものを"
        "「認めた」とします（`app/eval/table_tells.py`）。比べるために、発言したが"
        "認めなかった席と日の組も並べます。",
        "",
        "| 陣営 | 認めた席日 | うち同じ日に処刑 | うち翌日に処刑 | 認めなかった席日 | "
        "うち同じ日に処刑 | うち翌日に処刑 |",
        "|---|---|---|---|---|---|---|",
    ]
    for camp in CAMPS:
        if not found.conceded[camp] and not found.other[camp]:
            continue
        out.append(
            f"| {team_label(camp)} | {found.conceded[camp]} | "
            f"{found.conceded_executed_same[camp]} | {found.conceded_executed_next[camp]} | "
            f"{found.other[camp]} | {found.other_executed_same[camp]} | "
            f"{found.other_executed_next[camp]} |"
        )
    if found.examples:
        out += ["", "認めたと読んだ文（席日ごとに最初の一文、先頭から最大20件）:", ""]
        out += [
            f"- seed {seed if seed is not None else '—'} {day}日目 {who}: 「{sentence}」"
            for seed, day, who, sentence in found.examples
        ]
    return out


def _render_personalities(found: PersonalityExecutions) -> list[str]:
    out = [
        "### 人格ごとの処刑（陣営別）",
        "",
        "各欄は「処刑された数／その人格・陣営で座った数」です。",
        "",
        "| 人格 | 村人陣営 | 人狼陣営 | 妖狐陣営 | 計 |",
        "|---|---|---|---|---|",
    ]
    personalities = sorted({personality for personality, _camp in found.seated})
    for personality in personalities:
        cells = [
            f"{found.executed[(personality, camp)]}/{found.seated[(personality, camp)]}"
            if found.seated[(personality, camp)]
            else "—"
            for camp in CAMPS
        ]
        total_executed = sum(found.executed[(personality, camp)] for camp in CAMPS)
        total_seated = sum(found.seated[(personality, camp)] for camp in CAMPS)
        out.append(f"| {personality} | {' | '.join(cells)} | {total_executed}/{total_seated} |")
    return out
