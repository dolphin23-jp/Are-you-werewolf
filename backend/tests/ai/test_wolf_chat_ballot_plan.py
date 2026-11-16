"""Which wolf-chat lines are a plan to vote a partner, read through the real ballot path.

A werewolf's vote on its own partner passes the release gate only when it was
planned: the team agreed on it in the wolf chat, or the table already marks
the partner. The first version of the wolf-chat rule accepted any sentence
that named the partner and contained a ballot word anywhere, so a line about
the partner's own vote ("ダイキはミオに投票して"), a line refusing to vote the
partner ("ダイキを吊るのは避けたい") or a second vocative counted as a plan,
and a wrong game passed. It also missed ordinary plans (に入れる, を落とす,
に乗る, a partner addressed as お前), so a correct game failed. The partner
must now be the object of the ballot phrase, and the clause must not turn it
around. Every case below goes through `_cast_vote` on a board where nothing
public marks the partner, so the wolf chat alone decides.
"""

from __future__ import annotations

import pytest

from app.ai.coordinator import AICoordinator
from app.eval.reasoning_analyzer import ReasoningTranscriptAnalyzer
from tests.ai import first_claude_game as game1
from tests.ai.test_vote_audit import (
    _ally_vote,
    _latest_audit,
    _name,
    _wolf_line,
    _wolf_pair_board,
)

# {a}: the partner the voter then votes. {o}, {b}: two other seats.
NOT_A_PLAN = [
    # the partner is not the object of the ballot phrase
    "{a}と私で{o}に投票しよう。",
    "{a}は{o}に投票して。",
    "{a}は票を{o}に集めて。",
    "{a}が進行を仕切ってくれ。",
    "{a}は吊り先を{o}に誘導して。",
    # vocatives, one or several
    "{b}、{a}、明日は{o}を吊ろう。",
    "{a}、明日は{o}に投票しよう。",
    "{a}、今夜はケント噛みでいこう。",
    # the clause turns the ballot around
    "{a}を吊るのは避けたい。",
    "{a}の吊りだけは絶対に避けたい。",
    "{a}への投票は絶対ダメ。",
    "{a}を処刑から守りたい。",
    "{a}が処刑されないように動こう。",
    "{a}を犠牲にはできない。",
    "{a}吊りは避けたい。",
    "{a}には投票しない。",
]

A_PLAN = [
    "明日は{a}を切って信用を取ってくれ。",
    "悪いけど明日は{a}に入れるね。",
    "{a}を落とそう。",
    "{a}に乗る。",
    "明日は{a}に票を合わせて身内票で白く見せる。",
    "{a}吊りでいこう。",
    "{a}を吊ろう。",
    "{a}への投票で信用を取る。",
    "{a}は切ろう。",
    "{a}を見捨てる。",
    # the partner addressed alone, then called お前
    "{a}、すまんが明日はお前に投票する。",
]

# Beyond the reviewed cases: shapes the same rule has to get right.
MORE_NOT_A_PLAN = [
    # the partner is the one voting, or the one followed
    "{a}は処刑票を{o}に入れて。",
    "{a}は吊りに賛成して。",
    "{a}に合わせて動こう。",
    "{a}の案に乗る。",
    "{a}は切り札だ。",
    # done to the partner by the village, not by the team
    "村が{a}を吊りに来たら{o}に流そう。",
    "{a}を吊られたら負けだ。",
    # decided against, not obliged to
    "{a}を吊らないと決めた。",
    "{a}を切らずに残そう。",
    "{a}を吊るのは待とう。",
    "{a}には悪いけど明日は{o}に投票する。",
    # お前 is the partner only when the partner alone was addressed
    "{b}、{a}、明日はお前に投票する。",
    "{b}、明日はお前に投票する。",
]

MORE_A_PLAN = [
    "{a}を切るしかない。",
    "{a}を吊らないといけない。",
    "{a}に一票。",
    "{a}さんに投票しよう。",
    "{a}には悪いが切る。",
    # the address carries to the next sentence of the same message
    "{a}、すまん。明日はお前を切る。",
]


def _planned(template: str, author: str | None = None) -> bool:
    state, voter, ally = _wolf_pair_board()
    others = [pid for pid in state.alive_ids() if pid not in (voter, ally, "p0")]
    text = template.format(
        a=_name(state, ally), o=_name(state, others[0]), b=_name(state, others[1])
    )
    _wolf_line(state, author or voter, text, day=1)

    recorder, _ = _ally_vote(state, voter, ally)

    audit = _latest_audit(recorder, voter, state.day)
    assert audit.ally_vote is True
    report = ReasoningTranscriptAnalyzer().analyze(recorder.transcript)
    assert report.unplanned_wolf_ally_vote_count == int(not audit.ally_vote_planned)
    if audit.ally_vote_planned:
        plan = recorder.transcript.wolf_ally_vote_plans[-1]
        assert plan.basis == "wolf_chat"
    return audit.ally_vote_planned


@pytest.mark.parametrize("template", NOT_A_PLAN + MORE_NOT_A_PLAN)
def test_a_line_that_does_not_put_the_partner_up_for_the_ballot_is_no_plan(template: str):
    assert _planned(template) is False


@pytest.mark.parametrize("template", A_PLAN + MORE_A_PLAN)
def test_a_line_that_puts_the_partner_up_for_the_ballot_is_a_plan(template: str):
    assert _planned(template) is True


def test_the_reviewed_cases_are_all_here():
    """The 26 cases of the review's fix script, and the お前 case of its repro."""
    assert len(NOT_A_PLAN) + len(A_PLAN) == 27


@pytest.mark.parametrize(
    ("template", "planned"),
    [
        ("明日は俺を切って信用を取れ。", True),
        ("明日は私に投票して。", True),
        ("{a}と私で{o}に投票しよう。", False),
        ("俺は潜伏で{o}偽寄りに流す。", False),
        ("お前は俺を支えてくれ。", False),
    ],
)
def test_the_partner_offering_itself_up_in_the_first_person_is_a_plan(
    template: str, planned: bool
):
    """Written by the partner: 俺/私 is the partner."""
    assert _planned(template, author="p2") is planned


def test_first_person_from_anyone_but_the_partner_is_not_the_partner():
    assert _planned("明日は俺を切って信用を取れ。") is False


def test_the_first_games_wolf_chat_plans_no_vote_on_a_partner():
    """Three vocatives in a row, お前 and 俺 throughout: none is a ballot on a partner."""
    roles = game1.load_transcript()["roles"]
    wolves = [pid for pid, role in roles.items() if role == "werewolf"]
    for day in range(2, 7):
        state = game1.state_at_vote(day)
        for ally in wolves:
            assert AICoordinator._wolf_chat_ballot_plan(state, ally) is None, (day, ally)
