"""The vote prompt's memo-target line on realistic lines and on the first Claude game.

The previous line judged whether the seat had put its memo target up for the
hang, and a review ran it on two sets: a battery of 67 hand-written lines
about one seat, and every (seat, day) of the first live Claude game. It was
wrong both ways. It missed pushes worded 「灰はレン優先のまま押す」 or
「名前を挙げるならミオ」 and so told those seats they had not named a target
they had pushed. It also counted polite refusals and reported speech as pushes.

The line now states only what code knows: whether the seat's own lines today
contain the name, and if they do, the seat's own sentences that name it. Each
case below pins that fact. The name is absent, so the flag shows; the name is
present, so its sentences are quoted verbatim, and the line says so when they
are an excerpt.

Two sentences cannot hold a seat's day. The first game has seat-days whose
designation the quote leaves out, and in that game a seat wrote a true
『昼に言った通り』 about such a designation. So the vote instruction scopes
『昼に言った』 to the day's log, which the prompt carries whole, and the tests
below pin that the left-out designations are there.
"""

from __future__ import annotations

import re

import pytest

from app.engine.state import ChatChannel, GameState
from tests.ai.first_claude_game import load_transcript, state_at_vote
from tests.ai.test_vote_memo_target import (
    HEADING,
    MEMO,
    NOT_SAID,
    SAID_IN_DAYS_LOG,
    _chat_builder,
    _day_two,
    _say,
    excerpted,
    flagged,
    quoted,
)

# -- the review's battery: 67 lines about ソウタ, some about レン too --

NAMES = {
    "p0": "マコト",
    "p1": "アカリ",
    "p2": "ハルト",
    "p3": "ユイ",
    "p4": "ソウタ",
    "p5": "ミオ",
    "p6": "レン",
    "p7": "サクラ",
    "p8": "カイト",
    "p9": "ノゾミ",
    "p10": "リク",
    "p11": "ツムギ",
    "p12": "ダイキ",
    "p13": "ホノカ",
    "p14": "シオン",
    "p15": "アオイ",
    "p16": "ケント",
}
SPEAKER = "p0"

# Lines that push ソウタ. The old detector missed 「ソウタに一票」「ソウタは黒だと
# 思う」「ソウタは狼候補」 and every clause that said 昨日.
PUSHING = [
    "今日はソウタ吊りで。",
    "ソウタさんに入れます。",
    "投票先はソウタ。",
    "ソウタさんを吊りたい。",
    "ソウタに一票。",
    "ソウタに1票入れる。",
    "私はソウタ推し。",
    "ソウタを推します。",
    "ソウタが一番怪しい。",
    "ソウタさんは人狼だと思う。",
    "ソウタは黒だと思う。",
    "ソウタ狼で見てる。",
    "ソウタさんで決まりかな。",
    "ソウタ一択。",
    "ソウタを吊るべき。",
    "ソウタを吊ろう。",
    "ソウタさんに投票しようと思う。",
    "ソウタさんに投票します。",
    "みんなソウタに入れてね。",
    "ソウタに入れてほしい。",
    "昨日の発言からソウタが一番怪しい。",
    "昨日の投票先を見るとソウタが狼っぽい。",
    "ソウタさん、やっぱり怪しいです。",
    "ソウタを吊りに行きたい。",
    "今日の吊りはソウタさんでお願いします。",
    "吊るならソウタ。",
    "ソウタさんが怪しいので、ソウタさんに投票します。",
    "ソウタの発言が怪しい。",
    "ソウタは偽物だと思う。",
    "ソウタさんは偽だと思う。",
    "ソウタは狼候補。",
    "ソウタが最有力。",
    "ソウタが本命。",
    "ソウタ吊り希望。",
    "ソウタ吊り。",
    "ソウタさんを処刑したい。",
    "ソウタは怪しすぎる。",
    "ソウタの動きが一番狼らしい。",
]
# Lines that name ソウタ without pushing: refusals in plain and polite form,
# reported speech, questions. The old detector read 12 of them as pushes.
NOT_PUSHING = [
    "ソウタは吊らない。",
    "ソウタ吊りはない。",
    "ソウタは怪しくない。",
    "ソウタは怪しいとは思わない。",
    "ソウタが怪しいという意見には反対です。",
    "ソウタを吊るなんてありえない。",
    "ソウタを吊る理由が見当たらない。",
    "ソウタに投票するつもりはありません。",
    "ソウタじゃなくてレンを吊りたい。",
    "ソウタよりレンが怪しい。",
    "レンさんはソウタが怪しいと言っていたけど根拠が弱い。",
    "みんなソウタが怪しいと言ってるけど、私は違うと思う。",
    "ソウタさん、私の投票先はレンです。",
    "ソウタさんの意見に賛成で、レン吊りでいい。",
    "ソウタさんは白なので吊り対象から外します。",
    "ソウタは村っぽい。",
    "ソウタさんが言う通りレンを吊ろう。",
    "ソウタさんの言う通り、レンが怪しい。",
    "ソウタを吊る派の人、理由を教えて。",
    "ソウタさん吊りを推してる人が多いけど、まだ決めたくない。",
    "ソウタを疑う声もあるけど、私は信じてる。",
    "ソウタさんはむしろ信用できる。",
    "ソウタの占い結果は白だった。",
    "占い師のソウタが白を出したレンは吊らない。",
    "ソウタを吊りたい人は多いと思うけど、私は反対。",
    "ソウタ吊りは最終手段。",
    "ソウタさんは昨日怪しかったけど今日は白っぽい。",
    "ソウタさんが占いCOしたので、占い候補は2人。",
]
# Polite refusals the review added on top of the battery.
POLITE_REFUSALS = [
    "ソウタさんを吊るつもりはありません。",
    "ソウタさんを吊る理由はありません。",
    "ソウタさんに投票する気はありません。",
    "ソウタさんを吊るのは賛成できません。",
    "ソウタさんを疑う根拠は特にありません。",
    "ソウタさんが怪しいとは思っていません。",
    "ソウタさんを吊るべきとは考えていません。",
    "ソウタさん吊りの流れには乗りません。",
]
ONE_SENTENCE = PUSHING + NOT_PUSHING + POLITE_REFUSALS


def _battery_state() -> GameState:
    state = _day_two()
    for player_id, name in NAMES.items():
        state.players[player_id].name = name
    return state


def _battery_line(said: str, target: str) -> str:
    state = _battery_state()
    _say(state, SPEAKER, said, day=2)
    _system, messages = _chat_builder(state).build_vote_context(
        state, SPEAKER, ["p4", "p6"], board_memo=MEMO, stated_target=target
    )
    return next(line for line in messages[0].content.splitlines() if line.startswith(HEADING))


def test_the_battery_is_the_reviews_whole_set():
    assert (len(PUSHING), len(NOT_PUSHING) + 1) == (38, 29)  # + the question below
    assert all(line.count("。") == 1 and line.endswith("。") for line in ONE_SENTENCE)


@pytest.mark.parametrize("said", ONE_SENTENCE)
def test_a_line_naming_the_target_is_quoted_as_said(said: str) -> None:
    assert _battery_line(said, "p4") == quoted("ソウタ", said.removesuffix("。"))


@pytest.mark.parametrize("said", ONE_SENTENCE)
def test_another_seat_is_quoted_only_where_the_line_names_it(said: str) -> None:
    expected = quoted("レン", said.removesuffix("。")) if "レン" in said else flagged("レン")

    assert _battery_line(said, "p6") == expected


def test_a_question_to_the_target_quotes_the_sentence_that_names_it():
    said = "ソウタさんへの質問です。昨日の投票理由を教えて。"

    assert _battery_line(said, "p4") == quoted("ソウタ", "ソウタさんへの質問です")
    assert _battery_line(said, "p6") == flagged("レン")


# -- the first Claude game: every (seat, day) with a memo target --

# What the vote prompt's line quotes for each seat on each day it spoke, from
# the committed transcript, with the memo target the seat's last discussion turn
# that day held; () is the flag. The flag shows on exactly the four pairs whose
# lines never contain the name. ユイ and サクラ on day 2 and カイト on day 6
# pushed the name in words the old detector lacked, and it flagged them; each
# now sees the sentence it said. Every other line is labelled an excerpt, except
# the pairs in SHOWN_WHOLE.
FIRST_GAME: dict[tuple[int, str, str], tuple[str, ...]] = {
    (2, "p3", "p6"): (  # ユイ, memo target レン; a push the old detector missed
        "灰はレン優先のまま押す！",
        "訂正は素直だったけど、灰の中ではレン優先で見る方針は変えない！",
    ),
    (2, "p5", "p6"): (),  # ミオ, memo target レン
    (2, "p6", "p12"): (  # レン, memo target ダイキ
        "呪殺の筋が通るのはアオイの方だから、今んとこダイキがいちばん気になってるよ〜",
        "ダイキとアオイが昨夜どこを占って、それが今朝の死体と合ってるか、結果待ってから考えるよ〜",
    ),
    (2, "p7", "p6"): (  # サクラ, memo target レン; likewise
        "ユイと同じく、灰ではレン優先のまま",
        "灰で見るならレン",
    ),
    (2, "p11", "p5"): (  # ツムギ, memo target ミオ
        "マコトへの答え、灰で一番気になるのはミオかな",
    ),
    (2, "p12", "p6"): (),  # ダイキ, memo target レン
    (2, "p15", "p6"): (),  # アオイ, memo target レン
    (3, "p0", "p12"): (  # マコト, memo target ダイキ; its designation is left out
        "ダイキさん、最後に反論があればどうぞ",
        "アオイさんのグレー詰めの意見も分かるので、ホノカさんと相談して、"
        "ダイキさん吊りか灰吊りかを仮指定で出しますね",
    ),
    (3, "p1", "p12"): (  # アカリ, memo target ダイキ
        "ダイキさん、この点はどう説明されますか？",
    ),
    (3, "p5", "p11"): (  # ミオ, memo target ツムギ
        "1日目時点でハルトさんへの決め手が弱く、"
        "ツムギさんの発言が様子見に寄っていたのが気になったためです",
    ),
    (3, "p10", "p12"): (  # リク, memo target ダイキ
        "俺はダイキ吊りで変えない",
        "ダイキ、反論があるなら聞こう",
    ),
    (3, "p11", "p12"): (  # ツムギ, memo target ダイキ
        "今日はダイキ吊りで行くよ",
        "ただ一個足すと、ダイキは0日目にソウタを占って溶かせてないから、どっちにしろ偽寄り",
    ),
    (3, "p12", "p15"): (  # ダイキ, memo target アオイ
        "でもアオイのソウタ結果が出たのは二死体を見た後の2日目朝だ",
        "アオイ、俺が白を出したソウタを重ねて占ったのはなぜだ？",
    ),
    (3, "p15", "p11"): (),  # アオイ, memo target ツムギ
    (4, "p0", "p11"): (  # マコト, memo target ツムギ; its designation is left out
        "ツムギさん、アオイさんの黒にどう反論しますか？",
        "ただ、ツムギさんの反論はまだ聞けていないので、先に聞かせてください…",
    ),
    (4, "p8", "p11"): (  # カイト, memo target ツムギ
        "ツムギ吊りで検証、白ならアオイ狐か狂で即処理",
        "今日はツムギ吊りでいい",
    ),
    (4, "p11", "p8"): (  # ツムギ, memo target カイト
        "仲間に吊り票は入れにくいし、囲いならシオンよりカイトの方を見たいな",
    ),
    (4, "p15", "p11"): (  # アオイ, memo target ツムギ
        "今日はツムギさん吊りをお願いしたいです",
        "お待たせしました、昨夜はツムギさんを占って、結果は黒です",
    ),
    (5, "p0", "p15"): (  # マコト, memo target アオイ
        "でも霊媒対抗が一度も出てないので、変えずにアオイさん本指定のままにします",
        "狼でも狂でも縄に余裕があるうちに処理したいので、アオイさん本指定にします…",
    ),
    (5, "p1", "p15"): (  # アカリ, memo target アオイ
        "それでもアオイさん吊りは変えません",
        "アオイさん、反論があれば聞かせてください",
    ),
    (5, "p3", "p15"): (  # ユイ, memo target アオイ
        "流れは変えない、今日はアオイで押す！",
        "だからアオイ真で呪殺、アカリ偽という筋も一応は成立する",
    ),
    (5, "p10", "p15"): (  # リク, memo target アオイ; the designation comes from the same line
        "そのうえで、シオンがアオイを外したい理由を聞かせてくれ",
        "今日はアオイ吊りでいい",
    ),
    (5, "p14", "p15"): (  # シオン, memo target アオイ
        "縄にも余裕があるので、今日はアオイさん吊りに同意します",
        "そうなるとアオイさんは狂人寄りです",
    ),
    (5, "p15", "p5"): (  # アオイ, memo target ミオ; cut at its start
        "…で残り縄は4本ほどなので、今日はまだ白黒の出ていないミオさんやサクラさんから選んで、"
        "私は明日以降でも遅くないと思います",
    ),
    (6, "p0", "p8"): (  # マコト, memo target カイト; its designation is left out
        "カイトさん、最後に反論あればお願いします…",
        "シオンさん、カイトさん以外で気になる人はいますか…？",
    ),
    (6, "p3", "p8"): (  # ユイ, memo target カイト; the designation comes from the same line
        "ずっと票乗りばかりで、今日もカイトの理由がみんなの丸写しだから",
        "ただ流れはカイトで押す",
    ),
    (6, "p5", "p8"): (  # ミオ, memo target カイト
        "私もカイトさん吊りで動きます",
        "カイトさんの問いですが、最後の狼はカイトさん本人と見ています",
    ),
    (6, "p7", "p8"): (  # サクラ, memo target カイト
        "カイトの重ね白を『囲いと決めつけられない』と崩しつつ、投票はカイトで二番手は自分、"
        "と両取りしている",
        "票の乗り方は認めるが、今日はカイト吊りで動く",
    ),
    (6, "p8", "p5"): (  # カイト, memo target ミオ; a push the old detector missed
        "リク、名前を挙げるならミオ",
    ),
    (6, "p10", "p8"): (  # リク, memo target カイト
        "カイト、他に誰が最後の狼だと見ているのか、名前を一人挙げてほしい",
        "一方カイトは、ダイキの白に加えて狂目のアオイからも白をもらっている",
    ),
    (6, "p14", "p8"): (  # シオン, memo target カイト; the first sentence is cut at its end
        "カイトさんのミオさん挙げは、偽陣営の票を根拠にしている点は理解しますが、"
        "ツムギさんは霊白なので押し付けの説明としては弱…",
        "カイトさん、1日目にマコトさんと同じくレンさんへ投票した理由を聞かせてください",
    ),
}
# The pairs whose quote is every sentence naming the target, each whole.
SHOWN_WHOLE = {
    (2, "p11", "p5"),
    (3, "p1", "p12"),
    (3, "p5", "p11"),
    (4, "p11", "p8"),
    (6, "p5", "p8"),
    (6, "p8", "p5"),
}

# The review's seat-days where the seat put its memo target up in a sentence
# other than the one it closed with. With one sentence per line, none of these
# was quoted. The second sentence now comes from the same line when no earlier
# line names the target, which brings in リク's and ユイ's. マコト's stay out:
# an earlier line names the target (days 3 and 6), or two later sentences in
# the line do (day 4). On day 5 リク, and on day 6 マコト, wrote a true
# 『昼に言った通り』 in their vote reasons. The instruction allows it: the
# designation is in the day's log, and the line says the quote is an excerpt.
DESIGNATIONS: dict[tuple[int, str, str], str] = {
    (3, "p0", "p12"): "なので今日はダイキさん仮指定にします",
    (4, "p0", "p11"): "ツムギ黒で、今日はツムギさん吊りが第一候補だと思います",
    (5, "p10", "p15"): "今日はアオイ吊りでいい",
    (6, "p0", "p8"): (
        "今日の第一候補はカイトさんで、理由は偽目のダイキさんとアオイさんからの重ね白です"
    ),
    (6, "p3", "p8"): "ただ流れはカイトで押す",
}
QUOTED_FROM_THE_SAME_LINE = {(5, "p10", "p15"), (6, "p3", "p8")}


def _memo_targets() -> dict[tuple[int, str], str]:
    """Each seat's execution target as its last discussion turn of the day held it."""
    targets: dict[tuple[int, str], str] = {}
    for utterance in load_transcript()["utterances"]:
        memo = utterance.get("reasoning_memo") or {}
        if utterance["kind"] == "discussion" and memo.get("execution_target"):
            targets[(utterance["day"], utterance["player_id"])] = memo["execution_target"]
    return targets


def _first_game_body(day: int, seat: str, target: str) -> str:
    state = state_at_vote(day)
    candidates = [pid for pid in state.alive_ids() if pid != seat]
    _system, messages = _chat_builder(state).build_vote_context(
        state, seat, candidates, board_memo=MEMO, stated_target=target
    )
    return messages[0].content


def _first_game_line(day: int, seat: str, target: str) -> tuple[str, list[str]]:
    state = state_at_vote(day)
    body = _first_game_body(day, seat, target)
    line = next(row for row in body.splitlines() if row.startswith(HEADING))
    own = [
        message.content
        for message in state.chat_log
        if message.channel is ChatChannel.PUBLIC
        and message.author_id == seat
        and message.day == day
    ]
    return line, own


def test_the_pinned_pairs_are_every_seat_day_with_a_memo_target():
    pinned = {(day, seat): target for day, seat, target in FIRST_GAME}

    assert pinned == _memo_targets()


@pytest.mark.parametrize(("day", "seat", "target"), sorted(FIRST_GAME))
def test_the_first_game_line_is_pinned(day: int, seat: str, target: str) -> None:
    line, _own = _first_game_line(day, seat, target)
    name = load_transcript()["names"][target]
    sentences = FIRST_GAME[(day, seat, target)]

    if not sentences:
        assert line == flagged(name)
    elif (day, seat, target) in SHOWN_WHOLE:
        assert line == quoted(name, *sentences)
    else:
        assert line == excerpted(name, *sentences)


def test_the_first_game_flags_only_the_four_seats_that_never_said_the_name():
    never_said = sorted(key for key, sentences in FIRST_GAME.items() if not sentences)

    assert never_said == [(2, "p12", "p6"), (2, "p15", "p6"), (2, "p5", "p6"), (3, "p15", "p11")]


@pytest.mark.parametrize(("day", "seat", "target"), sorted(DESIGNATIONS))
def test_a_designation_the_quote_leaves_out_is_in_the_days_log_the_instruction_names(
    day: int, seat: str, target: str
) -> None:
    designation = DESIGNATIONS[(day, seat, target)]
    body = _first_game_body(day, seat, target)
    line, _own = _first_game_line(day, seat, target)
    names = load_transcript()["names"]
    log = body.split("【当日のログ】\n", 1)[1].split("\n\n", 1)[0]

    assert any(
        row.split("] ", 1)[1].startswith(f"{names[seat]}: ") and designation in row
        for row in log.splitlines()
    )
    assert body.endswith(SAID_IN_DAYS_LOG)
    if (day, seat, target) in QUOTED_FROM_THE_SAME_LINE:
        assert f"「{designation}」" in line
    else:
        assert designation not in line
        assert "自分の発言から抜粋:" in line


@pytest.mark.parametrize(("day", "seat", "target"), sorted(FIRST_GAME))
def test_the_first_game_line_states_facts_about_the_seats_own_lines(
    day: int, seat: str, target: str
) -> None:
    """In that game no seat's name is inside another's or inside a longer
    katakana word, so a plain substring test is an independent check of the
    name match, and a plain split on sentence ends of whether the quote is
    every sentence naming it."""
    line, own = _first_game_line(day, seat, target)
    name = load_transcript()["names"][target]

    if not any(name in content for content in own):
        assert line == flagged(name)
        return
    assert NOT_SAID not in line
    quotes = line.split(":", 1)[1].removesuffix(")").removeprefix("「").removesuffix("」")
    nested = [content.translate(str.maketrans("「」", "『』")) for content in own]
    for quote in quotes.split("」「"):
        assert name in quote
        assert len(quote) <= 60
        assert any(quote.strip("…") in content for content in nested)
    naming = {
        sentence.strip().removesuffix("。")
        for content in nested
        for sentence in re.split(r"(?<=[。！？!?])(?![。！？!?])|\n", content)
        if name in sentence
    }
    whole = set(quotes.split("」「")) == naming
    assert ("自分の発言から抜粋:" not in line) == whole
