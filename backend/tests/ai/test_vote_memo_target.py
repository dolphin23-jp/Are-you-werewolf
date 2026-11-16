"""The v3 vote prompt sets the seat's memo target beside what it said at the table.

In v3 the "stated target" the vote prompt shows is the seat's private plan
note. The first Claude game labelled it "昼にあなたが推した吊り先", and the model
believed the label: it wrote "昼の通りレンに入れます" and "昼はレンさんと言いました"
about names it had never said in public, and one such line was audited as an
unexplained change of vote. The label now says what the value is.

Code then states only what it can know: whether the seat's own public lines
today contain the name, and if they do, the seat's own sentence that names it.
It does not judge whether that sentence pushed the name. An earlier version
did, with word lists, and was wrong both ways on realistic lines: it missed
「灰はレン優先のまま押す」 and read a polite refusal as a push.
"""

from __future__ import annotations

import pytest

from app.ai.context import CHAT_VOTE_OUTPUT_INSTRUCTION, ContextBuilder, DaySummaryManager
from app.ai.deception import FakeClaimGuard, WolfDeceptionAssignment
from app.ai.personalities import assign_personalities
from app.engine.roles import RoleName
from app.engine.state import ChatChannel, ChatMessage, GameState
from tests.conftest import make_controller

MEMO = "【盤面メモ】"
HEADING = "【今日のメモの吊り先】"
NOT_SAID = "(今日の卓ではまだこの名前を出していない)"


def _chat_builder(state: GameState) -> ContextBuilder:
    wolf_ids = [p.player_id for p in state.players_by_role(RoleName.WEREWOLF)]
    return ContextBuilder(
        personalities=assign_personalities(list(state.players), seed=1),
        day_summaries=DaySummaryManager(),
        wolf_deception=WolfDeceptionAssignment(
            pattern_name="all_lurk",
            pattern_label="全潜伏",
            fake_role_by_player={},
            lurking_player_ids=wolf_ids,
        ),
        madman_fake_role=None,
        fake_claim_guard=FakeClaimGuard(wolf_team_ids=set(wolf_ids)),
        engine="v3",
    )


def _say(state: GameState, author: str, content: str, day: int) -> None:
    state.chat_log.append(
        ChatMessage(
            message_id=f"m{len(state.chat_log) + 1}",
            author_id=author,
            content=content,
            channel=ChatChannel.PUBLIC,
            day=day,
        )
    )


def _vote_body(state: GameState, stated_target: str | None, voter: str = "p1") -> str:
    candidates = [pid for pid in ("p1", "p2", "p11") if pid != voter]
    _system, messages = _chat_builder(state).build_vote_context(
        state, voter, candidates, board_memo=MEMO, stated_target=stated_target
    )
    return messages[0].content


def _memo_line(body: str) -> str:
    return next(line for line in body.splitlines() if line.startswith(HEADING))


def _day_two() -> GameState:
    state = make_controller(seed=4).state
    state.day = 2
    return state


def flagged(label: str) -> str:
    return f"{HEADING}{label}{NOT_SAID}"


def quoted(label: str, *sentences: str) -> str:
    quotes = "".join(f"「{sentence}」" for sentence in sentences)
    return f"{HEADING}{label}(今日の卓でこの名前に触れた自分の発言:{quotes})"


def _line_after(*said: str, target: str = "p2", voter: str = "p1") -> str:
    state = _day_two()
    for content in said:
        _say(state, voter, content, day=2)
    return _memo_line(_vote_body(state, target, voter=voter))


# -- the name absent: the flag --


def test_the_memo_target_is_labelled_as_the_memo_and_flagged_when_never_said():
    state = _day_two()
    _say(state, "p1", "占いの内訳が気になります。", day=2)

    body = _vote_body(state, "p2")

    assert _memo_line(body) == (
        "【今日のメモの吊り先】Player2(今日の卓ではまだこの名前を出していない)"
    )
    assert "昼にあなたが推した" not in body
    assert "名指ししていない" not in body


def test_only_the_seats_own_lines_from_today_count():
    state = _day_two()
    _say(state, "p3", "Player1が怪しい。", day=1)  # yesterday
    _say(state, "p4", "Player1を吊ろう。", day=2)  # someone else
    _say(state, "p3", "Player11の票が気になる。", day=2)  # Player11 is not Player1

    line = _memo_line(_vote_body(state, "p1", voter="p3"))

    assert line == flagged("Player1")


def test_a_seat_that_has_not_spoken_today_is_flagged():
    assert _line_after(target="p2") == flagged("Player2")


def test_no_memo_target_is_said_plainly():
    assert _memo_line(_vote_body(_day_two(), None)) == "【今日のメモの吊り先】なし"


# -- the name present: the seat's own sentence --


def test_a_memo_target_the_seat_named_today_is_quoted_from_its_own_line():
    line = _line_after("今日はPlayer2さんを吊りたい。")

    assert line == (
        "【今日のメモの吊り先】Player2"
        "(今日の卓でこの名前に触れた自分の発言:「今日はPlayer2さんを吊りたい」)"
    )


@pytest.mark.parametrize(
    ("said", "sentence"),
    [
        # The review's flag_repro2.py: once read as a push by a bare-mention
        # check. Now the seat reads its own words.
        (
            "Player2さん、私の投票理由は発言の少なさです。",
            "Player2さん、私の投票理由は発言の少なさです",
        ),
        ("Player2の説明で納得した。Player2は吊らない。", "Player2は吊らない"),
        # flag_repro.py: talked about, cleared, asked about.
        ("Player2さんは白っぽいし信用してる。", "Player2さんは白っぽいし信用してる"),
        ("Player3さんに質問。Player2をどう見てる？", "Player2をどう見てる？"),
        # The previous detector's own cases, each one a sentence naming Player2.
        ("Player2さんは吊るべきではない。", "Player2さんは吊るべきではない"),
        ("Player2さんへの投票はしない。", "Player2さんへの投票はしない"),
        ("Player2はまだ吊るのは早い。", "Player2はまだ吊るのは早い"),
        ("Player2さんの投票先が気になる。", "Player2さんの投票先が気になる"),
        ("Player2さんが投票したのはPlayer3。", "Player2さんが投票したのはPlayer3"),
        ("昨日Player2に投票した。", "昨日Player2に投票した"),
        (
            "僕もPlayer2に入れてたので、そこは材料にならない。",
            "僕もPlayer2に入れてたので、そこは材料にならない",
        ),
        ("Player2は白だからグレー吊り。", "Player2は白だからグレー吊り"),
        ("今日はPlayer2吊り。", "今日はPlayer2吊り"),
        ("Player2に投票します。", "Player2に投票します"),
        ("Player2怪しい、吊りたい。", "Player2怪しい、吊りたい"),
        ("今日の吊り先はPlayer2。", "今日の吊り先はPlayer2"),
        ("吊りたいのはPlayer2さん。", "吊りたいのはPlayer2さん"),
        ("Player2さんが第一候補。", "Player2さんが第一候補"),
        ("Player2を吊るしかない。", "Player2を吊るしかない"),
        ("俺はPlayer2吊りで変えない。", "俺はPlayer2吊りで変えない"),
        ("Player3かPlayer2に投票。", "Player3かPlayer2に投票"),
        ("現時点の第一処刑候補はPlayer2(p2)です。", "現時点の第一処刑候補はPlayer2(p2)です"),
    ],
)
def test_a_line_that_names_the_target_is_quoted_whatever_it_says(said: str, sentence: str) -> None:
    assert _line_after(said) == quoted("Player2", sentence)


# -- matching the name --


def test_a_name_followed_by_a_digit_is_still_named():
    """「ソウタ1択」: the trailing-digit guard is for labels that end in a digit."""
    state = _day_two()
    state.players["p2"].name = "ソウタ"
    _say(state, "p1", "今日はソウタ1択。", day=2)

    assert _memo_line(_vote_body(state, "p2")) == quoted("ソウタ", "今日はソウタ1択")


def test_a_name_inside_a_longer_seats_name_is_that_seat():
    """A seat called サク is not named by a line about サクラ."""
    state = _day_two()
    state.players["p2"].name = "サク"
    state.players["p3"].name = "サクラ"
    _say(state, "p1", "サクラさん吊りで。", day=2)

    assert _memo_line(_vote_body(state, "p2")) == flagged("サク")
    assert _memo_line(_vote_body(state, "p3")) == quoted("サクラ", "サクラさん吊りで")


def test_a_digit_label_is_not_read_inside_a_longer_one():
    assert _line_after("Player11吊り。Player21も気になる。") == flagged("Player2")
    assert _line_after("Player11吊り。", target="p11") == quoted("Player11", "Player11吊り")


@pytest.mark.parametrize("honorific", ["さん", "くん", "君", "ちゃん", "氏", "様"])
def test_a_name_with_an_honorific_is_named(honorific: str) -> None:
    said = f"ソウタ{honorific}に一票。"
    state = _day_two()
    state.players["p2"].name = "ソウタ"
    _say(state, "p1", said, day=2)

    assert _memo_line(_vote_body(state, "p2")) == quoted("ソウタ", f"ソウタ{honorific}に一票")


# -- which sentences --


def test_the_sentence_comes_from_the_most_recent_line_that_names_the_target():
    line = _line_after("Player2が怪しい。", "今日はPlayer2吊りで。", "おはよう。")

    assert line == quoted("Player2", "今日はPlayer2吊りで", "Player2が怪しい")


def test_one_sentence_per_line_the_last_one_that_names_the_target():
    line = _line_after("Player2が怪しい。だからPlayer2に投票する。Player3は白。")

    assert line == quoted("Player2", "だからPlayer2に投票する")


def test_at_most_two_sentences_are_quoted():
    line = _line_after("Player2は白っぽい。", "Player2が怪しい。", "Player2吊りで。")

    assert line == quoted("Player2", "Player2吊りで", "Player2が怪しい")


def test_a_sentence_said_again_is_quoted_once_and_an_earlier_different_one_follows():
    line = _line_after("Player2が怪しい。", "Player2吊り。", "Player2吊り。")

    assert line == quoted("Player2", "Player2吊り", "Player2が怪しい")


def test_an_earlier_line_gives_its_sentence_that_differs_from_the_one_quoted():
    line = _line_after("Player2が怪しい。Player2吊り。", "Player2吊り。")

    assert line == quoted("Player2", "Player2吊り", "Player2が怪しい")


def test_a_quote_in_the_seats_line_stays_whole_and_is_nested_in_double_brackets():
    """Someone else's words inside the seat's sentence: the sentence does not
    break at their 。, and their 「」 becomes 『』 inside the seat's quote."""
    line = _line_after("Player3さんの「Player2吊り。」は早いと思う。")

    assert line == quoted("Player2", "Player3さんの『Player2吊り。』は早いと思う")


def test_a_run_of_sentence_ends_stays_with_its_sentence():
    line = _line_after("Player2って本当に狼！？違う気がする。")

    assert line == quoted("Player2", "Player2って本当に狼！？")


# -- length --

LONG_HEAD = "昨日からずっと票の動きとCOの順番を見比べていて、"  # 25 characters
LONG_TAIL = "に投票するつもりで、ほかの灰はまだそこまで材料が揃っていないので明日以降に回したい"


def test_a_long_sentence_is_cut_around_the_name_with_an_ellipsis_where_cut():
    sentence = f"{LONG_HEAD}今日はPlayer2さん{LONG_TAIL}"
    assert len(sentence) > 60

    line = _line_after(f"{sentence}。")

    quote = line.split("「", 1)[1].rsplit("」", 1)[0]
    assert quote.startswith("…") and quote.endswith("…")
    assert len(quote) <= 60
    assert "Player2さん" in quote
    assert quote.strip("…") in sentence
    # Twenty characters before the name, the rest of the room after it.
    assert quote == (
        "…の動きとCOの順番を見比べていて、今日はPlayer2さん"
        "に投票するつもりで、ほかの灰はまだそこまで材料が揃っていな…"
    )


def test_a_long_sentence_that_starts_with_the_name_is_cut_only_at_its_end():
    sentence = f"Player2さん{LONG_TAIL}、{LONG_HEAD}"

    quote = _line_after(sentence).split("「", 1)[1].rsplit("」", 1)[0]

    assert not quote.startswith("…") and quote.endswith("…")
    assert len(quote) == 60
    assert sentence.startswith(quote[:-1])


def test_a_long_sentence_that_ends_with_the_name_is_cut_only_at_its_start():
    sentence = f"{LONG_HEAD}{LONG_HEAD}今日の吊り先はPlayer2さん"

    quote = _line_after(sentence).split("「", 1)[1].rsplit("」", 1)[0]

    assert quote.startswith("…") and not quote.endswith("…")
    assert len(quote) == 60
    assert sentence.endswith(quote[1:])


def test_the_memo_line_is_bounded_however_much_the_seat_said():
    """Two quotes of at most 60 characters: the vote prompt grows by a bounded
    amount, not by the seat's day of talk."""
    long = f"{LONG_HEAD}今日はPlayer2さん{LONG_TAIL}。" * 3
    line = _line_after(long, f"Player2が{LONG_TAIL}。", long)

    assert len(line) <= len(quoted("Player2", "x" * 60, "x" * 60))
    assert line.count("「") == 2


# -- the instruction --


def test_the_vote_instruction_points_at_the_quote_not_the_memo():
    body = _vote_body(_day_two(), "p2")

    assert "卓で名指しした吊り先と違う相手に入れるなら" in body
    assert "括弧内の自分の発言に無いことを『昼に言った』と書かない" in body
    assert "昼に言った吊り先と違う相手" not in body


def test_the_vote_instruction_stays_plain_and_short():
    sentence = "括弧内の自分の発言に無いことを『昼に言った』と書かない。"
    assert CHAT_VOTE_OUTPUT_INSTRUCTION.endswith(sentence)
    # The sentence it replaced was 24 characters; this one is 28.
    assert len(CHAT_VOTE_OUTPUT_INSTRUCTION) <= 327
    for word in ("思考", "推論", "考え", "thought", "reasoning"):
        assert word not in CHAT_VOTE_OUTPUT_INSTRUCTION
