"""The v3 vote prompt compares the ballot with what the seat said at the table.

In v3 the "stated target" the vote prompt shows is the seat's private plan
note. The first Claude game labelled it "昼にあなたが推した吊り先", and the model
believed the label: it wrote "昼の通りレンに入れます" and "昼はレンさんと言いました"
about names it had never said in public, and one such line was audited as an
unexplained change of vote. The label now says what the value is, and code
checks whether the seat actually named it today.
"""

from __future__ import annotations

from app.ai.context import ContextBuilder, DaySummaryManager
from app.ai.deception import FakeClaimGuard, WolfDeceptionAssignment
from app.ai.personalities import assign_personalities
from app.engine.roles import RoleName
from app.engine.state import ChatChannel, ChatMessage, GameState
from tests.conftest import make_controller

MEMO = "【盤面メモ】"


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
    return next(line for line in body.splitlines() if line.startswith("【今日のメモの吊り先】"))


def _day_two() -> GameState:
    state = make_controller(seed=4).state
    state.day = 2
    return state


def test_the_memo_target_is_labelled_as_the_memo_and_flagged_when_never_said():
    state = _day_two()
    _say(state, "p1", "占いの内訳が気になります。", day=2)

    body = _vote_body(state, "p2")

    assert _memo_line(body) == "【今日のメモの吊り先】Player2(卓ではまだ名指ししていない)"
    assert "昼にあなたが推した" not in body


def test_a_memo_target_the_seat_named_today_carries_no_flag():
    state = _day_two()
    _say(state, "p1", "今日はPlayer2さんを吊りたい。", day=2)

    assert _memo_line(_vote_body(state, "p2")) == "【今日のメモの吊り先】Player2"


def test_only_the_seats_own_lines_from_today_count_as_naming_the_target():
    state = _day_two()
    _say(state, "p3", "Player1が怪しい。", day=1)  # yesterday
    _say(state, "p4", "Player1を吊ろう。", day=2)  # someone else
    _say(state, "p3", "Player11の票が気になる。", day=2)  # Player11 is not Player1

    line = _memo_line(_vote_body(state, "p1", voter="p3"))

    assert line == "【今日のメモの吊り先】Player1(卓ではまだ名指ししていない)"


def test_no_memo_target_is_said_plainly():
    assert _memo_line(_vote_body(_day_two(), None)) == "【今日のメモの吊り先】なし"


def test_the_vote_instruction_asks_for_a_reason_against_the_table_not_the_memo():
    body = _vote_body(_day_two(), "p2")

    assert "卓で名指しした吊り先と違う相手に入れるなら" in body
    assert "卓で言っていないことを『昼に言った』と書かない" in body
    assert "昼に言った吊り先と違う相手" not in body
