"""The ballot audit reads seat names exactly as the vote prompt does.

Whether a seat named its memo target at the table decides whether a change
of vote can fail the release gate, and the vote prompt quotes the same lines
to the seat. The audit runs on transcripts, without a live game state, so it
keeps its own copy of `app.ai.context._mention_spans`; these hold the copy to
the original's answers on the cases the prompt's tests pin.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ai.context import _mention_spans
from app.eval.reasoning_analyzer import mention_spans, named_seat_ids

NAMES = {
    "p1": "Player1",
    "p11": "Player11",
    "p2": "ソウタ",
    "p3": "サクラ",
    "p4": "サク",
    "p5": "レン",
    "p6": "リク",
    "p7": "Ken",
    "p8": "Al",
}

LINES = [
    "今日はソウタ1択。",
    "サクラさん吊りで。サクも気になる。",
    "Player11吊り。Player1も気になる。p1とp11の票を見たい。",
    "ソウタさんに一票。ソウタくん、ソウタ君、ソウタちゃん、ソウタ氏、ソウタ様。",
    "占いチャレンジしてもいいかも。オレンジ色のカード。カレンダー。トレンド。",
    "リクエストがあればどうぞ。今日はサクッと決めよう。ローソウタって何？",
    "Kenjiって誰？ Also, I think so.",
    "ソウター、どう思う。ねーソウタ、どう思う。灰はハルト・ソウタ・レン。",
    "ソウタCOは偽だと思う。灰ソウタ吊りで。Kenさんに一票。Ken1択。灰はKen、ほかは白。",
    "レンさんに入れるよ。リク、レンの票の理由は？",
]


def _state() -> SimpleNamespace:
    return SimpleNamespace(
        players={pid: SimpleNamespace(name=name) for pid, name in NAMES.items()}
    )


@pytest.mark.parametrize("text", LINES)
def test_the_audit_reads_the_same_seats_as_the_vote_prompt(text: str) -> None:
    assert mention_spans(text, NAMES) == _mention_spans(_state(), text)  # type: ignore[arg-type]


def test_each_seat_is_listed_once_in_the_order_first_named() -> None:
    assert named_seat_ids("レンさんに入れるよ。リク、レンの票の理由は？", NAMES) == ("p5", "p6")
    assert named_seat_ids("サクラさん吊りで。", NAMES) == ("p3",)
    assert named_seat_ids("Player11吊り。", NAMES) == ("p11",)
