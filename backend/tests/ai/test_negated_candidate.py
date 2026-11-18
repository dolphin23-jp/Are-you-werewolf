"""「Xは吊るべきではない」 does not put X forward for execution.

`rendering._DECLARES_CANDIDATE` found 吊るべき inside 吊るべきではない, so
`declared_candidates` read a seat arguing *against* hanging X as backing X, and
`enforce_execution_target` stripped that sentence as a competing declaration.
Where stated targets are public (v2), the day summary every seat reads could
then report the seat as backing the player it had just defended.

A declaring word followed by a plain negation (ではない / でない / じゃない /
ではありません / とは思わない / つもりはない ...) is no declaration. The
rhetorical 「〜ではないか」 / 「〜ではないでしょうか」, which in Japanese
proposes rather than denies, still declares.
"""

from __future__ import annotations

import pytest

from app.ai.reasoning.facts import PublicFactLedger
from app.ai.reasoning.rendering import declared_candidates, enforce_execution_target
from app.ai.reasoning.runtime import ReasoningRuntime
from app.engine.roles import RoleName
from app.engine.state import ChatChannel, ChatMessage
from tests.ai.reasoning.solver import boards
from tests.conftest import make_controller


def _ledger() -> PublicFactLedger:
    return PublicFactLedger(make_controller(seed=4).state)


@pytest.mark.parametrize(
    "sentence",
    [
        "Player5は吊るべきではない。",
        "Player5を吊るべきでない。",
        "Player5は吊るべきじゃない。",
        "Player5を吊るべきではありません。",
        "Player5は吊るべきでもない。",
        "Player5を吊るべきではなく、まず話を聞きたい。",
        "Player5は処刑すべきではないと思う。",
        "Player5を吊りたいとは思わない。",
        "Player5を吊るべきとは言えない。",
        "Player5は第一候補ではない。",
        "Player5は本命ではないです。",
        "Player5は最有力ではない。",
        "Player5は投票先ではない。",
        "Player5は処刑先でもない。",
        "Player5に投票するつもりはない。",
        "Player5は第一候補ではないかもしれない。",
    ],
)
def test_a_negated_declaration_puts_nobody_forward(sentence: str):
    assert declared_candidates(sentence, _ledger(), speaker_id="p2") == set()


@pytest.mark.parametrize(
    "sentence",
    [
        "Player5を吊るべき。",
        "Player5を吊るべきです。",
        "Player5を吊るべきだと思う。",
        "Player5を吊るべきでしょう。",
        "Player5を吊りたい。",
        "第一処刑候補はPlayer5。",
        # Rhetorical: "shouldn't we hang Player5?" proposes the hang.
        "Player5を吊るべきではないか。",
        "Player5が本命ではないでしょうか。",
        "Player5が第一候補ではありませんか？",
        "Player5を吊るべきじゃない？",
        "Player5に投票します。",
    ],
)
def test_an_affirmed_declaration_still_puts_the_seat_forward(sentence: str):
    assert declared_candidates(sentence, _ledger(), speaker_id="p2") == {"p5"}


def test_a_negated_and_an_affirmed_sentence_are_told_apart():
    message = "Player5は吊るべきではない。Player7を吊りたい。"

    assert declared_candidates(message, _ledger(), speaker_id="p2") == {"p7"}


def test_enforcement_keeps_a_sentence_that_argues_against_hanging_someone_else():
    ledger = _ledger()

    enforced = enforce_execution_target(
        "Player5は吊るべきではない。Player9を吊りたい。", "p7", ledger, speaker_id="p2"
    )

    # The competing declaration goes; the defence of Player5 is not one and stays.
    assert "Player5は吊るべきではない。" in enforced
    assert "Player9" not in enforced
    assert enforced.startswith("現時点の第一処刑候補は")


def test_v2_does_not_report_a_seat_backing_the_player_it_defended():
    """v2 states targets in public, so the day summary reads declarations.

    p2's plan names p5, but in public p2 said p5 should not be hanged; p3
    really put p7 forward. There is no public split for the summary to report.
    """
    state = boards.deal(
        {"p1": RoleName.WEREWOLF, "p4": RoleName.SEER, "p6": RoleName.FOX}, day=2
    )
    runtime = ReasoningRuntime(state, [f"p{i}" for i in range(1, 17)], seed=3)
    runtime.refresh(state)
    for speaker, text, target in (
        ("p2", "Player5は吊るべきではない。", "p5"),
        ("p3", "Player7を吊りたい。", "p7"),
    ):
        state.chat_log.append(
            ChatMessage(
                message_id=f"m{len(state.chat_log)}",
                author_id=speaker,
                content=text,
                channel=ChatChannel.PUBLIC,
                day=2,
            )
        )
        runtime.record_stated_target(speaker, target, 2)

    points = runtime.conflict_points(state, stated_targets_are_public=True)

    assert not any("割れている" in point for point in points)
    assert not any("p5←p2" in point for point in points)
