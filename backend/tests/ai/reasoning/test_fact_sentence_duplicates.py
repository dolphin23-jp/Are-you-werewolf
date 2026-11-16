"""The canonical fact sentence is for what the prose left out, not a second copy.

In the first Claude game eleven lines opened with a code-written "XXCO。" for a
CO the speaker had made days earlier, and three stated a result twice because
the prose said it in a way the matcher did not read -- for example
「霊媒CO。霊媒結果、ダイキは黒(人狼)です。おはようございます。霊媒結果です。
昨日のダイキさんは「黒」でした。」. The table hears that as a machine talking.
"""

from __future__ import annotations

import pytest

from app.ai.reasoning import PublicFactLedger
from app.ai.reasoning.claims import build_claim_drafts, ensure_fact_sentences
from app.ai.schemas import DiscussionOutput, PublicResultClaim
from app.engine.roles import RoleName
from app.engine.state import GameState
from tests.ai.reasoning.fixtures import declare_co, execute, make_state

SPEAKER = "p3"
TARGET = "p5"


def _fixed(
    state: GameState,
    message: str,
    *,
    role: str,
    is_werewolf: bool,
    target: str = TARGET,
) -> str:
    ledger = PublicFactLedger(state)
    output = DiscussionOutput(
        public_message=message,
        public_claim_role=role,
        public_results=[
            PublicResultClaim(result_type=role, target_id=target, is_werewolf=is_werewolf)
        ],
    )
    drafts = build_claim_drafts(output, ledger, speaker_id=SPEAKER, names_only=True)
    return ensure_fact_sentences(message, drafts, ledger, speaker_id=SPEAKER)


def _medium_board() -> GameState:
    state = make_state(day=4)
    declare_co(state, SPEAKER, RoleName.MEDIUM, day=1)
    execute(state, TARGET, day=3)
    return state


def _seer_board() -> GameState:
    state = make_state(day=4)
    declare_co(state, SPEAKER, RoleName.SEER, day=1)
    return state


# -- a CO the speaker already holds is not announced again --


def test_a_standing_co_is_not_announced_again():
    message = "おはようございます。今日も結果から出します。"

    fixed = _fixed(_medium_board(), message, role="medium", is_werewolf=True)

    assert fixed == "霊媒結果、Player5は黒(人狼)です。" + message


def test_a_first_co_is_still_written_in():
    state = make_state(day=2)

    fixed = _fixed(state, "結果を出します。", role="seer", is_werewolf=False)

    assert fixed == "占いCO。占い結果、Player5は白(人狼ではない)です。結果を出します。"


def test_a_co_for_a_different_role_than_the_standing_one_is_still_written_in():
    state = make_state(day=2)
    declare_co(state, SPEAKER, RoleName.HUNTER, day=1)
    ledger = PublicFactLedger(state)
    output = DiscussionOutput(public_message="話を変えます。", public_claim_role="seer")
    drafts = build_claim_drafts(output, ledger, speaker_id=SPEAKER, names_only=True)

    fixed = ensure_fact_sentences(output.public_message, drafts, ledger, speaker_id=SPEAKER)

    assert fixed.startswith("占いCO。")


# -- the ordinary ways a result is said count as saying it --


@pytest.mark.parametrize(
    ("board", "role", "message"),
    [
        # The first game, nearly verbatim.
        ("medium", "medium", "おはようございます。霊媒結果です。昨日のPlayer5さんは「黒」でした。"),
        ("seer", "seer", "お待たせしました、昨夜はPlayer5さんを占って、結果は黒です。"),
        ("seer", "seer", "昨夜の占いはPlayer5さんで、黒でした。"),
        # Other everyday forms.
        ("seer", "seer", "Player5は黒でした。"),
        ("seer", "seer", "Player5を占って結果は黒。"),
        ("seer", "seer", "Player5を占った結果、人狼でした。"),
        ("seer", "seer", "Player5さんを占いました。結果は黒です。"),
        ("seer", "seer", "Player5さんの結果は『黒』です。"),
        ("seer", "seer", "占い結果、Player5は●。"),
        ("medium", "medium", "霊媒結果、Player5は黒。"),
        ("medium", "medium", "昨日吊られたPlayer5さん、霊媒の結果は人狼でした。"),
    ],
)
def test_a_result_said_in_an_everyday_way_is_not_said_twice(
    board: str, role: str, message: str
) -> None:
    state = _medium_board() if board == "medium" else _seer_board()

    assert _fixed(state, message, role=role, is_werewolf=True) == message


@pytest.mark.parametrize(
    "message",
    [
        "Player5さんを占って、結果は白です。",
        "Player5は人狼ではありませんでした。",
        "昨夜の占いはPlayer5さんで、白でした。",
    ],
)
def test_a_white_result_said_in_an_everyday_way_is_not_said_twice(message: str) -> None:
    assert _fixed(_seer_board(), message, role="seer", is_werewolf=False) == message


@pytest.mark.parametrize(
    "message",
    [
        # The opposite verdict: the canonical one has to be heard.
        "Player5は白でした。",
        # Impressions and hypotheticals are not a published result.
        "Player5は黒っぽい。",
        "Player5が黒なら吊る。",
        "Player5は黒寄りに見える。",
        "Player5は黒だと思う。",
        "Player5の黒要素は票だけ。",
        # Somebody else's verdict, quoted.
        "Player7のPlayer5黒は信用しない。",
        # A verdict about someone else that follows the target's name.
        "Player5を占いました。Player6は黒です。",
    ],
)
def test_impressions_and_other_verdicts_do_not_count_as_stating_it(message: str) -> None:
    fixed = _fixed(_seer_board(), message, role="seer", is_werewolf=True)

    assert fixed == "占い結果、Player5は黒(人狼)です。" + message


def test_a_neighbouring_name_does_not_state_the_targets_result():
    message = "Player11は黒でした。"

    fixed = _fixed(_seer_board(), message, role="seer", is_werewolf=True, target="p1")

    assert fixed == "占い結果、Player1は黒(人狼)です。Player11は黒でした。"
