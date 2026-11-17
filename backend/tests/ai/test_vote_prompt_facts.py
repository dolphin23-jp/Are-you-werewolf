"""What a v3 ballot is cast beside: facts the vote can be checked against, and
no advice on how to use them (seed-1 review 2-B6, 2-A2, 3-B1, 4-B6, 2-B7).

Of the seed-1 game's 83 ballots, 37 came from seats without a turn that day;
none of their prompts said who had not had one, that the ballot would be seen
by everyone, or what the day's summary and designation had been -- only the
day's log, in which the designation is one line among dozens. The candidates
were bare names, so a confirmed freemason read like anyone else.
"""

from __future__ import annotations

import asyncio

from app.ai.schemas import VoteOutput
from app.engine.phases import Phase
from tests.ai.seed1_replay import captured_prompts
from tests.ai.test_v3_coordinator import _setup as _v3_setup


def _line(user: str, heading: str) -> str:
    return next(line for line in user.splitlines() if line.startswith(heading))


def test_the_ballot_is_said_to_be_public():
    user = captured_prompts()["d4-vote-kaito"].user

    assert "- 投票先は全員に公開され、記録に残る" in user


def test_the_seats_without_a_turn_today_are_listed_and_humans_are_left_out():
    user = captured_prompts()["d4-vote-kaito"].user

    assert (
        "- AI席の手番は進行が割り振る。今日まだ手番が回っていない席: "
        "ミオ、レン、サクラ、ツムギ、シオン（人間の席は含めない）"
    ) in user


def test_the_days_direction_is_quoted():
    user = captured_prompts()["d4-vote-kaito"].user

    direction = user.split("【進行役の方針】\n", 1)[1].split("\n\n", 1)[0].splitlines()
    assert direction == [
        "- 進行役のまとめ（マコト m69）:「今日はハルトさんに投票します。ダイキさん票でケントさんに"
        "入れた理由が、まだ納得できないからです。最後に言いたいことはありますか…？」",
        "- 本指定（マコト m71）:「ハルトさんの最後の話も、ケントさんに入れた理由の説明には"
        "なっていないので、今日はハルトさんに投票します。」",
    ]


def test_day_one_quotes_its_summary_and_designation_too():
    user = captured_prompts()["d1-vote-tsumugi"].user

    assert "【進行役の方針】\n- 進行役のまとめ（マコト m23）" in user
    assert "\n- 本指定（マコト m25）:「今日はグレランで、ユイさんに仮指定します。" in user


def test_a_day_without_those_lines_has_no_direction_section():
    controller, state, _runtime, coordinator, provider, speaker, _coded, _rogue = _v3_setup()
    state.phase = Phase.VOTING

    asyncio.run(coordinator._cast_vote(controller, state, speaker))

    _schema, _system, user = next(p for p in provider.prompts if p[0] is VoteOutput)
    assert "【進行役の方針】" not in user
    assert "- 投票先は全員に公開され、記録に残る" in user


def test_a_human_seat_is_never_listed_as_waiting_for_a_turn():
    controller, state, _runtime, coordinator, provider, speaker, _coded, _rogue = _v3_setup()
    state.players["p0"].is_human = True
    state.phase = Phase.VOTING

    asyncio.run(coordinator._cast_vote(controller, state, speaker))

    _schema, _system, user = next(p for p in provider.prompts if p[0] is VoteOutput)
    waiting = _line(user, "- AI席の手番は進行が割り振る。")
    assert state.players["p0"].name not in waiting
    assert state.players["p2"].name in waiting
    assert waiting.endswith("（人間の席は含めない）")


def test_a_voter_without_a_turn_finds_itself_as_jibun():
    controller, state, _runtime, coordinator, provider, speaker, _coded, _rogue = _v3_setup()
    state.phase = Phase.VOTING

    asyncio.run(coordinator._cast_vote(controller, state, speaker))

    _schema, _system, user = next(p for p in provider.prompts if p[0] is VoteOutput)
    listed = _line(user, "- AI席の手番は進行が割り振る。").split(": ", 1)[1]
    seats = listed.split("（", 1)[0].split("、")
    assert "自分" in seats
    assert state.players[speaker].name not in seats


def test_each_candidate_carries_its_public_facts():
    candidates = _line(captured_prompts()["d4-vote-kaito"].user, "【投票候補】")

    for annotated in (
        "マコト（共有・相互確認済み）",
        "ホノカ（共有・相互確認済み）",
        "ケント（占いCO）",
        "リク（ケントの白）",
        "ハルト（ケントの黒）",
        "ソウタ（アオイの白）",
        "レン（グレー）",
    ):
        assert annotated in candidates
    # Only the living are candidates, and never the voter.
    assert "カイト" not in candidates
    assert "アカリ" not in candidates


def test_a_runoff_says_how_the_first_round_tied():
    header = _line(captured_prompts()["d2-runoff-makoto"].user, "【決選投票")

    assert header.startswith(
        "【決選投票(2回目)】前回の投票でアオイとレンが各3票の同数だったため、"
        "候補は同数だったプレイヤーに限られる。次の中から選ぶ: "
    )
    assert header.endswith("アオイ（占いCO・ケントの白）、レン（グレー）")


def test_no_advice_comes_with_the_facts():
    user = captured_prompts()["d4-vote-kaito"].user
    facts = user.split("【投票について】\n", 1)[1].split("\n\n", 1)[0]

    assert facts.splitlines() == [
        "- 投票先は全員に公開され、記録に残る",
        "- AI席の手番は進行が割り振る。今日まだ手番が回っていない席: "
        "ミオ、レン、サクラ、ツムギ、シオン（人間の席は含めない）",
    ]
