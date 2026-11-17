"""v3's past days, one block each: what happened, not the board again (review
1-D3, cause R3).

The v3 day summary used to repeat the board memo -- the living, the dead, every
CO and verdict -- and its living list was frozen at the day's vote: on day 4 the
prompt still listed アカリ, killed on night 3. The verdict list under it dated a
night-0 result "1日目" and wrote seer and medium results alike, under the
【公開事実】 heading. What the memo cannot hold was missing: the day's summary
and designation, and what the executed seat said last (5-B3, 2-B8).

Now each past day is the execution with its vote spread, the night before, the
facilitator's lines and the executed seat's last line, quoted as said; the
day's results and COs follow under their own heading, dated by the night they
are about. v2 and legacy keep the summary they were measured with.
"""

from __future__ import annotations

import asyncio

from app.ai.reasoning.summaries import FACTS_HEADING
from app.engine.phases import Phase
from app.engine.state import DeathCause, DeathRecord, VoteRecord
from tests.ai.seed1_replay import captured_prompts
from tests.ai.test_v3_coordinator import _setup as _v3_setup


def _past_days(user: str) -> str:
    return user.split("【過去日の要約】\n", 1)[1].split("\n\n", 1)[0]


def _block(past: str, day: int) -> list[str]:
    """Day `day`'s facts and its results heading, as lines."""
    start = past.index(f"{FACTS_HEADING}{day}日目")
    end = past.find(f"{FACTS_HEADING}{day + 1}日目")
    return past[start : end if end >= 0 else None].splitlines()


def test_day_five_reads_each_past_day_as_it_happened():
    past = _past_days(captured_prompts()["d5-m81-kento-rebuttal"].user)

    assert _block(past, 1) == [
        f"{FACTS_HEADING}1日目",
        "- 0日目夜の結果（1日目公開）: 初日犠牲者ノゾミ",
        "- 進行役のまとめ（マコト m23）:「今日はグレランで、占い3人は残したいです。"
        "ソウタさん、アオイさん以外で気になるグレーはいますか…？」",
        "- 本指定（マコト m25）:「今日はグレランで、ユイさんに仮指定します。"
        "反論があれば聞かせてください。」",
        "- 投票: ユイ15票、ソウタ1票 → 処刑 ユイ",
        "- 処刑されたユイの最後の発言（m15）:「みんなホノカ待ちばかりだけど、"
        "ホノカが出てこないまま流れるなら要警戒！反応が遅いのは放置できないよ！」",
        "【1日目に出た結果・CO】CO: 占い=ケント、ダイキ、アオイ / 共有=マコト、ホノカ。"
        "占い 0日目夜の結果（1日目公開）: ケント→リク=白、ダイキ→シオン=白、アオイ→ソウタ=白",
    ]


def test_a_runoff_day_shows_both_rounds_and_keeps_seer_and_medium_apart():
    past = _past_days(captured_prompts()["d5-m81-kento-rebuttal"].user)

    day2 = _block(past, 2)
    assert "- 1日目夜の結果（2日目公開）: 死体なし" in day2
    assert (
        "- 投票: レン3票、アオイ3票、ホノカ2票、シオン2票、ケント2票、ハルト1票、ツムギ1票、"
        "ダイキ1票（同数）→ 決選: アオイ10票、レン5票 → 処刑 アオイ"
    ) in day2
    assert (
        "- 処刑されたアオイの最後の発言（m34）:「昨夜はカイトさんを占って白でした。"
        "ケントさん、私への白は了解です。ただ占い先の理由は聞きたいです。」"
    ) in day2
    results = day2[-1]
    assert results.startswith("【2日目に出た結果・CO】CO: 霊媒=アカリ。")
    seer = "占い 1日目夜の結果（2日目公開）: ケント→アオイ=白、ダイキ→ミオ=白、アオイ→カイト=白"
    assert f"。{seer}。" in results
    assert results.endswith("。霊媒 1日目処刑の結果（2日目公開）: アカリ→ユイ=白")


def test_the_night_a_seat_died_is_dated_and_no_living_list_goes_stale():
    past = _past_days(captured_prompts()["d5-m81-kento-rebuttal"].user)

    day4 = _block(past, 4)
    assert "- 3日目夜の結果（4日目公開）: アカリが死亡" in day4
    assert day4[2].startswith("- 進行役のまとめ（マコト m69）:「今日はハルトさんに")
    assert day4[3].startswith("- 本指定（マコト m71）:「ハルトさんの最後の話も")
    assert "- 投票: ハルト10票、ホノカ1票、ケント1票 → 処刑 ハルト" in day4
    assert day4[-1] == "【4日目に出た結果・CO】占い 3日目夜の結果（4日目公開）: ケント→ハルト=黒"
    # What the board memo already holds is not repeated, and nothing dated
    # at the day's vote is left to go stale.
    for duplicated in ("生存者", "死亡者:", "本日のCO", "本日公開された判定", "投票R1"):
        assert duplicated not in past
    assert f"{FACTS_HEADING}5日目" not in past


def test_a_day_whose_executed_seat_never_spoke_says_so():
    controller, state, _runtime, coordinator, _provider, _speaker, _coded, rogue = _v3_setup()
    state.phase = Phase.VOTE_RESULT
    state.vote_records.append(VoteRecord(voter_id="p1", target_id=rogue, day=2, round=1))
    state.players[rogue].alive = False
    state.players[rogue].death_cause = DeathCause.EXECUTED
    state.players[rogue].death_day = 2
    state.death_records.append(DeathRecord(player_id=rogue, cause=DeathCause.EXECUTED, day=2))

    asyncio.run(coordinator._generate_day_summary(controller, state))

    facts = coordinator._day_summaries.facts[2]
    name = state.players[rogue].name
    assert f"- 投票: {name}1票 → 処刑 {name}" in facts
    assert f"- 処刑された{name}はこの日発言していない" in facts


def test_the_coordinator_quotes_the_days_summary_line():
    controller, state, _runtime, coordinator, provider, speaker, _coded, rogue = _v3_setup()
    asyncio.run(coordinator._speak(controller, state, speaker, "consensus_summary"))
    posted = next(m for m in reversed(state.chat_log) if m.author_id == speaker)
    state.phase = Phase.VOTE_RESULT

    asyncio.run(coordinator._generate_day_summary(controller, state))

    facts = coordinator._day_summaries.facts[2]
    who = state.players[speaker].name
    assert f"- 進行役のまとめ（{who} {posted.message_id}）:「{posted.content}」" in facts


def test_v2_keeps_the_measured_summary():
    controller, state, _runtime, coordinator, _provider, _speaker, _coded, _rogue = _v3_setup()
    coordinator.model_decides = False
    state.phase = Phase.VOTE_RESULT

    asyncio.run(coordinator._generate_day_summary(controller, state))

    assert "- 生存者(" in coordinator._day_summaries.facts[2]
