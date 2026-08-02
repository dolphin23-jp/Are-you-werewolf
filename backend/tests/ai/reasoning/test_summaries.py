"""The factual half of a day summary is code-generated, so it must be exact,
deterministic, and free of both opinion and private information."""

from __future__ import annotations

from app.ai.reasoning import (
    PublicFactLedger,
    compose_day_summary,
    render_public_fact_summary,
    split_day_summary,
)
from app.ai.reasoning.summaries import FACTS_HEADING, OPINION_HEADING
from app.engine.roles import RoleName
from tests.ai.reasoning.fixtures import (
    cast_vote,
    declare_co,
    execute,
    kill_at_night,
    make_first_victim,
    make_state,
    publish_result,
)


def _played_out_day() -> PublicFactLedger:
    state = make_state(
        day=2,
        roles={"p3": RoleName.WEREWOLF, "p5": RoleName.SEER, "p9": RoleName.FOX},
    )
    make_first_victim(state, "p0")
    kill_at_night(state, "p6", day=1)
    declare_co(state, "p5", RoleName.SEER, day=2)
    publish_result(state, "p5", "seer", "p3", is_werewolf=True, day=2)
    cast_vote(state, "p5", "p3", day=2, round_number=1)
    cast_vote(state, "p1", "p3", day=2, round_number=1)
    execute(state, "p3", day=2)
    return PublicFactLedger(state)


def test_the_public_fact_summary_is_deterministic():
    ledger = _played_out_day()

    assert render_public_fact_summary(ledger) == render_public_fact_summary(ledger)


def test_the_public_fact_summary_states_what_happened_and_nothing_else():
    summary = render_public_fact_summary(_played_out_day())

    assert summary.startswith(f"{FACTS_HEADING}2日目")
    assert "- 本日のCO: Player5(p5)=占い師" in summary
    assert "- 本日公開された判定: Player5(p5)の占い: Player3(p3)=黒" in summary
    assert "- 投票R1: Player5(p5)→Player3(p3)、Player1(p1)→Player3(p3)" in summary
    assert "- 処刑結果: Player3(p3)" in summary
    assert "Player0(p0)[0日目初日犠牲者]" in summary
    assert "Player6(p6)[1日目夜死亡]" in summary


def test_no_private_role_leaks_into_the_public_fact_summary():
    summary = render_public_fact_summary(_played_out_day())

    # p3 is really a werewolf and p9 a fox, but only p5's *claimed* seer role
    # was ever said out loud.
    assert "人狼" not in summary
    assert "妖狐" not in summary
    assert "襲撃" not in summary
    assert summary.count("占い師") == 1


def test_no_speculation_or_opinion_enters_the_fact_block():
    facts = render_public_fact_summary(_played_out_day())
    combined = compose_day_summary(facts, "p3が黒だったので占いは真だと思われる。")

    assert combined.startswith(facts)
    assert OPINION_HEADING in combined
    # The generated commentary is quarantined behind its own heading, so nothing
    # downstream can read it back as something that happened.
    assert split_day_summary(combined) == (facts, "p3が黒だったので占いは真だと思われる。")
    assert "思われる" not in split_day_summary(combined)[0]


def test_an_empty_commentary_leaves_the_facts_untouched():
    facts = render_public_fact_summary(_played_out_day())

    assert compose_day_summary(facts, "   ") == facts
    assert split_day_summary(facts) == (facts, "")


def test_a_summary_needs_no_llm_and_no_network():
    # Nothing here touches a provider: the ledger goes straight to a string.
    state = make_state(day=1)
    execute(state, "p2", day=1)

    summary = render_public_fact_summary(PublicFactLedger(state), day=1)

    assert "- 処刑結果: Player2(p2)" in summary
    assert "- 生存者(11人):" in summary
