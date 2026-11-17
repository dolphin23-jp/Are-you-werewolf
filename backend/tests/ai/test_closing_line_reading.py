"""How the day's closing lines are read: what they designate, who defends, who leads.

The pre-run integration review of the seed-1 fixes found three ways the close
could go wrong without anything failing:

- the summary was skipped when the replies had used the day's budget -- 6 of
  39 days closed with nobody named, against 2 of 41 before;
- 「今日はグレランで、仮指定はダイキ…」 read as a grey-random declaration, so
  the seat it had just designated never defended and there was no final
  designation (run 13, seed 3, twice);
- a plan-note hang target the line never named got a defence turn, against an
  accusation nobody at the table had heard.

And the freemason chat was asked 「どちらがまとめ役か」 though the code, not
the chat, picks who leads. These pin each.
"""

from __future__ import annotations

import asyncio

import pytest

from app.ai.context import ContextBuilder
from app.ai.coordinator import AICoordinator
from app.ai.metrics import MetricsCollector
from app.ai.provider.mock import MockProvider
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.sessions.models import DiscussionRoundState
from tests.ai.test_day_turn_schedule import (
    DEFENCE,
    DESIGNATION,
    PROVISIONAL,
    SUMMARY,
    _after,
    _asking,
    _day_two_table,
    _line,
    _session,
    _Table,
)
from tests.ai.test_listening import _v3_builder
from tests.conftest import make_controller

# -- the summary is never skipped for want of budget --


def test_a_day_whose_replies_used_the_budget_still_gets_its_summary() -> None:
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.DISCUSSION
    controller.co("p1", "freemason")
    metrics = MetricsCollector()
    coordinator = AICoordinator(
        state, ["p1", "p2", "p3"], MockProvider(seed=1, metrics=metrics), seed=1
    )
    spoken = [(seat, _line("見解を述べます。")) for seat in ("p1", "p2", "p3", "p2", "p3")]
    round_state = DiscussionRoundState(
        day=state.day,
        order=["p1", "p2", "p3"],
        cursor=3,
        stage="initial_view",
        outputs=spoken,
        speech_counts={"p1": 1, "p2": 2, "p3": 2},
        major_targets_ready=True,
        max_total=len(spoken),
    )

    speaker, stage = coordinator._next_discussion_speaker(state, round_state)

    assert (speaker, stage) == ("p1", SUMMARY)
    # Given outside the budget, so counted: the progress total grows by one.
    assert round_state.max_total == len(spoken) + 1
    assert "summary_without_speaker" not in metrics.events


def test_a_summary_with_nobody_to_give_it_is_counted() -> None:
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.DISCUSSION
    metrics = MetricsCollector()
    coordinator = AICoordinator(state, ["p1"], MockProvider(seed=1, metrics=metrics), seed=1)
    round_state = DiscussionRoundState(
        day=state.day, order=[], cursor=0, stage="initial_view", major_targets_ready=True
    )

    assert coordinator._next_discussion_speaker(state, round_state) == (None, SUMMARY)
    assert metrics.events.get("summary_without_speaker") == 1


# -- グレラン and a designation in the same breath is a designation --


@pytest.mark.parametrize(
    "line",
    [
        "今日はグレランで。",
        "今日はグレランにします。投票先は各自で。",
        "今日は指定なしのグレランでいきましょう。",
        "指定はしないので、グレランで。",
        "灰ラン。",
    ],
)
def test_a_plain_grey_random_line_is_a_declaration(line: str) -> None:
    assert AICoordinator._declares_grey_random(line, names_candidate=False)


@pytest.mark.parametrize(
    "line",
    [
        # Run 13, seed 3: both closed the day with the named seat unheard.
        "今日はグレランで、仮指定はダイキさんにします。反論があればどうぞ。",
        "今日はグレランでツムギさんを仮指定します。",
        "グレランでいいと思うけど、本指定はハルトで。",
    ],
)
def test_a_grey_random_line_that_designates_is_not_a_declaration(line: str) -> None:
    assert not AICoordinator._declares_grey_random(line, names_candidate=False)


def test_a_grey_random_line_that_names_its_own_candidate_is_not_a_declaration() -> None:
    assert not AICoordinator._declares_grey_random(
        "今日はグレランで。ツムギさんが一番怪しい。", names_candidate=True
    )


def test_grey_random_with_a_provisional_designation_hears_the_candidate() -> None:
    """The seed-3 shape at the table: the designated seat defends, then the final designation."""
    table = _Table(
        {
            SUMMARY: _line("今日はグレランで、仮指定はp5さん。弁明をどうぞ。", target="p5"),
            DEFENCE: _line("私は村人です。", target="p1"),
            DESIGNATION: _line("本指定はp5さん。", target="p5"),
        }
    )
    controller, coordinator = _day_two_table(table)
    session = _session(controller)

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(session), 30))

    assert _after(table.turns) == [("p1", SUMMARY), ("p5", DEFENCE), ("p1", DESIGNATION)]
    assert session.discussion_round.complete is True


def test_a_declared_grey_random_day_still_closes_without_a_designation() -> None:
    table = _Table({SUMMARY: _line("今日はグレランで。", target="p5")})
    controller, coordinator = _day_two_table(table)
    session = _session(controller)

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(session), 30))

    assert _after(table.turns) == [("p1", SUMMARY)]
    assert session.discussion_round.complete is True


# -- the candidate is a seat the line names --


def test_a_plan_note_target_the_summary_never_names_does_not_defend() -> None:
    """The note leans on p5; the line names nobody. No defence for p5 -- a provisional turn."""
    table = _Table(
        {
            SUMMARY: _line("ここまでの議論を整理します。皆さん気になる人はいますか？", target="p5"),
            PROVISIONAL: _line("仮指定はp7さん。", target="p7"),
            DEFENCE: _line("私は村人です。", target="p1"),
            DESIGNATION: _line("本指定はp7さん。", target="p7"),
        }
    )
    controller, coordinator = _day_two_table(table)
    session = _session(controller)

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(session), 30))

    after = _after(table.turns)
    assert ("p5", DEFENCE) not in after
    assert after == [
        ("p1", SUMMARY),
        ("p1", PROVISIONAL),
        ("p7", DEFENCE),
        ("p1", DESIGNATION),
    ]


def test_a_final_designation_whose_target_goes_unnamed_opens_no_safety_net() -> None:
    table = _Table(
        {
            SUMMARY: _asking("p7さん、どう思う？", "p7"),
            DESIGNATION: _line("では投票に移りましょう。", target="p9"),
        }
    )
    controller, coordinator = _day_two_table(table)
    session = _session(controller)

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(session), 30))

    assert ("p9", DEFENCE) not in _after(table.turns)
    assert session.discussion_round.complete is True


# -- the freemason chat is told who leads, not asked --


def _freemason_table():  # type: ignore[no-untyped-def]
    controller = make_controller(seed=4)
    state = controller.state
    masons = [p.player_id for p in state.players_by_role(RoleName.FREEMASON)]
    assert len(masons) == 2
    return controller, state, masons


def test_the_freemason_chat_is_not_asked_who_leads() -> None:
    controller, state, masons = _freemason_table()
    builder = _v3_builder(state)

    _system, messages = builder.build_freemason_chat_context(state, masons[0])
    prompt = "\n".join(str(m.content) for m in messages)

    assert "どちらがまとめ役か" not in prompt
    assert "まとめ役（進行役）は先に共有COした方が務める" in prompt


def test_the_freemason_chat_names_the_leader_the_scheduler_will_pick() -> None:
    controller, state, masons = _freemason_table()
    first, second = masons
    if state.players[first].is_human or state.players[second].is_human:
        pytest.skip("this seed seats the human as a freemason")
    controller.co(second, "freemason")
    controller.co(first, "freemason")
    ai_ids = [pid for pid in state.players if not state.players[pid].is_human]
    coordinator = AICoordinator(state, ai_ids, MockProvider(seed=1), seed=1)
    assert coordinator._facilitator(state) == second

    for seat, expected in (
        (second, "まとめ役（進行役）はあなたが務める"),
        (first, f"まとめ役（進行役）は{state.players[second].name}"),
    ):
        note = ContextBuilder._freemason_lead_note(state, seat)
        assert note.startswith(expected)


def test_with_a_human_partner_the_ai_freemason_leads() -> None:
    controller, state, masons = _freemason_table()
    ai_seat, human_seat = masons
    state.players[human_seat].is_human = True
    state.players[ai_seat].is_human = False

    assert ContextBuilder._freemason_lead_note(state, ai_seat).startswith(
        "まとめ役（進行役）はあなたが務める"
    )
