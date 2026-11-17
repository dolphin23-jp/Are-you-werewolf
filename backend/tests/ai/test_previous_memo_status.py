"""The previous plan note, shown with where it was written and what has become of
its names since (seed-1 review 2-A4, cause R3).

A note is only rewritten on a discussion turn, so a seat without one votes from
whatever it last wrote -- in the seed-1 game 37 of 83 ballots. ハルト (a wolf)
wrote his on day 1 at m8, before ホノカ confirmed her freemason claim, and was
still shown `"execution_target": "p13"` when he voted on day 2: an id, no date,
nothing to say that the seat it names had since been confirmed. The note is
kept (the intent it records is the seat's), but it is dated and its names carry
their current public status.
"""

from __future__ import annotations

import asyncio

from app.ai.context import ContextBuilder, DaySummaryManager
from app.ai.deception import FakeClaimGuard, WolfDeceptionAssignment
from app.ai.personalities import assign_personalities
from app.ai.schemas import VoteOutput
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.state import DeathCause, DeathRecord
from tests.ai.reasoning.fixtures import declare_co, publish_result
from tests.ai.seed1_replay import captured_prompts
from tests.ai.test_v3_coordinator import _setup as _v3_setup
from tests.conftest import make_controller


def _section(text: str, heading: str) -> str:
    """From `heading` to the next blank line."""
    return text.split(heading, 1)[1].split("\n\n", 1)[0]


# -- the replayed seed-1 moments --


def test_the_day_two_ballot_shows_the_day_one_note_dated_and_named():
    user = captured_prompts()["d2-vote-haruto"].user

    note = _section(user, "【前回の方針メモ")
    assert note.startswith("（1日目 m8・初回意見で書いたもの）】")
    assert "それ以降の出来事は【盤面メモ】にある" in note
    assert "- execution_target: ホノカ［共有・相互確認済み］" in note
    assert (
        "- suspects: ホノカ［共有・相互確認済み］、ケント［占いCO］、アオイ［占いCO・ケントの白］"
        in note
    )
    # No id survives into the note: the old echo was the JSON with ids.
    assert "p13" not in note
    assert '"execution_target"' not in note


def test_a_seat_back_after_two_silent_days_reads_its_old_target_as_it_stands():
    user = captured_prompts()["d4-m61-haruto-initial"].user

    digest = _section(user, "【前回の発言のあとに出たこと】")
    assert "- 前回のメモの吊り先（1日目 m8）: ホノカ［共有・相互確認済み］" in digest
    assert "あなたが推していた吊り先" not in digest
    note = _section(user, "【前回の方針メモ")
    assert note.startswith("（1日目 m8・初回意見で書いたもの）】")


def test_a_note_written_today_is_not_sent_to_the_board_memo():
    user = captured_prompts()["d5-m81-kento-rebuttal"].user

    note = _section(user, "【前回の方針メモ")
    assert note.startswith("（5日目 m75・朝一CO・結果発表で書いたもの）】")
    assert "それ以降の出来事" not in note


# -- the pieces --


def _builder(state) -> ContextBuilder:  # type: ignore[no-untyped-def]
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


def test_each_name_carries_its_public_status():
    controller = make_controller(seed=4)
    state = controller.state
    state.day = 3
    builder = _builder(state)
    declare_co(state, "p3", RoleName.SEER, day=1)
    publish_result(state, "p3", "seer", "p5", False, day=1)
    state.players["p6"].alive = False
    state.players["p6"].death_cause = DeathCause.EXECUTED
    state.players["p6"].death_day = 2
    state.death_records.append(DeathRecord(player_id="p6", cause=DeathCause.EXECUTED, day=2))
    state.players["p7"].alive = False
    state.players["p7"].death_cause = DeathCause.ATTACKED
    state.players["p7"].death_day = 2
    state.death_records.append(DeathRecord(player_id="p7", cause=DeathCause.ATTACKED, day=2))
    message = controller.chat("p1", "様子見", "public")
    builder.set_reasoning_memo(
        "p1",
        {
            "execution_target": "p6",
            "suspects": ["p6", "p7", "p8"],
            "trusted": ["p5"],
            "trusted_seer": "p3",
            "fox_candidates": [],
            "overall_thought": "p6が怪しい",
            "private_team_thought": "",
        },
        day=2,
        stage="initial_view",
    )
    builder.record_key_point(2, message, "p1", "", stage="initial_view")

    layer = builder._layer_previous_memo(state, "p1")

    assert layer.startswith(f"【前回の方針メモ（2日目 {message}・初回意見で書いたもの）】")
    assert "- execution_target: Player6［2日目処刑］" in layer
    assert "- suspects: Player6［2日目処刑］、Player7［2日目夜死亡］、Player8［グレー］" in layer
    assert "- trusted: Player5［Player3の白］" in layer
    assert "- trusted_seer: Player3［占いCO］" in layer
    assert "- fox_candidates: なし" in layer
    # Free text is the seat's own; an id typed into it is still read as a name.
    assert "- plan: Player6が怪しい" in layer
    assert "team_plan" not in layer


def test_the_coordinator_dates_the_note_with_the_turn_that_wrote_it():
    controller, state, _runtime, coordinator, provider, speaker, _coded, rogue = _v3_setup()
    asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))
    posted = next(
        m.message_id for m in reversed(state.chat_log) if m.author_id == speaker
    )
    state.phase = Phase.VOTING

    asyncio.run(coordinator._cast_vote(controller, state, speaker))

    _schema, _system, user = next(p for p in provider.prompts if p[0] is VoteOutput)
    note = _section(user, "【前回の方針メモ")
    assert note.startswith(f"（2日目 {posted}・初回意見で書いたもの）】")
    assert f"- execution_target: {state.players[rogue].name}［" in note


def test_a_note_without_a_known_source_keeps_the_plain_heading():
    state = make_controller(seed=4).state
    builder = _builder(state)
    builder.set_reasoning_memo("p1", {"execution_target": "p2"})

    layer = builder._layer_previous_memo(state, "p1")

    assert layer.startswith("【前回の方針メモ】\n")
    assert "- execution_target: Player2［グレー］" in layer
