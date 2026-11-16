"""The first morning's death line under the night-0 curse.

The owner ruled (2026-11-16) that a seer's night-0 divination curse-kills the
fox, as it does on every other night. Night 0 has no attack, so a body on the
first morning other than the scripted first victim can only be the cursed fox.
The death line used to say "attack or curse, can't tell" on every morning;
on this one that would hand the table a doubt the rules have settled.
"""

from __future__ import annotations

from app.ai.context import ContextBuilder, DaySummaryManager
from app.ai.deception import FakeClaimGuard, WolfDeceptionAssignment
from app.ai.personalities import assign_personalities
from app.engine.roles import RoleName
from app.engine.state import DeathCause, DeathRecord, GameState
from tests.conftest import make_controller

MEMO = "【盤面メモ】"


def _builder(state: GameState, engine: str) -> ContextBuilder:
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
        engine=engine,
    )


def _cursed_fox_on_night_zero() -> tuple[GameState, str]:
    state = make_controller(seed=4).state
    state.day = 1
    fox = next(p.player_id for p in state.players_by_role(RoleName.FOX))
    state.players[fox].alive = False
    state.death_records.append(DeathRecord(player_id=fox, cause=DeathCause.CURSED, day=0))
    return state, state.players[fox].name


def test_a_death_on_night_zero_is_named_as_the_cursed_fox():
    state, name = _cursed_fox_on_night_zero()

    _system, messages = _builder(state, "v3").build_discussion_context(
        state, "p1", "initial_view", board_memo=MEMO
    )
    body = messages[0].content
    line = next(item for item in body.splitlines() if item.startswith("【今朝の死体】"))

    assert name in line
    assert "噛みか呪殺かの区別はつかない" not in line
    assert "呪殺" in line and "妖狐" in line


def test_a_later_mornings_death_still_cannot_be_told_apart():
    state = make_controller(seed=4).state
    state.day = 3
    state.players["p5"].alive = False
    state.death_records.append(DeathRecord(player_id="p5", cause=DeathCause.ATTACKED, day=2))

    _system, messages = _builder(state, "v3").build_discussion_context(
        state, "p1", "initial_view", board_memo=MEMO
    )

    assert "【今朝の死体】Player5。公開情報では噛みか呪殺かの区別はつかない。" in (
        messages[0].content
    )


def test_the_structured_prompt_names_it_too_and_is_unchanged_on_later_mornings():
    state, name = _cursed_fox_on_night_zero()

    _system, messages = _builder(state, "legacy").build_discussion_context(state, "p1")
    line = next(
        item
        for item in messages[0].content.splitlines()
        if item.startswith("【今朝の公開死体（最優先で考察）】")
    )

    assert name in line and "呪殺" in line and "妖狐" in line
    assert "死因の区別はつきません" not in line

    later = make_controller(seed=4).state
    later.day = 3
    later.players["p5"].alive = False
    later.death_records.append(DeathRecord(player_id="p5", cause=DeathCause.ATTACKED, day=2))
    _system, messages = _builder(later, "legacy").build_discussion_context(later, "p1")
    assert (
        "Player5(p5)。公開情報では死因の区別はつきません。死体数と公開占い結果を照合してください。"
        in messages[0].content
    )
