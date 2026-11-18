"""Each faction's win, stated on its v3 role card as the rule it is, and the fox
kept quiet about its aim like the other non-village roles (seed-1 review 1-C1,
5-B2).

ソウタ (the fox) argued at the table about who his seer had chosen and why --
the instruction to keep one's real aim out of speech named wolves and madmen
only. His card said "生き残れば勝利する" without saying when the game ends, and
on day 5 nobody, the fox included, read the last wolf's execution as the
moment the fox would take the win. The win conditions are those of
`app/engine/victory.py`.
"""

from __future__ import annotations

from app.ai.context import ContextBuilder, DaySummaryManager
from app.ai.deception import FakeClaimGuard, WolfDeceptionAssignment
from app.ai.personalities import assign_personalities
from app.engine.roles import RoleName, Team
from app.engine.state import DeathCause
from app.engine.victory import VictoryChecker
from tests.ai.seed1_replay import captured_prompts
from tests.conftest import make_controller

FOX_WIN = (
    "【勝利条件】人狼が0人になった時点、または人狼が人狼以外と同数以上になった時点で、"
    "生きていれば妖狐の勝ち（村・人狼の勝ちを奪う）"
)
VILLAGE_WIN = "【勝利条件】人狼が0人になった時点で村の勝ち。ただし妖狐が生きていれば妖狐の勝ち"
WOLF_WIN = (
    "【勝利条件】人狼が人狼以外（狂人・妖狐も数える）と同数以上になった時点で人狼陣営の勝ち。"
    "ただし妖狐が生きていれば妖狐の勝ち。人狼が0人になれば負け"
)
SECRECY = "人狼・狂人・妖狐は本当の目的を発言で漏らさない"


def test_the_fox_is_told_when_it_wins_and_to_keep_its_aim_out_of_speech():
    system = captured_prompts()["d1-m14-souta-initial"].system

    assert FOX_WIN in system
    assert SECRECY in system


def test_every_faction_reads_its_own_win():
    prompts = captured_prompts()
    for tag, line in (
        ("d5-m81-kento-rebuttal", VILLAGE_WIN),
        ("d2-m35-makoto-initial", VILLAGE_WIN),
        ("d4-m61-haruto-initial", WOLF_WIN),
        ("d2-m34-aoi-immediate", WOLF_WIN),
    ):
        system = prompts[tag].system
        assert line in system, tag
        assert SECRECY in system, tag
        others = {FOX_WIN, VILLAGE_WIN, WOLF_WIN} - {line}
        assert not any(other in system for other in others), tag


def test_the_cards_say_what_the_engine_decides():
    """The fox steals both wins; without it each side wins as its line says."""
    checker = VictoryChecker()
    state = make_controller(seed=4).state
    wolves = [p.player_id for p in state.players_by_role(RoleName.WEREWOLF)]
    for wolf in wolves:
        state.players[wolf].alive = False
        state.players[wolf].death_cause = DeathCause.EXECUTED
    result = checker.check(state)
    assert result is not None and result.winner is Team.FOX
    fox = state.players_by_role(RoleName.FOX)[0].player_id
    state.players[fox].alive = False
    result = checker.check(state)
    assert result is not None and result.winner is Team.VILLAGE


def test_legacy_and_v2_cards_are_unchanged():
    state = make_controller(seed=4).state
    wolf_ids = [p.player_id for p in state.players_by_role(RoleName.WEREWOLF)]
    builder = ContextBuilder(
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
        engine="v2",
    )
    fox = state.players_by_role(RoleName.FOX)[0].player_id

    assert "【勝利条件】" not in builder._layer_b_role(state, fox)
