"""When the game ends, and what ends it: the memo's win rules, rope count and
conditional endgame lines, pinned against the engine and against run #13 seed 1.

The owner's seed-1 review (5-B1, 5-B2) found the village hanging カイト, the
last wolf, without a word anywhere in the prompt that the execution would end
the game and hand it to a living fox. Every number and clause below is checked
against `app/engine` -- a rule the memo states and the engine does not play is
worse than no rule at all.
"""

from __future__ import annotations

import itertools

from app.ai.strategy import StrategyAnalyzer
from app.engine.game import GameController, PlayerSpec
from app.engine.phases import Phase
from app.engine.roles import RoleName, Team
from app.engine.state import DeathCause, DeathRecord, GameState, PlayerState
from app.engine.victory import VictoryChecker
from app.engine.vote import DEFAULT_MAX_VOTE_ROUNDS
from tests.ai.reasoning import run13_seed1_boards as s1

_END_RULE = (
    "- 勝敗: 処刑のあとと夜明けに判定。人狼が0人になるか、人狼が人狼以外の生存者"
    "（狂人・妖狐も数える）と同数以上になった時点で試合終了。"
    "その時点で妖狐が生きていれば妖狐の勝ち。妖狐が死んでいれば、人狼0人なら村、"
    "同数以上なら人狼陣営（狂人を含む）の勝ち。"
    f"投票が決選を含め{DEFAULT_MAX_VOTE_ROUNDS}回続けて同数なら引き分け"
)


def _line(public: str, prefix: str) -> str:
    return next(line for line in public.splitlines() if line.startswith(prefix))


# -- 1. the rules of the end, and the rope count --


def test_the_memo_states_when_the_game_ends_and_who_wins():
    public = s1.public_part(s1.memo(s1.board(1), s1.REN))

    assert _END_RULE in public.splitlines()


def _rule_as_written(wolves: int, non_wolves: int, fox_alive: bool) -> Team | None:
    """The end-rule line, clause by clause."""
    if wolves == 0 or wolves >= non_wolves:
        if fox_alive:
            return Team.FOX
        return Team.VILLAGE if wolves == 0 else Team.WEREWOLF
    return None


def test_the_end_rule_line_says_what_the_victory_checker_decides():
    """Every small table the rule line can describe, decided both ways."""
    for wolves, villagers, madman, fox in itertools.product(range(3), range(4), (0, 1), (0, 1)):
        roles = (
            [RoleName.WEREWOLF] * wolves
            + [RoleName.VILLAGER] * villagers
            + [RoleName.MADMAN] * madman
            + [RoleName.FOX] * fox
        )
        if not roles:
            continue
        players = {
            f"p{index}": PlayerState(player_id=f"p{index}", name=f"P{index}", role=role)
            for index, role in enumerate(roles)
        }
        state = GameState(session_id="rule", players=players)
        result = VictoryChecker().check(state)
        expected = _rule_as_written(wolves, len(roles) - wolves, bool(fox))
        assert (result.winner if result else None) == expected, (wolves, villagers, madman, fox)


def _table(
    wolves: int, villagers: int, *, fox: bool
) -> tuple[GameController, list[str], list[str], str | None]:
    """A day-2 discussion with only the named seats alive; every other is dead."""
    specs = [PlayerSpec(player_id=f"p{i}", name=f"P{i}") for i in range(17)]
    controller = GameController("end", specs, seed=3)
    players = controller.state.players
    wolf_ids = [pid for pid, p in players.items() if p.role is RoleName.WEREWOLF][:wolves]
    village_ids = [pid for pid, p in players.items() if p.role is RoleName.VILLAGER][:villagers]
    fox_id = next(pid for pid, p in players.items() if p.role is RoleName.FOX) if fox else None
    keep = {*wolf_ids, *village_ids, *([fox_id] if fox_id else [])}
    for pid, player in players.items():
        if pid in keep:
            continue
        player.alive = False
        player.death_cause = DeathCause.EXECUTED
        player.death_day = 1
        controller.state.death_records.append(
            DeathRecord(player_id=pid, cause=DeathCause.EXECUTED, day=1)
        )
        if player.role is RoleName.WEREWOLF:
            controller._alpha_tracker.on_wolf_death(pid)
    controller.state.phase = Phase.DISCUSSION
    controller.state.day = 2
    return controller, wolf_ids, village_ids, fox_id


def test_the_game_is_decided_right_after_an_execution():
    """Hanging the last wolf ends the game there: no night follows."""
    controller, (wolf,), villagers, fox = _table(1, 2, fox=True)
    assert fox is not None
    controller.end_discussion()
    for voter in (*villagers, fox):
        controller.vote(voter, wolf)
    controller.vote(wolf, villagers[0])
    controller.resolve_votes()

    assert controller.state.phase is Phase.GAME_OVER
    assert controller.state.winner is Team.FOX


def test_the_game_is_decided_at_dawn():
    """A bite that brings the wolves level ends the game when the night resolves."""
    controller, wolves, villagers, _ = _table(2, 4, fox=False)
    hanged, bitten, *others = villagers
    controller.end_discussion()
    for voter in (*wolves, bitten, *others):
        controller.vote(voter, hanged)
    controller.vote(hanged, bitten)
    controller.resolve_votes()
    assert controller.state.phase is Phase.VOTE_RESULT
    controller.start_night()
    controller.submit_night_action(controller.alpha_wolf_id, "attack", bitten)
    controller.resolve_night()

    assert controller.state.phase is Phase.GAME_OVER
    assert controller.state.winner is Team.WEREWOLF


def test_the_rope_count_is_half_the_living_less_the_inhumans_still_to_hang():
    """17A: 16 alive on day 1 is 7 executions for 3 wolves and the fox -- 3 spare."""
    day_one = s1.public_part(s1.memo(s1.board(1), s1.REN))
    assert (
        "縄: 生存16人で残り7回（(生存-1)÷2、毎晩1人死ぬ場合）。"
        "吊るべき人外は最大4人（人狼3・妖狐1）→ 外してよい吊りは3回"
    ) in day_one

    day_five = s1.public_part(s1.memo(s1.board(5), s1.REN))
    assert "縄: 生存11人で残り5回" in day_five
    assert "外してよい吊りは1回" in day_five


def test_at_night_the_rope_count_is_for_the_morning_after_a_death():
    night_four = s1.public_part(s1.memo(s1.board(4, "night"), s1.REN))

    assert "縄: 今夜1人死ねば明日は生存10人で残り4回" in night_four
    assert "外してよい吊りは0回" in night_four


def test_the_legacy_rope_count_uses_the_same_formula():
    analyzer = StrategyAnalyzer()

    assert analyzer._rope_count(3, 16) == 3
    assert analyzer._rope_count(3, 11) == 1
    assert analyzer._rope_count(3, 11, fox_may_be_alive=False) == 2
    assert analyzer._rope_count(3, 6) == 0
