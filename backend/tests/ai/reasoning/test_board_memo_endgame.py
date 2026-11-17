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

from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import PlayerPrivatePerspective
from app.ai.reasoning.solver import build_solver
from app.ai.strategy import StrategyAnalyzer
from app.engine.game import GameController, PlayerSpec
from app.engine.phases import Phase
from app.engine.roles import RoleName, Team
from app.engine.state import DeathCause, DeathRecord, GameState, PlayerState
from app.engine.victory import VictoryChecker
from app.engine.vote import DEFAULT_MAX_VOTE_ROUNDS
from tests.ai.reasoning import run13_seed1_boards as s1
from tests.ai.reasoning.solver import boards

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
    public = s1.shared_part(s1.memo(s1.board(1), s1.REN), s1.REN)

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
    day_one = s1.shared_part(s1.memo(s1.board(1), s1.REN), s1.REN)
    assert (
        "縄: 生存16人で残り7回（(生存-1)÷2、毎晩1人死ぬ場合）。"
        "吊るべき人外は最大4人（人狼3・妖狐1）→ 外してよい吊りは3回"
    ) in day_one

    day_five = s1.shared_part(s1.memo(s1.board(5), s1.REN), s1.REN)
    assert "縄: 生存11人で残り5回" in day_five
    assert "外してよい吊りは1回" in day_five


def test_at_night_the_rope_count_is_for_the_morning_after_a_death():
    night_four = s1.shared_part(s1.memo(s1.board(4, "night"), s1.REN), s1.REN)

    assert "縄: 今夜1人死ねば明日は生存10人で残り4回" in night_four
    assert "外してよい吊りは0回" in night_four


def test_the_legacy_rope_count_uses_the_same_formula():
    analyzer = StrategyAnalyzer()

    assert analyzer._rope_count(3, 16) == 3
    assert analyzer._rope_count(3, 11) == 1
    assert analyzer._rope_count(3, 11, fox_may_be_alive=False) == 2
    assert analyzer._rope_count(3, 6) == 0


# -- 2. what ends the game, under each seer claim and from each seat --

_KENT_IF = "ケントが真の占い師で結果も正しいなら: "
_FOX_OPEN = (
    "妖狐の死亡は未確定（生存の候補8人: マコト、ソウタ、ミオ、レン、サクラ、ツムギ、ホノカ、シオン"
    "／死亡の候補: ユイ）"
)


def _conditional_lines(public: str) -> list[str]:
    head = "- 仮定つきで言えること（仮定が正しい場合だけ）:"
    lines = public.splitlines()
    if head not in lines:
        return []
    out: list[str] = []
    for line in lines[lines.index(head) + 1 :]:
        if not line.startswith("  - "):
            break
        out.append(line[4:])
    return out


def test_kents_seat_on_day_five_is_told_that_hanging_kaito_ends_the_game():
    """5-B1: ケント knew all three wolves after m75, and that a fox might live."""
    private = s1.private_part(s1.memo(s1.board(5, "m75"), s1.KENT))

    line = _line(private, "- 終局（あなたの視点）: ")
    assert (
        "死亡した人狼はハルト、ダイキ。生存の人狼はカイトだけ。"
        "カイトを処刑すると人狼0人で試合が終わる"
        "（妖狐が生きていれば妖狐の勝ち、死んでいれば村の勝ち）。" + _FOX_OPEN
    ) in line


def test_the_public_section_says_the_same_under_kents_claim_and_names_the_assumption():
    public = s1.shared_part(s1.memo(s1.board(5, "m75"), s1.REN), s1.REN)

    (line,) = _conditional_lines(public)
    assert line.startswith(_KENT_IF)
    assert "カイトを処刑すると人狼0人で試合が終わる" in line
    assert _FOX_OPEN in line
    # A supposition, not a fact: it is never filed with the certainties.
    settled = public.split("- 公開情報だけで確定すること:")[1].split("\n- ")[0]
    assert "試合が終わる" not in settled


def test_on_day_four_kents_claim_names_a_living_wolf_but_no_end_yet():
    public = s1.shared_part(s1.memo(s1.board(4, "m57"), s1.REN), s1.REN)

    (line,) = _conditional_lines(public)
    assert line.startswith(
        _KENT_IF + "死亡した人狼はダイキ。生存の人狼にハルトを含む（生存の人狼は全部で1〜2人）"
    )
    assert "試合が終わる" not in line


def test_a_seer_claim_that_settles_nothing_gets_no_line():
    """Day 2: three claimants, whites only. Nothing follows about wolves or the
    fox from any one of them, so there is nothing to say yet."""
    public = s1.shared_part(s1.memo(s1.board(2, "m34"), s1.REN), s1.REN)

    assert _conditional_lines(public) == []


def test_the_endgame_line_agrees_with_the_victory_checker():
    """Hang カイト in any world consistent with ケント's view: the checker ends
    the game. Hang anyone else: it does not."""
    state = s1.board(5, "m75")
    solver = build_solver(ObservationSet.from_state(state), PlayerPrivatePerspective(s1.KENT))
    worlds = [state]
    for model in solver.representative_models([], limit=8):
        world = s1.board(5, "m75")
        for pid, role in model.roles:
            world.players[pid].role = role
        worlds.append(world)
    assert len(worlds) > 2
    for world in worlds:
        fox_alive = any(p.alive and p.role is RoleName.FOX for p in world.players.values())
        for hanged in world.alive_ids():
            world.players[hanged].alive = False
            result = VictoryChecker().check(world)
            world.players[hanged].alive = True
            if hanged == s1.KAITO:
                assert result is not None
                assert result.winner is (Team.FOX if fox_alive else Team.VILLAGE)
            else:
                assert result is None, hanged


def test_wolves_and_the_fox_are_told_their_own_way_to_win():
    state = s1.board(4, "m57")

    wolf = _line(s1.private_part(s1.memo(state, s1.KAITO)), "- 終局（あなたの視点）: ")
    assert wolf.startswith(
        "- 終局（あなたの視点）: 生存の人狼はハルト、自分の2人、人狼以外は10人。"
    )
    assert "人狼以外があと8人減ると同数で試合が終わる" in wolf
    assert "人狼陣営の勝ちには妖狐の死亡が要る" in wolf
    assert "候補" not in wolf

    fox = _line(s1.private_part(s1.memo(state, s1.SOTA)), "- 終局（あなたの視点）: ")
    assert (
        "人狼が0人になるか、人狼が人狼以外と同数以上になった時点であなた(妖狐)が生きていれば、あなたの勝ち"
    ) in fox
    assert "候補" not in fox


def test_a_villager_with_nothing_settled_gets_no_endgame_line():
    private = s1.private_part(s1.memo(s1.board(4, "m57"), s1.REN))

    assert "終局" not in private


def _two_wolves_left(*, phase: Phase) -> GameState:
    """Five alive: the claimant p4, wolves p1 and p2 (both blacked by p4), p5, p6.
    p3, the third wolf, was hanged and blacked too."""
    state = boards.deal(
        {"p1": RoleName.WEREWOLF, "p2": RoleName.WEREWOLF, "p3": RoleName.WEREWOLF,
         "p4": RoleName.SEER, "p7": RoleName.FOX},
        day=4,
    )
    boards.kill_first_victim(state, "p16")
    boards.claim(state, "p4", RoleName.SEER, day=1)
    for night, target in enumerate(("p3", "p1", "p2")):
        boards.divine(state, "p4", target, night=night)
        boards.verdict(state, "p4", "seer", target, True, day=night + 1)
    for pid in ("p0", "p3", "p7", "p8", "p9", "p10", "p11"):
        boards.execute(state, pid, day=1)
    for pid in ("p12", "p13", "p14", "p15"):
        boards.die_by_attack(state, pid, night=1)
    state.phase = phase
    return state


def test_parity_is_said_for_the_next_execution_by_day_and_for_the_bite_by_night():
    """With two wolves among five, hanging a non-wolf brings them level. At
    night the bite comes first, so the line has to say that instead."""
    assert len(_two_wolves_left(phase=Phase.DISCUSSION).alive_ids()) == 5
    by_day = s1.public_part(s1.memo(_two_wolves_left(phase=Phase.DISCUSSION), "p5"))
    assert "人狼以外を処刑すると人狼が同数になり試合が終わる" in by_day

    by_night = s1.public_part(s1.memo(_two_wolves_left(phase=Phase.NIGHT), "p5"))
    assert "今夜の襲撃が通ると人狼が同数になり試合が終わる" in by_night
    assert "人狼以外を処刑すると" not in by_night
