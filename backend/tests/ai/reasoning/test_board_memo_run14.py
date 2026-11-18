"""The owner's review of run #14: board facts, derived conclusions and skill.

Each problem is pinned at the real moment of the run #14 games
(`run14_boards`), and the curse cases on a hand-dealt board:

* X1 -- contradictions and endgame consequences reach the board: the game
  still running is a rule, conditional rows exist for medium and dead
  claimants, and a broken account says "成り立たない" with its reason;
* X2 -- every night corpse is "not a wolf", and when each claimant died is a
  record with no conclusion drawn;
* X4 -- each night's grey sits beside the seer claimants' targets;
* X12 -- two corpses in one night, read against each claimant's account;
* 5-B2 -- the fox is told only the real seer's divination kills it;
* skill -- derived conclusions reach each seat at its own pace, records at
  once, and nothing role-private leaks through the new lines.
"""

from __future__ import annotations

from app.ai.personalities import assign_personalities
from app.ai.reasoning.board_memo import PRIVATE_HEADING, SELF, render_board_memo
from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import CommonPublicPerspective, PlayerPrivatePerspective
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.reasoning.skill import (
    DerivationClock,
    DerivedItem,
    Skill,
    Stamp,
    assign_skills,
    stamp_now,
    stated_at_table,
)
from app.ai.reasoning.solver import build_solver, has_role, not_role
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.state import ChatChannel, ChatMessage, GameState
from tests.ai.reasoning import run14_boards as r14
from tests.ai.reasoning.solver import boards

_IF_HEAD = "- 仮定つきで言えること（仮定が正しい場合だけ）:"


def _public(text: str) -> str:
    return text.split(PRIVATE_HEADING)[0]


def _private(text: str) -> str:
    return text.split(PRIVATE_HEADING)[1]


def _say(state: GameState, author: str, content: str) -> None:
    state.chat_log.append(
        ChatMessage(
            message_id=f"t{len(state.chat_log)}",
            author_id=author,
            content=content,
            channel=ChatChannel.PUBLIC,
            day=state.day,
        )
    )


# -- X1: the game still running, and the accounts that break on it --


def test_seed1_day6_daiki_genuine_and_truthful_is_impossible():
    """ダイキ had given white to all five other living seats: genuine and
    truthful, no wolf would be alive and the game would be over."""
    public = _public(r14.memo(r14.board(1, 6, "vote"), r14.MIO))

    assert (
        "  - ダイキが真の占い師で結果も正しいなら: 成り立たない"
        "（生存者に人狼が1人もいないことになるが、それなら試合はもう終わっている）"
    ) in public.splitlines()


def test_seed1_the_break_appears_only_once_the_sixth_white_is_published():
    """That morning ダイキ had five whites; one living seat was still unjudged."""
    public = _public(r14.memo(r14.board(1, 6), r14.MIO))

    assert "ダイキが真の占い師で結果も正しいなら: 成り立たない" not in public


def test_seed2_day6_kent_genuine_medium_is_impossible():
    """ケント called all five executions white; with every night corpse and the
    first victim no wolf, all three wolves would be among the six alive."""
    public = _public(r14.memo(r14.board(2, 6, "vote"), r14.KAITO))

    assert (
        "  - ケントが真の霊媒師で結果も正しいなら: 成り立たない"
        "（生存6人のうち人狼が3人以上になるが、それなら試合はもう終わっている）"
    ) in public.splitlines()


def test_seed2_dead_claimants_get_rows_too():
    """The real seer was bitten on night 1; a dead claimant's account is still
    an account (the section used to vanish once no seer claimant lived)."""
    public = _public(r14.memo(r14.board(2, 6, "vote"), r14.KAITO))

    assert "  - ダイキ(死亡)が真の占い師で結果も正しいなら: 死亡した人狼はリク" in public


def test_seed3_day6_ren_genuine_leaves_tsumugi_the_last_wolf():
    public = _public(r14.memo(r14.board(3, 6, "vote"), r14.HARUTO))

    (row,) = [line for line in public.splitlines() if line.startswith("  - レンが真の占い師")]
    assert row.startswith(
        "  - レンが真の占い師で結果も正しいなら: 生存の人狼はツムギだけ。"
        "ツムギを処刑すると人狼0人で試合が終わる"
    )
    # The rope count under the account, which the public line cannot give.
    assert "吊るべき人外は最大1人で、外してよい吊りは1回" in row


def test_seed3_the_conditional_rows_are_never_filed_as_certainties():
    public = _public(r14.memo(r14.board(3, 6, "vote"), r14.HARUTO))

    settled = public.split("- 公開情報だけで確定すること:")[1].split(_IF_HEAD)[0]
    assert "ツムギ" not in settled
    assert "試合が終わる" not in settled


def test_the_execution_line_states_what_ends_the_game_today():
    public = _public(r14.memo(r14.board(3, 6, "vote"), r14.HARUTO))

    assert (
        "- 今日の処刑で試合が終わるか（公開情報から）: 処刑した人が最後の人狼なら人狼0人で"
        "試合が終わる（生存者の誰でもありうる。妖狐が生きていれば妖狐の勝ち、死んでいれば"
        "村の勝ち）。人狼以外の処刑では今日は終わらない"
    ) in public.splitlines()


def test_the_execution_line_is_for_the_day_and_for_the_endgame():
    night = _public(r14.memo(r14.board(3, 5, "night"), r14.HARUTO))
    early = _public(r14.memo(r14.board(3, 2, "vote"), r14.HARUTO))

    assert "今日の処刑で試合が終わるか" not in night
    assert "今日の処刑で試合が終わるか" not in early


# -- X2: night corpses and when each claimant died --


def test_every_night_corpse_is_not_a_wolf_claimants_included():
    public = _public(r14.memo(r14.board(3, 6), r14.HARUTO))

    assert (
        "  - 夜に死んだアカリ、ソウタ、サクラ、カイト、ノゾミは人狼ではない"
        "（夜の死因は人狼以外への襲撃か、妖狐の呪殺だけ）"
    ) in public.splitlines()


def test_when_each_claimant_died_is_a_record_with_no_conclusion():
    public = _public(r14.memo(r14.board(2, 6), r14.KAITO))

    (line,) = [row for row in public.splitlines() if row.startswith("- 役職COの夜の死亡")]
    assert line == (
        "- 役職COの夜の死亡（夜の死体は襲撃か呪殺）: 占いCO ハルト=1日目夜に死亡、"
        "ダイキ=5日目処刑(夜の死亡なし) / 霊媒CO ケント=夜の死亡なし(生存)、レン=2日目夜に死亡"
        " / 共有CO アカリ=3日目夜に死亡、ホノカ=4日目夜に死亡"
    )
    # Owner decision 3: the pattern is shown, never read.
    for word in ("真", "偽", "寄り", "噛まれ"):
        assert word not in line


# -- X4: the grey beside each night's targets --


def test_each_nights_grey_is_shown_with_the_targets_chosen_from_it():
    """Seed 1, night 4: one grey seat left (カイト, a wolf partner), and
    ダイキ divined ホノカ, a confirmed freemason."""
    public = _public(r14.memo(r14.board(1, 6), r14.MIO))

    assert "  - 4日目夜（死体: マコト）〔カイト〕: ダイキ→ホノカ=白[5日目公開]" in public
    assert "  - 3日目夜（死体: アカリ）〔ハルト、カイト〕: ダイキ→レン=白[4日目公開]" in public
    # Night 0 is everyone, and says so by saying nothing.
    assert "  - 0日目夜（初日犠牲者: ノゾミ）: " in public


# -- 5-B2: the fox's own reading of the curse --


def test_the_fox_is_told_a_fake_seer_cannot_curse_it():
    private = _private(r14.memo(r14.board(3, 6), r14.HONOKA))

    (line,) = [row for row in private.splitlines() if row.startswith("- 呪殺")]
    assert "妖狐が死ぬのは真の占い師に占われた夜だけ（真でない占いCOに占われても死なない）" in line
    assert "あなたの視点で真の占い師でない占いCO: レン" in line
    assert "生存の占いCOに真の可能性は残っていない" in line


def test_only_the_fox_gets_the_curse_line():
    for seat in (r14.HARUTO, r14.REN, r14.TSUMUGI):
        assert "- 呪殺" not in r14.memo(r14.board(3, 6), seat)


# -- no leaks: every seat's public section is the same --


def test_every_seat_reads_the_same_public_section_on_day_six():
    """The new lines come from the public solver only: a wolf, the fox and a
    villager read the same public section, word for word."""
    for seed in (1, 2, 3):
        state = r14.board(seed, 6, "vote")
        sections = set()
        for seat in state.alive_ids():
            text = _public(r14.memo(state, seat)).partition("\n")[2]
            sections.add(text.replace(SELF, r14.name(seed, seat)))
        assert len(sections) == 1, seed


# -- X12: two corpses in one night --


def _curse_night() -> GameState:
    """p0 the real seer, p1 the fox, p5 the madman, p2 a wolf; three seer COs.

    Night 0: p0 divines p8; p5 claims p9 and p2 claims p1 (both white).
    Night 1: p0 divines p1 (cursed) and the wolves bite p0: two corpses.
    Day 2: p5 says night 1 was p10 (alive); p2 says it was p0 (a corpse).
    """
    state = boards.deal(
        {
            "p0": RoleName.SEER,
            "p1": RoleName.FOX,
            "p2": RoleName.WEREWOLF,
            "p3": RoleName.WEREWOLF,
            "p4": RoleName.WEREWOLF,
            "p5": RoleName.MADMAN,
        },
        day=2,
    )
    boards.kill_first_victim(state, "p16")
    for seat in ("p0", "p5", "p2"):
        boards.claim(state, seat, RoleName.SEER, day=1)
    boards.divine(state, "p0", "p8", night=0)
    boards.verdict(state, "p0", "seer", "p8", False, day=1, referenced_day=0)
    boards.verdict(state, "p5", "seer", "p9", False, day=1, referenced_day=0)
    boards.verdict(state, "p2", "seer", "p1", False, day=1, referenced_day=0)
    boards.execute(state, "p12", day=1)
    boards.divine(state, "p0", "p1", night=1)
    boards.die_by_curse(state, "p1", night=1)
    boards.attack(state, "p2", "p0", night=1, succeeded=True)
    boards.die_by_attack(state, "p0", night=1)
    boards.verdict(state, "p5", "seer", "p10", False, day=2, referenced_day=1)
    boards.verdict(state, "p2", "seer", "p0", False, day=2, referenced_day=1)
    return state


def _memo(state: GameState, seat: str, **kwargs: object) -> str:
    observations = ObservationSet.from_state(state)
    return render_board_memo(
        state,
        seat,
        observations=observations,
        public_solver=build_solver(observations, CommonPublicPerspective()),
        seat_solver=build_solver(observations, PlayerPrivatePerspective(seat)),
        **kwargs,  # type: ignore[arg-type]
    )


def test_a_claimant_whose_target_is_neither_corpse_cannot_be_the_seer():
    public = _public(_memo(_curse_night(), "p7"))

    assert "Player5は真の占い師ではありえない（1日目夜は死体2つ（Player0、Player1）" in public
    # p2 named a corpse: his account survives the curse.
    assert "Player2は真の占い師ではありえない" not in public


def test_the_two_corpses_themselves_are_a_record_every_seat_reads():
    """One of the two is the cursed fox, so the fox is dead: that much is the
    record and the rules, not a reading of anyone's account."""
    public = _public(_memo(_curse_night(), "p7", skill=Skill.NOVICE, clock=DerivationClock()))

    assert "1日目夜は死体が2つ（Player0、Player1）" in public
    assert "妖狐(1人)は死亡済み。生存者に妖狐はいない（Player0かPlayer1のどちらか）" in public
    assert "夜に死んだPlayer0、Player1は人狼ではない" in public


def test_a_white_on_the_cursed_fox_breaks_the_claimant_once_the_fox_is_known():
    """p2 published a night-0 white on p1, who lived through night 0. If p1
    is the fox, p2 cannot be the real seer: the real one would have cursed it
    that night."""
    solver = build_solver(boards.observe(_curse_night()), CommonPublicPerspective())

    assert solver.is_possible(has_role("p2", RoleName.SEER))
    assert solver.implies(has_role("p1", RoleName.FOX), not_role("p2", RoleName.SEER))
    assert solver.implies(has_role("p2", RoleName.SEER), has_role("p0", RoleName.FOX))


# -- skill: when a seat is shown a derived conclusion --

_BROKEN = "ダイキが真の占い師で結果も正しいなら: 成り立たない"
_RECORD = "夜に死んだマコト、アカリ、ホノカ、アオイ、ケントは人狼ではない"


def test_an_expert_sees_a_conclusion_the_moment_it_is_derivable():
    state = r14.board(1, 6, "vote")
    clock = DerivationClock()

    assert _BROKEN in r14.memo(state, r14.MIO, skill=Skill.EXPERT, clock=clock)


def test_an_average_seat_sees_it_the_next_day_and_a_novice_not_at_all():
    state = r14.board(1, 6, "vote")
    clock = DerivationClock()
    r14.memo(state, r14.MIO, skill=Skill.EXPERT, clock=clock)  # stamps it today

    for skill in (Skill.AVERAGE, Skill.NOVICE):
        text = r14.memo(state, r14.MIO, skill=skill, clock=clock)
        assert _BROKEN not in text, skill
        assert _RECORD in text, skill  # records reach every seat at once

    state.day = 7  # the same conclusion, a day later
    assert _BROKEN in r14.memo(state, r14.MIO, skill=Skill.AVERAGE, clock=clock)
    assert _BROKEN not in r14.memo(state, r14.MIO, skill=Skill.NOVICE, clock=clock)


def test_once_stated_at_the_table_every_seat_sees_it():
    state = r14.board(1, 6, "vote")
    clock = DerivationClock()
    r14.memo(state, r14.MIO, skill=Skill.EXPERT, clock=clock)
    _say(state, r14.SAKURA, "ダイキの白が全員に出てる。ダイキが真なら人狼がいないから破綻してる")

    for skill in (Skill.AVERAGE, Skill.NOVICE):
        assert _BROKEN in r14.memo(state, r14.MIO, skill=skill, clock=clock), skill


def test_a_derived_fact_hidden_from_a_seat_is_not_restated_as_its_own():
    """The curse refutation is public logic: a novice does not see it yet, and
    its private tail must not hand it over as "your own" conclusion."""
    state = _curse_night()
    text = _memo(state, "p7", skill=Skill.NOVICE, clock=DerivationClock())

    assert "Player5は真の占い師ではありえない" not in text
    assert "Player5は真の占い師ではありえない" in _memo(state, "p7", skill=Skill.EXPERT)


def test_the_table_statement_rule():
    state = r14.board(1, 6, "vote")
    names = {pid: player.name for pid, player in state.players.items()}
    item = DerivedItem(key=("broken", r14.DAIKI), text="", seats=(r14.DAIKI,))
    since = stamp_now(state)

    assert not stated_at_table(state, item, since, names)
    _say(state, r14.SAKURA, "ダイキさんは怪しいと思う")  # the seat, no contradiction word
    _say(state, r14.SAKURA, "誰かが破綻してる気がする")  # the word, no seat
    assert not stated_at_table(state, item, since, names)
    _say(state, r14.SAKURA, "ダイキは偽確定でしょ")
    assert stated_at_table(state, item, since, names)
    # Said before the conclusion was derivable: not a statement of it.
    later = Stamp(day=state.day, position=len(state.chat_log))
    assert not stated_at_table(state, item, later, names)


def test_skills_are_dealt_from_the_seed_apart_from_the_personas():
    seats = [f"p{i}" for i in range(16)]

    first, again = assign_skills(seats, seed=3), assign_skills(seats, seed=3)
    assert first == again
    assert {level: list(first.values()).count(level) for level in Skill} == {
        Skill.EXPERT: 6,
        Skill.AVERAGE: 5,
        Skill.NOVICE: 5,
    }
    assert assign_skills(seats, seed=4) != first
    # Its own random stream: dealing skills leaves the persona deal as it was.
    before = assign_personalities(seats, seed=3)
    assign_skills(seats, seed=3)
    assert assign_personalities(seats, seed=3) == before


def test_the_runtime_shows_each_seat_the_memo_its_skill_allows():
    state = r14.board(3, 6, "vote")
    state.phase = Phase.VOTING
    seats = [pid for pid in state.players if pid != r14.MIO]
    runtime = ReasoningRuntime(state, seats, seed=1)
    runtime.skills[r14.HARUTO] = Skill.EXPERT
    runtime.skills[r14.AOI] = Skill.NOVICE

    expert = runtime.board_memo(state, r14.HARUTO)
    novice = runtime.board_memo(state, r14.AOI)

    assert "レンが真の占い師で結果も正しいなら: 生存の人狼はツムギだけ" in expert
    assert "レンが真の占い師で結果も正しいなら" not in novice
    assert "今日の処刑で試合が終わるか" not in novice
    assert "夜に死んだアカリ、ソウタ、サクラ、カイト、ノゾミは人狼ではない" in novice
