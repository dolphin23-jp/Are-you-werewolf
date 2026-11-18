"""盤面メモ: the board the way a player keeps it in their own notes.

The v2 engine handed the model a *brief* -- conclusions in fixed wording, to be
repeated. This module hands it the *board* instead: who is alive, who claimed
what, every published verdict, every ballot, what the rules make certain, what
the calendar makes impossible, and what this seat alone knows. Then the model
thinks, which is the part it is good at, and the part the brief had taken away.

Three properties are load-bearing:

* **Names, never ids.** Every public line the table reads is written in names.
  The id form (`ユイ(p3)`) was a prompt convention that leaked into speech; a
  roster rendered once by `render_roster` is where the model looks ids up for
  its structured answer.
* **Public first, private last and labelled.** Everything above the final
  section is derived from the ledger and a public-perspective solver, so the
  model cannot mistake its own card for something the table can see. The
  final section is this seat's own knowledge, read through its perspective.
* **Logic is stated as logic.** "Two seer claims, one seat" is arithmetic the
  rules settle; "the attacked claimant was probably real" is not, and does not
  appear here. The memo says what is certain and what is impossible, and leaves
  what is likely to the player.

The first live Claude game (2026-11-16, analysis.md P1) showed what a model does
when the memo leaves arithmetic to it: it counts roles it was never told,
argues from opposite readings of the night-0 curse on consecutive days,
forgets a dead seer's unpublished night, and keeps "caring" for a fox that died
on night 1. So the memo now also carries, every time:

* the composition and the handful of rules the table reasons from, each
  checked against `app/engine` (a rule the engine does not play is worse than
  none);
* every seer claimant's account night by night, with the day each verdict
  was published and the nights nobody published, next to that night's corpses;
* what the public solver settles about the dead as well as the living -- a
  role whose seats are all in the graveyard is said in so many words -- and
  which claimants the night's curse rules out as the real seer.

The owner's review of run #13 seed 1 (seed1-review-causes.md, R2) found the
memo stating the board and leaving out what follows from it. It now also says
what code can guarantee and nothing it cannot -- facts, and conclusions under a
named supposition, never advice:

* when the game ends and who wins then, and the rope count;
* each execution's medium read, or that none was published (and, with no
  medium claimant alive, that none will be); each night's corpses, and that an
  empty morning means the bite hit the guard target or the fox;
* the grey, the CO count with its zeros, and every ballot read both ways;
* under "if this living seer claimant is real and truthful", which wolves are
  dead or alive, which execution ends the game, and where the fox may be --
  in a block of its own, never among the certainties;
* in the seat's own section: the same endgame from its own solver (or its own
  side's way to win), a seer's "neither wolf nor fox", published colours it
  knows are false, the hunter's empty mornings, and conclusions under
  `NoVillageBluff` / `TruthfulIfGenuine`, each opening with its supposition --
  under a header that tells the seat to keep them off the table.

The owner's review of run #14 found the contradictions and endgame
consequences never reaching the board, and decided how they should: records
reach every seat at once, conclusions derived from them reach each seat at its
own pace (`skill.py`). So the public section now splits the two:

* records -- when each claimant died (no reading of the pattern: "only the
  rival was bitten" is no rule, the survivor may be the madman), every night
  corpse "not a wolf", each night's grey beside the seer claimants' targets;
* derived -- facts the solver reaches only with the curse read against a
  claimant's account or with the game still running, the conditional rows
  (now for medium and dead claimants too, and "成り立たない" with its reason
  when an account cannot be genuine and truthful), and the execution line.
  Code computes them every time; `skill` and `clock` decide whether this seat
  is shown them yet. A hidden one is not restated in the private tail either.

The reader is 自分 in its own memo, past the header.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from app.ai.reasoning.facts import MEDIUM_RESULT, SEER_RESULT, PublicFactLedger
from app.ai.reasoning.observations import ObservationSet, PublicVerdict
from app.ai.reasoning.perspectives import PlayerPrivatePerspective
from app.ai.reasoning.skill import (
    DerivationClock,
    DerivedItem,
    Skill,
    is_visible,
)
from app.ai.reasoning.solver.assumptions import (
    NON_VILLAGE_ROLES,
    GenuineClaim,
    HonestResults,
    NoLatentClaim,
    NoVillageBluff,
    TruthfulIfGenuine,
)
from app.ai.reasoning.solver.backend import (
    AnyOf,
    Hypothesis,
    RoleCountAmong,
    RoleIs,
    RoleIsNot,
    has_role,
)
from app.ai.reasoning.solver.queries import RoleSolver
from app.ai.reasoning.solver.rules import default_rule_modules, plain_rule_modules
from app.ai.reasoning.solver.rules.claimed_divination import (
    ClaimedDivinationRuleModule,
    SeerClaimRefutation,
    curse_nights,
    seer_claim_refutations,
)
from app.ai.reasoning.solver.rules.game_running import GameRunningRuleModule
from app.ai.reasoning.timeline import (
    can_act_on_night,
    could_publish_on_day,
    find_timeline_conflicts,
)
from app.engine.phases import Phase
from app.engine.roles import ROLE_DEFINITIONS, TOTAL_PLAYERS, RoleName
from app.engine.state import GameState, PublicDeathCause
from app.engine.vote import DEFAULT_MAX_VOTE_ROUNDS

MEMO_HEADING = "【盤面メモ】"
PRIVATE_HEADING = "【あなただけが知っていること】"
# How a seat's memo names the seat itself, past the header (review 2-C1).
SELF = "自分"
# On every header of conclusions only this seat can draw (review 1-C1: the fox
# argued in public from its own curse reasoning). Not on the seat's card or
# results, which a seer or medium is there to publish.
_KEEP_UNSAID = "卓ではこの結論を知らないものとして話し、匂わせない"


class _SeatLedger(PublicFactLedger):
    """The ledger as one seat reads it: its own name is 自分.

    The memo used to write the reader in the third person -- ソウタ's own memo
    said 「アオイ→ソウタ=白」 -- and seats came to speak of themselves the same
    way. Every name in the memo goes through `name_of`, so this one override
    covers the tables, the facts and the solver's own explanations alike.
    """

    def __init__(self, state: GameState, self_id: str) -> None:
        super().__init__(state)
        self._self_id = self_id

    def name_of(self, player_id: str) -> str:
        return SELF if player_id == self._self_id else super().name_of(player_id)

_ROLE_LABELS: dict[RoleName, str] = {
    RoleName.VILLAGER: "村人",
    RoleName.WEREWOLF: "人狼",
    RoleName.MADMAN: "狂人",
    RoleName.SEER: "占い",
    RoleName.MEDIUM: "霊媒",
    RoleName.HUNTER: "狩人",
    RoleName.FOX: "妖狐",
    RoleName.FREEMASON: "共有",
}

# The claimable roles, in the order a 17A table lists them.
_CLAIM_ROLE_ORDER = (RoleName.SEER, RoleName.MEDIUM, RoleName.HUNTER, RoleName.FREEMASON)

# The composition in the order a 17A table announces it: the opposing teams
# first, then the village's roles. Counts and labels come from ROLE_DEFINITIONS.
_COMPOSITION_ORDER = (
    RoleName.WEREWOLF,
    RoleName.MADMAN,
    RoleName.FOX,
    RoleName.SEER,
    RoleName.MEDIUM,
    RoleName.HUNTER,
    RoleName.FREEMASON,
    RoleName.VILLAGER,
)
assert set(_COMPOSITION_ORDER) == set(RoleName)

# The rules a player reasons from, in a few lines. Every clause is what
# `app/engine` actually does -- game.resolve_night / submit_night_action (from
# night 1 the wolves bite a non-wolf every night: no wolf target, no night
# without a bite), night_resolver (curse on every night including
# night 0; the attack never kills the fox; one attack and one curse per night;
# the medium reads only the day's execution; results are wolf / not wolf),
# game.submit_night_action (night 0 is divination only; the hunter may repeat a
# guard but not guard themselves), roles (the first victim is never a wolf or
# the fox), victory (who wins, counting the madman and the fox as non-wolves)
# and game (the end is checked after an execution and after a night; a tie that
# survives every runoff is a draw). Change one only together with the engine.
_RULE_LINES = (
    "- ルール: 占い・霊媒の結果は人狼かどうかだけ（狂人・妖狐は白）。"
    "霊媒が知るのは処刑された人の結果だけ。"
    "狩人は1日目夜から護衛でき、同じ人の連続護衛も可（狩人自身は守れない）。"
    "初日犠牲者は人狼でも妖狐でもない",
    "- 死体のルール: 人狼は1日目夜から毎晩、人狼以外の生存者を必ず1人襲撃する"
    "（襲撃なしの夜はない）。"
    "襲撃先が護衛先か妖狐なら死体は出ない。占い先が妖狐ならその夜に呪殺（0日目夜も）。"
    "妖狐は襲撃では死なない。夜の死体は襲撃と呪殺（どちらも1夜1人）だけなので、"
    "1夜に2つなら片方は呪殺された妖狐。0日目夜は襲撃がなく、初日犠牲者以外の死体は呪殺",
    "- 勝敗: 処刑のあとと夜明けに判定。人狼が0人になるか、人狼が人狼以外の生存者"
    "（狂人・妖狐も数える）と同数以上になった時点で試合終了。"
    "その時点で妖狐が生きていれば妖狐の勝ち。妖狐が死んでいれば、人狼0人なら村、"
    "同数以上なら人狼陣営（狂人を含む）の勝ち。"
    f"投票が決選を含め{DEFAULT_MAX_VOTE_ROUNDS}回続けて同数なら引き分け",
)

_PLAYER_ID_RE = re.compile(r"(?<![0-9A-Za-z])p(\d+)(?![0-9])")


def role_label(role: RoleName) -> str:
    """The short label the memo writes a role with: 占い, not `seer`.

    Public so the day summary every seat reads names roles the way the memo
    does, instead of leaking the enum value into Japanese text.
    """
    return _ROLE_LABELS.get(role, role.value)


def fake_claim_floor(role: RoleName, claimant_count: int) -> int:
    """How many claims on `role` must be fake: those beyond the role's seat count.

    Zero means the claims do not conflict at all. The village holds two
    freemasons, so a pair claiming it is the composition, not a counter-claim;
    "more than one claimant" is the wrong test for every role but the unique ones.
    """
    return max(0, claimant_count - ROLE_DEFINITIONS[role].count)


def render_roster(ledger: PublicFactLedger, self_id: str) -> str:
    """The one place ids appear: the lookup table for structured fields."""
    pairs = "、".join(
        f"{player.name}={player.player_id}"
        for player in ledger.players()
        if player.player_id != self_id
    )
    return f"【名簿(JSONのplayer_id用)】あなた={ledger.name_of(self_id)}({self_id})、{pairs}"


def namify(text: str, ledger: PublicFactLedger) -> str:
    """Rewrite `pN` tokens in code-generated explanations as names."""
    return _PLAYER_ID_RE.sub(
        lambda match: (
            ledger.name_of(f"p{match.group(1)}")
            if ledger.is_known(f"p{match.group(1)}")
            else match.group(0)
        ),
        text,
    )


def render_board_memo(
    state: GameState,
    player_id: str,
    *,
    observations: ObservationSet,
    public_solver: RoleSolver | None = None,
    seat_solver: RoleSolver | None = None,
    skill: Skill = Skill.EXPERT,
    clock: DerivationClock | None = None,
) -> str:
    """The full memo for one seat. Pure function of the board and the seat,
    except that `clock`, when given, records when each derived conclusion was
    first computed -- the runtime keeps it across the game so a seat's `skill`
    can decide when it is shown one (`app/ai/reasoning/skill.py`)."""
    lines: list[str] = [
        f"{MEMO_HEADING}{state.day}日目 / あなた: {PublicFactLedger(state).name_of(player_id)}"
    ]
    ledger = _SeatLedger(state, player_id)
    plain_solver = _plain_solver(public_solver)
    lines.extend(_rules_lines())
    lines.extend(_roster_lines(ledger, plain_solver))
    lines.extend(_claim_lines(state, ledger))
    lines.extend(_claimant_death_lines(ledger, observations))
    lines.extend(_verdict_lines(ledger, observations))
    lines.extend(_night_lines(ledger, observations))
    lines.extend(_grey_lines(ledger, observations))
    lines.extend(_vote_lines(ledger))
    board = _derived_board(ledger, observations, public_solver, plain_solver)
    if clock is not None:
        clock.observe(board.items, state)
    names = {pid: player.name for pid, player in state.players.items()}

    def shown(item: DerivedItem) -> bool:
        return is_visible(item, skill, state, clock, names)

    lines.extend(_logic_lines(board.visible_facts(shown)))
    lines.extend(_conditional_block(board.visible_rows(shown)))
    lines.extend(board.visible_execution(shown))
    lines.extend(_timeline_lines(ledger, observations))
    lines.extend(
        _private_lines(
            state,
            player_id,
            ledger,
            observations,
            seat_solver,
            public_keys=board.fact_keys,
            public_solver=public_solver,
        )
    )
    return "\n".join(lines)


def _plain_solver(public_solver: RoleSolver | None) -> RoleSolver | None:
    """The public solver without the derived rules: what the record settles."""
    if public_solver is None:
        return None
    return RoleSolver(
        public_solver.observations,
        public_solver.perspective,
        rules=plain_rule_modules(),
        cache=public_solver.cache,
    )


# -- public sections --


def _rules_lines() -> list[str]:
    """Composition and rules: the same for every seat and every day.

    The memo used to give only "人狼は最大3人"; a model left to recall the rest
    argued from opposite readings of the night-0 curse and hunted a fox that was
    already dead. Read from ROLE_DEFINITIONS so a composition change cannot
    leave the memo describing a game nobody is playing.
    """
    composition = "・".join(
        f"{ROLE_DEFINITIONS[role].label_ja}{ROLE_DEFINITIONS[role].count}"
        for role in _COMPOSITION_ORDER
    )
    return [f"- 配役({TOTAL_PLAYERS}人): {composition}", *_RULE_LINES]


def _roster_lines(ledger: PublicFactLedger, public_solver: RoleSolver | None) -> list[str]:
    alive = [player for player in ledger.players() if player.alive]
    dead = [player for player in ledger.players() if not player.alive]
    name = ledger.name_of
    lines = [f"- 生存({len(alive)}人): " + "、".join(name(player.player_id) for player in alive)]
    if dead:
        lines.append(
            "- 死亡: "
            + "、".join(
                f"{name(player.player_id)}({_death_label(player.death_cause, player.death_day)})"
                for player in dead
            )
        )
    lines.append(_rope_line(ledger, public_solver, len(alive)))
    return lines


def ropes_left(alive_count: int) -> int:
    """Executions left if every night costs one life: (alive - 1) // 2.

    16 alive on day 1 hang on 16, 14, ..., 4: seven times, and the seventh has
    to find the last wolf or three seats become two at the next dawn.
    """
    return max(0, (alive_count - 1) // 2)


def _rope_line(
    ledger: PublicFactLedger, public_solver: RoleSolver | None, alive_count: int
) -> str:
    """縄: executions left, and how many of them may miss.

    The inhumans the village must hang are the wolves that may still live and
    the fox while it may live -- the fox dies to the seer as well, but nobody
    can count on it. The old `alive - 7` formula said nine misses on day 1 of
    a game that allows three. At night the count is for the morning after the
    bite, since the wolves bite every night.

    Read from the record (the plain solver): late in a game it counts every
    wolf that may live and so reads 0 spare. What one claimant's account would
    leave to hang is a conditional fact, and goes with that claimant's row
    (`_conditional_rope_clauses`).
    """
    dead_wolves, fox_may_live = _rope_inhumans(ledger, public_solver)
    wolves = ROLE_DEFINITIONS[RoleName.WEREWOLF].count - len(dead_wolves)
    ropes, to_hang = _rope_counts(ledger, public_solver)
    if dead_wolves:
        head = f"- 生存の人狼は最大{wolves}人（{_names(ledger, dead_wolves)}は人狼で死亡確定）"
    else:
        head = f"- 人狼は最大{wolves}人（公開情報では何人死んだか確定しない）"
    if ledger.phase in (Phase.NIGHT, Phase.VOTE_RESULT):
        count = f"今夜1人死ねば明日は生存{alive_count - 1}人で残り{ropes}回"
    else:
        count = f"生存{alive_count}人で残り{ropes}回"
    breakdown = f"人狼{wolves}" + ("・妖狐1" if fox_may_live else "。妖狐は死亡済み")
    return (
        f"{head} / 縄: {count}（(生存-1)÷2、毎晩1人死ぬ場合）。"
        f"吊るべき人外は最大{to_hang}人（{breakdown}）→ 外してよい吊りは{max(0, ropes - to_hang)}回"
    )


def _rope_inhumans(
    ledger: PublicFactLedger, solver: RoleSolver | None
) -> tuple[list[str], bool]:
    """The dead seats certain to be wolves, and whether the fox may live."""
    if solver is None or not solver.is_possible(Hypothesis()):
        return [], True
    dead_wolves = [
        pid for pid in ledger.dead_ids() if solver.certain_role(pid) is RoleName.WEREWOLF
    ]
    return dead_wolves, _anyone_alive_could_be(ledger, solver, RoleName.FOX)


def _rope_counts(ledger: PublicFactLedger, solver: RoleSolver | None) -> tuple[int, int]:
    """(executions left, inhumans the solver says may still need hanging).

    At night, and between the execution and the night, the count is for the
    morning after tonight's bite."""
    alive_count = len(ledger.alive_ids())
    if ledger.phase in (Phase.NIGHT, Phase.VOTE_RESULT):
        alive_count -= 1
    dead_wolves, fox_may_live = _rope_inhumans(ledger, solver)
    wolves = ROLE_DEFINITIONS[RoleName.WEREWOLF].count - len(dead_wolves)
    return ropes_left(alive_count), wolves + (1 if fox_may_live else 0)


def _anyone_alive_could_be(ledger: PublicFactLedger, solver: RoleSolver, role: RoleName) -> bool:
    alive = ledger.alive_ids()
    if not alive:
        return False
    return solver.is_possible(
        Hypothesis(
            claims=(AnyOf(tuple(RoleIs(player_id=pid, role=role) for pid in alive)),),
            label=f"a living {role.value}",
        )
    )


def _death_label(cause: PublicDeathCause | None, day: int | None) -> str:
    if cause is PublicDeathCause.FIRST_VICTIM:
        return "初日犠牲者"
    if cause is PublicDeathCause.EXECUTED:
        return f"{day}日目処刑"
    if cause is PublicDeathCause.NIGHT:
        return f"{day}日目夜死亡"
    return "死亡"


def _claim_lines(state: GameState, ledger: PublicFactLedger) -> list[str]:
    by_role: dict[RoleName, list[str]] = defaultdict(list)
    for claim in ledger.co_declarations():
        by_role[claim.claimed_role].append(claim.player_id)
    if not by_role:
        return ["- CO: まだ誰もCOしていない"]
    parts: list[str] = []
    for role in _CLAIM_ROLE_ORDER:
        claimants = by_role.get(role)
        if not claimants:
            if role is RoleName.MEDIUM:
                # A latent medium is a choice the table should see it making.
                parts.append(f"{_ROLE_LABELS[role]}=まだなし")
            continue
        rendered = "、".join(
            f"{ledger.name_of(pid)}{'' if ledger.is_alive(pid) else '(死亡)'}" for pid in claimants
        )
        capacity = ROLE_DEFINITIONS[role].count
        note = ""
        fakes = fake_claim_floor(role, len(claimants))
        if fakes:
            note = f"（{_ROLE_LABELS[role]}は{capacity}人なので少なくとも{fakes}人は偽）"
        elif role is RoleName.FREEMASON:
            note = _freemason_note(state, ledger, claimants)
        parts.append(f"{_ROLE_LABELS[role]}={rendered}{note}")
    for role, claimants in by_role.items():
        if role in _CLAIM_ROLE_ORDER:
            continue
        rendered = "、".join(ledger.name_of(pid) for pid in claimants)
        parts.append(f"{_ROLE_LABELS.get(role, role.value)}={rendered}")
    return ["- CO: " + " / ".join(parts), *_co_count_lines(ledger, by_role)]


def _co_count_lines(ledger: PublicFactLedger, by_role: dict[RoleName, list[str]]) -> list[str]:
    """占い・霊媒 counts, zeros included, and the "3-0" a table calls them by.

    The review's day-1 table said "3-1" before any medium had claimed (1-B1):
    a role nobody claimed was simply missing from the CO line.
    """
    seers = by_role.get(RoleName.SEER, [])
    mediums = by_role.get(RoleName.MEDIUM, [])
    if not seers and not mediums:
        return []

    def count(role: RoleName, claimants: list[str]) -> str:
        if not claimants:
            return f"{_ROLE_LABELS[role]}0"
        living = sum(1 for pid in claimants if ledger.is_alive(pid))
        return f"{_ROLE_LABELS[role]}{len(claimants)}(生存{living})"

    return [
        f"- CO数: {count(RoleName.SEER, seers)}・{count(RoleName.MEDIUM, mediums)}"
        f" → 内訳の呼び方は{len(seers)}-{len(mediums)}"
    ]


def _freemason_note(state: GameState, ledger: PublicFactLedger, claimants: list[str]) -> str:
    partner_claims = state.freemason_partner_claims
    named = set(claimants) | {
        claim.partner_id for claim in partner_claims if claim.claimant_id in claimants
    }
    # Both freemason seats have a name on the table: the "is a day-1 black the
    # freemason trap" question is closed (1-C5). Three names is a contest.
    both_out = "。共有は2人とも名前が出ている" if len(named) == 2 else ""
    confirmed = [claim for claim in partner_claims if claim.confirmed]
    if confirmed:
        pair = confirmed[0]
        return (
            f"（{ledger.name_of(pair.claimant_id)}と{ledger.name_of(pair.partner_id)}が"
            f"互いに相方と確認済み{both_out}）"
        )
    pending = [claim for claim in partner_claims if not claim.confirmed]
    if pending:
        claim = pending[0]
        return (
            f"（{ledger.name_of(claim.claimant_id)}は相方を{ledger.name_of(claim.partner_id)}と"
            f"名指ししたが本人の確認待ち{both_out}）"
        )
    if len(claimants) == 1:
        return "（相方は未公開）"
    return f"（{both_out.removeprefix('。')}）" if both_out else ""


def _claimant_death_lines(ledger: PublicFactLedger, observations: ObservationSet) -> list[str]:
    """When each role claimant died, and which have never died at night.

    A record, not a reading: run #14 seed 2's village took "only the rival
    claimant was bitten" as a sign the survivor was real, and the owner ruled
    that no convention (the survivor may be the madman). So the line gives
    the pattern and stops -- a night death is an attack or a curse, and that
    is all it says. Printed once some claimant has died at night.
    """
    night_of = {death.player_id: death.night for death in observations.night_deaths}
    by_role: dict[RoleName, list[str]] = defaultdict(list)
    for claim in ledger.co_declarations():
        if claim.claimed_role in _CLAIM_ROLE_ORDER and claim.player_id not in by_role[
            claim.claimed_role
        ]:
            by_role[claim.claimed_role].append(claim.player_id)
    claimants = {pid for pids in by_role.values() for pid in pids}
    if not claimants & set(night_of):
        return []

    def status(pid: str) -> str:
        if pid in night_of:
            return f"{night_of[pid]}日目夜に死亡"
        if ledger.is_alive(pid):
            return "夜の死亡なし(生存)"
        day = ledger.execution_day(pid)
        return f"{day}日目処刑(夜の死亡なし)" if day is not None else "死亡"

    parts = [
        f"{_ROLE_LABELS[role]}CO "
        + "、".join(f"{ledger.name_of(pid)}={status(pid)}" for pid in pids)
        for role in _CLAIM_ROLE_ORDER
        if (pids := by_role.get(role))
    ]
    return ["- 役職COの夜の死亡（夜の死体は襲撃か呪殺）: " + " / ".join(parts)]


def _verdict_lines(ledger: PublicFactLedger, observations: ObservationSet) -> list[str]:
    """Seer accounts night by night, then the medium's read of each execution."""
    lines = _divination_lines(ledger, observations)
    if not observations.verdicts:
        lines.append("- 公開された判定: なし")
    return lines + _medium_lines(ledger, observations)


def _seer_claimants(ledger: PublicFactLedger) -> list[str]:
    """Everyone standing on a seer claim or who has published a seer verdict,
    in the order they first spoke as one."""
    standing = {
        claim.player_id
        for claim in ledger.co_declarations()
        if claim.claimed_role is RoleName.SEER
    }
    publishers = {
        result.claimant_id
        for result in ledger.public_results()
        if result.result_type == SEER_RESULT
    }
    wanted = standing | publishers
    order: list[str] = []
    for event in ledger.speech_events():
        if event.role is RoleName.SEER and event.actor_id in wanted and event.actor_id not in order:
            order.append(event.actor_id)
    order.extend(_seat_ordered_ids(ledger, wanted - set(order)))
    return order


def _seat_ordered_ids(ledger: PublicFactLedger, player_ids: Iterable[str]) -> list[str]:
    """Seating order -- never the order deaths were recorded.

    The engine applies the curse before the bite, so "the first corpse listed"
    would tell every seat which one was the fox.
    """
    wanted = set(player_ids)
    return [pid for pid in ledger.known_player_ids() if pid in wanted]


def _names(ledger: PublicFactLedger, player_ids: Iterable[str]) -> str:
    return "、".join(ledger.name_of(pid) for pid in _seat_ordered_ids(ledger, player_ids))


def _night_header(ledger: PublicFactLedger, observations: ObservationSet, night: int) -> str:
    if night >= observations.day:
        return ""
    corpses = _names(ledger, (death.player_id for death in observations.deaths_on(night)))
    if night == 0 and observations.first_victim_id is not None:
        parts = [f"初日犠牲者: {ledger.name_of(observations.first_victim_id)}"]
        if corpses:
            parts.append(f"死体: {corpses}")
        return f"（{' / '.join(parts)}）"
    return f"（死体: {corpses}）" if corpses else "（死体なし）"


def _divination_lines(ledger: PublicFactLedger, observations: ObservationSet) -> list[str]:
    """Every seer claimant's account, one row per night, next to its corpses.

    A row per night rather than per claimant, because the question a table asks
    is "whose target fits this morning's corpses". The day each verdict was
    published is shown -- a result first said after the corpses were seen reads
    differently from one said before -- and a night a claimant could have
    divined but never published is written out: the first live game's village
    concluded "only two seats were divined on night 1" by forgetting a dead
    seer's unpublished night. A night a claimant could not act (already dead or
    hanged) has no cell.

    Verdicts are read from the observations, not the raw ledger: the projection
    dates a restated verdict by the statement it repeats, so mentioning
    yesterday's result again does not move it into last night's row.
    """
    claimants = _seer_claimants(ledger)
    if not claimants:
        return []
    published: dict[tuple[str, int], list[PublicVerdict]] = defaultdict(list)
    for result in observations.verdicts:
        if result.result_type == SEER_RESULT:
            published[(result.claimant_id, result.source_night)].append(result)
    nights = sorted(set(range(max(0, observations.day))) | {night for _, night in published})
    rows: list[str] = []
    for night in nights:
        cells: list[str] = []
        for claimant in claimants:
            results = sorted(published.get((claimant, night), []), key=lambda r: r.day)
            cell = _divination_cell(ledger, observations, claimant, night, results)
            if cell is not None:
                cells.append(cell)
        if cells:
            header = _night_header(ledger, observations, night)
            grey = _grey_cell(ledger, observations, night)
            rows.append(f"  - {night}日目夜{header}{grey}: " + " / ".join(cells))
    if not rows:
        return []
    return [
        "- 占いCOの夜ごとの占い先（[ ]は公開日。未公開の夜も書く。〔 〕はその夜の時点のグレー）:",
        *rows,
    ]


def _grey_on_night(
    ledger: PublicFactLedger, observations: ObservationSet, night: int
) -> list[str]:
    """The grey as it stood when night `night`'s divinations were chosen.

    Alive that night (not hanged by day `night`, not dead on an earlier
    night), no CO made by day `night`, and no seer verdict on them published
    by then -- the grey line's definition, read at that moment. Run #14's
    fake seers never divined the seat the table suspected, and one divined a
    confirmed freemason with one grey left; neither can be read from a target
    without the grey beside it (X4).
    """
    gone = {
        execution.player_id for execution in observations.executions if execution.day <= night
    }
    gone |= {death.player_id for death in observations.night_deaths if death.night < night}
    if observations.first_victim_id is not None:
        gone.add(observations.first_victim_id)
    claimed = {claim.player_id for claim in ledger.co_declarations() if claim.day <= night}
    judged = {
        verdict.target_id
        for verdict in observations.verdicts
        if verdict.result_type == SEER_RESULT and verdict.day <= night
    }
    return [
        pid
        for pid in ledger.known_player_ids()
        if pid not in gone and pid not in claimed and pid not in judged
    ]


def _grey_cell(ledger: PublicFactLedger, observations: ObservationSet, night: int) -> str:
    """〔その夜のグレー〕 for a resolved night from night 1; night 0 is everyone."""
    if night < 1 or night >= observations.day:
        return ""
    grey = _grey_on_night(ledger, observations, night)
    return f"〔{_names(ledger, grey) or 'なし'}〕"


def _divination_cell(
    ledger: PublicFactLedger,
    observations: ObservationSet,
    claimant: str,
    night: int,
    results: list[PublicVerdict],
) -> str | None:
    name = ledger.name_of(claimant)
    if results:
        return f"{name}→" + "、".join(
            f"{ledger.name_of(r.target_id)}={'黒' if r.is_werewolf else '白'}[{r.day}日目公開]"
            for r in results
        )
    if night >= observations.day or not can_act_on_night(observations, claimant, night):
        return None
    if not could_publish_on_day(observations, claimant, night + 1):
        return f"{name}→未公開({night}日目夜に死亡)"
    return f"{name}→未公開"


def _medium_lines(ledger: PublicFactLedger, observations: ObservationSet) -> list[str]:
    """One cell per execution: the colour each medium claimant published, or
    that nothing was published -- and, once no medium claimant is alive, that
    nothing will be unless somebody new claims it.

    By execution rather than by claimant, because the question is "what do we
    know of the people we hanged". The review's day-5 village kept reasoning
    from "the medium will check it" two days after the medium was bitten; the
    old section listed published results only, so a missing one was silent.
    """
    results = [r for r in observations.verdicts if r.result_type == MEDIUM_RESULT]
    executions = ledger.executions()
    if not executions and not results:
        return []
    by_target: dict[str, list[PublicVerdict]] = defaultdict(list)
    for result in sorted(results, key=lambda r: (r.day, r.source_night)):
        by_target[result.target_id].append(result)

    def published(target_id: str) -> str:
        return "、".join(
            f"{'黒' if r.is_werewolf else '白'}({ledger.name_of(r.claimant_id)}・{r.day}日目公開)"
            for r in by_target.pop(target_id)
        )

    cells = [
        f"{execution.day}日目処刑 {ledger.name_of(execution.player_id)}="
        + (published(execution.player_id) if execution.player_id in by_target else "未公開")
        for execution in executions
    ]
    # A medium verdict on someone never hanged is a claim like any other.
    cells.extend(
        f"処刑されていない{ledger.name_of(target)}={published(target)}"
        for target in _seat_ordered_ids(ledger, list(by_target))
    )
    line = "- 処刑者の霊媒結果: " + " / ".join(cells)
    living_mediums = [
        claim.player_id
        for claim in ledger.co_declarations()
        if claim.claimed_role is RoleName.MEDIUM and ledger.is_alive(claim.player_id)
    ]
    if executions and not living_mediums:
        line += "。生存する霊媒COなし: 新たに霊媒COが出ない限り、処刑者の結果は公開されない"
    return [line]


def _night_lines(ledger: PublicFactLedger, observations: ObservationSet) -> list[str]:
    """Every resolved night from night 1: its corpses, or none, and what none means.

    The only record of an empty morning used to be a "(死体なし)" in the seer
    table's row headers -- read by nobody in the seed-1 review, and absent
    altogether on a night no seer claimant could act. With the engine making
    the wolves bite a non-wolf every night, an empty morning is logic, not a
    guess: the bite hit the guard target or the fox.
    """
    nights = range(1, observations.day)
    if not nights:
        return []
    cells: list[str] = []
    for night in nights:
        corpses = [death.player_id for death in observations.deaths_on(night)]
        found = f"死体: {_names(ledger, corpses)}" if corpses else "死体なし"
        cells.append(f"{night}日目夜 {found}")
    line = "- 夜の結果: " + " / ".join(cells)
    if any(not observations.deaths_on(night) for night in nights):
        line += "（死体なし＝襲撃が護衛で防がれたか、襲撃先が妖狐）"
    return [line]


def _grey_lines(ledger: PublicFactLedger, observations: ObservationSet) -> list[str]:
    """The grey -- alive, no claim, judged by no seer claimant -- and, apart
    from it, the unclaimed seats somebody has judged, with whose verdict.

    The review's day 2 argued over "the grey" with two definitions in play and
    no list on the board (2-B5). One definition, computed: a seat stays grey
    until some seer claimant publishes a verdict on it, whoever that is.
    """
    claimed = {claim.player_id for claim in ledger.co_declarations()}
    judged: dict[str, list[str]] = defaultdict(list)
    for verdict in sorted(observations.verdicts, key=lambda v: (v.day, v.source_night)):
        if verdict.result_type == SEER_RESULT:
            colour = "黒" if verdict.is_werewolf else "白"
            judged[verdict.target_id].append(f"{ledger.name_of(verdict.claimant_id)}の{colour}")
    unclaimed = [pid for pid in ledger.alive_ids() if pid not in claimed]
    if not claimed and not judged:
        return []
    grey = [pid for pid in unclaimed if pid not in judged]
    line = f"- グレー(生存・CO無し・どの占いCOの判定も無し): {_names(ledger, grey) or 'なし'}"
    seen = [pid for pid in unclaimed if pid in judged]
    if seen:
        line += " / 判定を受けた非CO者: " + "、".join(
            f"{ledger.name_of(pid)}({'・'.join(judged[pid])})" for pid in seen
        )
    return [line]


def _vote_lines(ledger: PublicFactLedger) -> list[str]:
    """Ballots read both ways: per target, then per living voter.

    「ケント←ハルト、カイト、ダイキ」 was read backwards in the review (4-B2), and
    a seat that lied about its own day-3 ballot (m61) was never caught because
    nobody's history sat in one place. (処刑) marks the seat hanged that day,
    as a target and as a voter.
    """
    votes = ledger.votes()
    if not votes:
        return []
    executed_on = {execution.day: execution.player_id for execution in ledger.executions()}
    seat = {pid: index for index, pid in enumerate(ledger.known_player_ids())}
    lines = ["- 投票（N日目: 投票先へ票数: 投票した人。(処刑)はその日に処刑された人）:"]
    rounds = sorted({(vote.day, vote.round) for vote in votes})
    last_round = {day: round_number for day, round_number in rounds}
    for day, round_number in rounds:
        hanged = executed_on.get(day)

        def label(pid: str, *, hanged: str | None = hanged) -> str:
            return ledger.name_of(pid) + ("(処刑)" if pid == hanged else "")

        by_target: dict[str, list[str]] = defaultdict(list)
        for vote in ledger.votes_on(day, round_number):
            by_target[vote.target_id].append(vote.voter_id)
        ordered = sorted(by_target.items(), key=lambda item: (-len(item[1]), seat[item[0]]))
        cells = []
        for target, voters in ordered:
            mark = "(処刑)" if target == hanged and round_number == last_round[day] else ""
            names = "、".join(label(pid) for pid in sorted(voters, key=seat.__getitem__))
            cells.append(f"{ledger.name_of(target)}へ{len(voters)}票{mark}: {names}")
        lines.append(f"  - {day}日目{_runoff_label(round_number)}: " + " / ".join(cells))
    lines.append("- 生存者ごとの投票先:")
    for voter in ledger.alive_ids():
        ballots: dict[int, list[str]] = defaultdict(list)
        for vote in sorted(votes, key=lambda v: (v.day, v.round)):
            if vote.voter_id == voter:
                ballots[vote.day].append(
                    _runoff_label(vote.round) + ledger.name_of(vote.target_id)
                )
        if ballots:
            lines.append(
                f"  - {ledger.name_of(voter)}: "
                + " / ".join(f"{day}日目" + "→".join(ballots[day]) for day in sorted(ballots))
            )
    if lines[-1] == "- 生存者ごとの投票先:":
        lines.pop()
    return lines


def _runoff_label(round_number: int) -> str:
    """'' for the first ballot of a day, 決選 for the runoff, 決選2... after."""
    if round_number <= 1:
        return ""
    return "決選" if round_number == 2 else f"決選{round_number - 1}"


@dataclass(frozen=True)
class _Fact:
    """One settled conclusion, with a key naming what it settles.

    The key is what lets a seat's private tail drop a conclusion the public
    section already states -- comparing the wording would break the first time
    the two sections phrase the same fact differently.
    """

    key: tuple[str, ...]
    text: str
    # Facts of the form "<seat> is not X" share a predicate, and a run of them
    # is written once: 「リク、アオイは人狼でも妖狐でもない」.
    predicate: str = ""
    # The seats the fact is about: a table statement naming one of them makes
    # a derived fact count as stated (`skill.stated_at_table`), and a derived
    # fact about a seat replaces the record's weaker fact about it.
    seats: tuple[str, ...] = ()


def _fact_texts(facts: Iterable[_Fact]) -> list[str]:
    """Each fact's text, with seats sharing a predicate merged at the first."""
    texts: list[str] = []
    merged: dict[str, int] = {}
    subjects: dict[str, list[str]] = defaultdict(list)
    for fact in facts:
        if not fact.predicate:
            texts.append(fact.text)
            continue
        if fact.predicate not in merged:
            merged[fact.predicate] = len(texts)
            texts.append("")
        subjects[fact.predicate].append(fact.text)
    for predicate, index in merged.items():
        texts[index] = "、".join(subjects[predicate]) + predicate
    return texts


def _public_logic_facts(
    ledger: PublicFactLedger,
    observations: ObservationSet,
    public_solver: RoleSolver | None,
) -> list[_Fact]:
    """What the rules settle from public information alone."""
    facts: list[_Fact] = []
    if observations.first_victim_id is not None:
        facts.append(
            _Fact(
                ("first_victim",),
                f"初日犠牲者の{ledger.name_of(observations.first_victim_id)}は人狼でも妖狐でもない",
            )
        )
    stated: set[str] = set()
    for night, corpses in sorted(curse_nights(observations).items()):
        names = _names(ledger, corpses)
        if night == 0 and len(corpses) == 1:
            stated.update(corpses)
            facts.append(
                _Fact(
                    ("role", corpses[0]),
                    f"0日目夜は襲撃がないので、初日犠牲者以外の死体の{names}は呪殺された妖狐",
                )
            )
        elif night > 0 and len(corpses) == 2:
            facts.append(
                _Fact(
                    ("curse_night", str(night)),
                    f"{night}日目夜は死体が2つ（{names}）。襲撃は一晩1件、"
                    "呪殺も1件なので、このうち1人は占われて死んだ妖狐",
                )
            )
    # Every night corpse, claimants included: the solver always knew it
    # (`DeathRuleModule`), but the section listed living seats only, so the
    # table read "the first victim" and nothing else for six days (run #14, X2).
    night_dead = [d.player_id for d in observations.night_deaths if d.night > 0]
    if night_dead:
        facts.append(
            _Fact(
                ("night_dead_not_wolf",),
                f"夜に死んだ{_names(ledger, night_dead)}は人狼ではない"
                "（夜の死因は人狼以外への襲撃か、妖狐の呪殺だけ）",
                seats=tuple(night_dead),
            )
        )
    if public_solver is not None:
        facts.extend(
            _settled_facts(
                ledger,
                observations,
                public_solver,
                exclude=frozenset(stated),
                wolf_view=False,
                refute_claims=True,
                viewer_id=None,
            )
        )
    return facts


def _logic_lines(facts: list[_Fact]) -> list[str]:
    if not facts:
        return []
    return ["- 公開情報だけで確定すること:"] + [f"  - {text}" for text in _fact_texts(facts)]


def _settled_facts(
    ledger: PublicFactLedger,
    observations: ObservationSet,
    solver: RoleSolver,
    *,
    exclude: frozenset[str],
    wolf_view: bool,
    refute_claims: bool,
    viewer_id: str | None,
    fox_view: bool = False,
) -> list[_Fact]:
    """Hard conclusions a solver reaches about claimants, roles and seats.

    Shared by the public section and a seat's private tail, which differ only
    in the solver and in what they leave out. A board that admits no world at
    all settles nothing: read naively, "nobody can be the fox" would be true of
    every seat and printed as a fact.

    A living seat that cannot be the wolf, or the fox, or either, is cleared in
    those words. The fox is never told who is not the fox -- that is everyone
    -- as the wolves are never told who is not a wolf; and once no living seat
    can be the fox, the filled-role line already says so for all of them.
    """
    if not solver.is_possible(Hypothesis()):
        return []
    facts: list[_Fact] = []
    if refute_claims:
        facts.extend(_refuted_claim_facts(ledger, observations, solver, exclude, viewer_id))
    filled = _filled_role_facts(ledger, solver)
    facts.extend(filled)
    fox_settled = ("filled", RoleName.FOX.value) in {fact.key for fact in filled}
    for pid in ledger.known_player_ids():
        if pid in exclude:
            continue
        alive = ledger.is_alive(pid)
        role = solver.certain_role(pid)
        if role is not None and not (wolf_view and role is RoleName.WEREWOLF):
            label = ledger.name_of(pid) + ("" if alive else "(死亡)")
            facts.append(
                _Fact(("role", pid), f"{label}は{_ROLE_LABELS[role]}で確定", seats=(pid,))
            )
            continue
        if not alive or role is not None:
            continue
        not_wolf = not wolf_view and not solver.is_possible(has_role(pid, RoleName.WEREWOLF))
        not_fox = (
            not fox_view
            and not fox_settled
            and not solver.is_possible(has_role(pid, RoleName.FOX))
        )
        if not_wolf or not_fox:
            cleared = "wolf+fox" if not_wolf and not_fox else ("wolf" if not_wolf else "fox")
            predicate = {
                "wolf+fox": "は人狼でも妖狐でもない",
                "wolf": "は人狼ではない",
                "fox": "は妖狐ではない",
            }[cleared]
            facts.append(
                _Fact(("cleared", pid, cleared), ledger.name_of(pid), predicate, seats=(pid,))
            )
    return facts


def _refuted_claim_facts(
    ledger: PublicFactLedger,
    observations: ObservationSet,
    solver: RoleSolver,
    exclude: frozenset[str],
    viewer_id: str | None,
) -> list[_Fact]:
    """Seer claimants who cannot be the real seer, each with the reason.

    The solver decides; `seer_claim_refutations` supplies the wording for the
    one-step case (a curse night, and a published target who did not die on
    it). Anything the solver reaches some other way is explained from its own
    unsat core, so no line ever states a conclusion without a reason behind it.
    """
    direct: dict[str, SeerClaimRefutation] = {}
    for refutation in seer_claim_refutations(observations, viewer_id=viewer_id):
        direct.setdefault(refutation.claimant_id, refutation)
    facts: list[_Fact] = []
    for claimant in _seer_claimants(ledger):
        if claimant in exclude or claimant == viewer_id:
            continue
        if solver.is_possible(has_role(claimant, RoleName.SEER)):
            continue
        facts.append(
            _Fact(
                ("not_seer", claimant),
                _refutation_text(ledger, solver, claimant, direct.get(claimant)),
                seats=(claimant,),
            )
        )
    return facts


def _refutation_text(
    ledger: PublicFactLedger,
    solver: RoleSolver,
    claimant: str,
    refutation: SeerClaimRefutation | None,
) -> str:
    name = ledger.name_of(claimant)
    head = f"{name}は真の占い師ではありえない"
    if refutation is not None:
        night = refutation.night
        target = ledger.name_of(refutation.target_id)
        corpses = _names(ledger, refutation.corpse_ids)
        if night == 0:
            why = (
                f"0日目夜は襲撃がなく{corpses}の死は呪殺なので、真の占い師の0日目夜の占い先は"
                f"{corpses}。{name}が公開した0日目夜の占い先{target}はその夜に死んでいない"
            )
        else:
            why = (
                f"{night}日目夜は死体{len(refutation.corpse_ids)}つ（{corpses}）で片方は呪殺なので、"
                f"真の占い師のその夜の占い先は死体のどちらか。"
                f"{name}が公開したその夜の占い先{target}は死んでいない"
            )
        return f"{head}（{why}）"
    core = solver.explain_contradiction(has_role(claimant, RoleName.SEER))
    module_prefix = f"{ClaimedDivinationRuleModule.module_id}:"
    reasons = [
        namify(text, ledger)
        for cid, text in zip(core.constraint_ids, core.explanations, strict=True)
        if cid.startswith(module_prefix)
    ]
    return f"{head}（{reasons[0]}）" if reasons else head


def _filled_role_facts(ledger: PublicFactLedger, solver: RoleSolver) -> list[_Fact]:
    """Roles whose every seat is already in the graveyard, said plainly.

    17A has one fox. Once the fox can only be among the dead, "no living fox"
    is as settled as a confirmed role -- and the first live game spent three
    days "caring" for a fox that died on night 1 because the memo never said so.
    """
    alive = ledger.alive_ids()
    if not alive:
        return []
    dead = ledger.dead_ids()
    facts: list[_Fact] = []
    for role in _COMPOSITION_ORDER:
        anyone_alive = Hypothesis(
            claims=(AnyOf(tuple(RoleIs(player_id=pid, role=role) for pid in alive)),),
            label=f"a living {role.value}",
        )
        if solver.is_possible(anyone_alive):
            continue
        definition = ROLE_DEFINITIONS[role]
        label = definition.label_ja
        holders = [pid for pid in dead if solver.is_possible(has_role(pid, role))]
        if definition.count == 1:
            text = f"{label}(1人)は死亡済み。生存者に{label}はいない{_candidates(ledger, holders)}"
        else:
            text = f"{label}({definition.count}人)は全員死亡済み。生存者に{label}はいない"
        facts.append(_Fact(("filled", role.value), text, seats=tuple(holders)))
    return facts


def _candidates(ledger: PublicFactLedger, player_ids: list[str]) -> str:
    names = [ledger.name_of(pid) for pid in _seat_ordered_ids(ledger, player_ids)]
    if not names:
        return ""
    if len(names) == 1:
        return f"（{names[0]}）"
    if len(names) == 2:
        return f"（{names[0]}か{names[1]}のどちらか）"
    return f"（{'、'.join(names)}のいずれか）"


@dataclass(frozen=True)
class _Endgame:
    """What one solver settles about how the game can end.

    The number of living wolves is counted exactly: the smallest and largest
    count some world allows (`RoleCountAmong` over the living), which also
    reads the game still running (`GameRunningRuleModule`). A clause below is
    printed only when the bound alone makes it true in every world the solver
    allows.
    """

    alive: tuple[str, ...]
    dead_wolves: tuple[str, ...]
    living_wolves: tuple[str, ...]
    wolves_min: int
    wolves_max: int
    fox_alive: tuple[str, ...]
    fox_dead: tuple[str, ...]

    @property
    def wolves_exact(self) -> bool:
        return self.wolves_min == self.wolves_max == len(self.living_wolves)

    @property
    def fox_dead_for_sure(self) -> bool:
        return not self.fox_alive

    @property
    def fox_alive_for_sure(self) -> bool:
        return bool(self.fox_alive) and not self.fox_dead


def _living_wolf_counts(solver: RoleSolver, alive: tuple[str, ...]) -> list[int]:
    """Every number of living wolves some world allows."""
    wolf = RoleName.WEREWOLF
    return [
        count
        for count in range(ROLE_DEFINITIONS[wolf].count + 1)
        if solver.is_possible(
            Hypothesis(
                claims=(RoleCountAmong(alive, wolf, at_least=count, at_most=count),),
                label=f"{count} living wolves",
            )
        )
    ]


def _endgame(ledger: PublicFactLedger, solver: RoleSolver) -> _Endgame | None:
    if not solver.is_possible(Hypothesis()):
        return None
    alive = ledger.alive_ids()
    dead = ledger.dead_ids()
    wolf, fox = RoleName.WEREWOLF, RoleName.FOX
    maybe_dead_wolves = [pid for pid in dead if solver.is_possible(has_role(pid, wolf))]
    maybe_living_wolves = [pid for pid in alive if solver.is_possible(has_role(pid, wolf))]
    dead_wolves = tuple(p for p in maybe_dead_wolves if solver.is_forced(has_role(p, wolf)))
    living_wolves = tuple(p for p in maybe_living_wolves if solver.is_forced(has_role(p, wolf)))
    counts = _living_wolf_counts(solver, alive) or [len(living_wolves)]
    return _Endgame(
        alive=alive,
        dead_wolves=dead_wolves,
        living_wolves=living_wolves,
        wolves_min=min(counts),
        wolves_max=max(counts),
        fox_alive=tuple(pid for pid in alive if solver.is_possible(has_role(pid, fox))),
        fox_dead=tuple(pid for pid in dead if solver.is_possible(has_role(pid, fox))),
    )


@dataclass(frozen=True)
class _Clause:
    """One clause of an endgame summary. `kind` and `seats` name what it
    settles, so a seat's skill can be applied clause by clause: a conditional
    line whose fox candidates shrink is the same conclusion, while "ツムギ is
    the only living wolf" is a new one."""

    kind: str
    text: str
    seats: tuple[str, ...] = ()


def _wolf_clauses(ledger: PublicFactLedger, end: _Endgame) -> list[_Clause]:
    """Who the wolves are and what execution ends the game, as far as settled.

    "Hanging X ends it" is printed only when every world agrees: X is the one
    living wolf left. "Hanging any non-wolf ends it" only when even the fewest
    living wolves would then stand level with the rest.
    """
    clauses: list[_Clause] = []
    if end.dead_wolves:
        clauses.append(
            _Clause(
                "dead_wolves", f"死亡した人狼は{_names(ledger, end.dead_wolves)}", end.dead_wolves
            )
        )
    count = (
        f"{end.wolves_min}"
        if end.wolves_min == end.wolves_max
        else f"{end.wolves_min}〜{end.wolves_max}"
    )
    if end.living_wolves and end.wolves_exact:
        clauses.append(
            _Clause(
                "living_wolves_exact",
                f"生存の人狼は{_names(ledger, end.living_wolves)}だけ",
                end.living_wolves,
            )
        )
    elif end.living_wolves:
        clauses.append(
            _Clause(
                "living_wolves_some",
                f"生存の人狼に{_names(ledger, end.living_wolves)}を含む"
                f"（生存の人狼は全部で{count}人）",
                end.living_wolves,
            )
        )
    elif end.dead_wolves and end.wolves_min == end.wolves_max:
        clauses.append(_Clause("living_wolf_count", f"生存の人狼は{count}人"))
    if end.wolves_exact and len(end.living_wolves) == 1:
        (last,) = end.living_wolves
        clauses.append(
            _Clause(
                "ends_by_hanging",
                f"{ledger.name_of(last)}を処刑すると人狼0人で試合が終わる"
                f"（{_winner_if_wolves_gone(end)}）",
                (last,),
            )
        )
    clauses.extend(_parity_clauses(ledger, end))
    return clauses


def _parity_clauses(ledger: PublicFactLedger, end: _Endgame) -> list[_Clause]:
    """When even the fewest living wolves would stand level with the rest.

    By day the next event is an execution, then the bite; by night (and
    between the execution and the night) the bite comes first. Each wording
    is printed only when it holds with `wolves_min`, so in every world.
    """
    wolves, alive = end.wolves_min, len(end.alive)
    if not wolves:
        return []
    winner = f"（{_winner_at_parity(end)}）"
    if ledger.phase in (Phase.NIGHT, Phase.VOTE_RESULT):
        if 2 * wolves >= alive - 1:
            text = "今夜の襲撃が通ると人狼が同数になり試合が終わる"
        elif 2 * wolves >= alive - 2:
            text = "今夜の襲撃が通り、次の処刑が人狼以外なら人狼が同数になり試合が終わる"
        else:
            return []
    elif 2 * wolves >= alive - 1:
        text = "人狼以外を処刑すると人狼が同数になり試合が終わる"
    elif 2 * wolves >= alive - 2:
        text = "人狼以外を処刑すると、その夜の襲撃が通れば人狼が同数になり試合が終わる"
    else:
        return []
    return [_Clause("parity", text + winner)]


def _winner_if_wolves_gone(end: _Endgame) -> str:
    if end.fox_dead_for_sure:
        return "妖狐は死亡済みなので村の勝ち"
    if end.fox_alive_for_sure:
        return "妖狐が生きているので妖狐の勝ち"
    return "妖狐が生きていれば妖狐の勝ち、死んでいれば村の勝ち"


def _winner_at_parity(end: _Endgame) -> str:
    if end.fox_dead_for_sure:
        return "妖狐は死亡済みなので人狼陣営の勝ち"
    if end.fox_alive_for_sure:
        return "妖狐が生きているので妖狐の勝ち"
    return "妖狐が生きていれば妖狐の勝ち、死んでいれば人狼陣営の勝ち"


def _fox_clause(ledger: PublicFactLedger, end: _Endgame) -> _Clause:
    if end.fox_dead_for_sure:
        return _Clause("fox_dead", "妖狐は死亡済み")
    if end.fox_alive_for_sure:
        return _Clause("fox_alive", "妖狐は生存")
    return _Clause(
        "fox_open",
        f"妖狐の死亡は未確定（生存の候補{len(end.fox_alive)}人: {_names(ledger, end.fox_alive)}"
        f"／死亡の候補: {_names(ledger, end.fox_dead)}）",
    )


def _endgame_clauses(
    ledger: PublicFactLedger, end: _Endgame, *, fox_dead_in_record: bool
) -> list[_Clause]:
    """The village-eye summary, or [] when it settles nothing worth a line.

    A claimant with only whites, early on, settles neither a wolf nor the
    fox; three such lines of fox candidates would be noise. A fox the record
    already buries is not buried again.
    """
    clauses = _wolf_clauses(ledger, end)
    if not clauses and not (end.fox_dead_for_sure and not fox_dead_in_record):
        return []
    if not fox_dead_in_record:
        clauses.append(_fox_clause(ledger, end))
    return clauses


def _fox_dead_in_record(ledger: PublicFactLedger, plain_solver: RoleSolver | None) -> bool:
    if plain_solver is None or not plain_solver.is_possible(Hypothesis()):
        return False
    return not _anyone_alive_could_be(ledger, plain_solver, RoleName.FOX)


# -- derived conclusions: computed for everyone, shown by skill --


@dataclass(frozen=True)
class _ConditionalRow:
    """One "if this claimant is genuine and truthful" row, clause by clause."""

    head: str
    items: tuple[DerivedItem, ...]


# A derived fact about one seat replaces the record's weaker fact about it.
_SEAT_FACT_KINDS = frozenset({"role", "cleared"})


@dataclass(frozen=True)
class _DerivedBoard:
    """The public section's conclusions, split into what the record settles
    (every seat reads it at once) and what is derived from it (each seat
    notices at its own pace, `app/ai/reasoning/skill.py`).

    "Derived" is decided by the rules, not by wording: a fact the solver
    reaches only with the curse applied to a claimant's own account, or with
    the game still running (`DERIVED_RULE_MODULE_IDS`), is derived; so is
    every conditional row and the execution line.
    """

    plain_facts: tuple[_Fact, ...] = ()
    derived_facts: tuple[tuple[_Fact, DerivedItem], ...] = ()
    rows: tuple[_ConditionalRow, ...] = ()
    execution: DerivedItem | None = None

    @property
    def items(self) -> tuple[DerivedItem, ...]:
        found = [item for _fact, item in self.derived_facts]
        for row in self.rows:
            found.extend(row.items)
        if self.execution is not None:
            found.append(self.execution)
        return tuple(found)

    @property
    def fact_keys(self) -> frozenset[tuple[str, ...]]:
        """Every public fact, shown or not: a seat's private tail must not
        restate as its own a public conclusion it has not noticed yet."""
        return frozenset(
            [fact.key for fact in self.plain_facts]
            + [fact.key for fact, _item in self.derived_facts]
        )

    def visible_facts(self, shown: Callable[[DerivedItem], bool]) -> list[_Fact]:
        derived = [fact for fact, item in self.derived_facts if shown(item)]
        settled = {
            seat for fact in derived if fact.key[0] in _SEAT_FACT_KINDS for seat in fact.seats
        }
        plain = [
            fact
            for fact in self.plain_facts
            if not (fact.key[0] in _SEAT_FACT_KINDS and settled.intersection(fact.seats))
        ]
        return plain + derived

    def visible_rows(self, shown: Callable[[DerivedItem], bool]) -> list[str]:
        rows: list[str] = []
        for row in self.rows:
            texts = [item.text for item in row.items if shown(item)]
            if texts:
                rows.append(f"  - {row.head}" + "。".join(texts))
        return rows

    def visible_execution(self, shown: Callable[[DerivedItem], bool]) -> list[str]:
        if self.execution is None or not shown(self.execution):
            return []
        return [self.execution.text]


def _derived_board(
    ledger: PublicFactLedger,
    observations: ObservationSet,
    public_solver: RoleSolver | None,
    plain_solver: RoleSolver | None,
) -> _DerivedBoard:
    plain = _public_logic_facts(ledger, observations, plain_solver)
    if public_solver is None:
        return _DerivedBoard(plain_facts=tuple(plain))
    plain_keys = {fact.key for fact in plain}
    derived = [
        (
            fact,
            DerivedItem(
                key=("fact", *fact.key),
                text=fact.text + fact.predicate,
                seats=fact.seats,
            ),
        )
        for fact in _public_logic_facts(ledger, observations, public_solver)
        if fact.key not in plain_keys
    ]
    refuted = {fact.key[1] for fact, _item in derived if fact.key[0] == "not_seer"}
    fox_dead = _fox_dead_in_record(ledger, plain_solver)
    public_end = _endgame(ledger, public_solver)
    return _DerivedBoard(
        plain_facts=tuple(plain),
        derived_facts=tuple(derived),
        rows=tuple(
            _conditional_rows(
                ledger,
                public_solver,
                plain_solver,
                refuted=refuted,
                fox_dead_in_record=fox_dead,
            )
        ),
        execution=_execution_item(ledger, public_solver, public_end),
    )


def _conditional_block(rows: list[str]) -> list[str]:
    if not rows:
        return []
    return ["- 仮定つきで言えること（仮定が正しい場合だけ）:", *rows]


def _conditional_claimants(ledger: PublicFactLedger) -> list[tuple[str, RoleName]]:
    """Every seer and medium claimant, alive or dead, standing on the claim.

    The dead count: a dead claimant's account can still break down, and with
    the real seer bitten early it is the only account left to read (run #14
    seed 2, where the section vanished on day 6).
    """
    found: list[tuple[str, RoleName]] = []
    for claimant in _seer_claimants(ledger):
        if ledger.claimed_role_of(claimant) is RoleName.SEER:
            found.append((claimant, RoleName.SEER))
    mediums = [
        claim.player_id
        for claim in ledger.co_declarations()
        if claim.claimed_role is RoleName.MEDIUM
    ]
    for claimant in _seat_ordered_ids(ledger, mediums):
        if ledger.claimed_role_of(claimant) is RoleName.MEDIUM:
            found.append((claimant, RoleName.MEDIUM))
    return found


def _conditional_rows(
    ledger: PublicFactLedger,
    public_solver: RoleSolver,
    plain_solver: RoleSolver | None,
    *,
    refuted: set[str],
    fox_dead_in_record: bool,
) -> list[_ConditionalRow]:
    """What follows if one claimant is real and told the truth -- or that it
    cannot be so.

    Never filed under the certainties: "the game ends if カイト hangs" is true
    only in the worlds where ケント is the seer and his blacks are real, and
    the row says so in its first words. One claimant at a time, since two
    claims on one role cannot both hold. A claimant the certainties already
    rule out as the holder gets no row.
    """
    if not public_solver.is_possible(Hypothesis()):
        return []
    spare = _spare_ropes(ledger, plain_solver)
    rows: list[_ConditionalRow] = []
    for claimant, role in _conditional_claimants(ledger):
        if claimant in refuted or not public_solver.is_possible(has_role(claimant, role)):
            continue
        name = ledger.name_of(claimant) + ("" if ledger.is_alive(claimant) else "(死亡)")
        head = f"{name}が真の{ROLE_DEFINITIONS[role].label_ja}で結果も正しいなら: "
        supposed = public_solver.assuming(GenuineClaim(claimant, role), HonestResults(claimant))
        if not supposed.is_possible(Hypothesis()):
            reason = _broken_reason(ledger, public_solver, supposed, claimant, role)
            item = DerivedItem(
                key=("broken", claimant, role.value),
                text=f"成り立たない（{reason}）",
                seats=(claimant,),
            )
            rows.append(_ConditionalRow(head=head, items=(item,)))
            continue
        end = _endgame(ledger, supposed)
        if end is None:
            continue
        clauses = _endgame_clauses(ledger, end, fox_dead_in_record=fox_dead_in_record)
        if not clauses:
            continue
        clauses.extend(_conditional_rope_clauses(ledger, end, spare))
        rows.append(
            _ConditionalRow(
                head=head,
                items=tuple(
                    DerivedItem(
                        key=("if", claimant, role.value, clause.kind, *clause.seats),
                        text=clause.text,
                        seats=(claimant, *clause.seats),
                    )
                    for clause in clauses
                ),
            )
        )
    return rows


def _broken_reason(
    ledger: PublicFactLedger,
    public_solver: RoleSolver,
    supposed: RoleSolver,
    claimant: str,
    role: RoleName,
) -> str:
    """Why "this claimant is real and truthful" admits no world, in a phrase.

    The common case is the game itself: the account leaves no living wolf, or
    leaves the wolves level with the rest, and the engine would have ended the
    game. That is checked by dropping the game-running rule and counting the
    living wolves under the same supposition. Otherwise the solver's own core
    names the rule the account collides with.
    """
    unbounded = RoleSolver(
        public_solver.observations,
        public_solver.perspective,
        rules=tuple(
            module
            for module in default_rule_modules()
            if module.module_id != GameRunningRuleModule.module_id
        ),
        cache=public_solver.cache,
    ).assuming(GenuineClaim(claimant, role), HonestResults(claimant))
    alive = ledger.alive_ids()
    if unbounded.is_possible(Hypothesis()):
        counts = _living_wolf_counts(unbounded, alive)
        if counts and max(counts) == 0:
            return "生存者に人狼が1人もいないことになるが、それなら試合はもう終わっている"
        if counts and 2 * min(counts) >= len(alive):
            return (
                f"生存{len(alive)}人のうち人狼が{min(counts)}人以上になるが、"
                "それなら試合はもう終わっている"
            )
        return (
            "それまでの処刑か夜明けの時点で人狼が人狼以外と同数以上になり、"
            "試合が終わっていたことになる"
        )
    core = supposed.explain_contradiction(Hypothesis())
    pairs = list(zip(core.constraint_ids, core.explanations, strict=True))
    rule = [
        namify(text, ledger)
        for cid, text in pairs
        if not cid.startswith(("genuine:", "honest:", "role_count:"))
    ]
    supposition = [namify(text, ledger) for cid, text in pairs if cid.startswith("honest:")]
    return "".join([*rule[:1], *supposition[:1]]).rstrip("。") or "公開情報と矛盾する"


def _spare_ropes(ledger: PublicFactLedger, plain_solver: RoleSolver | None) -> int:
    ropes, to_hang = _rope_counts(ledger, plain_solver)
    return max(0, ropes - to_hang)


def _conditional_rope_clauses(
    ledger: PublicFactLedger, end: _Endgame, public_spare: int
) -> list[_Clause]:
    """The rope count under the supposition, when it says more than the
    public one. Late in a game the public line counts every wolf that may live
    and so reads 0; under one claimant's account the inhumans left to hang can
    be fewer. A fact about that account, not a plan."""
    ropes, _ = _rope_counts(ledger, None)
    to_hang = end.wolves_max + (0 if end.fox_dead_for_sure else 1)
    spare = max(0, ropes - to_hang)
    if spare <= public_spare:
        return []
    return [_Clause("rope", f"吊るべき人外は最大{to_hang}人で、外してよい吊りは{spare}回")]


_EXECUTION_LINE_ROPES = 3


def _execution_item(
    ledger: PublicFactLedger, solver: RoleSolver, end: _Endgame | None
) -> DerivedItem | None:
    """Before the vote, near the end: whose execution can end the game today.

    Per living seat, from public logic only: could they be the last living
    wolf (hanging them leaves none), and could hanging them as a non-wolf
    bring the wolves level. Facts about the rules, not a recommendation.
    Printed in the day's discussion and vote with three ropes or fewer left;
    earlier the list is the whole table and says nothing.
    """
    if end is None or ledger.phase not in (Phase.DISCUSSION, Phase.VOTING, Phase.RUNOFF):
        return None
    alive = ledger.alive_ids()
    if not alive or ropes_left(len(alive)) > _EXECUTION_LINE_ROPES:
        return None
    wolf = RoleName.WEREWOLF
    last = [
        pid
        for pid in alive
        if solver.is_possible(
            Hypothesis(
                claims=(RoleIs(pid, wolf), RoleCountAmong(alive, wolf, at_least=1, at_most=1))
            )
        )
    ]
    # Hanging a non-wolf leaves `alive - 1` seats with the same wolves:
    # level once 2 * wolves >= alive - 1.
    level = len(alive) // 2
    parity = [
        pid
        for pid in alive
        if solver.is_possible(
            Hypothesis(claims=(RoleIsNot(pid, wolf), RoleCountAmong(alive, wolf, at_least=level)))
        )
    ]
    if not last and not parity:
        return None
    parts: list[str] = []
    if last:
        who = (
            "生存者の誰でもありうる"
            if len(last) == len(alive)
            else f"ありうるのは{_names(ledger, last)}"
        )
        parts.append(
            f"処刑した人が最後の人狼なら人狼0人で試合が終わる（{who}。{_winner_if_wolves_gone(end)}）"
        )
    else:
        parts.append("今日の処刑で人狼0人にはならない")
    if parity:
        who = "誰でも" if len(parity) == len(alive) else _names(ledger, parity)
        parts.append(
            f"人狼以外を処刑すると人狼が同数になり試合が終わることがある（{who}。"
            f"{_winner_at_parity(end)}）"
        )
    else:
        parts.append("人狼以外の処刑では今日は終わらない")
    return DerivedItem(
        key=("execution_end",),
        text="- 今日の処刑で試合が終わるか（公開情報から）: " + "。".join(parts),
        seats=tuple(dict.fromkeys([*last, *parity])),
    )


def _seat_endgame_lines(
    ledger: PublicFactLedger,
    seat_solver: RoleSolver,
    own_role: RoleName,
    public_solver: RoleSolver | None,
) -> list[str]:
    """The same summary from the seat's own solver, with no supposition.

    A wolf already knows every wolf, and the fox is itself the fox: each is
    told how its own side wins instead of a list of fox candidates. A
    village-side seat is told only what its own knowledge adds: a clause the
    public solver reaches as well is a public derived conclusion, which the
    seat sees in the public section when its skill gets there.
    """
    end = _endgame(ledger, seat_solver)
    if end is None:
        return []
    head = "- 終局（あなたの視点）: "
    if own_role is RoleName.WEREWOLF:
        wolves = len(end.living_wolves)
        others = len(end.alive) - wolves
        text = (
            f"生存の人狼は{_names(ledger, end.living_wolves)}の{wolves}人、人狼以外は{others}人。"
            f"人狼以外があと{others - wolves}人減ると同数で試合が終わる"
        )
        if end.fox_dead_for_sure:
            # The fox's death is already a fact line; this only draws the end.
            text += "（そのとき人狼陣営の勝ち）"
        else:
            text += (
                "。その時点で妖狐が生きていれば妖狐の勝ちなので、"
                "人狼陣営の勝ちには妖狐の死亡が要る"
            )
        return [head + text]
    if own_role is RoleName.FOX:
        clauses = [
            "人狼が0人になるか、人狼が人狼以外と同数以上になった時点で"
            "あなた(妖狐)が生きていれば、あなたの勝ち",
            *(clause.text for clause in _wolf_clauses(ledger, end)),
        ]
        return [head + "。".join(clauses)]
    public = _endgame(ledger, public_solver) if public_solver is not None else None
    fox_dead = public is not None and public.fox_dead_for_sure
    shared = (
        {clause.text for clause in _wolf_clauses(ledger, public)} if public is not None else set()
    )
    own = [clause for clause in _wolf_clauses(ledger, end) if clause.text not in shared]
    if not own and not (end.fox_dead_for_sure and not fox_dead):
        return []
    if not fox_dead:
        own.append(_fox_clause(ledger, end))
    return [head + "。".join(clause.text for clause in own)]


def _fox_curse_lines(
    player_id: str,
    ledger: PublicFactLedger,
    seat_solver: RoleSolver,
) -> list[str]:
    """The fox's own reading of the curse (run #14 seed 3, 5-B2).

    ホノカ worked out that レン, who said he had divined her, was fake -- she
    was alive -- and went on attacking レン "so as not to be divined". Only the
    real seer's divination kills the fox, so this line says which seer
    claimants cannot be the real one from the fox's seat (they cannot curse
    it), which living claimants still may be, and whether the real seer can
    be alive at all.
    """
    claimants = [pid for pid in _seer_claimants(ledger) if pid != player_id]
    if not seat_solver.is_possible(Hypothesis()):
        return []
    fake = [pid for pid in claimants if not seat_solver.is_possible(has_role(pid, RoleName.SEER))]
    living = [pid for pid in claimants if ledger.is_alive(pid) and pid not in fake]
    seer_alive = _anyone_alive_could_be(ledger, seat_solver, RoleName.SEER)
    parts = ["妖狐が死ぬのは真の占い師に占われた夜だけ（真でない占いCOに占われても死なない）"]
    if fake:
        parts.append(f"あなたの視点で真の占い師でない占いCO: {_names(ledger, fake)}")
    if not seer_alive:
        parts.append("真の占い師はもう生存していない")
    elif living:
        parts.append(f"生存の占いCOで真の可能性が残るのは{_names(ledger, living)}")
    elif claimants:
        parts.append(
            "生存の占いCOに真の可能性は残っていない（真の占い師が生きていればCOしていない誰か）"
        )
    return [f"- 呪殺（{_KEEP_UNSAID}）: " + "。".join(parts)]


def _timeline_lines(ledger: PublicFactLedger, observations: ObservationSet) -> list[str]:
    conflicts = find_timeline_conflicts(observations)
    if not conflicts:
        return []
    lines = ["- 公開された判定と公開の時系列が合わない点（真役職の言い間違いもあり得る）:"]
    seen: set[str] = set()
    for conflict in conflicts:
        text = namify(conflict.explanation, ledger)
        if text in seen:
            continue
        seen.add(text)
        lines.append(f"  - {text}")
    return lines


# -- the seat's own knowledge --


def _private_lines(
    state: GameState,
    player_id: str,
    ledger: PublicFactLedger,
    observations: ObservationSet,
    seat_solver: RoleSolver | None,
    *,
    public_keys: frozenset[tuple[str, ...]],
    public_solver: RoleSolver | None = None,
) -> list[str]:
    perspective = PlayerPrivatePerspective(player_id)
    own_role = state.players[player_id].role
    lines = [PRIVATE_HEADING, f"- あなたの本当の役職: {ROLE_DEFINITIONS[own_role].label_ja}"]
    known = perspective.known_roles(observations)
    allies = sorted(pid for pid in known if pid != player_id)
    if allies:
        label = "仲間の人狼" if own_role is RoleName.WEREWOLF else "共有の相方"
        lines.append(
            f"- {label}: "
            + "、".join(
                f"{ledger.name_of(pid)}{'' if ledger.is_alive(pid) else '(死亡)'}" for pid in allies
            )
        )
    for result in perspective.known_divine_results(observations):
        colour = "黒(人狼)" if result.is_werewolf else "白(人狼ではない)"
        status = (
            "公開済み"
            if ledger.find_result(player_id, "seer", result.target_id) is not None
            else "未公開"
        )
        target = ledger.name_of(result.target_id)
        lines.append(f"- 占い結果 {result.night}日目夜: {target}={colour}（{status}）")
    for medium in perspective.known_medium_results(observations):
        colour = "黒(人狼)" if medium.is_werewolf else "白(人狼ではない)"
        status = (
            "公開済み"
            if ledger.find_result(player_id, MEDIUM_RESULT, medium.target_id) is not None
            else "未公開"
        )
        target = ledger.name_of(medium.target_id)
        lines.append(f"- 霊媒結果 {medium.day}日目処刑: {target}={colour}（{status}）")
    nights = perspective.known_night_actions(observations)
    if nights.guards:
        lines.append(
            "- 護衛履歴: "
            + " / ".join(
                f"{night}日目夜 {ledger.name_of(target)}"
                for night, target in sorted(nights.guards.items())
            )
        )
    if own_role is RoleName.HUNTER:
        lines.extend(_quiet_guarded_nights(ledger, observations, nights.guards))
    if nights.attacks and own_role is RoleName.WEREWOLF:
        outcome_by_night = {record.day: record.succeeded for record in state.attack_records}
        lines.append(
            "- 襲撃履歴: "
            + " / ".join(
                f"{night}日目夜 {ledger.name_of(target)}"
                + ("" if outcome_by_night.get(night, True) else "（失敗=護衛か妖狐）")
                for night, target in sorted(nights.attacks.items())
            )
        )
    false_colours = (
        _false_colours(player_id, ledger, observations, allies, seat_solver)
        if seat_solver is not None
        else []
    )
    lines.extend(
        _private_certainty_lines(
            player_id,
            own_role,
            ledger,
            observations,
            allies,
            seat_solver,
            public_keys,
            extra=[text for _verdict, text in false_colours],
        )
    )
    if seat_solver is not None:
        lines.extend(_seat_endgame_lines(ledger, seat_solver, own_role, public_solver))
        if own_role is RoleName.FOX:
            lines.extend(_fox_curse_lines(player_id, ledger, seat_solver))
        lines.extend(
            _seat_supposition_lines(
                player_id,
                own_role,
                ledger,
                observations,
                allies,
                seat_solver,
                false_colours=[verdict for verdict, _text in false_colours],
            )
        )
    return lines


def _false_colours(
    player_id: str,
    ledger: PublicFactLedger,
    observations: ObservationSet,
    allies: Iterable[str],
    seat_solver: RoleSolver,
) -> list[tuple[PublicVerdict, str]]:
    """Published colours this seat knows to be false, said as fact.

    The wolves knew m34's 「カイト=白」 was false the moment アオイ said it; the
    memo had no way to say so (2-B4). Verdicts by the seat or a known ally are
    left out: the wolves wrote their own fakes.
    """
    skip = {player_id, *allies}
    found: list[tuple[PublicVerdict, str]] = []
    for verdict in sorted(observations.verdicts, key=lambda v: (v.day, v.source_night)):
        if verdict.claimant_id in skip:
            continue
        target = ledger.name_of(verdict.target_id)
        wolf = has_role(verdict.target_id, RoleName.WEREWOLF)
        if verdict.is_werewolf and not seat_solver.is_possible(wolf):
            truth = f"{target}は人狼ではない"
        elif not verdict.is_werewolf and seat_solver.is_forced(wolf):
            truth = f"{target}は人狼"
        else:
            continue
        colour = "黒" if verdict.is_werewolf else "白"
        found.append(
            (
                verdict,
                f"{ledger.name_of(verdict.claimant_id)}の『{target}={colour}』は事実と違う（{truth}）",
            )
        )
    return found


def _seat_supposition_lines(
    player_id: str,
    own_role: RoleName,
    ledger: PublicFactLedger,
    observations: ObservationSet,
    allies: Iterable[str],
    seat_solver: RoleSolver,
    *,
    false_colours: list[PublicVerdict],
) -> list[str]:
    """What this seat may conclude under a named supposition, kept apart from
    what it knows. Each row opens with its supposition: the design contract's
    rule that a conditional conclusion states what it rests on."""
    rows = _truthful_holder_rows(
        player_id, own_role, ledger, observations, allies, seat_solver, false_colours
    )
    rows += _no_village_bluff_rows(player_id, ledger, observations, allies, seat_solver)
    if not rows:
        return []
    head = f"- あなたの視点で、仮定つきで言えること（仮定が外れれば成り立たない。{_KEEP_UNSAID}）:"
    return [head, *rows]


_NON_VILLAGE_LABELS = {RoleName.WEREWOLF: "人狼", RoleName.MADMAN: "狂人", RoleName.FOX: "妖狐"}

_RESULT_ROLES = ((SEER_RESULT, RoleName.SEER), (MEDIUM_RESULT, RoleName.MEDIUM))


def _truthful_holder_rows(
    player_id: str,
    own_role: RoleName,
    ledger: PublicFactLedger,
    observations: ObservationSet,
    allies: Iterable[str],
    seat_solver: RoleSolver,
    false_colours: list[PublicVerdict],
) -> list[str]:
    """Who cannot be the real seer (or medium), if the real one tells true
    colours -- and who must be, if nobody holds the card in silence.

    Two suppositions, each named (2-B4): a real seer may lie about a colour,
    and a real seer may never claim. Asked only once the seat holds a colour
    it knows is false; a seat holding the card itself already knows.
    """
    rows: list[str] = []
    skip = {player_id, *allies}
    for result_type, role in _RESULT_ROLES:
        if own_role is role or not any(v.result_type == result_type for v in false_colours):
            continue
        claimants = _ability_claimants(ledger, observations, result_type, role)
        possible = [
            pid for pid in claimants
            if pid not in skip and seat_solver.is_possible(has_role(pid, role))
        ]
        truthful = seat_solver.assuming(TruthfulIfGenuine())
        if not truthful.is_possible(Hypothesis()):
            continue
        ruled_out = [pid for pid in possible if not truthful.is_possible(has_role(pid, role))]
        if not ruled_out:
            continue
        label = ROLE_DEFINITIONS[role].label_ja
        row = f"  - 真の{label}が色を偽らないなら、{_names(ledger, ruled_out)}は真の{label}ではない"
        no_latent = truthful.assuming(NoLatentClaim(role))
        if no_latent.is_possible(Hypothesis()):
            left = [pid for pid in claimants if no_latent.is_possible(has_role(pid, role))]
            if len(left) == 1:
                row += f"。潜伏の{label}もいないなら、真の{label}は{ledger.name_of(left[0])}"
        rows.append(row)
    return rows


def _ability_claimants(
    ledger: PublicFactLedger, observations: ObservationSet, result_type: str, role: RoleName
) -> list[str]:
    if role is RoleName.SEER:
        return _seer_claimants(ledger)
    wanted = set(observations.claimants_of(role)) | {
        v.claimant_id for v in observations.verdicts if v.result_type == result_type
    }
    return _seat_ordered_ids(ledger, wanted)


def _no_village_bluff_rows(
    player_id: str,
    ledger: PublicFactLedger,
    observations: ObservationSet,
    allies: Iterable[str],
    seat_solver: RoleSolver,
) -> list[str]:
    """The claims this seat knows are false, read as "a village seat would not
    bluff a role".

    ケント knew ダイキ and アオイ were not the seer, and that アオイ was neither
    wolf nor fox. "So アオイ is the madman" is the table's habit, not a rule (a
    hunter may claim seer), which is why the row names the supposition (2-B3).
    A claim the seat cannot yet call false, or whose holder is already settled,
    gets nothing; nor does a claimant left with all three non-village roles.
    """
    skip = {player_id, *allies}
    fakes = [
        claim.player_id
        for claim in observations.claims
        if claim.player_id not in skip
        and claim.role is not RoleName.VILLAGER
        and not seat_solver.is_possible(has_role(claim.player_id, claim.role))
        and seat_solver.certain_role(claim.player_id) is None
    ]
    if not fakes:
        return []
    supposed = seat_solver.assuming(NoVillageBluff())
    if not supposed.is_possible(Hypothesis()):
        return []
    cells: list[str] = []
    for pid in _seat_ordered_ids(ledger, fakes):
        roles = [role for role in NON_VILLAGE_ROLES if supposed.is_possible(has_role(pid, role))]
        if not roles or len(roles) == len(NON_VILLAGE_ROLES):
            continue
        dead = "" if ledger.is_alive(pid) else "(死亡)"
        cells.append(
            f"{ledger.name_of(pid)}{dead}は" + "か".join(_NON_VILLAGE_LABELS[r] for r in roles)
        )
    if not cells:
        return []
    return ["  - 村側の騙りがなければ: " + "、".join(cells)]


def _quiet_guarded_nights(
    ledger: PublicFactLedger, observations: ObservationSet, guards: Mapping[int, str]
) -> list[str]:
    """The empty mornings, each with the seat the hunter covered that night.

    The engine never tells the hunter whether a guard held, and this says no
    more than the rules give: the wolves bit someone, so the bite went to the
    covered seat or to the fox (3-B2). That the covered seat is no wolf does
    not follow -- the fox may have been the one bitten.
    """
    quiet = [
        (night, guards.get(night))
        for night in range(1, observations.day)
        if not observations.deaths_on(night)
    ]
    if not quiet:
        return []
    cells = "・".join(
        f"{night}日目夜({ledger.name_of(target) if target else '護衛なし'})"
        for night, target in quiet
    )
    return [f"- 死体なしの夜: {cells}。襲撃先は護衛先か妖狐（護衛の成否は知らされない）"]


def _private_certainty_lines(
    player_id: str,
    own_role: RoleName,
    ledger: PublicFactLedger,
    observations: ObservationSet,
    allies: Iterable[str],
    seat_solver: RoleSolver | None,
    public_keys: frozenset[tuple[str, ...]],
    *,
    extra: Iterable[str] = (),
) -> list[str]:
    """Hard conclusions only this seat can draw, beyond its card and results.

    Read through the seat's own solver, and only what the public section has
    not already said. The dead count too: a wolf that bit one of two corpses
    knows the other was the fox, and in the first live game the wolves planned
    to have their own madman hanged "to get rid of the fox" for want of it.

    Werewolves get every settled role except "not a wolf" -- they hold the whole
    wolf roster, so that is true of everyone else and tells them nothing. The
    real seer is not told which claimants are fake; knowing its own card, that
    is every one of them.
    """
    if seat_solver is None:
        return []
    facts = _settled_facts(
        ledger,
        observations,
        seat_solver,
        exclude=frozenset({player_id, *allies}),
        wolf_view=own_role is RoleName.WEREWOLF,
        refute_claims=own_role is not RoleName.SEER,
        viewer_id=player_id,
        fox_view=own_role is RoleName.FOX,
    )
    texts = _fact_texts(fact for fact in facts if fact.key not in public_keys)
    texts.extend(extra)
    if not texts:
        return []
    return [
        f"- あなたの視点で論理的に確定すること（卓には言えない根拠を含む。{_KEEP_UNSAID}）: "
        + " / ".join(texts)
    ]


__all__ = [
    "MEMO_HEADING",
    "PRIVATE_HEADING",
    "SELF",
    "namify",
    "render_board_memo",
    "render_roster",
]
