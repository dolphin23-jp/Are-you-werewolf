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
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from app.ai.reasoning.facts import MEDIUM_RESULT, SEER_RESULT, PublicFactLedger
from app.ai.reasoning.observations import ObservationSet, PublicVerdict
from app.ai.reasoning.perspectives import PlayerPrivatePerspective
from app.ai.reasoning.solver.backend import AnyOf, Certainty, Hypothesis, RoleIs, has_role
from app.ai.reasoning.solver.queries import RoleSolver
from app.ai.reasoning.solver.rules.claimed_divination import (
    ClaimedDivinationRuleModule,
    SeerClaimRefutation,
    curse_nights,
    seer_claim_refutations,
)
from app.ai.reasoning.timeline import (
    can_act_on_night,
    could_publish_on_day,
    find_timeline_conflicts,
)
from app.engine.roles import ROLE_DEFINITIONS, TOTAL_PLAYERS, RoleName
from app.engine.state import GameState, PublicDeathCause

MEMO_HEADING = "【盤面メモ】"
PRIVATE_HEADING = "【あなただけが知っていること】"

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
# `app/engine` actually does -- night_resolver (curse on every night including
# night 0; the attack never kills the fox; one attack and one curse per night;
# the medium reads only the day's execution; results are wolf / not wolf),
# game.submit_night_action (night 0 is divination only; the hunter may repeat a
# guard but not guard themselves) and roles (the first victim is never a wolf or
# the fox). Change one only together with the engine.
_RULE_LINES = (
    "- ルール: 占い・霊媒の結果は人狼かどうかだけ（狂人・妖狐は白）。"
    "霊媒が知るのは処刑された人の結果だけ。"
    "狩人は1日目夜から護衛でき、同じ人の連続護衛も可（自分は守れない）。"
    "初日犠牲者は人狼でも妖狐でもない",
    "- 死体のルール: 占い先が妖狐ならその夜に呪殺（0日目夜も）。妖狐は襲撃では死なない。"
    "夜の死体は襲撃（1夜1人まで）と呪殺だけなので、1夜に2つなら片方は呪殺された妖狐。"
    "0日目夜は襲撃がなく、初日犠牲者以外の死体は呪殺",
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
) -> str:
    """The full memo for one seat. Pure function of the board and the seat."""
    ledger = PublicFactLedger(state)
    name = ledger.name_of
    lines: list[str] = [f"{MEMO_HEADING}{state.day}日目 / あなた: {name(player_id)}"]
    lines.extend(_rules_lines())
    lines.extend(_roster_lines(ledger))
    lines.extend(_claim_lines(state, ledger))
    lines.extend(_verdict_lines(ledger, observations))
    lines.extend(_vote_lines(ledger))
    public_facts = _public_logic_facts(ledger, observations, public_solver)
    lines.extend(_logic_lines(public_facts))
    lines.extend(_timeline_lines(ledger, observations))
    lines.extend(
        _private_lines(
            state,
            player_id,
            ledger,
            observations,
            seat_solver,
            public_keys=frozenset(fact.key for fact in public_facts),
        )
    )
    return "\n".join(lines)


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


def _roster_lines(ledger: PublicFactLedger) -> list[str]:
    alive = [player for player in ledger.players() if player.alive]
    dead = [player for player in ledger.players() if not player.alive]
    lines = [f"- 生存({len(alive)}人): " + "、".join(player.name for player in alive)]
    if dead:
        lines.append(
            "- 死亡: "
            + "、".join(
                f"{player.name}({_death_label(player.death_cause, player.death_day)})"
                for player in dead
            )
        )
    wolves = ROLE_DEFINITIONS[RoleName.WEREWOLF].count
    rope = max(0, (len(alive) - wolves) - wolves - 1)
    lines.append(
        f"- 人狼は最大{wolves}人（公開情報では何人死んだか確定しない）/ "
        f"外してよい吊りの目安: {rope}回"
    )
    return lines


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
    return ["- CO: " + " / ".join(parts)]


def _freemason_note(state: GameState, ledger: PublicFactLedger, claimants: list[str]) -> str:
    confirmed = [claim for claim in state.freemason_partner_claims if claim.confirmed]
    if confirmed:
        pair = confirmed[0]
        return (
            f"（{ledger.name_of(pair.claimant_id)}と{ledger.name_of(pair.partner_id)}が"
            "互いに相方と確認済み）"
        )
    named = [claim for claim in state.freemason_partner_claims if not claim.confirmed]
    if named:
        claim = named[0]
        return (
            f"（{ledger.name_of(claim.claimant_id)}は相方を{ledger.name_of(claim.partner_id)}と"
            "名指ししたが本人の確認待ち）"
        )
    if len(claimants) == 1:
        return "（相方は未公開）"
    return ""


def _verdict_lines(ledger: PublicFactLedger, observations: ObservationSet) -> list[str]:
    """Seer accounts night by night, then medium results by claimant."""
    lines = _divination_lines(ledger, observations) + _medium_lines(ledger, observations)
    return lines or ["- 公開された判定: なし"]


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
            rows.append(f"  - {night}日目夜{header}: " + " / ".join(cells))
    if not rows:
        return []
    return ["- 占いCOの夜ごとの占い先（[ ]は公開日。未公開の夜も書く）:", *rows]


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
    results = [r for r in observations.verdicts if r.result_type == MEDIUM_RESULT]
    if not results:
        return []
    by_claimant: dict[str, list[str]] = defaultdict(list)
    for result in sorted(results, key=lambda r: (r.source_night, r.day)):
        colour = "黒" if result.is_werewolf else "白"
        by_claimant[result.claimant_id].append(
            f"{result.source_night}日目処刑 {ledger.name_of(result.target_id)}={colour}"
            f"[{result.day}日目公開]"
        )
    lines = ["- 霊媒結果:"]
    for claimant_id in _seat_ordered_ids(ledger, by_claimant):
        role = ledger.claimed_role_of(claimant_id)
        role_label = _ROLE_LABELS.get(role, "") if role is not None else ""
        lines.append(
            f"  - {ledger.name_of(claimant_id)}({role_label}CO): "
            + " / ".join(by_claimant[claimant_id])
        )
    return lines


def _vote_lines(ledger: PublicFactLedger) -> list[str]:
    votes = ledger.votes()
    if not votes:
        return []
    lines = ["- 投票履歴（吊り先←投票者）:"]
    rounds = sorted({(vote.day, vote.round) for vote in votes})
    for day, round_number in rounds:
        by_target: dict[str, list[str]] = defaultdict(list)
        for vote in ledger.votes_on(day, round_number):
            by_target[vote.target_id].append(ledger.name_of(vote.voter_id))
        ordered = sorted(by_target.items(), key=lambda item: (-len(item[1]), item[0]))
        rendered = " / ".join(
            f"{ledger.name_of(target)}←{'、'.join(voters)}({len(voters)})"
            for target, voters in ordered
        )
        label = f"{day}日目" + (f"決選{round_number - 1}" if round_number > 1 else "")
        lines.append(f"  - {label}: {rendered}")
    return lines


@dataclass(frozen=True)
class _Fact:
    """One settled conclusion, with a key naming what it settles.

    The key is what lets a seat's private tail drop a conclusion the public
    section already states -- comparing the wording would break the first time
    the two sections phrase the same fact differently.
    """

    key: tuple[str, ...]
    text: str


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
    return ["- 公開情報だけで確定すること:"] + [f"  - {fact.text}" for fact in facts]


def _settled_facts(
    ledger: PublicFactLedger,
    observations: ObservationSet,
    solver: RoleSolver,
    *,
    exclude: frozenset[str],
    wolf_view: bool,
    refute_claims: bool,
    viewer_id: str | None,
) -> list[_Fact]:
    """Hard conclusions a solver reaches about claimants, roles and seats.

    Shared by the public section and a seat's private tail, which differ only
    in the solver and in what they leave out. A board that admits no world at
    all settles nothing: read naively, "nobody can be the fox" would be true of
    every seat and printed as a fact.
    """
    if not solver.is_possible(Hypothesis()):
        return []
    facts: list[_Fact] = []
    if refute_claims:
        facts.extend(_refuted_claim_facts(ledger, observations, solver, exclude, viewer_id))
    facts.extend(_filled_role_facts(ledger, solver))
    for pid in ledger.known_player_ids():
        if pid in exclude:
            continue
        alive = ledger.is_alive(pid)
        role = solver.certain_role(pid)
        if role is not None and not (wolf_view and role is RoleName.WEREWOLF):
            label = ledger.name_of(pid) + ("" if alive else "(死亡)")
            facts.append(_Fact(("role", pid), f"{label}は{_ROLE_LABELS[role]}で確定"))
        elif (
            alive
            and role is None
            and not wolf_view
            and solver.assess(has_role(pid, RoleName.WEREWOLF)) is Certainty.IMPOSSIBLE
        ):
            facts.append(_Fact(("not_wolf", pid), f"{ledger.name_of(pid)}は人狼ではないと確定"))
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
        if definition.count == 1:
            holders = [pid for pid in dead if solver.is_possible(has_role(pid, role))]
            text = f"{label}(1人)は死亡済み。生存者に{label}はいない{_candidates(ledger, holders)}"
        else:
            text = f"{label}({definition.count}人)は全員死亡済み。生存者に{label}はいない"
        facts.append(_Fact(("filled", role.value), text))
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
    lines.extend(
        _private_certainty_lines(
            player_id, own_role, ledger, observations, allies, seat_solver, public_keys
        )
    )
    return lines


def _private_certainty_lines(
    player_id: str,
    own_role: RoleName,
    ledger: PublicFactLedger,
    observations: ObservationSet,
    allies: Iterable[str],
    seat_solver: RoleSolver | None,
    public_keys: frozenset[tuple[str, ...]],
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
    )
    texts = [fact.text for fact in facts if fact.key not in public_keys]
    if not texts:
        return []
    return [
        "- あなたの視点で論理的に確定すること（卓には言えない根拠を含む）: " + " / ".join(texts)
    ]


__all__ = ["MEMO_HEADING", "PRIVATE_HEADING", "namify", "render_board_memo", "render_roster"]
