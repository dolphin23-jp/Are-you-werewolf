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
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from app.ai.reasoning.facts import MEDIUM_RESULT, PublicFactLedger
from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import PlayerPrivatePerspective
from app.ai.reasoning.solver.backend import Certainty, has_role
from app.ai.reasoning.solver.queries import RoleSolver
from app.ai.reasoning.timeline import find_timeline_conflicts
from app.engine.roles import ROLE_DEFINITIONS, RoleName
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

_PLAYER_ID_RE = re.compile(r"(?<![0-9A-Za-z])p(\d+)(?![0-9])")


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
    lines.extend(_roster_lines(ledger))
    lines.extend(_claim_lines(state, ledger))
    lines.extend(_verdict_lines(ledger))
    lines.extend(_vote_lines(ledger))
    lines.extend(_logic_lines(ledger, observations, public_solver))
    lines.extend(_timeline_lines(ledger, observations))
    lines.extend(_private_lines(state, player_id, ledger, observations, seat_solver))
    return "\n".join(lines)


# -- public sections --


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
        if len(claimants) > capacity:
            fakes = len(claimants) - capacity
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


def _verdict_lines(ledger: PublicFactLedger) -> list[str]:
    results = ledger.public_results()
    if not results:
        return ["- 公開された判定: なし"]
    by_claimant: dict[str, list[str]] = defaultdict(list)
    for result in sorted(results, key=lambda r: (r.claimant_id, r.source_night, r.day)):
        colour = "黒" if result.is_werewolf else "白"
        if result.result_type == MEDIUM_RESULT:
            when = f"{result.source_night}日目処刑"
        else:
            when = f"{result.source_night}日目夜"
        by_claimant[result.claimant_id].append(
            f"{when} {ledger.name_of(result.target_id)}={colour}"
        )
    lines = ["- 公開された判定:"]
    for claimant_id, items in by_claimant.items():
        role = ledger.claimed_role_of(claimant_id)
        role_label = _ROLE_LABELS.get(role, "") if role is not None else ""
        lines.append(f"  - {ledger.name_of(claimant_id)}({role_label}CO): " + " / ".join(items))
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


def _logic_lines(
    ledger: PublicFactLedger,
    observations: ObservationSet,
    public_solver: RoleSolver | None,
) -> list[str]:
    """What the rules settle from public information alone."""
    facts: list[str] = []
    if observations.first_victim_id is not None:
        facts.append(
            f"初日犠牲者の{ledger.name_of(observations.first_victim_id)}は人狼でも妖狐でもない"
        )
    for night in observations.nights_with_deaths():
        deaths = observations.deaths_on(night)
        if len(deaths) >= 2:
            names = "、".join(ledger.name_of(death.player_id) for death in deaths)
            facts.append(
                f"{night}日目夜は死体が{len(deaths)}つ（{names}）。襲撃は一晩1件、"
                "呪殺も1件なので、このうち1人は占われて死んだ妖狐"
            )
    if public_solver is not None:
        for pid in ledger.alive_ids():
            role = public_solver.certain_role(pid)
            if role is not None:
                facts.append(f"{ledger.name_of(pid)}は{_ROLE_LABELS[role]}で確定")
                continue
            if public_solver.assess(has_role(pid, RoleName.WEREWOLF)) is Certainty.IMPOSSIBLE:
                facts.append(f"{ledger.name_of(pid)}は人狼ではないと確定")
    if not facts:
        return []
    return ["- 公開情報だけで確定すること:"] + [f"  - {fact}" for fact in facts]


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
    lines.extend(_private_certainty_lines(player_id, own_role, ledger, allies, seat_solver))
    return lines


def _private_certainty_lines(
    player_id: str,
    own_role: RoleName,
    ledger: PublicFactLedger,
    allies: Iterable[str],
    seat_solver: RoleSolver | None,
) -> list[str]:
    """Hard conclusions only this seat can draw, beyond its card and results.

    Skipped for werewolves: they hold the whole wolf roster, so "X is not a
    wolf" is true of everyone else and says nothing a wolf needs telling.
    """
    if seat_solver is None or own_role is RoleName.WEREWOLF:
        return []
    skip = {player_id, *allies}
    facts: list[str] = []
    for pid in ledger.alive_ids():
        if pid in skip:
            continue
        role = seat_solver.certain_role(pid)
        if role is not None:
            facts.append(f"{ledger.name_of(pid)}は{_ROLE_LABELS[role]}で確定")
        elif seat_solver.assess(has_role(pid, RoleName.WEREWOLF)) is Certainty.IMPOSSIBLE:
            facts.append(f"{ledger.name_of(pid)}は人狼ではない")
    if not facts:
        return []
    return [
        "- あなたの視点で論理的に確定すること（卓には言えない根拠を含む）: " + " / ".join(facts)
    ]


__all__ = ["MEMO_HEADING", "PRIVATE_HEADING", "namify", "render_board_memo", "render_roster"]
