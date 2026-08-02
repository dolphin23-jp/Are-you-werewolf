"""Deterministic, code-generated public-fact summaries.

The factual half of a day summary is not a language task. Who died, who COed,
which verdicts were published and who voted for whom are all in the ledger
already, and asking a model to restate them only adds a chance of getting them
wrong. So this module renders those facts from `PublicFactLedger` -- same input,
same string, no LLM, no network -- and the model is left with the part it is
actually needed for: what the table argued about.

Opinion never enters the fact block. `compose_day_summary` keeps the generated
commentary in a separate, explicitly labelled section so nothing downstream can
mistake it for something that happened.
"""

from __future__ import annotations

from app.ai.reasoning.facts import MEDIUM_RESULT, PublicFactLedger
from app.engine.roles import RoleName
from app.engine.state import PublicDeathCause

FACTS_HEADING = "【公開事実】"
OPINION_HEADING = "【議論の要点(生成)】"

_ROLE_LABELS: dict[RoleName, str] = {
    RoleName.VILLAGER: "村人",
    RoleName.WEREWOLF: "人狼",
    RoleName.MADMAN: "狂人",
    RoleName.SEER: "占い師",
    RoleName.MEDIUM: "霊媒師",
    RoleName.HUNTER: "狩人",
    RoleName.FOX: "妖狐",
    RoleName.FREEMASON: "共有者",
}

_DEATH_LABELS: dict[PublicDeathCause, str] = {
    PublicDeathCause.EXECUTED: "処刑",
    PublicDeathCause.NIGHT: "夜死亡",
    PublicDeathCause.FIRST_VICTIM: "初日犠牲者",
}


def render_public_fact_summary(ledger: PublicFactLedger, day: int | None = None) -> str:
    """Render the day's established public facts. Pure function of the ledger."""
    target_day = ledger.day if day is None else day
    lines = [f"{FACTS_HEADING}{target_day}日目"]

    alive = ledger.alive_ids()
    lines.append(f"- 生存者({len(alive)}人): {_labels(ledger, alive)}")

    dead = [ledger.player(pid) for pid in ledger.dead_ids()]
    if dead:
        rendered = "、".join(
            f"{player.label}[{player.death_day}日目{_death_label(player.death_cause)}]"
            for player in dead
            if player is not None
        )
        lines.append(f"- 死亡者: {rendered}")

    todays_co = [claim for claim in ledger.co_declarations() if claim.day == target_day]
    if todays_co:
        rendered = "、".join(
            f"{ledger.label_of(claim.player_id)}={_role_label(claim.claimed_role)}"
            for claim in todays_co
        )
        lines.append(f"- 本日のCO: {rendered}")

    todays_results = [result for result in ledger.public_results() if result.day == target_day]
    if todays_results:
        rendered = "、".join(
            f"{ledger.label_of(result.claimant_id)}の"
            f"{'霊媒' if result.result_type == MEDIUM_RESULT else '占い'}: "
            f"{ledger.label_of(result.target_id)}={'黒' if result.is_werewolf else '白'}"
            for result in todays_results
        )
        lines.append(f"- 本日公開された判定: {rendered}")

    for round_number in sorted({vote.round for vote in ledger.votes_on(target_day)}):
        rendered = "、".join(
            f"{ledger.label_of(vote.voter_id)}→{ledger.label_of(vote.target_id)}"
            for vote in ledger.votes_on(target_day, round_number)
        )
        lines.append(f"- 投票R{round_number}: {rendered}")

    executed_today = [
        execution for execution in ledger.executions() if execution.day == target_day
    ]
    if executed_today:
        rendered = "、".join(
            ledger.label_of(execution.player_id) for execution in executed_today
        )
        lines.append(f"- 処刑結果: {rendered}")

    night_deaths = ledger.night_death_ids(target_day)
    if night_deaths:
        lines.append(f"- 夜の死亡: {_labels(ledger, night_deaths)}")

    return "\n".join(lines)


def compose_day_summary(facts: str, opinion: str) -> str:
    """Facts first, generated commentary second and clearly labelled as such."""
    commentary = opinion.strip()
    if not commentary:
        return facts
    return f"{facts}\n{OPINION_HEADING}\n{commentary}"


def split_day_summary(summary: str) -> tuple[str, str]:
    """Inverse of :func:`compose_day_summary`, so the bounded-memory compressor
    can shrink the generated commentary without touching the facts."""
    head, separator, tail = summary.partition(f"\n{OPINION_HEADING}\n")
    if not separator:
        return summary, ""
    return head, tail


def _labels(ledger: PublicFactLedger, player_ids: tuple[str, ...]) -> str:
    return "、".join(ledger.label_of(pid) for pid in player_ids) or "なし"


def _death_label(cause: PublicDeathCause | None) -> str:
    return _DEATH_LABELS.get(cause, "死亡") if cause is not None else "死亡"


def _role_label(role: RoleName) -> str:
    return _ROLE_LABELS.get(role, role.value)
