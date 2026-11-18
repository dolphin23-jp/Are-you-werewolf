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

from collections.abc import Sequence

from app.ai.reasoning.board_memo import role_label
from app.ai.reasoning.facts import MEDIUM_RESULT, SEER_RESULT, PublicFactLedger
from app.ai.reasoning.observations import _dated_by_first_statement
from app.engine.roles import RoleName
from app.engine.state import ChatChannel, GameState, PublicDeathCause

FACTS_HEADING = "【公開事実】"
OPINION_HEADING = "【議論の要点(生成)】"

# v3: the claimable roles in the order a 17A table lists them.
_CLAIM_ORDER = (RoleName.SEER, RoleName.MEDIUM, RoleName.HUNTER, RoleName.FREEMASON)
# A quoted line is the seat's own words; a line in this game is at most 160
# characters, and a human's longer one is cut rather than left to grow the layer.
_QUOTE_MAX_CHARS = 160
_NESTED_QUOTES = str.maketrans("「」", "『』")

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


def render_public_fact_summary(
    ledger: PublicFactLedger, day: int | None = None, *, names_only: bool = False
) -> str:
    """Render the day's established public facts. Pure function of the ledger.

    `names_only` drops the `(pN)` suffix from every label. The chat-register
    prompt uses it so the model never sees the id form it is told not to speak.
    """
    target_day = ledger.day if day is None else day
    label = ledger.name_of if names_only else ledger.label_of
    lines = [f"{FACTS_HEADING}{target_day}日目"]

    alive = ledger.alive_ids()
    lines.append(f"- 生存者({len(alive)}人): {_labels(ledger, alive, names_only=names_only)}")

    dead = [ledger.player(pid) for pid in ledger.dead_ids()]
    if dead:
        rendered = "、".join(
            f"{label(player.player_id)}[{player.death_day}日目{_death_label(player.death_cause)}]"
            for player in dead
            if player is not None
        )
        lines.append(f"- 死亡者: {rendered}")

    todays_co = [claim for claim in ledger.co_declarations() if claim.day == target_day]
    if todays_co:
        rendered = "、".join(
            f"{label(claim.player_id)}={_role_label(claim.claimed_role)}" for claim in todays_co
        )
        lines.append(f"- 本日のCO: {rendered}")

    # Dated as `ObservationSet` dates them: a verdict said again on a later day
    # without naming a night belongs to the day it was first published, so it
    # is neither listed as today's nor lost from its own day's summary.
    todays_results = [
        verdict
        for verdict in (
            _dated_by_first_statement(ledger, result) for result in ledger.public_results()
        )
        if verdict.day == target_day
    ]
    if todays_results:
        rendered = "、".join(
            f"{label(result.claimant_id)}の"
            f"{'霊媒' if result.result_type == MEDIUM_RESULT else '占い'}: "
            f"{label(result.target_id)}={'黒' if result.is_werewolf else '白'}"
            for result in todays_results
        )
        lines.append(f"- 本日公開された判定: {rendered}")

    for round_number in sorted({vote.round for vote in ledger.votes_on(target_day)}):
        rendered = "、".join(
            f"{label(vote.voter_id)}→{label(vote.target_id)}"
            for vote in ledger.votes_on(target_day, round_number)
        )
        lines.append(f"- 投票R{round_number}: {rendered}")

    executed_today = [
        execution for execution in ledger.executions() if execution.day == target_day
    ]
    if executed_today:
        rendered = "、".join(label(execution.player_id) for execution in executed_today)
        lines.append(f"- 処刑結果: {rendered}")

    night_deaths = ledger.night_death_ids(target_day)
    if night_deaths:
        lines.append(f"- 夜の死亡: {_labels(ledger, night_deaths, names_only=names_only)}")

    return "\n".join(lines)


def render_day_record(
    state: GameState, day: int, direction: Sequence[tuple[str, str]] = ()
) -> str:
    """v3: one past day as it happened, for the 【過去日の要約】 layer.

    The board memo already holds the living, the dead, every CO and verdict
    and every ballot, so this block repeats none of them -- a living list here
    was frozen at the day's vote and went stale overnight. It holds what the
    memo does not: the night before (published that morning), the lines the
    day was steered by (`direction`: label and message id, in order), the
    vote spread with the execution, and the executed seat's last line. Lines
    are quoted as they were said. Public facts only: the same for every seat.
    """
    ledger = PublicFactLedger(state)
    name = ledger.name_of
    public = {
        message.message_id: message
        for message in state.chat_log
        if message.channel is ChatChannel.PUBLIC
    }
    lines = [f"{FACTS_HEADING}{day}日目", f"- {_night_line(ledger, day - 1)}"]
    for label, message_id in direction:
        message = public.get(message_id)
        if message is not None:
            lines.append(
                f"- {label}（{name(message.author_id)} {message_id}）:{_quote(message.content)}"
            )
    executed = [execution.player_id for execution in ledger.executions() if execution.day == day]
    rounds = sorted({vote.round for vote in ledger.votes_on(day)})
    if rounds:
        spreads = [_vote_spread(ledger, day, round_number) for round_number in rounds]
        tally = "（同数）→ 決選: ".join(spreads)
        outcome = "、".join(name(pid) for pid in executed) or "なし"
        lines.append(f"- 投票: {tally} → 処刑 {outcome}")
    for player_id in executed:
        last = next(
            (
                message
                for message in reversed(public.values())
                if message.author_id == player_id and message.day == day
            ),
            None,
        )
        if last is None:
            lines.append(f"- 処刑された{name(player_id)}はこの日発言していない")
        else:
            lines.append(
                f"- 処刑された{name(player_id)}の最後の発言（{last.message_id}）:"
                f"{_quote(last.content)}"
            )
    return "\n".join(lines)


def day_results_and_claims(ledger: PublicFactLedger, day: int) -> list[str]:
    """v3: the COs made and the verdicts first published on `day`, by role.

    A verdict is dated by the night it is about and the day it came out --
    "1日目夜の結果（2日目公開）" -- and seer and medium results are kept on
    separate lines: written alike, a medium's read of an execution looked
    like one more divination. Ids are left in; the caller writes names.
    """
    points: list[str] = []
    claims: dict[RoleName, list[str]] = {}
    # From the claim history, not the standing claims: a CO said again on a
    # later day is the same CO, not a new one that day.
    for claim in ledger.role_claim_history():
        if claim.day == day and claim.role is not None and not claim.reaffirmed:
            claimants = claims.setdefault(claim.role, [])
            if claim.player_id not in claimants:
                claimants.append(claim.player_id)
    if claims:
        ordered = [role for role in _CLAIM_ORDER if role in claims] + [
            role for role in claims if role not in _CLAIM_ORDER
        ]
        points.append(
            "CO: "
            + " / ".join(f"{role_label(role)}={'、'.join(claims[role])}" for role in ordered)
        )
    verdicts = [
        verdict
        for verdict in (
            _dated_by_first_statement(ledger, result) for result in ledger.public_results()
        )
        if verdict.day == day
    ]
    for result_type, label, event in ((SEER_RESULT, "占い", "夜"), (MEDIUM_RESULT, "霊媒", "処刑")):
        by_night: dict[int, list[str]] = {}
        for verdict in verdicts:
            if verdict.result_type == result_type:
                by_night.setdefault(verdict.source_night, []).append(
                    f"{verdict.claimant_id}→{verdict.target_id}="
                    f"{'黒' if verdict.is_werewolf else '白'}"
                )
        for night in sorted(by_night):
            points.append(
                f"{label} {night}日目{event}の結果（{day}日目公開）: {'、'.join(by_night[night])}"
            )
    return points


def _night_line(ledger: PublicFactLedger, night: int) -> str:
    """What the morning after `night` showed: its bodies, or none."""
    heading = f"{night}日目夜の結果（{night + 1}日目公開）: "
    deaths = [ledger.name_of(pid) for pid in ledger.night_death_ids(night)]
    if night == 0 and ledger.first_victim_id is not None:
        others = f"。ほかに死体: {'、'.join(deaths)}" if deaths else ""
        return f"{heading}初日犠牲者{ledger.name_of(ledger.first_victim_id)}{others}"
    if deaths:
        return f"{heading}{'、'.join(deaths)}が死亡"
    return f"{heading}死体なし"


def _vote_spread(ledger: PublicFactLedger, day: int, round_number: int) -> str:
    counts: dict[str, int] = {}
    for vote in ledger.votes_on(day, round_number):
        counts[vote.target_id] = counts.get(vote.target_id, 0) + 1
    seating = {pid: index for index, pid in enumerate(ledger.known_player_ids())}
    ranked = sorted(counts, key=lambda pid: (-counts[pid], seating.get(pid, len(seating))))
    return "、".join(f"{ledger.name_of(pid)}{counts[pid]}票" for pid in ranked)


def _quote(text: str) -> str:
    """A line as said, on one line, inside 「」 (its own 「」 become 『』)."""
    flat = " ".join(text.split()).translate(_NESTED_QUOTES)
    if len(flat) > _QUOTE_MAX_CHARS:
        flat = flat[: _QUOTE_MAX_CHARS - 1] + "…"
    return f"「{flat}」"


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


def _labels(
    ledger: PublicFactLedger, player_ids: tuple[str, ...], *, names_only: bool = False
) -> str:
    label = ledger.name_of if names_only else ledger.label_of
    return "、".join(label(pid) for pid in player_ids) or "なし"


def _death_label(cause: PublicDeathCause | None) -> str:
    return _DEATH_LABELS.get(cause, "死亡") if cause is not None else "死亡"


def _role_label(role: RoleName) -> str:
    return _ROLE_LABELS.get(role, role.value)
