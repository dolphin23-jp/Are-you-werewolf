"""The game record, written out in full.

`transcript-seedN.md` is the record a person reads after a run, and the live
log is what they watch during it. Both used to leave things out: the live log
cut every line at 90 characters and printed votes and night actions without
their reasons, and the record had no private notes, no message ids, no stage,
no divination or medium results, no vote tallies and no audit verdicts, and it
named targets by seat id. Reading a game then meant opening the JSON.

This renders everything a transcript holds, in the order the table saw it,
with every seat named. Nothing here is shortened; what a seat wrote is quoted
whole. It reads only the transcript, so a record saved by an older run renders
the same way, with whatever that run recorded.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from app.engine.roles import ROLE_DEFINITIONS, RoleName
from app.engine.speech_events import (
    RoleClaimState,
    RoleClaimStatus,
    SpeechEvent,
    SpeechEventType,
    role_claim_history,
)
from app.eval.public_record import describe_record, undeclared_records
from app.eval.reasoning_analyzer import AllyVoteKind, audit_ballots
from app.eval.transcript import DecisionAuditRecord, GameTranscript, Utterance

_DEATH_CAUSE = {
    "first_victim": "初日犠牲者",
    "executed": "処刑",
    "attacked": "襲撃",
    "cursed": "呪殺",
    "night_death": "夜の死亡",
}

# The factions as the table says them. The one table every display of a
# faction or a winner reads: the record, the report's per-game table and the
# live log of `scripts/evaluate.py`.
TEAM_LABELS: dict[str, str] = {"village": "村人陣営", "werewolf": "人狼陣営", "fox": "妖狐陣営"}


def team_label(team: object) -> str:
    """A faction id as the table says it: `fox` -> 妖狐陣営. Anything else as it is."""
    return TEAM_LABELS.get(str(team), str(team))

_NIGHT_ACTION = {
    RoleName.SEER.value: "占い",
    RoleName.HUNTER.value: "護衛",
    RoleName.WEREWOLF.value: "襲撃",
}

# Each seat's skill (`app/ai/reasoning/skill.py`), as the cast table shows it.
_SKILL_LABELS = {"expert": "熟練", "average": "普通", "novice": "初心者"}

_CHANGE_KIND = {
    "none": "変更なし",
    "new_public_evidence": "新しい公開情報による変更",
    "target_became_invalid": "吊り先が投票できなくなった",
    "runoff_restriction": "決選の候補の制限",
    "public_result_changed": "公開結果が変わった",
    "correction_accepted": "訂正を受け入れた",
    "strategic_team_decision": "陣営の方針",
    "public_argument": "公開の論による変更",
    "grey_random_policy": "進行役のグレラン宣言に沿った変更（グレー外からグレーへ）",
    "final_designation": "進行役の最後の指定に沿った変更",
    "unexplained": "理由のない変更",
    # Compared with nothing: no turn that day, or a memo that named nobody.
    "no_basis": "比較できない",
    "memo_target_unspoken": "卓で言っていないメモの吊り先からの変更（合否に数えない）",
    "memo_target_only_reported_or_asked": (
        "メモの吊り先に自分の結果の発表か本人への質問でだけ触れた変更（合否に数えない）"
    ),
}

# A discussion utterance's private note, field by field, in reading order.
_MEMO_FIELDS: tuple[tuple[str, str], ...] = (
    ("overall_thought", "方針"),
    ("execution_target", "吊り先"),
    ("suspects", "疑い"),
    ("trusted", "信用"),
    ("trusted_seer", "真と見る占い"),
    ("fox_candidates", "妖狐候補"),
    ("role_hypotheses", "役職の見立て"),
    ("private_team_thought", "陣営内の方針"),
)


def role_label(role: str | None) -> str:
    """A role id as the table says it: `seer` -> 占い師."""
    if not role:
        return "?"
    try:
        return ROLE_DEFINITIONS[RoleName(role)].label_ja
    except (ValueError, KeyError):
        return role


def seat_label(names: Mapping[str, str], seat: str | None) -> str:
    """A seat by name, with its id: 'ハルト(p2)'. Unknown ids stay ids."""
    if not seat:
        return "?"
    name = names.get(seat)
    return f"{name}({seat})" if name else seat


def _seat_list(names: Mapping[str, str], seats: Iterable[Any]) -> str:
    return "、".join(seat_label(names, str(seat)) for seat in seats if seat)


def render_memo(memo: Mapping[str, Any] | None, names: Mapping[str, str]) -> list[str]:
    """Every filled field of a private note, one per line, seats named."""
    if not memo:
        return []
    lines: list[str] = []
    seen: set[str] = set()
    for key, label in _MEMO_FIELDS:
        seen.add(key)
        value = memo.get(key)
        if value in (None, "", [], {}):
            continue
        lines.append(f"{label}: {_memo_value(key, value, names)}")
    for key, value in memo.items():
        # A field added to the note later still shows, under its own name.
        if key in seen or value in (None, "", [], {}):
            continue
        lines.append(f"{key}: {_memo_value(key, value, names)}")
    return lines


def _memo_value(key: str, value: Any, names: Mapping[str, str]) -> str:
    if key in ("execution_target", "trusted_seer") and isinstance(value, str):
        return seat_label(names, value)
    if key in ("suspects", "trusted", "fox_candidates") and isinstance(value, list):
        return _seat_list(names, value)
    if key == "role_hypotheses" and isinstance(value, list):
        return " / ".join(_hypothesis(item, names) for item in value)
    if isinstance(value, list):
        return "、".join(str(item) for item in value)
    return str(value)


def _hypothesis(item: Any, names: Mapping[str, str]) -> str:
    if not isinstance(item, Mapping):
        return str(item)
    seat = seat_label(names, str(item.get("player_id") or ""))
    role = role_label(str(item.get("role") or "")) if item.get("role") else "?"
    rest = {
        key: value
        for key, value in item.items()
        if key not in ("player_id", "role") and value not in (None, "", [])
    }
    extra = "".join(f" {key}={value}" for key, value in rest.items())
    return f"{seat}={role}{extra}"


def _flat(text: str) -> str:
    """One line, nothing removed: newlines become ' / '."""
    return " / ".join(part.strip() for part in text.splitlines() if part.strip()) or text


def render_transcript_full(transcript: GameTranscript, *, partial: bool = False) -> str:
    """The whole game as one readable page. See the module docstring."""
    return "\n".join(_Renderer(transcript, partial=partial).render()) + "\n"


class _Renderer:
    def __init__(self, transcript: GameTranscript, *, partial: bool) -> None:
        self.t = transcript
        self.partial = partial
        self.names = transcript.names
        state = transcript.final_state or {}
        self.state = state
        self.message_ids = _match_message_ids(transcript.utterances, state.get("chat_log", []))
        self.vote_rounds = _assign_vote_rounds(transcript.utterances, state.get("vote_records", []))
        self.ballot_audits = _ballot_audits(transcript.decision_audits)
        self.verdicts = {
            (verdict.player_id, verdict.day, verdict.vote_round): verdict
            for verdict in audit_ballots(transcript)
        }
        self.plans = {
            (plan.voter_id, plan.day, plan.round_number): plan
            for plan in transcript.wolf_ally_vote_plans
        }
        self.claim_history = role_claim_history(_speech_events(state))
        # Claims, results and partner namings their own line did not declare.
        self.undeclared = {record.event_id: record for record in undeclared_records(transcript)}
        self.claims_by_message: dict[str, list[RoleClaimState]] = defaultdict(list)
        for claim in self.claim_history:
            self.claims_by_message[claim.source_message_id].append(claim)

    # -- the page --

    def render(self) -> list[str]:
        lines = [f"# 対戦記録 (seed={self.t.seed})\n"]
        if self.partial:
            lines.append(
                "> **途中経過です。** 対戦はまだ終わっていません。"
                "実行が止まった場合は、ここまでが残った分です。\n"
            )
        lines.append(
            "> 省略なしの記録です。各発言の非公開メモ（そのときの吊り先・疑い・方針）、"
            "投票の理由・決め手・反対仮説、夜行動の理由と結果、監査の判定を、"
            "記録にあるとおりに載せています。`m12` はその発言のメッセージIDです。"
            "夜行動の「実際の結果」は記録から書いた神の視点で、その席に知らされた内容とは"
            "限りません（狩人は護衛先が襲われたかを知らされず、人狼は噛みが通らなかった"
            "ことしか分かりません）。\n"
        )
        lines += self._roster()
        lines += self._timeline()
        lines += self._system_records()
        lines += self._result()
        return lines

    def _roster(self) -> list[str]:
        lines = [
            "## 配役\n",
            "| プレイヤー | 役職 | 陣営 | 人格 | 腕前 | 騙り役 |",
            "|---|---|---|---|---|---|",
        ]
        deception = self.t.deception or {}
        fake_roles = deception.get("fake_role_by_player", {})
        lurkers = set(deception.get("lurking_player_ids", []))
        for pid, name in self.names.items():
            fake = fake_roles.get(pid)
            label = f"偽{role_label(fake)}" if fake else ("潜伏" if pid in lurkers else "-")
            team = team_label(self.t.teams.get(pid, "-"))
            lines.append(
                f"| {name} (`{pid}`) | {role_label(self.t.roles.get(pid))} | {team} | "
                f"{self.t.personalities.get(pid, '-')} | "
                f"{_SKILL_LABELS.get(self.t.skills.get(pid, ''), '-')} | {label} |"
            )
        lines.append(f"\n人狼陣営の方針: {deception.get('wolf_pattern_label', '-')}\n")
        return lines

    def _timeline(self) -> list[str]:
        lines: list[str] = []
        current_day: int | None = None
        section: str | None = None
        vote_day: int | None = None
        for index, utterance in enumerate(self.t.utterances):
            if utterance.day != current_day:
                if vote_day is not None:
                    lines += self._vote_outcome(vote_day)
                    vote_day = None
                current_day = utterance.day
                section = None
                lines.append(f"\n## {utterance.day}日目\n")
                lines += self._morning(utterance.day)
            group = _section_of(utterance)
            if group != section:
                if vote_day is not None and group != "vote":
                    lines += self._vote_outcome(vote_day)
                    vote_day = None
                section = group
                lines.append(f"\n### {_SECTION_TITLE[group]}\n")
            if group == "vote":
                vote_day = utterance.day
            lines += self._utterance(index, utterance)
        if vote_day is not None:
            lines += self._vote_outcome(vote_day)
        return lines

    def _morning(self, day: int) -> list[str]:
        """Who was found dead this morning: the deaths of the night before."""
        if day < 1:
            return []
        deaths = [
            death
            for death in self.state.get("death_records", [])
            if death.get("day") == day - 1 and death.get("cause") != "executed"
        ]
        if not deaths:
            return ["- 朝: 死体なし"]
        return [
            "- 朝の死体: "
            + "、".join(
                f"{seat_label(self.names, death['player_id'])}"
                f"（{role_label(self.t.roles.get(death['player_id']))}・"
                f"{_DEATH_CAUSE.get(death.get('cause', ''), death.get('cause', ''))}）"
                for death in deaths
            )
        ]

    # -- one utterance --

    def _utterance(self, index: int, u: Utterance) -> list[str]:
        who = f"{u.player_name}（{role_label(u.role)}／{u.personality}）"
        if u.kind == "discussion":
            return self._discussion(index, u, who)
        if u.kind == "vote":
            return self._vote(index, u, who)
        if u.kind == "vote_change":
            return [f"  - 投票の記録（{u.player_name}）: {_flat(u.text)}"]
        if u.kind == "night_action":
            return self._night_action(u, who)
        if u.kind in ("wolf_chat", "freemason_chat"):
            channel = "人狼チャット" if u.kind == "wolf_chat" else "共有チャット"
            mid = self.message_ids.get(index)
            prefix = f"`{mid}` " if mid else ""
            return [f"- {prefix}[{channel}] **{who}**: {_flat(u.text)}"]
        if u.kind == "summary":
            # A summary written in code has no author at the table; the seat it
            # is filed under did not write it.
            author = "コードが作成" if u.stage == "code" else f"{u.player_name}が作成"
            return [f"- 要約（{author}）: {_flat(u.text)}"]
        return [f"- [{u.kind}] {who}: {_flat(u.text)}"]

    def _discussion(self, index: int, u: Utterance, who: str) -> list[str]:
        mid = self.message_ids.get(index)
        stage = f"[{u.stage}] " if u.stage else ""
        flags = []
        if u.used_fallback:
            flags.append("フォールバック")
        if u.model_text_chars == 0:
            flags.append("コードが書いた行")
        flag = f" [{'・'.join(flags)}]" if flags else ""
        head = f"`{mid}` " if mid else ""
        lines = [f"- {head}{stage}**{who}**{flag}: {_flat(u.text)}"]
        detail: list[str] = []
        if u.reply_to:
            detail.append(f"返信先: {u.reply_to}")
        if u.directed_question_targets:
            detail.append(f"名指しの質問: {_seat_list(self.names, u.directed_question_targets)}")
        if u.agrees_with:
            detail.append(f"同意: {_seat_list(self.names, u.agrees_with)}")
        if u.public_claim_role:
            claim = self._co_label(mid, u.public_claim_role)
            detail.append(f"{claim}: {role_label(u.public_claim_role)}")
        for result in u.public_results:
            detail.append("公開した結果: " + _result_text(result, self.names))
        if u.key_point:
            detail.append(f"要点: {u.key_point}")
        if u.ready_to_vote:
            detail.append("投票へ進んでよい")
        if u.model_text_chars is not None or u.effective_length_limit is not None:
            model = "?" if u.model_text_chars is None else u.model_text_chars
            limit = "-" if u.effective_length_limit is None else u.effective_length_limit
            detail.append(f"字数: 全体{len(u.text)}字・モデル{model}字（上限{limit}字）")
        if detail:
            lines.append("  - " + " ／ ".join(detail))
        memo = render_memo(u.reasoning_memo, self.names)
        if memo:
            lines.append("  - 非公開メモ:")
            lines += [f"    - {line}" for line in memo]
        return lines

    def _co_label(self, message_id: str | None, role: str) -> str:
        """「CO」, or 「CO（再表明）」 when this line only said an earlier claim again.

        A restatement is not a new claim: the claim stands from the day it was
        first made (`speech_events.current_role_claim`), and the record says so
        rather than presenting a second CO.
        """
        claims = [
            claim
            for claim in self.claims_by_message.get(message_id or "", [])
            if claim.role is not None and claim.role.value == role
        ]
        if claims and all(claim.reaffirmed for claim in claims):
            return "CO（再表明）"
        return "CO"

    def _vote(self, index: int, u: Utterance, who: str) -> list[str]:
        lines = [f"- **{who}** → {seat_label(self.names, u.target)}"]
        if u.text:
            lines.append(f"  - 理由: {_flat(u.text)}")
        if u.decision_evidence:
            lines.append(f"  - 決め手: {_flat(u.decision_evidence)}")
        if u.countercase:
            lines.append(f"  - 反対仮説: {_flat(u.countercase)}")
        if u.alternative_target:
            lines.append(f"  - 次点: {seat_label(self.names, u.alternative_target)}")
        round_number = self.vote_rounds.get(index)
        audit = self.ballot_audits.get((u.player_id, u.day, round_number or 0)) or (
            self.ballot_audits.get((u.player_id, u.day, 0))
        )
        if audit is not None:
            lines += self._audit_lines(audit, round_number)
        return lines

    def _audit_lines(self, audit: DecisionAuditRecord, round_number: int | None) -> list[str]:
        """The ballot's audit as the analyzer judges it now (`audit_ballots`).

        Not the classification recorded live: a record written before the
        audit's rules were fixed would otherwise keep printing a ballot with
        nothing to compare against as 「変更なし」.
        """
        parts: list[str] = []
        verdict = self.verdicts.get((audit.player_id, audit.day, audit.vote_round))
        basis_id = verdict.basis if verdict is not None else audit.decision_target
        if basis_id:
            basis = seat_label(self.names, basis_id)
            parts.append(f"比較の基準（メモの吊り先）: {basis}")
        if audit.model_target_was_overridden:
            parts.append(
                f"モデルが書いた吊り先 {seat_label(self.names, audit.model_requested_target)} "
                "をコードが差し替え"
            )
        kind = verdict.kind.value if verdict is not None else audit.target_change_classification
        why = verdict.why if verdict is not None else audit.target_change_reason
        if kind:
            text = _CHANGE_KIND.get(kind, kind)
            if why:
                text += f"（{why}）"
            parts.append(f"判定: {text}")
        if audit.ally_vote:
            plan = self.plans.get((audit.player_id, audit.day, round_number or 1))
            planned = "計画あり" if audit.ally_vote_planned else "計画なし"
            if plan is not None:
                planned += f"（{plan.basis}: {_flat(plan.story_reason)}）"
            elif verdict is not None and verdict.ally_kind is AllyVoteKind.DESIGNATED:
                planned += "・本指定どおりで、この票を除いても処刑は同じ（合否に数えない）"
            parts.append(f"仲間への投票: {planned}")
        if verdict is not None and verdict.reason_names_other:
            others = _seat_list(self.names, verdict.reason_names_other)
            parts.append(f"理由が別の相手（{others}）への投票を述べている")
        return [f"  - 監査: {' ／ '.join(parts)}"] if parts else []

    def _vote_outcome(self, day: int) -> list[str]:
        records = [vote for vote in self.state.get("vote_records", []) if vote.get("day") == day]
        lines: list[str] = []
        for round_number in sorted({vote.get("round", 1) for vote in records}):
            tally = Counter(
                vote["target_id"] for vote in records if vote.get("round", 1) == round_number
            )
            label = "1回目" if round_number == 1 else f"決選{round_number - 1}"
            text = " / ".join(
                f"{seat_label(self.names, target)} {count}票"
                for target, count in tally.most_common()
            )
            lines.append(f"- 集計（{label}）: {text}")
        executed = [
            death
            for death in self.state.get("death_records", [])
            if death.get("day") == day and death.get("cause") == "executed"
        ]
        for death in executed:
            pid = death["player_id"]
            role = role_label(self.t.roles.get(pid))
            lines.append(f"- 処刑: {seat_label(self.names, pid)}（{role}）")
        for record in self.state.get("medium_records", []):
            if record.get("day") == day:
                colour = "人狼" if record.get("is_werewolf") else "人狼ではない"
                medium = seat_label(self.names, record.get("medium_id"))
                target = seat_label(self.names, record.get("target_id"))
                lines.append(f"- 霊媒の結果（{medium}が翌朝に知る）: {target}は{colour}")
        return lines

    def _night_action(self, u: Utterance, who: str) -> list[str]:
        action = _NIGHT_ACTION.get(u.role, "夜行動")
        lines = [f"- **{who}** {action} → {seat_label(self.names, u.target)}"]
        if u.text:
            lines.append(f"  - 理由: {_flat(u.text)}")
        outcome = self._night_outcome(u)
        if outcome:
            # A seer is told its result; a hunter is never told whether its
            # target was attacked, and wolves learn only that a bite failed.
            # Labelled so a reader does not take the outcome for the seat's
            # knowledge.
            label = (
                "結果" if u.role == RoleName.SEER.value else "実際の結果（本人には知らされない）"
            )
            lines.append(f"  - {label}: {outcome}")
        return lines

    def _night_outcome(self, u: Utterance) -> str:
        if u.role == RoleName.SEER.value:
            for record in self.state.get("divine_records", []):
                if (
                    record.get("seer_id") == u.player_id
                    and record.get("day") == u.day
                    and record.get("target_id") == u.target
                ):
                    colour = "黒（人狼）" if record.get("is_werewolf") else "白（人狼ではない）"
                    cursed = any(
                        death.get("player_id") == u.target
                        and death.get("day") == u.day
                        and death.get("cause") in ("cursed", "night_death")
                        and self.t.roles.get(str(u.target)) == RoleName.FOX.value
                        for death in self.state.get("death_records", [])
                    )
                    return colour + ("。妖狐なので呪殺" if cursed else "")
            return "記録なし"
        if u.role == RoleName.WEREWOLF.value:
            guarded = {
                (record.get("day"), record.get("target_id"))
                for record in self.state.get("guard_records", [])
            }
            for attack in self.state.get("attack_records", []):
                if attack.get("day") == u.day and attack.get("wolf_id") == u.player_id:
                    target = attack.get("target_id")
                    if attack.get("succeeded"):
                        return f"{seat_label(self.names, target)}を襲撃して成功"
                    why = "護衛された" if (u.day, target) in guarded else "妖狐なので死なない"
                    return f"{seat_label(self.names, target)}への襲撃は失敗（{why}）"
            return ""
        if u.role == RoleName.HUNTER.value:
            attacked = {
                (attack.get("day"), attack.get("target_id"))
                for attack in self.state.get("attack_records", [])
            }
            if (u.day, u.target) in attacked:
                return "護衛先が襲撃され、守った"
            return "護衛先は襲撃されなかった"
        return ""

    # -- the end --

    def _system_records(self) -> list[str]:
        lines = ["\n## システム記録\n"]
        names = self.names
        for death in self.state.get("death_records", []):
            pid = death["player_id"]
            role = role_label(self.t.roles.get(pid))
            cause = _DEATH_CAUSE.get(death["cause"], death["cause"])
            lines.append(f"- {death['day']}日目: {seat_label(names, pid)}（{role}）が{cause}で死亡")
        guards = {
            (record["day"], record["target_id"]) for record in self.state.get("guard_records", [])
        }
        for attack in self.state.get("attack_records", []):
            target = attack["target_id"]
            guarded = (attack["day"], target) in guards
            outcome = "護衛され失敗" if guarded else ("成功" if attack["succeeded"] else "失敗")
            lines.append(
                f"- {attack['day']}日目夜: {seat_label(names, attack['wolf_id'])}が"
                f"{seat_label(names, target)}を襲撃 → {outcome}"
            )
        for record in self.state.get("divine_records", []):
            colour = "黒" if record.get("is_werewolf") else "白"
            lines.append(
                f"- {record['day']}日目夜の占い: {seat_label(names, record['seer_id'])} → "
                f"{seat_label(names, record['target_id'])} = {colour}"
            )
        for record in self.state.get("guard_records", []):
            lines.append(
                f"- {record['day']}日目夜の護衛: {seat_label(names, record['hunter_id'])} → "
                f"{seat_label(names, record['target_id'])}"
            )
        lines += self._claim_records()
        for record in self.undeclared.values():
            counted = "" if record.changed_record else "・記録は変わらず、合否に数えない"
            lines.append(
                f"- {record.day}日目: 本人が宣言していない記録"
                f"（{record.source_message_id or '-'}{counted}）: "
                f"{describe_record(record, names)}"
            )
        audits = [
            audit
            for audit in self.t.decision_audits
            if audit.vote_target is None
            and (audit.model_target_was_overridden or audit.target_change_classification)
        ]
        if audits:
            lines.append("\n### 議論の手番の監査（票以外）\n")
            for audit in audits:
                parts = [f"{audit.day}日目 {seat_label(names, audit.player_id)}"]
                if audit.model_target_was_overridden:
                    parts.append(
                        f"メモの吊り先 {seat_label(names, audit.model_requested_target)} を "
                        f"{seat_label(names, audit.decision_target)} に差し替え"
                    )
                kind = audit.target_change_classification
                if kind:
                    parts.append(f"判定: {_CHANGE_KIND.get(kind, kind)}")
                lines.append("- " + " ／ ".join(parts))
        return lines

    def _claim_records(self) -> list[str]:
        """Every claim transition in the order it was spoken, restatements labelled.

        Read from the speech-event log, so a claim said again keeps the day it
        was first made. A record written without that log falls back to the
        final CO list it carries.
        """
        names = self.names
        if not self.claim_history:
            return [
                f"- {claim['day']}日目のCO: {seat_label(names, claim['player_id'])} = "
                f"{role_label(claim.get('claimed_role'))}"
                f"（{claim.get('source_message_id') or '-'}）"
                for claim in self.state.get("co_declarations", [])
            ]
        lines: list[str] = []
        for claim in self.claim_history:
            seat = seat_label(names, claim.player_id)
            source = claim.source_message_id or "-"
            if claim.status is RoleClaimStatus.RETRACTED:
                lines.append(f"- {claim.day}日目のCO撤回: {seat}（{source}）")
                continue
            role = role_label(claim.role.value if claim.role is not None else None)
            if claim.reaffirmed:
                kind = "CO（再表明）"
            elif claim.status is RoleClaimStatus.SWITCHED and claim.previous_role is not None:
                kind = f"CO（{role_label(claim.previous_role.value)}から乗り換え）"
            else:
                kind = "CO"
            undeclared = "【本人の宣言なし】" if claim.event_id in self.undeclared else ""
            lines.append(f"- {claim.day}日目の{kind}: {seat} = {role}（{source}）{undeclared}")
        return lines

    def _result(self) -> list[str]:
        if self.partial:
            return ["\n## 結果\n\n(対戦は未完了)"]
        winner = self.state.get("winner")
        reason = self.state.get("victory_reason", "")
        return [f"\n## 結果\n\n勝者: {team_label(winner)}（{winner}） / {reason}"]


_SECTION_TITLE = {
    "discussion": "昼の議論",
    "vote": "投票",
    "night": "夜",
    "summary": "要約",
    "other": "その他",
}


def _section_of(u: Utterance) -> str:
    if u.kind == "discussion":
        return "discussion"
    if u.kind in ("vote", "vote_change"):
        return "vote"
    if u.kind in ("night_action", "wolf_chat", "freemason_chat"):
        return "night"
    if u.kind == "summary":
        return "summary"
    return "other"


def _result_text(result: Mapping[str, Any], names: Mapping[str, str]) -> str:
    kind = {"seer": "占い", "medium": "霊媒"}.get(str(result.get("result_type")), "結果")
    colour = "黒" if result.get("is_werewolf") else "白"
    night = result.get("referenced_day")
    when = f"（{night}日目夜）" if night is not None else ""
    return f"{kind} {seat_label(names, str(result.get('target_id') or ''))}={colour}{when}"


def _speech_events(state: Mapping[str, Any]) -> list[SpeechEvent]:
    """The final state's speech-event log, back as events; [] when it has none."""
    events: list[SpeechEvent] = []
    for raw in state.get("speech_events", []) or []:
        try:
            events.append(
                SpeechEvent(
                    event_id=str(raw["event_id"]),
                    source_message_id=str(raw.get("source_message_id") or ""),
                    actor_id=str(raw["actor_id"]),
                    event_type=SpeechEventType(raw["event_type"]),
                    day=int(raw["day"]),
                    target_id=raw.get("target_id"),
                    role=RoleName(raw["role"]) if raw.get("role") else None,
                    result_is_werewolf=raw.get("result_is_werewolf"),
                    referenced_day=raw.get("referenced_day"),
                    confidence=float(raw.get("confidence", 1.0)),
                )
            )
        except (KeyError, ValueError):
            continue
    return events


def _match_message_ids(
    utterances: Sequence[Utterance], chat_log: Sequence[Mapping[str, Any]]
) -> dict[int, str]:
    """Each spoken utterance's message id, found in the engine's chat log.

    An utterance is recorded before its line is posted, so it does not carry
    the id itself. It is the next unclaimed message on the same channel, by the
    same seat, on the same day, with the same text.
    """
    channel_of = {"discussion": "public", "wolf_chat": "wolf", "freemason_chat": "freemason"}
    pool: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for message in chat_log:
        key = (
            str(message.get("channel")),
            str(message.get("author_id")),
            int(message.get("day", -1)),
        )
        pool[key].append(message)
    found: dict[int, str] = {}
    for index, utterance in enumerate(utterances):
        channel = channel_of.get(utterance.kind)
        if channel is None:
            continue
        candidates = pool.get((channel, utterance.player_id, utterance.day), [])
        match = next(
            (m for m in candidates if m.get("content") == utterance.text),
            None,
        )
        if match is None:
            continue
        candidates.remove(match)
        found[index] = str(match.get("message_id"))
    return found


def _assign_vote_rounds(
    utterances: Sequence[Utterance], vote_records: Sequence[Mapping[str, Any]]
) -> dict[int, int]:
    """Which ballot round each vote utterance belongs to, from the engine's records."""
    remaining: dict[tuple[int, str, str | None], list[int]] = defaultdict(list)
    for record in vote_records:
        key = (int(record.get("day", -1)), str(record.get("voter_id")), record.get("target_id"))
        remaining[key].append(int(record.get("round", 1)))
    rounds: dict[int, int] = {}
    for index, utterance in enumerate(utterances):
        if utterance.kind != "vote":
            continue
        queue = remaining.get((utterance.day, utterance.player_id, utterance.target))
        if queue:
            rounds[index] = queue.pop(0)
    return rounds


def _ballot_audits(
    audits: Sequence[DecisionAuditRecord],
) -> dict[tuple[str, int, int], DecisionAuditRecord]:
    """The audit record of each ballot, by (seat, day, round); the latest wins."""
    found: dict[tuple[str, int, int], DecisionAuditRecord] = {}
    for audit in audits:
        if audit.vote_target is None:
            continue
        found[(audit.player_id, audit.day, audit.vote_round)] = audit
    return found
