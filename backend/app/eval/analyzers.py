"""Rule-based transcript checks -- the metrics that need no LLM judge.

Everything here is decidable from the transcript plus hidden role/deception
assignments, so it is cheap, deterministic and not subject to a judge
model's opinion. Subjective qualities (Japanese naturalness, whether a
persona *feels* consistent) live in `judge.py` instead.

Findings are counts plus concrete examples, because a bare rate is not
actionable -- you need the offending line to know whether it is a real
problem or a false positive of the pattern.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.ai.co_detection import detect_claimed_role
from app.eval.transcript import GameTranscript, Utterance


def _claimed_role(utterance: Utterance, t: GameTranscript) -> str | None:
    """Same detector the engine uses to register a CO, so the evaluation
    can never disagree with what the game actually recorded."""
    if utterance.public_claim_role:
        return utterance.public_claim_role
    others = [name for pid, name in t.names.items() if pid != utterance.player_id]
    role = detect_claimed_role(utterance.text, others)
    return role.value if role is not None else None


_P0_IDENTITY_RE = re.compile(r"(?:私|俺|僕|自分)が(?:本物の)?p0(?:本人)?|p0本人")

# Rough "is this actually Japanese" signal: share of CJK/kana characters.
_JA_CHAR_RE = re.compile(r"[぀-ヿ一-鿿]")

_META_LEAK_RE = re.compile(
    r"(AIとして|言語モデル|アシスタントとして|as an AI|システムプロンプト)", re.IGNORECASE
)


@dataclass
class Finding:
    check: str
    severity: str  # "high" | "medium" | "low"
    player_id: str
    day: int
    detail: str
    text: str = ""


@dataclass
class AnalysisResult:
    findings: list[Finding] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def count(self, check: str) -> int:
        return sum(1 for f in self.findings if f.check == check)


def analyze(transcript: GameTranscript) -> AnalysisResult:
    result = AnalysisResult()
    _check_role_consistency(transcript, result)
    _check_contradictions(transcript, result)
    _check_wolf_deception(transcript, result)
    _check_identity_confusion(transcript, result)
    _check_true_role_result_accuracy(transcript, result)
    _collect_format_stats(transcript, result)
    return result


# -- 自分の役職との整合性 --------------------------------------------------


def _check_role_consistency(t: GameTranscript, result: AnalysisResult) -> None:
    fake_roles = t.deception.get("fake_role_by_player", {})

    for u in t.by_kind("discussion"):
        # Reporting a divination/medium *result* is only legitimate for the
        # real seer/medium, or for a wolf/madman deliberately assigned to
        # fake that role. Anyone else doing it is out of character.
        if u.public_results:
            assigned_fake = fake_roles.get(u.player_id)
            claimed_result_roles = {item.get("result_type") for item in u.public_results}
            legitimate_roles = {u.role, assigned_fake}
            legitimate = claimed_result_roles <= legitimate_roles
            if not legitimate:
                result.add(
                    Finding(
                        check="result_claim_without_role",
                        severity="high",
                        player_id=u.player_id,
                        day=u.day,
                        detail=(
                            f"役職={u.role} / 騙り役={assigned_fake or 'なし'} なのに"
                            "占い・霊媒の結果を主張している"
                        ),
                        text=u.text,
                    )
                )

        claimed_role = _claimed_role(u, t)
        if claimed_role is None:
            continue
        assigned_fake = fake_roles.get(u.player_id)
        if u.role == claimed_role or assigned_fake == claimed_role:
            continue
        # A village-side player claiming someone else's role is a genuine
        # inconsistency; a wolf improvising a claim outside its assigned
        # plan is weaker but still worth surfacing.
        severity = "medium" if u.team == "werewolf" else "high"
        result.add(
            Finding(
                check="co_role_mismatch",
                severity=severity,
                player_id=u.player_id,
                day=u.day,
                detail=f"役職={u.role} だが {claimed_role} をCOしている",
                text=u.text,
            )
        )


# -- 過去発言との矛盾 ------------------------------------------------------


def _check_contradictions(t: GameTranscript, result: AnalysisResult) -> None:
    claimed_roles: dict[str, tuple[str, int]] = {}
    for u in t.by_kind("discussion"):
        claimed_role = _claimed_role(u, t)
        if claimed_role is None:
            continue
        previous = claimed_roles.get(u.player_id)
        if previous is not None and previous[0] != claimed_role:
            result.add(
                Finding(
                    check="co_role_changed",
                    severity="high",
                    player_id=u.player_id,
                    day=u.day,
                    detail=f"{previous[1]}日目に {previous[0]} CO、今回は {claimed_role} CO",
                    text=u.text,
                )
            )
        claimed_roles.setdefault(u.player_id, (claimed_role, u.day))

    # Only structured current targets count. Historical prose about a dead
    # player's execution, vote or medium result is legitimate analysis.
    died_on: dict[str, int] = {
        d["player_id"]: d["day"] for d in t.final_state.get("death_records", [])
    }
    for u in t.by_kind("discussion"):
        memo = u.reasoning_memo or {}
        current_targets = set(u.directed_question_targets)
        execution_target = memo.get("execution_target")
        if isinstance(execution_target, str):
            current_targets.add(execution_target)
        for pid in current_targets:
            death_day = died_on.get(pid)
            if death_day is not None and u.day > death_day:
                result.add(
                    Finding(
                        check="treats_dead_player_as_active",
                        severity="medium",
                        player_id=u.player_id,
                        day=u.day,
                        detail=f"{t.names.get(pid, pid)} は死亡済みだが現在の対象にしている",
                        text=u.text,
                    )
                )

    # The reasoning memo is the model's own stated intent; voting against it
    # the same day is a self-inconsistency signal.
    intent_by_player_day: dict[tuple[str, int], str] = {}
    for u in t.by_kind("discussion"):
        memo = u.reasoning_memo or {}
        target = memo.get("execution_target")
        if isinstance(target, str) and target:
            intent_by_player_day[(u.player_id, u.day)] = target

    for u in t.by_kind("vote"):
        intent = intent_by_player_day.get((u.player_id, u.day))
        if intent and u.target and intent != u.target and intent in t.names:
            result.add(
                Finding(
                    check="vote_contradicts_stated_intent",
                    severity="low",
                    player_id=u.player_id,
                    day=u.day,
                    detail=f"思考メモでは {intent} を処刑したいとしたが {u.target} に投票",
                    text=u.text,
                )
            )


# -- 人狼側の欺瞞と連携 ----------------------------------------------------


def _check_wolf_deception(t: GameTranscript, result: AnalysisResult) -> None:
    wolf_ids = {pid for pid, role in t.roles.items() if role == "werewolf"}
    fake_roles = t.deception.get("fake_role_by_player", {})
    lurkers = set(t.deception.get("lurking_player_ids", []))
    name_by_id = t.names

    # Prefer the structured suspect memo. Text fallback requires the teammate
    # and an accusation to occur in the same short sentence; a remote word
    # such as another player's "black result" must not trigger this check.
    for u in t.by_kind("discussion"):
        if u.player_id not in wolf_ids:
            continue
        for teammate in wolf_ids - {u.player_id}:
            teammate_name = name_by_id.get(teammate, teammate)
            explicit_text = any(
                teammate_name in sentence
                and re.search(r"(人狼だ|狼だ|黒だ|黒い|狼で見る|人狼で見る)", sentence)
                for sentence in re.split(r"[。！？!?\n]", u.text)
            )
            if explicit_text:
                result.add(
                    Finding(
                        check="wolf_named_teammate_with_wolf_word",
                        severity="high",
                        player_id=u.player_id,
                        day=u.day,
                        detail=f"仲間の人狼 {teammate_name} に言及しつつ「人狼/黒」に触れている",
                        text=u.text,
                    )
                )
                break

        memo_suspects = set((u.reasoning_memo or {}).get("suspects", []))
        teammate_suspects = memo_suspects & (wolf_ids - {u.player_id})
        if teammate_suspects:
            teammate = sorted(teammate_suspects)[0]
            result.add(
                Finding(
                    check="wolf_memo_suspects_teammate",
                    severity="low",
                    player_id=u.player_id,
                    day=u.day,
                    detail=f"非公開メモのsuspectsに仲間 {name_by_id.get(teammate, teammate)}",
                    text=f"reasoning_memo.suspects={sorted(memo_suspects)}",
                )
            )

    for u in t.by_kind("vote"):
        if u.player_id in wolf_ids and u.target in wolf_ids:
            result.add(
                Finding(
                    check="wolf_voted_teammate",
                    severity="low",
                    player_id=u.player_id,
                    day=u.day,
                    detail=f"仲間の人狼 {name_by_id.get(u.target or '', u.target)} に投票した",
                    text=u.text,
                )
            )

    def _roles_claimed_by(player_id: str) -> set[str]:
        claimed: set[str] = set()
        for u in t.by_player(player_id):
            if u.kind != "discussion":
                continue
            role = _claimed_role(u, t)
            if role is not None:
                claimed.add(role)
        return claimed

    # Did the pre-committed deception plan actually get executed?
    plan: dict[str, Any] = {}
    for wolf_id, fake_role in fake_roles.items():
        executed = fake_role in _roles_claimed_by(wolf_id)
        plan[wolf_id] = {"assigned": f"fake_{fake_role}", "executed": executed}
        if not executed:
            result.add(
                Finding(
                    check="assigned_faker_never_claimed",
                    severity="medium",
                    player_id=wolf_id,
                    day=0,
                    detail=f"{fake_role} を騙る担当だが一度もCOしなかった",
                )
            )

    for wolf_id in lurkers:
        claimed = sorted(_roles_claimed_by(wolf_id))
        plan[wolf_id] = {"assigned": "lurker", "executed": not claimed}
        if claimed:
            result.add(
                Finding(
                    check="lurker_broke_cover",
                    severity="medium",
                    player_id=wolf_id,
                    day=0,
                    detail=f"潜伏担当だが {', '.join(claimed)} をCOした",
                )
            )

    result.stats["wolf_plan_execution"] = plan
    result.stats["wolf_pattern"] = t.deception.get("wolf_pattern_label")


def _check_identity_confusion(t: GameTranscript, result: AnalysisResult) -> None:
    for u in t.by_kind("discussion"):
        memo = u.reasoning_memo or {}
        structured_targets = set(memo.get("suspects", [])) | set(u.directed_question_targets)
        if memo.get("execution_target"):
            structured_targets.add(memo["execution_target"])
        structured_targets.update(item.get("target_id") for item in u.public_results)
        if u.player_id in structured_targets:
            result.add(
                Finding(
                    check="self_treated_as_other_player",
                    severity="high",
                    player_id=u.player_id,
                    day=u.day,
                    detail="構造化された疑い先・行動対象に自分自身を指定している",
                    text=u.text,
                )
            )

        if u.player_id != "p0" and _P0_IDENTITY_RE.search(u.text):
            result.add(
                Finding(
                    check="claimed_p0_identity",
                    severity="high",
                    player_id=u.player_id,
                    day=u.day,
                    detail="p0ではないプレイヤーが自分をp0本人だと主張している",
                    text=u.text,
                )
            )


def _check_true_role_result_accuracy(t: GameTranscript, result: AnalysisResult) -> None:
    records_by_owner: dict[str, dict[str, tuple[int, bool]]] = {}
    for record in t.final_state.get("divine_records", []):
        records_by_owner.setdefault(record["seer_id"], {})[record["target_id"]] = (
            record["day"],
            record["is_werewolf"],
        )
    for record in t.final_state.get("medium_records", []):
        records_by_owner.setdefault(record["medium_id"], {})[record["target_id"]] = (
            record["day"],
            record["is_werewolf"],
        )

    for u in t.by_kind("discussion"):
        if u.role not in ("seer", "medium"):
            continue
        known = records_by_owner.get(u.player_id, {})
        for claim in u.public_results:
            target_id = claim.get("target_id")
            if not isinstance(target_id, str):
                continue
            actual = known.get(target_id)
            if actual is None:
                continue
            record_day, actual_is_wolf = actual
            if record_day > u.day or claim.get("is_werewolf") == actual_is_wolf:
                continue
            result.add(
                Finding(
                    check="true_role_result_conflict",
                    severity="high",
                    player_id=u.player_id,
                    day=u.day,
                    detail=(
                        f"実際の結果は {target_id}="
                        f"{'人狼' if actual_is_wolf else '人狼ではない'} だが逆の色を主張"
                    ),
                    text=u.text,
                )
            )


# -- 形式・言語の客観指標 --------------------------------------------------


def _collect_format_stats(t: GameTranscript, result: AnalysisResult) -> None:
    speech = [u for u in t.utterances if u.kind in ("discussion", "wolf_chat", "freemason_chat")]
    if not speech:
        result.stats["speech"] = None
        return

    lengths = sorted(len(u.text) for u in speech)
    fallbacks = [u for u in speech if u.used_fallback]

    for u in speech:
        if _META_LEAK_RE.search(u.text):
            result.add(
                Finding(
                    check="meta_phrase_leaked",
                    severity="high",
                    player_id=u.player_id,
                    day=u.day,
                    detail="AIであることを示すメタ発言がフィルタを抜けている",
                    text=u.text,
                )
            )

    low_japanese = [
        u
        for u in speech
        if len(u.text) >= 10 and len(_JA_CHAR_RE.findall(u.text)) / max(len(u.text), 1) < 0.3
    ]
    for u in low_japanese:
        result.add(
            Finding(
                check="low_japanese_ratio",
                severity="medium",
                player_id=u.player_id,
                day=u.day,
                detail="日本語(かな/漢字)の比率が3割未満",
                text=u.text,
            )
        )

    # Verbatim self-repetition reads as broken immersion even when each line
    # is individually fine.
    seen: dict[tuple[str, str], int] = {}
    repeats = 0
    for u in speech:
        key = (u.player_id, u.text.strip())
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 2:
            repeats += 1

    over_limit = [u for u in speech if len(u.text) > 200]

    result.stats["speech"] = {
        "utterances": len(speech),
        "mean_length": round(sum(lengths) / len(lengths), 1),
        "median_length": lengths[len(lengths) // 2],
        "max_length": lengths[-1],
        "over_200_chars": len(over_limit),
        "fallback_lines": len(fallbacks),
        "fallback_rate": round(len(fallbacks) / len(speech), 4),
        "verbatim_repeat_players": repeats,
    }
