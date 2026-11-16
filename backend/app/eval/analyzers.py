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
import statistics
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
# An assertion that someone is a wolf. "黒だけで決められない" is the particle だけ, not
# "黒だ": reading it that way flagged two wolves for merely discussing a black result.
_WOLF_WORD_RE = re.compile(r"(?:人狼だ|狼だ|黒だ)(?!け)|黒い|狼で見る|人狼で見る")

# Rough "is this actually Japanese" signal: share of CJK/kana characters.
_JA_CHAR_RE = re.compile(r"[぀-ヿ一-鿿]")

# Wording the v2 runtime injects or hands the model as fixed phrasing. A human
# at a chat table never opens every line with "現時点の第一処刑候補は", so the
# share of lines carrying any of these is the most direct measure of how much
# of what the table hears was written by code rather than said by a player.
_TEMPLATE_PHRASE_RE = re.compile(
    r"現時点の第一処刑候補は"
    r"|という見方も残"
    r"|処刑しない最強の理由"
    r"|独立した根拠"
    r"|反対材料"
    r"|第二候補は"
    # The v3 prompt's own checklist words until 2026-11-16 (the first Claude
    # game repeated them back); kept so a regression to recited labels shows.
    r"|反対仮説"
    r"|二番手"
    r"|吊らない一番の理由"
)
# The canonical fact sentences `ensure_fact_sentences` (app/ai/reasoning/claims.py)
# prepends when the prose does not state a declared CO or result -- the
# `render_*_sentence` wording, at the very start of the line. The transcript
# does not record what the code added, so this reads the wording: the labels
# are the code's (`占いCO。`, where a player writes 占い師CO), and a verdict is
# spelled `黒(人狼)` / `白(人狼ではない)`. In the first Claude game these opened 11
# of 77 lines, one of them announcing the same result twice; the template rate
# counted none of them.
_CLAIM_LABEL = r"(?:占い|霊媒|狩人|共有)"
_VERDICT = r"(?:黒\(人狼\)|白\(人狼ではない\))"
_CODE_FACT_SENTENCE_RE = re.compile(
    rf"(?:{_CLAIM_LABEL}CO。"
    rf"|{_CLAIM_LABEL}COします。"
    rf"|先ほどの{_CLAIM_LABEL}COを撤回します。"
    rf"|{_CLAIM_LABEL}COを取り下げ、{_CLAIM_LABEL}COに変更します。"
    rf"|(?:占い|霊媒)結果、[^、。]+?は{_VERDICT}です。"
    rf"|[^、。]+?への(?:占い|霊媒)結果を撤回します。"
    rf"|訂正します。[^、。]+?への(?:占い|霊媒)結果は{_VERDICT}です。)+"
)
# `名前(pN)` in public speech. Humans say names; the id form is a prompt
# convention that leaked into what the table reads.
_ID_MENTION_RE = re.compile(r"\(p\d+\)")
# Bigram Jaccard at or above which the next seat is taken to restate the line
# before it. Set on the first Claude game: the one restatement a reader flagged
# (day 3, マコト then アカリ) scored 0.43, the next two echoes 0.25-0.26, and
# the median adjacent pair 0.09.
_ADJACENT_RESTATEMENT_JACCARD = 0.25
_EXAMPLE_TEXT_CHARS = 80


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
    _check_ledger_results(transcript, result)
    _collect_stance_stats(transcript, result)
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
                and _WOLF_WORD_RE.search(sentence)
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


def _actual_result_records(t: GameTranscript) -> dict[str, dict[str, tuple[int, bool]]]:
    """What each seer and medium really saw: owner -> target -> (day, is_werewolf)."""
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
    return records_by_owner


def _check_true_role_result_accuracy(t: GameTranscript, result: AnalysisResult) -> None:
    records_by_owner = _actual_result_records(t)

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


def _check_ledger_results(t: GameTranscript, result: AnalysisResult) -> None:
    """The results on the table's public record, against what was said and seen.

    The public record is built from what players say. A seer who quotes a rival's
    verdict ("アオイのソウタ白は…") was once recorded as having divined it, which put a
    result the real seer never had on the record; the "two results in one night"
    conflict that created discredited the one seer telling the truth for the rest
    of the game. The declared `public_results` were empty in that case, so no check
    that reads them could see it -- this reads the record itself.
    """
    events = [
        event
        for event in t.final_state.get("speech_events", [])
        if event.get("event_type") == "ability_result"
    ]
    if not events:
        return
    records = _actual_result_records(t)
    messages = {
        message.get("message_id", ""): message.get("content", "")
        for message in t.final_state.get("chat_log", [])
    }
    for event in events:
        actor = event.get("actor_id", "")
        target = event.get("target_id", "")
        who = t.names.get(actor, actor)
        whom = t.names.get(target, target)
        color = "人狼" if event.get("result_is_werewolf") else "人狼ではない"
        day = int(event.get("day", 0))
        text = messages.get(event.get("source_message_id", ""), "")
        if float(event.get("confidence", 1.0)) < 1.0:
            result.add(
                Finding(
                    check="undeclared_spoken_result",
                    severity="medium",
                    player_id=actor,
                    day=day,
                    detail=(
                        f"{who} が宣言していない {whom}={color} が、"
                        "発言文から公開の結果として記録された"
                    ),
                    text=text,
                )
            )
        role = t.roles.get(actor)
        if role in ("seer", "medium") and target not in records.get(actor, {}):
            result.add(
                Finding(
                    check="true_role_result_without_record",
                    severity="high",
                    player_id=actor,
                    day=day,
                    detail=(
                        f"本物の{'占い師' if role == 'seer' else '霊媒師'}の {who} が、"
                        f"実際には出していない {whom}={color} を公開した扱いになっている"
                    ),
                    text=text,
                )
            )


# -- 聞き入れて考えを変えているか --------------------------------------------

# Words a player uses to say they took a point or are dropping one. Counted next
# to what they then did: saying it and not moving is the pattern worth seeing.
# The last three are how a seat says it changed its mind; the first Claude game's
# clearest change of mind (「リクさんの指摘で考えを変えます」) matched nothing.
_ACK_RE = re.compile(
    r"確かに|なるほど|その通り|言う通り|言うとおり|納得|認め|撤回|考え直|見直|取り下げ"
    r"|考えを変え|指摘で変え|引っ込め"
)
# 認め that is not the speaker taking a point: thanking or judging someone
# else's admission (「認めてくださって」「認めたのは評価する」), or pointing back to
# a concession already made (「前に認めた通り」 -- the short reply the prompt asks
# for when the same objection comes back). All three were counted as "took the
# point and did not move" in the first Claude game.
_ACK_NOT_TAKEN_RE = re.compile(
    r"認めて(?:くださ|くれ|もら|いただ)"
    r"|認めた(?:の|こと)は"
    r"|(?:前に|前にも|さっき|さっきも|先ほど|先程|昨日|昨日も)認め"
)
# A turn after the acknowledgement: "that is fair, *but* ...". でも and ただ also
# occur inside words (「狼でも」「ただの」), so they count only at a clause start.
_ADVERSATIVE_RE = re.compile(
    r"(?:^|[。、，,！？!?…\s])(?:でも|ただ(?!の)|だが|しかし|それでも|とはいえ)"
    r"|けど|けれど|(?:ます|です|だ|る)が[、,]"
)
# A reason given after that turn. Surface markers only: a concession that keeps
# the view *and says why* reads differently from one that just keeps it.
_REASON_RE = re.compile(r"から|ので|ため|なら|以上|理由|根拠|証拠|矛盾")


def _stated_candidate(utterance: Utterance) -> str | None:
    value = (utterance.reasoning_memo or {}).get("execution_target")
    return value if isinstance(value, str) and value else None


def _acknowledgement(text: str) -> re.Match[str] | None:
    """The first acknowledgement word that is the speaker taking a point."""
    not_taken = [match.span() for match in _ACK_NOT_TAKEN_RE.finditer(text)]
    for match in _ACK_RE.finditer(text):
        if not any(start <= match.start() < end for start, end in not_taken):
            return match
    return None


def _keeps_view_with_reason(text: str, after: int) -> bool:
    """Whether a turn (でも, ただ, …) and then a reason follow position `after`."""
    turn = _ADVERSATIVE_RE.search(text, after)
    return turn is not None and _REASON_RE.search(text, turn.end()) is not None


def _collect_stance_stats(t: GameTranscript, result: AnalysisResult) -> None:
    """How often a seat's stated candidate really moved, and what went with it.

    A change because the old candidate died is no change of mind, so those are
    counted apart. What is left is what a person at the table would call
    listening: the candidate moved while the old one was still alive.

    "Unchanged" can only be seen where both turns state a candidate. The model
    leaves `execution_target` blank on a third of its turns, and in the first
    Claude game 11 of the 19 "took the point and did not move" turns had a
    blank side -- some of them had in fact moved. So the observable pairs are
    the denominator (`observable_follow_ups`), a blank side is its own bucket,
    and a same-candidate acknowledgement is split by whether the speaker
    turned and gave a reason (a partial concession) or did not (the empty
    promise this metric exists to find).
    """
    deaths = {
        record["player_id"]: record["day"] for record in t.final_state.get("death_records", [])
    }
    by_player: dict[str, list[Utterance]] = {}
    for u in t.by_kind("discussion"):
        by_player.setdefault(u.player_id, []).append(u)
    # Transcripts from before the reply target was recorded cannot say.
    replies_known = any(u.reply_to is not None for u in t.by_kind("discussion"))

    follow_ups = observable = forced = moved = same_day = replying = acknowledging = 0
    unmoved = unmoved_with_reason = unstated = unstated_acknowledged = adopted = 0
    for turns in by_player.values():
        for before, after in zip(turns, turns[1:], strict=False):
            follow_ups += 1
            was, now = _stated_candidate(before), _stated_candidate(after)
            acknowledgement = _acknowledgement(after.text)
            if not (was and now):
                unstated += 1
                if acknowledgement is not None:
                    unstated_acknowledged += 1
                    adopted += was is None and now is not None
                continue
            died_on = deaths.get(was)
            if died_on is not None and before.day <= died_on < after.day:
                # Moving off a dead candidate is no change of mind; keeping one
                # is a different defect (`treats_dead_player_as_active`).
                forced += was != now
                continue
            observable += 1
            if was != now:
                moved += 1
                same_day += before.day == after.day
                replying += after.reply_to is not None
                acknowledging += acknowledgement is not None
            elif acknowledgement is not None:
                unmoved += 1
                unmoved_with_reason += _keeps_view_with_reason(after.text, acknowledgement.end())
    result.stats["stance"] = {
        "follow_up_turns": follow_ups,
        "observable_follow_ups": observable,
        "forced_changes": forced,
        "voluntary_changes": moved,
        "voluntary_changes_same_day": same_day,
        "voluntary_changes_replying": replying if replies_known else None,
        "voluntary_changes_acknowledged": acknowledging,
        "acknowledged_without_change": unmoved,
        "acknowledged_without_change_with_reason": unmoved_with_reason,
        "acknowledged_without_change_no_reason": unmoved - unmoved_with_reason,
        "target_unstated_follow_ups": unstated,
        "target_unstated_acknowledged": unstated_acknowledged,
        "target_adopted_acknowledged": adopted,
    }


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

    over_limit = [
        u
        for u in speech
        if u.effective_length_limit is not None and len(u.text) > u.effective_length_limit
    ]
    discussions = [u for u in speech if u.kind == "discussion"]
    counts = [
        sum(utterance.player_id == player_id for utterance in discussions)
        for player_id in t.names
        if player_id in {utterance.player_id for utterance in discussions}
    ]
    cross_pairs = [
        _jaccard(_bigrams(left.text), _bigrams(right.text))
        for index, left in enumerate(discussions)
        for right in discussions[index + 1 :]
        if left.player_id != right.player_id and _bigrams(left.text) and _bigrams(right.text)
    ]
    public_chat = [
        message
        for message in t.final_state.get("chat_log", [])
        if message.get("channel") == "public"
    ]
    # AI public lines only: the human seat's wording is not the AI layer's to
    # answer for, and the id/template rates are claims about the AI layer.
    ai_public = [u.text for u in discussions]
    prompt_template_lines = [bool(_TEMPLATE_PHRASE_RE.search(text)) for text in ai_public]
    code_fact_lines = [bool(_CODE_FACT_SENTENCE_RE.match(text)) for text in ai_public]
    public_days = {int(message.get("day", 0)) for message in public_chat}
    question_count = sum(len(u.directed_question_targets) for u in discussions)
    pending = [
        question
        for questions in t.final_state.get("pending_questions", {}).values()
        for question in questions
    ]
    adjacent = _adjacent_similarities(discussions)

    result.stats["speech"] = {
        "utterances": len(speech),
        "mean_length": round(sum(lengths) / len(lengths), 1),
        "median_length": lengths[len(lengths) // 2],
        "max_length": lengths[-1],
        "over_length_limit": len(over_limit),
        "fallback_lines": len(fallbacks),
        "fallback_rate": round(len(fallbacks) / len(speech), 4),
        "verbatim_repeat_players": repeats,
        "length_variance": round(statistics.pvariance(lengths), 1),
        "speech_count_gini": round(_gini(counts), 4),
        "reply_rate": round(
            sum(bool(message.get("reply_to")) for message in public_chat)
            / max(len(public_chat), 1),
            4,
        ),
        "unanswered_question_rate": round(len(pending) / max(question_count, 1), 4),
        **_unanswered_split(pending, question_count),
        "cross_player_topic_overlap_rate": round(
            sum(similarity >= 0.35 for similarity in cross_pairs) / max(len(cross_pairs), 1), 4
        ),
        "cross_player_mean_jaccard": round(
            sum(cross_pairs) / max(len(cross_pairs), 1), 4
        ),
        # The next seat restating the line before it. Averaged over every pair
        # of lines (above), one echo disappears among thousands of unrelated pairs.
        "adjacent_pairs": len(adjacent),
        "adjacent_mean_jaccard": round(
            sum(similarity for similarity, _, _ in adjacent) / max(len(adjacent), 1), 4
        ),
        "adjacent_overlap_rate": round(
            sum(similarity >= _ADJACENT_RESTATEMENT_JACCARD for similarity, _, _ in adjacent)
            / max(len(adjacent), 1),
            4,
        ),
        "adjacent_overlap_examples": _adjacent_examples(adjacent),
        "utterances_by_stage": _stage_counts(discussions),
        # -- how much the table reads like a chat between people --
        # Lines carrying wording no player chose: prompt vocabulary the model
        # repeated, or a fact sentence the code wrote in front of the line.
        "template_phrase_rate": round(
            sum(
                prompt or code
                for prompt, code in zip(prompt_template_lines, code_fact_lines, strict=True)
            )
            / max(len(ai_public), 1),
            4,
        ),
        # The two sources apart. The prompt-vocabulary rate alone is what this
        # report called the template rate before 2026-11-16 (luna run #11).
        "prompt_template_phrase_rate": round(
            sum(prompt_template_lines) / max(len(ai_public), 1), 4
        ),
        "code_fact_sentence_lines": sum(code_fact_lines),
        "code_fact_sentence_rate": round(sum(code_fact_lines) / max(len(ai_public), 1), 4),
        "id_mention_rate": round(
            sum(bool(_ID_MENTION_RE.search(text)) for text in ai_public)
            / max(len(ai_public), 1),
            4,
        ),
        "public_messages_per_day": round(len(public_chat) / max(len(public_days), 1), 1),
        "public_mean_length": round(
            sum(len(str(message.get("content", ""))) for message in public_chat)
            / max(len(public_chat), 1),
            1,
        ),
    }


def _unanswered_split(
    pending: list[dict[str, Any]], question_count: int
) -> dict[str, float | None]:
    """The questions still pending at the end, by why they are still pending.

    A question leaves the queue only when its target replies with a link to it
    (`reply_to` or a reference). One still there either never reached its
    target (`served` is false: the target got no turn after it, or died) or
    reached it and got an answer with no link, or none. The first is the turn
    structure failing the accused; the second is mostly the model answering in
    prose. In the first Claude game they were 4 and 10 of the 14, and the 4 were
    the questions that mattered most (last words before a vote). Transcripts
    that predate `served` cannot be split and say None.
    """
    if any("served" not in question for question in pending):
        return {
            "unanswered_never_served_rate": None,
            "unanswered_served_without_reply_link_rate": None,
        }
    never_served = sum(1 for question in pending if not question["served"])
    denominator = max(question_count, 1)
    return {
        "unanswered_never_served_rate": round(never_served / denominator, 4),
        "unanswered_served_without_reply_link_rate": round(
            (len(pending) - never_served) / denominator, 4
        ),
    }


def _adjacent_similarities(
    discussions: list[Utterance],
) -> list[tuple[float, Utterance, Utterance]]:
    """Bigram Jaccard of each line with the one before it, same day, other seat."""
    pairs: list[tuple[float, Utterance, Utterance]] = []
    for first, second in zip(discussions, discussions[1:], strict=False):
        if first.day != second.day or first.player_id == second.player_id:
            continue
        left, right = _bigrams(first.text), _bigrams(second.text)
        if left and right:
            pairs.append((_jaccard(left, right), first, second))
    return pairs


def _adjacent_examples(
    pairs: list[tuple[float, Utterance, Utterance]], limit: int = 3
) -> list[dict[str, Any]]:
    """The strongest restatements, so a rate can be checked against the lines."""
    over = sorted(
        (pair for pair in pairs if pair[0] >= _ADJACENT_RESTATEMENT_JACCARD),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return [
        {
            "day": first.day,
            "first": first.player_id,
            "second": second.player_id,
            "jaccard": round(similarity, 4),
            "first_text": first.text[:_EXAMPLE_TEXT_CHARS],
            "second_text": second.text[:_EXAMPLE_TEXT_CHARS],
        }
        for similarity, first, second in over[:limit]
    ]


def _stage_counts(discussions: list[Utterance]) -> dict[str, int] | None:
    """Discussion lines per coordinator stage, or None when none was recorded.

    The stage (`immediate`, `rebuttal…`, `consensus_summary`, …) says which
    slot of the turn structure a line filled -- what the first Claude game's
    read needed to tell whether a rebuttal turn was ever scheduled. The
    transcript does not record it yet; read through `getattr` so the count
    starts as soon as `Utterance` carries a `stage`.
    """
    stages = [getattr(u, "stage", None) for u in discussions]
    if not any(isinstance(stage, str) and stage for stage in stages):
        return None
    counts: dict[str, int] = {}
    for stage in stages:
        key = stage if isinstance(stage, str) and stage else "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _bigrams(text: str) -> set[str]:
    normalized = re.sub(r"[\s、。！？,.・「」『』（）()]+", "", text.lower())
    return {normalized[index : index + 2] for index in range(max(len(normalized) - 1, 0))}


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(len(left | right), 1)


def _gini(values: list[int]) -> float:
    if not values or sum(values) == 0:
        return 0.0
    ordered = sorted(values)
    count = len(ordered)
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2 * weighted) / (count * sum(ordered)) - (count + 1) / count
