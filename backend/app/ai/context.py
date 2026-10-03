"""5-layer prompt context assembly, one method per phase.

  [A] system prompt / personality
  [B] role-specific info
  [C] game state + StrategyAnalyzer board analysis (+ conditional doctrine)
  [D] rolling compressed summaries of past days (DaySummaryManager)
  [E] full current-day chat log verbatim

Bounded rolling-summary memory: `DaySummaryManager.compress_if_needed()`
keeps total summary size bounded instead of ever-growing transcript replay.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

from app.ai.deception import FakeClaimGuard, WolfDeceptionAssignment
from app.ai.knowledge_base import KnowledgeBase, KnowledgeContext
from app.ai.personalities import Personality, discussion_length_range
from app.ai.provider.base import Message
from app.ai.reasoning.board_memo import render_roster
from app.ai.reasoning.facts import PublicFactLedger
from app.ai.strategy import (
    StrategyAnalyzer,
    player_label,
    player_labels,
    render_board_analysis,
)
from app.engine.roles import ROLE_DEFINITIONS, RoleName
from app.engine.state import ChatChannel, ChatMessage, GameState

# Upper bound for the key-point digest layer. The same statements appear in full in
# the current-day log, so an unbounded digest just doubles the prompt as a day runs on.
_MAX_KEY_POINTS_SHOWN = 12

DISCUSSION_OUTPUT_INSTRUCTION = """以下のJSON形式で回答してください:
{"public_message": "あなたの発言(人格に合った口調)", \
"reasoning_memo": {"trusted_seer": "信頼する占い師のplayer_idまたはnull", \
"suspects": ["怪しいと思うplayer_idの配列"], "trusted": ["信頼するplayer_idの配列"], \
"execution_target": "処刑したい相手のplayer_idまたはnull", "overall_thought": "現在の考えの要約", \
"role_hypotheses": ["各CO者を真と仮定した内訳・矛盾の短い比較"], \
"fox_candidates": ["妖狐候補のplayer_id"], \
"private_team_thought": "非公開。人狼・狂人は本当の陣営と目的を隠さず書く"}, \
"contains_co_claim": true または false, \
"public_claim_role": "今回公開COする役職(seer/medium/hunter/freemason)またはnull", \
"public_results": [{"result_type": "seerまたはmedium", "target_id": "pN", \
"is_werewolf": trueまたはfalse}], \
"reply_to": "反応する相手の発言ID(mN)。特定の発言に反応するなら必ず入れる", \
"quote": "reply_toを入れたときは相手の該当箇所を10〜40字で原文のまま抜粋。それ以外はnull", \
"key_point": "今回新たに出す論点を1行で。新規論点がなければ空文字", \
"agrees_with": ["同意する既出発言ID(mN)"], \
"directed_questions": [{"target_id": "質問相手pN", "question": "質問", \
"source_message_id": "質問のきっかけになった発言IDまたはnull", \
"topic": "execution_candidate|fox_candidate|claim_reason|timeline|other"}], \
"ready_to_vote": trueまたはfalse, "needs_another_statement": trueまたはfalse, \
"reassessments": [{"player_id": "pN", "accepted_point": \
"反論で妥当だった点", "remaining_reason": "それでも残る独立した疑い。なければ空", \
"changed_mind": trueまたはfalse}], \
"alternative_execution_target": "第二候補のpNまたはnull", \
"strongest_case_against_execution": "第一候補を処刑しない最も強い理由"}
反論を読んだ場合はreassessmentsを入れ、主要処刑候補がいる場合は第二候補と、
第一候補を処刑しない最も強い理由を必ず入れる。
主要候補が反論し、各視点と未解決質問を検討し終えた場合だけready_to_vote=true。
新規論点がなくagrees_withだけの場合はreactionとして60文字以内の短い同意にする。
まだ反論・再評価が必要ならfalseとし、自分も追加発言が必要ならneeds_another_statement=true。"""

BRIEF_DISCUSSION_OUTPUT_INSTRUCTION = """【簡易形式】直前の指定は取り消します。分析欄は不要です。
次のJSONだけを返してください: {"public_message": "あなたの発言(人格に合った口調)"}"""

MORNING_INTENT_OUTPUT_INSTRUCTION = """公開発言前の非公開判断です。JSONで回答してください:
{"timing": "immediate|after_results|normal|hold", \
"intent": "publish_result|claim|lead|question|normal", \
"public_claim_role": "seer|medium|hunter|freemasonまたはnull", \
"priority_reason": "簡潔な内部理由"}
新しい占い・霊媒結果を持つCO済み役職はimmediate。朝一COを決めた役職・騙りもimmediate。
占霊結果を見てから出たい共有などはafter_results。意図的潜伏はholdを選んでください。"""

VOTE_OUTPUT_INSTRUCTION = """以下のJSON形式で回答してください:
{"vote_target": "投票する相手のplayer_id", "reason": "簡潔な理由", \
"decisive_evidence": "人数や口調ではない、投票を決めた最も強い独立根拠", \
"countercase": "その相手が村側・真役職でも説明できる最も強い反対仮説", \
"alternative_target": "次点候補のplayer_idまたはnull"}
多数派、共有指定、他者への同意だけを投票理由にしてはいけません。現在の最多候補を
投票する場合も、自分で確認した独立根拠と反対仮説を比較してください。黒を受けた
占いCO者の生存欲、自吊り拒否、COの短さ、初日占い理由の薄さはdecisive_evidenceに
できません。処刑で得る霊結果だけでなく、真役職だった場合に失う結果も比較します。"""

NIGHT_ACTION_OUTPUT_INSTRUCTION = """以下のJSON形式で回答してください:
{"target": "対象プレイヤーのplayer_id", "reason": "簡潔な理由"}"""

WOLF_CHAT_OUTPUT_INSTRUCTION = """以下のJSON形式で回答してください:
{"message": "内輪チャットでの発言(100文字以内)"}"""

SUMMARY_OUTPUT_INSTRUCTION = """以下のJSON形式で回答してください:
{"summary": "その日の出来事の要約(500文字以内)"}"""

# -- v3 (chat register) contracts --
#
# Deliberately a fraction of `DISCUSSION_OUTPUT_INSTRUCTION`. On the seed-11
# live game the full contract plus a reasoning model's hidden thinking ran out
# of completion budget on half of all turns; every field cut from here is a
# field that cannot truncate the sentence the table was waiting for.
CHAT_DISCUSSION_OUTPUT_INSTRUCTION = """次のJSONだけを返してください（前後に説明文を付けない）:
{"public_message": "卓に送るチャット1通。相手は名前で呼び、player_idは書かない", \
"reply_to": "直前の誰かの発言に返すならその発言ID(mN)、なければnull", \
"reasoning_memo": {"execution_target": "今日吊りたい相手のplayer_idまたはnull", \
"suspects": ["怪しいと思う順のplayer_id"], "trusted": ["信用しているplayer_id"], \
"trusted_seer": "真と見ている占いCO者のplayer_idまたはnull", \
"fox_candidates": ["妖狐候補のplayer_id"], \
"overall_thought": "非公開の思考メモ。内訳の見立てと今日の方針を2〜3文", \
"private_team_thought": "非公開。人狼・狂人は本当の陣営の狙いを隠さず書く"}, \
"public_claim_role": "この発言でCOする役職(seer/medium/hunter/freemason)。しないならnull", \
"public_results": [{"result_type": "seerまたはmedium", "target_id": "pN", \
"is_werewolf": trueまたはfalse, "referenced_day": その結果の夜(霊媒は処刑日)の日数}], \
"directed_questions": [{"target_id": "pN", "question": "名指しで聞く一つの質問"}], \
"ready_to_vote": trueまたはfalse}
player_idは【名簿】で引いてください。public_messageの中にはplayer_idを書きません。
public_resultsは本当に自分が持っている結果（騙りなら騙りとして出す結果）だけを入れます。"""

CHAT_VOTE_OUTPUT_INSTRUCTION = """次のJSONだけを返してください:
{"vote_target": "投票する相手のplayer_id", "reason": "一言の理由（チャットで言う調子）", \
"decisive_evidence": "決め手になった一つの根拠", \
"countercase": "その相手が村側でも説明がつく最も強い見方", \
"alternative_target": "次点のplayer_idまたはnull"}
昼に言った吊り先と違う相手に入れるなら、reasonに変えた理由を書いてください。"""

CHAT_PRIVATE_CHAT_INSTRUCTION = """次のJSONだけを返してください:
{"message": "内輪チャットの1通。1〜2文、80文字以内。相手は名前で呼ぶ"}"""


class DaySummaryManager:
    """Bounded rolling-summary memory: full current-day log stays verbatim,
    older days degrade to compressed summaries instead of ever-growing
    transcript replay.

    Facts and commentary are stored apart. The public-fact block is generated
    deterministically from the ledger and must survive compression intact --
    truncating it would leave the AI recalling half a vote history. Only the
    generated commentary, which is opinion, gets shortened."""

    def __init__(self) -> None:
        self.summaries: dict[int, str] = {}
        self.facts: dict[int, str] = {}

    def set_summary(self, day: int, summary: str, facts: str = "") -> None:
        self.summaries[day] = summary
        if facts:
            self.facts[day] = facts

    def compress_if_needed(self, max_total_chars: int = 3000) -> None:
        for day in sorted(self.summaries):
            if self._total_chars() <= max_total_chars:
                return
            current = self.summaries[day]
            if len(current) > 200:
                self.summaries[day] = current[:200] + "…(省略)"

    def render(self) -> str:
        days = sorted(set(self.facts) | set(self.summaries))
        if not days:
            return "(まだ過去日の要約はありません)"
        blocks = []
        for day in days:
            facts = self.facts.get(day, "")
            summary = self.summaries.get(day, "")
            if facts:
                body = "\n".join(part for part in (facts, summary) if part)
                blocks.append(f"{day}日目:\n{body}")
            else:
                blocks.append(f"{day}日目: {summary}")
        return "\n".join(blocks)

    def _total_chars(self) -> int:
        return sum(len(s) for s in self.summaries.values()) + sum(
            len(s) for s in self.facts.values()
        )


class ContextBuilder:
    def __init__(
        self,
        personalities: dict[str, Personality],
        day_summaries: DaySummaryManager,
        wolf_deception: WolfDeceptionAssignment,
        madman_fake_role: RoleName | None,
        fake_claim_guard: FakeClaimGuard,
        observer_player_ids: set[str] | None = None,
        engine: str = "legacy",
    ) -> None:
        self._personalities = personalities
        self._day_summaries = day_summaries
        self._analyzer = StrategyAnalyzer()
        # v3 speaks in the chat register: names only, short lines, the board
        # handed over as a memo rather than as a brief. Everything else keeps
        # the structured prompt that legacy and v2 were measured on.
        self._engine = engine
        self._chat = engine == "v3"
        self._wolf_deception = wolf_deception
        self._madman_fake_role = madman_fake_role
        self._fake_claim_guard = fake_claim_guard
        self._observer_player_ids = observer_player_ids or set()
        self._reasoning_memos: dict[str, dict[str, Any]] = {}
        self._key_points: dict[int, list[tuple[str, str, str]]] = {}
        self._knowledge = KnowledgeBase()

    def set_reasoning_memo(self, player_id: str, memo: dict[str, Any]) -> None:
        self._reasoning_memos[player_id] = memo

    def get_reasoning_memo(self, player_id: str) -> dict[str, Any] | None:
        """The player's last persisted memo. Free-text fields in it are opinion,
        not fact -- only the validated id fields may be relied on."""
        return self._reasoning_memos.get(player_id)

    def record_key_point(
        self, day: int, message_id: str, player_id: str, key_point: str
    ) -> None:
        normalized = key_point.strip()
        if normalized:
            self._key_points.setdefault(day, []).append((message_id, player_id, normalized))

    def _layer_previous_memo(self, player_id: str) -> str:
        memo = self._reasoning_memos.get(player_id)
        if memo is None:
            return "【前回の非公開思考メモ】(まだありません)"
        return "【前回の非公開思考メモ】\n" + json.dumps(memo, ensure_ascii=False)

    def _label(self, state: GameState, player_id: str) -> str:
        """How a player is written in the prompt: `名前(pN)` normally, the name
        alone in the chat register, where the id form must never be modelled."""
        if self._chat:
            player = state.players.get(player_id)
            return player.name if player is not None else player_id
        return player_label(state, player_id)

    def _labels(self, state: GameState, player_ids: Iterable[str]) -> str:
        return "、".join(self._label(state, pid) for pid in player_ids) or "なし"

    # -- layer [A] --

    def _layer_a_system(self, state: GameState, player_id: str) -> str:
        if self._chat:
            return self._layer_a_chat(state, player_id)
        player = state.players[player_id]
        personality = self._personalities[player_id]
        return (
            f"あなたは人狼ゲームに参加しているプレイヤー「{player.name}」です。\n"
            f"あなた自身のplayer_idは {player_id} です。"
            f"「{player.name}({player_id})」はあなた自身であり、別人ではありません。\n"
            "17人参加のオンラインチャット型人狼ゲームです。\n"
            f"{personality.to_prompt_section()}\n"
            "【重要な制約】\n"
            "- 「AIとして」「言語モデルとして」「プロンプト」等のメタ発言は絶対に禁止です\n"
            "- 他のプレイヤーの発言内容に具体的に言及してください\n"
            "- 他プレイヤーを示すときは必ず「名前(pN)」の形で書いてください\n"
            "- 特定の誰かの発言に反応するときは必ずreply_toにその発言ID(mN)を入れてください。"
            "反論・同意・質問への回答・質問のきっかけは、すべてこれに当たります\n"
            "- reply_toを使うときはquoteに相手の該当箇所を10〜40字でそのまま抜き出してください。"
            "要約や言い換えではなく原文のまま抜き出します\n"
            "- 誰に向けた発言か曖昧なまま論評を続けないでください。"
            "宛先のない一般論より、特定の発言への具体的な反応を優先します\n"
            "- 自分自身を疑い先・処刑先・能力対象として扱ってはいけません\n"
            "- 名指しの質問には1回だけ追加返信の機会があります。返信前の相手を"
            "『答えられない』と評価せず、同じ要求を繰り返さないでください\n"
            "- 初日・0日目の占い先は発言情報がないランダム選択でも自然です。"
            "理由がないこと、CO文が短いこと、丁寧さや強い口調そのものを偽要素・狼要素に"
            "してはいけません。初日占い理由は一度回答済みなら再質問しないでください\n"
            "- 黒判定を受けた占いCO者が生存と呪殺機会を求めるのは真偽どちらでも自然です。"
            "自吊りを拒むこと自体を黒要素にせず、判定・結果・視点・投票など独立した根拠と、"
            "その日に吊る場合と一日残す場合の損失を比較してください\n"
            "- 処刑霊結果は占い判定を支持または反証する材料ですが、黒一致だけで占い師の真は"
            "確定しません。身内切り、誤爆、別騙りを残し、『情報が落ちる』だけで処刑を正当化"
            "せず、失う能力結果も同じ重さで評価してください\n"
            "- CO待ちだけで発言を消費せず、処刑希望・妖狐候補・新しく判明した矛盾の"
            "いずれかを具体化してください。既出のCO内訳や両視点で同じ成立条件は再掲せず、"
            "必要ならagrees_withで参照して新しい含意だけを述べてください\n"
            "- reasoning_memoは非公開です。人狼・狂人は本当の役職と陣営目的を隠さず考えてください"
        )

    def _layer_a_chat(self, state: GameState, player_id: str) -> str:
        """The v3 system prompt: a person at a chat table, not a form to fill.

        No fixed openers, no candidate declaration every turn, no ids. The
        structured fields carry the private state; the message is just what the
        player says. Everything about *what is true* arrives in the board memo,
        so this layer is only about *how to be at the table*.
        """
        player = state.players[player_id]
        personality = self._personalities[player_id]
        minimum, maximum = discussion_length_range(personality.verbosity)
        return (
            f"あなたは17人村(17A)のチャット人狼に参加しているプレイヤー「{player.name}」です。"
            "相手は人間のプレイヤーだと思って、人間のプレイヤーとして話してください。\n"
            f"{personality.to_prompt_section(chat=True)}\n"
            "【チャットの話し方】\n"
            f"- 1発言は短く。ふだんは1〜3文、{minimum}〜{max(minimum + 20, maximum // 3)}字くらい。"
            f"CO・結果発表・投票前の整理のときだけ長くてよい(最大{maximum}字)\n"
            "- 相手は名前で呼ぶ。「(p3)」のようなIDは絶対に書かない\n"
            "- 毎回同じ書き出しをしない。毎回「吊り候補は〇〇」と宣言しない。盤面の復唱をしない。"
            "いま言いたいことを一つだけ言う\n"
            "- 誰かの発言に反応するときは、その人の名前を出して具体的に反応する"
            "(同意・反論・質問・回答)。名指しで聞かれたことには最初に答える\n"
            "- 村で普通に使う言葉を使う(CO、対抗、真/偽、狂、内訳、グレー、グレラン、指定、"
            "ローラー、囲い、身内切り、噛み、呪殺、GJ、縄、PP、狐ケア など)\n"
            "- 「AIとして」「言語モデル」「プロンプト」などのメタ発言は絶対に禁止\n"
            "【考え方】\n"
            "- 【盤面メモ】の事実(CO・判定・投票・死亡)は正確に使う。メモにない判定や投票を"
            "作らない。思い込みで誰かのCOや結果を言い換えない\n"
            "- 思考はreasoning_memoに書く。発言はあなたの人格と、あなたの役職(騙っているなら"
            "騙り役)として自然な範囲で選ぶ\n"
            "- 発言で推した吊り先と投票先は原則そろえる。変えるときは一言理由を言う\n"
            "- 自分の陣営の勝ちを目指す。人狼・狂人は本当の目的を発言で漏らさない。"
            "妖狐は占われないこと・最後まで生き残ることを目指す\n"
            "- 「もっとも狼らしい人」と「今日吊るべき人」は別。縄数、霊結果の価値、真役職を"
            "失う損失、狐の生存を考えてから決める\n"
            "- 自分自身を疑い先・吊り先・能力の対象にしない"
        )

    # -- layer [B] --

    def _layer_b_role(self, state: GameState, player_id: str) -> str:
        player = state.players[player_id]
        definition = ROLE_DEFINITIONS[player.role]
        lines = [
            f"【役職情報】あなたの役職は「{definition.label_ja}」です。{definition.description_ja}"
        ]

        if player.role == RoleName.WEREWOLF:
            allies = [
                self._status_label(state, ally.player_id)
                for ally in state.players_by_role(RoleName.WEREWOLF)
                if ally.player_id != player_id
            ]
            lines.append(f"仲間の人狼: {'、'.join(allies) if allies else 'なし'}")
            lines.append(f"あなたたちの欺瞞方針: {self._wolf_deception.pattern_label}")
            if player_id in self._wolf_deception.fake_role_by_player:
                fake_role = self._wolf_deception.fake_role_by_player[player_id]
                lines.append(
                    f"あなたは「{ROLE_DEFINITIONS[fake_role].label_ja}」を騙る担当です。"
                    "仲間を黒だと嘘の結果で名指ししてはいけません。"
                )
            else:
                lines.append("あなたは潜伏担当です。無理にCOせず村人として振る舞ってください。")

        if player.role == RoleName.MADMAN and self._madman_fake_role is not None:
            fake_label = ROLE_DEFINITIONS[self._madman_fake_role].label_ja
            lines.append(f"あなたの戦略: 「{fake_label}」を騙ってください。")

        if player.role == RoleName.FREEMASON:
            partners = [
                self._status_label(state, partner.player_id)
                for partner in state.players_by_role(RoleName.FREEMASON)
                if partner.player_id != player_id
            ]
            lines.append(f"共有者の相方: {'、'.join(partners) if partners else 'なし'}")

        divine_results = [r for r in state.divine_records if r.seer_id == player_id]
        if divine_results:
            rendered = [
                f"{r.day}日目 {self._label(state, r.target_id)}="
                f"{'人狼' if r.is_werewolf else '人狼ではない'}"
                for r in divine_results
            ]
            lines.append("【あなただけが知る占い結果】" + "、".join(rendered))

        medium_results = [r for r in state.medium_records if r.medium_id == player_id]
        if medium_results:
            rendered = [
                f"{r.day}日目 {self._label(state, r.target_id)}="
                f"{'人狼' if r.is_werewolf else '人狼ではない'}"
                for r in medium_results
            ]
            lines.append("【あなただけが知る霊媒結果】" + "、".join(rendered))
            black_count = sum(1 for result in medium_results if result.is_werewolf)
            if black_count >= 2:
                lines.append(
                    "【霊媒の仕事終了】人狼判定を2回出したため、ゲームが続く限り今後の"
                    "処刑結果は白です。自分の価値は結果ではなく確定白・進行役である点です。"
                )

        return "\n".join(lines)

    def _status_label(self, state: GameState, player_id: str) -> str:
        player = state.players[player_id]
        status = "生存" if player.alive else f"{player.death_day}日目死亡済み"
        return f"{self._label(state, player_id)}[{status}]"

    # -- layer [C] --

    def _layer_c_state(
        self,
        state: GameState,
        player_id: str,
        extra_guides: list[str],
        board_memo: str | None = None,
    ) -> str:
        if self._chat and board_memo is not None:
            return self._layer_c_chat(state, board_memo, extra_guides)
        analysis = self._analyzer.analyze(state)
        parts = [render_board_analysis(analysis, state)]
        # Night N deaths are announced after start_discussion increments the
        # public day to N+1; executions remain attached to their discussion day.
        todays_deaths = [
            death
            for death in state.death_records
            if death.day == state.day - 1 and death.cause.value in ("attacked", "cursed")
        ]
        if todays_deaths:
            night_names = [
                player_label(state, death.player_id)
                for death in todays_deaths
                if death.cause.value in ("attacked", "cursed")
            ]
            if night_names:
                parts.append(
                    "【今朝の公開死体（最優先で考察）】"
                    + "、".join(night_names)
                    + "。公開情報では死因の区別はつきません。"
                    "死体数と公開占い結果を照合してください。"
                )
        if state.public_result_claims:
            result_lines = [
                f"{claim.day}日目 {player_label(state, claim.claimant_id)}の"
                f"{'占い' if claim.result_type == 'seer' else '霊媒'}主張: "
                f"{player_label(state, claim.target_id)}={'黒' if claim.is_werewolf else '白'}"
                for claim in state.public_result_claims
            ]
            parts.append("【公開された判定主張】" + " / ".join(result_lines))
        if state.vote_records:
            recent_days = sorted({vote.day for vote in state.vote_records}, reverse=True)[:2]
            vote_lines = [
                f"{vote.day}日目R{vote.round}: {player_label(state, vote.voter_id)} → "
                f"{player_label(state, vote.target_id)}"
                for vote in state.vote_records
                if vote.day in recent_days
            ]
            parts.append("【投票履歴】\n" + "\n".join(vote_lines))
        seer_claimants = [
            declaration.player_id
            for declaration in state.co_declarations
            if declaration.claimed_role == RoleName.SEER
            and state.players[declaration.player_id].alive
        ]
        if len(seer_claimants) >= 2:
            parts.append(
                "【占い視点比較課題】占いCO: "
                + player_labels(state, seer_claimants)
                + "。各人を真と仮定したとき、他の対抗が狼・狂人・狐のどれなら自然か、"
                "結果の矛盾、次に検証すべき処刑・占いを比較してください。"
            )
        observers = [
            player_label(state, pid)
            for pid in sorted(self._observer_player_ids)
            if pid in state.players
        ]
        if observers:
            parts.append(
                "【非参戦席】"
                + "、".join(observers)
                + "は評価用の無言席で、発言・投票をしません。"
                "沈黙や未回答を疑い理由にせず、返答も求めないでください。"
            )
        parts.extend(extra_guides)
        return "\n\n".join(parts)

    def _layer_c_chat(self, state: GameState, board_memo: str, extra_guides: list[str]) -> str:
        """The board as a memo. Facts and hard logic only; the model weighs them."""
        parts = [board_memo]
        todays_deaths = [
            death.player_id
            for death in state.death_records
            if death.day == state.day - 1 and death.cause.value in ("attacked", "cursed")
        ]
        if todays_deaths:
            parts.append(
                "【今朝の死体】"
                + self._labels(state, todays_deaths)
                + "。公開情報では噛みか呪殺かの区別はつかない。"
            )
        observers = [
            self._label(state, pid)
            for pid in sorted(self._observer_player_ids)
            if pid in state.players
        ]
        if observers:
            parts.append(
                "【非参戦席】"
                + "、".join(observers)
                + "は評価用の無言席で、発言・投票をしない。沈黙を疑い理由にしない。"
            )
        parts.extend(extra_guides)
        return "\n\n".join(parts)

    # -- layer [D] --

    def _layer_d_summaries(self) -> str:
        return f"【過去日の要約】\n{self._day_summaries.render()}"

    # -- layer [E] --

    def _layer_e_current_log(self, state: GameState, channel: ChatChannel) -> str:
        todays = [m for m in state.chat_log if m.channel == channel and m.day == state.day]
        if not todays:
            return "【当日のログ】(まだ発言はありません)"
        lines = [self._format_chat_line(state, message) for message in todays]
        return "【当日のログ】\n" + "\n".join(lines)

    def _layer_private_history(self, state: GameState, channel: ChatChannel) -> str:
        messages = [m for m in state.chat_log if m.channel == channel]
        if not messages:
            return "【過去を含む内輪ログ】(まだ発言はありません)"
        lines = [f"{m.day}日目 {self._format_chat_line(state, m)}" for m in messages[-30:]]
        return "【過去を含む内輪ログ】\n" + "\n".join(lines)

    def _format_chat_line(self, state: GameState, message: ChatMessage) -> str:
        reply = f" →{message.reply_to}" if message.reply_to else ""
        references = f" refs={','.join(message.references)}" if message.references else ""
        return (
            f"[{message.message_id}{reply}{references}] "
            f"{self._label(state, message.author_id)}: {message.content}"
        )

    def _layer_pending_questions(self, state: GameState, player_id: str) -> str:
        questions = [item for item in state.pending_questions.get(player_id, []) if not item.served]
        if not questions:
            return "【あなたへの未回答の質問】(ありません)"
        lines = [
            f"[{item.source_message_id}] {self._label(state, item.asker)} →あなた:"
            f"「{item.question}」"
            for item in questions
        ]
        return (
            "【あなたへの未回答の質問】\n"
            + "\n".join(lines)
            + "\n最初にこれへ直接答えてください。答えられない場合は理由を述べてください。"
            "同じことを何人かに聞かれているなら、1回の発言でまとめて答えます。"
            "答えるときはreply_toにその発言ID(mN)を入れてください。"
            "これらの質問はこの発言のあとで片づく扱いなので、次の発言で同じ答えを繰り返さないでください。"
        )

    def _layer_existing_key_points(self, state: GameState) -> str:
        points = self._key_points.get(state.day, [])
        if not points:
            return "【すでに卓に出ている論点】(まだありません)"
        # The full text of every one of these is already in the current-day log, so
        # this layer is a digest, not a second transcript. Keep the most recent ones.
        lines = [
            f"[{message_id}] {self._label(state, player_id)}: {key_point}"
            for message_id, player_id, key_point in points[-_MAX_KEY_POINTS_SHOWN:]
        ]
        return (
            "【すでに卓に出ている論点】\n"
            + "\n".join(lines)
            + "\n既出と同じ論点ならagrees_withへ発言IDを入れ、短い同意で済ませてください。"
            "同じ内容を別の言葉で繰り返してはいけません。"
        )

    def _assemble(
        self, system_layers: list[str], user_layers: list[str]
    ) -> tuple[str, list[Message]]:
        system = "\n\n".join(system_layers)
        user_content = "\n\n".join(user_layers)
        return system, [Message(role="user", content=user_content)]

    # -- public, phase-specific builders --

    def build_discussion_context(
        self,
        state: GameState,
        player_id: str,
        stage: str = "initial",
        *,
        board_memo: str | None = None,
        required_lines: Sequence[str] = (),
    ) -> tuple[str, list[Message]]:
        if self._chat:
            return self._build_chat_discussion_context(
                state, player_id, stage, board_memo=board_memo, required_lines=required_lines
            )
        guides = self._role_specific_guides(state, player_id)
        personality = self._personalities[player_id]
        minimum, maximum = discussion_length_range(personality.verbosity)
        target_chars = f"{minimum}〜{maximum}"
        stage_instruction = (
            "reaction段階では、新論点を無理に作らず、短い同意・驚き・反論・回答だけでも構いません。"
            "その場合もreply_toとquoteで対象の発言を必ず指してください。"
            if stage == "reaction"
            else "未検討の論点を一つ提示する、具体的な相手へ根拠を問う、直前の意見へ反論する、"
            "または発言から処刑候補を絞る、のいずれかを行ってください。"
        )
        if stage == "rebuttal_or_reassessment":
            stage_instruction = (
                "対象の反論を最も強い形で捉え、妥当な点を一つ認めてください。"
                "疑いを維持するなら反論後にも残る独立根拠を示し、それがなければ候補順位を下げます。"
            )
        elif stage == "freemason_confirmation":
            stage_instruction = (
                "共有相方として名指しされています。真の相方なら確認共有COを最初に述べ、"
                "相方でなければ明確に否定してください。他の論点を先に話してはいけません。"
            )
        elif stage.startswith("minority_review:"):
            target_id = stage.split(":", 1)[1]
            target = player_label(state, target_id) if target_id in state.players else target_id
            stage_instruction = (
                f"票・疑いが{target}へ集中しています。多数派の理由を再掲せず、"
                "この人物が村側である反対仮説、本人の反論で妥当な点、別の処刑候補を"
                "必ず比較してください。疑いを維持する場合も独立根拠だけを述べます。"
            )
        elif stage == "consensus_summary":
            stage_instruction = (
                "第一候補だけでなく第二候補、第一候補を処刑しない最強の理由、"
                "本人の反論後にも残った独立根拠、少数意見を整理してください。"
            )
        return self._assemble(
            [self._layer_a_system(state, player_id), self._layer_b_role(state, player_id)],
            [
                self._layer_c_state(state, player_id, guides),
                self._layer_pending_questions(state, player_id),
                self._layer_existing_key_points(state),
                self._layer_d_summaries(),
                self._layer_previous_memo(player_id),
                self._layer_e_current_log(state, ChatChannel.PUBLIC),
                f"【議論段階】{self._stage_label(stage)}。発言長の目安は{target_chars}字です。"
                + stage_instruction
                + "直近の複数人がすでに述べた結論・質問を言い換えて繰り返してはいけません。"
                "同じ処刑候補を支持する場合も、未提示の投票履歴・死体・能力結果・発言差を"
                "一つ追加してください。名指しされた本人は質問への直接回答を最初に述べてください。"
                "当日のログは各行の先頭に[mN]の発言IDが付いています。"
                "誰かの発言を受けて話すなら、そのIDをreply_toに、該当箇所の原文をquoteに入れてください。"
                "consensus_summary段階では新説を広げず、対立点と処刑候補を根拠付きでまとめてください。",
                DISCUSSION_OUTPUT_INSTRUCTION,
            ],
        )

    def _build_chat_discussion_context(
        self,
        state: GameState,
        player_id: str,
        stage: str,
        *,
        board_memo: str | None,
        required_lines: Sequence[str],
    ) -> tuple[str, list[Message]]:
        guides = self._role_specific_guides(state, player_id)
        ledger = PublicFactLedger(state)
        stage_instruction = self._chat_stage_instruction(state, stage)
        duties = ""
        if required_lines:
            # Decided in code: a held result goes out, and a result needs its CO.
            # Said once, plainly; how to say it is the model's.
            duties = "【この発言で必ず言うこと】\n- " + "\n- ".join(required_lines)
        user_layers = [
            self._layer_c_state(state, player_id, guides, board_memo=board_memo),
            self._layer_pending_questions(state, player_id),
            self._layer_d_summaries(),
            self._layer_previous_memo(player_id),
            self._layer_e_current_log(state, ChatChannel.PUBLIC),
            f"【いまの場面】{self._stage_label(stage)}。{stage_instruction}"
            "当日のログは各行の先頭に[mN]の発言IDが付いている。誰かの発言を受けて話すなら"
            "そのIDをreply_toに入れる。",
        ]
        if duties:
            user_layers.append(duties)
        user_layers.append(render_roster(ledger, player_id))
        user_layers.append(CHAT_DISCUSSION_OUTPUT_INSTRUCTION)
        return self._assemble(
            [self._layer_a_system(state, player_id), self._layer_b_role(state, player_id)],
            user_layers,
        )

    def _chat_stage_instruction(self, state: GameState, stage: str) -> str:
        if stage == "immediate":
            return "朝一。結果やCOがあるなら最初にそれを言い、必要なら一言添える。"
        if stage == "reaction":
            return "直前の発言への短い反応(同意・驚き・一言の反論・質問)。1〜2文でよい。"
        if stage == "rebuttal_or_reassessment":
            return (
                "自分が疑われている、または反論を受けた場面。相手の言い分の妥当な点は認め、"
                "それでも残る根拠か反論を短く返す。"
            )
        if stage == "freemason_confirmation":
            return (
                "共有の相方として名指しされた。本当の相方なら最初に確認COし、"
                "違うなら明確に否定する。他の話は後。"
            )
        if stage.startswith("minority_review:"):
            target_id = stage.split(":", 1)[1]
            target = self._label(state, target_id) if target_id in state.players else target_id
            return (
                f"票や疑いが{target}に集中している。多数派の繰り返しではなく、"
                f"{target}が村側である可能性と別の候補を一言で出す。"
            )
        if stage == "consensus_summary":
            return "投票前の整理。今日の吊り先候補と理由、残る対立点を短くまとめる。"
        if stage == "human_followup":
            return "人間のプレイヤーの発言に応答する。"
        return (
            "今日の最初の発言。盤面を見て一番言いたいこと(内訳の見立て、吊り方針、"
            "誰かへの質問のどれか)を一つ。"
        )

    def build_morning_intent_context(
        self, state: GameState, player_id: str
    ) -> tuple[str, list[Message]]:
        return self._assemble(
            [self._layer_a_system(state, player_id), self._layer_b_role(state, player_id)],
            [
                self._layer_c_state(state, player_id, self._role_specific_guides(state, player_id)),
                self._layer_d_summaries(),
                self._layer_previous_memo(player_id),
                MORNING_INTENT_OUTPUT_INSTRUCTION,
            ],
        )

    def build_vote_context(
        self,
        state: GameState,
        player_id: str,
        candidate_ids: list[str],
        *,
        board_memo: str | None = None,
        stated_target: str | None = None,
    ) -> tuple[str, list[Message]]:
        if self._chat:
            return self._build_chat_vote_context(
                state, player_id, candidate_ids, board_memo=board_memo, stated_target=stated_target
            )
        candidates = player_labels(state, candidate_ids)
        # Without saying why the field shrank, a runoff looks to the model like
        # an arbitrarily truncated ballot.
        header = (
            f"【決選投票({state.vote_round}回目)】前回の投票が同数だったため、"
            "候補は同数だったプレイヤーに限られます。次の中から選んでください: "
            if state.runoff_candidates
            else "【投票候補】"
        )
        return self._assemble(
            [self._layer_a_system(state, player_id), self._layer_b_role(state, player_id)],
            [
                # The vote is where a faction's objective actually costs it something,
                # so it needs the same doctrine the discussion and night phases get.
                # Without it a fake seer argues its bluff all day and then votes like
                # a plain villager.
                self._layer_c_state(state, player_id, self._role_specific_guides(state, player_id)),
                self._layer_d_summaries(),
                self._layer_previous_memo(player_id),
                self._layer_e_current_log(state, ChatChannel.PUBLIC),
                f"{header}{candidates}",
                VOTE_OUTPUT_INSTRUCTION,
            ],
        )

    def _build_chat_vote_context(
        self,
        state: GameState,
        player_id: str,
        candidate_ids: list[str],
        *,
        board_memo: str | None,
        stated_target: str | None,
    ) -> tuple[str, list[Message]]:
        ledger = PublicFactLedger(state)
        header = (
            f"【決選投票({state.vote_round}回目)】前回の投票が同数だったため、"
            "候補は同数だったプレイヤーに限られる。次の中から選ぶ: "
            if state.runoff_candidates
            else "【投票候補】"
        )
        stated = (
            f"【昼にあなたが推した吊り先】{self._label(state, stated_target)}"
            if stated_target is not None and stated_target in state.players
            else "【昼にあなたが推した吊り先】特に名指ししていない"
        )
        return self._assemble(
            [self._layer_a_system(state, player_id), self._layer_b_role(state, player_id)],
            [
                self._layer_c_state(
                    state,
                    player_id,
                    self._role_specific_guides(state, player_id),
                    board_memo=board_memo,
                ),
                self._layer_d_summaries(),
                self._layer_previous_memo(player_id),
                self._layer_e_current_log(state, ChatChannel.PUBLIC),
                stated,
                f"{header}{self._labels(state, candidate_ids)}",
                render_roster(ledger, player_id),
                CHAT_VOTE_OUTPUT_INSTRUCTION,
            ],
        )

    def build_night_action_context(
        self,
        state: GameState,
        player_id: str,
        action_type: str,
        candidate_ids: list[str],
        *,
        board_memo: str | None = None,
    ) -> tuple[str, list[Message]]:
        guides = self._role_specific_guides(state, player_id)
        candidates = self._labels(state, candidate_ids)
        extra = ""
        if action_type == "attack":
            wolf_log = self._layer_private_history(state, ChatChannel.WOLF)
            extra = f"\n\n{wolf_log}"
        action_label = {"divine": "占い", "guard": "護衛", "attack": "襲撃"}.get(
            action_type, action_type
        )
        user_layers = [
            self._layer_c_state(state, player_id, guides, board_memo=board_memo),
            self._layer_d_summaries(),
            self._layer_previous_memo(player_id),
            f"【夜行動: {action_label}({action_type})】候補: {candidates}{extra}",
        ]
        if self._chat:
            user_layers.append(render_roster(PublicFactLedger(state), player_id))
        user_layers.append(NIGHT_ACTION_OUTPUT_INSTRUCTION)
        return self._assemble(
            [self._layer_a_system(state, player_id), self._layer_b_role(state, player_id)],
            user_layers,
        )

    def build_wolf_chat_context(
        self, state: GameState, player_id: str, *, board_memo: str | None = None
    ) -> tuple[str, list[Message]]:
        return self._build_private_chat_context(
            state,
            player_id,
            ChatChannel.WOLF,
            "【指示】内輪チャットで襲撃先や騙り戦略、潜伏戦略を100文字以内で相談してください。",
            board_memo=board_memo,
        )

    def build_freemason_chat_context(
        self, state: GameState, player_id: str, *, board_memo: str | None = None
    ) -> tuple[str, list[Message]]:
        return self._build_private_chat_context(
            state,
            player_id,
            ChatChannel.FREEMASON,
            "【指示】共有者チャットで方針を100文字以内で相談してください。",
            board_memo=board_memo,
        )

    def _build_private_chat_context(
        self,
        state: GameState,
        player_id: str,
        channel: ChatChannel,
        instruction: str,
        *,
        board_memo: str | None,
    ) -> tuple[str, list[Message]]:
        user_layers = [
            self._layer_c_state(
                state,
                player_id,
                self._role_specific_guides(state, player_id),
                board_memo=board_memo,
            ),
            self._layer_previous_memo(player_id),
            self._layer_private_history(state, channel),
        ]
        if self._chat:
            user_layers.append(
                "【指示】内輪チャットの1通。仲間と、今夜の行動と明日の方針を人間のチャットのように"
                "短く相談する。"
            )
            user_layers.append(CHAT_PRIVATE_CHAT_INSTRUCTION)
        else:
            user_layers.append(instruction)
            user_layers.append(WOLF_CHAT_OUTPUT_INSTRUCTION)
        return self._assemble(
            [self._layer_a_system(state, player_id), self._layer_b_role(state, player_id)],
            user_layers,
        )

    def build_summary_context(self, state: GameState, player_id: str) -> tuple[str, list[Message]]:
        return self._assemble(
            [self._layer_a_system(state, player_id)],
            [
                self._layer_e_current_log(state, ChatChannel.PUBLIC),
                "【指示】本日の議論・投票の要点を500文字以内で要約してください。",
                SUMMARY_OUTPUT_INSTRUCTION,
            ],
        )

    def _role_specific_guides(self, state: GameState, player_id: str) -> list[str]:
        player = state.players[player_id]
        fake_role = self._wolf_deception.fake_role_by_player.get(player_id)
        if player.role == RoleName.MADMAN:
            fake_role = self._madman_fake_role
        own_claims = {
            declaration.claimed_role
            for declaration in state.co_declarations
            if declaration.player_id == player_id
        }
        perspective_in_public_log = any(
            message.day == state.day
            and any(marker in message.content for marker in ("視点", "真と仮定", "内訳"))
            for message in state.chat_log
            if message.channel == ChatChannel.PUBLIC
        )
        perspective_needed = bool(
            own_claims.intersection({RoleName.SEER, RoleName.MEDIUM})
            or player.role in (RoleName.MEDIUM, RoleName.FREEMASON)
            or (not perspective_in_public_log)
        )
        context = KnowledgeContext(
            state=state,
            player_id=player_id,
            fake_role=fake_role,
            perspective_needed=perspective_needed,
            engine=self._engine,
        )
        return [doctrine.body for doctrine in self._knowledge.select(context)]

    @staticmethod
    def _stage_label(stage: str) -> str:
        if stage.startswith("minority_review:"):
            return "多数派への反証・代替候補の比較"
        return {
            "immediate": "朝一CO・結果発表",
            "initial_view": "初回意見",
            "reaction": "短い反応",
            "rebuttal_or_reassessment": "反論・再評価",
            "freemason_confirmation": "共有相方の確認",
            "consensus_summary": "議論の整理",
            "human_followup": "人間発言への応答",
        }.get(stage, "議論")
