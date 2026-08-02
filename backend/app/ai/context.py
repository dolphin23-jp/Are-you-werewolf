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
from typing import Any

from app.ai.deception import FakeClaimGuard, WolfDeceptionAssignment
from app.ai.knowledge_base import KnowledgeBase, KnowledgeContext
from app.ai.personalities import Personality, discussion_length_range
from app.ai.provider.base import Message
from app.ai.strategy import (
    StrategyAnalyzer,
    player_label,
    player_labels,
    render_board_analysis,
)
from app.engine.roles import ROLE_DEFINITIONS, RoleName
from app.engine.state import ChatChannel, ChatMessage, GameState

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
"reply_to": "反論・回答対象の発言ID(mN)またはnull", "quote": "必要なら短い引用またはnull", \
"directed_questions": [{"target_id": "質問相手pN", "question": "質問", \
"source_message_id": "質問のきっかけになった発言IDまたはnull"}], \
"ready_to_vote": trueまたはfalse, "needs_another_statement": trueまたはfalse}
主要候補が反論し、各視点と未解決質問を検討し終えた場合だけready_to_vote=true。
まだ反論・再評価が必要ならfalseとし、自分も追加発言が必要ならneeds_another_statement=true。"""

MORNING_INTENT_OUTPUT_INSTRUCTION = """公開発言前の非公開判断です。JSONで回答してください:
{"timing": "immediate|after_results|normal|hold", \
"intent": "publish_result|claim|lead|question|normal", \
"public_claim_role": "seer|medium|hunter|freemasonまたはnull", \
"priority_reason": "簡潔な内部理由"}
新しい占い・霊媒結果を持つCO済み役職はimmediate。朝一COを決めた役職・騙りもimmediate。
占霊結果を見てから出たい共有などはafter_results。意図的潜伏はholdを選んでください。"""

VOTE_OUTPUT_INSTRUCTION = """以下のJSON形式で回答してください:
{"vote_target": "投票する相手のplayer_id", "reason": "簡潔な理由"}"""

NIGHT_ACTION_OUTPUT_INSTRUCTION = """以下のJSON形式で回答してください:
{"target": "対象プレイヤーのplayer_id", "reason": "簡潔な理由"}"""

WOLF_CHAT_OUTPUT_INSTRUCTION = """以下のJSON形式で回答してください:
{"message": "内輪チャットでの発言(100文字以内)"}"""

SUMMARY_OUTPUT_INSTRUCTION = """以下のJSON形式で回答してください:
{"summary": "その日の出来事の要約(500文字以内)"}"""


class DaySummaryManager:
    """Bounded rolling-summary memory: full current-day log stays verbatim,
    older days degrade to compressed summaries instead of ever-growing
    transcript replay."""

    def __init__(self) -> None:
        self.summaries: dict[int, str] = {}

    def set_summary(self, day: int, summary: str) -> None:
        self.summaries[day] = summary

    def compress_if_needed(self, max_total_chars: int = 3000) -> None:
        total = sum(len(s) for s in self.summaries.values())
        while total > max_total_chars and self.summaries:
            oldest = min(self.summaries.keys())
            current = self.summaries[oldest]
            if len(current) <= 200:
                break
            self.summaries[oldest] = current[:200] + "…(省略)"
            total = sum(len(s) for s in self.summaries.values())

    def render(self) -> str:
        if not self.summaries:
            return "(まだ過去日の要約はありません)"
        return "\n".join(f"{day}日目: {summary}" for day, summary in sorted(self.summaries.items()))


class ContextBuilder:
    def __init__(
        self,
        personalities: dict[str, Personality],
        day_summaries: DaySummaryManager,
        wolf_deception: WolfDeceptionAssignment,
        madman_fake_role: RoleName | None,
        fake_claim_guard: FakeClaimGuard,
        observer_player_ids: set[str] | None = None,
    ) -> None:
        self._personalities = personalities
        self._day_summaries = day_summaries
        self._analyzer = StrategyAnalyzer()
        self._wolf_deception = wolf_deception
        self._madman_fake_role = madman_fake_role
        self._fake_claim_guard = fake_claim_guard
        self._observer_player_ids = observer_player_ids or set()
        self._reasoning_memos: dict[str, dict[str, Any]] = {}
        self._knowledge = KnowledgeBase()

    def set_reasoning_memo(self, player_id: str, memo: dict[str, Any]) -> None:
        self._reasoning_memos[player_id] = memo

    def _layer_previous_memo(self, player_id: str) -> str:
        memo = self._reasoning_memos.get(player_id)
        if memo is None:
            return "【前回の非公開思考メモ】(まだありません)"
        return "【前回の非公開思考メモ】\n" + json.dumps(memo, ensure_ascii=False)

    # -- layer [A] --

    def _layer_a_system(self, state: GameState, player_id: str) -> str:
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
            "- 反論や質問への回答では、対象ログの発言IDをreply_toに入れてください\n"
            "- 自分自身を疑い先・処刑先・能力対象として扱ってはいけません\n"
            "- 名指しの質問には1回だけ追加返信の機会があります。返信前の相手を"
            "『答えられない』と評価せず、同じ要求を繰り返さないでください\n"
            "- CO待ちだけで発言を消費せず、各CO者を真と仮定した内訳、矛盾、"
            "処刑希望、妖狐候補のいずれかを具体化してください\n"
            "- reasoning_memoは非公開です。人狼・狂人は本当の役職と陣営目的を隠さず考えてください"
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
                f"{r.day}日目 {player_label(state, r.target_id)}="
                f"{'人狼' if r.is_werewolf else '人狼ではない'}"
                for r in divine_results
            ]
            lines.append("【あなただけが知る占い結果】" + "、".join(rendered))

        medium_results = [r for r in state.medium_records if r.medium_id == player_id]
        if medium_results:
            rendered = [
                f"{r.day}日目 {player_label(state, r.target_id)}="
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

    @staticmethod
    def _status_label(state: GameState, player_id: str) -> str:
        player = state.players[player_id]
        status = "生存" if player.alive else f"{player.death_day}日目死亡済み"
        return f"{player_label(state, player_id)}[{status}]"

    # -- layer [C] --

    def _layer_c_state(self, state: GameState, player_id: str, extra_guides: list[str]) -> str:
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

    @staticmethod
    def _format_chat_line(state: GameState, message: ChatMessage) -> str:
        reply = f" →{message.reply_to}" if message.reply_to else ""
        return (
            f"[{message.message_id}{reply}] "
            f"{player_label(state, message.author_id)}: {message.content}"
        )

    def _layer_pending_questions(self, state: GameState, player_id: str) -> str:
        questions = state.pending_questions.get(player_id, [])
        if not questions:
            return "【あなたへの未回答の質問】(ありません)"
        lines = [
            f"[{item.source_message_id}] {player_label(state, item.asker)} →あなた:"
            f"「{item.question}」"
            for item in questions
        ]
        return (
            "【あなたへの未回答の質問】\n"
            + "\n".join(lines)
            + "\n最初にこれへ直接答えてください。答えられない場合は理由を述べてください。"
        )

    def _assemble(
        self, system_layers: list[str], user_layers: list[str]
    ) -> tuple[str, list[Message]]:
        system = "\n\n".join(system_layers)
        user_content = "\n\n".join(user_layers)
        return system, [Message(role="user", content=user_content)]

    # -- public, phase-specific builders --

    def build_discussion_context(
        self, state: GameState, player_id: str, stage: str = "initial"
    ) -> tuple[str, list[Message]]:
        guides = self._role_specific_guides(state, player_id)
        personality = self._personalities[player_id]
        minimum, maximum = discussion_length_range(personality.verbosity)
        target_chars = f"{minimum}〜{maximum}"
        stage_instruction = (
            "reaction段階では、新論点を無理に作らず、短い同意・驚き・反論・回答だけでも構いません。"
            if stage == "reaction"
            else "未検討の論点を一つ提示する、具体的な相手へ根拠を問う、直前の意見へ反論する、"
            "または発言から処刑候補を絞る、のいずれかを行ってください。"
        )
        return self._assemble(
            [self._layer_a_system(state, player_id), self._layer_b_role(state, player_id)],
            [
                self._layer_c_state(state, player_id, guides),
                self._layer_pending_questions(state, player_id),
                self._layer_d_summaries(),
                self._layer_previous_memo(player_id),
                self._layer_e_current_log(state, ChatChannel.PUBLIC),
                f"【議論段階】{self._stage_label(stage)}。発言長の目安は{target_chars}字です。"
                + stage_instruction
                + "直近の複数人がすでに述べた結論・質問を言い換えて繰り返してはいけません。"
                "同じ処刑候補を支持する場合も、未提示の投票履歴・死体・能力結果・発言差を"
                "一つ追加してください。名指しされた本人は質問への直接回答を最初に述べてください。"
                "consensus_summary段階では新説を広げず、対立点と処刑候補を根拠付きでまとめてください。",
                DISCUSSION_OUTPUT_INSTRUCTION,
            ],
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
        self, state: GameState, player_id: str, candidate_ids: list[str]
    ) -> tuple[str, list[Message]]:
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
                self._layer_c_state(state, player_id, []),
                self._layer_d_summaries(),
                self._layer_previous_memo(player_id),
                self._layer_e_current_log(state, ChatChannel.PUBLIC),
                f"{header}{candidates}",
                VOTE_OUTPUT_INSTRUCTION,
            ],
        )

    def build_night_action_context(
        self, state: GameState, player_id: str, action_type: str, candidate_ids: list[str]
    ) -> tuple[str, list[Message]]:
        guides = self._role_specific_guides(state, player_id)
        candidates = player_labels(state, candidate_ids)
        extra = ""
        if action_type == "attack":
            wolf_log = self._layer_private_history(state, ChatChannel.WOLF)
            extra = f"\n\n{wolf_log}"
        return self._assemble(
            [self._layer_a_system(state, player_id), self._layer_b_role(state, player_id)],
            [
                self._layer_c_state(state, player_id, guides),
                self._layer_d_summaries(),
                self._layer_previous_memo(player_id),
                f"【夜行動: {action_type}】候補: {candidates}{extra}",
                NIGHT_ACTION_OUTPUT_INSTRUCTION,
            ],
        )

    def build_wolf_chat_context(
        self, state: GameState, player_id: str
    ) -> tuple[str, list[Message]]:
        return self._assemble(
            [self._layer_a_system(state, player_id), self._layer_b_role(state, player_id)],
            [
                self._layer_c_state(state, player_id, self._role_specific_guides(state, player_id)),
                self._layer_previous_memo(player_id),
                self._layer_private_history(state, ChatChannel.WOLF),
                "【指示】内輪チャットで襲撃先や騙り戦略、潜伏戦略を100文字以内で相談してください。",
                WOLF_CHAT_OUTPUT_INSTRUCTION,
            ],
        )

    def build_freemason_chat_context(
        self, state: GameState, player_id: str
    ) -> tuple[str, list[Message]]:
        return self._assemble(
            [self._layer_a_system(state, player_id), self._layer_b_role(state, player_id)],
            [
                self._layer_c_state(state, player_id, self._role_specific_guides(state, player_id)),
                self._layer_previous_memo(player_id),
                self._layer_private_history(state, ChatChannel.FREEMASON),
                "【指示】共有者チャットで方針を100文字以内で相談してください。",
                WOLF_CHAT_OUTPUT_INSTRUCTION,
            ],
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
        )
        return [doctrine.body for doctrine in self._knowledge.select(context)]

    @staticmethod
    def _stage_label(stage: str) -> str:
        return {
            "immediate": "朝一CO・結果発表",
            "initial_view": "初回意見",
            "reaction": "短い反応",
            "rebuttal_or_reassessment": "反論・再評価",
            "consensus_summary": "議論の整理",
            "human_followup": "人間発言への応答",
        }.get(stage, "議論")
