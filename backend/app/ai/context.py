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

from app.ai.deception import FakeClaimGuard, WolfDeceptionAssignment
from app.ai.personalities import Personality
from app.ai.provider.base import Message
from app.ai.strategy import (
    FREEMASON_CHAT_GUIDE,
    HUNTER_NIGHT_GUIDE,
    MEDIUM_ROLA_KNOWLEDGE,
    SEER_NIGHT_GUIDE,
    WOLF_ATTACK_GUIDE,
    StrategyAnalyzer,
    player_label,
    player_labels,
    render_board_analysis,
)
from app.engine.roles import ROLE_DEFINITIONS, RoleName
from app.engine.state import ChatChannel, GameState

DISCUSSION_OUTPUT_INSTRUCTION = """以下のJSON形式で回答してください:
{"public_message": "あなたの発言(200文字以内、人格に合った口調)", \
"reasoning_memo": {"trusted_seer": "信頼する占い師のplayer_idまたはnull", \
"suspects": ["怪しいと思うplayer_idの配列"], "trusted": ["信頼するplayer_idの配列"], \
"execution_target": "処刑したい相手のplayer_idまたはnull", "overall_thought": "現在の考えの要約"}, \
"contains_co_claim": true または false}"""

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
            "- 発言は200文字以内を目安にしてください\n"
            "- 他のプレイヤーの発言内容に具体的に言及してください\n"
            "- 自分自身を疑い先・処刑先・能力対象として扱ってはいけません\n"
            "- 名指しの質問には1回だけ追加返信の機会があります。返信前の相手を"
            "『答えられない』と評価せず、同じ要求を繰り返さないでください"
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
                player_label(state, pid)
                for pid in (p.player_id for p in state.players_by_role(RoleName.WEREWOLF))
                if pid != player_id
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
                player_label(state, pid)
                for pid in (p.player_id for p in state.players_by_role(RoleName.FREEMASON))
                if pid != player_id
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

        return "\n".join(lines)

    # -- layer [C] --

    def _layer_c_state(self, state: GameState, player_id: str, extra_guides: list[str]) -> str:
        analysis = self._analyzer.analyze(state)
        parts = [render_board_analysis(analysis, state)]
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
        lines = [
            f"{player_label(state, m.author_id)}: {m.content}"
            for m in todays
        ]
        return "【当日のログ】\n" + "\n".join(lines)

    def _assemble(
        self, system_layers: list[str], user_layers: list[str]
    ) -> tuple[str, list[Message]]:
        system = "\n\n".join(system_layers)
        user_content = "\n\n".join(user_layers)
        return system, [Message(role="user", content=user_content)]

    # -- public, phase-specific builders --

    def build_discussion_context(
        self, state: GameState, player_id: str
    ) -> tuple[str, list[Message]]:
        guides = self._role_specific_guides(state, player_id)
        return self._assemble(
            [self._layer_a_system(state, player_id), self._layer_b_role(state, player_id)],
            [
                self._layer_c_state(state, player_id, guides),
                self._layer_d_summaries(),
                self._layer_e_current_log(state, ChatChannel.PUBLIC),
                DISCUSSION_OUTPUT_INSTRUCTION,
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
            wolf_log = self._layer_e_current_log(state, ChatChannel.WOLF)
            extra = f"\n\n{wolf_log}"
        return self._assemble(
            [self._layer_a_system(state, player_id), self._layer_b_role(state, player_id)],
            [
                self._layer_c_state(state, player_id, guides),
                self._layer_d_summaries(),
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
                self._layer_c_state(state, player_id, [WOLF_ATTACK_GUIDE]),
                self._layer_e_current_log(state, ChatChannel.WOLF),
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
                self._layer_c_state(state, player_id, [FREEMASON_CHAT_GUIDE]),
                self._layer_e_current_log(state, ChatChannel.FREEMASON),
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
        guides: list[str] = []
        if player.role == RoleName.SEER:
            guides.append(SEER_NIGHT_GUIDE)
        if player.role == RoleName.HUNTER:
            guides.append(HUNTER_NIGHT_GUIDE)
        if player.role == RoleName.WEREWOLF:
            guides.append(WOLF_ATTACK_GUIDE)
        medium_co_count = sum(1 for c in state.co_declarations if c.claimed_role == RoleName.MEDIUM)
        if medium_co_count >= 2:
            guides.append(MEDIUM_ROLA_KNOWLEDGE)
        return guides
