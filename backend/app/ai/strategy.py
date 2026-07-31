"""Two-layer strategy guidance.

Layer 1 (`StrategyAnalyzer`): code-computed hard facts derived from
`GameState` -- rope/parity math, the gray-suspect list, CO composition.
This is pure Python, no LLM involved, and is genuinely portable Werewolf
domain knowledge independent of any specific model.

Layer 2: static strategic doctrine text blocks, injected into a prompt
conditionally by role/situation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from app.engine.roles import RoleName
from app.engine.state import GameState


def player_label(state: GameState, player_id: str) -> str:
    """Renders a player as `名前(pN)`.

    The prompt talks about players by name everywhere (chat log, wolf allies,
    freemason partner) but every machine-readable answer is a player_id. With
    no roster tying the two together the model was being asked to vote for
    "p7" having only ever heard of "アカリ" -- a wolf could not even reliably
    avoid voting for its own partner. Carrying both forms everywhere removes
    the guesswork without giving up the unambiguous id in the output.
    """
    player = state.players.get(player_id)
    return f"{player.name}({player_id})" if player is not None else player_id


def player_labels(state: GameState, player_ids: Iterable[str]) -> str:
    return "、".join(player_label(state, pid) for pid in player_ids) or "なし"


@dataclass
class BoardAnalysis:
    alive_count: int
    alive_wolves_estimate_max: int  # unknown to non-wolves; upper bound from role count
    rope_count: int
    gray_player_ids: list[str]
    co_composition: str


class StrategyAnalyzer:
    """Layer 1: hard facts computed from the current GameState."""

    def analyze(self, state: GameState) -> BoardAnalysis:
        alive = state.alive_players()
        alive_ids = {p.player_id for p in alive}

        claimed_ids = {c.player_id for c in state.co_declarations}
        named_ids = {r.target_id for r in state.divine_records} | {
            r.target_id for r in state.medium_records
        }
        # Seating order, not sorted(): lexicographic ids read as
        # "p0、p1、p10、p11、…、p2" once rendered into the prompt.
        excluded = claimed_ids | named_ids
        gray_ids = [p.player_id for p in alive if p.player_id not in excluded]

        # From a village-side vantage point the true wolf count is hidden;
        # use the fixed initial wolf count as the working upper bound.
        max_wolves = 3
        rope = self._rope_count(max_wolves, len(alive))

        co_counts: dict[RoleName, int] = {}
        for c in state.co_declarations:
            if c.player_id in alive_ids:
                co_counts[c.claimed_role] = co_counts.get(c.claimed_role, 0) + 1
        composition = (
            "-".join(
                f"{ROLE_LABELS.get(role, role.value)}{count}" for role, count in co_counts.items()
            )
            or "CO無し"
        )

        return BoardAnalysis(
            alive_count=len(alive),
            alive_wolves_estimate_max=max_wolves,
            rope_count=rope,
            gray_player_ids=gray_ids,
            co_composition=composition,
        )

    @staticmethod
    def _rope_count(estimated_wolves: int, alive_count: int) -> int:
        """Rough number of daytime executions the village can still afford
        to "miss" a wolf before wolves reach parity/majority."""
        non_wolves = alive_count - estimated_wolves
        return max(0, non_wolves - estimated_wolves - 1)


ROLE_LABELS: dict[RoleName, str] = {
    RoleName.VILLAGER: "村人",
    RoleName.WEREWOLF: "人狼",
    RoleName.MADMAN: "狂人",
    RoleName.SEER: "占い",
    RoleName.MEDIUM: "霊媒",
    RoleName.HUNTER: "狩人",
    RoleName.FOX: "妖狐",
    RoleName.FREEMASON: "共有",
}


def render_board_analysis(analysis: BoardAnalysis, state: GameState) -> str:
    return (
        "【盤面分析】\n"
        f"- 生存者数: {analysis.alive_count}人\n"
        f"- 生存者一覧: {player_labels(state, state.alive_ids())}\n"
        f"- 推定残り縄数(村が許容できる誤処刑の目安): {analysis.rope_count}\n"
        f"- CO構成: {analysis.co_composition}\n"
        f"- グレー(無CO・未言及)のプレイヤー: {player_labels(state, analysis.gray_player_ids)}"
    )


SEER_NIGHT_GUIDE = (
    "【占い師の定石】占い先は初日はランダムでも構いませんが、2日目以降はCOした人物や"
    "議論で怪しまれている人物を優先しましょう。すでに黒を出した相手を放置せず、"
    "村の情報になるよう占い結果は速やかに公開してください。"
)

HUNTER_NIGHT_GUIDE = (
    "【狩人の定石】占い師CO者や霊媒師CO者など、村にとって重要な人物を優先的に護衛しましょう。"
    "同じ人物を連続で護衛すると読まれやすいので、状況に応じて護衛先を変えることも検討してください。"
)

WOLF_ATTACK_GUIDE = (
    "【人狼の襲撃方針】大きく2つの路線があります。"
    "「信用勝負路線」は村人として振る舞い続け、疑われにくい人物を残しつつ村の信頼を得る戦略です。"
    "「即噛み路線」は占い師や霊媒師など村の情報源になり得る人物を早期に排除し、村の情報を減らす戦略です。"
    "内輪チャットの流れと現在の縄数を踏まえて、どちらの路線を取るか判断してください。"
)

FREEMASON_CHAT_GUIDE = (
    "【共有者チャットの定石】お互いが確定白であることを活かし、占い師や霊媒師のCOタイミングを"
    "すり合わせたり、怪しい人物についての情報を共有しましょう。共有であることをいつ公開するかも"
    "重要な駆け引きです。"
)

MEDIUM_ROLA_KNOWLEDGE = (
    "【対抗(ロラ)の知識】霊媒師のCOが2人以上いる場合、どちらかは人狼か狂人の偽者です。"
    "過去の霊媒結果の一貫性や、CO順・COタイミングの不自然さから真贋を見極めてください。"
)
