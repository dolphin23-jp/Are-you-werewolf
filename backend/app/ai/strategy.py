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
        # Seating order, not sorted(): lexicographic ids read as
        # "p0、p1、p10、p11、…、p2" once rendered into the prompt.
        # Private divine/medium records must never influence a public board
        # analysis. A result only becomes public once its owner says it in the
        # public chat; until then every non-CO living player remains gray.
        excluded = claimed_ids
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
    "【占い師の定石】新結果は朝早く公開する。対抗は通常すでに人外濃厚なので、"
    "狐騙り濃厚・相互占い指示などの特殊事情がなければ対抗占いより灰、対抗の囲い候補、"
    "未占い狐候補を優先し、グレーを狭めて呪殺を狙う。各対抗を狼・狂人・狐のどれで見るか"
    "理由を示す。"
)

HUNTER_NIGHT_GUIDE = (
    "【狩人の定石】占い師CO者や霊媒師CO者など、村にとって重要な人物を優先的に護衛しましょう。"
    "同じ人物を連続で護衛すると読まれやすいので状況に応じて変える。3狼村で黒を2回出した"
    "霊媒は能力上仕事終了で、以後ゲームが続く限り白しか出ない。守るなら結果目的ではなく、"
    "ほぼ確定白・進行役としての価値を理由にする。"
)

WOLF_ATTACK_GUIDE = (
    "【人狼の襲撃方針】『占い即噛み』は真占い候補を早期襲撃すること。霊媒襲撃は"
    "『霊媒噛み・色隠し』であり即噛みとは呼ばない。『信用勝負』は真占いを残して騙りと"
    "判定・発言で競い、霊媒噛みを伴うことが多い。仲間と真占い候補・狂人候補・狐候補を"
    "本音で整理し、選んだ路線と実際の襲撃先を一致させる。真占いを噛むなら狼側も狐処理を担う。"
)

FREEMASON_CHAT_GUIDE = (
    "【共有者チャットの定石】お互いが確定白であることを活かし、占い師や霊媒師のCOタイミングを"
    "すり合わせたり、怪しい人物についての情報を共有しましょう。共有であることをいつ公開するかも"
    "重要な駆け引きです。"
)

LEADER_GUIDE = (
    "【進行役】対抗なし霊媒や相互確認済み共有として信用が高い場合、仮の処刑候補を示し、"
    "黒を出された本人・対立する役職者・少数意見へ具体的に質問する。命令だけで議論を閉じず、"
    "主要候補の反論と各視点の検証方法が出るまで投票準備完了にしない。"
)

MEDIUM_ROLA_KNOWLEDGE = (
    "【対抗(ロラ)の知識】霊媒師のCOが2人以上いる場合、どちらかは人狼か狂人の偽者です。"
    "過去の霊媒結果の一貫性や、CO順・COタイミングの不自然さから真贋を見極めてください。"
)

PERSPECTIVE_GUIDE = (
    "【視点整理】CO者を一人ずつ真と仮定し、他のCO者が狼・狂人・狐のどれなら自然か、"
    "公開結果と矛盾しないか、次に何を処刑・占い・霊視すれば検証できるかを比較する。"
    "結論だけでなく少なくとも有力な2視点を検討する。共有潜伏中の偽占いの初日黒は"
    "共有トラップで破綻する危険がある。"
)

FOX_GUIDE = (
    "【妖狐ケア】狼が2人死亡した後は最終狼と妖狐候補を分ける。未占い生存者を確認し、"
    "妖狐生存中に最後の狼を先に処刑すると妖狐勝利になる。狼側も真占いを噛んだ後は"
    "妖狐候補を処理する必要がある。"
)
