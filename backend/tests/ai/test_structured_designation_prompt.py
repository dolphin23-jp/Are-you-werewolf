"""The v2/legacy closing designation is told one consistent thing.

The structured (v2/legacy) discussion prompt ends every turn with a shared
trailer: a per-persona length band (80〜240字 for a normal speaker) and "even
when backing the same candidate, add one point nobody has raised yet". The
closing designation's own text says the opposite -- name one seat, briefly, no
new point, no question -- so the same 【議論段階】 line told the model both.

The designation now gets its own short line: one target, no new point, no
question. Every other structured stage keeps the shared trailer, and the v3
scene, which was already clean, is pinned unchanged.
"""

from __future__ import annotations

import re

import pytest

from app.ai.context import ContextBuilder, DaySummaryManager
from app.ai.deception import FakeClaimGuard, WolfDeceptionAssignment
from app.ai.personalities import assign_personalities
from app.engine.roles import RoleName
from app.engine.state import GameState
from tests.conftest import make_controller

DESIGNATION = "final_designation"


def _builder(state: GameState, engine: str) -> ContextBuilder:
    wolf_ids = [p.player_id for p in state.players_by_role(RoleName.WEREWOLF)]
    return ContextBuilder(
        personalities=assign_personalities(list(state.players), seed=1),
        day_summaries=DaySummaryManager(),
        wolf_deception=WolfDeceptionAssignment(
            pattern_name="all_lurk",
            pattern_label="全潜伏",
            fake_role_by_player={},
            lurking_player_ids=wolf_ids,
        ),
        madman_fake_role=None,
        fake_claim_guard=FakeClaimGuard(wolf_team_ids=set(wolf_ids)),
        engine=engine,
    )


def _scene(engine: str, stage: str) -> str:
    state = make_controller(seed=4).state
    state.day = 2
    kwargs = {"board_memo": "x"} if engine == "v3" else {}
    _system, messages = _builder(state, engine).build_discussion_context(
        state, "p1", stage, **kwargs
    )
    marker = "【いまの場面】" if engine == "v3" else "【議論段階】"
    return next(line for line in messages[0].content.splitlines() if line.startswith(marker))


@pytest.mark.parametrize("engine", ["legacy", "v2"])
def test_the_structured_designation_asks_for_no_length_band_and_no_new_point(engine: str):
    scene = _scene(engine, DESIGNATION)

    assert scene.startswith("【議論段階】本指定。")
    # The verbosity band (80〜240字 for this persona) is gone ...
    assert "発言長の目安" not in scene
    assert re.search(r"[0-9]+〜[0-9]+字", scene) is None
    # Short is said in sentences, not in characters.
    assert "1〜2文" in scene
    # ... and so is "add one point nobody has raised yet".
    assert "未提示" not in scene
    assert "追加してください" not in scene
    # The consensus-summary reminder belongs to another stage.
    assert "consensus_summary" not in scene


@pytest.mark.parametrize("engine", ["legacy", "v2"])
def test_the_structured_designation_says_one_target_no_new_point_no_question(engine: str):
    scene = _scene(engine, DESIGNATION)

    assert "一人だけ" in scene
    assert "新しい論点" in scene
    assert "新しい質問" in scene
    assert "directed_questionsは空" in scene
    # Still told how to point at the message it answers.
    assert "reply_to" in scene


@pytest.mark.parametrize("engine", ["legacy", "v2"])
@pytest.mark.parametrize("stage", ["initial_view", "reaction", "consensus_summary"])
def test_every_other_structured_stage_keeps_the_shared_trailer(engine: str, stage: str):
    scene = _scene(engine, stage)

    assert "発言長の目安は" in scene
    assert "未提示の投票履歴・死体・能力結果・発言差を一つ追加してください" in scene
    assert "consensus_summary段階では新説を広げず" in scene


def test_the_v3_designation_scene_is_unchanged():
    # Two sentences were added after the owner's review of seed 1 (1-A2 /
    # 1-B2): a grey-random day may close without a name, and the turn says
    # the vote starts right after it. It is still one line with no length
    # band and no new point.
    assert _scene("v3", DESIGNATION) == (
        "【いまの場面】本指定。まとめのあとの締め。ここまでに出た返答を踏まえ、"
        "今日の吊り先を一言で決める。グレランにするなら「グレランで」でよい。"
        "質問はしない。この発言のあとすぐ投票に入る。1文、60字以内。"
        "当日のログは各行の先頭に[mN]の発言IDが付いている。"
        "誰かの発言を受けて話すならそのIDをreply_toに入れる。"
    )
