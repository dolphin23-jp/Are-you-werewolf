"""What the plan note's target fields mean, for every seat (seed-1 review R5,
1-C5, 4-B6).

The note's `execution_target` and `suspects` were described the same way for
every role ("今いちばん吊りたい相手", "怪しいと思う順"), which leaves a wolf to
choose between its real read and the seat it means to push. Wolves wrote the
push -- and that value is what the vote prompt anchors on and the vote audit
compares against. Both fields now mean the seat's position at the table, for
every role; a wolf or madman is told where its real read goes. The schema is
unchanged: only the contract text, switched in code by role.
"""

from __future__ import annotations

from app.ai.context import CHAT_DISCUSSION_OUTPUT_INSTRUCTION
from app.ai.schemas import PlanNote
from tests.ai.seed1_replay import captured_prompts

TARGET = '"execution_target": "今日投票するつもりの相手のplayer_id。'
SUSPECTS = '"suspects": ["卓で疑いを向ける順のplayer_id"]'
TEAM_READ = "本当の見立て（真占い・狩人・狐の候補）はteam_planに書く"


def test_every_seat_reads_the_target_fields_as_its_table_position():
    assert TARGET in CHAT_DISCUSSION_OUTPUT_INSTRUCTION
    assert SUSPECTS in CHAT_DISCUSSION_OUTPUT_INSTRUCTION
    for tag in ("d5-m81-kento-rebuttal", "d4-m61-haruto-initial", "d2-m34-aoi-immediate"):
        prompt = captured_prompts()[tag]
        assert TARGET in prompt.user, tag
        assert SUSPECTS in prompt.user, tag
        assert "今日投票するつもりの相手" in prompt.system, tag
        assert "今いちばん吊りたい相手" not in prompt.text, tag


def test_a_wolf_and_a_madman_are_told_where_their_real_read_goes():
    for tag in ("d4-m61-haruto-initial", "d2-m33-daiki-immediate", "d2-m34-aoi-immediate"):
        assert TEAM_READ in captured_prompts()[tag].user, tag


def test_the_village_and_the_fox_are_not():
    for tag in ("d5-m81-kento-rebuttal", "d2-m35-makoto-initial", "d1-m14-souta-initial"):
        assert TEAM_READ not in captured_prompts()[tag].text, tag


def test_the_schema_is_unchanged():
    assert list(PlanNote.model_fields) == [
        "plan",
        "execution_target",
        "suspects",
        "trusted",
        "trusted_seer",
        "fox_candidates",
        "team_plan",
    ]
