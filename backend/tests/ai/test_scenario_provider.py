"""`ScenarioProvider` only earns its keep if it targets ids the prompt really
offered and never the speaker itself -- otherwise the coordinator discards its
output and the paths it exists to exercise stay dark."""

from __future__ import annotations

import asyncio

from app.ai.provider.scenario import ScenarioProvider
from app.ai.schemas import DiscussionOutput, VoteOutput

_PROMPT = """あなたは人狼ゲームに参加しているプレイヤー「P3」です。
あなた自身のplayer_idは p3 です。
【盤面分析】
- 生存者一覧: P1(p1)、P2(p2)、P3(p3)、P4(p4)
【あなたへの未回答の質問】(ありません)
【当日のログ】
[m1] P1(p1): おはよう
[m2 →m1] P2(p2): 挨拶だけですね
"""

_PROMPT_WITH_QUESTION = _PROMPT.replace(
    "【あなたへの未回答の質問】(ありません)",
    "【あなたへの未回答の質問】\n[m2] P2(p2) →あなた:「理由は?」\n最初にこれへ直接答えて",
)


def _discussion(prompt: str, seed: int = 1) -> DiscussionOutput:
    provider = ScenarioProvider(seed=seed)
    result = asyncio.run(
        provider.generate_structured(
            system=prompt, messages=[], response_schema=DiscussionOutput
        )
    )
    assert result is not None
    return result


def test_never_targets_itself():
    for seed in range(25):
        output = _discussion(_PROMPT, seed=seed)
        assert output.reasoning_memo.execution_target != "p3"
        assert "p3" not in output.reasoning_memo.suspects
        for question in output.directed_questions:
            assert question.target_id != "p3"


def test_targets_and_message_ids_come_from_the_prompt():
    for seed in range(25):
        output = _discussion(_PROMPT, seed=seed)
        if output.reasoning_memo.execution_target is not None:
            assert output.reasoning_memo.execution_target in {"p1", "p2", "p4"}
        if output.reply_to is not None:
            assert output.reply_to in {"m1", "m2"}
        for message_id in output.agrees_with:
            assert message_id in {"m1", "m2"}


def test_an_open_question_is_answered_by_replying_to_its_message():
    output = _discussion(_PROMPT_WITH_QUESTION, seed=7)
    assert output.reply_to == "m2"
    assert output.key_point


def test_same_seed_reproduces_the_same_output():
    first = _discussion(_PROMPT, seed=11)
    second = _discussion(_PROMPT, seed=11)
    assert first.model_dump() == second.model_dump()


def test_vote_targets_a_player_other_than_itself():
    provider = ScenarioProvider(seed=3)
    result = asyncio.run(
        provider.generate_structured(system=_PROMPT, messages=[], response_schema=VoteOutput)
    )
    assert result is not None
    assert result.vote_target in {"p1", "p2", "p4"}
