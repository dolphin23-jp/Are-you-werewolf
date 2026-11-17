"""In v3 a persona changes how a seat sounds and how much it says, and nothing
else (owner decision on the seed-1 review 1-C2, 1-C3, 3-B4).

The v3 persona carried judgement and procedure as well: a thinking habit
("素直に多数派へ同調しがち"), a discussion style, how far an argument has to go
to move the seat, and example lines that promised content. Two of those lines
came back nearly word for word -- ダイキ's 「根拠言い切るぞ！」 (m46) from
「根拠言い切ろう！」, ユイ's 「放置できないよ！」 (m15) from 「この矛盾は放置できない！」.
The chat persona is now tone, how emotion shows in speech, the amount, and
tone-only examples. legacy and v2 keep the persona they were measured with.
"""

from __future__ import annotations

from app.ai.personalities import PERSONALITIES
from tests.ai.seed1_replay import captured_prompts

# The axes that carried judgement or procedure, by their v3 labels.
_DROPPED_AXES = ("考え方の癖", "議論スタイル", "意見の聞き方")
# Content an example line used to promise: a stance, a procedure, a verdict.
_PROMISES = (
    "根拠",
    "放置",
    "指定",
    "グレラン",
    "2択",
    "外す",
    "下げる",
    "変える",
    "変更",
    "譲れ",
    "引かない",
    "押す",
    "疑う",
    "怪しい",
    "票",
    "結論は伏せる",
)


def test_the_chat_persona_is_tone_and_amount_only():
    for personality in PERSONALITIES:
        chat = personality.to_prompt_section(chat=True)
        for axis in _DROPPED_AXES:
            assert axis not in chat, personality.name
        assert personality.thinking_style not in chat, personality.name
        assert personality.discussion_style not in chat, personality.name
        assert f"- 口調: {personality.tone}" in chat
        assert "- 感情の出し方: " in chat
        assert "- 話す量: " in chat
        assert "思考" not in chat


def test_no_chat_example_promises_content():
    for personality in PERSONALITIES:
        assert len(personality.chat_lines) >= 2, personality.name
        for line in personality.chat_lines:
            for promise in _PROMISES:
                assert promise not in line, (personality.name, line)


def test_the_replayed_prompts_carry_the_tone_only_persona():
    for prompt in captured_prompts().values():
        persona = prompt.system.split("【あなたの人格】", 1)[1].split("【チャットの話し方】", 1)[0]
        for axis in _DROPPED_AXES:
            assert axis not in persona
        assert "根拠言い切ろう" not in prompt.system
        assert "放置できない" not in prompt.system


def test_the_structured_register_keeps_every_axis():
    for personality in PERSONALITIES:
        section = personality.to_prompt_section()
        assert personality.thinking_style in section
        assert personality.discussion_style in section
        assert personality.emotional_tendency in section
        assert all(line in section for line in personality.sample_lines)
