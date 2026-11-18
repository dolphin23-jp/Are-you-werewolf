"""What the v3 chat prompt asks for, after the first Claude game.

That game's table read as a polite analysis forum, and the read traced most of
it to the prompt rather than the model: it asked for 80-100 and 180-200
character turns, printed the persona's raw cadence values, handed out stock
phrases ("そこは認めるけど", "確かに") and checklists (counter-hypothesis, second
candidate) that came back verbatim as speech. Each test below pins one of
those levers. (The vote prompt's label is pinned in test_vote_memo_target.py.)
"""

from __future__ import annotations

import re

from app.ai.context import ContextBuilder, DaySummaryManager
from app.ai.deception import FakeClaimGuard, WolfDeceptionAssignment
from app.ai.knowledge_base import KnowledgeBase
from app.ai.personalities import (
    CHAT_STAGE_MAX_CHARS,
    CHAT_TURN_CHARS,
    PERSONA_CADENCE,
    PERSONALITIES,
    Personality,
    assign_personalities,
    chat_stage_max_chars,
    discussion_length_range,
)
from app.engine.roles import RoleName
from app.engine.state import GameState
from tests.conftest import make_controller

MEMO = "【盤面メモ】"

# Wording that came back as speech in the first Claude game: the
# acknowledge-then-refuse template, the stock acknowledgement, and the
# comparison checklist meant for the private note.
_RECITED = (
    "そこは認めるけど",
    "確かに",
    "第一候補",
    "第二候補",
    "二番手",
    "反対仮説",
    "吊らない一番の理由",
    "処刑しない最",
)

_STAGES = (
    "immediate",
    "initial_view",
    "reaction",
    "rebuttal_or_reassessment",
    "freemason_confirmation",
    "consensus_summary",
    "human_followup",
    "open_floor",
    "facilitator_followup",
    "provisional_designation",
    "summary_answer",
)


def _chat_builder(
    state: GameState, personalities: dict[str, Personality] | None = None
) -> ContextBuilder:
    wolf_ids = [p.player_id for p in state.players_by_role(RoleName.WEREWOLF)]
    return ContextBuilder(
        personalities=personalities or assign_personalities(list(state.players), seed=1),
        day_summaries=DaySummaryManager(),
        wolf_deception=WolfDeceptionAssignment(
            pattern_name="all_lurk",
            pattern_label="全潜伏",
            fake_role_by_player={},
            lurking_player_ids=wolf_ids,
        ),
        madman_fake_role=None,
        fake_claim_guard=FakeClaimGuard(wolf_team_ids=set(wolf_ids)),
        engine="v3",
    )


# -- length: one short band for every persona, a cap per stage --


def _system_for(personality: Personality) -> str:
    state = make_controller(seed=4).state
    state.day = 1
    personalities = assign_personalities(list(state.players), seed=1)
    personalities["p1"] = personality
    system, _messages = _chat_builder(state, personalities).build_discussion_context(
        state, "p1", "initial_view", board_memo=MEMO
    )
    return system


def test_every_persona_is_asked_for_the_same_short_ordinary_turn():
    low, high = CHAT_TURN_CHARS
    length_lines = set()
    for personality in PERSONALITIES:
        system = _system_for(personality)
        line = next(item for item in system.splitlines() if item.startswith("- 1発言は短く"))
        length_lines.add(line)
        assert f"{low}〜{high}字" in line
        for old in ("80〜100", "180〜200", "30〜50", "1〜3文"):
            assert old not in system, (personality.name, old)

    assert len(length_lines) == 1
    assert (low, high) == (15, 60)


def test_each_stage_carries_its_own_cap():
    state = make_controller(seed=4).state
    state.day = 2
    builder = _chat_builder(state)
    expected = {
        "reaction": 30,
        "initial_view": 60,
        "rebuttal_or_reassessment": 60,
        "immediate": 120,
        "consensus_summary": 120,
        "open_floor": 60,
        "facilitator_followup": 60,
        "provisional_designation": 60,
        "summary_answer": 60,
    }
    for stage, cap in expected.items():
        _system, messages = builder.build_discussion_context(state, "p1", stage, board_memo=MEMO)
        scene = next(
            line for line in messages[0].content.splitlines() if line.startswith("【いまの場面】")
        )
        assert f"{cap}字以内" in scene, stage
        assert chat_stage_max_chars(stage) == cap


def test_the_closing_summary_is_one_line_and_the_comparison_stays_in_the_note():
    state = make_controller(seed=4).state
    state.day = 2
    _system, messages = _chat_builder(state).build_discussion_context(
        state, "p1", "consensus_summary", board_memo=MEMO
    )
    scene = next(
        line for line in messages[0].content.splitlines() if line.startswith("【いまの場面】")
    )

    assert "一行" in scene
    assert "plan_note" in scene


def test_the_chat_cap_is_common_and_above_every_stage_while_the_structured_ranges_stay():
    chat = {discussion_length_range(v, chat=True) for v in ("terse", "normal", "wordy", "?")}

    assert len(chat) == 1
    low, cap = chat.pop()
    assert low == CHAT_TURN_CHARS[0]
    assert cap > max(CHAT_STAGE_MAX_CHARS.values())
    # The structured register keeps the numbers legacy and v2 were measured on.
    assert discussion_length_range("terse") == (30, 100)
    assert discussion_length_range("normal") == (80, 240)
    assert discussion_length_range("wordy") == (180, 400)


def test_the_chat_persona_states_its_cadence_in_words_and_the_structured_one_keeps_values():
    for personality in PERSONALITIES:
        chat = personality.to_prompt_section(chat=True)
        assert "発言頻度" not in chat and "発言量" not in chat, personality.name
        assert not re.search(r"\d\.\d", chat), personality.name
        assert not re.search(r"terse|normal|wordy", chat), personality.name
        assert "- 話す量: " in chat

        structured = personality.to_prompt_section()
        assert f"- 発言頻度: {personality.talkativeness:.1f}" in structured
        assert f"- 発言量: {personality.verbosity}" in structured


def test_cadence_follows_what_each_persona_says_about_itself():
    by_name = {p.name: p for p in PERSONALITIES}

    # Every preset is mapped by meaning; none is left to its list position.
    assert set(PERSONA_CADENCE) == set(by_name)
    for quiet in ("慎重な観察者", "のんびり屋", "無口な実務家", "素朴な聞き手"):
        assert by_name[quiet].verbosity == "terse", quiet
        assert by_name[quiet].talkativeness < 1.0, quiet
    assert by_name["長考する参謀"].verbosity == "wordy"
    for lively in ("元気なムードメーカー", "情熱的な扇動家"):
        assert by_name[lively].talkativeness > 1.0, lively
    # The quiet listener and the long-form analyst must not read alike.
    quiet_line = by_name["慎重な観察者"].to_prompt_section(chat=True).split("- 話す量: ")[1]
    assert "長" not in quiet_line.split("\n", 1)[0]


# -- no checklist or stock phrase to recite --


def test_no_chat_stage_hands_the_model_a_phrase_to_recite():
    state = make_controller(seed=4).state
    state.day = 2
    builder = _chat_builder(state)
    villager = next(p.player_id for p in state.players_by_role(RoleName.VILLAGER))
    for stage in _STAGES:
        system, messages = builder.build_discussion_context(
            state, villager, stage, board_memo=MEMO
        )
        prompt = system + "\n" + messages[0].content
        for phrase in _RECITED:
            assert phrase not in prompt, (stage, phrase)


def test_doctrine_the_chat_register_reads_carries_no_checklist_to_recite():
    for doctrine in KnowledgeBase().doctrines:
        engines = {e.strip() for e in doctrine.metadata.get("engines", "").split(",")}
        if engines - {""} and "v3" not in engines:
            continue
        for phrase in _RECITED:
            assert phrase not in doctrine.body, (doctrine.metadata["id"], phrase)


def test_no_persona_is_handed_a_stock_acknowledgement():
    for personality in PERSONALITIES:
        assert all("確かに" not in line for line in personality.chat_lines), personality.name
        chat = personality.to_prompt_section(chat=True)
        assert "毎回は使わない" in chat


def test_the_chat_prompt_keeps_the_listening_rules_and_asks_for_a_one_word_agreement():
    system = _system_for(PERSONALITIES[0])

    assert "【聞き入れ方】" in system
    assert "認めたなら疑い先・吊り先も実際に変え" in system
    assert "強く言われただけで折れない" in system
    assert "直前の人と同じ結論なら『〇〇に同意』と一言で" in system


def test_the_contract_asks_for_todays_target_every_turn_and_the_comparison_in_the_note():
    state = make_controller(seed=4).state
    state.day = 2
    system, messages = _chat_builder(state).build_discussion_context(
        state, "p1", "initial_view", board_memo=MEMO
    )
    body = messages[0].content
    target_field = body.split('"execution_target": "', 1)[1].split('"', 1)[0]

    assert "決めきれていなくても" in target_field
    assert "plan_noteのexecution_targetには" in system
    assert "候補を比べるのもplan_noteで" in system
