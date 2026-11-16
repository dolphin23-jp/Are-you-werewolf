"""The v3 wire schema: what the provider is asked for, and what callers get.

`ChatDiscussionOutput` exists for two provider constraints -- a schema size the
API will compile, and key names that never ask for the model's thinking -- but
everything past the provider keeps working on `DiscussionOutput`. These tests
pin the conversion both ways and the parts of the schema that leave the process.
"""

from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace

from anthropic import transform_schema

from app.ai.coordinator import AICoordinator
from app.ai.personalities import PERSONALITIES
from app.ai.player_agent import AIPlayerAgent
from app.ai.provider.base import Message
from app.ai.provider.mock import MockProvider
from app.ai.provider.scenario import ScenarioProvider
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.schemas import (
    PLAN_NOTE_MEMO_FIELDS,
    ChatDiscussionOutput,
    DirectedQuestion,
    DiscussionOutput,
    PlanNote,
    PublicResultClaim,
    ReasoningMemo,
)
from tests.conftest import make_controller

_WORDING = re.compile(r"thought|reasoning|思考", re.IGNORECASE)


def _full_wire() -> ChatDiscussionOutput:
    return ChatDiscussionOutput(
        public_message="占い師CO。Player2は白。",
        reply_to="m3",
        plan_note=PlanNote(
            execution_target="p4",
            suspects=["p4", "p5"],
            trusted=["p2"],
            trusted_seer="p1",
            fox_candidates=["p6"],
            plan="2-1で見る。今日はグレランでp4。",
            team_plan="",
        ),
        public_claim_role="seer",
        public_results=[
            PublicResultClaim(
                result_type="seer", target_id="p2", is_werewolf=False, referenced_day=0
            )
        ],
        directed_questions=[DirectedQuestion(target_id="p4", question="昨日の票の理由は?")],
        ready_to_vote=False,
        needs_another_statement=True,
    )


def test_wire_form_converts_to_the_engine_form_field_by_field():
    wire = _full_wire()

    output = wire.to_discussion_output()

    assert isinstance(output, DiscussionOutput)
    assert output.public_message == wire.public_message
    assert output.reply_to == "m3"
    assert output.public_claim_role == "seer"
    assert output.contains_co_claim is True
    assert output.public_results == wire.public_results
    assert output.directed_questions == wire.directed_questions
    assert output.ready_to_vote is False
    assert output.needs_another_statement is True
    memo = output.reasoning_memo
    assert memo.execution_target == "p4"
    assert memo.suspects == ["p4", "p5"]
    assert memo.trusted == ["p2"]
    assert memo.trusted_seer == "p1"
    assert memo.fox_candidates == ["p6"]
    assert memo.overall_thought == "2-1で見る。今日はグレランでp4。"
    assert memo.private_team_thought == ""
    # What v3 never asks for stays at its default.
    assert memo.role_hypotheses == []
    assert output.key_point == ""
    assert output.agrees_with == []
    assert output.quote is None
    assert output.claim_action is None
    assert output.result_actions == []
    assert output.reassessments == []
    assert output.alternative_execution_target is None
    assert output.strongest_case_against_execution == ""


def test_a_turn_without_a_claim_is_not_a_co():
    output = ChatDiscussionOutput(public_message="様子見。").to_discussion_output()

    assert output.contains_co_claim is False
    assert output.reasoning_memo == ReasoningMemo()


def test_round_trip_through_the_engine_form_is_lossless_for_wire_fields():
    wire = _full_wire()

    assert ChatDiscussionOutput.from_discussion_output(wire.to_discussion_output()) == wire


def test_engine_form_to_wire_drops_only_what_the_wire_has_no_room_for():
    output = DiscussionOutput(
        public_message="同意です。",
        key_point="新論点",
        agrees_with=["m1"],
        quote="原文",
        reasoning_memo=ReasoningMemo(
            execution_target="p3",
            overall_thought="方針",
            private_team_thought="陣営の狙い",
            role_hypotheses=["p1真なら..."],
        ),
        alternative_execution_target="p5",
    )

    wire = ChatDiscussionOutput.from_discussion_output(output)

    assert wire.plan_note.execution_target == "p3"
    assert wire.plan_note.plan == "方針"
    assert wire.plan_note.team_plan == "陣営の狙い"
    back = wire.to_discussion_output()
    assert back.reasoning_memo.role_hypotheses == []
    assert back.key_point == "" and back.agrees_with == [] and back.quote is None


def test_converted_output_shares_no_mutable_state_with_the_wire_form():
    wire = _full_wire()
    output = wire.to_discussion_output()

    # The coordinator repairs memos and results in place.
    output.reasoning_memo.suspects.append("p9")
    output.public_results[0].target_id = "p9"
    output.directed_questions[0].question = "changed"

    assert wire.plan_note.suspects == ["p4", "p5"]
    assert wire.public_results[0].target_id == "p2"
    assert wire.directed_questions[0].question == "昨日の票の理由は?"


def test_the_field_table_covers_the_note_and_lands_on_the_memo():
    assert set(PLAN_NOTE_MEMO_FIELDS) == set(PlanNote.model_fields)
    assert set(PLAN_NOTE_MEMO_FIELDS.values()) <= set(ReasoningMemo.model_fields)
    assert set(ReasoningMemo.model_fields) - set(PLAN_NOTE_MEMO_FIELDS.values()) == {
        "role_hypotheses"
    }


def _property_names(node: object) -> set[str]:
    names: set[str] = set()
    if isinstance(node, dict):
        properties = node.get("properties")
        if isinstance(properties, dict):
            names.update(properties)
        for value in node.values():
            names |= _property_names(value)
    elif isinstance(node, list):
        for value in node:
            names |= _property_names(value)
    return names


def test_no_key_sent_on_the_wire_names_thought_or_reasoning():
    schema = transform_schema(ChatDiscussionOutput)

    names = _property_names(schema)
    assert "plan_note" in names and "plan" in names and "team_plan" in names
    assert not [name for name in names if _WORDING.search(name)]
    # Not just the keys: the maintainer docstrings that explain why must not
    # ride along as schema descriptions either.
    assert not _WORDING.search(json.dumps(schema, ensure_ascii=False))


def _discussion_via_agent(provider: object) -> tuple[DiscussionOutput | None, list[str]]:
    seen: list[str] = []

    class _Recording:
        async def generate_structured(self, **kwargs):  # type: ignore[no-untyped-def]
            seen.append(kwargs["response_schema"].__name__)
            return await provider.generate_structured(**kwargs)  # type: ignore[attr-defined]

    agent = AIPlayerAgent(_Recording(), PERSONALITIES[0], chat_contract=True)
    prompt = (
        "【名簿(JSONのplayer_id用)】あなた=Player3(p3)、Player1=p1、Player2=p2、Player4=p4\n"
        "【当日のログ】\n[m1] Player1: おはよう\n[m2 →m1] Player2: 挨拶だけですね"
    )
    messages = [Message(role="user", content=prompt)]
    result = asyncio.run(agent.generate_discussion("system", messages))
    return result, seen


def test_chat_contract_agent_gets_a_valid_discussion_from_the_mock():
    result, seen = _discussion_via_agent(MockProvider(seed=5))

    assert seen == ["ChatDiscussionOutput"]
    assert isinstance(result, DiscussionOutput)
    assert result.public_message
    assert result.contains_co_claim is (result.public_claim_role is not None)


def test_chat_contract_agent_gets_a_valid_discussion_from_the_scenario_provider():
    for seed in range(10):
        result, seen = _discussion_via_agent(ScenarioProvider(seed=seed))

        assert seen == ["ChatDiscussionOutput"]
        assert isinstance(result, DiscussionOutput)
        assert result.public_message
        # The v3 prompt names the seat only in the roster; it is still not a target.
        assert result.reasoning_memo.execution_target != "p3"
        assert all(q.target_id != "p3" for q in result.directed_questions)


class _CapturingScenario:
    """ScenarioProvider answers, with every prompt it was handed kept."""

    def __init__(self) -> None:
        self.inner = ScenarioProvider(seed=3)
        self.prompts: list[tuple[str, str]] = []

    async def generate_structured(self, *, system, messages, response_schema, **kwargs):  # type: ignore[no-untyped-def]
        text = system + "\n" + "\n".join(m.content for m in messages)
        self.prompts.append((response_schema.__name__, text))
        return await self.inner.generate_structured(
            system=system, messages=messages, response_schema=response_schema, **kwargs
        )


def test_a_v3_day_and_night_never_ask_for_thinking_and_the_night_reads_the_day():
    """Built the way the v3 coordinator builds them, real board memo included."""
    ai_ids = [f"p{i}" for i in range(1, 17)]
    controller = make_controller(seed=4)
    state = controller.state
    session = SimpleNamespace(controller=controller)
    controller.start_game()
    runtime = ReasoningRuntime(state, ai_ids, seed=4)
    provider = _CapturingScenario()
    coordinator = AICoordinator(
        state, ai_ids, provider, seed=4, reasoning=runtime, model_decides=True
    )
    asyncio.run(coordinator.run_night_phase(session))
    controller.start_discussion()
    runtime.refresh(state)
    for speaker in ("p1", "p2", "p3"):
        asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))
    said = [m.content for m in state.chat_log if m.channel.value == "public"]
    controller.end_discussion()
    for voter in ("p1", "p2"):
        asyncio.run(coordinator._cast_vote(controller, state, voter))
    target = next(pid for pid in state.alive_ids() if pid not in ("p0", "p1", "p2"))
    for voter in state.alive_ids():
        if voter not in state.pending_votes:
            controller.vote(voter, target if voter != target else "p0")
    controller.resolve_votes()
    controller.start_night()
    day_prompts = len(provider.prompts)
    asyncio.run(coordinator.run_night_phase(session))

    schemas = {name for name, _ in provider.prompts}
    assert {"VoteOutput", "NightActionOutput", "WolfChatOutput"} <= schemas
    assert schemas & {"DiscussionOutput", "ChatDiscussionOutput"}
    offending = [
        (name, text[max(0, match.start() - 30) : match.end() + 10])
        for name, text in provider.prompts
        for match in _WORDING.finditer(text)
    ]
    assert offending == []
    night = provider.prompts[day_prompts:]
    assert said
    assert {name for name, _ in night} >= {"NightActionOutput", "WolfChatOutput"}
    for name, text in night:
        assert "【今日の公開議論のログ】" in text, name
        assert all(line in text for line in said), name


def test_offline_providers_answer_both_contracts_with_the_same_turn():
    for provider_cls in (MockProvider, ScenarioProvider):
        prompt = [Message(role="user", content="あなた自身のplayer_idは p3 です\np1 p2 p4\n[m1]")]
        full = asyncio.run(
            provider_cls(seed=8).generate_structured(
                system="", messages=prompt, response_schema=DiscussionOutput
            )
        )
        wire = asyncio.run(
            provider_cls(seed=8).generate_structured(
                system="", messages=prompt, response_schema=ChatDiscussionOutput
            )
        )

        assert isinstance(full, DiscussionOutput)
        assert wire == ChatDiscussionOutput.from_discussion_output(full)
