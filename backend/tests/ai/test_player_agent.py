import pytest

from app.ai.personalities import PERSONALITIES
from app.ai.player_agent import AIPlayerAgent
from app.ai.provider.base import Message
from app.ai.provider.budget import BudgetExceeded
from app.ai.schemas import (
    BriefDiscussionOutput,
    ChatDiscussionOutput,
    DiscussionOutput,
    MorningIntentOutput,
    NightActionOutput,
    PlanNote,
    SummaryOutput,
    VoteOutput,
    WolfChatOutput,
)


class _AlwaysNoneProvider:
    def __init__(self):
        self.calls = 0

    async def generate_structured(self, **kwargs):
        self.calls += 1
        return None


class _MetaLeakProvider:
    async def generate_structured(self, *, response_schema, **kwargs):
        if response_schema is DiscussionOutput:
            return DiscussionOutput(public_message="AIとしてお答えします。人狼だと思います。")
        raise AssertionError("unexpected schema")


class _HallucinatingVoteProvider:
    async def generate_structured(self, *, response_schema, **kwargs):
        if response_schema is VoteOutput:
            return VoteOutput(vote_target="not-a-real-player", reason="hallucinated")
        raise AssertionError("unexpected schema")


class _AlternativeVoteProvider:
    async def generate_structured(self, *, response_schema, **kwargs):
        if response_schema is VoteOutput:
            return VoteOutput(
                vote_target="not-a-real-player",
                reason="第一候補が無効なら次点",
                decisive_evidence="投票履歴",
                countercase="村でも起こりうる",
                alternative_target="p2",
            )
        raise AssertionError("unexpected schema")


@pytest.mark.asyncio
async def test_discussion_returns_none_on_total_failure():
    personality = PERSONALITIES[0]
    provider = _AlwaysNoneProvider()
    agent = AIPlayerAgent(provider, personality, max_retries=99)
    result = await agent.generate_discussion("system", [Message(role="user", content="hi")])
    assert result is None
    # Exactly one full-contract request and one brief-contract fallback.
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_strips_meta_phrases():
    agent = AIPlayerAgent(_MetaLeakProvider(), PERSONALITIES[0])
    result = await agent.generate_discussion("system", [Message(role="user", content="hi")])
    assert result is not None
    assert "AIとして" not in result.public_message
    assert "人狼だと思います" in result.public_message


@pytest.mark.asyncio
async def test_invalid_vote_target_falls_back_deterministically():
    agent = AIPlayerAgent(_HallucinatingVoteProvider(), PERSONALITIES[0])
    valid = ["p1", "p2", "p3"]

    first = await agent.generate_vote("system", [Message(role="user", content="hi")], valid)
    second = await agent.generate_vote("system", [Message(role="user", content="hi")], valid)

    # A random substitute turns a bad response into a vote nobody can account
    # for afterwards; the same invalid output must always land the same way.
    assert first.vote_target == second.vote_target == "p1"


@pytest.mark.asyncio
async def test_invalid_vote_target_prefers_the_players_own_stated_beliefs():
    agent = AIPlayerAgent(_HallucinatingVoteProvider(), PERSONALITIES[0])

    result = await agent.generate_vote(
        "system",
        [Message(role="user", content="hi")],
        ["p1", "p2", "p3"],
        preferred_targets=["p9", "p3"],
    )

    assert result.vote_target == "p3"


@pytest.mark.asyncio
async def test_invalid_vote_target_prefers_the_models_valid_alternative():
    agent = AIPlayerAgent(_AlternativeVoteProvider(), PERSONALITIES[0])

    result = await agent.generate_vote("system", [Message(role="user", content="hi")], ["p1", "p2"])

    assert result.vote_target == "p2"
    assert result.decisive_evidence == "投票履歴"
    assert result.countercase == "村でも起こりうる"
    assert result.alternative_target is None


class _TruncatingProvider:
    """Fails the full contract the way a cut-off JSON response does, but answers
    the minimal one. Records the token budget it was asked for."""

    def __init__(self) -> None:
        self.budgets: list[int] = []
        self.schemas: list[str] = []

    async def generate_structured(self, *, response_schema, max_tokens=800, **kwargs):
        self.budgets.append(max_tokens)
        self.schemas.append(response_schema.__name__)
        if response_schema is BriefDiscussionOutput:
            return BriefDiscussionOutput(public_message="占い理由を説明します。")
        return None


@pytest.mark.asyncio
async def test_a_truncated_full_contract_still_produces_a_real_turn():
    # Going silent here is what leaves "書き込み中" on screen resolving to nothing.
    agent = AIPlayerAgent(_TruncatingProvider(), PERSONALITIES[0], max_retries=1)
    result = await agent.generate_discussion("system", [Message(role="user", content="hi")])
    assert result is not None
    assert result.public_message == "占い理由を説明します。"
    assert len(agent._provider.schemas) == 2


@pytest.mark.asyncio
async def test_wordy_speakers_get_a_bigger_token_budget_than_terse_ones():
    # A flat budget fits a terse speaker and truncates a wordy one mid-JSON.
    terse = next(p for p in PERSONALITIES if p.verbosity == "terse")
    wordy = next(p for p in PERSONALITIES if p.verbosity == "wordy")

    terse_provider = _TruncatingProvider()
    await AIPlayerAgent(terse_provider, terse, max_retries=0).generate_discussion(
        "system", [Message(role="user", content="hi")]
    )
    wordy_provider = _TruncatingProvider()
    await AIPlayerAgent(wordy_provider, wordy, max_retries=0).generate_discussion(
        "system", [Message(role="user", content="hi")]
    )

    assert wordy_provider.budgets[0] > terse_provider.budgets[0]
    # The visible sentence alone can be 400 Japanese characters for a wordy
    # speaker, before the structured envelope around it.
    assert wordy_provider.budgets[0] >= 2000
    assert "BriefDiscussionOutput" in wordy_provider.schemas


# -- v3 wire contract, repair events, and the budget that is never swallowed --


class _Events:
    def __init__(self) -> None:
        self.names: list[str] = []

    def record_event(self, name: str) -> None:
        self.names.append(name)


class _ScriptedProvider:
    """Answers each schema from a table; records what it was asked for."""

    def __init__(self, answers: dict[type, object]) -> None:
        self.answers = answers
        self.schemas: list[type] = []

    async def generate_structured(self, *, response_schema, **kwargs):
        self.schemas.append(response_schema)
        answer = self.answers.get(response_schema)
        if isinstance(answer, BaseException):
            raise answer
        return answer


_HI = [Message(role="user", content="hi")]


@pytest.mark.asyncio
async def test_chat_contract_requests_the_wire_schema_and_returns_the_engine_form():
    wire = ChatDiscussionOutput(
        public_message="今日はPlayer4を吊りたい。",
        plan_note=PlanNote(execution_target="p4", plan="グレランでp4。"),
    )
    provider = _ScriptedProvider({ChatDiscussionOutput: wire})
    agent = AIPlayerAgent(provider, PERSONALITIES[0], chat_contract=True)

    result = await agent.generate_discussion("system", _HI)

    assert provider.schemas == [ChatDiscussionOutput]
    assert isinstance(result, DiscussionOutput)
    assert result.public_message == "今日はPlayer4を吊りたい。"
    assert result.reasoning_memo.execution_target == "p4"
    assert result.reasoning_memo.overall_thought == "グレランでp4。"


@pytest.mark.asyncio
async def test_without_the_chat_contract_the_full_schema_is_still_requested():
    provider = _ScriptedProvider({DiscussionOutput: DiscussionOutput(public_message="様子見。")})

    result = await AIPlayerAgent(provider, PERSONALITIES[0]).generate_discussion("system", _HI)

    assert provider.schemas == [DiscussionOutput]
    assert result is not None and result.public_message == "様子見。"


@pytest.mark.asyncio
async def test_a_failed_wire_contract_falls_back_to_the_brief_one_and_says_so():
    events = _Events()
    provider = _ScriptedProvider(
        {BriefDiscussionOutput: BriefDiscussionOutput(public_message="占い理由を説明します。")}
    )
    agent = AIPlayerAgent(provider, PERSONALITIES[0], chat_contract=True, metrics=events)

    result = await agent.generate_discussion("system", _HI)

    assert provider.schemas == [ChatDiscussionOutput, BriefDiscussionOutput]
    assert result is not None and result.public_message == "占い理由を説明します。"
    assert events.names == ["brief_fallback"]


@pytest.mark.asyncio
async def test_a_turn_that_needed_no_repair_records_nothing():
    events = _Events()
    provider = _ScriptedProvider(
        {
            ChatDiscussionOutput: ChatDiscussionOutput(public_message="おはよう。"),
            VoteOutput: VoteOutput(vote_target="p2"),
            NightActionOutput: NightActionOutput(target="p2"),
            WolfChatOutput: WolfChatOutput(message="今夜は占い師を噛もう。"),
        }
    )
    agent = AIPlayerAgent(provider, PERSONALITIES[0], chat_contract=True, metrics=events)

    await agent.generate_discussion("system", _HI)
    await agent.generate_vote("system", _HI, ["p1", "p2"])
    await agent.generate_night_action("system", _HI, ["p1", "p2"])
    await agent.generate_wolf_chat("system", _HI)

    assert events.names == []


@pytest.mark.asyncio
async def test_a_vote_code_had_to_cast_or_repair_is_counted():
    events = _Events()
    missing = AIPlayerAgent(_ScriptedProvider({}), PERSONALITIES[0], metrics=events)
    invalid = AIPlayerAgent(
        _ScriptedProvider({VoteOutput: VoteOutput(vote_target="p9")}),
        PERSONALITIES[0],
        metrics=events,
    )

    assert (await missing.generate_vote("system", _HI, ["p1", "p2"])).vote_target == "p1"
    assert (await invalid.generate_vote("system", _HI, ["p1", "p2"])).vote_target == "p1"

    assert events.names == ["vote_code_resolved", "vote_code_resolved"]


@pytest.mark.asyncio
async def test_a_night_target_code_had_to_choose_or_repair_is_counted():
    events = _Events()
    missing = AIPlayerAgent(_ScriptedProvider({}), PERSONALITIES[0], metrics=events)
    invalid = AIPlayerAgent(
        _ScriptedProvider({NightActionOutput: NightActionOutput(target="p9", reason="r")}),
        PERSONALITIES[0],
        metrics=events,
    )

    assert (await missing.generate_night_action("system", _HI, ["p1", "p2"])).target == "p1"
    repaired = await invalid.generate_night_action("system", _HI, ["p1", "p2"])

    assert repaired.target == "p1" and repaired.reason == "r"
    assert events.names == ["night_code_resolved", "night_code_resolved"]


@pytest.mark.asyncio
async def test_a_private_chat_line_that_is_the_personas_canned_line_is_counted():
    events = _Events()
    personality = PERSONALITIES[0]
    missing = AIPlayerAgent(_ScriptedProvider({}), personality, metrics=events)
    meta_only = AIPlayerAgent(
        _ScriptedProvider({WolfChatOutput: WolfChatOutput(message="AIとして")}),
        personality,
        metrics=events,
    )

    assert (await missing.generate_wolf_chat("system", _HI)).message == (
        personality.get_fallback_message()
    )
    assert (await meta_only.generate_wolf_chat("system", _HI)).message == (
        personality.get_fallback_message()
    )
    assert events.names == ["wolf_chat_fallback_line", "wolf_chat_fallback_line"]


@pytest.mark.asyncio
async def test_metrics_without_an_event_counter_are_ignored():
    # A plain MetricsCollector from before events existed, or anything else.
    agent = AIPlayerAgent(_ScriptedProvider({}), PERSONALITIES[0], metrics=object())

    vote = await agent.generate_vote("system", _HI, ["p1", "p2"])

    assert vote.vote_target == "p1"


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_contract", [False, True])
async def test_an_exhausted_budget_is_never_absorbed_into_a_repair(chat_contract: bool):
    exhausted = BudgetExceeded("HTTP request budget exhausted (3)")
    provider = _ScriptedProvider(
        {
            schema: exhausted
            for schema in (
                DiscussionOutput,
                ChatDiscussionOutput,
                BriefDiscussionOutput,
                VoteOutput,
                NightActionOutput,
                WolfChatOutput,
                MorningIntentOutput,
                SummaryOutput,
            )
        }
    )
    agent = AIPlayerAgent(provider, PERSONALITIES[0], chat_contract=chat_contract)

    with pytest.raises(BudgetExceeded):
        await agent.generate_discussion("system", _HI)
    # Raised by the first contract: no brief request is spent after it.
    assert len(provider.schemas) == 1
    with pytest.raises(BudgetExceeded):
        await agent.generate_vote("system", _HI, ["p1"])
    with pytest.raises(BudgetExceeded):
        await agent.generate_night_action("system", _HI, ["p1"])
    with pytest.raises(BudgetExceeded):
        await agent.generate_wolf_chat("system", _HI)
    with pytest.raises(BudgetExceeded):
        await agent.generate_morning_intent("system", _HI)
    with pytest.raises(BudgetExceeded):
        await agent.generate_summary("system", _HI)


@pytest.mark.asyncio
async def test_any_other_provider_error_is_still_absorbed():
    provider = _ScriptedProvider(
        {
            VoteOutput: RuntimeError("connection reset"),
            ChatDiscussionOutput: ValueError("bad json"),
            BriefDiscussionOutput: ValueError("bad json"),
        }
    )
    agent = AIPlayerAgent(provider, PERSONALITIES[0], chat_contract=True)

    assert (await agent.generate_vote("system", _HI, ["p1"])).vote_target == "p1"
    assert await agent.generate_discussion("system", _HI) is None
