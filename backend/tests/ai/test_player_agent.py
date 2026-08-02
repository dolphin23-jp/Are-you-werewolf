import pytest

from app.ai.personalities import PERSONALITIES
from app.ai.player_agent import AIPlayerAgent
from app.ai.provider.base import Message
from app.ai.schemas import BriefDiscussionOutput, DiscussionOutput, VoteOutput


class _AlwaysNoneProvider:
    async def generate_structured(self, **kwargs):
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
    agent = AIPlayerAgent(_AlwaysNoneProvider(), personality, max_retries=1)
    result = await agent.generate_discussion("system", [Message(role="user", content="hi")])
    assert result is None


@pytest.mark.asyncio
async def test_strips_meta_phrases():
    agent = AIPlayerAgent(_MetaLeakProvider(), PERSONALITIES[0])
    result = await agent.generate_discussion("system", [Message(role="user", content="hi")])
    assert result is not None
    assert "AIとして" not in result.public_message
    assert "人狼だと思います" in result.public_message


@pytest.mark.asyncio
async def test_invalid_vote_target_falls_back_to_random_valid_choice():
    agent = AIPlayerAgent(_HallucinatingVoteProvider(), PERSONALITIES[0])
    valid = ["p1", "p2", "p3"]
    result = await agent.generate_vote("system", [Message(role="user", content="hi")], valid)
    assert result.vote_target in valid


@pytest.mark.asyncio
async def test_invalid_vote_target_prefers_the_models_valid_alternative():
    agent = AIPlayerAgent(_AlternativeVoteProvider(), PERSONALITIES[0])

    result = await agent.generate_vote(
        "system", [Message(role="user", content="hi")], ["p1", "p2"]
    )

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
    # speaker, before the twelve-field envelope around it.
    assert wordy_provider.budgets[0] >= 2000
    assert "BriefDiscussionOutput" in wordy_provider.schemas
