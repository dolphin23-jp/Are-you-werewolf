import pytest

from app.ai.personalities import PERSONALITIES
from app.ai.player_agent import AIPlayerAgent
from app.ai.provider.base import Message
from app.ai.schemas import DiscussionOutput, VoteOutput


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
