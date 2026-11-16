"""`app/eval/prompt_capture.py`: recording prompts without changing the game."""

from __future__ import annotations

from typing import Any

from app.ai.metrics import MetricsCollector
from app.ai.provider.base import Message
from app.ai.schemas import VoteOutput
from app.eval.prompt_capture import CapturedPrompt, CapturingProvider, capture_offline_game


class _Inner:
    """A provider double that answers every call with one vote."""

    def __init__(self) -> None:
        self._metrics = MetricsCollector()
        self.model = "inner-model"
        self.calls: list[dict[str, Any]] = []

    async def generate_structured(self, **kwargs: Any) -> VoteOutput:
        self.calls.append(kwargs)
        return VoteOutput(vote_target="p1", reason="inner")


async def test_each_call_is_recorded_then_delegated_unchanged():
    inner = _Inner()
    where = iter([(2, "voting"), (2, "runoff")])
    provider = CapturingProvider(inner, context=lambda: next(where))
    messages = [Message(role="user", content="誰に投票しますか")]

    first = await provider.generate_structured(
        system="sys", messages=messages, response_schema=VoteOutput, max_tokens=321
    )
    messages.append(Message(role="assistant", content="later turn"))
    await provider.generate_structured(
        system="", messages=messages, response_schema=VoteOutput, temperature=0.1
    )

    assert first == VoteOutput(vote_target="p1", reason="inner")
    assert inner.calls[0]["max_tokens"] == 321
    assert inner.calls[1]["temperature"] == 0.1
    one, two = provider.captured
    assert (one.index, one.schema, one.system, one.max_tokens) == (0, "VoteOutput", "sys", 321)
    assert (one.day, one.phase) == (2, "voting")
    assert (two.index, two.max_tokens, two.day, two.phase) == (1, 800, 2, "runoff")
    # A copy: the caller appending to its own list later does not rewrite
    # what was captured for the first call.
    assert one.messages == (Message(role="user", content="誰に投票しますか"),)
    assert len(two.messages) == 2


async def test_the_inner_metrics_and_model_stay_visible_through_the_wrapper():
    """The coordinator finds the collector by duck typing on the provider."""
    inner = _Inner()
    provider = CapturingProvider(inner)
    assert provider._metrics is inner._metrics
    assert provider.model == "inner-model"
    assert CapturingProvider(object())._metrics is None  # type: ignore[arg-type]


async def test_without_a_context_the_game_position_is_unknown():
    provider = CapturingProvider(_Inner())
    await provider.generate_structured(system="s", messages=[], response_schema=VoteOutput)
    (only,) = provider.captured
    assert (only.day, only.phase) == (None, None)


def test_a_captured_prompt_serializes_to_its_wire_form():
    prompt = CapturedPrompt(
        index=4,
        response_schema=VoteOutput,
        system="規則",
        messages=(Message(role="user", content="盤面"),),
        max_tokens=800,
        day=1,
        phase="voting",
    )
    assert prompt.wire_messages() == [{"role": "user", "content": "盤面"}]
    assert prompt.input_chars == 4
    assert prompt.to_dict() == {
        "index": 4,
        "schema": "VoteOutput",
        "system": "規則",
        "messages": [{"role": "user", "content": "盤面"}],
        "max_tokens": 800,
        "day": 1,
        "phase": "voting",
    }


async def test_an_offline_v3_game_is_captured_end_to_end():
    metrics = MetricsCollector()
    result = await capture_offline_game(1, metrics=metrics)

    assert result.finished and result.engine == "v3"
    assert result.days >= 2
    counts = result.schema_counts()
    # v3 seats speak through the chat wire schema, never the full contract.
    assert counts.get("ChatDiscussionOutput", 0) > 0
    assert "DiscussionOutput" not in counts
    for schema in ("VoteOutput", "NightActionOutput", "WolfChatOutput"):
        assert counts.get(schema, 0) > 0, schema
    assert [p.index for p in result.prompts] == list(range(len(result.prompts)))
    # Every call carries where in the game it was made.
    phases = {p.schema: set() for p in result.prompts}
    for prompt in result.prompts:
        assert prompt.day is not None and prompt.phase is not None
        phases[prompt.schema].add(prompt.phase)
    assert phases["ChatDiscussionOutput"] == {"discussion"}
    assert phases["VoteOutput"] <= {"voting", "runoff"}
    assert phases["NightActionOutput"] == {"night"}
    # The board memo reaches the model: these are real v3 prompts.
    discussion = next(p for p in result.prompts if p.schema == "ChatDiscussionOutput")
    assert discussion.system and discussion.messages
    assert "【盤面メモ】" in discussion.messages[-1].content
    # The wrapper kept the inner provider's collector visible to the game.
    assert len(metrics.records) == len(result.prompts)
