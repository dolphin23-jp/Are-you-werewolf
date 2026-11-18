"""Every schema we may send must compile under Anthropic's structured outputs.

A schema over the limits is refused with a 400 ("Schema is too complex for
compilation") that no retry fixes, and only ever against the live API. These
counts are the offline guard: a field added to a contract shows up here first.
"""

from __future__ import annotations

import re

import pytest
from anthropic import transform_schema
from pydantic import BaseModel

from app.ai.context import (
    CHAT_VOTE_OUTPUT_INSTRUCTION,
    NIGHT_ACTION_OUTPUT_INSTRUCTION,
    VOTE_OUTPUT_INSTRUCTION,
)
from app.ai.schema_limits import (
    MAX_OPTIONAL_PARAMETERS,
    MAX_UNION_PARAMETERS,
    check_within_limits,
    count_schema_parameters,
)
from app.ai.schemas import (
    BriefDiscussionOutput,
    ChatDiscussionOutput,
    DiscussionOutput,
    MorningIntentOutput,
    NightActionOutput,
    SummaryOutput,
    VoteOutput,
    WolfChatOutput,
)
from app.eval.expert_scenarios import ExpertScenarioAnswer
from app.eval.expert_scenarios_v2 import ExpertScenarioV2Answer
from app.eval.judge import PersonaScore

# What a provider may be asked for in a v3 game or an evaluation run.
SENT_SCHEMAS: list[type[BaseModel]] = [
    ChatDiscussionOutput,
    BriefDiscussionOutput,
    VoteOutput,
    NightActionOutput,
    WolfChatOutput,
    MorningIntentOutput,
    SummaryOutput,
    PersonaScore,
    ExpertScenarioAnswer,
    ExpertScenarioV2Answer,
]


@pytest.mark.parametrize("schema", SENT_SCHEMAS, ids=lambda cls: cls.__name__)
def test_every_sent_schema_is_within_the_compilation_limits(schema: type[BaseModel]) -> None:
    optional, union = count_schema_parameters(transform_schema(schema))

    assert optional <= MAX_OPTIONAL_PARAMETERS, f"{schema.__name__}: {optional} optional"
    assert union <= MAX_UNION_PARAMETERS, f"{schema.__name__}: {union} union"
    assert check_within_limits(schema)


def test_the_full_discussion_contract_is_over_the_limit_which_is_why_v3_uses_the_wire_form():
    # If this ever passes, the wire schema is no longer needed for size -- but
    # it still is for its key names (see PlanNote).
    optional, union = count_schema_parameters(transform_schema(DiscussionOutput))

    assert optional == 36
    assert union == 12
    assert not check_within_limits(DiscussionOutput)


def test_the_wire_form_has_headroom_under_both_limits():
    assert count_schema_parameters(transform_schema(ChatDiscussionOutput)) == (17, 6)


def test_the_plan_note_is_written_before_the_message():
    """Structured outputs emit required properties first, in schema order. The
    note is required on the wire so the comparison precedes the line; inside
    it, the comparison (`plan`) precedes the pick."""
    sent = transform_schema(ChatDiscussionOutput)

    assert sent["required"] == ["plan_note", "public_message"]
    assert list(sent["properties"])[:2] == ["plan_note", "public_message"]
    assert list(sent["$defs"]["PlanNote"]["properties"])[:2] == ["plan", "execution_target"]
    # Parsing stays lenient: the json_object path is not grammar-enforced.
    assert ChatDiscussionOutput.model_validate({"public_message": "x"}).plan_note.plan == ""


def _keys_in_order(instruction: str) -> list[str]:
    return re.findall(r'"([a-z_]+)":', instruction.split("\n", 1)[1])


VOTE_ORDER = ["decisive_evidence", "vote_target", "reason", "countercase", "alternative_target"]


def test_the_vote_names_what_decided_it_before_the_target():
    """Seed-1 review R4: with `vote_target` the only required property, and
    first, the ballot was fixed before any reason was written; one seat's
    reason then argued against its own ballot (2-A3). The deciding evidence is
    now required and comes first, the target second."""
    sent = transform_schema(VoteOutput)

    assert sent["required"] == ["decisive_evidence", "vote_target"]
    assert list(sent["properties"]) == VOTE_ORDER
    # Parsing stays lenient: the json_object path is not grammar-enforced.
    parsed = VoteOutput.model_validate({"vote_target": "p1"})
    assert (parsed.vote_target, parsed.decisive_evidence) == ("p1", "")


def test_a_night_action_gives_its_reason_before_its_target():
    sent = transform_schema(NightActionOutput)

    assert sent["required"] == ["reason", "target"]
    assert list(sent["properties"]) == ["reason", "target"]
    assert NightActionOutput.model_validate({"target": "p1"}).reason == ""


def test_the_instructions_show_the_keys_in_the_order_they_are_sent():
    assert _keys_in_order(CHAT_VOTE_OUTPUT_INSTRUCTION) == VOTE_ORDER
    assert _keys_in_order(VOTE_OUTPUT_INSTRUCTION) == VOTE_ORDER
    assert _keys_in_order(NIGHT_ACTION_OUTPUT_INSTRUCTION) == ["reason", "target"]


def test_the_limits_are_the_documented_ones():
    assert (MAX_OPTIONAL_PARAMETERS, MAX_UNION_PARAMETERS) == (24, 16)


# -- the counter itself, on a hand-written schema --

_HAND_WRITTEN = {
    "type": "object",
    "properties": {
        "required_text": {"type": "string"},
        "maybe_text": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "typed_union": {"type": ["integer", "null"]},
        "flag": {"type": "boolean"},
        "inner": {"$ref": "#/$defs/Inner"},
        "maybe_inner": {"anyOf": [{"$ref": "#/$defs/Inner"}, {"type": "null"}]},
        "rows": {"type": "array", "items": {"$ref": "#/$defs/Row"}},
        "loose": {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
    },
    "required": ["required_text", "inner", "rows"],
    "$defs": {
        "Inner": {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a"],
        },
        "Row": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["id"],
        },
    },
}


def test_counter_applies_each_rule():
    optional, union = count_schema_parameters(_HAND_WRITTEN)

    # Top level: maybe_text, typed_union, flag, maybe_inner, loose (5).
    # Inner via `inner`: b (1). Inner again via `maybe_inner`: b (1) -- a
    # definition counts at every use. Row, once for the array: note (1).
    assert optional == 8
    # maybe_text, typed_union, maybe_inner, Row.note, and loose's items.
    assert union == 5


def test_counter_survives_a_self_referencing_definition():
    schema = {
        "type": "object",
        "properties": {"root": {"$ref": "#/$defs/Node"}},
        "required": ["root"],
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
                },
                "required": ["label"],
            }
        },
    }

    assert count_schema_parameters(schema) == (1, 0)


def test_counter_rejects_a_reference_it_cannot_resolve():
    with pytest.raises(ValueError, match="no definition"):
        count_schema_parameters(
            {"type": "object", "properties": {"x": {"$ref": "#/$defs/Missing"}}}
        )
    with pytest.raises(ValueError, match="only local"):
        count_schema_parameters(
            {"type": "object", "properties": {"x": {"$ref": "https://example.invalid/s"}}}
        )
