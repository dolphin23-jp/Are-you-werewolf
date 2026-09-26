"""Every labelled case in the CO corpus, as its own test.

Sensitivity (a claim is read as the right role) and specificity (nothing else
is read as a claim) are both required to be exact on these sets. A false
positive puts a phantom CO on every AI's board; a miss leaves a real one off.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ai.co_detection import detect_claimed_role
from app.engine.roles import RoleName
from tests.ai.fixtures.co_corpus import (
    HELD_OUT_NEGATIVE,
    HELD_OUT_POSITIVE,
    NEGATIVE,
    OTHER_NAMES,
    POSITIVE,
)

_LIVE = json.loads(
    (Path(__file__).parent / "fixtures" / "co_corpus_live.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("text,expected", POSITIVE + HELD_OUT_POSITIVE)
def test_self_claims_are_read_as_the_claimed_role(text: str, expected: RoleName):
    assert detect_claimed_role(text, OTHER_NAMES) is expected


@pytest.mark.parametrize("text", NEGATIVE + HELD_OUT_NEGATIVE)
def test_talk_about_roles_is_not_a_claim(text: str):
    assert detect_claimed_role(text, OTHER_NAMES) is None


@pytest.mark.parametrize(
    "item", _LIVE["items"], ids=[item["source"] for item in _LIVE["items"]]
)
def test_recorded_live_messages_are_read_as_labelled(item: dict[str, str | None]):
    names = [name for pid, name in _LIVE["names"].items() if pid != item["speaker"]]
    expected = RoleName(item["expected"]) if item["expected"] else None
    assert detect_claimed_role(str(item["text"]), names) is expected
