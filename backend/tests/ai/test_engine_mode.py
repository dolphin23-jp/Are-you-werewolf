"""One resolution of the engine flag, shared by the API, the scripts and the tests."""

from __future__ import annotations

import pytest

from app.ai.engine_mode import build_reasoning_runtime, model_decides, uses_reasoning_runtime
from app.ai.reasoning.runtime import ReasoningRuntime
from tests.conftest import make_controller

AI_IDS = [f"p{i}" for i in range(1, 17)]


def test_legacy_has_no_runtime_and_the_model_decides():
    state = make_controller(seed=1).state
    assert build_reasoning_runtime("legacy", state, AI_IDS, seed=1) is None
    assert model_decides("legacy") is True
    assert uses_reasoning_runtime("legacy") is False


def test_v2_has_a_runtime_and_code_decides():
    state = make_controller(seed=1).state
    assert isinstance(build_reasoning_runtime("v2", state, AI_IDS, seed=1), ReasoningRuntime)
    assert model_decides("v2") is False


def test_v3_has_a_runtime_and_the_model_decides():
    state = make_controller(seed=1).state
    assert isinstance(build_reasoning_runtime("v3", state, AI_IDS, seed=1), ReasoningRuntime)
    assert model_decides("v3") is True


def test_an_unknown_engine_fails_loudly():
    state = make_controller(seed=1).state
    with pytest.raises(ValueError, match="unknown reasoning engine"):
        build_reasoning_runtime("v4", state, AI_IDS, seed=1)
