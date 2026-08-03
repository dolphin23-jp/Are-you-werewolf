from pathlib import Path

import pytest

from app.eval.reasoning_analyzer import ReasoningQualityReport
from app.eval.release_gate import HARD_FAILURE_FIELDS, ReleaseDecision, ReleaseGate

CONFIG = Path(__file__).parents[2] / "config" / "reasoning_release_gate.toml"

QUALIFIED_KWARGS = dict(
    live_games=2,
    mock_llm_reduction=0.6,
    human_review_complete=True,
    operational_complete=True,
)


def test_release_gate_fails_hard_error():
    gate = ReleaseGate.from_toml(CONFIG)
    result = gate.evaluate(
        ReasoningQualityReport(private_evidence_exposed_count=1),
        {},
        live_games=3,
        mock_llm_reduction=0.6,
    )
    assert result.decision is ReleaseDecision.FAIL


def test_release_gate_is_inconclusive_without_enough_live_games():
    result = ReleaseGate.from_toml(CONFIG).evaluate(
        ReasoningQualityReport(), {}, live_games=0, mock_llm_reduction=0.6
    )
    assert result.decision is ReleaseDecision.INCONCLUSIVE


def test_release_gate_passes_only_qualified_live_evaluation():
    result = ReleaseGate.from_toml(CONFIG).evaluate(
        ReasoningQualityReport(),
        {},
        live_games=2,
        mock_llm_reduction=0.6,
        human_review_complete=True,
        operational_complete=True,
    )
    assert result.decision is ReleaseDecision.PASS


@pytest.mark.parametrize("field", HARD_FAILURE_FIELDS)
def test_release_gate_fails_on_each_hard_failure_field(field):
    result = ReleaseGate.from_toml(CONFIG).evaluate(
        ReasoningQualityReport(**{field: 1}), {}, **QUALIFIED_KWARGS
    )
    assert result.decision is ReleaseDecision.FAIL
    assert field in result.reasons


def test_release_gate_fails_on_complete_failure_rate_over_threshold():
    result = ReleaseGate.from_toml(CONFIG).evaluate(
        ReasoningQualityReport(), {"complete_failure_rate": 0.02}, **QUALIFIED_KWARGS
    )
    assert result.decision is ReleaseDecision.FAIL
    assert "complete_failure_rate" in result.reasons


def test_release_gate_does_not_fail_at_the_complete_failure_rate_boundary():
    result = ReleaseGate.from_toml(CONFIG).evaluate(
        ReasoningQualityReport(), {"complete_failure_rate": 0.01}, **QUALIFIED_KWARGS
    )
    assert result.decision is ReleaseDecision.PASS


def test_release_gate_fails_on_discussion_skip_rate_over_threshold():
    result = ReleaseGate.from_toml(CONFIG).evaluate(
        ReasoningQualityReport(), {"discussion_skip_rate": 0.05}, **QUALIFIED_KWARGS
    )
    assert result.decision is ReleaseDecision.FAIL
    assert "discussion_skip_rate" in result.reasons


def test_release_gate_fails_on_insufficient_mock_llm_reduction():
    kwargs = {**QUALIFIED_KWARGS, "mock_llm_reduction": 0.3}
    result = ReleaseGate.from_toml(CONFIG).evaluate(ReasoningQualityReport(), {}, **kwargs)
    assert result.decision is ReleaseDecision.FAIL
    assert "mock_logical_call_reduction" in result.reasons


def test_release_gate_skips_efficiency_check_when_reduction_is_not_supplied():
    kwargs = {**QUALIFIED_KWARGS, "mock_llm_reduction": None}
    result = ReleaseGate.from_toml(CONFIG).evaluate(ReasoningQualityReport(), {}, **kwargs)
    assert result.decision is ReleaseDecision.PASS


def test_release_gate_inconclusive_on_stale_transcript_schema_version():
    kwargs = {**QUALIFIED_KWARGS, "transcript_schema_version": 2}
    result = ReleaseGate.from_toml(CONFIG).evaluate(ReasoningQualityReport(), {}, **kwargs)
    assert result.decision is ReleaseDecision.INCONCLUSIVE
    assert "transcript_schema_version" in result.reasons


def test_release_gate_inconclusive_on_incomplete_operational_metrics():
    kwargs = {**QUALIFIED_KWARGS, "operational_complete": False}
    result = ReleaseGate.from_toml(CONFIG).evaluate(ReasoningQualityReport(), {}, **kwargs)
    assert result.decision is ReleaseDecision.INCONCLUSIVE
    assert "operational_metrics_missing" in result.reasons


def test_release_gate_inconclusive_on_incomplete_human_review():
    kwargs = {**QUALIFIED_KWARGS, "human_review_complete": False}
    result = ReleaseGate.from_toml(CONFIG).evaluate(ReasoningQualityReport(), {}, **kwargs)
    assert result.decision is ReleaseDecision.INCONCLUSIVE
    assert "human_review_incomplete" in result.reasons


def test_release_gate_reports_multiple_simultaneous_hard_failures():
    result = ReleaseGate.from_toml(CONFIG).evaluate(
        ReasoningQualityReport(private_evidence_exposed_count=1, public_fact_flip_count=1),
        {},
        **QUALIFIED_KWARGS,
    )
    assert result.decision is ReleaseDecision.FAIL
    assert "private_evidence_exposed_count" in result.reasons
    assert "public_fact_flip_count" in result.reasons


def test_release_gate_respects_custom_config_thresholds():
    gate = ReleaseGate(
        {
            "reliability": {"max_complete_failure_rate": 0.5},
            "qualification": {"minimum_live_games": 5},
        }
    )
    lenient_reliability = gate.evaluate(
        ReasoningQualityReport(),
        {"complete_failure_rate": 0.3},
        live_games=2,
        mock_llm_reduction=None,
        human_review_complete=True,
        operational_complete=True,
    )
    assert lenient_reliability.decision is ReleaseDecision.INCONCLUSIVE
    assert "live_games=2; required=5" in lenient_reliability.reasons
    assert "complete_failure_rate" not in lenient_reliability.reasons
