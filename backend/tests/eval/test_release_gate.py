from pathlib import Path

from app.eval.reasoning_analyzer import ReasoningQualityReport
from app.eval.release_gate import ReleaseDecision, ReleaseGate

CONFIG = Path(__file__).parents[2] / "config" / "reasoning_release_gate.toml"


def test_release_gate_fails_hard_error():
    gate = ReleaseGate.from_toml(CONFIG)
    result = gate.evaluate(
        ReasoningQualityReport(private_evidence_exposed_count=1), {}, live_pairs=3,
        mock_llm_reduction=0.6,
    )
    assert result.decision is ReleaseDecision.FAIL


def test_release_gate_is_inconclusive_without_live_pairs():
    result = ReleaseGate.from_toml(CONFIG).evaluate(
        ReasoningQualityReport(), {}, live_pairs=0, mock_llm_reduction=0.6
    )
    assert result.decision is ReleaseDecision.INCONCLUSIVE


def test_release_gate_passes_only_qualified_live_evaluation():
    result = ReleaseGate.from_toml(CONFIG).evaluate(
        ReasoningQualityReport(), {}, live_pairs=2, mock_llm_reduction=0.6
    )
    assert result.decision is ReleaseDecision.PASS
