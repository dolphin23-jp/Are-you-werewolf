"""Configuration-driven, deliberately conservative v2 release decision."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.eval.reasoning_analyzer import ReasoningQualityReport


class ReleaseDecision(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


HARD_FAILURE_FIELDS = (
    "private_evidence_exposed_count",
    "team_private_evidence_exposed_count",
    "dead_target_selection_count",
    "public_fact_flip_count",
    "missing_required_result_count",
    "duplicate_required_result_count",
    "displayed_target_mismatch_count",
    "stale_evidence_publicly_emitted_count",
    "unexplained_vote_change_count",
    "unplanned_wolf_ally_vote_count",
    "duplicate_night_action_count",
)


@dataclass(frozen=True)
class ReleaseGateResult:
    decision: ReleaseDecision
    reasons: tuple[str, ...]


class ReleaseGate:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @classmethod
    def from_toml(cls, path: str | Path) -> ReleaseGate:
        with Path(path).open("rb") as stream:
            return cls(tomllib.load(stream))

    def evaluate(
        self,
        report: ReasoningQualityReport,
        operational: dict[str, float],
        *,
        live_pairs: int,
        mock_llm_reduction: float | None = None,
        transcript_schema_version: int = 2,
        human_review_complete: bool = False,
        operational_complete: bool = False,
    ) -> ReleaseGateResult:
        failures = [field for field in HARD_FAILURE_FIELDS if getattr(report, field) > 0]
        reliability = self.config.get("reliability", {})
        if operational.get("complete_failure_rate", 0.0) > reliability.get(
            "max_complete_failure_rate", 0.01
        ):
            failures.append("complete_failure_rate")
        if operational.get("discussion_skip_rate", 0.0) > reliability.get(
            "max_discussion_skip_rate", 0.02
        ):
            failures.append("discussion_skip_rate")
        required_reduction = self.config.get("efficiency", {}).get(
            "min_mock_logical_call_reduction", 0.5
        )
        if mock_llm_reduction is not None and mock_llm_reduction < required_reduction:
            failures.append("mock_logical_call_reduction")
        if failures:
            return ReleaseGateResult(ReleaseDecision.FAIL, tuple(failures))
        incomplete: list[str] = []
        if transcript_schema_version < 2:
            incomplete.append("transcript_schema_version")
        if not operational_complete:
            incomplete.append("operational_metrics_missing")
        if not human_review_complete:
            incomplete.append("human_review_incomplete")
        minimum_pairs = self.config.get("qualification", {}).get("minimum_live_pairs", 2)
        if live_pairs < minimum_pairs:
            incomplete.append(f"live_pairs={live_pairs}; required={minimum_pairs}")
        if incomplete:
            return ReleaseGateResult(ReleaseDecision.INCONCLUSIVE, tuple(incomplete))
        return ReleaseGateResult(ReleaseDecision.PASS, ())
