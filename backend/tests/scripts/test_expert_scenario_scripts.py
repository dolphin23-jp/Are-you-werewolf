"""The expert cutoff scripts with the real provider's shape, offline.

`build_llm_provider` is replaced by a stand-in that prices and budgets each
call as the Anthropic provider does, so the cap, the labels and the metrics
are exercised without a network.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from app.ai.metrics import CallRecord, MetricsCollector, ParsePath
from app.ai.provider.budget import EvaluationBudget
from app.config import Settings
from scripts import evaluate_expert_scenarios as v1
from scripts import evaluate_expert_scenarios_v2 as v2

_MINIMAL = {
    "ExpertScenarioAnswer": {
        "recommended_action_id": "none",
        "next_observation": "",
        "confidence": "low",
        "rationale": "",
    },
    "ExpertScenarioV2Answer": {
        "world_judgments": [],
        "action_assessments": [],
        "phase_choices": [],
        "next_observation": "",
        "confidence": "low",
        "rationale": "",
    },
}


class _PricedAnswers:
    model = "claude-opus-5-5"

    def __init__(self, metrics: MetricsCollector | None, cost: float) -> None:
        self._metrics = metrics
        self._cost = cost
        self._budget: EvaluationBudget | None = None
        self.systems: list[str] = []

    def set_request_budget(self, budget: EvaluationBudget) -> None:
        self._budget = budget

    async def generate_structured(
        self, *, system: str, messages: Any, response_schema: Any, **_: Any
    ) -> Any:
        if self._budget is not None:
            self._budget.reserve(self._cost)
            self._budget.settle(self._cost, self._cost)
        self.systems.append(system)
        if self._metrics is not None:
            self._metrics.record(
                CallRecord(
                    schema=response_schema.__name__,
                    path=ParsePath.STRICT_SCHEMA,
                    latency_seconds=0.0,
                    attempt=0,
                    http_requests=1,
                    prompt_tokens=1000,
                    completion_tokens=100,
                    estimated_cost=self._cost,
                    served_model=self.model,
                    requested_model=self.model,
                    effort="medium",
                )
            )
        return response_schema.model_validate(_MINIMAL[response_schema.__name__])


def _install(monkeypatch: pytest.MonkeyPatch, cost: float) -> list[_PricedAnswers]:
    built: list[_PricedAnswers] = []

    def factory(
        settings: Settings, *, seed: int | None = None, metrics: MetricsCollector | None = None
    ) -> _PricedAnswers:
        assert settings.werewolf_llm_provider == "anthropic"
        provider = _PricedAnswers(metrics, cost)
        built.append(provider)
        return provider

    monkeypatch.setattr(v1, "build_llm_provider", factory)
    return built


@pytest.mark.parametrize("script", [v1, v2], ids=["v1", "v2"])
def test_anthropic_needs_a_cap(
    script: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stopped:
        asyncio.run(script.main(["--provider", "anthropic", "--out", str(tmp_path)]))

    assert stopped.value.code == 2
    assert "--max-cost" in capsys.readouterr().err


@pytest.mark.parametrize("script", [v1, v2], ids=["v1", "v2"])
def test_luna_cap_without_prices_is_refused(script: Any, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as stopped:
        asyncio.run(
            script.main(["--provider", "luna", "--max-cost", "3", "--out", str(tmp_path)])
        )

    assert stopped.value.code == 2


@pytest.mark.parametrize("script", [v1, v2], ids=["v1", "v2"])
def test_anthropic_answers_are_labelled_metered_and_budgeted(
    script: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = _install(monkeypatch, cost=0.01)

    code = asyncio.run(
        script.main(["--provider", "anthropic", "--max-cost", "5", "--out", str(tmp_path)])
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert code in (0, 1)  # scored; whether it clears --fail-under is not the point
    assert summary["model"] == "claude-opus-5-5"
    assert summary["metrics"]["total_calls"] == len(built[0].systems) > 0
    assert summary["metrics"]["pricing_source"] == "table"
    assert summary["budget"]["spent_cost"] == pytest.approx(0.01 * len(built[0].systems))
    assert summary["stopped_by_budget"] is False
    # Same prompt contract as luna: the schema travels in the system prompt.
    assert all("BEGIN_RESPONSE_JSON_SCHEMA" in system for system in built[0].systems)


@pytest.mark.parametrize("script", [v1, v2], ids=["v1", "v2"])
def test_the_cap_leaves_the_rest_unanswered_and_exits_3(
    script: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = _install(monkeypatch, cost=1.0)

    code = asyncio.run(
        script.main(
            [
                "--provider",
                "anthropic",
                "--max-cost",
                "1.5",
                "--concurrency",
                "1",
                "--out",
                str(tmp_path),
            ]
        )
    )

    assert code == 3
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert summary["stopped_by_budget"] is True
    assert len(built[0].systems) == 1  # the second answer was never sent
    assert summary["budget"]["spent_cost"] <= 1.5
    assert report.startswith("> **費用上限に達したため中断")
