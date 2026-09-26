"""The live release gate counts only evidence it can trust.

Before this, an offline `scenario` game, a game cut off at the loop limit and a
review that answered "no" to every question all counted toward PASS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.ai.provider.budget import EvaluationBudget
from app.eval.release_report import REVIEW_ITEMS, HumanTranscriptReview
from scripts.live_ab_reasoning_check import _save_aggregate


def _row(seed: int, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "seed": seed,
        "engine": "v2",
        "game_id": f"live-{seed}-v2-abcd{seed}",
        "provider": "luna",
        "game_over": True,
        "reasoning_quality": {},
        "operational_metrics": {"total_calls": 10, "complete_failures": 0},
    }
    row.update(overrides)
    return row


def _review(directory: Path, game_id: str, answer: bool) -> None:
    HumanTranscriptReview(
        game_id=game_id,
        reviewer="reviewer",
        reviewed_at="2026-09-26T00:00:00Z",
        answers={item: answer for item in REVIEW_ITEMS},
    ).write_json(directory / f"{game_id}.json")


def _decide(tmp_path: Path, rows: list[dict[str, Any]], answer: bool = True) -> dict[str, Any]:
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    for row in rows:
        _review(reviews, row["game_id"], answer)
    budget = EvaluationBudget(
        max_requests=100,
        max_estimated_cost=5.0,
        input_price_per_million=1.0,
        output_price_per_million=1.0,
    )
    _save_aggregate(tmp_path, rows, budget, reviews)
    return json.loads((tmp_path / "aggregate.json").read_text(encoding="utf-8"))


def test_two_finished_live_games_with_approving_reviews_pass(tmp_path):
    decision = _decide(tmp_path, [_row(11), _row(12)])
    assert decision["release_decision"] == "pass"


def test_reviews_that_answer_no_fail_the_gate(tmp_path):
    decision = _decide(tmp_path, [_row(11), _row(12)], answer=False)
    assert decision["release_decision"] == "fail"
    assert "human_review_rejected" in decision["release_reasons"]


def test_offline_doubles_and_unfinished_games_are_not_live_evidence(tmp_path):
    decision = _decide(
        tmp_path, [_row(11, provider="scenario"), _row(12, game_over=False)]
    )
    assert decision["release_decision"] == "inconclusive"
    assert any(reason.startswith("live_games=0") for reason in decision["release_reasons"])
