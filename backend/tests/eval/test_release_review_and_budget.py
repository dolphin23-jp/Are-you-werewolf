import asyncio

import pytest
from pydantic import BaseModel

from app.ai.provider.base import Message
from app.ai.provider.budget import BudgetedProvider, BudgetExceeded, EvaluationBudget
from app.eval.release_report import REVIEW_ITEMS, HumanTranscriptReview, empty_review
from app.eval.transcript import TranscriptRecorder, WolfAllyVotePlan


def test_budget_stops_before_the_request_beyond_limit():
    budget = EvaluationBudget(max_requests=2)
    budget.claim_request()
    budget.claim_request()
    with pytest.raises(BudgetExceeded):
        budget.claim_request()
    assert budget.used_requests == 2


def test_budget_tracks_usage_and_rejects_cost_overrun():
    budget = EvaluationBudget(
        max_requests=10,
        max_estimated_cost=0.001,
        input_price_per_million=1.0,
        output_price_per_million=1.0,
    )
    with pytest.raises(BudgetExceeded):
        budget.record_usage(600, 600)
    assert budget.estimated_cost == pytest.approx(0.0012)
    assert budget.snapshot()["pricing_supplied"] is True


def test_missing_prices_are_unknown_not_free():
    budget = EvaluationBudget(max_requests=1)
    budget.record_usage(100, 100)
    assert budget.estimated_cost is None
    assert budget.pricing_supplied is False


def test_review_is_complete_only_when_every_answer_is_present(tmp_path):
    review = empty_review("game-1")
    assert review.complete is False
    completed = HumanTranscriptReview(
        game_id="game-1",
        reviewer="reviewer",
        reviewed_at="2026-08-03T00:00:00Z",
        answers={item: True for item in REVIEW_ITEMS},
    )
    path = tmp_path / "review.json"
    completed.write_json(path)
    assert HumanTranscriptReview.from_json(path).complete is True


def test_exact_request_provider_owns_budget_claims():
    class Output(BaseModel):
        value: str

    class ExactProvider:
        def set_request_budget(self, budget):
            self.budget = budget

        async def generate_structured(self, **kwargs):
            self.budget.claim_request()
            self.budget.claim_request()  # strict failure plus fallback
            return Output(value="ok")

    budget = EvaluationBudget(2)
    provider = BudgetedProvider(ExactProvider(), budget)
    result = asyncio.run(
        provider.generate_structured(
            system="", messages=[Message(role="user", content="x")], response_schema=Output
        )
    )
    assert result == Output(value="ok")
    assert budget.used_requests == 2


def test_wolf_ally_plan_requires_complete_audit_basis():
    recorder = TranscriptRecorder()
    complete = WolfAllyVotePlan(
        "plan", "p1", "p2", 2, 1, ("public:pressure",),
        "公開物語を維持する", "身内切りを選択", "wolf-chat-1",
    )
    recorder.record_wolf_ally_vote_plan(complete)
    assert recorder.matching_wolf_ally_vote_plan("p1", "p2", 2, 1) == complete
    assert recorder.matching_wolf_ally_vote_plan("p1", "p3", 2, 1) is None
