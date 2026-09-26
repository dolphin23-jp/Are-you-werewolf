import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.ai.coordinator import AICoordinator
from app.ai.provider.base import Message
from app.ai.provider.budget import BudgetedProvider, BudgetExceeded, EvaluationBudget
from app.ai.provider.mock import MockProvider
from app.eval.release_report import REVIEW_ITEMS, HumanTranscriptReview, empty_review
from app.eval.transcript import TranscriptRecorder, WolfAllyVotePlan
from tests.conftest import make_controller


def test_budget_stops_before_the_request_beyond_limit():
    budget = EvaluationBudget(max_requests=2)
    budget.claim_request()
    budget.claim_request()
    with pytest.raises(BudgetExceeded):
        budget.claim_request()
    assert budget.used_requests == 2


def test_budget_refuses_the_next_request_once_the_cost_cap_is_reached():
    budget = EvaluationBudget(
        max_requests=10,
        max_estimated_cost=0.001,
        input_price_per_million=1.0,
        output_price_per_million=1.0,
    )
    budget.claim_request()
    # The response is paid for, so it is recorded and kept, not thrown away.
    budget.record_usage(600, 600)
    assert budget.estimated_cost == pytest.approx(0.0012)
    with pytest.raises(BudgetExceeded):
        budget.claim_request()
    assert budget.used_requests == 1
    assert budget.snapshot()["pricing_supplied"] is True


def test_a_cost_cap_without_prices_is_refused_up_front():
    with pytest.raises(ValueError):
        EvaluationBudget(max_requests=10, max_estimated_cost=5.0)


def test_budget_exhaustion_is_not_swallowed_by_generic_handlers():
    # Every layer between a provider call and the run keeps the game alive with
    # `except Exception`; the budget stop has to pass through all of them.
    budget = EvaluationBudget(max_requests=0)
    with pytest.raises(BudgetExceeded):
        try:
            budget.claim_request()
        except Exception:
            pytest.fail("BudgetExceeded was caught as an ordinary Exception")


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


def test_a_budgeted_game_stops_when_the_budget_runs_out():
    """Exhaustion has to end the game, not leave it running on fallback turns.

    It used to be caught by the agent and the game played on to the end, with
    the run counted as a completed live game.
    """
    controller = make_controller(seed=1)
    budget = EvaluationBudget(max_requests=3)
    coordinator = AICoordinator(
        controller.state,
        [f"p{i}" for i in range(1, 17)],
        BudgetedProvider(MockProvider(seed=1), budget),
        seed=1,
        pacing_scale=0.0,
    )
    session = SimpleNamespace(
        controller=controller,
        coordinator=coordinator,
        human_id="p0",
        discussion_lock=asyncio.Lock(),
        discussion_round=None,
        discussion_paused=False,
        discussion_pause_requested=False,
        discussion_step_budget=None,
    )

    async def play() -> None:
        controller.start_game()
        await coordinator.run_night_phase(session)
        controller.start_discussion()
        await coordinator.run_discussion_round(session)

    with pytest.raises(BudgetExceeded):
        asyncio.run(play())
    assert budget.used_requests == 3
