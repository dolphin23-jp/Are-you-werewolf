"""Call-level budget enforcement for manual live evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from app.ai.provider.base import LLMProvider, Message, SchemaT


class BudgetExceeded(BaseException):
    """Stop the run. Raised deep inside provider calls, so it must get past every
    `except Exception` that keeps a game alive on one failed generation.

    As a RuntimeError it was swallowed by the agent (`return None`) and by the
    provider's own retry handling: the game carried on with fallback turns,
    every later call was still sent and billed, and the caller's
    `except BudgetExceeded` never ran. `asyncio.CancelledError` is a
    BaseException for the same reason.
    """


@dataclass
class EvaluationBudget:
    """Request and estimated-cost caps, both checked *before* a request is sent.

    The cost cap used to be checked only once a response had come back, and its
    exception thrown away with the paid response. Now `record_usage` only
    accumulates and `claim_request` refuses the next request, so a run can
    exceed the cap by at most one response and never discards one it paid for.
    """

    max_requests: int
    used_requests: int = 0
    max_estimated_cost: float | None = None
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __post_init__(self) -> None:
        if self.max_estimated_cost is not None and not self.pricing_supplied:
            # Without prices the estimate is unknown, and an unknown estimate
            # never reaches the cap: the cost limit would silently not exist.
            raise ValueError(
                "a cost cap needs --input-price-per-million and --output-price-per-million "
                "(or LLM_INPUT_PRICE_PER_MILLION / LLM_OUTPUT_PRICE_PER_MILLION)"
            )

    def claim_request(self) -> None:
        if self.used_requests >= self.max_requests:
            raise BudgetExceeded(f"HTTP request budget exhausted ({self.max_requests})")
        cost = self.estimated_cost
        if (
            cost is not None
            and self.max_estimated_cost is not None
            and cost >= self.max_estimated_cost
        ):
            raise BudgetExceeded(
                f"estimated cost budget exhausted ({cost:.6f} >= {self.max_estimated_cost:.6f})"
            )
        self.used_requests += 1

    @property
    def pricing_supplied(self) -> bool:
        return (
            self.input_price_per_million is not None and self.output_price_per_million is not None
        )

    @property
    def estimated_cost(self) -> float | None:
        if not self.pricing_supplied:
            return None
        assert self.input_price_per_million is not None
        assert self.output_price_per_million is not None
        return (
            self.prompt_tokens * self.input_price_per_million
            + self.completion_tokens * self.output_price_per_million
        ) / 1_000_000

    def record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    def snapshot(self) -> dict[str, int | float | bool | None]:
        return {
            "max_requests": self.max_requests,
            "used_requests": self.used_requests,
            "max_estimated_cost": self.max_estimated_cost,
            "input_price_per_million": self.input_price_per_million,
            "output_price_per_million": self.output_price_per_million,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "estimated_cost": self.estimated_cost,
            "pricing_supplied": self.pricing_supplied,
        }


class BudgetedProvider:
    """Conservative proxy: every logical generation claims at least one HTTP request."""

    def __init__(self, inner: LLMProvider, budget: EvaluationBudget) -> None:
        self.inner = inner
        self.budget = budget
        attach = getattr(inner, "set_request_budget", None)
        self._claims_exact_requests = callable(attach)
        if self._claims_exact_requests:
            cast(Callable[[EvaluationBudget], None], attach)(budget)

    @property
    def _metrics(self) -> object | None:
        # The coordinator records discussion attempts and skips on the provider's
        # metrics. Hidden behind this proxy, every budgeted run reported a 0%
        # skip rate, so the gate's skip-rate threshold could never fail.
        return getattr(self.inner, "_metrics", None)

    async def generate_structured(
        self,
        *,
        system: str,
        messages: list[Message],
        response_schema: type[SchemaT],
        max_tokens: int = 800,
        temperature: float = 0.9,
    ) -> SchemaT | None:
        if not self._claims_exact_requests:
            self.budget.claim_request()
        return await self.inner.generate_structured(
            system=system,
            messages=messages,
            response_schema=response_schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )
