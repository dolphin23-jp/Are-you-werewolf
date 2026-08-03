"""Call-level budget enforcement for manual live evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from app.ai.provider.base import LLMProvider, Message, SchemaT


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class EvaluationBudget:
    max_requests: int
    used_requests: int = 0
    max_estimated_cost: float | None = None
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def claim_request(self) -> None:
        if self.used_requests >= self.max_requests:
            raise BudgetExceeded(f"HTTP request budget exhausted ({self.max_requests})")
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
        cost = self.estimated_cost
        if (
            cost is not None
            and self.max_estimated_cost is not None
            and cost > self.max_estimated_cost
        ):
            raise BudgetExceeded(
                f"estimated cost budget exhausted ({cost:.6f} > {self.max_estimated_cost:.6f})"
            )

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
