"""Call-level budget enforcement for manual live evaluation.

Two ways to spend against it, one per provider generation:

  - `claim_request` + `record_usage` (luna): count the request up front, add
    the tokens afterwards, and price them with the flat per-million prices
    given here. The cost check can only fire *after* the money is spent.
  - `reserve` + `settle` (anthropic): before each HTTP attempt the provider
    reserves that attempt's worst-case cost, priced from its own table, and
    the request is refused *before* it is sent if the reservation could cross
    `max_estimated_cost`; afterwards the reservation is replaced by what the
    attempt actually cost. Concurrent calls each hold their reservation, so a
    burst of sixteen parallel votes cannot all pass a check that only one of
    them fits under.

`BudgetExceeded` is a hard stop, never a degraded result: providers let it
propagate rather than turn it into a None the agent would paper over with a
fallback line.
"""

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
    # Reserve/settle accounting, in USD. `reserved_cost` is what in-flight
    # attempts could still cost at worst; `spent_cost` is what settled
    # attempts did cost.
    spent_cost: float = 0.0
    reserved_cost: float = 0.0
    settled_attempts: int = 0

    def claim_request(self) -> None:
        if self.used_requests >= self.max_requests:
            raise BudgetExceeded(f"HTTP request budget exhausted ({self.max_requests})")
        self.used_requests += 1

    def reserve(self, worst_case_cost: float) -> None:
        """Claim one request and hold its worst-case cost, or refuse to send it.

        Both checks run before anything is recorded, so a refused reservation
        leaves the budget exactly as it was.
        """
        if self.used_requests >= self.max_requests:
            raise BudgetExceeded(f"HTTP request budget exhausted ({self.max_requests})")
        if self.max_estimated_cost is not None:
            committed = self.spent_cost + self.reserved_cost + worst_case_cost
            if committed > self.max_estimated_cost:
                raise BudgetExceeded(
                    "estimated cost budget would be exceeded "
                    f"({committed:.6f} > {self.max_estimated_cost:.6f}: spent "
                    f"{self.spent_cost:.6f}, in flight {self.reserved_cost:.6f}, "
                    f"this request up to {worst_case_cost:.6f})"
                )
        self.used_requests += 1
        self.reserved_cost += worst_case_cost

    def settle(self, reserved: float, actual_cost: float) -> None:
        """Replace a reservation with what the attempt actually cost.

        Never raises: the response has already been paid for, and throwing it
        away would waste the spend. If an under-estimated attempt pushed the
        total past the cap, the next `reserve` refuses.
        """
        self.reserved_cost = max(0.0, self.reserved_cost - reserved)
        self.spent_cost += actual_cost
        self.settled_attempts += 1

    @property
    def pricing_supplied(self) -> bool:
        # A settled attempt was priced by the provider's own table, so the
        # cost is known even without per-million prices given here.
        return self.settled_attempts > 0 or (
            self.input_price_per_million is not None and self.output_price_per_million is not None
        )

    @property
    def estimated_cost(self) -> float | None:
        if self.settled_attempts > 0:
            return self.spent_cost
        if self.input_price_per_million is None or self.output_price_per_million is None:
            return None
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
            "spent_cost": self.spent_cost,
            "reserved_cost": self.reserved_cost,
            "settled_attempts": self.settled_attempts,
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
        """The coordinator finds the collector by duck typing (`provider._metrics`).
        Wrapping a provider in a budget must not hide it, or every event the
        coordinator and agents record would silently vanish."""
        return getattr(self.inner, "_metrics", None)

    @property
    def model(self) -> str | None:
        return getattr(self.inner, "model", None)

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
