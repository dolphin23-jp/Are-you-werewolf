"""Claude list prices, and the arithmetic that turns a response's usage into USD.

Kept apart from the provider so the budget, the metrics summary and the
evaluation reports all price a call the same way -- a cost that two modules
compute independently is a cost that two modules eventually disagree on.

Every figure is USD per million tokens. Sources:

  - Verified on the live pricing page
    (platform.claude.com/docs/en/about-claude/pricing) on 2026-11-15:
    `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`.
  - Taken from the bundled claude-api skill reference, not re-fetched:
    `claude-opus-5`, `claude-opus-4-8`, `claude-sonnet-5` (the models a
    server-side refusal fallback may serve a turn with), `claude-fable-5-2`,
    and Opus 5.5 fast mode.

A price here can go stale without any error, so a model this table does not
know is priced at the most expensive standard row and reported as unknown
(`price_for(...)[1] is False`): over-estimating spend stops a budgeted run
early, under-estimating it lets one overspend.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, fields

_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens for each way a token can be billed."""

    input: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float
    output: float

    def elementwise_max(self, other: ModelPrice) -> ModelPrice:
        """The dearer of two prices in every column -- a worst case when it is
        not known in advance which of two models will serve a request."""
        return ModelPrice(
            *(max(getattr(self, f.name), getattr(other, f.name)) for f in fields(self))
        )


# Standard-speed prices. Keys are exact model ids; a dated snapshot such as
# `claude-opus-5-5-20261101` resolves to its undated row (see `price_for`).
STANDARD_PRICES: dict[str, ModelPrice] = {
    # Verified live 2026-11-15.
    "claude-opus-5-5": ModelPrice(4.0, 5.0, 8.0, 0.20, 20.0),
    "claude-sonnet-5-5": ModelPrice(2.0, 2.50, 4.0, 0.10, 10.0),
    # The listed rate for prompts up to 100k tokens. Werewolf prompts are far
    # below that; a longer prompt would be under-priced by this row.
    "claude-haiku-5-5": ModelPrice(0.10, 0.125, 0.20, 0.01, 0.50),
    # From the skill reference.
    "claude-opus-5": ModelPrice(5.0, 6.25, 10.0, 0.50, 25.0),
    "claude-opus-4-8": ModelPrice(5.0, 6.25, 10.0, 0.50, 25.0),
    # Sonnet 5.5's prices, except cache reads at twice Sonnet 5.5's rate
    # (skill reference, "Migrating to Claude Sonnet 5.5").
    "claude-sonnet-5": ModelPrice(2.0, 2.50, 4.0, 0.20, 10.0),
    "claude-fable-5-2": ModelPrice(8.0, 10.0, 16.0, 0.20, 40.0),
}

# `speed="fast"` prices (from the skill reference; 2x Opus 5.5 standard).
FAST_PRICES: dict[str, ModelPrice] = {
    "claude-opus-5-5": ModelPrice(8.0, 10.0, 16.0, 0.40, 40.0),
}

# The models a `fallbacks="default"` turn may be re-run on after a refusal, by
# the model that was asked (skill reference): a declined Sonnet 5.5 turn goes
# to Sonnet 5, a declined Opus 5.5 turn to Opus 5 or Opus 4.8, and Haiku 5.5
# has no server-side fallback. Keyed per requested model because one fixed
# Opus pair priced every Sonnet call's worst case at Opus 5 rates -- 2.5x the
# output rate -- and a per-game cap then refused calls far below real spend.
_REFUSAL_FALLBACKS_BY_MODEL: dict[str, tuple[str, ...]] = {
    "claude-sonnet-5-5": ("claude-sonnet-5",),
    "claude-opus-5-5": ("claude-opus-5", "claude-opus-4-8"),
    "claude-haiku-5-5": (),
}

# What a model id may carry after a known table key and still be that model:
# a dated snapshot (`-20261101`), optionally followed by anything a platform
# appends (`-v1:0`, `@...`). A bare `-1` is NOT accepted -- that is a different
# model (`claude-opus-5` + `-1` would be a hypothetical Opus 5.1), and pricing
# it as its predecessor would be a silent guess.
_SNAPSHOT_SUFFIX_RE = re.compile(r"-\d{8}(?:\b.*)?")


def is_same_model(a: str, b: str) -> bool:
    """Whether two model ids name the same model, a dated snapshot included.

    Not a plain prefix test: `claude-opus-5` is a prefix of `claude-opus-5-5`
    and is a different (and dearer) model -- exactly what a refusal fallback
    serves a turn with.
    """
    a, b = a.strip().lower(), b.strip().lower()
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    return longer.startswith(shorter) and bool(
        _SNAPSHOT_SUFFIX_RE.fullmatch(longer[len(shorter) :])
    )


def _most_expensive(table: dict[str, ModelPrice]) -> ModelPrice:
    return max(table.values(), key=lambda p: (p.output, p.input, p.cache_write_1h))


_UNKNOWN_MODEL_PRICE = _most_expensive(STANDARD_PRICES)


def _table_key(model: str, keys: Iterable[str]) -> str | None:
    """The table key a model id names: itself, or the longest key it extends
    by a snapshot suffix. `claude-opus-5` is a prefix of both
    `claude-opus-5-5-20261101` and `claude-opus-5-20260601`, which is why the
    longest key wins."""
    normalized = model.strip().lower()
    keys = list(keys)
    if normalized in keys:
        return normalized
    candidates = [
        key
        for key in keys
        if normalized.startswith(key) and _SNAPSHOT_SUFFIX_RE.fullmatch(normalized[len(key) :])
    ]
    return max(candidates, key=len) if candidates else None


def price_for(model: str, speed: str | None = None) -> tuple[ModelPrice, bool]:
    """Return `(price, known)` for a model id as the API reports it.

    Matches the exact id first, then the longest table key the id extends by
    a snapshot suffix, so `claude-opus-5-5-20261101` is Opus 5.5 while
    `claude-opus-5-20260601` is Opus 5. Anything else is unknown and priced at
    the most expensive standard row.
    """
    table = FAST_PRICES if speed == "fast" else STANDARD_PRICES
    key = _table_key(model, table)
    if key is not None:
        return table[key], True
    return _UNKNOWN_MODEL_PRICE, False


def refusal_fallback_models(model: str) -> tuple[str, ...]:
    """The models a refused turn of `model` may be re-served by.

    A model this module has no fallback route for may be routed anywhere, so
    it gets every fallback model listed: an over-reservation stops a budgeted
    run early, an under-reservation lets one call cross its cap.
    """
    key = _table_key(model, _REFUSAL_FALLBACKS_BY_MODEL)
    if key is not None:
        return _REFUSAL_FALLBACKS_BY_MODEL[key]
    union: list[str] = []
    for fallbacks in _REFUSAL_FALLBACKS_BY_MODEL.values():
        union.extend(fallback for fallback in fallbacks if fallback not in union)
    return tuple(union)


def worst_case_price(model: str, *, fallbacks: bool) -> ModelPrice:
    """The dearest rate, column by column, any attempt of one call can bill.

    With refusal fallbacks on, the requested model's attempt may be followed
    by one on a fallback model, and which one is not known in advance.
    """
    price = price_for(model)[0]
    if fallbacks:
        for fallback in refusal_fallback_models(model):
            price = price.elementwise_max(price_for(fallback)[0])
    return price


def cost_usd(
    price: ModelPrice,
    *,
    input_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    output_tokens: int = 0,
) -> float:
    """USD for one response's usage.

    `input_tokens` is the *uncached* input only, exactly as the API reports
    it; cache writes and reads are separate counts with their own rates.
    Cache writes are priced at the 5-minute rate because this application
    only ever uses the default TTL. Thinking tokens are already inside
    `output_tokens` and billed as output.
    """
    return (
        input_tokens * price.input
        + cache_creation_tokens * price.cache_write_5m
        + cache_read_tokens * price.cache_read
        + output_tokens * price.output
    ) / _PER_MILLION
