"""Pricing table lookups and USD arithmetic (app/ai/provider/pricing.py)."""

from __future__ import annotations

import dataclasses

import pytest

from app.ai.provider.pricing import (
    FAST_PRICES,
    STANDARD_PRICES,
    ModelPrice,
    cost_usd,
    is_same_model,
    price_for,
)


def test_the_default_model_has_its_verified_list_price():
    price, known = price_for("claude-opus-5-5")
    assert known is True
    assert price == ModelPrice(
        input=4.0, cache_write_5m=5.0, cache_write_1h=8.0, cache_read=0.20, output=20.0
    )


@pytest.mark.parametrize("model", sorted(STANDARD_PRICES))
def test_every_table_row_is_found_by_its_exact_id(model: str):
    assert price_for(model) == (STANDARD_PRICES[model], True)


def test_a_dated_snapshot_resolves_to_the_longest_matching_row():
    # `claude-opus-5` is a prefix of both ids; the longer key must win.
    assert price_for("claude-opus-5-5-20261101") == (STANDARD_PRICES["claude-opus-5-5"], True)
    assert price_for("claude-opus-5-20260601") == (STANDARD_PRICES["claude-opus-5"], True)
    assert price_for(" Claude-Opus-5-5 ")[1] is True


def test_an_unknown_model_is_priced_at_the_dearest_row_and_flagged():
    dearest = max(STANDARD_PRICES.values(), key=lambda p: p.output)
    assert price_for("claude-opus-9") == (dearest, False)
    # A minor-version-looking suffix is a different model, not a snapshot --
    # pricing it as its predecessor would be a silent guess.
    assert price_for("claude-opus-5-1") == (dearest, False)
    assert price_for("gpt-6-luna") == (dearest, False)


def test_fast_mode_has_its_own_rows():
    assert price_for("claude-opus-5-5", speed="fast") == (FAST_PRICES["claude-opus-5-5"], True)
    assert price_for("claude-opus-5-5", speed="standard") == (
        STANDARD_PRICES["claude-opus-5-5"],
        True,
    )
    assert price_for("claude-sonnet-5-5", speed="fast")[1] is False


def test_cost_prices_each_token_kind_at_its_own_rate():
    price = STANDARD_PRICES["claude-opus-5-5"]
    cost = cost_usd(
        price,
        input_tokens=1_000_000,
        cache_creation_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    # Cache writes at the 5-minute rate: the only TTL this application uses.
    assert cost == pytest.approx(4.0 + 5.0 + 0.20 + 20.0)
    assert cost_usd(price) == 0.0


def test_prices_are_immutable_and_combine_into_a_worst_case():
    price = STANDARD_PRICES["claude-opus-5-5"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        price.input = 0.0  # type: ignore[misc]
    worst = price.elementwise_max(STANDARD_PRICES["claude-opus-5"])
    assert worst == ModelPrice(5.0, 6.25, 10.0, 0.50, 25.0)


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("claude-opus-5-5", "claude-opus-5-5", True),
        ("claude-opus-5-5", "claude-opus-5-5-20261101", True),
        ("CLAUDE-OPUS-5-5", "claude-opus-5-5", True),
        # The refusal fallback: a prefix, and a different model.
        ("claude-opus-5-5", "claude-opus-5", False),
        ("claude-opus-5-5", "claude-sonnet-5-5", False),
    ],
)
def test_is_same_model(a: str, b: str, same: bool):
    assert is_same_model(a, b) is same
    assert is_same_model(b, a) is same
