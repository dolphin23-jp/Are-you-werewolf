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
    refusal_fallback_models,
    worst_case_price,
)
from app.config import DEFAULT_ANTHROPIC_MODEL


def test_the_default_model_has_a_known_price():
    # A default the table did not know would be priced at its dearest row.
    assert price_for(DEFAULT_ANTHROPIC_MODEL)[1] is True


def test_opus_5_5_has_its_verified_list_price():
    price, known = price_for("claude-opus-5-5")
    assert known is True
    assert price == ModelPrice(
        input=4.0, cache_write_5m=5.0, cache_write_1h=8.0, cache_read=0.20, output=20.0
    )


def test_sonnet_5_5_has_its_verified_list_price():
    price, known = price_for("claude-sonnet-5-5")
    assert known is True
    assert price == ModelPrice(
        input=2.0, cache_write_5m=2.50, cache_write_1h=4.0, cache_read=0.10, output=10.0
    )


def test_sonnet_5_the_sonnet_fallback_has_its_own_row():
    # Same rates as Sonnet 5.5 except cache reads, at twice Sonnet 5.5's.
    assert price_for("claude-sonnet-5") == (ModelPrice(2.0, 2.50, 4.0, 0.20, 10.0), True)
    # `claude-sonnet-5` is a prefix of `claude-sonnet-5-5`; the longer key wins.
    assert price_for("claude-sonnet-5-5-20261101")[0] == STANDARD_PRICES["claude-sonnet-5-5"]
    assert price_for("claude-sonnet-5-20260601")[0] == STANDARD_PRICES["claude-sonnet-5"]


@pytest.mark.parametrize(
    ("requested", "fallbacks"),
    [
        ("claude-sonnet-5-5", ("claude-sonnet-5",)),
        ("claude-sonnet-5-5-20261101", ("claude-sonnet-5",)),
        ("claude-opus-5-5", ("claude-opus-5", "claude-opus-4-8")),
        # No server-side fallback exists for Haiku 5.5.
        ("claude-haiku-5-5", ()),
    ],
)
def test_the_refusal_fallbacks_follow_the_requested_model(
    requested: str, fallbacks: tuple[str, ...]
):
    assert refusal_fallback_models(requested) == fallbacks


def test_an_unknown_model_may_fall_back_to_any_known_fallback_model():
    union = refusal_fallback_models("claude-opus-9")
    assert set(union) == {"claude-sonnet-5", "claude-opus-5", "claude-opus-4-8"}


def test_every_fallback_model_is_priced():
    for model in refusal_fallback_models("an-unknown-model"):
        assert price_for(model)[1] is True, model


def test_the_worst_case_price_covers_the_requested_model_and_its_fallbacks():
    sonnet = worst_case_price("claude-sonnet-5-5", fallbacks=True)
    assert sonnet == ModelPrice(2.0, 2.50, 4.0, 0.20, 10.0)
    assert worst_case_price("claude-sonnet-5-5", fallbacks=False) == (
        STANDARD_PRICES["claude-sonnet-5-5"]
    )
    assert worst_case_price("claude-opus-5-5", fallbacks=True) == STANDARD_PRICES["claude-opus-5"]
    # An unknown model is priced at the dearest row, and may fall back anywhere.
    dearest = max(STANDARD_PRICES.values(), key=lambda p: p.output)
    assert worst_case_price("claude-opus-9", fallbacks=True) == dearest.elementwise_max(
        STANDARD_PRICES["claude-opus-5"]
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
