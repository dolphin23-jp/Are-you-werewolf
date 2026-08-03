from scripts.evaluate_reasoning_campaign import parse_seeds, wilson_interval


def test_seed_range_is_inclusive():
    assert parse_seeds("2:4") == [2, 3, 4]


def test_wilson_interval_contains_observed_ratio():
    low, high = wilson_interval(6, 10)
    assert 0 <= low < 0.6 < high <= 1
    assert wilson_interval(0, 0) == (0.0, 0.0)
