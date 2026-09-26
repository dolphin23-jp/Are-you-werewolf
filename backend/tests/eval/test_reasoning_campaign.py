from scripts.evaluate_reasoning_campaign import parse_seeds, wilson_interval


def test_seed_range_is_inclusive():
    assert parse_seeds("2:4") == [2, 3, 4]


def test_wilson_interval_contains_observed_ratio():
    low, high = wilson_interval(6, 10)
    assert 0 <= low < 0.6 < high <= 1
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_pooled_reports_recompute_rates_and_averages_instead_of_adding_them():
    """Three games with 4/10, 5/10 and 4/10 changed votes used to aggregate to a
    vote-change rate of 1.30; the pooled rate is 13/30."""
    from app.eval.reasoning_analyzer import ReasoningQualityReport

    games = [
        ReasoningQualityReport(
            vote_count=10,
            justified_vote_change_count=justified,
            unexplained_vote_change_count=unexplained,
            mean_belief_delta_per_affected_seat=sum(deltas) / len(deltas),
            median_belief_delta_per_affected_seat=sorted(deltas)[len(deltas) // 2],
            belief_deltas=deltas,
        ).to_dict()
        for justified, unexplained, deltas in (
            (1, 3, (0.1,)),
            (2, 3, (0.2, 0.4, 0.9)),
            (0, 4, (0.3,)),
        )
    ]

    pooled = ReasoningQualityReport.combine(games)

    assert pooled.vote_count == 30
    assert pooled.vote_change_rate == 13 / 30
    assert pooled.unexplained_vote_change_rate == 10 / 30
    assert pooled.mean_belief_delta_per_affected_seat == (0.1 + 0.2 + 0.4 + 0.9 + 0.3) / 5
    assert pooled.median_belief_delta_per_affected_seat == 0.3


def test_campaign_wins_use_completed_games_only(monkeypatch):
    import asyncio

    import scripts.evaluate_reasoning_campaign as campaign_module

    class _Run:
        def __init__(self, seed: int) -> None:
            self.seed = seed

        def as_dict(self):
            return {
                "engine": "v2",
                "seed": self.seed,
                "winner": None if self.seed == 3 else "village",
                "llm_requests": 1,
                "http_requests": 1,
                "reasoning_quality": {"vote_count": 10, "unexplained_vote_change_count": 2},
            }

    async def fake_play(engine: str, seed: int):
        return _Run(seed)

    monkeypatch.setattr(campaign_module, "play", fake_play)

    result = asyncio.run(campaign_module.campaign([1, 2, 3], ["v2"]))
    v2 = result["engines"]["v2"]

    village = v2["wins_by_team"]["village"]
    assert set(v2["wins_by_team"]) == {"village"}
    assert (village["wins"], village["trials"], village["ratio"]) == (2, 2, 1.0)
    assert v2["reasoning_quality"]["unexplained_vote_change_rate"] == 6 / 30
