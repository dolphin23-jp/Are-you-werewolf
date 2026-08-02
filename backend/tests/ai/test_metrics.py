from __future__ import annotations

from app.ai.metrics import CallRecord, MetricsCollector, ParsePath


def _record(path: ParsePath, latency: float = 0.1, **kwargs) -> CallRecord:
    return CallRecord(schema="VoteOutput", path=path, latency_seconds=latency, attempt=0, **kwargs)


def test_empty_collector_reports_no_calls():
    assert MetricsCollector().summary() == {"total_calls": 0}


def test_success_rate_separates_strict_schema_from_fallbacks():
    collector = MetricsCollector()
    collector.record(_record(ParsePath.STRICT_SCHEMA))
    collector.record(_record(ParsePath.JSON_OBJECT))
    collector.record(_record(ParsePath.PERMISSIVE))
    collector.record(_record(ParsePath.FAILED, error="boom"))

    summary = collector.summary()
    assert summary["total_calls"] == 4
    assert summary["success_rate"] == 0.75
    # The distinction that matters: only one call honoured strict schema,
    # so a 75% "success" rate is being carried by the fallback path.
    assert summary["strict_schema_rate"] == 0.25
    assert summary["parse_path_counts"]["permissive"] == 1
    assert summary["errors"] == [{"error": "boom", "count": 1}]


def test_discussion_skip_rate_is_reported():
    collector = MetricsCollector()
    collector.record(_record(ParsePath.FAILED))
    collector.record_discussion_result(skipped=True)
    collector.record_discussion_result(skipped=False)

    summary = collector.summary()
    assert summary["discussion_skips"] == 1
    assert summary["discussion_skip_rate"] == 0.5


def test_tokens_and_cost_are_none_when_the_endpoint_reports_no_usage():
    collector = MetricsCollector()
    collector.record(_record(ParsePath.STRICT_SCHEMA))
    summary = collector.summary(price_per_1m_input=1.0, price_per_1m_output=2.0)
    # Reporting 0 here would read as "this run was free", which is a
    # materially different claim from "usage was not reported".
    assert summary["tokens"] is None
    assert summary["estimated_cost"] is None


def test_cost_is_computed_from_reported_tokens():
    collector = MetricsCollector()
    collector.record(
        _record(ParsePath.STRICT_SCHEMA, prompt_tokens=1_000_000, completion_tokens=500_000)
    )
    summary = collector.summary(price_per_1m_input=2.0, price_per_1m_output=4.0)
    assert summary["tokens"]["total"] == 1_500_000
    assert summary["estimated_cost"] == 4.0  # 1.0*2 + 0.5*4
    assert summary["pricing_supplied"] is True


def test_latency_percentiles_and_per_schema_breakdown():
    collector = MetricsCollector()
    for latency in (0.1, 0.2, 0.3, 0.4, 10.0):
        collector.record(_record(ParsePath.STRICT_SCHEMA, latency=latency))
    collector.record(
        CallRecord(schema="DiscussionOutput", path=ParsePath.FAILED, latency_seconds=1.0, attempt=1)
    )

    summary = collector.summary()
    assert summary["latency_seconds"]["max"] == 10.0
    assert summary["latency_seconds"]["p50"] <= summary["latency_seconds"]["p95"]
    assert summary["by_schema"]["DiscussionOutput"]["success_rate"] == 0.0
    assert summary["by_schema"]["VoteOutput"]["calls"] == 5
    assert summary["retry_calls"] == 1
