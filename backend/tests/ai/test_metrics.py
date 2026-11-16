from __future__ import annotations

import threading

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


# --- events -------------------------------------------------------------------


def test_events_are_counted_and_reported():
    collector = MetricsCollector()
    collector.record(_record(ParsePath.STRICT_SCHEMA))
    collector.record_event("vote_code_resolved")
    collector.record_event("vote_code_resolved")
    collector.record_event("brief_fallback")

    assert collector.summary()["events"] == {"brief_fallback": 1, "vote_code_resolved": 2}


def test_events_survive_a_run_with_no_llm_calls():
    """Everything resolved in code is exactly when the events matter most."""
    collector = MetricsCollector()
    collector.record_event("night_code_resolved")
    assert collector.summary() == {"total_calls": 0, "events": {"night_code_resolved": 1}}


def test_record_event_is_thread_safe():
    collector = MetricsCollector()

    def hammer() -> None:
        for _ in range(1000):
            collector.record_event("discussion_stalled")

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert collector.events == {"discussion_stalled": 8000}


# --- anthropic usage, refusals and served models --------------------------------


def test_cache_usage_and_share():
    collector = MetricsCollector()
    collector.record(
        _record(
            ParsePath.STRICT_SCHEMA,
            prompt_tokens=1000,
            completion_tokens=10,
            cache_creation_tokens=200,
            cache_read_tokens=600,
        )
    )
    summary = collector.summary()
    assert summary["cache_creation_tokens"] == 200
    assert summary["cache_read_tokens"] == 600
    assert summary["cache_read_share"] == 0.6


def test_refusals_are_counted_by_category():
    collector = MetricsCollector()
    for category in ("reasoning_extraction", "reasoning_extraction", None):
        collector.record(
            _record(
                ParsePath.FAILED,
                finish_reason="refusal",
                stop_reason="refusal",
                refusal_category=category,
            )
        )
    collector.record(_record(ParsePath.FAILED, finish_reason="length", stop_reason="max_tokens"))
    assert collector.summary()["refusals"] == {"reasoning_extraction": 2, "uncategorized": 1}


def test_served_models_and_fallback_served_calls():
    collector = MetricsCollector()
    requested = "claude-opus-5-5"
    collector.record(
        _record(ParsePath.STRICT_SCHEMA, served_model=requested, requested_model=requested)
    )
    # A dated snapshot of the requested model is not a fallback.
    collector.record(
        _record(
            ParsePath.STRICT_SCHEMA,
            served_model="claude-opus-5-5-20261101",
            requested_model=requested,
        )
    )
    # A refusal fallback served the turn on another model.
    collector.record(
        _record(ParsePath.STRICT_SCHEMA, served_model="claude-opus-5", requested_model=requested)
    )
    # Unknown requested model: cannot be called a fallback.
    collector.record(_record(ParsePath.STRICT_SCHEMA, served_model="claude-opus-5"))

    summary = collector.summary()
    assert summary["served_model_counts"] == {
        "claude-opus-5": 2,
        "claude-opus-5-5": 1,
        "claude-opus-5-5-20261101": 1,
    }
    assert summary["fallback_served_calls"] == 1


# --- pricing sources --------------------------------------------------------------


def test_provider_priced_calls_are_summed_from_the_table():
    collector = MetricsCollector()
    ok = ParsePath.STRICT_SCHEMA
    collector.record(_record(ok, prompt_tokens=100, completion_tokens=10, estimated_cost=0.25))
    # A rejected request: no usage, priced at zero by the provider.
    collector.record(_record(ParsePath.FAILED, estimated_cost=0.0))
    collector.record(_record(ok, prompt_tokens=50, completion_tokens=5, estimated_cost=0.5))
    # Price arguments are ignored when the provider priced every call itself.
    summary = collector.summary(price_per_1m_input=100.0, price_per_1m_output=100.0)
    assert summary["estimated_cost"] == 0.75
    assert summary["pricing_supplied"] is True
    assert summary["pricing_source"] == "table"
    assert summary["tokens"]["total"] == 165


def test_a_partially_priced_run_falls_back_to_the_price_arguments():
    """Summing only the priced calls would understate the spend."""
    collector = MetricsCollector()
    million = 1_000_000
    ok = ParsePath.STRICT_SCHEMA
    collector.record(_record(ok, prompt_tokens=million, completion_tokens=0, estimated_cost=9.0))
    collector.record(_record(ok, prompt_tokens=million, completion_tokens=0))
    summary = collector.summary(price_per_1m_input=1.0, price_per_1m_output=1.0)
    assert summary["estimated_cost"] == 2.0
    assert summary["pricing_source"] == "arguments"


def test_pricing_source_is_none_when_nothing_is_priced():
    collector = MetricsCollector()
    collector.record(_record(ParsePath.STRICT_SCHEMA, prompt_tokens=10, completion_tokens=10))
    summary = collector.summary()
    assert summary["pricing_supplied"] is False
    assert summary["pricing_source"] is None

    empty = MetricsCollector()
    empty.record(_record(ParsePath.STRICT_SCHEMA))
    assert empty.summary()["pricing_source"] is None
