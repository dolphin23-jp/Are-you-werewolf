"""The preflight, offline: masking, the schema check, the prompt capture, and
the anthropic path end to end against a scripted Messages API.

The anthropic tests drive the real provider and the real SDK client over an
`httpx2.MockTransport`, so request counting, parse paths and refusals are the
provider's own -- only the server is fake.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from app.ai.metrics import CallRecord, MetricsCollector, ParsePath
from app.ai.provider.anthropic_claude import AnthropicClaudeProvider
from app.ai.provider.base import Message
from app.ai.schemas import DiscussionOutput
from app.config import Settings
from scripts import check_llm

KEY = "sk-ant-api03-SECRETSECRET-0123456789-tail"
PROMPT = ("SYSTEM-PROMPT-TEXT-秘密", [Message(role="user", content="USER-PROMPT-TEXT-秘密")])

_ANSWERS = {
    "plan_note": {"public_message": "よろしくお願いします。", "plan_note": {"suspects": ["p2"]}},
    "vote_target": {"vote_target": "p2", "reason": "様子見"},
    "target": {"target": "p3", "reason": "様子見"},
    "message": {"message": "今夜はp2で。"},
    "public_message": {"public_message": "よろしくお願いします。"},
}


def _answer_for(body: dict[str, Any]) -> dict[str, Any]:
    schema = (body.get("output_config") or {}).get("format", {}).get("schema", {})
    properties = schema.get("properties") or {}
    for key in ("plan_note", "vote_target", "target", "message", "public_message"):
        if key in properties:
            return _ANSWERS[key]
    # A degraded request carries no grammar: answer as the discussion turn.
    return _ANSWERS["plan_note"]


def _message(payload: dict[str, Any], *, stop_reason: str = "end_turn", **extra: Any) -> Any:
    body: dict[str, Any] = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
        ],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 40},
        **extra,
    }
    return httpx2.Response(200, headers={"request-id": "req_ok"}, json=body)


class _Server:
    """A scripted Messages API. `override(body)` may answer a request instead."""

    def __init__(self, override: Any = None) -> None:
        self.override = override
        self.paths: list[str] = []

    async def handler(self, request: httpx2.Request) -> Any:
        self.paths.append(request.url.path)
        if request.url.path.endswith("/count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 5_000})
        body = json.loads(request.content)
        if self.override is not None:
            answered = self.override(body)
            if answered is not None:
                return answered
        return _message(_answer_for(body))


def _builder(server: _Server) -> Any:
    def build(settings: Settings, metrics: MetricsCollector) -> AnthropicClaudeProvider:
        client = anthropic.AsyncAnthropic(
            api_key="test-key",
            base_url="https://example.invalid",
            max_retries=0,
            http_client=anthropic.DefaultAsyncHttpxClient(
                transport=httpx2.MockTransport(server.handler)
            ),
        )

        async def no_sleep(seconds: float) -> None:
            return None

        return AnthropicClaudeProvider(
            api_key="test-key",
            base_url="https://example.invalid",
            model=settings.werewolf_anthropic_model,
            effort_by_kind=settings.anthropic_effort_by_kind,
            metrics=metrics,
            client=client,
            sleep=no_sleep,
        )

    return build


def _settings(**overrides: Any) -> Settings:
    fields: dict[str, Any] = {
        "werewolf_llm_provider": "anthropic",
        "werewolf_anthropic_api_key": KEY,
        **overrides,
    }
    return Settings(**fields)


def _run(server: _Server, settings: Settings | None = None) -> tuple[int, list[str]]:
    lines: list[str] = []
    code = asyncio.run(
        check_llm.run_anthropic(
            settings or _settings(), build=_builder(server), prompt=PROMPT, out=lines.append
        )
    )
    return code, lines


def _assert_nothing_secret(lines: list[str]) -> None:
    text = "\n".join(lines)
    assert KEY not in text
    # No fragment either: the old mask printed the first four and last two.
    assert KEY[:4] not in text and KEY[-6:] not in text
    assert "秘密" not in text and "PROMPT-TEXT" not in text
    for prompt in check_llm.PROBE_PROMPTS.values():
        assert prompt not in text


# --- masking -----------------------------------------------------------------


def test_a_secret_is_shown_as_presence_and_length_only() -> None:
    assert check_llm.mask_secret("") == "<空>"
    masked = check_llm.mask_secret(KEY)
    assert masked == f"設定済み(長さ{len(KEY)})"
    assert not any(KEY[i : i + 3] in masked for i in range(len(KEY) - 2))


def test_only_the_host_of_a_base_url_is_shown() -> None:
    assert check_llm.host_of("https://api.anthropic.com/v1?token=abc") == "api.anthropic.com"
    assert "abc" not in check_llm.host_of("https://proxy.example/path/abc")


def test_luna_never_prints_its_key_or_base_url_path() -> None:
    settings = Settings(
        werewolf_llm_provider="luna",
        luna_api_key="sk-luna-SECRETVALUE-xyz",
        luna_base_url="https://api.example.com/v1/tenant-secret",
    )
    lines: list[str] = []

    # The placeholder base URL stops it before any network access.
    assert asyncio.run(check_llm.run_luna(settings, out=lines.append)) == 1

    text = "\n".join(lines)
    assert "設定済み(長さ23)" in text
    assert "sk-l" not in text and "xyz" not in text and "tenant-secret" not in text


PRIVATE_LUNA_URL = "https://my-private-endpoint.internal.example/v1"
PRIVATE_LUNA_HOST = "my-private-endpoint.internal.example"


def test_luna_prints_only_whether_its_base_url_is_set() -> None:
    """LUNA_BASE_URL is an Actions secret, and Actions masks only the whole
    value: its host alone is a substring and went into the public log as is."""
    lines: list[str] = []
    settings = Settings(
        werewolf_llm_provider="luna", luna_api_key="", luna_base_url=PRIVATE_LUNA_URL
    )

    # No key stops it before any network access.
    assert asyncio.run(check_llm.run_luna(settings, out=lines.append)) == 1

    text = "\n".join(lines)
    assert "my-private-endpoint" not in text and "internal.example" not in text
    assert any("LUNA_BASE_URL" in line and "設定済み" in line for line in lines)

    lines.clear()
    asyncio.run(check_llm.run_luna(Settings(werewolf_llm_provider="luna"), out=lines.append))
    assert any("LUNA_BASE_URL" in line and "プレースホルダー" in line for line in lines)


class _FailingOpenAI:
    """Stands in for `openai.AsyncOpenAI`: every call fails, and the error
    repeats the endpoint and the key, as client errors can."""

    def __init__(self, *, api_key: str, base_url: str, timeout: float) -> None:
        self._detail = f"cannot reach {base_url} ({PRIVATE_LUNA_HOST}) with key {api_key}"
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        raise RuntimeError(self._detail)


def test_a_luna_error_never_carries_the_base_url_or_key_into_the_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", _FailingOpenAI)
    key = "sk-luna-SECRETVALUE-xyz"
    settings = Settings(
        werewolf_llm_provider="luna", luna_api_key=key, luna_base_url=PRIVATE_LUNA_URL
    )
    lines: list[str] = []

    assert asyncio.run(check_llm.run_luna(settings, out=lines.append)) == 1

    text = "\n".join(lines)
    assert "接続に失敗しました" in text and "RuntimeError" in text
    assert PRIVATE_LUNA_HOST not in text and "my-private-endpoint" not in text
    assert key not in text


# --- offline checks ------------------------------------------------------------


def test_every_schema_a_game_sends_is_within_the_limits() -> None:
    checks = check_llm.check_schema_limits()

    assert [check.name for check in checks] == [
        "ChatDiscussionOutput",
        "BriefDiscussionOutput",
        "VoteOutput",
        "NightActionOutput",
        "WolfChatOutput",
    ]
    assert all(check.ok for check in checks)
    assert not check_llm.check_schema_limits([DiscussionOutput])[0].ok


def test_a_real_v3_discussion_prompt_is_rendered_offline() -> None:
    prompt = asyncio.run(check_llm.capture_discussion_prompt(seed=1))

    assert prompt is not None
    system, messages = prompt
    assert system and messages and all(m.content for m in messages)
    # A board memo prompt, not a toy: the size of a real turn.
    assert len(system) + sum(len(m.content) for m in messages) > 1_000


def test_the_projection_scales_one_prompt_by_a_games_calls() -> None:
    assert check_llm.project_game_input_tokens(5_000) == 5_000 * 197
    assert check_llm.project_game_input_tokens(10, calls=3) == 30


def _record(**overrides: Any) -> CallRecord:
    fields: dict[str, Any] = {
        "schema": "VoteOutput",
        "path": ParsePath.STRICT_SCHEMA,
        "latency_seconds": 1.0,
        "attempt": 0,
        "stop_reason": "end_turn",
        "finish_reason": "end_turn",
        "served_model": "claude-opus-5-5",
        "requested_model": "claude-opus-5-5",
    }
    fields.update(overrides)
    return CallRecord(**fields)


@pytest.mark.parametrize(
    ("record", "grammar_rejected", "expected"),
    [
        (
            _record(finish_reason="refusal", stop_reason="refusal", refusal_category="cyber"),
            False,
            "拒否",
        ),
        (_record(finish_reason="length", stop_reason="max_tokens"), False, "途中切れ"),
        (_record(path=ParsePath.JSON_OBJECT), True, "400"),
        (_record(path=ParsePath.PERMISSIVE), False, "降格"),
        (_record(served_model="claude-opus-5"), False, "claude-opus-5"),
    ],
)
def test_anything_short_of_a_clean_strict_answer_is_a_failure(
    record: CallRecord, grammar_rejected: bool, expected: str
) -> None:
    problems = check_llm.probe_problems(
        answered=True, record=record, grammar_rejected=grammar_rejected
    )

    assert any(expected in problem for problem in problems)


def test_a_clean_strict_answer_passes() -> None:
    assert check_llm.probe_problems(answered=True, record=_record(), grammar_rejected=False) == []


# --- the anthropic path end to end ---------------------------------------------


def test_a_healthy_endpoint_passes_in_six_requests_without_printing_secrets() -> None:
    server = _Server()

    code, lines = _run(server)

    assert code == 0, "\n".join(lines)
    messages = [path for path in server.paths if not path.endswith("/count_tokens")]
    assert len(messages) == len(check_llm.GAME_SCHEMAS)
    assert len(server.paths) == 6  # five calls and one count_tokens
    text = "\n".join(lines)
    assert "設定済み(長さ" in text and "example.invalid" not in text.split("=== 2.")[1]
    assert "strict_schema" in text and "stop=end_turn" in text and "model=claude-opus-5-5" in text
    assert "5,000 トークン" in text and f"{5_000 * 197:,}" in text
    _assert_nothing_secret(lines)


def test_a_refusal_fails_the_preflight() -> None:
    def refuse_votes(body: dict[str, Any]) -> Any:
        properties = body["output_config"]["format"]["schema"]["properties"]
        if "vote_target" in properties:
            return _message({}, stop_reason="refusal", stop_details={"category": "cyber"})
        return None

    code, lines = _run(_Server(refuse_votes))

    assert code == 1
    assert any("拒否" in line and "cyber" in line for line in lines)
    _assert_nothing_secret(lines)


def test_a_schema_the_grammar_rejects_fails_the_preflight() -> None:
    def reject_discussion(body: dict[str, Any]) -> Any:
        schema = (body.get("output_config") or {}).get("format", {}).get("schema", {})
        if "plan_note" in (schema.get("properties") or {}):
            return httpx2.Response(
                400,
                json={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Schema is too complex for compilation",
                    },
                },
            )
        return None

    code, lines = _run(_Server(reject_discussion))

    assert code == 1
    assert any("400" in line for line in lines)
    assert any("json_object" in line or "降格" in line for line in lines)


def test_a_transient_error_that_needs_a_retry_stays_within_the_request_cap() -> None:
    def overloaded(body: dict[str, Any]) -> Any:
        return httpx2.Response(
            529, json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
        )

    server = _Server(overloaded)
    code, lines = _run(server)

    assert code == 1
    messages = [path for path in server.paths if not path.endswith("/count_tokens")]
    assert len(messages) <= check_llm.MAX_PROBE_REQUESTS
    assert any("要求数の上限" in line for line in lines)


def _reject_fallbacks(body: dict[str, Any]) -> Any:
    """A key without the server-side fallback beta: any request that asks for
    it is a 400 naming it, as `_is_fallback_rejection` expects."""
    if "fallbacks" in body:
        return httpx2.Response(
            400,
            json={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "fallbacks: 'default' is not available for this organization",
                },
            },
        )
    return None


def test_a_key_without_the_fallback_beta_passes_with_one_negotiation_request() -> None:
    """The provider drops the beta and resends; a game works the same way, so
    the preflight must pass, count that one request, and say what it was --
    not run out of requests and blame a 429."""
    server = _Server(_reject_fallbacks)

    code, lines = _run(server)

    assert code == 0, "\n".join(lines)
    messages = [path for path in server.paths if not path.endswith("/count_tokens")]
    assert len(messages) == len(check_llm.GAME_SCHEMAS) + 1
    assert check_llm.FALLBACK_NEGOTIATION_REQUESTS == 1
    text = "\n".join(lines)
    assert "フォールバック(beta)はこのキーでは使えません" in text
    assert "fallback beta unavailable for this key" in text
    assert "要求数の上限" not in text and "429" not in text
    assert f"HTTP要求 {len(messages)}回" in text
    _assert_nothing_secret(lines)


def test_after_the_fallback_negotiation_a_retry_still_fails_within_a_fixed_cap() -> None:
    overloaded_once: list[bool] = []

    def reject_fallbacks_then_overload_a_vote(body: dict[str, Any]) -> Any:
        rejected = _reject_fallbacks(body)
        if rejected is not None:
            return rejected
        properties = body["output_config"]["format"]["schema"]["properties"]
        if "vote_target" in properties and not overloaded_once:
            overloaded_once.append(True)
            return httpx2.Response(
                529,
                json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}},
            )
        return None

    server = _Server(reject_fallbacks_then_overload_a_vote)
    code, lines = _run(server)

    assert code == 1
    messages = [path for path in server.paths if not path.endswith("/count_tokens")]
    assert (
        len(messages)
        == check_llm.MAX_PROBE_REQUESTS + check_llm.FALLBACK_NEGOTIATION_REQUESTS
    )
    assert any("要求数の上限" in line for line in lines)


def test_without_a_fallback_rejection_no_extra_request_is_allowed() -> None:
    """The negotiation allowance is for the beta alone: a retry on a key that
    has the beta still exhausts the plain five."""
    overloaded_once: list[bool] = []

    def overload_the_first_call(body: dict[str, Any]) -> Any:
        if not overloaded_once:
            overloaded_once.append(True)
            return httpx2.Response(
                529,
                json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}},
            )
        return None

    server = _Server(overload_the_first_call)
    code, lines = _run(server)

    assert code == 1
    messages = [path for path in server.paths if not path.endswith("/count_tokens")]
    assert len(messages) == check_llm.MAX_PROBE_REQUESTS
    assert any("要求数の上限(5)" in line for line in lines)
    assert not any("フォールバック(beta)" in line for line in lines)


def test_no_key_means_no_request() -> None:
    server = _Server()

    code, lines = _run(server, _settings(werewolf_anthropic_api_key=""))

    assert code == 1
    assert server.paths == []
    assert any("WEREWOLF_ANTHROPIC_API_KEY が空" in line for line in lines)


def test_a_schema_over_the_limits_stops_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        check_llm,
        "check_schema_limits",
        lambda: [check_llm.SchemaCheck("DiscussionOutput", 36, 12)],
    )
    server = _Server()

    code, lines = _run(server)

    assert code == 1
    assert server.paths == []
    assert any("DiscussionOutput" in line and "[NG]" in line for line in lines)


# --- which provider is checked ---------------------------------------------------


def test_the_configured_real_provider_is_checked_and_mock_falls_back_to_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser_args = type("Args", (), {"provider": None, "model": None})

    assert check_llm.resolve_settings(parser_args()).werewolf_llm_provider == "anthropic"
    monkeypatch.setenv("WEREWOLF_LLM_PROVIDER", "luna")
    assert check_llm.resolve_settings(parser_args()).werewolf_llm_provider == "luna"


def test_model_override_goes_to_the_checked_provider() -> None:
    args = type("Args", (), {"provider": "anthropic", "model": "claude-sonnet-5-5"})()
    assert check_llm.resolve_settings(args).werewolf_anthropic_model == "claude-sonnet-5-5"
    args = type("Args", (), {"provider": "luna", "model": "gpt-7-luna"})()
    assert check_llm.resolve_settings(args).luna_model == "gpt-7-luna"
