#!/usr/bin/env python3
"""Preflight: a handful of API calls that answer "can we actually talk to this model".

Run this before `evaluate.py`. A full 17-player game is a few hundred
calls; this is a few, so a wrong key, a wrong base URL, or a model that
ignores structured output costs you a handful of requests to discover instead
of a whole game's spend.

    WEREWOLF_LLM_PROVIDER=anthropic python scripts/check_llm.py
    python scripts/check_llm.py --provider luna

`anthropic` (the real provider) checks, in order of what blocks you first:
  1. Which settings arrived -- provider, model, the base URL's host, and
     whether the key is set (its length only) -- catches secrets not reaching
     the process at all.
  2. Offline: that every schema a game sends is within Anthropic's
     structured-output limits (`app/ai/schema_limits.py`). A schema over them
     is a 400 no retry fixes, and in a game it degrades silently.
  3. One real call per schema through the same provider the game builds, at
     most five requests, plus one when the key turns out not to have the
     server-side fallback beta (reported, not a failure): success, parse
     path, latency, stop reason, the model that answered, tokens and cost.
     Any failure, refusal, degraded parse path or schema rejection fails the
     preflight.
  4. `count_tokens` (free, not a generation) on one real v3 discussion prompt,
     rendered offline by playing a ScenarioProvider game in-process, and a
     per-game projection of the input tokens from it.

`luna` (dormant) checks the endpoint, authentication and whether strict JSON
schema or only JSON-object mode works.

The key is never printed, not even in part -- a fragment of one once reached
a public Actions log -- and neither is any prompt: only lengths and counts.
`LUNA_BASE_URL` is an Actions secret too, and Actions masks only a secret's
whole value, so not even its host is printed: only whether it is set.
Exits 0 when everything passed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel  # noqa: E402

from app.ai.coordinator import AICoordinator  # noqa: E402
from app.ai.dialect import EndpointDialect  # noqa: E402
from app.ai.engine_mode import build_reasoning_runtime, model_decides  # noqa: E402
from app.ai.metrics import CallRecord, MetricsCollector, ParsePath  # noqa: E402
from app.ai.provider.base import LLMProvider, Message, SchemaT  # noqa: E402
from app.ai.provider.budget import (  # noqa: E402
    BudgetedProvider,
    BudgetExceeded,
    EvaluationBudget,
)
from app.ai.provider.factory import LLMProviderConfigError, build_llm_provider  # noqa: E402
from app.ai.provider.pricing import price_for  # noqa: E402
from app.ai.provider.scenario import ScenarioProvider  # noqa: E402
from app.ai.schema_limits import (  # noqa: E402
    MAX_OPTIONAL_PARAMETERS,
    MAX_UNION_PARAMETERS,
    count_schema_parameters,
)
from app.ai.schemas import (  # noqa: E402
    BriefDiscussionOutput,
    ChatDiscussionOutput,
    NightActionOutput,
    VoteOutput,
    WolfChatOutput,
)
from app.config import DEFAULT_ANTHROPIC_BASE_URL, Settings  # noqa: E402
from app.engine.game import GameController, PlayerSpec  # noqa: E402
from app.engine.phases import Phase  # noqa: E402

PROMPT = (
    "あなたは人狼ゲームのプレイヤーです。候補 p1, p2, p3 の中から1人に投票してください。"
    '以下のJSON形式で回答してください: {"vote_target": "p1などのID", "reason": "簡潔な理由"}'
)

OK = "  [OK]  "
NG = "  [NG]  "
INFO = "  [--]  "

PROVIDERS = ("anthropic", "luna")

# Every schema a v3 game sends. `ChatDiscussionOutput` is the discussion turn
# itself; `BriefDiscussionOutput` is what a failed turn degrades to.
GAME_SCHEMAS: tuple[type[BaseModel], ...] = (
    ChatDiscussionOutput,
    BriefDiscussionOutput,
    VoteOutput,
    NightActionOutput,
    WolfChatOutput,
)

# HTTP requests the per-schema calls may make in all: one per schema, no
# retries. With the one `count_tokens` request that makes six. A call that
# needs a retry (a 429, a 529) runs out of budget and fails the preflight,
# which is the right answer: the endpoint is not healthy enough to start a
# game on.
MAX_PROBE_REQUESTS = len(GAME_SCHEMAS)

# Added to that cap only when the key rejects the server-side fallback beta:
# the provider then drops the beta and resends the same call once, exactly as
# in a game (see `ProbeBudget`). So the preflight never sends more than
# MAX_PROBE_REQUESTS + 1 generations and one `count_tokens`.
FALLBACK_NEGOTIATION_REQUESTS = 1

# Logical LLM calls in one v3 game, from the only complete live v3 game on
# record (run #11, gpt-6-luna, 2026-10-03: 197 calls; see
# `backend/artifacts/live-v3-luna-run11/report.md`). A projection, not a
# measurement: Claude's own count comes from the first game's report.
REFERENCE_CALLS_PER_GAME = 197

_SYSTEM = (
    "あなたは人狼ゲームの参加者です。これは接続確認のための短い手番です。"
    "指定された形式で、短く答えてください。"
)

# Tiny prompts, one per schema. Plain game instructions only: nothing asks
# the model to write out how it reached its answer.
PROBE_PROMPTS: dict[str, str] = {
    "ChatDiscussionOutput": (
        "1日目の議論です。生存者は p1(あなた)・p2・p3 です。"
        "公開の場での一言を public_message に書いてください。"
        "気になる相手がいれば plan_note.suspects に入れてください。"
    ),
    "BriefDiscussionOutput": (
        "1日目の議論です。公開の場での一言を public_message に書いてください。"
    ),
    "VoteOutput": (
        "投票の時間です。p2 か p3 のどちらかを vote_target に入れ、理由を一言添えてください。"
    ),
    "NightActionOutput": (
        "夜です。あなたは占い師です。占う相手を p2 か p3 から選び、target に入れてください。"
    ),
    "WolfChatOutput": "夜です。あなたは人狼です。仲間への一言を message に書いてください。",
}


def mask_secret(value: str) -> str:
    """Whether a secret is set, and its length. Never any part of it."""
    return f"設定済み(長さ{len(value)})" if value else "<空>"


def host_of(url: str) -> str:
    """The host alone: a base URL's path or query can carry more than it should."""
    return urlparse(url).hostname or "(URLとして読めません)"


def secret_url_state(url: str, placeholder_marker: str = "example.com") -> str:
    """Whether a secret URL is set. Never any part of it.

    Not even the host: Actions masks only a secret's whole value, so a host
    cut out of a secret URL is printed in clear in a public log.
    """
    if not url:
        return "<空>"
    if placeholder_marker in url:
        return "プレースホルダーのまま"
    return "設定済み"


def redact(text: str, secrets: Sequence[str]) -> str:
    """`text` with every secret, and every secret URL's host, replaced by ***.

    For messages this script did not write -- a client's exception text can
    repeat the endpoint it failed to reach, or the key it sent.
    """
    fragments = set()
    for secret in secrets:
        if not secret:
            continue
        fragments.add(secret)
        host = urlparse(secret).hostname
        if host:
            fragments.add(host)
    for fragment in sorted(fragments, key=len, reverse=True):
        text = text.replace(fragment, "***")
    return text


# --- offline: schema limits -----------------------------------------------


@dataclass(frozen=True)
class SchemaCheck:
    name: str
    optional: int
    union: int

    @property
    def ok(self) -> bool:
        return self.optional <= MAX_OPTIONAL_PARAMETERS and self.union <= MAX_UNION_PARAMETERS


def check_schema_limits(schemas: Sequence[type[BaseModel]] = GAME_SCHEMAS) -> list[SchemaCheck]:
    """Counts each schema exactly as the provider sends it (after the SDK's
    `transform_schema`), against the documented per-request limits."""
    from anthropic import transform_schema

    checks = []
    for schema in schemas:
        optional, union = count_schema_parameters(transform_schema(schema))
        checks.append(SchemaCheck(schema.__name__, optional, union))
    return checks


# --- offline: one real v3 discussion prompt -------------------------------


class _PromptCapture:
    """Passes every call to the offline double and keeps the discussion prompts."""

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.prompts: list[tuple[str, list[Message]]] = []

    async def generate_structured(
        self,
        *,
        system: str,
        messages: list[Message],
        response_schema: type[SchemaT],
        max_tokens: int = 800,
        temperature: float = 0.9,
    ) -> SchemaT | None:
        if response_schema is ChatDiscussionOutput:
            self.prompts.append((system, list(messages)))
        return await self._inner.generate_structured(
            system=system,
            messages=messages,
            response_schema=response_schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )


async def capture_discussion_prompt(seed: int = 1) -> tuple[str, list[Message]] | None:
    """A v3 discussion prompt exactly as a game renders it, made offline.

    Plays a ScenarioProvider game (free, no network) through the first day's
    discussion and returns the median-length prompt of that day: early turns
    are shorter than a real game's average, so the first one would
    understate it. None if the game never reached a discussion.
    """
    specs = [PlayerSpec(player_id=f"p{i}", name=f"P{i}", is_human=False) for i in range(17)]
    controller = GameController(session_id=f"check-llm-{seed}", player_specs=specs, seed=seed)
    ai_ids = [spec.player_id for spec in specs]
    capture = _PromptCapture(ScenarioProvider(seed=seed))
    coordinator = AICoordinator(
        controller.state,
        ai_ids,
        capture,
        seed=seed,
        reasoning=build_reasoning_runtime("v3", controller.state, ai_ids, seed=seed),
        model_decides=model_decides("v3"),
        pacing_scale=0.0,
    )
    session = SimpleNamespace(
        controller=controller,
        human_id="p0",
        coordinator=coordinator,
        discussion_lock=asyncio.Lock(),
    )
    controller.start_game()
    for _ in range(10):
        phase = controller.state.phase
        if phase is Phase.NIGHT:
            await coordinator.run_night_phase(session)
        elif phase is Phase.DAWN:
            controller.start_discussion()
        elif phase is Phase.DISCUSSION:
            await coordinator.run_discussion_round(session)
            break
        else:
            break
    if not capture.prompts:
        return None
    ordered = sorted(
        capture.prompts, key=lambda p: len(p[0]) + sum(len(m.content) for m in p[1])
    )
    return ordered[len(ordered) // 2]


def project_game_input_tokens(
    tokens_per_call: int, calls: int = REFERENCE_CALLS_PER_GAME
) -> int:
    return tokens_per_call * calls


# --- live: one call per schema ---------------------------------------------


@dataclass
class ProbeResult:
    schema: str
    record: CallRecord | None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def probe_problems(
    *,
    answered: bool,
    record: CallRecord | None,
    grammar_rejected: bool,
) -> list[str]:
    """Everything that makes one preflight call a failure, in plain words.

    Stricter than a game: a game survives a degraded parse or a refusal with
    a fallback, which is exactly why they must be caught here instead.
    """
    problems: list[str] = []
    if grammar_rejected:
        problems.append("スキーマが構造化出力の文法に拒否されました(400)")
    if record is None:
        problems.append("呼び出しの記録がありません")
        return problems
    if record.refused:
        problems.append(f"拒否されました(category={record.refusal_category or '不明'})")
    if record.truncated:
        problems.append("出力上限で途中切れになりました(WEREWOLF_ANTHROPIC_MAX_TOKENS)")
    if record.served_by_fallback:
        problems.append(
            f"要求したモデルではなく {record.served_model} が答えました(拒否からのフォールバック)"
        )
    if record.path is not ParsePath.STRICT_SCHEMA and record.path is not ParsePath.FAILED:
        problems.append(f"strict schema ではなく降格した経路({record.path.value})で読みました")
    if not answered:
        detail = f": {record.error}" if record.error else ""
        problems.append(f"使える応答が得られませんでした{detail}")
    return problems


class ProbeBudget(EvaluationBudget):
    """The preflight's request cap, with room for the fallback-beta negotiation.

    A key without the server-side fallback beta answers the first request
    that asks for it with a 400. The provider then drops the beta for good
    and resends the same call, as it does in a game, so that resend is a
    negotiation and not a retry. Without room for it the last schema ran out
    of requests and the preflight failed, blaming a 429/529, on a key a game
    works with -- and the evaluate workflows, which need the preflight, could
    not start at all.

    The extra request is granted only when the provider is seen to have
    dropped the beta, at the reservation for the resend itself, so any other
    retry still exhausts the plain cap.
    """

    def __init__(self, max_requests: int, fallbacks_enabled: Callable[[], bool]) -> None:
        super().__init__(max_requests=max_requests)
        self._fallbacks_enabled = fallbacks_enabled
        self._fallbacks_offered = fallbacks_enabled()
        self.fallback_rejected = False

    def reserve(self, worst_case_cost: float) -> None:
        if (
            self._fallbacks_offered
            and not self.fallback_rejected
            and not self._fallbacks_enabled()
        ):
            self.fallback_rejected = True
            self.max_requests += FALLBACK_NEGOTIATION_REQUESTS
        super().reserve(worst_case_cost)


async def probe_schemas(
    provider: LLMProvider,
    collector: MetricsCollector,
    rejected: Callable[[], frozenset[str]],
    schemas: Sequence[type[BaseModel]] = GAME_SCHEMAS,
    *,
    request_limit: Callable[[], int] = lambda: MAX_PROBE_REQUESTS,
) -> list[ProbeResult]:
    """One call per schema, in order, each judged by its own `CallRecord`.

    `request_limit` reads the cap when it is hit, since a fallback-beta
    negotiation can raise it during the run.
    """
    results: list[ProbeResult] = []
    for schema in schemas:
        name = schema.__name__
        before = len(collector.records)
        try:
            answer = await provider.generate_structured(
                system=_SYSTEM,
                messages=[Message(role="user", content=PROBE_PROMPTS[name])],
                response_schema=schema,
                max_tokens=400,
            )
        except BudgetExceeded:
            results.append(
                ProbeResult(
                    name,
                    collector.records[-1] if len(collector.records) > before else None,
                    [
                        f"要求数の上限({request_limit()})に達しました。"
                        "再試行が必要だった呼び出しがあります(429・529・接続エラーなど)"
                    ],
                )
            )
            break
        record = collector.records[-1] if len(collector.records) > before else None
        problems = probe_problems(
            answered=answer is not None,
            record=record,
            grammar_rejected=name in rejected(),
        )
        results.append(ProbeResult(name, record, problems))
    return results


def describe_record(record: CallRecord | None) -> str:
    if record is None:
        return "(記録なし)"
    parts = [f"{record.path.value}", f"{record.latency_seconds:.2f}秒"]
    parts.append(f"stop={record.stop_reason or '-'}")
    parts.append(f"model={record.served_model or '-'}")
    if record.prompt_tokens is not None:
        thinking = f"(思考{record.reasoning_tokens})" if record.reasoning_tokens else ""
        parts.append(f"入力{record.prompt_tokens}/出力{record.completion_tokens}{thinking}")
    if record.estimated_cost is not None:
        parts.append(f"${record.estimated_cost:.4f}")
    if record.effort:
        parts.append(f"effort={record.effort}")
    return " ".join(parts)


async def count_prompt_tokens(
    client: Any, model: str, prompt: tuple[str, list[Message]], effort: str
) -> int:
    """Input tokens for one rendered discussion prompt, with the schema the
    provider would attach. `count_tokens` is free and generates nothing."""
    from anthropic import transform_schema

    system, messages = prompt
    result = await client.messages.count_tokens(
        model=model,
        system=system,
        messages=[{"role": m.role, "content": m.content} for m in messages],
        output_config={
            "effort": effort,
            "format": {"type": "json_schema", "schema": transform_schema(ChatDiscussionOutput)},
        },
    )
    return int(result.input_tokens)


ProviderBuilder = Callable[[Settings, MetricsCollector], LLMProvider]


def _factory_builder(settings: Settings, metrics: MetricsCollector) -> LLMProvider:
    return build_llm_provider(settings, metrics=metrics)


async def run_anthropic(
    settings: Settings,
    *,
    build: ProviderBuilder = _factory_builder,
    prompt: tuple[str, list[Message]] | None = None,
    out: Callable[[str], None] = print,
) -> int:
    model = settings.werewolf_anthropic_model
    out("=== 1. 設定の確認 ===")
    out(f"{INFO}WEREWOLF_LLM_PROVIDER         = {settings.werewolf_llm_provider}")
    out(f"{INFO}WEREWOLF_ANTHROPIC_MODEL      = {model}")
    out(f"{INFO}WEREWOLF_ANTHROPIC_BASE_URL   = {host_of(settings.werewolf_anthropic_base_url)}")
    out(f"{INFO}WEREWOLF_ANTHROPIC_API_KEY    = {mask_secret(settings.werewolf_anthropic_api_key)}")
    out(f"{INFO}思考量(effort)                = {settings.anthropic_effort_by_kind}")
    out(f"{INFO}フォールバック                = {settings.werewolf_anthropic_fallbacks}")
    if not settings.werewolf_anthropic_api_key:
        out(
            f"{NG}WEREWOLF_ANTHROPIC_API_KEY が空です。Actions では secret の名前を、"
            "Codespaces では secret の登録と Codespace の再起動を確認してください"
            "(SDK の ANTHROPIC_API_KEY は読みません)。"
        )
        return 1
    if host_of(settings.werewolf_anthropic_base_url) != host_of(DEFAULT_ANTHROPIC_BASE_URL):
        out(f"{INFO}既定({host_of(DEFAULT_ANTHROPIC_BASE_URL)})と違う接続先です。意図どおりか確認してください。")

    out("\n=== 2. スキーマの確認(通信なし) ===")
    checks = check_schema_limits()
    for check in checks:
        mark = OK if check.ok else NG
        out(
            f"{mark}{check.name}: 任意 {check.optional}/{MAX_OPTIONAL_PARAMETERS}・"
            f"union {check.union}/{MAX_UNION_PARAMETERS}"
        )
    if not all(check.ok for check in checks):
        out(f"{NG}上限を超えるスキーマは API が400で拒否します。実呼び出しの前に止めます。")
        return 1

    out(
        f"\n=== 3. スキーマごとに1回ずつ実呼び出し(最大{MAX_PROBE_REQUESTS}要求、"
        f"フォールバックbetaが使えないキーなら+{FALLBACK_NEGOTIATION_REQUESTS}) ==="
    )
    collector = MetricsCollector()
    try:
        inner = build(settings, collector)
    except (LLMProviderConfigError, ValueError) as exc:
        out(f"{NG}プロバイダを作れませんでした: {exc}")
        return 1
    budget = ProbeBudget(
        MAX_PROBE_REQUESTS, lambda: bool(getattr(inner, "fallbacks_enabled", False))
    )
    provider = BudgetedProvider(inner, budget)

    def rejected() -> frozenset[str]:
        return frozenset(getattr(inner, "grammar_rejected_schemas", frozenset()))

    results = await probe_schemas(
        provider, collector, rejected, request_limit=lambda: budget.max_requests
    )
    if budget.fallback_rejected:
        out(
            f"{INFO}サーバー側フォールバック(beta)はこのキーでは使えません。"
            "フォールバックなしで続けます(fallback beta unavailable for this key"
            " — continuing without it)。試合でも同じく1要求で切り替わります。"
            "その1要求も省くなら WEREWOLF_ANTHROPIC_FALLBACKS=off を設定してください。"
        )
    for result in results:
        mark = OK if result.ok else NG
        out(f"{mark}{result.schema}: {describe_record(result.record)}")
        for problem in result.problems:
            out(f"{NG}    {problem}")
    probes_ok = len(results) == len(GAME_SCHEMAS) and all(result.ok for result in results)
    spent = sum(r.record.estimated_cost or 0.0 for r in results if r.record is not None)
    negotiation = "(うちフォールバックbetaの切り替え1回)" if budget.fallback_rejected else ""
    out(f"{INFO}HTTP要求 {budget.used_requests}回{negotiation}・推定費用 ${spent:.4f}")

    out("\n=== 4. 議論プロンプトのトークン数(count_tokens) ===")
    tokens_ok = True
    if prompt is None:
        prompt = await capture_discussion_prompt()
    if prompt is None:
        out(f"{NG}通信なしの対戦から議論プロンプトを作れませんでした。")
        tokens_ok = False
    else:
        # The provider's own client: the same explicit key and base URL, with
        # the SDK's ambient ANTHROPIC_* inputs already neutralised.
        client = getattr(inner, "_client", None)
        chars = len(prompt[0]) + sum(len(m.content) for m in prompt[1])
        try:
            if client is None:
                raise RuntimeError("provider exposes no client")
            tokens = await count_prompt_tokens(
                client, model, prompt, settings.anthropic_effort_by_kind["discussion"]
            )
        except Exception as exc:
            out(f"{NG}count_tokens に失敗しました: {type(exc).__name__}: {str(exc)[:160]}")
            tokens_ok = False
        else:
            projected = project_game_input_tokens(tokens)
            price, _known = price_for(model)
            out(
                f"{OK}v3 の議論プロンプト1件(1日目・中央値の長さ、{chars:,}文字): "
                f"入力 {tokens:,} トークン"
            )
            out(
                f"{INFO}1試合の入力の見込み: {tokens:,} × {REFERENCE_CALLS_PER_GAME}回 ≈ "
                f"{projected:,} トークン(キャッシュ無しなら入力だけで約 "
                f"${projected * price.input / 1_000_000:.2f})"
            )
            out(
                f"{INFO}回数は run #11(gpt-6-luna の v3 1試合)の値。出力と思考は含まず、"
                "投票などは議論より短いので粗い見込みです。実額は最初の試合のレポートで。"
            )

    out("\n=== 判定 ===")
    if probes_ok and tokens_ok:
        out(f"{OK}すべて通りました。評価を始められます。")
        out("\n次のステップ:")
        out(
            "  python scripts/evaluate.py --games 1 --provider anthropic --max-cost 15 "
            "--out eval-out"
        )
        return 0
    out(f"{NG}通らなかった項目があります。上の [NG] を直してから評価してください。")
    return 1


# --- luna (dormant) --------------------------------------------------------


async def _one_call(
    client: Any,
    model: str,
    response_format: dict[str, Any] | None,
    dialect: EndpointDialect,
    secrets: Sequence[str] = (),
) -> dict[str, Any]:
    """Same parameter negotiation the real provider uses, so what this
    reports is what the game will actually send. A failure's text has
    `secrets` redacted, since it is printed."""
    started = perf_counter()
    while True:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
        }
        dialect.apply(kwargs, max_tokens=200, temperature=0.0)
        if response_format is not None:
            kwargs["response_format"] = response_format
        try:
            response = await client.chat.completions.create(**kwargs)
            break
        except Exception as exc:
            if dialect.adapt(exc):
                print(f"{INFO}このエンドポイント向けに調整: {dialect.describe()}")
                continue
            return {"ok": False, "error": redact(f"{type(exc).__name__}: {exc}", secrets)}
    elapsed = perf_counter() - started

    content = response.choices[0].message.content if response.choices else None
    usage = getattr(response, "usage", None)
    return {
        "ok": True,
        "latency": elapsed,
        "content": content,
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
    }


def _validates(content: str | None) -> bool:
    if not content:
        return False
    try:
        VoteOutput.model_validate_json(content)
        return True
    except Exception:
        return False


async def run_luna(settings: Settings, *, out: Callable[[str], None] = print) -> int:
    model = settings.luna_model
    out("=== 1. 設定の確認 ===")
    out(f"{INFO}WEREWOLF_LLM_PROVIDER = {settings.werewolf_llm_provider}")
    out(f"{INFO}LUNA_BASE_URL         = {secret_url_state(settings.luna_base_url)}")
    out(f"{INFO}LUNA_MODEL            = {model}")
    out(f"{INFO}LUNA_API_KEY          = {mask_secret(settings.luna_api_key)}")

    problems = []
    if not settings.luna_api_key:
        problems.append(
            "LUNA_API_KEY が空です。Codespaces secrets に登録したあと、"
            "Codespace を再起動しないと反映されません。"
        )
    if "example.com" in settings.luna_base_url:
        problems.append(
            "LUNA_BASE_URL がプレースホルダーのままです。"
            "実際のエンドポイントURLを設定してください。"
        )
    if problems:
        out("")
        for problem in problems:
            out(f"{NG}{problem}")
        return 1

    from openai import AsyncOpenAI  # imported late so the checks above run without it

    client = AsyncOpenAI(
        api_key=settings.luna_api_key,
        base_url=settings.luna_base_url,
        timeout=settings.luna_timeout_seconds,
    )

    dialect = EndpointDialect()
    secrets = (settings.luna_api_key, settings.luna_base_url)

    out("\n=== 2. 疎通と認証 ===")
    plain = await _one_call(client, model, None, dialect, secrets)
    if not plain["ok"]:
        out(f"{NG}接続に失敗しました: {plain['error']}")
        out("\n確認してください:")
        out("  - LUNA_BASE_URL は末尾が /v1 になっていますか")
        out("  - キーは有効ですか(失効・コピーミス)")
        out("  - モデル名は正しいですか")
        return 1
    out(f"{OK}応答あり ({plain['latency']:.2f} 秒)")
    out(f"{INFO}送信パラメータ: {dialect.describe()}")
    out(f"{INFO}生の応答: {str(plain['content'])[:160]}")

    if plain["prompt_tokens"] is not None:
        out(
            f"{OK}usage を返しています "
            f"(入力{plain['prompt_tokens']} / 出力{plain['completion_tokens']}トークン)"
        )
    else:
        out(f"{NG}usage を返していません → コスト集計は「不明」表示になります")

    out("\n=== 3. 構造化出力の対応状況 ===")
    strict = await _one_call(
        client,
        model,
        {
            "type": "json_schema",
            "json_schema": {
                "name": "VoteOutput",
                "schema": VoteOutput.model_json_schema(),
                "strict": True,
            },
        },
        dialect,
        secrets,
    )
    strict_ok = strict["ok"] and _validates(strict.get("content"))
    if strict_ok:
        out(f"{OK}strict JSON-schema に対応 ({strict['latency']:.2f} 秒)")
    elif strict["ok"]:
        out(f"{NG}strict は通ったがスキーマに合いません: {str(strict['content'])[:120]}")
    else:
        out(f"{NG}strict JSON-schema 非対応: {strict['error'][:160]}")

    loose = await _one_call(client, model, {"type": "json_object"}, dialect, secrets)
    loose_ok = loose["ok"] and _validates(loose.get("content"))
    if loose_ok:
        out(f"{OK}json_object モードに対応 ({loose['latency']:.2f} 秒)")
    elif loose["ok"]:
        out(f"{INFO}json_object は通りましたが直接の検証は失敗 → 寛容パーサーが処理します")
    else:
        out(f"{NG}json_object 非対応: {loose['error'][:160]}")

    out("\n=== 判定 ===")
    if strict_ok:
        out(f"{OK}そのまま評価を実行できます。JSON成功率は高くなる見込みです。")
    elif loose_ok or (plain["ok"] and _validates(plain.get("content"))):
        out(f"{INFO}strict は使えませんが、フォールバック経路で動作します。")
        out(f"{INFO}レポートの「strict schema で成功」が低く出るのは想定どおりです。")
    else:
        out(f"{NG}JSON形式の応答が安定しません。評価前にプロンプト調整が必要かもしれません。")

    out("\n次のステップ:")
    out("  python scripts/evaluate.py --games 1 --provider luna --out eval-out")
    tokens = plain["prompt_tokens"]
    if tokens:
        out(
            f"  (1ゲームは約200回の呼び出し。今回の1回が入力{tokens}トークンだったので、"
            "レポートの合計値で実コストを確認してください)"
        )
    return 0


def resolve_settings(args: argparse.Namespace) -> Settings:
    """The provider to check: `--provider`, else the configured one when it is
    real, else anthropic -- this preflight exists for the real providers, and
    the configured default (`mock`) has nothing to connect to."""
    configured = Settings().werewolf_llm_provider
    provider = args.provider or (configured if configured in PROVIDERS else "anthropic")
    overrides: dict[str, Any] = {"werewolf_llm_provider": provider}
    if args.model:
        overrides["werewolf_anthropic_model" if provider == "anthropic" else "luna_model"] = (
            args.model
        )
    return Settings(**overrides)


async def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default=None,
        help="既定: WEREWOLF_LLM_PROVIDER が anthropic / luna ならそれ、そうでなければ anthropic",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="既定: WEREWOLF_ANTHROPIC_MODEL(anthropic)/ LUNA_MODEL(luna)",
    )
    args = parser.parse_args(argv)
    settings = resolve_settings(args)
    if settings.werewolf_llm_provider == "anthropic":
        return await run_anthropic(settings)
    return await run_luna(settings)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
