#!/usr/bin/env python3
"""Preflight: one API call that answers "can we actually talk to this model".

Run this before `evaluate.py`. A full 17-player game is a few hundred
calls; this is one, so a wrong key, a wrong base URL, or a model that
ignores structured output costs you a single request to discover instead
of a whole game's spend.

    python scripts/check_llm.py

It reports, in order of what blocks you first:
  1. Which settings arrived (key masked) -- catches secrets not reaching
     the process at all.
  2. Whether the endpoint answers and authenticates.
  3. Whether strict JSON-schema structured output works, or only loose
     JSON mode, or neither -- this decides how much the permissive parser
     will be carrying at runtime.
  4. Latency and whether `usage` is reported (needed for cost tracking).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.dialect import EndpointDialect  # noqa: E402
from app.ai.schemas import VoteOutput  # noqa: E402
from app.config import Settings  # noqa: E402

PROMPT = (
    "あなたは人狼ゲームのプレイヤーです。候補 p1, p2, p3 の中から1人に投票してください。"
    '以下のJSON形式で回答してください: {"vote_target": "p1などのID", "reason": "簡潔な理由"}'
)

OK = "  [OK]  "
NG = "  [NG]  "
INFO = "  [--]  "


def _mask(value: str) -> str:
    if not value:
        return "<空>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-2:]} (長さ{len(value)})"


async def _one_call(
    client: Any,
    model: str,
    response_format: dict[str, Any] | None,
    dialect: EndpointDialect,
) -> dict:
    """Same parameter negotiation the real provider uses, so what this
    reports is what the game will actually send."""
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
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
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


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="既定: LUNA_MODEL")
    args = parser.parse_args()

    settings = Settings(werewolf_llm_provider="luna")
    model = args.model or settings.luna_model

    print("=== 1. 設定の確認 ===")
    print(f"{INFO}WEREWOLF_LLM_PROVIDER = {settings.werewolf_llm_provider}")
    print(f"{INFO}LUNA_BASE_URL         = {settings.luna_base_url}")
    print(f"{INFO}LUNA_MODEL            = {model}")
    print(f"{INFO}LUNA_API_KEY          = {_mask(settings.luna_api_key)}")

    problems = []
    if not settings.luna_api_key:
        problems.append(
            "LUNA_API_KEY が空です。Codespaces secrets に登録したあと、"
            "Codespace を再起動しないと反映されません。"
        )
    if "example.com" in settings.luna_base_url:
        problems.append(
            f"LUNA_BASE_URL がプレースホルダーのままです ({settings.luna_base_url})。"
            "実際のエンドポイントURLを設定してください。"
        )
    if problems:
        print()
        for problem in problems:
            print(f"{NG}{problem}")
        return 1

    from openai import AsyncOpenAI  # imported late so the checks above run without it

    client = AsyncOpenAI(
        api_key=settings.luna_api_key,
        base_url=settings.luna_base_url,
        timeout=settings.luna_timeout_seconds,
    )

    dialect = EndpointDialect()

    print("\n=== 2. 疎通と認証 ===")
    plain = await _one_call(client, model, None, dialect)
    if not plain["ok"]:
        print(f"{NG}接続に失敗しました: {plain['error']}")
        print("\n確認してください:")
        print("  - LUNA_BASE_URL は末尾が /v1 になっていますか")
        print("  - キーは有効ですか(失効・コピーミス)")
        print("  - モデル名は正しいですか")
        return 1
    print(f"{OK}応答あり ({plain['latency']:.2f} 秒)")
    print(f"{INFO}送信パラメータ: {dialect.describe()}")
    print(f"{INFO}生の応答: {str(plain['content'])[:160]}")

    if plain["prompt_tokens"] is not None:
        print(
            f"{OK}usage を返しています "
            f"(入力{plain['prompt_tokens']} / 出力{plain['completion_tokens']}トークン)"
        )
    else:
        print(f"{NG}usage を返していません → コスト集計は「不明」表示になります")

    print("\n=== 3. 構造化出力の対応状況 ===")
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
    )
    strict_ok = strict["ok"] and _validates(strict.get("content"))
    if strict_ok:
        print(f"{OK}strict JSON-schema に対応 ({strict['latency']:.2f} 秒)")
    elif strict["ok"]:
        print(f"{NG}strict は通ったがスキーマに合いません: {str(strict['content'])[:120]}")
    else:
        print(f"{NG}strict JSON-schema 非対応: {strict['error'][:160]}")

    loose = await _one_call(client, model, {"type": "json_object"}, dialect)
    loose_ok = loose["ok"] and _validates(loose.get("content"))
    if loose_ok:
        print(f"{OK}json_object モードに対応 ({loose['latency']:.2f} 秒)")
    elif loose["ok"]:
        print(f"{INFO}json_object は通りましたが直接の検証は失敗 → 寛容パーサーが処理します")
    else:
        print(f"{NG}json_object 非対応: {loose['error'][:160]}")

    print("\n=== 判定 ===")
    if strict_ok:
        print(f"{OK}そのまま評価を実行できます。JSON成功率は高くなる見込みです。")
    elif loose_ok or (plain["ok"] and _validates(plain.get("content"))):
        print(f"{INFO}strict は使えませんが、フォールバック経路で動作します。")
        print(f"{INFO}レポートの「strict schema で成功」が低く出るのは想定どおりです。")
    else:
        print(f"{NG}JSON形式の応答が安定しません。評価前にプロンプト調整が必要かもしれません。")

    print("\n次のステップ:")
    print("  python scripts/evaluate.py --games 1 --provider luna --out eval-out")
    tokens = plain["prompt_tokens"]
    if tokens:
        print(
            f"  (1ゲームは約280回の呼び出し。今回の1回が入力{tokens}トークンだったので、"
            "レポートの合計値で実コストを確認してください)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
