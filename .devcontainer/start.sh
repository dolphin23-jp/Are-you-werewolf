#!/usr/bin/env bash
# Builds the frontend if needed, then serves the whole app (API + UI) on a
# single port -- which is what lets Codespaces forward just one URL.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -d frontend/dist ]; then
  echo "==> frontend/dist が無いのでビルドします"
  (cd frontend && pnpm build)
fi

# Lower-case and drop quotes/spaces, so the check below reads a Codespaces
# secret the same way the backend does (app/config.py normalises it too).
normalize() {
  printf '%s' "$1" | tr -d "\"' " | tr '[:upper:]' '[:lower:]'
}

PROVIDER="$(normalize "${WEREWOLF_LLM_PROVIDER:-mock}")"
case "$PROVIDER" in
  # A model name typed into the provider field means its provider, as in
  # the backend: "claude" / "claude-opus-5-5" -> anthropic, "gpt-6-luna" -> luna.
  claude | claude-*) PROVIDER="anthropic" ;;
  gpt-*-luna) PROVIDER="luna" ;;
esac
echo "==> LLM プロバイダ: ${PROVIDER}"

case "$PROVIDER" in
  anthropic)
    if [ -z "${WEREWOLF_ANTHROPIC_API_KEY:-}" ]; then
      echo "!!! WEREWOLF_LLM_PROVIDER=anthropic ですが WEREWOLF_ANTHROPIC_API_KEY が空です。" >&2
      echo "!!! GitHub の Settings → Codespaces → Secrets を確認してください。" >&2
      echo "!!! (ANTHROPIC_API_KEY という名前では読みません。WEREWOLF_ANTHROPIC_API_KEY で登録します)" >&2
    fi
    # Same truthy spellings as the backend's boolean setting.
    case "$(normalize "${WEREWOLF_ALLOW_OPEN_ACCESS:-}")" in
      1 | on | t | true | y | yes) OPEN_ACCESS="true" ;;
      *) OPEN_ACCESS="false" ;;
    esac
    if [ -z "${WEREWOLF_ACCESS_PASSWORD:-}" ] && [ "$OPEN_ACCESS" != "true" ]; then
      echo "!!! WEREWOLF_ACCESS_PASSWORD が空です。anthropic ではパスワード無しでは起動しません" >&2
      echo "!!! (届いた人が誰でも課金の試合を作れてしまうため)。Codespaces Secrets に" >&2
      echo "!!! WEREWOLF_ACCESS_PASSWORD を登録してください。自分のパソコンだけで使う場合に限り" >&2
      echo "!!! WEREWOLF_ALLOW_OPEN_ACCESS=true で外せます。" >&2
    fi
    ;;
  luna)
    if [ -z "${LUNA_API_KEY:-}" ]; then
      echo "!!! WEREWOLF_LLM_PROVIDER=luna ですが LUNA_API_KEY が空です。" >&2
      echo "!!! GitHub の Settings → Codespaces → Secrets を確認してください。" >&2
    fi
    ;;
  mock)
    echo "    (モックAIで動作します。実APIを使うには Codespaces Secrets を設定)"
    ;;
esac

cd backend
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
