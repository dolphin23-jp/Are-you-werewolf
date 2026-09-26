#!/usr/bin/env bash
# Builds the frontend if needed, then serves the whole app (API + UI) on a
# single port -- which is what lets Codespaces forward just one URL.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Rebuild when the bundle is missing or older than its sources: after a pull
# the old check kept serving the previous UI.
if [ ! -f frontend/dist/index.html ] || [ -n "$(find frontend/src frontend/index.html \
    frontend/package.json -newer frontend/dist/index.html -print -quit 2>/dev/null)" ]; then
  echo "==> frontend/dist が無いか古いのでビルドします"
  (cd frontend && pnpm build)
fi

# Same normalisation as Settings: trim, drop quotes, lower-case, and accept the
# model name typed into the provider field.
PROVIDER="$(printf '%s' "${WEREWOLF_LLM_PROVIDER:-mock}" | tr -d "\"'" | tr '[:upper:]' '[:lower:]' | xargs)"
if [ "$PROVIDER" = "gpt-5.6-luna" ]; then
  PROVIDER="luna"
fi
echo "==> LLM プロバイダ: ${PROVIDER}"
if [ "$PROVIDER" = "luna" ] && [ -z "${LUNA_API_KEY:-}" ]; then
  echo "!!! WEREWOLF_LLM_PROVIDER=luna ですが LUNA_API_KEY が空です。" >&2
  echo "!!! GitHub の Settings → Codespaces → Secrets を確認してください。" >&2
elif [ "$PROVIDER" = "mock" ]; then
  echo "    (モックAIで動作します。実APIを使うには Codespaces Secrets を設定)"
fi

cd backend
UVICORN="uvicorn"
if [ -x .venv/bin/uvicorn ]; then
  UVICORN=".venv/bin/uvicorn"
fi
exec "$UVICORN" app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
