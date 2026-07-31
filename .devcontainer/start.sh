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

PROVIDER="${WEREWOLF_LLM_PROVIDER:-mock}"
echo "==> LLM プロバイダ: ${PROVIDER}"
if [ "$PROVIDER" = "luna" ] && [ -z "${LUNA_API_KEY:-}" ]; then
  echo "!!! WEREWOLF_LLM_PROVIDER=luna ですが LUNA_API_KEY が空です。" >&2
  echo "!!! GitHub の Settings → Codespaces → Secrets を確認してください。" >&2
elif [ "$PROVIDER" = "mock" ]; then
  echo "    (モックAIで動作します。実APIを使うには Codespaces Secrets を設定)"
fi

cd backend
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
