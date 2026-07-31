#!/usr/bin/env bash
# Runs once when the Codespace is created (devcontainer postCreateCommand).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Installing backend dependencies"
pip install --no-cache-dir -e "./backend[dev]"

echo "==> Installing frontend dependencies"
corepack enable
corepack prepare pnpm@10 --activate
cd frontend
pnpm install --frozen-lockfile

echo "==> Building frontend"
pnpm build

cd "$REPO_ROOT"
cat <<'EOF'

============================================================
  セットアップ完了

  起動するには、ターミナルで次を実行してください:

      bash .devcontainer/start.sh

  起動後、下部の「ポート」タブに出る 8000 番のURLを開くと
  ブラウザで遊べます。

  API キーは未設定でもモックAI相手に遊べます。
  実際の gpt-5.6-luna を使うには、GitHub の
  Settings → Codespaces → Secrets に次を登録してください:

      WEREWOLF_LLM_PROVIDER = luna
      LUNA_API_KEY          = 実際のキー
      LUNA_BASE_URL         = エンドポイントURL

  登録後は Codespace を作り直す(または再起動する)と反映されます。
============================================================

EOF
