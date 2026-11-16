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
  実際の Claude を使うには、GitHub の
  Settings → Codespaces → Secrets に次を登録してください
  (Repository access はこのリポジトリだけに):

      WEREWOLF_LLM_PROVIDER      = anthropic
      WEREWOLF_ANTHROPIC_API_KEY = 実際のキー
      WEREWOLF_ACCESS_PASSWORD   = 長いランダムなパスワード

  パスワードが空だと anthropic ではサーバーが起動しません。
  モデルを変えるときだけ WEREWOLF_ANTHROPIC_MODEL を登録します
  (既定 claude-opus-5-5)。キーは、Anthropic Console で月の
  利用上限を設定した専用ワークスペースで発行するのがおすすめです。
  luna (gpt-6-luna) を使う場合は README を参照してください。

  登録後は Codespace を作り直す(または再起動する)と反映されます。
============================================================

EOF
