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
# A Claude model id typed into the provider field, which the backend then
# asks for (app/config.py) -- the line below shows it, so a run that bills
# a different model than the owner meant is visible before it starts.
PROVIDER_MODEL=""
case "$PROVIDER" in
  # A model name typed into the provider field means its provider, as in
  # the backend: "claude" / "claude-sonnet-5-5" -> anthropic, "gpt-6-luna" -> luna.
  claude) PROVIDER="anthropic" ;;
  claude-*)
    PROVIDER_MODEL="$PROVIDER"
    PROVIDER="anthropic"
    ;;
  gpt-*-luna) PROVIDER="luna" ;;
esac
echo "==> LLM プロバイダ: ${PROVIDER}"

case "$PROVIDER" in
  anthropic)
    # Quotes and spaces dropped as the backend does, case kept: it sends the
    # model id as written. The default is DEFAULT_ANTHROPIC_MODEL in
    # app/config.py (tests/unit/test_config.py keeps the two in step).
    CONFIGURED_MODEL="$(printf '%s' "${WEREWOLF_ANTHROPIC_MODEL:-}" | tr -d "\"' ")"
    if [ -n "$PROVIDER_MODEL" ] && [ -n "$CONFIGURED_MODEL" ] \
      && [ "$(normalize "$CONFIGURED_MODEL")" != "$PROVIDER_MODEL" ]; then
      echo "!!! WEREWOLF_LLM_PROVIDER は ${PROVIDER_MODEL}、WEREWOLF_ANTHROPIC_MODEL は ${CONFIGURED_MODEL} で、" >&2
      echo "!!! モデルが食い違っています。どちらか一方だけにしてください(このままではサーバーが起動しません)。" >&2
    fi
    echo "==> モデル: ${CONFIGURED_MODEL:-${PROVIDER_MODEL:-claude-sonnet-5-5}}"
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
    case "$(normalize "${WEREWOLF_ALLOW_OPEN_ACCESS:-}")" in
      1 | on | t | true | y | yes) OPEN_ACCESS="true" ;;
      *) OPEN_ACCESS="false" ;;
    esac
    if [ -z "${WEREWOLF_ACCESS_PASSWORD:-}" ] && [ "$OPEN_ACCESS" != "true" ]; then
      echo "!!! WEREWOLF_ACCESS_PASSWORD が空です。課金のあるプロバイダ(luna も)ではパスワード無しでは" >&2
      echo "!!! 起動しません。Codespaces Secrets に WEREWOLF_ACCESS_PASSWORD を登録してください。" >&2
      echo "!!! 自分のパソコンだけで使う場合に限り WEREWOLF_ALLOW_OPEN_ACCESS=true で外せます。" >&2
    fi
    ;;
  mock)
    echo "    (モックAIで動作します。実APIを使うには Codespaces Secrets を設定)"
    ;;
esac

cd backend
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
