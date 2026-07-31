# Are you werewolf?

AI 16人と挑む、本格チャット人狼ゲーム。人間1人 + 生成AI16人の17人人狼を、
低コスト・高性能な LLM である **gpt-5.6-luna**(OpenAI互換API経由)で動かします。

過去の実装(Claude Haiku ベース)の設計思想 —
エンジン/AI分離、5層プロンプト構成、人格システム、人狼陣営の欺瞞パターン
事前コミット、モックLLMファーストのテスト戦略 — を引き継ぎつつ、コードは
一から新規に書き直しています。

## 構成

```
are-you-werewolf/
  backend/    Python + FastAPI。ゲームエンジン(LLM非依存) + AIプレイヤー層
  frontend/   Vite + React + TypeScript
```

## ゲームルール(17人)

| 役職 | 人数 | 陣営 |
|---|---|---|
| 村人 | 7 | 村人陣営 |
| 人狼 | 3 | 人狼陣営 |
| 狂人 | 1 | 人狼陣営 |
| 占い師 | 1 | 村人陣営 |
| 霊媒師 | 1 | 村人陣営 |
| 狩人 | 1 | 村人陣営 |
| 妖狐 | 1 | 妖狐陣営 |
| 共有者 | 2 | 村人陣営 |

## セットアップ

### バックエンド

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # 必要に応じて編集(デフォルトは mock プロバイダ)
uvicorn app.main:app --reload
```

`WEREWOLF_LLM_PROVIDER=mock`(デフォルト)ならAPIキー不要でネットワーク費用ゼロで動作します。
実際に gpt-5.6-luna を使う場合は `WEREWOLF_LLM_PROVIDER=luna` にして
`LUNA_API_KEY` / `LUNA_BASE_URL` / `LUNA_MODEL` を設定してください
(`.env.example` の値はプレースホルダーです)。

### フロントエンド

```bash
cd frontend
pnpm install
cp .env.example .env.local   # 必要に応じて編集
pnpm dev
```

`http://localhost:5173` を開くとブラウザで遊べます(バックエンドは
`http://localhost:8000` で起動している前提)。

## テスト

```bash
# バックエンド(全てモックプロバイダ、ネットワーク不要)
cd backend && pytest

# フロントエンド
cd frontend && pnpm test
```

ヘッドレスでAIの挙動を確認したい場合:

```bash
cd backend && python scripts/dry_run.py --seed 1 --provider mock
```

## 設計ドキュメント

実装計画・設計思想の詳細は `docs/architecture.md` を参照してください。
