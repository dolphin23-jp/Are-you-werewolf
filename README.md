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

`WEREWOLF_LLM_PROVIDER=mock`(デフォルト)ならAPIキー不要・ネットワーク費用ゼロで動作します。

## APIキーの設定手順

> **このリポジトリは公開(public)です。APIキーは絶対にコミットしないでください。**
> 手順は「gitが追跡しない `backend/.env` に書く」だけです。

1. 雛形をコピーする(既に `.env` があればこの手順は不要)

   ```bash
   cd backend
   cp .env.example .env
   ```

2. `backend/.env` を編集し、次の4項目を実際の値にする

   ```
   WEREWOLF_LLM_PROVIDER=luna
   LUNA_API_KEY=<発行された実際のキー>
   LUNA_BASE_URL=<gpt-5.6-luna のOpenAI互換エンドポイント>
   LUNA_MODEL=gpt-5.6-luna
   ```

3. 念のため、gitから見えないことを確認する(**何も出力されなければ安全**)

   ```bash
   git status --porcelain backend/.env
   ```

4. 起動する(`.env` は `backend/` を作業ディレクトリとして読まれます)

   ```bash
   uvicorn app.main:app --reload
   ```

   キーが未設定のまま `luna` を選ぶと、誤って課金APIを叩く前に起動時エラーで止まります。

### なぜ公開リポジトリでも安全か

- `.env` は `.gitignore` で除外済み。追跡されるのは値が空の `.env.example` のみです。
- APIキーは**バックエンドのプロセス内のみ**で使われます。フロントエンドは
  バックエンドのURL(`VITE_API_BASE_URL` / `VITE_WS_BASE_URL`)しか知りません。
- **`VITE_` 接頭辞の変数にキーを書かないでください。** Viteは `VITE_*` を
  ブラウザ向けバンドルに埋め込むため、書くと全訪問者にキーが露出します。
  LLMのキーは常にバックエンド側(`LUNA_*`)に置いてください。

### デプロイ時

`.env` ファイルを置く代わりに、ホスティング先(Render / Fly.io / Cloud Run 等)の
環境変数設定画面に `LUNA_API_KEY` 等を登録してください。実環境変数は `.env` より
優先されるため、コードの変更は不要です。

万一キーをコミットしてしまった場合は、履歴から消すより先に
**発行元でそのキーを失効(revoke)させ、新しいキーを再発行**してください。

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
