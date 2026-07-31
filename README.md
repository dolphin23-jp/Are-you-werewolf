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

## 遊び方(GitHub Codespaces / ローカル環境不要)

ブラウザだけで完結します。パソコンへのインストールは一切不要です。

1. GitHub のリポジトリページで **Code → Codespaces → Create codespace** をクリック
2. ブラウザ上で VS Code が開き、自動セットアップが走る(数分)
3. ターミナルで次を実行

   ```bash
   bash .devcontainer/start.sh
   ```

4. 下部の **ポート** タブに出る **8000番** のURLを開くと遊べます

APIキーを設定しなくても、**モックAI相手にそのまま遊べます**(通信費ゼロ)。
実際の gpt-5.6-luna を使う場合は次項へ。

## APIキーの入力手順(GitHub の設定画面)

> **このリポジトリは公開(public)です。APIキーはリポジトリのファイルには絶対に書きません。**
> GitHub の Secrets 画面に入力すると、Codespace に環境変数として渡されます。

1. GitHub の **Settings**(自分のアカウント設定)→ **Codespaces** →
   **Codespaces secrets** → **New secret** を開く
2. 次の3つを登録し、いずれも Repository access でこのリポジトリを許可する

   | Name | Value |
   |---|---|
   | `WEREWOLF_LLM_PROVIDER` | `luna` |
   | `LUNA_API_KEY` | 発行された実際のキー |
   | `LUNA_BASE_URL` | gpt-5.6-luna のOpenAI互換エンドポイントURL |

   `WEREWOLF_LLM_PROVIDER` はモデル名ではなくプロバイダ種別です。値には
   `gpt-5.6-luna` や `WEREWOLF_LLM_PROVIDER=luna` ではなく、`luna` だけを入力します。

3. **Codespace を再起動**(または作り直す)と反映されます
4. 反映確認は `/api/health` を開き、`"llm_provider": "luna"` になっていればOK

キーを入れずに `WEREWOLF_LLM_PROVIDER=luna` だけ設定した場合は、
誤って課金APIを叩く前に起動時の警告とエラーで止まります。

### ローカルで動かす場合(任意)

```bash
cd backend
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env      # ここにキーを書く(.env は gitignore 済み)
cd ../frontend && pnpm install && pnpm build
cd ../backend && uvicorn app.main:app --reload
```

`.env` が git から見えないことは次で確認できます(**何も出なければ安全**)。

```bash
git status --porcelain backend/.env
```

### なぜ公開リポジトリでも安全か

- キーはリポジトリのファイルに一切書きません。GitHub の Secrets 画面(または
  gitignore 済みの `backend/.env`)にだけ入力します。
- APIキーは**バックエンドのプロセス内のみ**で使われます。フロントエンドは
  自分と同じオリジンにリクエストするだけで、キーを一切知りません。
- **`VITE_` 接頭辞の変数にキーを書かないでください。** Viteは `VITE_*` を
  ブラウザ向けバンドルに埋め込むため、書くと全訪問者にキーが露出します。
  LLMのキーは常にバックエンド側(`LUNA_*`)に置いてください。
- 実環境変数は `.env` より優先されます。Codespaces や各種ホスティングの
  環境変数設定に登録すれば、コードも `.env` も不要です。

万一キーをコミットしてしまった場合は、履歴から消すより先に
**発行元でそのキーを失効(revoke)させ、新しいキーを再発行**してください。

## 構成上のポイント: 単一ポート

バックエンド(FastAPI)がビルド済みのフロントエンドも配信します。
API・画面・WebSocket がすべて同一オリジンなので:

- 転送するポートが1つで済む(Codespaces のポート転送と相性が良い)
- CORS 設定が不要
- WebSocket はページのスキームを継承するため、Codespaces の HTTPS 転送でも
  `wss://` が自動的に使われる

フロントエンドを開発サーバーで動かす場合(`pnpm dev`)も、Vite が `/api` と
`/ws` をバックエンドへプロキシするので、コード側は常に同一オリジン前提の
1経路だけで済みます。

```bash
# 画面のホットリロードが欲しいときだけ、2つ起動する
cd backend && uvicorn app.main:app --reload   # ターミナル1
cd frontend && pnpm dev                        # ターミナル2 → localhost:5173
```

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

## AIの品質評価

日本語の自然さ・人格の維持・役職整合性・矛盾・人狼の連携・JSON成功率・応答時間・
実コストを測る評価ハーネスがあります。

**スマホだけで実行できます**: リポジトリの **Actions** タブ →
**AI評価** → **Run workflow** をタップすると評価が走り、レポートが実行ページに
そのまま表示されます(ターミナル不要)。

ターミナルから実行する場合:

```bash
cd backend
python scripts/check_llm.py                              # まず接続確認(1回だけ)
python scripts/evaluate.py --games 1 --provider luna --judge
```

詳細は `docs/evaluation.md` を参照してください。

## 設計ドキュメント

実装計画・設計思想の詳細は `docs/architecture.md` を参照してください。
