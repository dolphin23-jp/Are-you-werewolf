# Are you werewolf?

AI 16人と挑む、本格チャット人狼ゲーム。人間1人 + 生成AI16人の17人人狼を、
Anthropic API 経由の **Claude**(既定は Claude Opus 5.5)で動かします。

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
実際の Claude を使う場合は次項へ。

## APIキーの入力手順(GitHub の設定画面)

> **このリポジトリは公開(public)です。APIキーはリポジトリのファイルには絶対に書きません。**
> GitHub の Secrets 画面に入力すると、Codespace に環境変数として渡されます。

**最初に費用の上限を決めておくことを勧めます。** Anthropic Console でこのゲーム専用の
ワークスペースを作り、月の利用上限(spend limit)を設定してから、そのワークスペースで
APIキーを発行します。キーが漏れても、使われる額はその上限までです。

1. GitHub の **Settings**(自分のアカウント設定)→ **Codespaces** →
   **Codespaces secrets** → **New secret** を開く
2. 次の3つを登録し、いずれも Repository access で**このリポジトリだけ**を許可する

   | Name | Value |
   |---|---|
   | `WEREWOLF_LLM_PROVIDER` | `anthropic` |
   | `WEREWOLF_ANTHROPIC_API_KEY` | Anthropic Console で発行した実際のキー |
   | `WEREWOLF_ACCESS_PASSWORD` | 長いランダムなパスワード(画面を開くときに使う) |

   必要なら次も登録できます(どちらも省略可)。

   | Name | 既定値 | 意味 |
   |---|---|---|
   | `WEREWOLF_ANTHROPIC_MODEL` | `claude-opus-5-5` | 使うモデル |
   | `WEREWOLF_SESSION_MAX_COST_USD` | `30` | 1セッションあたりの推定API費用の上限(USD)。超えるとそのセッションではそれ以上APIを呼びません |

   `WEREWOLF_LLM_PROVIDER` はモデル名ではなくプロバイダ種別です。値には
   `anthropic` だけを入力します(モデルは `WEREWOLF_ANTHROPIC_MODEL` で指定します)。
   キーの名前は `ANTHROPIC_API_KEY` ではなく **`WEREWOLF_ANTHROPIC_API_KEY`** です。
   ほかのツール向けの `ANTHROPIC_*` を誤って使わないよう、このアプリは
   `WEREWOLF_ANTHROPIC_*` だけを読みます。

3. **Codespace を再起動**(または作り直す)と反映されます
4. 反映確認は `/api/health` を開き、`"llm_provider": "anthropic"` になっていればOK

`anthropic` では **`WEREWOLF_ACCESS_PASSWORD` が必須**です。空のままだとサーバーは
起動しません(誰でも課金の試合を作れる状態で動かさないため)。ブラウザの認証画面では、
ユーザー名 `werewolf` と、設定したパスワードを入力します。自分のパソコンだけで動かす
場合に限り、`WEREWOLF_ALLOW_OPEN_ACCESS=true` でこの確認を外せます(Codespaces や
公開デプロイでは使わないでください)。

キーを入れずに `WEREWOLF_LLM_PROVIDER=anthropic` だけ設定した場合は、
誤って課金APIを叩く前に起動時の警告とエラーで止まります。

### 別の選択肢: luna(gpt-6-luna、休眠中)

以前の実プロバイダ **gpt-6-luna**(OpenAI互換API)も選べます。これまでの計測
(`docs/approach-reset-2026-10.md` の §2.1b など)はこのモデルの値です。使う場合は
Codespaces secrets に `WEREWOLF_LLM_PROVIDER=luna`、`LUNA_API_KEY`(実際のキー)、
`LUNA_BASE_URL`(OpenAI互換エンドポイントURL)を登録します。モデルは `LUNA_MODEL`
(既定 `gpt-6-luna`)、思考量は `LUNA_REASONING_EFFORT=low` などで変えられます。

## 推論エンジン(v3 / v2 / legacy)

AIの疑い・投票・夜行動・発言は既定で **v3** エンジンが担います。事実台帳・論理
ソルバー・検証・発言順・公開義務(結果の公開、CO)はコードが保証し、**誰を疑うか・
誰に投票するか・夜に何をするか・何を言うかは、モデルが「盤面メモ」を読んで決めます**。
発言はチャット調で、定型文は挿入しません。同じモデル(gpt-6-luna)・同じシードで比べると、
v3 は v2 より約4倍速く(27分と1時間53分)、呼び出しの失敗がありませんでした
(`docs/approach-reset-2026-10.md` の §2.1b)。

**v2**(`WEREWOLF_REASONING_ENGINE=v2`)は、コードが結論を決め、モデルは言い回し
だけを担当する以前の方式です。比較用に残してあり、保守のみです。**legacy** は、
投票・夜行動・発言内容のすべてをモデルに任せる、さらに前の方式です。どちらも
`WEREWOLF_LLM_PROVIDER` と同じ要領で、Codespaces secrets・`backend/.env`・Docker の
環境変数のいずれかに設定すると切り替わります。反映確認は `/api/health` の
`"reasoning_engine"` で行えます(`v3` が既定)。

> Codespaces secrets や常設デプロイの環境変数に、以前の値(`v2` など)が
> 残っていると、そちらが優先されます。v3 にしたい場合は、その設定を削除するか
> `v3` に直してください。

Claude の思考量は、議論・投票・夜行動・チャットごとに `WEREWOLF_ANTHROPIC_EFFORT_*` で
変えられます(既定は議論とチャットが `low`、投票と夜行動が `medium`。`docs/evaluation.md`)。

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
  LLMのキーは常にバックエンド側(`WEREWOLF_ANTHROPIC_API_KEY`、luna なら `LUNA_API_KEY`)に
  置いてください。
- CI(`.github/workflows/ci.yml`)はモックだけで動き、キーを一切使いません。キーを使うのは
  手動実行だけの評価ワークフローで、GitHub の Actions secrets から渡します。
- 実環境変数は `.env` より優先されます。Codespaces や各種ホスティングの
  環境変数設定に登録すれば、コードも `.env` も不要です。

万一キーをコミットしてしまった場合は、履歴から消すより先に
**発行元(Anthropic Console)でそのキーを失効(revoke)させ、新しいキーを再発行**してください。

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

## 常設デプロイ(Docker)

Codespaceを毎回起動せず、スマートフォンなどから同じURLで遊ぶ場合は、付属の
`Dockerfile`をWebサービス型のホスティングへデプロイします。サービスには最低限、
次のSecretを設定してください。

```text
WEREWOLF_LLM_PROVIDER=anthropic
WEREWOLF_ANTHROPIC_API_KEY=<実際のキー>
WEREWOLF_ANTHROPIC_MODEL=claude-opus-5-5
WEREWOLF_ACCESS_PASSWORD=<長いランダムなパスワード>
WEREWOLF_SESSION_MAX_COST_USD=30
```

公開URLを第三者に使われるとAPI費用が発生するため、`WEREWOLF_ACCESS_PASSWORD`は
必ず設定してください(`anthropic` では、未設定だとサーバーが起動しません)。ブラウザの
認証画面ではユーザー名`werewolf`と、設定したパスワードを入力します。`/api/health`だけは
ホスティングの死活監視用に認証不要です。Anthropic Console 側の月の利用上限もあわせて
設定してください。

```bash
docker build -t are-you-werewolf .
docker run --rm -p 8000:8000 --env-file backend/.env are-you-werewolf
```

現在のゲームセッションはメモリ保存なので、サービスは1インスタンスで運用し、
プレイ中に再起動・スケールダウンしない設定にしてください。

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
python scripts/check_llm.py                                       # まず接続確認(少ない呼び出しだけ)
python scripts/evaluate.py --games 1 --provider anthropic --max-cost 20
```

`--provider anthropic` では、費用の上限 `--max-cost`(USD)の指定が必須です。推定費用が
上限を超えそうになると、その呼び出しを送らずに評価を止めます。

詳細は `docs/evaluation.md` を参照してください。モデル・思考量ごとの応答時間を測る
「レイテンシ測定」と、AIの発言を人間が見分けられるかを測る人間判定クイズは
`docs/measurement-scripts.md` にあります。

人間がプレイするゲームでは、`/api/health` が `llm_model` を返し、ゲーム画面の取得結果
(`/view`)に `ai_budget_exhausted`・`ai_cost_usd` が入ります。セッションの費用上限
(`WEREWOLF_SESSION_MAX_COST_USD`)に達するとAIは発言・投票を止め、画面に知らせが出ます。
`WEREWOLF_LOG_LEVEL` はアプリ自身のログ(プロバイダの失敗の警告を含む)にも効きます。

## 設計ドキュメント

実装計画・設計思想の詳細は `docs/architecture.md` を参照してください。
現在の優先順位と、実プロバイダを Anthropic に切り替えた理由・設定は
`docs/approach-reset-2026-10.md`(§11)にあります。
