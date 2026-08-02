# アーキテクチャ / 設計思想

## 背景

本プロジェクトは、過去に作られた人狼AIプレイヤー実装(Claude Haiku ベース)の
「理念」を抽出し、LLMプロバイダを **gpt-5.6-luna**(OpenAI互換API)に差し替えた
上で、コードは一から新規に設計・実装したものです。

引き継いだ設計思想:

- **エンジン/AI分離**: `backend/app/engine/` はLLMを一切知らない、純粋で
  決定的なゲームルール実装。人間もAIも同じ `GameController` API
  (`chat`/`vote`/`submit_night_action`/`co` 等)を呼ぶ。
- **5層プロンプト構成**(`backend/app/ai/context.py`):
  [A] 人格 / [B] 役職情報 / [C] 盤面分析(ロープ計算など) / [D] 過去日要約
  (ローリング圧縮) / [E] 当日全文ログ。
- **人格システム**(`backend/app/ai/personalities.py`): 口調×思考スタイル×
  議論スタイル×感情傾向の4軸から成る15プリセット。
- **2層の戦略ガイド**(`backend/app/ai/strategy.py`): コード計算(ロープ数・
  グレーリスト・CO構成)+ 静的ドクトリン文。
- **欺瞞パターン事前コミット**(`backend/app/ai/deception.py`): 人狼陣営は
  ゲーム開始時に欺瞞パターン(偽占い/偽霊媒/全潜伏など)を確率的に決定し、
  `FakeClaimGuard` で嘘の一貫性を検証する。
- **COは公開文と構造化公開宣言の一致で確定**(`backend/app/ai/coordinator.py`):
  `public_claim_role` だけの隠しCOは認めず、公開文から同じ役職COを確認できた場合だけ登録する。
- **フェーズ別並行性ポリシー**: 議論=有限多段階、投票=並列、夜=依存関係あり。
  議論前に非公開の朝発言意思を並列生成し、結果発表・朝一COを先行させる。その後、全員の
  初期見解、直接質問、主要処刑候補の反論、再評価を優先度付きキューで処理する。1人3回
  (主要候補4回)、総発言数は生存者数の2.5倍までのため必ず停止する。夜は占い/護衛/内輪
  チャットを並列実行し、人狼の襲撃決定はそれらを踏まえて後段で行う。
- **モックLLMファースト**(`backend/app/ai/provider/mock.py`): テストと
  `scripts/dry_run.py` はネットワーク費用ゼロで完結する。

新たに追加した基盤:

- **公開事実台帳**(`backend/app/ai/reasoning/`): AI側が「事実」として扱って
  よい唯一の投影。`facts.PublicFactLedger` は `GameState` を参照するだけの
  読み取り専用ビューで、`co_declarations` / `public_result_claims` /
  `vote_records` などの可変コピーは作らない(スナップショットを持つと、
  翌朝に死亡者が処刑候補へ戻るような乖離が起きる)。真の役職・真の死因・
  他人の非公開能力結果はこの境界を越えない。

## 推理基盤の分離

推理は3つの独立した層に分けている。

1. **事実**(`reasoning/facts.py`): 公開情報だけを投影する。
2. **整合性検証**(`reasoning/validation.py`): AI出力のうち「あり得ないもの」
   だけを拒否・修正する。死亡者を処刑候補にする、未処刑者へ霊媒結果を出す、
   公開済みの白判定を黒へ反転させる、といった状態破損が対象で、「誰が人狼か
   を読み違える」ことは人間らしい誤りなので触らない。修正は必ず決定的で、
   本人の前回の結論→本人の suspects→席順の先頭、という順に解決する
   (ランダムな代替を選ぶと、説明のつかない投票が残る)。
3. **決定的要約**(`reasoning/summaries.py`): 日次要約の公開事実部分は
   LLMではなくコードで生成し、生成された講評は別見出しへ隔離する。

各規則は小さなモジュールに分かれており、後続の推理機能が単一の巨大ファイルへ
集中しないようにしている。宣言した処刑先と実際の投票先の食い違いは
`VotePlanMismatch` として**記録するだけ**で修正しない — 心変わりは正当なプレイで、
それを取り違えないのは読み手の仕事だからである。

## LLMプロバイダ

`backend/app/ai/provider/base.py` の `LLMProvider` プロトコルを介して
プロバイダを抽象化。実装は2つ:

- `mock.py`: 決定的・シード可能なテスト用ダブル。
- `luna_openai.py`: `openai.AsyncOpenAI` を使った gpt-5.6-luna 実装。
  strict JSON-schema structured output を第一経路とし、失敗時は
  `json_object` モード + 寛容パーサー(生パース→フェンス付きJSON抽出→
  最初の`{`〜最後の`}`)にフォールバックする。

`WEREWOLF_LLM_PROVIDER` 環境変数で明示的に切り替える(`mock` | `luna`)。
`luna` 選択時に `LUNA_API_KEY` が未設定なら起動時に即エラーとする
(旧実装のAPIキー文字列を見て自動フォールバックする挙動はやめた)。

## 旧実装から修正した既知の問題

| 旧実装の問題 | 対応 |
|---|---|
| 同期SDKを`async def`内でブロッキング呼び出し | `AsyncOpenAI`を徹底使用 |
| プレーンテキスト+正規表現JSONパースのみ | structured output を第一経路に |
| 単一グローバルインメモリセッション | `SessionStore`で複数ゲーム対応 |
| プロバイダ抽象化なし | `LLMProvider`プロトコル+2実装+factory |
| 投票/夜フェーズでのLLM呼び出しが無制限並列 | `LUNA_MAX_CONCURRENCY`セマフォ |
| APIキー文字列の中身で自動モック判定 | `WEREWOLF_LLM_PROVIDER`明示指定 |

## セッション管理

`backend/app/sessions/store.py` の `SessionStore` インターフェースにより、
複数ゲームを同時に扱える(v1はインメモリ実装。永続化が必要になれば
同インターフェースの別実装に差し替え可能)。

## API / 通信

REST(セッションスコープ) + WebSocket(リアルタイムpush、`GameEvent`を
JSON化してブロードキャスト) + ポーリングfallback。AIターンの
オーケストレーションは `backend/app/api/orchestrator.py` に集約され、
人間の行動(チャット/投票/夜行動)をトリガーに `AICoordinator` を呼ぶ。

## テスト

- `backend/tests/unit/`: エンジン単体(純粋pytest、多シードでのfuzz)
- `backend/tests/ai/`: AI層(`MockProvider`使用)
- `backend/tests/e2e/`: API経由・WebSocket接続込みの結合テスト、
  17人ゲームの完走テスト(複数シード)
- `frontend/src/test/`: Vitest + React Testing Library
