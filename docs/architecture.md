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
- **COは発言テキストからの検出のみで確定**(`backend/app/ai/coordinator.py`):
  隠しチャネルでのCO操作を許さず、エマージェントな騙し合いを維持する。
- **フェーズ別並行性ポリシー**: 議論=逐次、投票=並列、夜=依存関係あり
  (占い/護衛/内輪チャットが並列 → 人狼の襲撃決定はそれらを踏まえて後段)。
- **モックLLMファースト**(`backend/app/ai/provider/mock.py`): テストと
  `scripts/dry_run.py` はネットワーク費用ゼロで完結する。

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
