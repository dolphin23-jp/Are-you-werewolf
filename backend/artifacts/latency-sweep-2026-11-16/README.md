# レイテンシ測定(2026-11-16)

Phase 1b(approach-reset §11.5)の速度スイープの結果です。記録済みの v3 プロンプト(オフライン対戦
シード1から層別に40件)を、5つの条件で同じ実行の中で交互に送り、ストリーミングで時間を測りました。

| 項目 | 値 |
|---|---|
| ワークフロー | GitHub Actions「レイテンシ測定」run #1(run id 43843464103) |
| コミット | `38cce564aef9df109830da297e7f380f25af56d6` |
| 条件 | `claude-opus-5-5:low`、`claude-opus-5-5:medium`、`claude-opus-5-5:low:fast`、`claude-sonnet-5-5:low`、`claude-haiku-5-5:low` |
| 要求 | 200(40件 × 5条件)、フォールバックと beta は送らない |
| 費用 | $6.1451(上限 $10) |

ファイル: `summary.md` / `summary.json`(条件 × スキーマごとの集計)、`plan.md` / `plan.json`(標本と
見積もり)、`rows.csv`(1要求1行の生データ)。プロンプト本文(`prompts.jsonl`)は再生成できるので
入れていません(`python scripts/latency_sweep.py --dry-run --seed 1 --prompts 40`)。

**時間だけの測定です。** 発言の質は測っていません。`claude-opus-5-5:low:fast` の議論で1件、
`end_turn` で終わった154字の本文がスキーマの検証を通りませんでした(本文は保存されていないため
原因は未特定)。モデルや effort の変更は、ゲームの評価で品質を確かめてから決めます(オーナーの判断)。
