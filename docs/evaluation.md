# AIプレイヤーの評価

`backend/scripts/evaluate.py` がN戦を実行し、レポートと対戦記録を出力します。

評価では人間席 `p0` もAIへ置き換え、17人全員をAIとして開始します。0日目に通常どおり
初日犠牲者が1人死亡し、1日目は生存AI16人で議論します。人間の発言や投票をランダム行動で
代用せず、実際の17人卓と同じ人数・配役・勝利条件でAI同士のプレイを観察します。

## スマホだけで実行する(GitHub Actions・ターミナル不要)

端末にもCodespacesにも入らず、ブラウザのボタンだけで評価を回せます。

### 初回のみ: シークレットの登録

GitHubのリポジトリページ → **Settings → Secrets and variables → Actions** で登録します。

| 種別 | 名前 | 値 |
|---|---|---|
| Secret | `WEREWOLF_ANTHROPIC_API_KEY` | Anthropic Console で発行した実際のキー |
| Variable | `WEREWOLF_ANTHROPIC_MODEL` | `claude-opus-5-5`(既定値なので省略可) |
| Variable | `WEREWOLF_ANTHROPIC_MAX_TOKENS` | `12000`(既定値なので省略可。途中切れが多いときに上げる。AI評価だけが読みます) |

> Codespaces secrets とは別枠です。Codespacesに登録済みでも、Actions用に改めて
> 登録が必要です。キーは手動実行だけの評価ワークフローに渡り、CI(`ci.yml`)には
> 渡りません。

キーは、月の利用上限(spend limit)を設定した専用のワークスペースで発行しておくと、
評価の回しすぎやキーの漏えいがあっても、使われる額がその上限で止まります。

**別の選択肢: luna(gpt-6-luna)。** luna で評価する場合だけ、次も登録します。

| 種別 | 名前 | 値 |
|---|---|---|
| Secret | `LUNA_API_KEY` | 発行された実際のキー |
| Secret | `LUNA_BASE_URL` | gpt-6-luna のエンドポイントURL |
| Variable | `LUNA_MODEL` | `gpt-6-luna`(既定値なので省略可) |
| Variable | `LUNA_PRICE_IN` / `LUNA_PRICE_OUT` | 入力/出力100万トークンあたりの単価(省略可。無いと luna の費用は計算されず、費用の上限も効きません) |

### 実行

1. リポジトリの **Actions** タブ → 左の **AI評価** を選ぶ
2. **Run workflow** をタップし、次の項目を入力して実行

   | 項目 | 既定 | 意味 |
   |---|---|---|
   | `provider` | `anthropic` | `anthropic`(Claude)か `luna`(休眠中の gpt-6-luna) |
   | `seeds` | `1` | シードをカンマ区切りで(例: `1,2,3`。最大20個)。**1シード=1ジョブ=1試合**で、最大4つを並列に回します |
   | `reasoning_engine` | `v3` | 推理エンジン(`v3` / `v2` / `legacy`)。`v2` と `legacy` は議論に `DiscussionOutput` を送るので、Claude では議論の手番が文法の強制なし(`json_object` 経路)で読まれます。v3 と比べるときはこの違いを含めて読んでください(`docs/approach-reset-2026-10.md` の §11.2)。接続確認は v3 のスキーマだけを確かめます |
   | `effort_discussion` | `low` | 議論と狼・共有チャットの思考量(`low` / `medium` / `high`。Claude のみ) |
   | `effort_decisions` | `medium` | 投票と夜行動の思考量(`low` / `medium` / `high`。Claude のみ) |
   | `judge` | オフ | 日本語・人格をLLMで採点する(追加費用。上限は対戦と共有) |
   | `judge_model` | 空 | 判定に使うモデル(空なら対戦と同じ。例: `claude-sonnet-5-5`) |
   | `max_cost_per_game` | `15` | 1試合(1ジョブ)あたりの推定費用の上限(USD) |
   | `model` | 空 | モデル。空なら変数 `WEREWOLF_ANTHROPIC_MODEL`、それも無ければ `claude-opus-5-5`(luna なら `LUNA_MODEL`、なければ `gpt-6-luna`) |

3. まず `plan` ジョブが入力を確かめてシードを展開し、`preflight` ジョブが接続確認
   (`scripts/check_llm.py`)を1回だけ行います。ここで失敗すれば、試合は1つも始まりません
4. シードごとの `evaluate (seed N)` ジョブが終わると、そのジョブの実行ページに
   **レポートがそのまま表示されます**(ダウンロード不要)。v3 の1戦は gpt-6-luna で
   約28分でした。Claude ではまだ計測していません(ジョブの上限は120分)
5. 全出力はシードごとに Artifacts の `eval-out-seedN` からダウンロードできます(保持は90日)

対戦記録も折りたたみで同じページに出るので、発言の日本語や人格の維持を
スマホの画面で直接確認できます。

課金が発生するため、pushでは起動せず**手動実行のみ**にしてあります。評価の2つの
ワークフロー(AI評価・Expert cutoff評価)は同じ順番待ち(`ai-evaluation`)に入り、
同時には1つしか走りません。走っている評価は取り消されず、次の1つは待ちます。
待てるのは1つだけで、さらに押すと、待っていたほうが取り消されて新しいほうが待ちます。

**費用の上限で止まったとき。** 上限に届く呼び出しは送らずに、その試合をそこで止めます。
レポートの先頭に「費用上限に達したため中断」と出て、対戦記録は「途中経過」として残り、
ジョブは失敗(終了コード3)になります。完走した試合と見分けるためです。

### 実行中の見方

レポートは対戦が終わってから出ますが、**実行中も進み具合は見えます**。
実行ページ → `evaluate (seed N)` ジョブ → **評価の実行** をタップすると、対戦がそのまま
流れます(次は表示の例です)。

```text
[01:12] ── 1日目 昼の議論 ── 呼び出し 14回(失敗 0・途中切れ 0) 平均 9.8秒 推論トークン 62%
[01:31] 1日目 ハルト: 占い結果、ミオさんは白でした。次はサクラさんを見たいです
[02:03] 1日目 サクラ [フォールバック]: 現時点の第一処刑候補は…
[04:15] … 生成中(最後の出来事から 02:00) 呼び出し 21回(失敗 1・途中切れ 1) 平均 11.4秒
```

- `──` の行は段階(夜・昼の議論・投票…)の切り替わりで、そこまでの**呼び出し回数・失敗・
  途中切れ・平均応答時間**が付きます。途中切れ(`finish_reason=length`。Claude の
  `stop_reason=max_tokens` もこれとして数えます)が増えているなら、思考量を下げるか
  (Claude は `WEREWOLF_ANTHROPIC_EFFORT_*`、luna は `LUNA_REASONING_EFFORT`)、Claude なら
  出力上限 `WEREWOLF_ANTHROPIC_MAX_TOKENS` を上げて再実行する合図です。Actions の AI評価
  では、思考量は入力の `effort_discussion` / `effort_decisions` で、出力上限はリポジトリの
  変数 `WEREWOLF_ANTHROPIC_MAX_TOKENS`(**Settings → Secrets and variables → Actions** の
  Variables)で変えます。
- `[フォールバック]` は、モデルの応答が使えず定型文で代用された発言です。
- 2分間なにも出ないと `… 生成中` の行が出ます。**出続けていれば遅いだけ**、
  **出ていなければ止まっている**と読めます。
- 段階が切り替わるたびに `transcript-seedN.md` / `.json` を「途中経過」として書き出します。
  キャンセルや時間切れで止まったときは、終了の合図を受けた時点でも `── 中断 ──` として
  書き出し、保存の手順(`if: always()`)が Artifacts の `eval-out-seedN` に載せます
  (`report.md` は完走したとき、または費用の上限で止まったときに出ます)。終了処理は
  数秒しか待ってもらえないため、確実に残るのは直前の段階の切り替わりまでです。

## まず接続確認(少ない呼び出しだけ)

いきなり評価を回すと1戦で数百回の呼び出しが発生します。その前に、キー・モデル名・
構造化出力の対応状況を、少ない呼び出しで確認してください。

```bash
cd backend
python scripts/check_llm.py --provider anthropic   # WEREWOLF_LLM_PROVIDER=anthropic でも同じ
```

`anthropic` では、次を順に確かめます。どれか1つでも通らなければ終了コード1です。

1. 届いた設定: プロバイダ、モデル、接続先のホスト名、キーが入っているか。キーは
   **長さだけ**を出し、一部も表示しません(以前の表示方法では、キーの断片が公開の
   Actions ログに残りました)。
2. 通信なしで、ゲームが送るスキーマ(`ChatDiscussionOutput`・`BriefDiscussionOutput`・
   `VoteOutput`・`NightActionOutput`・`WolfChatOutput`)が構造化出力の上限
   (任意パラメータ24・union 16)の内側にあるか。
3. スキーマごとに1回ずつの実呼び出し(合わせて最大5要求)。ゲームと同じプロバイダを
   通し、成否・読めた経路・応答時間・`stop_reason`・実際に答えたモデル・トークン・
   費用を出します。失敗、拒否、strict schema 以外の経路での読み取り、スキーマの400、
   別モデルによるフォールバック応答は、すべて不合格です。再試行が要った呼び出しも、
   要求数の上限で不合格になります(エンドポイントが試合を始められる状態ではないため)。
4. `count_tokens`(生成しない、無料)で、v3 の議論プロンプト1件の入力トークン数を数え、
   1試合の入力トークンの見込みを出します。プロンプトは通信なしの対戦
   (ScenarioProvider)を1日目の議論まで進めて作った本物で、内容は表示しません。

スキーマが通らないと、試合は動いて見えても、CO・結果の公開・質問・吊り先が黙って
簡易契約に落ちます(`docs/approach-reset-2026-10.md` の §11.2)。それを試合の前に
数要求で見つけるための確認です。

luna(`--provider luna`)では、`LUNA_API_KEY` が空、`LUNA_BASE_URL` が
プレースホルダーのまま、URLの誤り、認証エラーが、それぞれ具体的な対処つきで表示されます。
キーは長さだけ、接続先はホスト名だけを出します。

> Codespaces secrets は登録しただけでは反映されません。**Codespace の再起動**
> (または作り直し)が必要です。

## 評価の実行

```bash
cd backend

# 1) ハーネスの動作確認(通信なし・費用ゼロ)
python scripts/evaluate.py --games 2 --provider mock --out eval-out

# 2) 実際の Claude で評価(まず1戦から。--max-cost は必須)
python scripts/evaluate.py --games 1 --provider anthropic --max-cost 15 --out eval-out

# 3) 日本語/人格のLLM判定も行う(追加費用。判定は別モデルで。上限は対戦と共有)
python scripts/evaluate.py --games 1 --provider anthropic --max-cost 20 \
    --judge --judge-model claude-sonnet-5-5 --out eval-out

# 別の選択肢: luna(単価を渡すと金額換算。上限も単価があるときだけ使える)
python scripts/evaluate.py --games 1 --provider luna \
    --price-in 0.15 --price-out 0.60 --max-cost 5 --out eval-out
```

| 引数 | 既定 | 意味 |
|---|---|---|
| `--provider` | 環境変数 `WEREWOLF_LLM_PROVIDER`(なければ `mock`) | `mock` / `scenario` / `luna` / `anthropic` |
| `--games` / `--seed` | `1` / `1` | 対戦数と最初のシード(2戦目以降は1ずつ進む) |
| `--max-cost` | なし | 推定費用の上限(USD)。対戦と判定の全呼び出しで共有。**`anthropic` では必須** |
| `--judge` / `--judge-model` | オフ / 本体と同じ | LLM判定。`--judge-model` は `anthropic` なら `WEREWOLF_ANTHROPIC_MODEL`、`luna` なら `LUNA_MODEL` を判定のときだけ置き換える |
| `--price-in` / `--price-out` | `0` | luna の単価(100万トークンあたり)。`anthropic` は単価表から自動計算 |
| `--out` | `eval-out` | 出力先 |

`--provider anthropic` では **`--max-cost`(USD)が必須**です(`WEREWOLF_LLM_PROVIDER=anthropic`
で選んだときも同じ)。各呼び出しは送る前に最悪の費用を予約し、終わってから実際の費用で
精算するので、同時に走る呼び出しがそろって上限をすり抜けることはありません。上限を
超えそうになった時点で、その呼び出しを送らずに評価を止め、ほかの席の呼び出しも取り消して
精算してから、途中までの対戦記録と、先頭に「費用上限に達したため中断」と書いたレポートを
出し、**終了コード3**で終わります。費用はAPIが返す usage と単価表
(`backend/app/ai/provider/pricing.py`)から自動で計算するので、`--price-in` /
`--price-out` は要りません。luna で `--max-cost` を使うときは、単価が無いと上限が効かない
ので `--price-in` / `--price-out` が必須です。

`scripts/dry_run.py`・`scripts/live_reasoning_check.py`・専門家ベンチの2つのスクリプトも、
`anthropic` では同じく `--max-cost` が必須で、上限で止まると終了コード3です。
`scripts/live_ab_reasoning_check.py` は `--max-estimated-cost` が上限で、再開(`--resume`)
しても使った額を引き継ぎます。`--engines v3` のように1つだけ指定すれば、そのエンジンの
試合だけを回します。

1戦の費用はまだ計測していません。run #11(gpt-6-luna)のトークン量を Opus 5.5 の単価に
当てはめると、キャッシュ無しで約 $7.5 ですが、トークナイザと思考量が違うので±2倍程度の
幅があります。最初の1戦のレポートで実額を確かめてから、上限と対戦数を決めてください。

出力(`eval-out/`):

| ファイル | 内容 |
|---|---|
| `report.md` | 集計レポート |
| `summary.json` | 機械可読な集計(実行間の比較用) |
| `transcript-seedN.md` | 目視確認用の対戦記録(役職・人格・騙り役つき) |
| `transcript-seedN.json` | 生データ |

`report.md` の節:

| 節 | 中身 |
|---|---|
| 先頭 | プロバイダ・モデル・推理エンジン・思考量・コミット・費用の上限と使用額。上限で止まったときは「費用上限に達したため中断」 |
| 1. 動作指標 | JSON成功率、応答時間、所要時間、**1発言あたりの生成時間**(全席AI・ペース配分なしの評価での値。人間が卓で待つ時間ではない)、トークン、推定費用と単価の出どころ |
| 2. 呼び出し種別ごとの健全性 | スキーマごとの回数・成功率・平均応答・effort、コードによる代行(`brief_fallback` などの7項目。0でも行を出す)、拒否の種類、途中切れ、応答したモデルとフォールバックの数、キャッシュ読みの割合、主なエラー、試合ごとの表(シード・勝者・1日あたり発言数・所要・費用・ハード) |
| 3. ハード正しさ(すべて0が合格) | `release_gate` の HARD_FAILURE 項目を試合ごとに数えたもの。v2・v3 だけ(legacy は監査記録を持たないので「計測していません」)。参考として、コードが差し戻した誤りの数も出す |
| 4〜6 | ルールベース検査、発言の形式指標、日本語の自然さ・人格の維持(従来どおり) |

`summary.json` の主なキー: `provider`・`model`・`reasoning_engine`・`commit`
(Actions では `GITHUB_SHA`、手元では `git rev-parse HEAD`)・`budget`(上限・使用額)・
`stopped_by_budget`・`metrics`(全試合の合算)・`per_game`(試合ごとの `seed`・`winner`・
`wall_time_seconds`・`public_utterances`・`days`・`metrics`・`hard_failures`・
`reasoning_quality`)・`judge`。`anthropic` では `settings.anthropic_effort_by_kind` に
呼び出し種別ごとの思考量も残します。

> **費用の注意**: 17人ゲーム1戦で数百回のLLM呼び出しが発生します。
> まず `--games 1` で実行し、レポートのトークン数と費用を見てから増やしてください。

### Claude の設定(思考量・同時実行など)

どれも環境変数で、省略すれば既定値です。Codespaces やローカルでは Codespaces secrets・
`backend/.env` が渡します。下の表のうち Actions の AI評価が渡すのは次の2つだけで、
ほかの設定は既定値で走ります(リポジトリの変数に同じ名前で入れても読まれません)。

- 思考量: 入力の `effort_discussion` が `_DISCUSSION` と `_CHAT` に、`effort_decisions` が
  `_VOTE` と `_NIGHT` に入ります(`_DEFAULT` は既定値のまま)。
- `WEREWOLF_ANTHROPIC_MAX_TOKENS`: リポジトリの変数から。未設定なら `12000`。

Expert cutoff評価はこの表の設定を1つも渡さないので、すべて既定値です。

| 環境変数 | 既定 | 意味 |
|---|---|---|
| `WEREWOLF_ANTHROPIC_EFFORT_DISCUSSION` | `low` | 議論の発言の思考量 |
| `WEREWOLF_ANTHROPIC_EFFORT_VOTE` | `medium` | 投票の思考量 |
| `WEREWOLF_ANTHROPIC_EFFORT_NIGHT` | `medium` | 夜行動の思考量 |
| `WEREWOLF_ANTHROPIC_EFFORT_CHAT` | `low` | 狼・共有チャットの思考量 |
| `WEREWOLF_ANTHROPIC_EFFORT_DEFAULT` | `medium` | それ以外(要約・判定など)の思考量 |
| `WEREWOLF_ANTHROPIC_MAX_TOKENS` | `12000` | 1呼び出しの出力上限の下限。思考も出力に数えます |
| `WEREWOLF_ANTHROPIC_MAX_CONCURRENCY` | `8` | プロセス全体の同時呼び出し数 |
| `WEREWOLF_ANTHROPIC_TIMEOUT_SECONDS` | `120` | 1呼び出しのタイムアウト(秒) |
| `WEREWOLF_ANTHROPIC_MAX_RETRIES` | `2` | 再試行の回数(429 は `retry-after` を守る) |
| `WEREWOLF_ANTHROPIC_FALLBACKS` | `default` | 拒否された手番を別モデルが答え直すサーバー側フォールバック(`default` / `off`) |

思考量は `low` / `medium` / `high` / `xhigh` / `max` から選びます(空は既定値)。Opus 5.5 は
思考を切れないので、待ち時間と費用は主にこの値で調整します。既定値は仮置きで、
応答時間の計測をもとに見直す予定です(`docs/approach-reset-2026-10.md` の §11.2・§11.5)。

フォールバックが有効だと、拒否された手番を `claude-opus-5` などの別モデルが答えます。
実際に答えたモデルは呼び出しごとに記録され、`summary.json` の `metrics` に
`served_model_counts` と `fallback_served_calls` として出ます。基準線として比べる試合では、
この数を確かめてください。

### 専門家ベンチ(Actions の Expert cutoff評価)

**Actions** → **Expert cutoff評価** → **Run workflow**。入力は
`benchmark_version`(`v1` / `v2`)・`provider`(`baseline` / `anthropic` / `luna`)・`seed`・
`scenario_id`(空なら全件)・`max_cost`(USD、既定 `5`)・`price_in` / `price_out`
(luna の単価)。`baseline` は費用ゼロ、`anthropic` は Secret の
`WEREWOLF_ANTHROPIC_API_KEY` と変数の `WEREWOLF_ANTHROPIC_MODEL` を使い、上限は必ず
効きます。luna は単価を入れたときだけ上限が効きます。AI評価と同じ順番待ち
(`ai-evaluation`)に入り、成果物は90日保持です。手元では:

```bash
python scripts/evaluate_expert_scenarios_v2.py --provider anthropic --max-cost 5
```

## 評価項目と測り方

指標は性質が2つに分かれます。**機械的に判定できるもの**はルールベースで数え、
**判断が要るもの**だけをLLMまたは目視に回します。

### 機械的に判定(ルールベース・費用ゼロ・決定的)

| 項目 | 検査内容 |
|---|---|
| JSON成功率 | strict schema / json_object / 寛容パース / 失敗 の内訳。**「成功率」だけ見ると、実は全部フォールバックに救われている状態を見逃す**ため経路別に出す |
| 応答時間 | 呼び出し種別ごとの平均・p50・p95・最大 |
| 実コスト | APIが返す `usage` を集計。Claude は単価表から自動で金額換算(キャッシュの書き込み・読みも別に数える)。luna は単価を渡すと金額換算 |
| 拒否・フォールバック | 拒否の件数(種類別)、実際に答えたモデルの内訳、フォールバックで答えた呼び出し数(`summary.json` の `metrics`) |
| コードによる代行 | 簡易契約への降格、コードが決めた投票・夜行動、コードが出した公開義務の文などを名前つきで数える(`metrics` の `events`)。「動いたように見えて実はコードが代行していた」試合を見分けるため |
| ハード正しさ | v2・v3 の試合で、`release_gate` の HARD_FAILURE 項目(私的な根拠の漏えい、死亡者の指定、公開結果の白黒の言い換え、出すべき結果の欠落など11項目)を、対戦記録の監査記録から数え直す。すべて0が合格。`scripts/live_ab_reasoning_check.py` の判定と同じ計算 |
| 自分の役職との整合性 | 役職に無い占い/霊媒結果の主張、役職と異なるCO(騙り担当は除外) |
| 過去発言との矛盾 | CO役職の変更、死亡者を現在の行動対象にする発言、思考メモと投票先の不一致 |
| 人狼側の欺瞞・連携 | 仲間への明示的な狼視、仲間投票の観測、騙り担当がCOしたか、潜伏担当が黙れたか |
| 自己同一性 | 自分自身を疑い・処刑対象として扱う、p0以外がp0本人を名乗る |
| 形式 | 200文字超過率、フォールバック定型文の発生率、同一発言の繰り返し、日本語比率、メタ発言の漏れ |

これらは `backend/app/eval/analyzers.py` にあり、判定モデルの気分に左右されません。

### LLM判定または目視(`--judge`)

| 項目 | 測り方 |
|---|---|
| 日本語の自然さ | 5段階 |
| 人格の維持 | 指定人格をどれだけ保てたか、5段階 |

**2つの注意点**:

1. **自己採点バイアス** — 既定では発言したモデル自身が採点します。自分の出力には
   甘くなる傾向があるため、可能なら `--judge-model` で別モデル(たとえば
   `claude-sonnet-5-5`)を指定してください。
   絶対値より **プロンプト変更前後の相対比較** に使うのが実用的です。
2. **追加費用** — ゲーム本体とは別に呼び出しが発生します。判定分のコストは
   本体の集計には混ぜていません(本体の応答時間・コストが汚れないようにするため。
   `summary.json` の `judge.metrics` に別に出ます)。ただし `--max-cost` の上限は
   対戦と判定で共有します。上限は実行全体で使う額に対するものだからです。

判定を使わず `transcript-seedN.md` を読むのも有効です。各発言に話者の役職・人格・
騙り役が併記されているので、人格が崩れた箇所を目で追えます。

## 使い方の指針

プロンプトを変更したら、**同じ `--seed` で前後比較**してください。役職配置と人格割当は
シードで再現されるため、変更の効果を分離できます。

- Claude を温度0で動かすことはできません(サンプリングの引数を受け付けないため)。同じシード
  でも試合は毎回少しずつ変わるので、1試合の差で判断せず、異なるシードを複数、比べる条件の
  あいだで対にして使ってください。`--games N` は2戦目以降のシードを1ずつ進め、全体の指標は
  試合をまたいで合算します(試合ごとの値は `summary.json` の `per_game`)。Actions の
  AI評価は `seeds` の1つごとに1ジョブ1試合で回します(`docs/approach-reset-2026-10.md` の §11.5)。
- 基準線や A/B に使った試合は、`report.md`・`summary.json`・対戦記録を
  `backend/artifacts/` にコミットし、コミット SHA・モデル・思考量・シードを添えます
  (どれも `summary.json` に入っています)。Actions の成果物は90日で消えます
  (`docs/approach-reset-2026-10.md` の §11.6)。
- gpt-6-luna の記録(`backend/artifacts/live-v3-luna-run11/`)は文脈として読むためのもので、
  Claude の試合と前後比較しません。

`summary.json` は実行間の差分を取りやすい形にしてあります。

## 既知の限界

- ルールベース検査は表層パターンに基づくため、巧妙な論理矛盾は拾えません。
- 死亡者への霊媒結果、過去発言、過去投票の言及は正常な分析として検出しません。
  現在の処刑・投票・能力・返答対象として扱った場合だけ問題として計上します。
- 判定用プロンプトは推理の当否を評価しません(表現のみ)。強さの評価は勝率で見てください。

## 関連: 応答時間の測定と人間判定

モデル・effort・fast mode ごとの応答時間(最初のトークン・最初の可視テキスト・完了まで)を、記録済みのv3プロンプトで測る `scripts/latency_sweep.py` と、AIの発言を人間が見分けられるかを測る `scripts/human_judgment_quiz.py` の使い方は `docs/measurement-scripts.md` を参照。前者は時間だけを測り、品質は測らない(モデルや effort の変更は、この評価で確かめる)。
