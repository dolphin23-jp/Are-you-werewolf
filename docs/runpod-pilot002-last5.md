# RunPod で pilot-002 historical last-5 固定評価を行う

この手順は、population iteration 4 まで完了した `pilot-002` に対して、rolling window の妥当性を調べるための **historical last-5 fixed evaluation** を RunPod 上で行うものです。

学習は進めません。reward、observation、action mask、PPO、population 学習設定も変更しません。

## 評価対象

固定する policy cohort は次の通りです。

```text
Village:  g78, g79, g82, g85, g88
Werewolf: g78, g80, g83, g86, g89
Fox:      g78, g81, g84, g87, g90
```

各陣営5 policy なので、

```text
5 x 5 x 5 = 125 profiles
125 profiles x 20 games/profile = 2500 villages
```

を評価します。

この runner は単に「実行時点の最新5世代」を無条件に使うのではありません。上記 cohort が `TorchPolicyPool.policy_ids_for_team(..., last=5)` と完全一致することを実行前に検証します。iteration 5 以降が追加されて cohort がずれた場合は、別の実験を silently 実行せず停止します。

## 前提

RunPod の Network Volume 上に、iteration 4 まで完了した既存 pilot が存在している必要があります。

```text
/workspace/werewolf-training/pilot-002/
  pool/
    manifest.json
    g000078.npz
    ...
    g000090.npz
  population/
    population.run.json
  ...
```

runner は `completed_iterations >= 4` かつ population runner の `phase == "idle"` も検証します。

## 1コマンドで開始

PR merge 後、RunPod Web Terminal で次を実行します。

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/dolphin23-jp/Are-you-werewolf/main/tools/runpod_pilot002_last5.sh) start
```

自動で以下を行います。

1. 最新 `main` を `/workspace/Are-you-werewolf` に clone/update
2. Python 3.12 venv と RL/Transformer dependency を準備
3. NVIDIA GPU / CUDA / PyTorch を確認
4. iteration 4 完了・idle state を確認
5. historical last-5 cohort と全 checkpoint を厳密検証
6. 125 profiles × 20 games = 2500 villages を評価
7. incomplete な payoff table が既にあれば不足ゲームだけを再開
8. `temperature=0.25`, `iterations=100`, `damping=0.5` で meta strategy を計算
9. 単純勝率、policy別 marginal 勝率、meta weights、restricted deviation diagnostics、meta-mixture outcome rates を consolidated report に出力

評価設定は次の通りです。

```text
games_per_profile = 20
extra_games = 0
seed = 5101
parallel_games = 16
inference_batch_size = 64
device = auto
```

fixed evaluation なので adaptive extra sampling は行いません。

## 状態確認

```bash
bash /workspace/Are-you-werewolf/tools/runpod_pilot002_last5.sh status
```

完全ログ:

```text
/workspace/werewolf-training/pilot-002/runpod-last5-fixed-evaluation.log
```

評価データ:

```text
/workspace/werewolf-training/pilot-002/fixed-evaluation-last5-historical-v1/payoffs.json
/workspace/werewolf-training/pilot-002/fixed-evaluation-last5-historical-v1/meta.json
/workspace/werewolf-training/pilot-002/fixed-evaluation-last5-historical-v1/measure.log
/workspace/werewolf-training/pilot-002/fixed-evaluation-last5-historical-v1/meta.log
```

## Pod が途中停止した場合

Network Volume が残っていれば、同じ `start` をもう一度実行します。

```bash
bash /workspace/Are-you-werewolf/tools/runpod_pilot002_last5.sh start
```

`PopulationPayoffTable` に保存済みの profile/game 数を利用し、20 games/profile に不足している分だけを再実行します。完了済み profile を最初からやり直す必要はありません。

## 完了後

```bash
bash /workspace/Are-you-werewolf/tools/runpod_pilot002_last5.sh report
```

report は次に保存されます。

```text
/workspace/werewolf-training/pilot-002/PILOT-002-LAST5-FIXED-EVALUATION-REPORT.txt
```

この report 全体を次の解析に使います。特に確認する点は、g79 / g81 のような古い policy が、新しい相手を含む5世代cubeでも高い meta weight を維持するかどうかです。

- 古い policy が明確な weight を維持する: `recent_policies=3` は狭すぎる可能性が高い
- 古い policy がほぼ 0 weight になる: last-3 継続にも合理性がある
- 中間的: `recent=5` と価値ベース retention/pruning を比較検討する

この判断が終わるまでは iteration 5 の学習設定を変更しません。training payoff sampling noise の問題も、この window 問題とは分離して扱います。
