# RunPod で pilot-002 を継続する

この手順は、Google Drive 上にある既存の `pilot-002` を RunPod の永続ストレージへ一度だけ移し、population iteration 3–4 と固定評価を既存 checkpoint から exact resume するためのものです。

学習ルール、報酬、観測、action mask、モデル構造、PPO 設定、population 設定は変更しません。

## 推奨構成

- RunPod Secure Cloud
- NVIDIA RTX 3090 x1 を第一候補
  - 在庫がなければ RTX 4090 / RTX A6000 / A40 でも可
- Network Volume: 50 GB から開始
- 公式 PyTorch template
- Network Volume は Pod 作成時に接続する

RunPod の Pod 用 Network Volume は Secure Cloud で利用し、通常 `/workspace` にマウントされます。Network Volume は Pod 本体から独立しているため、Pod を削除しても研究データを残せます。

公式資料:

- https://docs.runpod.io/storage/network-volumes
- https://docs.runpod.io/pods/storage/types
- https://docs.runpod.io/pods/overview

## 1. Network Volume を作る

RunPod console の Storage から Network Volume を作成します。

推奨名:

```text
werewolf-training
```

容量:

```text
50 GB
```

GPU を選ぶ前に volume の datacenter を決めるため、その datacenter で希望 GPU が利用できることを確認してください。

## 2. GPU Pod を作る

Network Volume を選択した状態から Pod を Deploy します。

推奨:

```text
Cloud: Secure Cloud
GPU: RTX 3090 x1
Template: official RunPod PyTorch
Network Volume: werewolf-training
```

JupyterLab または Web Terminal に接続できれば十分です。

RunPod の Network Volume は通常 `/workspace` にマウントされます。

## 3. 既存 pilot-002 を一度だけ転送する

最終的に RunPod 側が次の構造になればよいです。

```text
/workspace/werewolf-training/pilot-002/
  pool/
  population/
  run.npz
  ...
```

最低限、次の2ファイルが存在する必要があります。

```text
/workspace/werewolf-training/pilot-002/pool/manifest.json
/workspace/werewolf-training/pilot-002/population/population.run.json
```

### PC から runpodctl で送る方法

Google Drive for desktop で `pilot-002` が PC に見えている場合が最も簡単です。

Windows 用 `runpodctl` は RunPod 公式 CLI から導入できます。

公式資料:

- https://docs.runpod.io/runpodctl/overview
- https://docs.runpod.io/pods/storage/transfer-files

PC の PowerShell / Terminal で、実際の Drive パスに置き換えて実行します。

```powershell
runpodctl send "G:\My Drive\werewolf-training\pilot-002"
```

表示された one-time code を控えます。

RunPod Web Terminal で:

```bash
mkdir -p /workspace/werewolf-training
cd /workspace/werewolf-training
runpodctl receive <表示されたコード>
```

転送後に確認します。

```bash
ls -l /workspace/werewolf-training/pilot-002/pool/manifest.json
ls -l /workspace/werewolf-training/pilot-002/population/population.run.json
```

Google Drive for desktop を使わない場合は、Drive から `pilot-002` を PC にダウンロードしてから同じ方法で送れます。

## 4. 学習を1コマンドで開始する

RunPod Web Terminal で次を実行します。

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/dolphin23-jp/Are-you-werewolf/main/tools/runpod_pilot002.sh) start
```

このコマンドは自動で以下を行います。

1. 最新 `main` を `/workspace/Are-you-werewolf` に clone/update
2. `/workspace` 上に Python 3.12 の永続 venv を作成
3. RL + Transformer 依存関係をインストール
4. NVIDIA CUDA / PyTorch を検証
5. `pilot-002` の run-state と pool を検証
6. population iteration 2 から iteration 4 へ exact resume
7. iteration 4 完了後、最新3 policy/faction の 27 profiles × 20 games = 540 villages を固定評価
8. meta strategy を再計算
9. ChatGPT に貼る consolidated report を保存

処理は `nohup` でバックグラウンド起動されます。JupyterLab や SSH 接続を閉じても、Pod 自体が RUNNING のままであれば処理は継続します。

## 5. 状態を確認する

```bash
bash /workspace/Are-you-werewolf/tools/runpod_pilot002.sh status
```

状態は概ね次のどれかです。

```text
RUNNING
COMPLETE
FAILED
NOT RUNNING
```

同時に最新80行のログを表示します。

完全ログ:

```text
/workspace/werewolf-training/pilot-002/runpod-iterations-3-4.log
```

## 6. Pod が途中で停止した場合

Network Volume 上の checkpoint は残ります。

Pod を再起動または同じ Network Volume で新しい Pod を作り、同じ start コマンドをもう一度実行します。

```bash
bash /workspace/Are-you-werewolf/tools/runpod_pilot002.sh start
```

既存の population exact-resume と incremental payoff table を使って、不足分から継続します。

手動で checkpoint や `population.run.json` を編集しないでください。

## 7. 完了後

次で report 全体を表示します。

```bash
bash /workspace/Are-you-werewolf/tools/runpod_pilot002.sh report
```

保存場所:

```text
/workspace/werewolf-training/pilot-002/PILOT-002-ITERATIONS-3-4-REPORT.txt
```

`===== PILOT-002 ITERATIONS 3-4 REPORT =====` から最後までを ChatGPT に貼り付けます。

## コスト管理

学習中だけ GPU Pod を RUNNING にします。完了後は Pod を停止または削除して GPU 課金を止め、Network Volume を残します。

RunPod の料金・GPU 在庫は変動するため、実際の Deploy 画面を正としてください。

公式料金:

- https://www.runpod.io/pricing
- https://docs.runpod.io/pods/pricing

Network Volume は長期バックアップ専用サービスではありません。重要な milestone は後で Google Drive / ローカル等にも複製してください。
