# Colab strategy observatory

This workflow separates self-play learning from post-hoc strategy inspection.
RunPod remains the authoritative training process. Colab receives only immutable
policy checkpoints and runs ordinary games for analysis. Observatory output must
never be fed back into rewards, action masks, retention, oracle-parent selection,
or any other learning decision.

## What is recorded

For each observed game the JSONL output contains:

- the frozen Village / Werewolf / Fox policy profile;
- post-hoc role assignment and terminal result;
- every public semantic event that was actually emitted;
- every vote and night action;
- for every living seat at every discussion tick, the information-safe
  `PolicyObservation`, sampled timing, sampled semantic bundle, policy sampling
  trace, and whether that seat was selected to speak.

The last item is intentionally richer than a normal transcript. A sampled bundle
from a non-selected seat is an **intent**, not an executed statement. Only the
semantic-event stream represents committed public speech.

The post-hoc role assignment is analysis metadata. It is never inserted into a
policy observation.

## 1. Export a frozen snapshot on RunPod

The exporter reads `population.run.json`, the pool manifest, and immutable NPZ
checkpoint files. It does not import PyTorch, allocate CUDA memory, mutate the
pool, mutate the run state, or write payoff data. It is therefore suitable for
exporting an already frozen active population while a later training phase is
running.

From `backend/`:

```bash
PYTHONPATH=. python scripts/export_strategy_snapshot.py \
  --pool-dir /workspace/werewolf-training/pilot-002/pool \
  --run-dir /workspace/werewolf-training/pilot-002/population \
  --output /workspace/werewolf-training/pilot-002/strategy-iteration-0012-active.tar.gz \
  --label pilot-002-iteration-12-active
```

When the population run is in `measure` or `oracle`, the exporter uses the exact
frozen faction policy IDs stored in the active run state. When it is `idle`, it
uses the latest completed iteration. Pass `--iteration N` to export a specific
completed iteration instead.

The archive contains:

```text
snapshot.json
pool/manifest.json
pool/gXXXXXX.npz
...
```

`snapshot.json` records the source iteration, run phase, Git commit, population,
and a SHA-256 digest for every checkpoint. Extraction verifies those hashes.

Copy the resulting `.tar.gz` to Google Drive or upload it to the Colab runtime.
This transfer is one-way with respect to learning: Colab output is analysis-only.

## 2. Prepare Colab

A typical Colab setup is:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!git clone https://github.com/dolphin23-jp/Are-you-werewolf.git /content/Are-you-werewolf
%cd /content/Are-you-werewolf/backend
!python -m pip install -q -e ".[rl]"
```

Colab normally already provides PyTorch. Installing only `[rl]` avoids replacing
that CUDA build. Verify it before observation:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

For exact replay, the repository checkout should match the `git_commit` stored in
`snapshot.json`. `observe_strategy_torch.py` refuses a mismatch by default.
`--allow-code-mismatch` exists only for deliberate compatibility experiments.

## 3. Observe a fixed profile

For an initial Village `g000079` inspection, choose frozen opponent policies from
the same snapshot and run, for example:

```bash
!PYTHONPATH=. python scripts/observe_strategy_torch.py \
  --snapshot /content/drive/MyDrive/werewolf/strategy-iteration-0012-active.tar.gz \
  --village-policy g000079 \
  --werewolf-policy g000104 \
  --fox-policy g000093 \
  --games 20 \
  --seed-start 120000 \
  --max-discussion-ticks 8 \
  --device auto \
  --output-dir /content/drive/MyDrive/werewolf/observatory/g79-vs-g104-g93
```

The profile semantics match empirical population evaluation: the selected
Village policy controls all Village-team seats, the Werewolf policy controls all
Werewolf-team seats, and the Fox policy controls the Fox seat. Role assignment
still varies with the game seed.

Outputs are:

```text
games.jsonl
transcripts/game-120000.txt
transcripts/game-120001.txt
...
```

`games.jsonl` is the authoritative analysis record. The text transcripts are a
compact human-readable view of executed speech, votes, night actions, and timing
intents.

## Research boundary

The observatory is deliberately downstream of learning. Do not use discovered
human-readable tactics to alter training reward, legal masks, retention rules,
policy selection, or oracle initialization. Human strategy labels and comparisons
belong only in later post-hoc analysis.

A future counterfactual-probe layer can branch an information-safe observation
and change one public event before querying the same frozen policy. That should
remain in this analysis boundary rather than the self-play learner.
