# Transformer historical-training recovery

Historical/population training can run for many learner batches and mutate both the
current Transformer and the immutable policy pool. This document describes the
batch-boundary recovery protocol used by `scripts/train_historical_torch.py`.

The recovery layer is operational only. It does not change `PolicyObservation`,
semantic actions, legal masks, logical discussion timing, or the terminal faction
reward (`+1 / -1 / 0`).

## What is saved

The historical run-state is a self-contained NPZ archive loaded with
`numpy.load(..., allow_pickle=False)`. It stores:

- Transformer architecture metadata and model tensors
- Adam optimizer tensors and parameter-group metadata
- PPO minibatch RNG state
- historical opponent-sampling RNG state
- PPO hyperparameters
- discussion-tick, parallel-game, inference-microbatch, and temperature settings
- the empirical opponent meta-strategy, when one is being used
- completed learner-batch count
- base episode seed and episodes per learner batch
- learner-faction rotation
- parent policy ID and next immutable policy-pool generation

The file is written through a temporary file and atomically replaced at a completed
learner-batch boundary.

## Starting a resumable historical run

```bash
python scripts/train_historical_torch.py \
  --load ./training-runs/current-transformer.npz \
  --pool-dir ./training-runs/torch-pool \
  --output ./training-runs/historical-transformer.npz \
  --run-state ./training-runs/historical.run.npz \
  --batches 1000 \
  --episodes-per-batch 32 \
  --parallel-games 16 \
  --inference-batch-size 64 \
  --team all \
  --meta-strategy ./training-runs/torch-meta.json
```

The initial run-state is written before the first learner batch. This gives the
run a committed recovery point even if the first rollout fails.

## Resuming

```bash
python scripts/train_historical_torch.py \
  --pool-dir ./training-runs/torch-pool \
  --output ./training-runs/historical-transformer.npz \
  --run-state ./training-runs/historical.run.npz \
  --batches 1000 \
  --resume
```

On resume, `--batches` is the total target rather than a number of additional
batches. It may be increased to extend the run, but it cannot be lower than the
already completed batch count.

The saved model, optimizer, RNG streams, learner-faction rotation, seed progress,
and rollout/learner runtime settings are authoritative. A saved meta-strategy is
restored from the run-state itself. If `--meta-strategy` is also supplied during
resume, it must describe the same strategy.

## Crash boundaries

### Crash during rollout or PPO

The current learner batch has not been committed. Resume restores the previous
batch-boundary model, Adam moments, PPO RNG, opponent RNG, and seed position, then
replays that learner batch.

### Crash after writing the output policy but before updating the pool

The output policy file may contain uncommitted newer tensors, but resume ignores it
and restores the committed model from the run-state. The batch is replayed and the
output file is replaced again.

### Crash after adding the policy-pool generation but before committing run-state

The old run-state still points to the generation that the replay is expected to
create. After deterministic replay, the trainer finds that generation already in
the immutable pool and verifies:

- parent policy ID
- specialized learner faction
- model tensors

If all match, the existing generation is reused and progress is committed instead
of creating a duplicate.

The generation written by the interrupted batch is specialized for that batch's
learner faction. Replaying the same batch trains that same faction, so the newly
written specialist is not one of the two opponent factions and does not alter that
batch's fallback historical-opponent candidate set.

### Unexpected pool mutation

Resume accepts the committed `next_pool_generation` boundary, plus the one-step
advanced state corresponding to the crash window described above. A pool that has
advanced further is rejected rather than silently changing the historical game.

## Reproducibility scope

The run-state restores the software-visible state needed for exact continuation:
model parameters, optimizer state, both training RNG streams, rollout batch
configuration, team rotation, and seed ranges.

Tests require an uninterrupted CPU continuation and a save/load continuation to
sample the same opponent policy IDs, produce the same PPO update statistics, and
end with bit-identical model parameters.

Bit-identical floating-point behavior should not be assumed after changing
hardware, PyTorch version, device type, parallel-game count, or inference
microbatch size. In particular, neural batch-shape changes can alter floating-point
rounding near stochastic sampling boundaries. Recovery therefore restores the
saved batching configuration instead of treating parallelism as a freely mutable
resume option.

## Relation to PSRO oracle training

`TorchPSROOracleTrainer` uses `TorchHistoricalTrainingLoop`, so it already shares
the vectorized mixed-model rollout and historical learner semantics. The current
historical run-state protects a single long-running historical learner sequence.
A follow-up orchestration layer can use the same state primitive to resume a
multi-faction Village/Werewolf/Fox oracle expansion at the active oracle and
sub-batch rather than restarting the whole expansion.
