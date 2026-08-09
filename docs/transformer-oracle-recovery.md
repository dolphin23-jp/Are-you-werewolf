# Transformer PSRO oracle-cycle recovery

The Transformer PSRO command trains one approximate response for each meta-game
faction in the fixed order Village, Werewolf, Fox. A long oracle can now be split
into recoverable learner sub-batches without adding a policy-pool generation after
every PPO update.

This layer is operational only. It does not change game rules, `PolicyObservation`,
semantic actions, legal masks, logical discussion timing, hidden information, or
the sparse terminal faction reward (`+1 / -1 / 0`).

## Semantics

A cycle freezes the supplied `PopulationMetaStrategy` before the first Village
rollout. Each faction then:

1. starts from that faction's dominant policy in the frozen strategy,
2. trains against opponents sampled from the same frozen strategy,
3. may perform several PPO sub-batch updates,
4. writes exactly one immutable specialist after all requested episodes finish.

Therefore a smaller `--oracle-batch-size` changes only the number of PPO updates
inside one response oracle. It does **not** create extra population generations.
Leaving `--oracle-batch-size` unset preserves the previous behavior: one PPO batch
containing all `--episodes-per-oracle` episodes.

New Village specialists created earlier in the same cycle do not become opponents
or parents for the Werewolf/Fox oracle. Parent selection and opponent sampling stay
anchored to the frozen meta-strategy until the whole cycle is complete.

## Starting a recoverable cycle

```bash
python scripts/train_psro_oracles_torch.py \
  --pool-dir ./training-runs/torch-pool \
  --meta-strategy ./training-runs/torch-meta.json \
  --run-state ./training-runs/psro-oracles.run.npz \
  --episodes-per-oracle 256 \
  --oracle-batch-size 32 \
  --parallel-games 16 \
  --inference-batch-size 64
```

The initial run-state is committed before the first rollout. It contains outer
cycle progress plus the full historical learner state for the active faction.

## Resuming

```bash
python scripts/train_psro_oracles_torch.py \
  --pool-dir ./training-runs/torch-pool \
  --run-state ./training-runs/psro-oracles.run.npz \
  --resume
```

The saved model, Adam state, PPO minibatch RNG, historical opponent RNG, frozen
meta-strategy, episode position, faction position, batching settings and pool
boundary are authoritative. Runtime training-shape flags from the original run do
not need to be repeated. If `--meta-strategy` is supplied on resume while a faction
is active, it must equal the strategy embedded in the run-state.

## What the outer state records

The outer NPZ is also loaded with `numpy.load(..., allow_pickle=False)`. It stores:

- faction order and active faction index,
- completed episodes inside the active oracle,
- episodes per oracle and oracle sub-batch size,
- rollout, opponent-sampling and trainer seed bases,
- the next expected immutable pool generation,
- active parent policy ID,
- already completed oracle policy IDs,
- cumulative active-oracle wins/losses/draws, days and decision counts.

When a faction is active, the archive embeds a second safe NPZ produced by the
historical-training recovery layer. That embedded state carries the Transformer
parameters, Adam tensors, both training RNG streams, PPO configuration, rollout
parallelism, inference microbatch limit, and frozen opponent strategy.

A completed cycle contains no active learner payload.

## Crash boundaries

### During rollout or PPO

The current sub-batch has not been committed. Resume restores the previous
sub-batch boundary and deterministically replays that episode slice under the same
software/hardware/batching conditions.

### After a sub-batch state commit

Resume continues with the next episode seed. No population generation has been
created yet unless the faction oracle itself had already finished.

### After the final faction sub-batch but before pool insertion

The state records that all episodes for the active faction are complete. Resume
inserts the expected specialist and advances to the next faction.

### After pool insertion but before advancing the outer state

This is the important two-resource crash window. The old state still contains the
finished learner and expected generation. Resume sees the one-generation-ahead
pool, replays no training, and verifies the existing generation's:

- generation number,
- parent policy ID,
- specialized faction,
- exact model tensors.

Only an exact match is reused. Otherwise recovery fails instead of silently
forking the population. This prevents duplicate oracle generations.

## Reproducibility scope

The recovery state preserves the software-visible state required for exact
continuation. Tests compare uninterrupted and save/load continuations for the next
oracle sub-batch, including opponent IDs, PPO statistics, progress counters and
bit-identical CPU model tensors.

As with historical self-play, bit-identical behavior is not promised after changing
hardware, PyTorch version, device type, parallel-game count or inference batch
shape. Those settings can alter floating-point rounding near stochastic sampling
boundaries, so they are restored from the saved active learner rather than treated
as resume-time tuning knobs.
