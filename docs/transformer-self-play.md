# Transformer self-play path

This document describes the optional PyTorch policy built on top of the
framework-agnostic self-play protocol in `docs/self-play-training.md`.

The rules, information boundaries, semantic action vocabulary, mechanical masks,
and terminal-only faction reward are unchanged. The Transformer is a model
replacement, not a new game or a new strategic reward function.

## Install

From `backend/`:

```bash
pip install -e ".[dev,rl,transformer]"
```

CI installs the PyTorch 2.13 CPU wheel in a dedicated job so the normal backend
job stays lightweight.

## Model structure

The policy receives the existing `EncodedPolicyObservation` as structured tokens:

```text
[CLS]
[global]
[17 player tokens]
[128 semantic-event tokens]
[128 vote-event tokens]
[16 dawn-event tokens]
```

Categorical fields use learned per-field embeddings. Bounded history padding is
masked in self-attention.

The CLS context produces non-pointer heads such as:

- speech timing
- semantic action type
- topic
- role claim
- result value
- quantity / referenced day / scope / stance
- night action topic
- state value

Target-like decisions use pointer heads over encoded tokens:

- `target`, `secondary_target`, `vote_target`, `night_target` -> player tokens
- `reference_event` -> semantic-event tokens

The output is converted back into the same framework-agnostic `PolicyLogits`
contract used by the lightweight NumPy baseline.

## PPO consistency constraints

The current Torch PPO implementation deliberately enforces:

```text
dropout = 0
temperature = 1
```

The rollout trace currently stores sampled indices, legal index sets, rollout
log probability and value estimate, but does not store sampling temperature or
neural dropout state. Allowing either to differ between rollout and update would
silently corrupt the PPO probability ratio, so the trainer rejects those cases.

The strategic reward remains only the final faction result: `+1 / -1 / 0`.

### Advantage estimation

Torch PPO supports per-seat GAE without adding intermediate rewards. Between a
seat's own recorded decisions the reward is zero; its faction payoff is attached
only after its final recorded decision.

The defaults are deliberately backward compatible:

```text
gamma = 1.0
gae_lambda = 1.0
normalize_advantages = false
```

With those values, GAE telescopes exactly to the previous Monte-Carlo target:
every value target is the final faction payoff and every advantage is
`terminal_payoff - rollout_value`.

For experiments, use for example:

```bash
--gamma 0.99 --gae-lambda 0.95 --normalize-advantages
```

Changing these parameters changes credit assignment, not the game reward. No
CO timing, kill, guard, role result, survival, or other strategic event receives
an authored bonus.

## Batched rollout

`TorchBatchedEpisodeRunner` always creates each seat's information-safe
observation independently. It then groups seats only when they use the exact same
Transformer instance and performs one batched forward for that group. Each seat
keeps its own `MaskedPolicySampler` and RNG stream.

This gives two useful cases without changing game semantics:

- all 17 seats share one policy -> normally one Transformer forward per decision point
- Village / Werewolf / Fox use different historical policies -> normally at most three forwards per decision point

A regression test runs the same model and seed through the generic sequential
runner and the batched runner and checks that winner, day count, semantic event
count, and the complete structured action sequence are identical. Batching is
therefore a neural-compute optimization, not an information-sharing mechanism.

### Cross-game vectorized collection

`TorchVectorizedEpisodeCollector` advances multiple independent games together.
At each logical phase it gathers every currently eligible seat observation
across those games. The environments never share state; only neural inference is
shared.

`TorchSelfPlayTrainingLoop` uses this collector directly. Two independent knobs
control rollout concurrency:

- `--parallel-games` controls how many independent game environments advance together.
- `--inference-batch-size` optionally caps how many seat observations enter one Transformer forward.

Without `--inference-batch-size`, the collector keeps the previous unbounded
behavior for backward compatibility. With a limit, one logical decision point
may be split into several inference microbatches while preserving request order,
seat-specific observations, sampler RNG streams, masks, and structured actions.
This is a memory/throughput control only; it does not become a policy input.

For example, 32 simultaneous games can produce far more than 32 eligible seat
observations at a phase boundary. A configuration such as:

```bash
--parallel-games 32 --inference-batch-size 64
```

still advances 32 games together but limits each Transformer forward to at most
64 observations. The correct value is hardware/model dependent and should be
chosen from measured utilization and memory use rather than treated as a game
hyperparameter.

Rollout output conversion is also batched: each Tensor policy head is transferred
from the accelerator to the framework-agnostic CPU `PolicyLogits` representation
once per inference microbatch instead of once per seat. Scalar and batched
adapters are regression-tested for exact equality.

## Checkpoints and long-run recovery

### Policy checkpoints

Transformer policy checkpoints are `.npz` archives containing:

- JSON architecture/head metadata encoded as `uint8`
- raw `state_dict` tensors as arrays

Loading uses `numpy.load(..., allow_pickle=False)`. `torch.save` object-pickle
loading is not required.

### Run-state snapshots

Long self-play runs additionally keep a self-contained run-state NPZ at every
completed batch boundary. The file stores:

- model tensors
- Adam optimizer tensors and parameter-group metadata
- PPO minibatch RNG state
- PPO configuration
- logical runtime settings such as discussion ticks, parallel-game count, and inference microbatch limit
- completed episode count and base seed
- immutable population lineage / next generation when a policy pool is enabled

The archive is written atomically and loaded with `allow_pickle=False`.
Therefore a crash can resume from the last committed batch without resetting
Adam moments, changing the minibatch permutation stream, or consuming a new seed
range.

The default run-state path is derived from `--output`. For example:

```text
current-transformer.npz
current-transformer.npz.run.npz
```

Resume with the same total episode target or a larger one:

```bash
python scripts/train_self_play_torch.py \
  --episodes 1000000 \
  --output ./training-runs/current-transformer.npz \
  --run-state ./training-runs/current-transformer.run.npz \
  --pool-dir ./training-runs/torch-pool \
  --metrics-jsonl ./training-runs/train.jsonl \
  --resume
```

On resume, the run-state's training hyperparameters, base seed, optimizer state,
parallel-game count, and inference microbatch limit are authoritative.
`--episodes` remains the total target, so it may be increased to extend a run.
If `--inference-batch-size` is explicitly supplied on resume, it must match the
saved value; changing batching in the middle of an exact reproducibility run is
rejected.

When a policy pool is enabled, the run state also records the intended next
immutable generation. If a crash happens after that generation was added but
before the run-state commit, deterministic replay recognizes the existing
entry, verifies that its tensors match, and continues instead of creating a
second generation.

## Training metrics

`train_self_play_torch.py` reports, and can append as JSONL with
`--metrics-jsonl`:

- rollout seconds
- rollout episodes / second
- rollout decisions / second
- inference forward-call count
- inference observation count
- mean and maximum actual inference batch size
- maximum pending observations before microbatch splitting
- learner seconds
- learner decisions / second
- faction wins and draws
- mean game days and decisions
- PPO policy/value loss, ratio, clipping fraction, and gradient norm
- sampled-policy approximate KL
- mean factorized action-path entropy
- rollout value explained variance against GAE value targets

The additional PPO diagnostics are observational only:

- approximate KL estimates how far the updated sampled-action probabilities have moved from their rollout probabilities; it does not currently trigger a KL penalty, reward, or automatic early stop
- path entropy sums the conditional entropy of the legal factorized heads actually traversed by timing, speech, vote, and night decisions; there is no entropy bonus in the loss or reward
- rollout value explained variance measures how much of the GAE value-target variation was explained by the value estimates frozen into rollout traces; `1` is ideal, `0` means no variance explained, and negative values are possible when the critic is worse than a constant predictor

These are runtime/research measurements only. Wall-clock timing, batching
statistics, KL, entropy, and value diagnostics are never placed inside
`PolicyObservation`, never alter action masks, and never become reward terms.

## Initial self-play

```bash
python scripts/train_self_play_torch.py \
  --episodes 20000 \
  --batch-size 256 \
  --parallel-games 32 \
  --inference-batch-size 64 \
  --discussion-ticks 8 \
  --pool-dir ./training-runs/torch-pool \
  --metrics-jsonl ./training-runs/train.jsonl \
  --run-state ./training-runs/current-transformer.run.npz \
  --output ./training-runs/current-transformer.npz
```

The default model is:

```text
d_model=96
heads=4
layers=3
feedforward=384
dropout=0
```

`--device auto` uses CUDA when available, otherwise CPU.

The example sizes above are starting points for throughput measurement, not
recommended final hyperparameters. Tune `--parallel-games`,
`--inference-batch-size`, `--batch-size`, and PPO minibatch size from observed
GPU/CPU utilization, inference batch metrics, and memory use. Track KL, entropy,
and explained variance to detect unstable policy updates, premature collapse,
or a critic that is not learning before scaling a run further.

Each completed learner batch can also be stored as a general immutable population
generation.

## Faction evaluation

```bash
python scripts/evaluate_policy_torch.py \
  ./training-runs/current-transformer.npz \
  --games 50 \
  --team all
```

Use `--opponent` to supply another saved Transformer. Without it, evaluation uses
the framework-agnostic uniform structured policy.

## Historical self-play

```bash
python scripts/train_historical_torch.py \
  --load ./training-runs/current-transformer.npz \
  --pool-dir ./training-runs/torch-pool \
  --output ./training-runs/current-transformer.npz \
  --batches 6 \
  --episodes-per-batch 2 \
  --team all
```

The learner faction rotates village -> werewolf -> fox when `--team all` is
used. Only that faction's decisions are sent to Torch PPO. Saved outputs are
tagged as faction specialists.

A solved opponent mixture can be supplied with:

```bash
--meta-strategy ./training-runs/torch-meta.json
```

## Measure the empirical three-faction game

```bash
python scripts/measure_population_payoffs_torch.py \
  --pool-dir ./training-runs/torch-pool \
  --table ./training-runs/torch-payoffs.json \
  --last 3 \
  --games-per-profile 10 \
  --extra-games 20
```

The shared payoff table records terminal outcomes for
`(village_policy, werewolf_policy, fox_policy)` profiles. Extra games are spent
on the currently most uncertain selected profiles.

## Solve the population mixture

```bash
python scripts/solve_population_meta_torch.py \
  --pool-dir ./training-runs/torch-pool \
  --table ./training-runs/torch-payoffs.json \
  --last 3 \
  --output ./training-runs/torch-meta.json
```

The current solver is still the empirical damped logit-response solver from the
foundation layer. It reports restricted-population unilateral deviation gains;
those are stability diagnostics inside the measured pool, not full-game
exploitability guarantees.

## Add response oracles

```bash
python scripts/train_psro_oracles_torch.py \
  --pool-dir ./training-runs/torch-pool \
  --meta-strategy ./training-runs/torch-meta.json \
  --episodes-per-oracle 20
```

One approximate response policy is independently trained and stored for Village,
Werewolf and Fox. Re-measure the new profiles, solve the meta-strategy again,
and repeat.

## Current complete loop

```text
vectorized Transformer self-play
  -> bounded inference microbatches if configured
  -> PPO / optional GAE learner update
  -> policy/value health diagnostics (observational only)
  -> atomic policy + resumable run-state snapshots
  -> save general generations
  -> historical self-play / faction specialists
  -> measure 3-faction payoff profiles
  -> allocate extra evaluation to uncertain profiles
  -> solve empirical meta-strategy
  -> inspect restricted deviation gains
  -> train Village/Werewolf/Fox response oracles
  -> add specialists to the pool
  -> measure new profiles
  -> solve again
  -> repeat
```

## Next performance/research steps

1. Benchmark CPU/GPU saturation across larger independent-game batches and tune observation-to-tensor staging.
2. Reduce Python-side rollout/sampling overhead after profiling identifies the dominant path.
3. Improve PPO learner data staging and minibatch execution for very large batches.
4. Improve long semantic-history representation and learned memory/compression.
5. Replace the heuristic meta-solver with a stronger multiplayer PSRO/JPSRO-style solver.
6. Add learned private wolf/freemason planning turns.
7. Build post-training strategy analysis over structured logs without feeding those concepts into reward.
8. Integrate semantic parsing and LLM surface realization only after strategic training is validated.
