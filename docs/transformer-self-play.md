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

## Safe checkpoints

Transformer checkpoints are `.npz` archives containing:

- JSON architecture/head metadata encoded as `uint8`
- raw `state_dict` tensors as arrays

Loading uses `numpy.load(..., allow_pickle=False)`. `torch.save` object-pickle
loading is not required.

## Initial self-play

```bash
python scripts/train_self_play_torch.py \
  --episodes 20 \
  --batch-size 2 \
  --discussion-ticks 8 \
  --pool-dir ./training-runs/torch-pool \
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

Each batch can be stored as a general immutable population generation.

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
Transformer initial self-play
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

1. Vectorize multiple independent games per GPU batch.
2. Add a better advantage estimator after the baseline is empirically stable.
3. Replace the heuristic meta-solver with a stronger multiplayer PSRO/JPSRO-style solver.
4. Add learned private wolf/freemason planning turns.
5. Integrate semantic parsing and LLM surface realization only after strategic training is validated.
