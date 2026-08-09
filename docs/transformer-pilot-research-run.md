# Transformer pilot research run

This runbook is the transition from infrastructure testing to repeated self-play
research. It uses the production `GameController` through the existing training
environment, structured semantic actions only, and sparse terminal faction reward
`+1 / -1 / 0`. No LLM is present in the rollout loop and no human Werewolf tactic
is added as a reward, utility, mask, or timing rule.

The first goal is not maximum strength. The first goal is to prove that a few
hundred to a few thousand complete 17-player villages can be trained, evaluated,
interrupted, resumed, and expanded through the empirical population loop without
silent corruption or pathological runtime behavior.

## Directory layout

Use separate bootstrap and population state while sharing one immutable policy
pool:

```text
training-runs/pilot-001/
  pool/
    manifest.json
    g000000.npz
    ...
  bootstrap/
    model.npz
    run.npz
    metrics.jsonl
  population/
    population.run.json
    payoffs.json
    iteration-0001/
      meta.json
      oracle.run.npz
      summary.json
    iteration-0002/
      ...
```

The pool is the lineage shared between initial general self-play and later
faction-specialized response oracles. The bootstrap run-state and population
run-state are intentionally separate because they have different recovery
boundaries.

## Stage 0: short throughput and recovery smoke run

Run from `backend/` on the machine that will do the actual training:

```bash
python scripts/train_self_play_torch.py \
  --episodes 128 \
  --batch-size 32 \
  --parallel-games 8 \
  --inference-batch-size 64 \
  --seed 1001 \
  --pool-dir ../training-runs/pilot-001/pool \
  --output ../training-runs/pilot-001/bootstrap/model.npz \
  --run-state ../training-runs/pilot-001/bootstrap/run.npz \
  --metrics-jsonl ../training-runs/pilot-001/bootstrap/metrics.jsonl \
  --device auto
```

This produces four general pool snapshots, one after each 32-game PPO batch. The
exact values of `--parallel-games` and `--inference-batch-size` are runtime tuning
parameters, not policy inputs. Start conservatively; increase them only if device
memory and throughput justify it.

A deliberate stop can be resumed with the same pool and saved run-state:

```bash
python scripts/train_self_play_torch.py \
  --episodes 128 \
  --pool-dir ../training-runs/pilot-001/pool \
  --output ../training-runs/pilot-001/bootstrap/model.npz \
  --run-state ../training-runs/pilot-001/bootstrap/run.npz \
  --metrics-jsonl ../training-runs/pilot-001/bootstrap/metrics.jsonl \
  --resume \
  --device auto
```

The saved batching configuration, model, Adam state, PPO minibatch RNG, progress,
and pool lineage are authoritative on exact resume.

## Stage 1: one small empirical population iteration

After the smoke bootstrap succeeds, exercise the complete payoff → meta-strategy →
Village/Werewolf/Fox oracle path:

```bash
python scripts/run_population_iterations_torch.py \
  --pool-dir ../training-runs/pilot-001/pool \
  --run-dir ../training-runs/pilot-001/population \
  --iterations 1 \
  --recent-policies 3 \
  --games-per-profile 3 \
  --extra-games 8 \
  --oracle-episodes 64 \
  --oracle-batch-size 16 \
  --parallel-games 8 \
  --inference-batch-size 64 \
  --evaluation-seed 1101 \
  --oracle-seed 1201 \
  --opponent-seed 1301 \
  --device auto
```

The population command freezes the eligible recent policy IDs before measuring an
iteration. A Village specialist written later in that same iteration therefore
cannot change the Werewolf/Fox restricted game, parent choice, or opponent mixture.
The solved meta-strategy remains fixed throughout the three-faction oracle cycle.

If interrupted:

```bash
python scripts/run_population_iterations_torch.py \
  --pool-dir ../training-runs/pilot-001/pool \
  --run-dir ../training-runs/pilot-001/population \
  --iterations 1 \
  --resume \
  --device auto
```

On resume, all research settings stored in `population.run.json` are authoritative.
Only the total iteration target and device selection need to be supplied again.

## Stage 2: first real pilot, roughly a few thousand villages

Once Stage 0 and Stage 1 are stable, a useful first optimization run is on the
order of thousands rather than millions of games. For example:

```bash
python scripts/train_self_play_torch.py \
  --episodes 1024 \
  --batch-size 64 \
  --parallel-games 16 \
  --inference-batch-size 64 \
  --seed 2001 \
  --pool-dir ../training-runs/pilot-002/pool \
  --output ../training-runs/pilot-002/bootstrap/model.npz \
  --run-state ../training-runs/pilot-002/bootstrap/run.npz \
  --metrics-jsonl ../training-runs/pilot-002/bootstrap/metrics.jsonl \
  --device auto
```

Then:

```bash
python scripts/run_population_iterations_torch.py \
  --pool-dir ../training-runs/pilot-002/pool \
  --run-dir ../training-runs/pilot-002/population \
  --iterations 2 \
  --recent-policies 3 \
  --games-per-profile 5 \
  --extra-games 32 \
  --oracle-episodes 256 \
  --oracle-batch-size 32 \
  --parallel-games 16 \
  --inference-batch-size 64 \
  --evaluation-seed 2101 \
  --oracle-seed 2201 \
  --opponent-seed 2301 \
  --device auto
```

With three recent eligible policies per faction, one full restricted cube has up
to 27 profiles. Five base games per profile plus 32 adaptive games and three
256-game response oracles is roughly 935 new games for an iteration before reuse
of already measured profiles is considered. Two iterations after a 1024-game
bootstrap therefore lands naturally in the low-thousands regime.

This is intentionally a pilot scale. It is large enough to expose optimization,
throughput, population, and strategy-diversity problems before spending the much
larger compute budget required for 100k+ games.

## What to watch before scaling

The bootstrap JSONL and population console/summary output already expose runtime
and learning-health signals. Before increasing the game count substantially,
check that:

- rollout throughput and inference batch sizes are stable rather than degrading
  over time,
- no NaN/Inf appears in policy loss, value loss, gradient norm, KL, entropy, or
  value explained variance,
- PPO approximate KL does not repeatedly jump by orders of magnitude from its own
  recent baseline,
- action-path entropy does not immediately collapse toward a near-deterministic
  policy across the whole run,
- value explained variance is at least capable of improving from an initially
  weak baseline rather than remaining structurally invalid,
- all three faction outcomes and draws are recorded from the actual terminal game
  result without auxiliary reward terms,
- payoff-table game counts grow as expected and restricted deviation diagnostics
  can be solved from a complete cube,
- stop/resume reproduces the next saved boundary under the same hardware/software
  and batching configuration.

None of these checks specifies what a Seer, Medium, Villager, Werewolf, Hunter, or
Fox should strategically do. They are optimizer/runtime integrity checks only.

## When to move past the pilot

Do not jump from a 128-game smoke test directly to a million games. A useful
progression is:

1. 128–500 games: throughput, memory, crash/recovery and numerical sanity,
2. 1k–5k games: first evidence that policy/population behavior changes under
   self-play rather than merely executing the rules,
3. 10k–50k games: compare generations and population mixtures with enough samples
   to see whether new response policies add measurable value,
4. 100k+ games: only after the preceding runs show stable learning and evaluation.

The exact thresholds are research-budget choices, not game-strategy rules.

## Reproducibility boundary

Exact recovery assumes the same device/software/batching shape saved for the
active learner. Changing GPU model, PyTorch version, parallel-game count, or
inference microbatch shape can alter floating-point rounding enough to cross a
stochastic sampling boundary. Such changes are appropriate between research runs,
not during an exact-resume run.
