# Vectorized Transformer population payoff evaluation

Population payoff measurement evaluates immutable
`(Village policy, Werewolf policy, Fox policy)` profiles using only completed game
outcomes. It is an evaluation/runtime layer: game rules, hidden information,
`PolicyObservation`, semantic actions, legal masks, discussion timing, and the
terminal faction reward remain unchanged.

## Why this path is vectorized

Self-play and historical training already advance independent games together, but
population measurement originally evaluated one game at a time and loaded three
Transformer checkpoints again for every profile. That becomes expensive once a
PSRO population contains many profiles and each profile needs repeated games.

Transformer payoff measurement now uses `TorchVectorizedEpisodeCollector` across
profiles as well as across repeated games of one profile.

For each active rollout chunk:

1. take up to `--parallel-games` pending profile games,
2. load each unique policy ID needed by that chunk once,
3. reuse the same model instance wherever that policy appears,
4. assign each game its own Village/Werewolf/Fox model mapping,
5. group inference requests only by exact model identity,
6. restore logits to original request order before seat-local sampling and actions,
7. record all completed outcomes for the chunk with one atomic payoff-table write,
8. release the chunk model cache before loading the next chunk.

The model cache is therefore bounded by active rollout concurrency rather than the
size of the whole measured population cube.

## CLI

```bash
python scripts/measure_population_payoffs_torch.py \
  --pool-dir ./training-runs/torch-pool \
  --table ./training-runs/torch-payoffs.json \
  --last 3 \
  --games-per-profile 10 \
  --extra-games 20 \
  --parallel-games 16 \
  --inference-batch-size 64
```

`--parallel-games` controls how many independent profile games advance together.
`--inference-batch-size` optionally caps the number of observations passed through
one exact-model Transformer forward. These values affect runtime batching only;
they are not visible to a policy.

The initial missing payoff cube is scheduled together across profiles. This is the
large, naturally parallel part of empirical evaluation.

`--extra-games` retains the previous adaptive uncertainty semantics. Each extra
game is committed before the next highest-uncertainty profile is selected, so
vectorization does not silently change the uncertainty-allocation algorithm.

## Duplicate seeds

The underlying vectorized collector requires unique game seeds within one collector
call. Population profiles derive their seed ranges independently, so a rare seed
collision between different profiles is possible.

The bulk evaluator does not rewrite such a seed. Instead it schedules colliding
seeds in different rollout chunks. Both profiles therefore keep exactly the seed
they were assigned while satisfying the collector's per-call uniqueness rule.

## Payoff persistence

`PopulationPayoffTable.record_result(...)` remains available for existing NumPy and
single-result callers. Internally, the table also supports a validated batch of
terminal outcomes.

All outcomes in a batch are validated before any in-memory mutation. A valid chunk
is then aggregated and the JSON table is atomically rewritten once, rather than
once per game. A crash can therefore lose at most the currently uncommitted rollout
chunk; rerunning the measurement command recomputes the still-missing games from
the persisted game counts.

No intermediate reward or strategic diagnostic is written into the payoff table.
It continues to contain only terminal winner/draw counts and game-day aggregates.

## Runtime metrics

The measurement command reports:

- newly measured games,
- rollout chunks,
- evaluation seconds and games/second,
- Transformer checkpoint loads,
- mean and maximum inference batch,
- maximum pending inference requests,
- configured parallel-game count and inference cap.

These metrics are observational only and never enter policy input or reward.

## Reproducibility scope

Game environments remain independent and seat-specific RNG streams remain local to
each game. Cross-game batching shares only neural inference.

As with historical vectorized rollout, changing parallel-game count or inference
microbatch shape can change floating-point reduction/order details enough to move a
stochastic sample near a probability boundary. The evaluator therefore does not
promise bit-identical outcomes across different batching configurations. Tests
instead verify model routing, profile accounting, checkpoint reuse, seed-collision
handling, and inference caps.
