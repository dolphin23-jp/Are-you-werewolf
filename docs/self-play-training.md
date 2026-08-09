# Self-play training architecture

This document describes the reinforcement-learning path for the 17-player game.
The production `GameController` remains the rules authority. Training code lives
under `backend/app/training/` and must not teach human werewolf doctrine through
hand-authored utility bonuses.

## Core invariants

### 1. Policies never receive the true world

A seat may observe only information it can legitimately know:

- public alive/dead state and public death category
- public CO / claim / result / vote / semantic event history
- logical discussion order (`discussion_tick`)
- its own true role
- wolf allies only when the seat is a werewolf
- the other freemason only when the seat is a freemason
- its own seer / medium results
- its own guard targets
- wolf attack-target history only for wolves
- private semantic messages only for their legitimate channel members

The policy observation deliberately excludes:

- other players' true roles
- whether a no-death night was caused by guard or fox
- whether a guard succeeded
- attack success/failure as hidden ground truth
- `is_human`

The value network in the current Phase-1 smoke implementation uses the same
encoded observation. A future centralized critic may be tested separately, but
must never feed privileged information back into the acting policy at runtime.

### 2. Speech order is part of the game

Discussion uses logical timing buckets instead of wall-clock milliseconds:

- `IMMEDIATE`
- `EARLY`
- `NORMAL`
- `LATE`
- `HOLD`

Every alive seat replans after each emitted public turn. A planned claim can
therefore be cancelled or changed after observing another player's CO.
Non-selected speakers still record their timing decision in the trajectory, so
waiting and speaking order can be learned rather than hard-coded.

### 3. Strategy is learned; malformed actions are masked

The semantic layer exposes mechanisms such as:

- role CO and role retraction
- seer / medium result reports and corrections
- partner claims
- remaining-wolf-count claims (including LW claims)
- suspicion/trust evaluations
- execution / divine / guard / CO proposals
- questions, reactions and declarations
- voting
- divine / guard / attack night actions
- wolf/freemason private semantic planning channels

Any role may be claimed publicly, including wolf, madman and fox. False results,
late claims, contradictory stories and self-execution proposals remain possible.
The masks remove only mechanically or semantically impossible combinations.

### 4. Reward is sparse

Current training reward is terminal only:

- winning faction: `+1`
- losing faction: `-1`
- draw: `0`

There is no bonus for CO timing, killing a wolf, successful guard, correct
suspicion, surviving a day, or any other human-authored strategic doctrine.

## Data flow

```text
GameController
    -> PolicyObservation (information-safe per seat)
    -> ObservationEncoder (fixed-shape integer features)
    -> policy/value model
    -> factorized logits
    -> mechanical + semantic masks
    -> structured action
    -> GameController / semantic event stream
```

A trajectory stores the encoded observation at decision time, the sampled head
indices and legal index sets, rollout log probability, value estimate, executed
structured action (when applicable), and final terminal reward.

## Phase-1 NumPy model

`NumpyMLPPolicy` is deliberately small and is not intended to be the final
werewolf AI. It exists to prove that the entire RL pipeline works before adding
a heavier Transformer implementation.

It currently provides:

- one shared role-conditioned policy/value network
- masked structured sampling
- clipped-PPO-style updates
- safe `.npz` checkpoint save/load without pickle objects
- full 17-seat LLM-free self-play
- faction-isolated evaluation
- immutable policy generations for historical self-play
- faction-specialized population entries
- empirical three-faction payoff tables
- payoff-driven population mixtures
- restricted-population deviation diagnostics
- PSRO-style faction-specific oracle expansion

## Install RL development dependencies

From `backend/`:

```bash
pip install -e ".[dev,rl]"
```

## Random environment smoke test

```bash
python scripts/self_play_smoke.py --episodes 20
```

This tests the game/training protocol without a learned model.

## Initial self-play training

```bash
python scripts/train_self_play_numpy.py \
  --episodes 20 \
  --batch-size 2 \
  --pool-dir ./training-runs/pool \
  --output ./training-runs/current.npz
```

Each completed batch can be saved as a general immutable pool generation
(`g000000`, `g000001`, ...). Training can be resumed with `--load`.

## Faction-isolated evaluation

```bash
python scripts/evaluate_policy_numpy.py \
  ./training-runs/current.npz \
  --games 50 \
  --team all
```

By default the fixed opponent is the uniform structured policy. A saved policy
can be supplied with `--opponent`.

Do not interpret the raw faction win distribution from all-current-model
self-play as evidence that the policy improved. Compare candidate generations
against the same fixed opponent and the same seed set, separately as village,
werewolf and fox.

## Historical self-play

After a pool contains at least one generation:

```bash
python scripts/train_historical_numpy.py \
  --load ./training-runs/current.npz \
  --pool-dir ./training-runs/pool \
  --output ./training-runs/current.npz \
  --batches 6 \
  --episodes-per-batch 2 \
  --team all
```

For each batch, one faction is controlled by the current trainable model. The
other factions are controlled by immutable generations sampled from the pool.
Only decisions made by the current model's faction are passed to PPO. With
`--team all`, learner factions rotate village -> werewolf -> fox.

Historical outputs are tagged with `specialized_team`. A village-specialized
checkpoint remains eligible for the village population but is not sampled as a
wolf or fox specialist. General initial self-play checkpoints remain eligible
for all three populations.

## Empirical three-faction meta-game

Measure the Cartesian product of the most recent eligible village, wolf and fox
strategies:

```bash
python scripts/measure_population_payoffs.py \
  --pool-dir ./training-runs/pool \
  --table ./training-runs/payoffs.json \
  --last 3 \
  --games-per-profile 10
```

A profile is `(village_policy, werewolf_policy, fox_policy)`. The table stores
only completed-game terminal outcomes, so the meta-game does not receive shaped
strategic scores either.

Solve a payoff-driven mixture:

```bash
python scripts/solve_population_meta.py \
  --table ./training-runs/payoffs.json \
  --output ./training-runs/meta.json
```

The current solver is an iterated damped logit-response mixture. It is an
intermediate empirical meta-strategy, not a Nash/JPSRO guarantee.

The solver also prints, for each faction:

- expected payoff of the current mixture
- best already-measured policy against the other two mixtures
- restricted-population unilateral deviation gain

`max_restricted_deviation_gain` is useful as a stability signal. It is **not**
full-game exploitability because the deviation search is restricted to policies
already present in the measured population.

Use the solved mixture for historical opponent sampling:

```bash
python scripts/train_historical_numpy.py \
  --load ./training-runs/current.npz \
  --pool-dir ./training-runs/pool \
  --meta-strategy ./training-runs/meta.json \
  --output ./training-runs/current.npz \
  --batches 6 \
  --episodes-per-batch 2 \
  --team all
```

## PSRO-style oracle expansion

Train one new approximate response for each meta-game player:

```bash
python scripts/train_psro_oracles_numpy.py \
  --pool-dir ./training-runs/pool \
  --meta-strategy ./training-runs/meta.json \
  --episodes-per-oracle 20
```

For each faction independently, the oracle:

1. clones that faction's highest-probability current meta policy,
2. plays only that faction against the other two factions' meta mixtures,
3. PPO-updates only the learner faction's decisions,
4. stores a new immutable checkpoint tagged for that faction.

This is an approximate best-response oracle, not an exact best response.
After oracle creation, re-run payoff measurement and meta solving. The practical
population loop is therefore:

```text
initial self-play
  -> measure faction-profile payoffs
  -> solve empirical meta-strategy
  -> inspect restricted deviation gains
  -> train village/wolf/fox response oracles
  -> add faction-specialized policies to pool
  -> measure new profiles
  -> solve again
  -> repeat
```

This is intentionally a PSRO-style scaffold. A later stage can replace the
logit-response solver with a better general-sum multiplayer meta-solver without
changing the game/observation/action interfaces.

## Human compatibility

The training environment does not expose whether a seat is human. The intended
production path is therefore:

```text
AI seat:    PolicyObservation -> learned structured action -> LLM wording
Human seat: human text -> semantic parser -> the same semantic event stream
```

The human and AI should alter the same `GameController` and discussion event
sequence. Self-play uses all AI seats only for speed.

## Next stages

1. Replace the smoke MLP with a player/event Transformer while preserving the same protocol.
2. Replace the heuristic logit-response meta-solver with a stronger multiplayer PSRO/JPSRO-style solver.
3. Add confidence intervals / adaptive sampling to the empirical payoff table.
4. Add private wolf/freemason planning decisions to learned rollouts.
5. Add human-text semantic parsing and final natural-language rendering.
6. Only after the learned path is validated, integrate it as an optional production AI mode.
