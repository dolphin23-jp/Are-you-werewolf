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

Each completed batch can be saved as an immutable pool generation (`g000000`,
`g000001`, ...). Training can be resumed with `--load`.

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

This is the first defense against latest-vs-latest strategy cycling. Uniform
pool sampling is only a baseline; a later population/meta-strategy stage should
replace it with measured matchup payoffs and PSRO/JPSRO-style opponent mixtures.

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

1. Add reproducible matchup matrices for policy-pool generations.
2. Derive opponent mixtures from measured payoff data rather than uniform pool sampling.
3. Replace the smoke MLP with a player/event Transformer while preserving the same protocol.
4. Add private wolf/freemason planning decisions to learned rollouts.
5. Add human-text semantic parsing and final natural-language rendering.
6. Only after the learned path is validated, integrate it as an optional production AI mode.
