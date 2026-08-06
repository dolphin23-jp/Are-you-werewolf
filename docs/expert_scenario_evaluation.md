# Expert cutoff benchmark

The repository keeps two deterministic closed-set benchmarks over the reviewed real-game cutoffs under `data/expert_scenarios/reviewed/`.

- **v1** preserves the original single-action baseline and historical scores.
- **v2** measures hard-rule application, graded action quality, and a multi-phase day/night plan.

Neither benchmark plays a full game. Each scenario is one frozen decision point, so the current eight scenarios need roughly eight model calls rather than hundreds.

## v1: stable historical baseline

Run with:

```bash
cd backend
python scripts/evaluate_expert_scenarios.py --provider baseline
python scripts/evaluate_expert_scenarios.py --provider luna
```

v1 presents curated facts, shuffled candidate worlds, and one mixed list of opaque candidate actions. It measures possible/impossible classification, belief ordering, exact agreement with one recommended action, catastrophic-action avoidance, and structured reference validity.

Keep v1 results for regression comparison. Its known limitation is that a day execution and a night ability can appear as if they were mutually exclusive, and a reasonable second-best action receives the same action score as a dominated move.

## v2: phase-aware plan benchmark

Run with:

```bash
cd backend

# Free harness smoke test
python scripts/evaluate_expert_scenarios_v2.py \
  --provider baseline \
  --out expert-eval-v2-out

# Current OpenAI-compatible endpoint
WEREWOLF_LLM_PROVIDER=luna \
LUNA_API_KEY=... \
LUNA_BASE_URL=... \
LUNA_MODEL=gpt-5.6-luna \
python scripts/evaluate_expert_scenarios_v2.py \
  --provider luna \
  --out expert-eval-v2-out
```

The reviewed v2 overlay is `data/expert_scenarios/v2_annotations.json`. It does not modify the original scenario files. For each cutoff it adds:

- generic 17A rule IDs used for hard contradiction checks;
- a phase and optional actor for every candidate action;
- expert action ratings: `optimal`, `acceptable`, `dominated`, or `catastrophic`;
- an optimal plan containing complementary day and night actions;
- exact fact/rule IDs supporting every impossible world.

The model receives only the perspective, observed facts, generic rules, unlabeled worlds, and opaque action candidates. It does **not** receive expert weighting, action ratings, loss conditions, the gold plan, later truth, post-game corrections, review status, or provenance.

### v2 answer structure

The model must:

1. judge every candidate world exactly once;
2. cite fact and rule IDs for each impossible judgment;
3. rank possible worlds into main and alternative sets;
4. assess every action as optimal, acceptable, dominated, or catastrophic;
5. choose one action for every `(phase, actor_id)` slot;
6. keep its selected plan consistent with its own action ratings.

This represents plans such as:

```text
Day: execute a cross-world non-LW target
Night: seer A divines the opposing claimant
Night: seer B divines the retained LW candidate
```

rather than forcing those three decisions into one mutually exclusive list.

### v2 scoring

The deterministic scorer reports:

- world-status accuracy and impossible-world recall;
- exact contradiction fact/rule support;
- main/alternative-world overlap;
- action-rating accuracy and ordinal distance;
- exact phase-plan agreement and plan utility;
- catastrophic-action identification and avoidance;
- internal consistency violations;
- unknown world/action/fact/rule references.

A selected `acceptable` action receives partial plan utility. A `dominated` action receives little utility. A `catastrophic` action receives none. This preserves the distinction between “not the expert optimum” and “immediate or forced faction loss.”

## Run one cutoff

Use the same seed for before/after comparison:

```bash
python scripts/evaluate_expert_scenarios_v2.py \
  --provider luna \
  --seed 1 \
  --scenario-id ruru-352698-d6-lw-hold-cross-divination
```

The seed changes candidate order only. It never changes the expert gold answer.

## GitHub Actions

Open **Actions → Expert cutoff評価 → Run workflow** and choose:

- `benchmark_version`: `v1` or `v2`;
- `provider`: `baseline` or `luna`;
- `seed`: normally `1`;
- optional `scenario_id` for one cutoff.

The report appears in the workflow summary and the complete output is uploaded as an artifact.

## Evaluate an external model

Both scripts write spoiler-safe prompts under:

```text
<output>/prompts/<scenario_id>.json
```

Save external answers using the same scenario filenames, then score without another API call:

```bash
python scripts/evaluate_expert_scenarios_v2.py \
  --answers-dir external-answers \
  --out external-v2-score
```

The v2 answer schema is `app.eval.expert_scenarios_v2.ExpertScenarioV2Answer`.

## Output

- `report.md`: aggregate and per-scenario results;
- `summary.json`: machine-readable regression data;
- `prompts/`: exact spoiler-safe inputs;
- `answers/`: validated structured model answers;
- `scores/`: detailed deterministic scores and consistency diagnostics.

`--fail-under` can turn either benchmark into a regression gate after repeated runs establish a stable baseline.

## Remaining limits

Both versions remain curated closed-set benchmarks. They do not yet measure:

- fact extraction from raw chronological chat/events;
- open-ended discovery of missing possible worlds or actions;
- full semantic correctness of free-text rationale;
- semantic perspective leakage outside structured references;
- action regret over a complete game tree.

Those require canonical append-only events, aligned public/full streams, and further independent expert review. v2 improves the integration of hard logic and temporal planning without claiming those missing layers are solved.
