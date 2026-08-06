# Expert cutoff benchmark

`backend/scripts/evaluate_expert_scenarios.py` evaluates a model on the reviewed real-game cutoff scenarios under `data/expert_scenarios/reviewed/`.

This is the first measurable bridge between the expert case studies and model improvement. It does not play a full game. Each scenario is one frozen decision point, so the current eight scenarios require eight model calls rather than hundreds.

## What v1 measures

The model receives only:

- the scenario perspective;
- curated observed facts available at the cutoff;
- shuffled, unlabeled candidate worlds;
- shuffled candidate actions with opaque IDs.

It does **not** receive the expert weighting, recommended action, AI correction, review status, provenance, later role table, or full-log postmortem.

The deterministic scorer measures:

- possible-world precision, recall, and F1;
- impossible-world precision, recall, and F1;
- whether every candidate was classified exactly once;
- main/alternative-world overlap with the expert assessment;
- exact action agreement;
- avoidance and identification of explicitly catastrophic actions;
- confidence distance from the expert assessment;
- invalid references to unknown world/action/fact IDs.

The free-text rationale and “next observation” are preserved in the artifact but are not semantically scored yet.

## Run locally

```bash
cd backend

# Free, deterministic harness smoke test
python scripts/evaluate_expert_scenarios.py \
  --provider baseline \
  --out expert-eval-out

# Current OpenAI-compatible endpoint
WEREWOLF_LLM_PROVIDER=luna \
LUNA_API_KEY=... \
LUNA_BASE_URL=... \
LUNA_MODEL=gpt-5.6-luna \
python scripts/evaluate_expert_scenarios.py \
  --provider luna \
  --out expert-eval-out
```

Use the same `--seed` for before/after comparisons. The seed changes only candidate order; it does not change the expert gold answer.

To run one cutoff:

```bash
python scripts/evaluate_expert_scenarios.py \
  --provider luna \
  --scenario-id ruru-352698-d6-lw-hold-cross-divination
```

## Evaluate any external model

The harness always writes spoiler-safe prompt payloads to:

```text
expert-eval-out/prompts/<scenario_id>.json
```

Give those files to any model and save each structured answer as:

```text
external-answers/<scenario_id>.json
```

Then score without another API call:

```bash
python scripts/evaluate_expert_scenarios.py \
  --answers-dir external-answers \
  --out external-score
```

The answer schema is `app.eval.expert_scenarios.ExpertScenarioAnswer`.

## Output

- `report.md`: phone-readable aggregate and per-scenario table;
- `summary.json`: machine-readable results for regression comparison;
- `prompts/`: the exact spoiler-safe inputs;
- `answers/`: validated model answers;
- `scores/`: per-scenario deterministic scoring details.

`--fail-under 0.70` makes the command exit with status 1 when the mean score is below a chosen regression threshold. Do not set a release threshold until a stable baseline has been measured across several runs.

## Important limits

This is a **closed-set, curated-fact benchmark**. It is intentionally smaller than the final target.

It does not yet measure:

- extraction from raw chronological chat/events;
- open-ended possible-world recall;
- semantic quality of the rationale;
- semantic perspective leakage in free text;
- calibrated action regret over a full search tree.

Those require canonical append-only events, aligned public/full streams, and additional expert annotations. Until then, this harness provides a repeatable measurement of hard-world classification, belief ordering, and action selection without pretending to solve the missing layers.
