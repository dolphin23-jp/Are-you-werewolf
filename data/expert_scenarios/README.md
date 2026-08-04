# Expert reasoning scenarios

This directory stores cutoff-local reasoning examples created while an expert and an AI read real game logs together.

The contract is intentionally stricter than a prose answer. A scenario must keep five layers separate:

1. observed public and actor-private facts;
2. logically possible and impossible worlds;
3. conditional deductions under explicit assumptions;
4. soft weighting among the remaining worlds;
5. the faction-aware action recommendation.

The canonical schema is `schemas/expert_scenario.schema.json`.

## Directory layout

```text
data/expert_scenarios/
  README.md
  examples/
    shared-claim-contradiction.json
  reviewed/
    <log_id>/
      <scenario_id>.json
```

Only files under `reviewed/` may become training or regression data. Example and draft files are never training-ready merely because they validate against the JSON Schema.

## Joint-reading workflow

For each important cutoff:

1. Freeze the event stream at `cutoff_event_id`.
2. Reconstruct the chosen perspective. Do not include later deaths, later verdicts, true roles, or private channels unavailable to that actor.
3. Record public and actor-private facts with source event IDs.
4. Enumerate representative possible worlds and explicitly impossible worlds.
5. Record conditional deductions separately from unconditional facts.
6. Ask the expert to rank worlds qualitatively and explain the weighting factors.
7. Record the recommended action separately from role suspicion.
8. If the AI was corrected, save the initial answer, correction, error categories, generalized prevention rule, and rule counterexamples.
9. Mark the scenario training-ready only after expert review and all blocking issues are cleared.

## Review states

- `draft`: AI-produced or incomplete; never train on it.
- `expert_reviewed`: one expert checked the content, but disagreement or data-quality questions may remain.
- `adjudicated`: disagreements were resolved or intentionally represented as alternatives.
- `rejected`: unsuitable because of source loss, perspective leakage, unresolved ambiguity, or incorrect reconstruction.

`review.training_ready` is an independent gate. A scenario may be expert-reviewed but still not training-ready.

## Hard rules for authors

- Do not convert an unlikely world into an impossible world.
- Do not use true roles or later events in a player-perspective scenario.
- Do not store a public claim as a true ability result without a system-event source.
- Do not count a repeated statement as fresh evidence.
- Give related soft evidence the same `causal_group_id` so later scoring cannot double-count it.
- Keep the expert's main line and plausible alternatives. Soft judgment is not required to have one universal answer.
- Use `true_world_evaluation` only for evaluation and debugging, never as a gameplay reasoning input.

## Training-ready checklist

A scenario is training-ready only when:

- source event IDs resolve against a training-ready parsed game;
- the cutoff and perspective can be reconstructed deterministically;
- public/private fact separation has been checked;
- impossible worlds have an explicit contradiction or unsat core;
- soft factors cite their source events and causal groups;
- expert alternatives and expected disagreement are represented;
- AI corrections include at least one prevention rule and any known counterexample;
- `review.blocking_issues` is empty.

The game itself must also satisfy the ten training-ready conditions in `docs/real_game_log_reasoning_plan.md`. A valid scenario cannot repair an incomplete or perspective-leaking source log.
