# Expert real-log study workflow — 2026-08 update

## Why this addendum exists

The first full expert review of ruru-jinro No.349517 exposed a risk in the original program: over-optimizing for direct role prediction from speech.

The updated target is an agent that makes strong faction-aware decisions under uncertainty. Role inference remains necessary, but is only one internal component.

This document supplements `docs/real_game_log_reasoning_plan.md`. When the two differ in emphasis, this addendum controls the expert-study curriculum and evaluation order; it does not weaken the original fact, perspective, or provenance requirements.

## Five separately scored layers

Every analyzed cutoff must keep these layers separate.

1. **Observation**
   - exact public and actor-private events available at the cutoff;
   - no later role, death, result, monologue, or private talk leakage.

2. **Hard logic**
   - possible and impossible worlds;
   - conditional deductions;
   - explicit unsat cores;
   - “unlikely” never becomes “impossible” without a rule contradiction.

3. **Soft weighting**
   - main line and alternatives;
   - source event and causal group for every factor;
   - qualitative strength unless numeric calibration has been validated.

4. **Action utility**
   - best execution, vote, divine, guard, attack, CO, or withholding action;
   - information gain;
   - faction objective;
   - loss in important alternative worlds;
   - effect on future execution count and controllability.

5. **True-world postmortem**
   - roles and private actions;
   - accidental correctness;
   - deceptive intent;
   - missed but unavailable information;
   - counterexamples and generalized prevention rules.

The fifth layer is evaluation-only and must never be copied into layers one through four.

## Mandatory distinction: belief vs action

The annotation UI and future engine should expose at least:

- role/faction belief;
- execution value;
- information value;
- endgame judgeability;
- ability-use value;
- catastrophic-loss flags.

A role can be the most likely wolf without being today's best execution. A seer can be more likely false yet still be the best guard target because the true-world loss is asymmetric.

## Two-pass review protocol

### Pass A: no-spoiler playable review

For each meaningful cutoff:

1. freeze the event stream;
2. reconstruct the selected perspective;
3. state public facts before interpretation;
4. enumerate representative worlds;
5. mark logical-only worlds separately;
6. add soft factors with causal groups;
7. recommend an action;
8. state the most valuable next observation;
9. record uncertainty and expected expert disagreement.

The reviewer must not read the role table, private talk, monologue, graveyard discussion, or later days.

### Pass B: full-log adjudication

After Pass A is frozen:

1. reveal the true world;
2. evaluate each prior inference without rewriting it;
3. mark correct-by-luck outcomes;
4. identify signals that were truly available;
5. record deceptive or strategic intent;
6. write counterexamples;
7. record AI corrections;
8. decide whether the scenario is training-ready.

## Evidence-quality rubric

Prefer factors that:

- distinguish among live worlds rather than merely fit one;
- cost the actor something;
- were produced before a designation or result made the action inevitable;
- update after new public information;
- lead to a testable future observation.

Discount factors that:

- are repeated statements from one causal event;
- appear after a partner's execution is nearly fixed;
- are self-meta claims such as “I would never do this as wolf”;
- infer one unique motive from an action with several strategic explanations;
- identify “role-holder pressure” but cannot separate wolf, fox, guard, or latent power.

## Evaluation metrics

Role-label accuracy is secondary. Track:

1. public fact reconstruction accuracy;
2. hard-logic contradiction rate;
3. possible-world recall;
4. calibration of main/alternative/thin labels;
5. causal double-counting rate;
6. action regret against expert adjudication;
7. expected information gain;
8. catastrophic-loss avoidance;
9. counterfactual robustness;
10. perspective leakage rate.

A model that guesses fewer roles but chooses safer, more informative actions can outperform a role classifier.

## Scenario bundle policy

A real-game bundle should contain:

- a human-readable case study;
- a source/provenance README;
- several cutoff-local schema-valid scenarios;
- review and training gates;
- a resume checklist.

Use `review.status = expert_reviewed` only after a human expert has checked the content. `training_ready` remains false until canonical event IDs resolve, public/full alignment succeeds, and blocking issues are empty.

Provisional source IDs must use a visibly provisional namespace and must not be silently treated as parser output.

## Stage-dependent treatment of unusual compositions

A composition can be true without deserving early operational weight.

For every unusual world — fox mixing, two wolf claimants, latent true roles, or a latent madman — record two separate labels:

- **logical status**: possible, impossible, or conditional;
- **operational status**: whether believing the world would change the current action.

Do not increase the early prior merely because the postmortem later reveals the unusual world. In an ordinary 3-1 start, fox mixing can remain logical-only when the same grey vote, guard allocation, and claimant deadline are optimal under true–wolf–madman. Escalate the world only when a concrete condition changes action, for example:

- four seer claims;
- claimant survival or results that create a composition-specific contradiction;
- two wolves publicly dead and the opposing claimant's fox status controls LW execution order;
- an inexpensive mutual or opponent divination can directly resolve the loss condition.

The training label should reward the timing of escalation, not simply whether the final composition was named.

## Attack interpretation: fit, intent, and quality

For each attack, store separately:

1. public strategic explanations that fit the attack;
2. the actual private intent, if available in the full log;
3. whether that intent was coherent and faction-optimal;
4. accidental outcomes such as removing a guard without targeting a guard.

Do not infer that wolves executed a sophisticated plan merely because the attack is consistent with one. Guard fear, safe-corpse preference, weak coordination, improvisation, and mistakes can produce the same public sequence.

Likewise, an explicitly deceptive or abnormal attack is not automatically good. Grade the intent and expected value separately from the fact that the wolves wanted to confuse the village.

## Curriculum after the first case

Do not select the next game by similarity alone. Build contrast.

Recommended order:

1. 3-1, medium alive, wolf seer claimant, credit battle;
2. initial black progression;
3. fox win;
4. latent madman;
5. genuine latent seer;
6. true seer loses credit;
7. correct inference but wrong execution order;
8. weak inference but robust procedure wins.

After each game, compare:

- which factors transferred;
- which tactics were composition-specific;
- whether a previously learned heuristic generated false positives;
- what new counterexamples are needed.

## Near-term implementation order

1. obtain paired public/full HTML;
2. parse append-only canonical events;
3. align both streams;
4. reconstruct actor snapshots;
5. migrate provisional scenario IDs;
6. add schema validation in CI;
7. evaluate hard logic before training soft weights;
8. add action-utility baselines;
9. only then learn speech and attack weighting from the reviewed corpus.

## Definition of done for one studied game

A game is complete only when:

- public and full sources are both preserved and hashed;
- parser output is deterministic;
- event streams align;
- at least four high-value cutoffs are annotated;
- no perspective leakage remains;
- hard worlds are independently checked;
- action recommendations include alternatives and risks;
- AI corrections include counterexamples;
- a second reviewer adjudicates disagreements;
- the bundle is schema-valid;
- training-ready status is explicit.
