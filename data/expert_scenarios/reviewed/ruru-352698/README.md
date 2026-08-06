# ruru-jinro No.352698 expert scenario bundle

This directory stores expert-reviewed cutoff-local scenarios for ruru-jinro No.352698「17A」.

## Source and status

- Game: No.352698
- Format: 17A
- Source reviewed: role-hidden public progression first, then the 67-page full log with roles, private talk, monologues, ability actions, graveyard discussion, and post-game discussion
- Review date: 2026-08-06
- Review status: `expert_reviewed`
- Training status: `false`

The event IDs in this bundle are provisional. They are human-readable references for the joint review, not canonical parser output.

Each scenario remains blocked by:

1. canonical event generation;
2. automatic public/full-stream alignment;
3. independent adjudication.

## Scenarios

- `d2-panda-guard-allocation.json`
  - 3-1-2 initial panda;
  - holding the panda to reduce medium-guard pressure;
  - execution value of low-information grey players;
  - why a true–wolf–fox ground truth does not justify early fox-mix over-weighting.

- `d4-low-credit-seer-before-panda.json`
  - low-credit seer claimant versus direct panda testing;
  - high information versus cross-world robustness;
  - preserving a possible kept wolf;
  - correction from “panda first” to “low-credit claimant first” as the safer default in this board.

- `d6-lw-hold-cross-divination.json`
  - two wolves publicly dead;
  - the same player is an LW candidate across the major seer worlds;
  - a non-LW safe execution purchases a night;
  - seer–fox mixing becomes operationally important only at the point where it changes execution order;
  - cross-divination resolves the board.

- `d7-double-death-convergence.json`
  - two corpses plus prior result chronology;
  - standard Variant-true worlds become impossible;
  - thin latent-seer worlds are retained but do not alter the final execution;
  - correction of an initially reversed fox/attack death assignment.

## Ground truth — evaluation only

- Seer: ずぶぬれお化け
- Medium: 時間遡行
- Werewolves: 参加できるよ、ぴゅりくれあ、ラッキーダガー
- Madman: 野村
- Guard: 17Aマン
- Freemasons: メンバー、ちっすー
- Fox: バリアントナイフ
- Village win on day 7

Do not copy this role table into public/player-perspective snapshots.

## Validation

All JSON files in this directory were parsed and validated against the current structural constraints of `schemas/expert_scenario.schema.json` during preparation. They must be revalidated after canonical event IDs replace the provisional namespace.

## Resume checklist

1. Preserve and hash the role-hidden/public source and the full source.
2. Parse canonical append-only events.
3. Align public statements with full-log private and postmortem events.
4. Replace every `ruru352698.provisional.*` ID.
5. Re-run schema validation.
6. Independently adjudicate the four scenarios.
7. Compare this case with No.349517:
   - common: role belief and action order diverge; the village wins through a controlled night;
   - different: No.352698 is a 3-1 initial-black credit game with a wolf claimant and a fox claimant, while No.349517 begins 2-0 with a dead medium and candidate attack;
   - new: unusual true composition need not be weighted early if it does not change action, but must be activated when the loss condition makes it operational.
