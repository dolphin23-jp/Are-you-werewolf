# v2 reasoning release qualification

PR15 measured the engines without tuning them. The configured default has since moved to `v2`
(`WEREWOLF_REASONING_ENGINE`); `scripts/evaluate.py` plays whichever engine that setting selects.

## Audit model

Debug transcripts contain decision, correction, and result-publication audit records. Decision
records distinguish the coded target, displayed target, ballot, model request, evidence by
visibility, stale evidence, required results, and planned ally votes. These fields never enter the
normal player API.

Corrections are counted once by correction id. Seat retractions and affected seats are separate
impact measures. Publication audits compare required and published result identities and retain
omissions and duplicates.

Schema version 3 also records accepted/rejected night actions with action type and legality,
decision-time public-result/correction/votable snapshots, evidence attempted at each rendering
stage, and fully grounded wolf ally-vote plans. Offline analysis therefore does not infer these
hard boundaries from prose.

## Denominators

Vote-change rates use audited ballots. Failure/skip rates use logical generation and discussion
attempt counts respectively. Win ratios and Wilson intervals use completed games for that engine;
campaigns below 100 games are descriptive, not evidence for tuning.

## Running evaluations

Mock campaign (safe for CI):

```bash
python scripts/evaluate_reasoning_campaign.py --provider mock --seeds 1:50 \
  --engines legacy v2 --output artifacts/mock-campaign.json
```

Live paired evaluation (manual only):

```bash
python scripts/live_ab_reasoning_check.py --seeds 11 12 13 --engines legacy v2 \
  --output-dir artifacts/live-ab --max-http-requests 4000 --max-estimated-cost 20 --resume
```

Prices must be supplied through CLI flags or `LLM_INPUT_PRICE_PER_MILLION` and
`LLM_OUTPUT_PRICE_PER_MILLION`; a cost cap without prices is refused at startup, because an unknown
estimate never reaches the cap. The runner writes after every game so interruption preserves work.
No live run means `INCONCLUSIVE`; mock results can never qualify a release alone.

The runner enforces Stage A (v2 smoke), Stage B (the matching legacy game), and then the remaining
paired seeds. A hard v2 error (any of `HARD_FAILURE_FIELDS`) stops later stages. Both caps are
checked before each HTTP request, so a run overshoots the cost cap by at most one response. On
exhaustion `BudgetExceeded` ends the game (it is a `BaseException`, so the per-turn fallbacks cannot
swallow it) and the game is recorded as `budget_exhausted`.

Only finished games (`GAME_OVER`) against a real provider (`luna`; the `mock` and `scenario`
doubles are refused) count as live games. Each run gets a unique `game_id`, so a review filed for one
run cannot approve another run of the same seed.

Human reviews are JSON files supplied with `--review-dir`. Every v2 game must have a review that
answers yes to every checklist item before the gate can pass. A finished review with any "no" is a
`FAIL`; missing or partial reviews are `INCONCLUSIVE`, never an implicit approval.

## Gate

Hard correctness failures immediately fail. Reliability and mock logical-call reduction thresholds
live in `backend/config/reasoning_release_gate.toml`. The reduction threshold is applied only when
the live runner is given the mock campaign's output (`--mock-campaign artifacts/mock-campaign.json`,
run with `--engines legacy v2`); `aggregate.json` records the value used, or `null` when the check
was skipped. Passing also requires the configured minimum
number of live pairs. Behavioural diversity and faction win rates are reported for human review and
are not one-directional gates.
