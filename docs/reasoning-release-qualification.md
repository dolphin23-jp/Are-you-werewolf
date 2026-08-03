# v2 reasoning release qualification

PR15 measures the engines; it does not tune their behaviour. The configured default remains
`legacy`. Only a later PR may change it after a human reviews live-model results.

## Audit model

Debug transcripts contain decision, correction, and result-publication audit records. Decision
records distinguish the coded target, displayed target, ballot, model request, evidence by
visibility, stale evidence, required results, and planned ally votes. These fields never enter the
normal player API.

Corrections are counted once by correction id. Seat retractions and affected seats are separate
impact measures. Publication audits compare required and published result identities and retain
omissions and duplicates.

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
`LLM_OUTPUT_PRICE_PER_MILLION`. The runner writes after every game so interruption preserves work.
No live run means `INCONCLUSIVE`; mock results can never qualify a release alone.

## Gate

Hard correctness failures immediately fail. Reliability and mock logical-call reduction thresholds
live in `backend/config/reasoning_release_gate.toml`. Passing also requires the configured minimum
number of live pairs. Behavioural diversity and faction win rates are reported for human review and
are not one-directional gates.
