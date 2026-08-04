# ruru-349517 expert scenario bundle

This directory contains cutoff-local scenarios extracted from the joint expert review of ruru-jinro No.349517「お気楽歓迎17A」.

## Files

- `d2-2-0-public-board.json`: initial 2-0 board, medium absence, grey-vote logic
- `d3-corona-death.json`: attacked seer candidate and treatment of the survivor
- `d6-yuki-black.json`: black-result execution under unresolved seer identity
- `d7-lwco-controlled-night.json`: guard peace, LW claim, and controlled-night design

The human-readable full case study is at:

- `docs/case_studies/ruru_349517.md`

The updated study method is at:

- `docs/expert_real_log_study_workflow.md`

## Source status

The source was reviewed from a saved full-log PDF. The original public no-spoiler HTML and full HTML have not yet been parsed and aligned by the repository pipeline.

All source event IDs beginning with `ruru349517.provisional.` are annotation placeholders. They are not canonical parser IDs.

## Review status

The content is marked `expert_reviewed` because it reflects a day-by-day review with human correction. It is not training-ready.

Before training:

1. preserve and hash both source HTML files;
2. parse canonical events;
3. align the public and full streams;
4. replace provisional IDs;
5. validate against the schema;
6. obtain independent adjudication;
7. clear all `review.blocking_issues`.
