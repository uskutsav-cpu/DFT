# Contributing

Use a review branch. Run `bash scripts/run_checks.sh` before committing. Keep
external engines optional and import them lazily. Add a regression test for each
bug. Tests using fabricated physics must say so. Never make tests pass by replacing
a missing high-level result with the known label.

Keep raw chemistry outputs immutable; use hashes and explicit failure records.
Any change to eligibility, grouping, thresholds, units, or cost semantics changes
the study protocol and requires a versioned explanation. Preserve a genuinely
untouched final test set. Do not add credentials, model weights, private personal
documents, institutional claims or generated large files to commits.
