# Reviewed experiment results

[`selected_results.csv`](selected_results.csv) contains the headline results,
with the evaluation protocol, participant count, calibration budget and source
report for every row. Each confidence interval refers to the quantity named in
`ci_target`. A blank field indicates that this summary does not report it.
The [comparison figure](calibration_comparison.svg) uses the two 16-shot rows.

This directory contains small, reviewable experiment summaries and comparison
tables. Raw datasets, checkpoints, per-sample biometric data, and large outputs
belong under ignored local storage or a private artifact store.

`experiment_index.csv` is generated from downloaded run manifests with
`scripts/build_experiment_index.py`. Review it before committing because source
paths or external identifiers may be machine-specific.

`task0_unigaze_b16_smoke_kaggle_v1.json` records the first successful Kaggle
end-to-end smoke run. It is plumbing evidence only, not a benchmark result.
The human-readable protocol, interpretation, known scheduler issue, and next
execution gate are documented in
[`docs/task0_smoke_test.md`](../docs/task0_smoke_test.md).
