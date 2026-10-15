# Reviewed experiment results

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
