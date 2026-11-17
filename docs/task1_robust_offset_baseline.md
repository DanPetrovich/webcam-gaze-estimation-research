# Task 1: robust residual-offset calibration

## Outcome

Leakage-safe Huber-weighted residual fitting completed for random calibration
and `quality_then_target_diversity` on all 15 subjects and five budgets.

On random samples, robust fitting provides small paired improvements over the
ordinary mean offset at `k=3`, `k=5`, and `k=9`:

| k | Robust error | Advantage over mean | 95% paired CI |
|---:|---:|---:|---:|
| 1 | 3.6551° | 0.0000° | 0.0000–0.0000° |
| 3 | 3.0150° | **0.0269°** | 0.0003–0.0587° |
| 5 | 2.8747° | **0.0078°** | 0.0003–0.0166° |
| 9 | 2.7496° | **0.0207°** | 0.0010–0.0478° |
| 16 | **2.7033°** | 0.0090° | -0.0127–0.0353° |

The absolute improvements are modest. Robust fitting is supported as a safer
default for `k=3–9`, not as a major accuracy advance.

For `quality_then_target_diversity`, robust fitting improves `k=16` by 0.0989°
(95% CI 0.0088–0.2173°), reducing error from 2.9452° to 2.8464°. This still
does not match robust random sampling at 2.7033°. Weighting mitigates but does
not remove the distribution shift introduced by the fixed 50% quality gate.

## Protocol

- residuals: known calibration target minus frozen Task 0 prediction;
- estimator: separate Huber iteratively reweighted locations for pitch and yaw;
- candidate deltas: 0.5°, 1°, 2°, 3°, 5°, and 10°;
- delta selection: mean guard error over 20 random draws from the other 14
  subjects only, separately for every held-out subject and budget;
- evaluation selectors: 100 matched random draws and the deterministic
  `quality_then_target_diversity` list;
- budgets: `k in {1, 3, 5, 9, 16}`;
- uncertainty: 100,000 paired subject-bootstrap replicates.

One-shot robust fitting is identical to a raw one-sample offset, so it cannot
replace MAP shrinkage at `k=1`.

The selected deltas were generally conservative: 10° for all one-shot folds,
mostly 3° at `k=3`, mostly 10° at `k=5`, mostly 5° at `k=9`, and 3–5° at
`k=16`. The mean effective weight fraction remains 0.98–1.00 for random draws,
which explains the small difference from the arithmetic mean.

## Interpretation

Residual consistency contains useful information, but gross residual outliers
are uncommon in the duplicate-safe candidate pool. Robust weighting therefore
acts as a small stability correction rather than changing the calibration
regime.

The larger benefit on quality-selected `k=16` indicates that hard filtering and
target coverage can still produce a few inconsistent residuals. Nevertheless,
the combined selector remains worse than random because its main problem is
representativeness, not only outlier contamination.

## Decision

- Keep MAP shrinkage for `k=1`.
- Use robust residual offset as a supported comparator/default for `k=3–9`.
- Treat ordinary mean and robust offset as practically similar at `k=16` under
  current evidence.
- Do not combine robust fitting with the fixed 50% quality gate as the final
  deployment policy.
- The next quality experiment should replace hard top-50% rejection with soft
  image-quality weights or adaptive reacquisition, while preserving random or
  stratified coverage.

## Threats to validity

- Huber weights use labelled calibration residuals and therefore require a
  guided calibration target; they do not apply to unlabeled passive frames.
- Delta tuning uses fixed Task 0 predictions from other identities in the same
  dataset.
- The paired bootstrap is across only 15 subjects.
- Residual robustness does not directly detect blur, blink, non-fixation, or
  landmark failure.
- The experiment uses normalized static crops rather than a live webcam stream.

## Reproduction

```bash
python3 scripts/run_task1_robust_offset.py \
  --predictions outputs/task0_lopo_baseline/all_predictions.csv \
  --mean-offset-results outputs/task1_mean_offset \
  --quality-selection-results outputs/task1_quality_selection \
  --selection-summary outputs/task1_quality_selection/selection_summary.json \
  --config configs/task1_robust_offset.toml \
  --output-dir outputs/task1_robust_offset
```
