# Task 1: ridge-affine calibration baseline

## Outcome

The leakage-safe ridge-affine experiment completed on the frozen Task 0
predictions for all 15 MPIIFaceGaze subjects. The best aggregate result is
random 16-shot calibration at **2.6861°** subject mean angular error, compared
with 4.6143° uncalibrated and 2.7123° for the paired mean-offset baseline.

The apparent 0.0262° advantage over mean offset is not conclusive: its paired
subject-bootstrap 95% CI is **-0.0366–0.0921°**. Ridge-affine therefore does not
establish a meaningful improvement over the simpler offset model under the
current protocol. At one shot, MAP shrinkage remains the preferred method.

## Hypothesis

A regularized two-output affine correction may capture subject-specific scale
and cross-axis error that a constant offset cannot, particularly as the
calibration budget increases.

The experiment supports the usefulness of calibration in general but leaves
this added-capacity hypothesis unresolved: the observed ridge-versus-offset
differences are small and almost all paired confidence intervals include zero.

## Protocol

- input: frozen Task 0 predictions, without UniGaze retraining;
- subjects: MPIIFaceGaze `p00`–`p14`;
- duplicate handling and temporal split: identical to the mean/MAP baseline;
- per-subject partitions: 20% candidate, 10% guard, 70% evaluation;
- budgets: `k in {1, 3, 5, 9, 16}`;
- selectors: 100 matched random draws and one fixed-uniform draw;
- affine inputs: predicted pitch and yaw plus an intercept;
- affine outputs: calibrated target pitch and yaw;
- regularization target: the identity affine map;
- candidate lambdas: `0.001, 0.01, 0.1, 1, 10, 100`;
- normalization scale: 30 degrees for both angular axes;
- uncertainty: 100,000 subject-bootstrap replicates.

For every held-out subject and budget, the ridge penalty is selected by mean
guard error over 20 random calibration draws from the other 14 subjects only.
The held-out subject's guard and evaluation labels never enter hyperparameter
selection. The selected penalty is then used for both random and fixed-uniform
selection on that subject. Lambda 0.1 was selected for every `k >= 3` fold and
for 13/15 one-shot folds; lambda 1.0 was selected for the other two one-shot
folds.

## Aggregate results

### Random selection

| k | Ridge-affine | Gain vs uncalibrated | Ridge advantage vs mean | 95% CI of advantage |
|---:|---:|---:|---:|---:|
| 1 | 3.4913° | 1.1230° | 0.1639° | 0.0071–0.2948° |
| 3 | 3.0285° | 1.5858° | 0.0135° | -0.0535–0.0827° |
| 5 | 2.8756° | 1.7387° | 0.0069° | -0.0604–0.0760° |
| 9 | 2.7487° | 1.8656° | 0.0216° | -0.0398–0.0874° |
| 16 | **2.6861°** | **1.9282°** | 0.0262° | -0.0366–0.0921° |

Positive advantage means lower error for ridge. Ridge is better than ordinary
mean offset at random `k=1`, but it is 0.0674° worse than MAP shrinkage there;
the ridge-versus-MAP CI is -0.1565–0.0350°. This does not replace MAP as the
one-shot default.

### Fixed-uniform selection

| k | Ridge-affine | Gain vs uncalibrated | Harmed subjects |
|---:|---:|---:|---:|
| 1 | 3.5510° | 1.0633° | 3/15 |
| 3 | 2.9803° | 1.6339° | 1/15 |
| 5 | 2.8167° | 1.7976° | 1/15 |
| 9 | **2.7175°** | **1.8968°** | 0/15 |
| 16 | 2.7589° | 1.8554° | 1/15 |

No fixed-uniform ridge comparison against mean or MAP has a paired confidence
interval excluding zero. As in the offset experiment, this fixed rule is not an
established selector improvement.

## Decision

- Keep MAP shrinkage as the one-shot baseline.
- Keep mean offset as the default simple baseline from `k >= 3`.
- Retain ridge-affine as a capacity-matched comparator, not as a demonstrated
  replacement for offset calibration.
- Proceed to quality and diversity selection at fixed calibration budgets,
  initially using offset/MAP so selector effects are not confounded with model
  capacity.
- Continue reporting harmed-subject and harmed-draw rates.

## Threats to validity

- Temporal order is approximated by minimum saved sample index; session-level
  boundaries are unavailable.
- Exact duplicates are removed, but adjacent near-duplicate frames cannot be
  identified from the saved predictions.
- The affine model uses only predicted pitch/yaw because head-pose features are
  not present in the frozen prediction artifact.
- Lambda selection uses only other subjects, but all predictions originate from
  the same dataset and Task 0 training recipe.
- The result measures 3D angular error, not screen-point accuracy or calibration
  duration in a live webcam workflow.

## Reproduction

```bash
python3 scripts/run_task1_ridge_affine.py \
  --predictions outputs/task0_lopo_baseline/all_predictions.csv \
  --mean-offset-results outputs/task1_mean_offset \
  --config configs/task1_ridge_affine.toml \
  --output-dir outputs/task1_ridge_affine
```

The generated directory contains `summary.json`, `ridge_tuning.csv`,
`per_draw_metrics.csv`, `per_subject_metrics.csv`, `aggregate_metrics.csv`, and
`paired_comparisons.csv`.
