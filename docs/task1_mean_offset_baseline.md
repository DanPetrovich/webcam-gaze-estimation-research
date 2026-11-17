# Task 1: few-shot mean-offset calibration baseline

## Outcome

The first few-shot personalization experiment completed successfully on the
fixed Task 0 predictions. A user-specific two-angle offset substantially reduces
identity-disjoint error without retraining UniGaze.

The strongest observed aggregate result uses 16 randomly selected calibration
observations and the ordinary sample-mean offset:

| Metric | Result |
|---|---:|
| Uncalibrated evaluation subject mean | 4.6143° |
| Calibrated subject mean | **2.7123°** |
| Absolute gain | **1.9020°** |
| 95% subject-bootstrap CI for gain | **1.3109–2.5109°** |
| Relative error reduction | 41.22% |
| Subjects harmed on average | 0 / 15 |
| Mean within-subject draw standard deviation | 0.1859° |

This is a new result under a calibration/evaluation split and must not be
confused with the 4.5799° Task 0 number evaluated on every saved sample.

## Leakage-safe protocol

The official pinned UniGaze preprocessing script upsamples each subject to 3,000
rows by sampling existing observations with replacement. Therefore, splitting
the 3,000 rows directly would leak copied images between calibration and
evaluation. The relevant implementation is
[`upsample_mpii.py`](https://raw.githubusercontent.com/ut-vision/UniGaze/9c240fbe33f3d6146970a77b7c8fa06a7e60019e/gazedata_preparation/upsample_mpii.py).

The experiment groups rows with identical target and predicted pitch/yaw. The
45,000 saved rows collapse to 37,667 unique observation groups. There are no
target-only collisions: every repeated target pair also has the same prediction,
which supports treating it as an exact duplicated observation.

For each subject, unique groups are ordered by their minimum saved sample index
and partitioned into:

- first 20%: calibration candidate pool;
- next 10%: unused guard interval;
- final 70%: evaluation set.

The guard reduces adjacency leakage at the calibration/evaluation boundary.
Exact duplicates cannot cross partitions. The smallest candidate pool contains
299 unique observations, so all tested budgets fit without replacement.

The split uses the saved index only as a temporal proxy. Original day/session
metadata is absent from Task 0 predictions, so a future dataset-backed rerun
should use actual session boundaries.

## Experimental factors

- subjects: `p00`–`p14`;
- calibration budgets: `k in {1, 3, 5, 9, 16}`;
- `random`: 100 deterministic draws per subject and `k`;
- `fixed_uniform`: one predeclared selection spread uniformly over candidate
  order;
- `mean_offset`: arithmetic mean of `target - prediction` in pitch/yaw;
- `map_offset`: dimension-wise Gaussian shrinkage toward a population prior;
- MAP prior mean, between-subject variance, and within-subject noise are fitted
  separately for every target subject using only the other 14 subjects;
- primary metric: equally weighted mean of per-subject evaluation mean angular
  errors;
- uncertainty: 100,000 subject-bootstrap replicates.

## Aggregate results

### Random calibration samples

| k | Mean offset | MAP offset | MAP/mean interpretation |
|---:|---:|---:|---|
| 1 | 3.6551° | **3.4239°** | MAP better by 0.2313°; 95% CI 0.0426–0.4100° |
| 3 | 3.0419° | **3.0137°** | difference 0.0283°; CI crosses zero |
| 5 | 2.8825° | **2.8770°** | effectively tied |
| 9 | **2.7703°** | 2.7717° | effectively tied |
| 16 | **2.7123°** | 2.7145° | effectively tied |

MAP shrinkage is useful specifically in the most data-starved one-shot case.
For random `k=1`, mean offset harms 4/15 subjects on average, while MAP harms
2/15. At `k=16`, both methods improve the average result for all 15 subjects,
although individual random draws can still hurt nearly unbiased subjects.

### Fixed-uniform calibration samples

| k | Mean offset | MAP offset |
|---:|---:|---:|
| 1 | 3.4841° | 3.4391° |
| 3 | 3.0092° | **2.9694°** |
| 5 | **2.7703°** | 2.8057° |
| 9 | **2.7664°** | 2.7688° |
| 16 | 2.8084° | **2.7993°** |

No fixed-versus-random paired confidence interval excludes zero at any tested
`k`. This experiment therefore does not establish that the simple fixed rule is
better than random selection. It establishes the calibration method baseline;
the calibration-sample-selection research question remains open.

## Per-subject interpretation

Random mean offset at `k=16` produces the largest gains for the identities that
showed systematic Task 0 bias:

| Subject | Uncalibrated | Calibrated | Gain |
|---|---:|---:|---:|
| p14 | 6.537° | 2.207° | 4.330° |
| p10 | 6.608° | 3.135° | 3.473° |
| p12 | 6.211° | 3.026° | 3.185° |
| p02 | 5.303° | 2.341° | 2.962° |

The smallest average gains occur for already-low-bias `p06` (0.046°) and `p00`
(0.194°). For `p06`, 36% of random `k=16` draws harm the subject even though the
average gain remains positive. A deployment policy should therefore shrink,
validate, or abstain from applying a noisy correction when estimated bias is
small.

## Decision

The minimum offset baseline is now established:

- use MAP shrinkage for `k=1`;
- treat MAP and ordinary mean offset as equivalent from `k>=3` under current
  evidence;
- retain random draws as the stochastic reference selector;
- do not claim a selection-method win from `fixed_uniform`;
- proceed to ridge-affine correction and then quality/diversity-based selection;
- continue reporting the fraction of harmed subjects and harmed draws, not only
  mean gain.

## Reproduction

```bash
python3 scripts/run_task1_mean_offset.py \
  --predictions outputs/task0_lopo_baseline/all_predictions.csv \
  --config configs/task1_mean_offset.toml \
  --output-dir outputs/task1_mean_offset
```

The script has no third-party runtime dependencies. It saves:

- `summary.json`;
- `split_manifest.csv`;
- `map_priors.csv`;
- `per_draw_metrics.csv`;
- `per_subject_metrics.csv`;
- `aggregate_metrics.csv`;
- `paired_comparisons.csv`;
- `calibration_curve.svg`.

## Threats to validity

- Sample index is only a temporal proxy; real session boundaries are not
  available in the prediction artifact.
- Exact duplicates are removed, but visually near-identical adjacent frames may
  remain within a partition.
- The experiment uses MPIIFaceGaze only and reuses Task 0 output angles; it does
  not establish cross-camera or real-webcam calibration performance.
- The fixed selector covers candidate order, not a known screen-target grid.
- The best observed method was read from the same experiment matrix; it requires
  confirmation on another dataset or real-webcam sequence before deployment.
