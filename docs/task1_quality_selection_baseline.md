# Task 1: quality-aware sample-selection baseline

## Outcome

Four deployable or diagnostic selectors were evaluated on the frozen Task 0
predictions with the unchanged duplicate-safe calibration/evaluation split:

- `high_quality`;
- deliberately adverse `low_quality`;
- normalized `pose_diversity`;
- `quality_then_target_diversity`.

The combined selector provides a statistically supported improvement only in
the one-shot case. With mean offset, it reduces the subject mean from the paired
random baseline by **0.3491°** (95% CI 0.0377–0.6382°). With MAP shrinkage, the
advantage is **0.3257°** (95% CI 0.1282–0.5444°).

The benefit does not persist as the budget grows. At `k=16`, the combined
selector is 0.2329° worse than random for mean offset (95% CI
-0.4237–-0.0658°) and 0.2354° worse for MAP (95% CI -0.4273–-0.0671°).
Quality-aware selection therefore cannot yet be claimed as a general reduction
in calibration cost.

## Protocol

- subjects: MPIIFaceGaze `p00`–`p14`;
- candidate/evaluation split: identical to mean/MAP, ridge, and target-diversity
  baselines;
- budgets: `k in {1, 3, 5, 9, 16}`;
- calibrators: mean offset and MAP shrinkage;
- primary reference: the existing 100-draw random average;
- quality inputs: brightness centrality, contrast, log Laplacian variance,
  dark-pixel fraction, and bright-pixel fraction;
- quality normalization: quantiles from the other 14 subjects only;
- pose normalization: mean and standard deviation from the other 14 subjects;
- combined selector: retain the highest-quality 50% of the candidate pool,
  then run target-space farthest-first selection;
- uncertainty: 100,000 paired subject-bootstrap replicates.

The server returned only 16 selected `group_id`/index pairs per subject and
selector. Images, gaze labels, head pose, predictions, and quality scores
remained private. Local evaluation matched every returned group against the
frozen candidate manifest before using Task 0 predictions.

## Mean-offset results

| Selector | k=1 | k=3 | k=5 | k=9 | k=16 |
|---|---:|---:|---:|---:|---:|
| `high_quality` | 3.8650° | 3.3998° | 3.3286° | 3.2509° | 3.2664° |
| `low_quality` | 4.8459° | 3.8562° | 3.5109° | 3.3056° | 3.2633° |
| `pose_diversity` | 3.4325° | 3.3899° | 3.0301° | 2.9177° | **2.8082°** |
| `quality_then_target_diversity` | **3.3060°** | **3.1245°** | **2.9792°** | 2.9517° | 2.9452° |

The best absolute result in this selector family is pose diversity at `k=16`,
2.8082°, but it remains 0.0959° worse than random mean offset; the paired CI
is -0.1723–-0.0089°.

## Interpretation

The negative control behaves as expected: `low_quality` is worse than random at
every budget, with confidence intervals below zero. The simple quality score
therefore contains a meaningful validity signal.

However, selecting only the highest-scoring frames is also worse than random
from `k=5` onward. This suggests that image cleanliness alone produces a
redundant or distribution-shifted calibration set. The combined method repairs
that problem for one shot, where avoiding a single bad sample is especially
valuable, but its fixed 50% gate still removes useful variation at larger
budgets.

Pose diversity does not improve the larger-budget result. The normalized
MPIIFaceGaze head-pose range may be too narrow, or extreme pose coverage may be
less important than representative subject-bias estimation for a constant
offset calibrator.

## Decision

- Retain `quality_then_target_diversity` as the supported one-shot selector.
- Do not claim a general selector win for `k >= 3`.
- Do not use top-quality-only sampling as the deployment default.
- Proceed to reliability-weighted robust offset, preserving coverage while
  reducing the influence of questionable samples.
- Tune any weight floor or robust-loss threshold on other subjects only.
- Compare weighting, selection, and their combination to determine whether the
  signals provide independent gains.

## Threats to validity

- Quality was validated against controlled corruptions, not annotated natural
  blink, non-fixation, landmark failure, or occlusion events.
- The fixed 50% gate is a protocol choice, not an optimized deployment threshold.
- Selecting a nested top-16 list is reproducible but may disadvantage methods
  whose optimal set is not nested across budgets.
- Sample quality is computed from normalized crops, not raw webcam frames.
- Candidate order still uses minimum sample index as an approximation to
  temporal structure.
- Results measure 3D angular error and not live calibration time or screen-point
  accuracy.

## Reproduction

Server-side selection:

```bash
python3 scripts/run_task1_quality_selection.py \
  --dataset-root /private/normalized/mpiifacegaze \
  --candidate-manifest configs/splits/task1_candidate_groups.csv \
  --quality-config configs/task1_quality_audit.toml \
  --config configs/task1_quality_selection.toml \
  --output-dir outputs/task1_quality_selection
```

Local evaluation uses `scripts/evaluate_task1_quality_selection.py` with the
frozen Task 0 predictions and the server-produced selection summary.
