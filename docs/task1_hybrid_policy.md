# Task 1: simple hybrid calibration policy

## Outcome

The completed simple baselines were consolidated into one prespecified
deployment candidate:

- `k=1`: soft quality/coverage weight 0.25 plus MAP offset;
- `k>=3`: random sampling plus the leakage-safe Huber residual offset.

The policy improves over the matching ordinary random reference at `k=1`,
`k=3`, `k=5`, and `k=9`, with paired subject-bootstrap confidence intervals
above zero. At `k=16`, Huber remains practically similar to the arithmetic mean
and its interval includes zero.

| k | Policy error | Reference | Advantage | 95% paired CI |
|---:|---:|---|---:|---:|
| 1 | **3.0797°** | random MAP | **0.3442°** | 0.1343–0.5704° |
| 3 | **3.0150°** | random mean | **0.0269°** | 0.0000–0.0587° |
| 5 | **2.8747°** | random mean | **0.0078°** | 0.0003–0.0166° |
| 9 | **2.7496°** | random mean | **0.0207°** | 0.0010–0.0478° |
| 16 | 2.7033° | random mean | 0.0090° | -0.0127–0.0354° |

The policy hypothesis is supported for `k=1–9`: a quality-aware one-shot
sample and robust larger-budget residual fitting provide small but consistent
improvements over their simple references. It remains unresolved at `k=16`.

## Protocol

- subjects: MPIIFaceGaze `p00`–`p14`;
- frozen evaluation: duplicate-safe Task 1 candidate/guard/evaluation split;
- one-shot selector: `soft_quality25_target_diversity`;
- one-shot calibrator: MAP shrinkage with prior fitted from the other subjects;
- larger-budget selector: the existing 100 matched random draws;
- larger-budget calibrator: Huber location with delta tuned on guard data from
  the other 14 subjects only;
- uncertainty: 100,000 paired subject-bootstrap replicates.

The one-shot reference is random MAP, because MAP is the supported ordinary
one-shot calibrator. The larger-budget reference is random arithmetic mean,
which isolates the Huber estimator's contribution on the same draws.

## Interpretation

The large one-shot contribution comes from avoiding an unreliable single frame
while retaining target coverage. For three or more samples, gross residual
outliers are uncommon, so robust fitting provides only a small correction. The
policy deliberately avoids hard quality gates and their reacquisition cost.

At `k=16`, the ordinary and Huber means are effectively interchangeable under
current evidence. A deployment implementation may retain Huber for a uniform
`k>=3` code path, but should not claim a resolved 16-shot accuracy gain.

This analysis consolidates results already measured on these subjects. It is
not an independent confirmation set and does not add statistical evidence
beyond the component experiments.

## Decision

- Adopt this hybrid rule as the current simple Task 1 deployment candidate.
- Keep the uncalibrated Task 0 result and ordinary random calibration as report
  baselines.
- Do not add further quality thresholds or weights on MPIIFaceGaze merely to
  optimize the same 15-subject result.
- Move the research work to calibration reliability/uncertainty, followed by
  independent cross-dataset or live-webcam confirmation.
- Preserve the full negative results: strong quality weights, top-quality-only
  selection, and the fixed 50% gate are not general defaults.

## Threats to validity

- The hybrid rule was designed after exploratory results on the same dataset.
- Only 15 subjects contribute to the paired bootstrap.
- The one-shot quality score was validated primarily with synthetic
  corruptions.
- Random results average 100 draws, while the one-shot soft selector is
  deterministic.
- Static normalized crops do not measure live reacquisition, fixation, or
  end-to-end screen calibration.
- The policy has not yet been confirmed on another dataset or real webcam data.

## Reproduction

```bash
python3 scripts/evaluate_task1_hybrid_policy.py \
  --soft-quality-results outputs/task1_soft_quality_selection \
  --robust-results outputs/task1_robust_offset \
  --mean-offset-results outputs/task1_mean_offset \
  --config configs/task1_quality_selection.toml \
  --output-dir outputs/task1_hybrid_policy
```
