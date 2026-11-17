# Task 1: residual-consistency calibration reliability

## Outcome

Within-calibration residual dispersion was tested as a label-free reliability
score for random Huber calibration attempts. It does **not** predict downstream
evaluation error under the frozen Task 1 protocol.

Mean within-subject Spearman correlations between residual dispersion and
post-calibration error are close to zero: 0.030 at `k=3`, 0.032 at `k=5`,
-0.033 at `k=9`, and -0.007 at `k=16`.

A leakage-safe 80% coverage gate slightly lowers mean error numerically, but
every paired subject-bootstrap interval includes zero:

| k | All attempts | Accepted attempts | Apparent gain | 95% paired CI |
|---:|---:|---:|---:|---:|
| 3 | 3.0150° | 3.0085° | 0.0065° | -0.0247–0.0356° |
| 5 | 2.8747° | 2.8468° | 0.0279° | -0.0139–0.0776° |
| 9 | 2.7496° | 2.7431° | 0.0065° | -0.0312–0.0412° |
| 16 | 2.7033° | 2.6995° | 0.0039° | -0.0295–0.0390° |

The hypothesis that internally consistent calibration residuals identify more
reliable calibration attempts is not supported.

## Protocol

- subjects: MPIIFaceGaze `p00`–`p14`;
- budgets: `k in {3, 5, 9, 16}`;
- attempts: the existing 100 matched random draws per subject and budget;
- calibrator: Huber residual offset with the existing leakage-safe delta;
- reliability score: median angular distance of each calibration residual from
  the fitted pitch/yaw offset;
- target coverage: 80%;
- held-subject threshold: the 80th percentile of scores from the other 14
  subjects only, without evaluation labels;
- uncertainty: 100,000 paired subject-bootstrap replicates.

Observed coverage is 79.1–79.7%, close to the target. Rejected attempts imply
an expected acquisition multiplier of about 1.26 if consecutive attempts were
independent.

## Harmful-attempt analysis

The fraction of draws that worsen the uncalibrated result is already small and
falls with budget:

| k | All draws harmed | Accepted draws harmed |
|---:|---:|---:|
| 3 | 10.87% | 10.68% |
| 5 | 7.53% | 6.52% |
| 9 | 3.93% | 3.60% |
| 16 | 2.80% | 2.65% |

The gate does not materially isolate harmful attempts. Residual agreement can
be high even when all samples share a systematic error, while genuine
subject-specific offsets can create large but useful residuals.

## Decision

- Do not expose residual dispersion as a calibration confidence score.
- Do not spend capture time rejecting 20% of attempts with this score.
- Preserve the result as a negative uncertainty baseline.
- Future uncertainty work must use information not reducible to within-set
  agreement alone, such as predictive ensembles, heteroscedastic outputs,
  landmark/face-detection confidence, temporal fixation stability, or external
  image-quality failure signals.
- Keep the current hybrid policy unchanged.

## Threats to validity

- The experiment reuses random calibration draws and static normalized crops.
- A scalar median dispersion may miss structured pitch/yaw disagreement.
- The score does not include model, landmark, temporal, or target-fixation
  uncertainty.
- Only one fixed 80% target coverage was tested intentionally to avoid
  post-hoc threshold search.
- Calibration attempts are simulated samples rather than consecutive live
  webcam captures.

## Reproduction

```bash
python3 scripts/run_task1_calibration_reliability.py \
  --predictions outputs/task0_lopo_baseline/all_predictions.csv \
  --robust-results outputs/task1_robust_offset \
  --config configs/task1_calibration_reliability.toml \
  --output-dir outputs/task1_calibration_reliability
```
