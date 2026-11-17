# Task 1: image-quality and head-pose audit

## Outcome

The aggregate-only audit completed on all 45,000 normalized MPIIFaceGaze
samples from 15 subjects. It evaluated simple transparent image statistics and
the existing `face_head_pose` labels without persisting images or per-sample
features outside the private GPU server.

On deterministic controlled corruptions, the intended one-dimensional signals
separate original from corrupted frames with AUC above 0.90:

| Corruption | Signal | Original-vs-corrupted AUC |
|---|---|---:|
| Gaussian blur | log Laplacian variance | **0.9973** |
| Darkening | mean brightness | **0.9826** |
| Overexposure | negative bright-pixel fraction | **0.9452** |
| Central occlusion | negative dark-pixel fraction | **0.9836** |

The minimum AUC is 0.9452, so the predeclared `all_auc_at_least_0_90` gate
passes. This validates the mechanics of the simple quality signals for the
injected defects. It does **not** establish that they detect natural fixation
failures, landmark errors, closed eyes, or samples that cause large gaze error.

## Protocol

- dataset: normalized MPIIFaceGaze `p00`–`p14`, 3,000 samples per subject;
- natural quality sample: 256 deterministic samples per subject, 3,840 total;
- controlled sample: 64 deterministic samples per subject, 960 per corruption;
- image shape: 224x224x3 normalized face patch;
- quality features: brightness, contrast, log Laplacian variance, dark-pixel
  fraction, and bright-pixel fraction;
- head pose: all 45,000 `face_head_pose` rows, converted from radians to degrees;
- corruptions: Gaussian blur, multiplicative darkening, additive/multiplicative
  brightening, and a fixed central black occlusion;
- privacy: aggregate JSON only; no image, per-sample feature, prediction, or
  checkpoint left the server.

The runner downloaded research commit
`e9c386d3393f7c3f570ad4a65a16279dc7eb70d0`, verified fixed SHA-256 hashes for
the script and configuration, and executed through the event-driven Mac runner.

## Natural quality distribution

| Feature | Mean | p05 | Median | p95 |
|---|---:|---:|---:|---:|
| Brightness [0,1] | 0.3885 | 0.1369 | 0.4037 | 0.5854 |
| Contrast [0,1] | 0.1782 | 0.0895 | 0.1795 | 0.2634 |
| Log Laplacian variance | 3.9843 | 2.1437 | 4.0317 | 5.7143 |
| Dark-pixel fraction | 0.0169 | 0.0000 | 0.0022 | 0.0862 |
| Bright-pixel fraction | 0.0081 | 0.0000 | 0.0000 | 0.0384 |

These quantiles describe the deterministic audit sample, not universal webcam
quality thresholds. Selector gates must be fitted on non-test subjects rather
than copying pooled test-inclusive quantiles into the final protocol.

## Head-pose distribution

| Quantity | Mean | p05 | Median | p95 |
|---|---:|---:|---:|---:|
| Pitch | 1.48° | -15.90° | 2.08° | 19.17° |
| Yaw | 1.72° | -9.97° | 2.35° | 10.16° |
| Absolute pitch | 9.25° | 0.79° | 7.91° | 22.26° |
| Absolute yaw | 5.05° | 0.39° | 4.09° | 12.81° |

The normalized dataset contains materially more pitch than yaw variation. A
pose-diversity selector should therefore normalize axes using training-subject
statistics instead of raw Euclidean degrees.

## Decision and next experiment

The next fixed-budget comparison should use the unchanged calibration and
evaluation splits and compare:

1. `random` as the primary stochastic reference;
2. `low_quality` as a deliberate negative control;
3. `high_quality` using thresholds/statistics fitted on the other 14 subjects;
4. `pose_diversity` with training-subject axis normalization;
5. `target_diversity`, already measured;
6. `quality_then_target_diversity`, filtering invalid candidates before
   farthest-first target selection.

Mean/MAP offset should remain the first calibrator so selection effects are not
confounded with ridge capacity. The primary comparison remains paired within
subject and budget. Report angular error, harmed-subject fraction, rejection
rate, candidate-pool coverage, and the effective number of frames that must be
captured to obtain `k` accepted samples.

## Threats to validity

- Controlled corruptions are intentionally obvious and do not prove natural
  failure detection.
- The fixed occlusion rectangle is only a proxy; it is not an eye-landmark or
  eyelid-openness measurement.
- Normalized face crops may suppress some capture-quality variation present in
  raw webcam frames.
- Quality statistics were audited on a deterministic subset, while head pose
  used all rows.
- Upsampled observations mean 45,000 rows are not 45,000 unique raw frames.
- No relationship between these features and held-out gaze error has yet been
  established.

## Reproduction

```bash
python3 scripts/run_task1_quality_audit.py \
  --dataset-root /private/normalized/mpiifacegaze \
  --config configs/task1_quality_audit.toml \
  --output-dir outputs/task1_quality_audit
```

The dataset path is intentionally illustrative; machine-specific paths remain
outside the versioned configuration.
