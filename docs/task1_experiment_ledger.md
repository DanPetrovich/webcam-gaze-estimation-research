# Task 1 experiment ledger

## Scope and frozen reference

This ledger summarizes the completed personalization experiments on the frozen
Task 0 UniGaze MAE-B16 predictions. Detailed protocols, tables, limitations,
and reproduction commands remain in the linked experiment documents.

The Task 0 identity-disjoint LOPO reference is 4.5799° over complete subject
folds. Task 1 uses a duplicate-safe temporal 20% candidate / 10% guard / 70%
evaluation split, whose uncalibrated subject mean is 4.6143°. These numbers are
not interchangeable.

## Completed experiments

| Experiment | Primary result | Status | Decision |
|---|---|---|---|
| Mean/MAP offset | Random 16-shot mean reaches 2.7123°, a 1.9020° gain | Supported | Constant user offset is the main simple baseline; MAP is preferred at one shot |
| Ridge-affine | 2.6861° at `k=16`; 0.0262° advantage over mean has CI crossing zero | Unresolved | Added affine capacity is not justified as the default |
| Target diversity | 2.6997° MAP at `k=16`; no overall paired win over random | Not supported overall | Keep as a coverage/peripheral diagnostic |
| Image-quality audit | Synthetic corruption AUC 0.9452–0.9973 | Mechanism supported | Signals detect injected blur/exposure/occlusion, not natural fixation failure |
| Hard top-50% quality gate | One-shot gain 0.3491° mean / 0.3257° MAP; `k=16` worsens by about 0.23° | Mixed | Useful only for one shot; reject as a general policy |
| Huber residual offset | Gains 0.0269°, 0.0078°, 0.0207° at `k=3,5,9`; `k=16` unresolved | Supported, small | Conservative default for `k=3–9` |
| Adaptive hard gate | 90% gate reaches 2.6665° MAP at `k=16`, but advantage CI crosses zero | Exploratory | Hard-gate strictness depends on budget; no general selector claim |
| Soft quality/coverage | Weight 0.25 one-shot gain 0.3776° mean / 0.3442° MAP; neutral at `k=16` | Supported for one shot | Prefer to top-50% when one selector must work across budgets |
| Hybrid simple policy | 3.0797°, 3.0150°, 2.8747°, 2.7496°, 2.7033° for `k=1,3,5,9,16` | Current candidate | Soft 0.25 + MAP at `k=1`; random + Huber at `k>=3` |
| Residual-consistency abstention | Spearman correlation with error near zero; 80% coverage gains all unresolved | Rejected | Do not expose residual dispersion as confidence |
| Eccentricity robustness | Hybrid improves all slices, but `k=16` peripheral remains 3.1506° | Supported with failure mode | Always report peripheral slice; `p00`, `p06`, `p09` are fixed failures |
| Ridge by eccentricity | Worse than Huber peripherally at `k=3,5,9` | Rejected for spatial repair | Ridge trades central improvement for peripheral degradation |
| Target diversity by eccentricity | Peripheral gain 0.1669° at `k=9`, 0.1577° at `k=16`; central tradeoff | Supported as tradeoff | Not a general default; useful evidence for spatial error structure |
| Equal region quotas | Significantly worse peripherally at `k=5,9` | Rejected | More peripheral samples alone do not solve peripheral error |
| Prediction-routed piecewise offsets | Overall worse by 0.4444° at `k=9`, 0.1983° at `k=16` | Rejected | Do not use hard region routing with few-shot local offsets |
| GazeGene initialization transfer | LOPO 4.6857°, 0.1058° worse than official MAE initialization | Negative transfer | Keep official MAE Task 0 initialization |

## Current supported policy

The simplest evidence-backed calibration policy is:

1. For one accepted calibration sample, use soft quality/coverage weight 0.25
   and MAP shrinkage.
2. For three to nine random samples, use the LOSO-tuned Huber residual offset.
3. At 16 samples, Huber and the ordinary mean are practically similar; keeping
   Huber gives a uniform implementation but not a resolved accuracy gain.
4. Do not reject a completed calibration attempt using residual dispersion.
5. Report target-eccentricity slices and the peripheral failure subjects.

This is a research candidate, not a fully validated deployment policy. The
one-shot quality choice and later spatial experiments were developed on the
same 15 subjects.

## Claims that are justified

- Few-shot user calibration substantially improves the frozen uncalibrated
  model under a duplicate-safe protocol.
- MAP shrinkage protects the one-shot case.
- Moderate image-quality preference can improve one-shot sample selection.
- Robust residual fitting gives small gains for `k=3–9`.
- Calibration gains persist across central, mid, and peripheral target slices,
  although peripheral absolute error remains higher.

## Claims that are not justified

- Ridge-affine is better than constant offset overall.
- Quality-only, target-diversity, or hard-gated selection is a universal
  replacement for random calibration.
- Residual agreement is a valid uncertainty estimate.
- The current policy generalizes to another camera, dataset, screen geometry,
  or live webcam capture.
- Angular error alone establishes reliable fixation or saccade detection.

## Next experimental boundary

Further threshold, weight, quota, or region-boundary tuning on the same 15
MPIIFaceGaze subjects should stop. The next result capable of changing the
scientific conclusion must add new evidence:

- independent cross-dataset personalization predictions;
- a consented live-webcam calibration protocol with screen geometry;
- model/landmark/temporal uncertainty unavailable in the current prediction
  artifact;
- or a continuous spatial personalization method evaluated on a genuinely held
  validation population before final testing.

Bulk per-sample predictions or checkpoints must not be exchanged through the
GitHub relay. Any server-side follow-up must compute bounded aggregate metrics
where the private data and model artifacts reside.

The first independent follow-up is now protocol-frozen and awaiting execution:
random-sample mean, MAP, and validation-tuned Huber offsets on the ten held-out
GazeGene test identities. Its camera-safe split, metrics, privacy boundary, and
decision rule are recorded in `docs/task1_gazegene_calibration_protocol.md`.

## Detailed records

- `docs/task1_mean_offset_baseline.md`
- `docs/task1_ridge_affine_baseline.md`
- `docs/task1_target_diversity_baseline.md`
- `docs/task1_quality_audit.md`
- `docs/task1_quality_selection_baseline.md`
- `docs/task1_robust_offset_baseline.md`
- `docs/task1_adaptive_quality_gate_baseline.md`
- `docs/task1_soft_quality_selection_baseline.md`
- `docs/task1_hybrid_policy.md`
- `docs/task1_calibration_reliability.md`
- `docs/task1_target_eccentricity_robustness.md`
- `docs/task1_gazegene_full_result.md`
- `docs/task1_gazegene_calibration_protocol.md`
