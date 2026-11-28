# Task 1: independent GazeGene few-shot calibration

## Status

Completed. The server-side audit, deterministic two-draw smoke run, and full
100-draw aggregate evaluation all passed. The full GitHub Actions run
`44266194125` completed on 2026-11-18 in five minutes without model inference or
training; it evaluated the frozen epoch-11 prediction artifacts in place.

## Result

The uncalibrated subject-macro error on the fixed temporal evaluation region is
4.8534°. This differs from the previously reported 4.8366° full-test result
because calibration reserves the first 30% of every camera sequence and reports
only frames 600–1999.

| k | Mean offset | MAP offset | Huber offset |
|---:|---:|---:|---:|
| 1 | 6.9124° (-2.0590°) | 4.8113° (+0.0420°) | 6.9124° (-2.0590°) |
| 3 | 5.6289° (-0.7755°) | **4.7166° (+0.1368°)** | 5.0908° (-0.2375°) |
| 5 | 5.2403° (-0.3870°) | **4.6498° (+0.2036°)** | 4.7681° (+0.0853°) |
| 9 | 4.9387° (-0.0853°) | **4.5833° (+0.2700°)** | **4.5485° (+0.3049°)** |
| 16 | 4.7499° (+0.1034°) | **4.5221° (+0.3313°)** | **4.4497° (+0.4037°)** |

Parentheses show uncalibrated-minus-calibrated gain; positive is better. Bold
marks gains whose paired 95% subject-bootstrap confidence interval is above
zero. The supported intervals are:

- MAP `k=3`: 0.0120–0.2742°;
- MAP `k=5`: 0.0552–0.3700°;
- MAP `k=9`: 0.1030–0.4525°;
- MAP `k=16`: 0.1545–0.5262°;
- Huber `k=9`: 0.1119–0.5166°;
- Huber `k=16`: 0.2204–0.6062°.

At `k=16`, MAP and Huber harm 0/10 subjects after averaging the 100 draws;
ordinary mean offset harms 5/10. At `k=1`, ordinary mean and Huber are identical
and harm all ten subjects on average. The one-shot MAP gain is small and
inconclusive: 0.0420° with CI -0.0417–0.1353°.

The validation-selected Huber delta is 10° at `k=1` and 3° at every larger
budget. Method-to-method differences were not bootstrapped in this run, so the
numerical Huber advantage over MAP at `k=9,16` remains exploratory.

The result independently supports shrinkage as essential for sparse random
calibration and supports robust residual fitting only at the larger tested
budgets in this domain. It does not reproduce the MPIIFaceGaze recommendation
of ordinary random Huber calibration at `k=3,5`; the deployment policy must
therefore remain dataset-dependent until live-webcam evidence is available.

## Audit result

The audit confirmed 108,000 validation rows and 180,000 test rows with exact
subject-camera-frame coverage, no duplicate keys, finite values, disjoint
identity sets, and the expected checkpoint SHA-256. The maximum difference
between saved mixed-precision angular error and error recomputed from serialized
CSV angles was 0.0117° on validation and 0.0299° on test. No row exceeded the
fixed 0.05° tolerance.

## Question

Does the simple few-shot offset calibration family developed on MPIIFaceGaze
also improve the independently held-out GazeGene test identities when the model,
checkpoint, predictions, temporal split, budgets, and hyperparameter-selection
population are fixed before test evaluation?

This experiment tests random-sample mean offset, MAP-shrunk offset, and
Huber-weighted offset. It does not test the MPIIFaceGaze image-quality selector,
because the existing frozen GazeGene prediction artifact contains no reviewed
natural-image quality score.

## Frozen inputs

- model checkpoint: GazeGene MAE-B16 epoch 11;
- checkpoint SHA-256:
  `d24c6cee737cd12dbc66264538d28d8d4fd8a4e71caca69a24fd79d2e906a87b`;
- validation identities: GazeGene subjects 41–46;
- final test identities: GazeGene subjects 47–56;
- cameras: 0–8 for every identity;
- frames: 0–1999 for every identity-camera sequence;
- frozen prediction files: `validation_epoch_11_predictions.csv` and
  `gazegene_test_predictions.csv`, retained only on the private compute server.

The audit must confirm disjoint identity sets, exact sequence coverage, unique
`subject/camera/frame` keys, finite values, and agreement between the saved and
recomputed angular error within the CSV serialization tolerance.

## Calibration unit and split

Each identity-camera sequence is treated as one deployment-like webcam stream.
Offsets are fit separately for every identity-camera sequence. This prevents a
calibration observation from one of GazeGene's nine cameras from silently
serving as a calibration observation for another camera. Results are averaged
first across the nine cameras of an identity and then equally across the ten
test identities.

Every 2,000-frame sequence uses the same chronological split:

- frames 0–399: calibration candidate pool (20%);
- frames 400–599: guard set (10%);
- frames 600–1999: final evaluation set (70%).

The final test guard set is not used by the tested methods. It is retained to
match the established Task 1 split contract and to leave room for future fixed
policies without changing the evaluation rows.

## Methods and leakage control

Budgets are `k = 1, 3, 5, 9, 16`, with 100 deterministic random draws for every
identity-camera sequence and budget.

- `mean_offset`: arithmetic mean of target-minus-prediction pitch/yaw residuals;
- `map_offset`: shrinkage prior fit only from the 54 validation
  identity-camera candidate pools;
- `huber_offset`: coordinate-wise Huber location with its delta selected for
  each budget only on the validation identity guard rows.

The Huber delta candidates are 0.5°, 1°, 2°, 3°, 5°, and 10°. The final test
identities, candidate rows, guard rows, and evaluation labels do not influence
the prior or delta choice. Test candidate labels are used only for the allowed
few-shot user-specific calibration fit.

## Metrics and decision rule

The primary metric is the equal-weighted mean of the ten per-identity evaluation
means after first averaging that identity's nine camera means. Report for each
method and budget:

- uncalibrated and calibrated subject-macro angular error;
- mean gain and a paired 95% subject-bootstrap confidence interval with 100,000
  replicates;
- subject standard deviation;
- harmed-subject fraction;
- mean within-subject draw standard deviation.

A cross-dataset calibration gain is supported only when the paired confidence
interval for uncalibrated-minus-calibrated error is above zero. Method-to-method
ranking remains exploratory unless a paired comparison is explicitly added.

## Privacy and artifact handling

Predictions, labels, images, checkpoints and per-subject tables remain in
private storage. Public artifacts contain reviewed aggregate results and
protocol metadata.

## Threats to validity

- GazeGene's nine cameras are not the same as nine independent users; inference
  is therefore performed at the identity level, not the sequence level.
- Chronological candidate and evaluation regions may differ in target or pose
  distribution. This is intentional deployment-like evidence, but it does not
  isolate the cause of any gain or failure.
- GazeGene labels are angular camera-coordinate targets and do not establish
  screen-point calibration quality.
- The model was trained on GazeGene subjects 1–40. Test identities remain
  disjoint, but this is within-dataset personalization evidence for a model
  trained in the same dataset domain, complementary to the MPIIFaceGaze result.
- The experiment validates random calibration and residual fitting, not the
  image-quality component of the current one-shot hybrid policy.
