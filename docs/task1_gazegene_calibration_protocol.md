# Task 1: independent GazeGene few-shot calibration

## Status

Prepared for a server-side audit, deterministic smoke run, and full aggregate
evaluation. No calibration result is claimed in this document until the fixed
server run completes.

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

Predictions, labels, images, and checkpoints remain on the private compute
server. The GitHub relay may return only the bounded audit and aggregate metric
summary. Server-local per-subject tables are allowed for reproducibility but are
not transferred through the relay.

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
