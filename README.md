# Webcam Gaze Estimation Research

Research project on robust, low-latency appearance-based gaze estimation and
few-shot personalization for commodity RGB webcams.

The long-term goal is to build a practical system that estimates a user's 3D
gaze direction and, when display geometry is available, the point of gaze on a
screen. The project targets ordinary 30–60 FPS webcams and consumer inference
hardware. It is not intended to reproduce the full capabilities of a dedicated
infrared eye tracker.

> **Project status:** early research and baseline-development stage. The current
> task is to fine-tune and evaluate a pretrained gaze-estimation model on a
> prepared dataset before introducing uncertainty-aware personalization.

## Motivation

Webcam-based gaze estimation is inexpensive and widely deployable, but it is
sensitive to changes in user identity, camera geometry, head pose, illumination,
distance, glasses, reflections, blur, and image quality. This project studies a
middle ground between coarse uncalibrated webcam estimators and specialized
infrared eye-tracking hardware:

- a strong pretrained appearance-based backbone;
- reproducible subject-independent evaluation;
- lightweight user calibration without retraining the full model;
- uncertainty estimates and an optional abstention mechanism;
- low-latency inference on consumer hardware.

## Research questions

1. How well does a pretrained gaze model transfer across users, cameras, and
   capture conditions?
2. How much can a short person-specific calibration improve an uncalibrated
   baseline?
3. How many calibration observations are needed, and how should they be
   selected?
4. Can predictive uncertainty identify informative calibration samples and
   unreliable predictions?
5. Which improvements remain useful in cross-dataset and real-webcam
   evaluation?
6. What accuracy–latency–coverage trade-off is practical on consumer hardware?

## Intended system

The planned demonstration follows a calibration-first workflow:

1. Display a small sequence of stimuli at known screen locations.
2. Capture synchronized RGB webcam observations while the user follows them.
3. Select a small calibration subset and fit a lightweight user-specific
   component.
4. Predict a 3D gaze direction and optionally project it onto the display.
5. Visualize the point of gaze together with confidence or abstain when the
   prediction is unreliable.

The number, placement, duration, and repetition of calibration stimuli are
experimental variables rather than fixed assumptions.

## Research plan

### Task 0 — reproducible pretrained baseline

The current task is to fine-tune a pretrained gaze-estimation model on a
prepared dataset using cluster compute. Its purpose is to establish a complete
and reproducible training and evaluation path before increasing model
complexity.

The baseline should record:

- dataset and split version;
- subject-independent train, validation, and test identities;
- preprocessing and coordinate conventions;
- model initialization and complete resolved configuration;
- random seed, environment, hardware, and code revision;
- checkpoint-selection criterion and final evaluation results.

The first end-to-end Kaggle smoke test has completed successfully. It exercised
the official UniGaze preprocessing path, MAE-B16 checkpoint loading, full-model
forward/backward training, identity-disjoint validation and test evaluation,
checkpointing, plots, aggregate metrics, and W&B synchronization on a deliberately
tiny 48-sample subset. Its 17.49° test error is a plumbing diagnostic, **not** a
benchmark result. See
[`docs/task0_smoke_test.md`](docs/task0_smoke_test.md) for the protocol, measured
outputs, known limitations, and next execution gate.

### Uncertainty-aware few-shot personalization

The next research stage will investigate a pretrained backbone with a prediction
head that outputs both gaze and uncertainty. A lightweight calibration module,
such as an offset predictor, will be adapted from a small set of user-specific
observations without retraining the full network.

Candidate public datasets include MPIIFaceGaze and GazeCapture. Their exact use,
licenses, preprocessing, and split definitions must be documented before any
reported experiment.

Planned calibration experiments include:

- separating observation validity from calibration informativeness;
- rejecting or down-weighting blinks, blur, failed landmarks, and likely
  non-fixations;
- comparing high- and low-uncertainty selection as deliberately simple
  baselines;
- random selection as a baseline;
- fixed-grid, target-diversity, head-pose-diversity, and feature-diversity
  selection;
- quality-gated diversity selection;
- varying the number of selected calibration samples;
- comparing calibrated and uncalibrated predictions on identical test samples.

The literature evidence, current gaps, baseline order, and fixed-`k` evaluation
matrix are documented in
[`docs/calibration_research_directions.md`](docs/calibration_research_directions.md).

Where screen geometry is available, a differentiable projection may connect the
predicted gaze vector to the point of gaze and allow evaluation in both spaces.

### Robustness and failure analysis

The evaluation will include defensible slices for difficult conditions whenever
annotations or reproducible proxies are available:

- strong illumination and overexposure;
- motion blur and low image quality;
- reflections and glasses;
- large head rotations;
- changes in camera, distance, and display geometry.

Saliency or attribution maps may be used for qualitative diagnosis, but they
will not be treated as proof that the model learned a causal or anatomically
meaningful strategy.

## Evaluation

The primary metric for 3D gaze estimation is angular error in degrees. For
screen-point estimation, pixel error and physical error in millimetres will also
be reported when the display geometry is known.

Evaluation is expected to include:

- mean, median, standard deviation, and relevant error percentiles;
- per-subject results and confidence intervals;
- calibration gain and performance versus calibration-set size;
- uncertainty quality using suitable calibration and ranking metrics;
- risk–coverage behaviour when uncertain predictions are rejected;
- preprocessing, model, postprocessing, and end-to-end latency;
- throughput, model size, and peak inference memory;
- robustness slices and documented failure cases.

Within-dataset, cross-dataset, calibrated, uncalibrated, static-frame, and
temporal results must be reported separately. Test identities and labels must
not influence training, normalization, hyperparameter selection, calibration
thresholds, or checkpoint selection.

## Repository layout

The repository is being organized around the following structure:

```text
configs/              versioned experiment configurations
docs/                 protocol, decisions, literature notes, and thesis material
notebooks/            exploratory work only
scripts/              thin data, training, evaluation, and export entry points
src/gaze_estimation/  reusable typed Python package
tests/                 unit and regression tests
data/                  local datasets; never committed
outputs/               local runs and generated artifacts; never committed
results/               reviewed small summaries and comparison tables
```

Reusable logic belongs in `src/gaze_estimation/`; notebooks are not intended to
become the implementation layer.

## Local development

The project targets Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The expected local checks are:

```bash
python -m pytest
python -m ruff check .
python -m mypy src
```

The Task 0 pipeline and smoke configuration are available in
[`notebooks/task0_unigaze_b16_finetune.ipynb`](notebooks/task0_unigaze_b16_finetune.ipynb)
and
[`configs/task0_unigaze_b16_smoke.toml`](configs/task0_unigaze_b16_smoke.toml).
No reproducible benchmark result is claimed yet; the completed smoke run only
establishes that the end-to-end experimental plumbing works.

The portable experiment-record format and Kaggle artifact workflow are
documented in [`docs/experiment_tracking.md`](docs/experiment_tracking.md).
Every reported run must have a unique manifest tied to both its resolved
configuration and executed source content; notebook filenames alone are not
experiment identifiers.

## Data, privacy, and artifacts

Face and eye images are sensitive biometric data. Raw datasets, personal webcam
recordings, derived biometric crops, model weights, checkpoints, secrets, and
large generated artifacts must not be committed to this repository.

Only data obtained and processed under appropriate consent and dataset-license
terms should be used. Dataset acquisition, retention, anonymization, and
deletion requirements will be documented alongside each experiment protocol.

## Reproducibility principles

- Use strict subject-independent splits unless an experiment explicitly studies
  personalization.
- Compare complex methods against simple baselines.
- Save resolved configurations and environment metadata with every run.
- Record random seeds, subject lists or split hashes, hardware, and code revision.
- Keep negative and failed experiments in the experiment log.
- Report dispersion across subjects rather than only a pooled mean.
- Measure inference on intended consumer hardware before making deployment
  claims.

## Current limitations

- Only the deliberately tiny Task 0 smoke protocol has completed; a complete
  subject-independent fold and the 15-fold LOPO evaluation have not yet run.
- The scheduler-order warning observed in the W&B smoke run has been fixed in
  source, but the corrected notebook still requires a repeat smoke run.
- The current Kaggle implementation uses one of the two allocated T4 GPUs and
  must be split into resumable jobs before a full LOPO experiment.
- Dataset choices and license reviews are not yet finalized.
- No benchmark numbers or real-time performance claims are currently provided.
- Reliable fixation or saccade detection cannot be inferred from angular error
  alone and is outside the initial baseline scope.

## License

A source-code license has not been selected yet. Until a license is added, the
repository is publicly viewable but no permission to reuse, modify, or
redistribute the source code is granted.
