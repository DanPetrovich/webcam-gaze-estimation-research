# Research protocol

## Scope

The project studies appearance-based gaze estimation from commodity RGB webcams
with optional user calibration. Infrared eye trackers are treated as a reference
class, not as a claim of attainable webcam accuracy.

## Research positioning

Dedicated infrared eye trackers commonly benefit from controlled near-infrared
illumination, specialized optics or sensors, and capture rates that may exceed
those of commodity webcams. These properties can support fine-grained temporal
analysis, including fixation and saccade research, but exact accuracy and
sampling-rate claims are device- and protocol-dependent and require cited
sources.

Commodity RGB webcams are substantially more accessible and commonly operate at
about 30--60 FPS, but appearance-based estimates are more exposed to domain
shift caused by identity, camera characteristics, illumination, head pose,
distance, glasses, and display geometry. The research opportunity is therefore
not to claim parity with infrared hardware, but to find a useful accuracy,
latency, calibration-cost, and accessibility trade-off for general-purpose
deployment.

The literature review should explicitly cover:

- domain generalization across users, datasets, cameras, and environments;
- supervised and unsupervised domain adaptation;
- lightweight personalization and few-point user calibration;
- transfer of useful infrared-tracking ideas to visible-spectrum imagery;
- temporal modeling and uncertainty-aware rejection where webcam data supports
  them.

The checked references and current project-specific gaps for calibration and
uncertainty are tracked in
[`calibration_research_directions.md`](calibration_research_directions.md).
Future literature additions should preserve the same separation between cited
evidence, project decisions, and untested hypotheses.

## Intended demonstration

The initial demo should guide a user through known reference stimuli, collect
calibration observations, fit a lightweight personalized mapping or correction,
and then visualize live predictions. The demo is evidence of system integration,
not by itself evidence of scientific accuracy. Its evaluation must report the
calibration protocol, post-calibration gain, end-to-end latency, failures, and
the hardware and capture settings used.

## Baseline protocol (to be finalized)

1. Select a public dataset and reproduce a documented baseline.
2. Use subject-independent train/validation/test splits.
3. Evaluate within-dataset generalization before cross-dataset transfer.
4. Add a real-webcam evaluation set with consent and recorded camera/display
   geometry.
5. Compare uncalibrated and calibrated inference using identical test samples.
6. Measure accuracy, uncertainty/coverage, and end-to-end latency.

The current working hypotheses, literature gaps, calibration baselines, and
fixed-`k` sample-selection protocol are maintained in
[`calibration_research_directions.md`](calibration_research_directions.md).
That note treats sample validity and sample informativeness as separate signals:
invalid observations should be rejected or down-weighted before coverage is
optimized among the remaining candidates.

## Required experiment record

- hypothesis and change relative to baseline;
- dataset version, license, preprocessing, and split hash;
- resolved configuration, seed, code revision, and environment;
- checkpoint selection rule fixed before test evaluation;
- aggregate and per-subject results;
- compute time, peak training memory, and inference measurements;
- failure analysis, limitations, and threats to validity.

## Open decisions

- second public dataset for cross-dataset or temporal failure evaluation;
- 3D gaze direction, screen point, or multi-task target;
- calibration protocol and number/layout of stimuli;
- target consumer inference hardware and latency budget;
- real-webcam data collection and privacy protocol.
