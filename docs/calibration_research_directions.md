# Few-shot calibration and unreliable-sample research directions

Last literature check: 2026-10-15.

## Status of this note

This document separates three kinds of statements:

- **Evidence** reports what a cited paper evaluates or claims.
- **Working decision** records the current project plan; it may change after the
  Task 0 baseline is measured.
- **Hypothesis** is a proposition that still requires an experiment in this
  repository.

The immediate goal is not to design the most complex personalization method. It
is to establish whether calibration-set construction and sample reliability
matter when the number of labelled user observations `k` is fixed.

## Main research question

Given a frozen person-independent gaze estimator and a budget of `k` labelled
calibration observations from a new user, can a selection rule produce a larger
and more stable post-calibration gain than a fixed or random rule, especially
when the candidate observations contain blinks, blur, occlusion, landmark
failures, or failed fixation?

The central distinction is between two properties that should not be collapsed
into one uncertainty score:

1. **Reliability / validity:** is the frame and its nominal target label usable
   for calibration?
2. **Information / coverage:** conditional on being usable, does the observation
   add new gaze-target, head-pose, appearance, or feature-space coverage?

Selecting the largest uncertainty can preferentially select corrupted frames.
Selecting only the lowest uncertainty can produce a clean but redundant set.
The current hypothesis is therefore:

> At fixed `k`, reject or down-weight invalid observations first, then maximize
> coverage among the remaining candidates. This should outperform uncertainty-
> only, confidence-only, random, and fixed-grid selection.

This is a project hypothesis, not an established result.

## Evidence from the supplied literature

### Calibration-set composition

**Pose-Robust Calibration Strategy for Point-of-Gaze Estimation on Mobile
Phones** (BMVC 2025) reports that head-pose diversity can matter more than the
number of screen targets: in its MobilePoG experiments, one target observed
under three or four poses can outperform nine or thirteen targets observed at a
single pose. It also shows that calibration from a narrow pose distribution can
make a newly trained SVR or linear probe worse than no calibration. The paper
provides [code and data](https://github.com/ZhaoYujie2002/MobilePoG) for iTracker
and AFFNet with SVR, linear-probe, MLP-finetuning, and full-finetuning
calibrators. This is direct motivation for pose-aware coverage, but it studies a
phone-specific 2D PoG setting and controlled pose acquisition rather than
model-driven selection under webcam artifacts. See the
[paper](https://arxiv.org/abs/2508.10268) and
[project page](https://mobile-pog.github.io/).

**Towards High Performance Low Complexity Calibration in Appearance Based Gaze
Estimation** studies the number of gaze targets, images per target, and head
positions. It compares offset/gaze-decomposition calibration with last-layer
fine-tuning, linear adaptation, third-order polynomial adaptation, differential
methods, latent-parameter fitting, and FAZE. Its results motivate a simple
offset baseline and again indicate that head-orientation variation can be more
valuable than merely adding targets or near-duplicate frames. See the
[paper](https://arxiv.org/abs/2001.09284).

### Few-shot personalization methods

**FAZE** uses rotation-aware gaze embeddings and MAML for person-specific
adaptation. Its support samples are randomly drawn, so it is a useful adaptation
baseline but does not answer the selection question. The official implementation
is public, but its original stack is old and the authors describe expensive
meta-training that must be repeated for each `k`; porting it is therefore a
substantial experiment, not the first calibration baseline. See the
[paper](https://arxiv.org/abs/1905.01941) and
[code](https://github.com/NVlabs/few_shot_gaze).

**Few-shot gaze estimation with model offset predictors** learns a
person-specific predictor for the difference between an agnostic model and a
many-shot personalized model, initialized from meta-learned offset predictors.
It is closely aligned with lightweight residual correction, but the publication
page does not expose an official implementation. See the
[publication](https://www.amazon.science/publications/few-shot-gaze-estimation-with-model-offset-predictors).

**Test-Time Personalization with Meta Prompt for Gaze Estimation** (AAAI 2024)
updates less than 1% of a ResNet-18 through a meta-learned prompt and uses an
unlabelled symmetry objective at test time. It has an
[official implementation](https://github.com/hmarkamcan/TPGaze). It is relevant
as a newer parameter-efficient personalization method, but it solves unlabelled
test-time adaptation rather than fixed-budget selection of labelled calibration
samples. Its original environment (`Python 3.8`, `PyTorch 1.10.2`, `CUDA 10.2`)
also makes direct integration non-trivial. See the
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/28151).

**WebEyeTrack / BlazeGaze** (2025 preprint) is the most deployment-oriented
recent reference found in this pass. It combines explicit head-pose estimates,
a small gaze model, and MAML-based adaptation with `k <= 9`, and describes
blink suppression and in-browser inference. It is a useful system-level
comparison for the eventual demo. The paper does not formulate calibrated
predictive uncertainty or compare active calibration-set selection rules, so it
does not close the proposed question. See the
[paper](https://arxiv.org/abs/2508.19544) and
[code](https://github.com/RedForestAI/WebEyeTrack).

### Unreliable images, labels, and uncertainty

The supplied uncertainty papers address related but distinct problems:

| Work | What it addresses | Gap relative to this project |
| --- | --- | --- |
| [SUGE](https://arxiv.org/abs/2412.12890) | During source-model training, derives image and label confidences from consistency among a prediction, ground-truth label, and labels of same-person feature neighbours; uses two-network co-training, label correction, and sample weighting. | It is not a new-user `k`-shot calibration method. Its confidence construction needs labelled neighbourhoods and two trained networks, and it does not test online selection among blink/blur/non-fixation candidates. |
| [SeeTN](https://arxiv.org/abs/2604.16562) | Detects source-domain label noise through feature/label affinity in a prototype manifold and regularizes clean/noisy samples differently to improve domain generalization. | It improves source training, not user calibration. Its synthetic test uses large injected angular noise and does not isolate real transient capture failures or decide which `k` samples a user should provide. |
| [FIUD](https://arxiv.org/abs/2607.20072) | Distils error-relevant quality factors such as illumination, sharpness, pose, eye openness, symmetry, and pupil visibility into a single-pass uncertainty head; evaluates ranking and selective prediction. | The teacher needs supervised error, landmark failures are excluded from the evaluated subset, factors are frame-wise, and the work does not evaluate uncertainty-weighted few-shot fitting or calibration-set acquisition. |
| [Zheng et al., CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zheng_Enhancing_Accuracy_of_Uncertainty_Estimation_in_Appearance-based_Gaze_Tracking_with_CVPR_2026_paper.html) | Uses a small labelled post-hoc set to calibrate a predicted conditional uncertainty distribution under domain shift and introduces Coverage Probability Error (CPE). | It calibrates uncertainty magnitude rather than the user's gaze mapping or the choice of calibration observations. Its labelled post-hoc set also consumes data, so uncertainty calibration and gaze personalization must use disjoint data or account for a shared budget. |

Together, these works leave a concrete intersection open: use an uncertainty or
quality signal to construct and weight a small person-specific calibration set,
while explicitly distinguishing corrupted observations from clean but
informative observations.

## Working baseline decisions

### Frozen gaze model

Use **UniGaze-B16** as the primary frozen estimator because Task 0 already uses
its official MAE-B16 initialization and preprocessing. Keeping the same backbone
isolates calibration effects from backbone changes. The public UniGaze package
also exposes jointly trained inference checkpoints, but switching to one would
be a separate baseline version, not an implicit replacement of the Task 0
checkpoint. See the [official implementation](https://github.com/ut-vision/UniGaze).

Do not add a second large model until the complete experiment works on
UniGaze-B16. If a second model is needed to test model dependence, prefer a
lightweight, reproducible estimator with the same 3D gaze target and
normalization. Treat WebEyeTrack as a later 2D PoG system comparison rather than
mixing its results directly with 3D angular error.

### Calibration methods, in order

1. **Mean offset / MAP shrinkage:** estimate a two-angle subject bias from
   `target - prediction`. This is the required lowest-complexity baseline.
2. **Weighted robust offset:** fit the same bias using reliability weights and a
   robust loss or outlier-resistant estimator. This isolates the value of
   sample quality without adding representation capacity.
3. **Ridge affine correction:** regress the target angles from the frozen model
   prediction, optionally adding head pose. Regularization is selected without
   test labels.
4. **Feature-space linear probe or small residual MLP:** use frozen UniGaze
   features only after the output-space baselines are stable.
5. **MAML / meta-learned residual correction:** evaluate only after the simpler
   methods establish whether the bottleneck is adaptation capacity or sample
   selection.

A third-order polynomial warp is not a fair default for very small `k`: two
input angles produce ten monomials through degree three for each output. With
`k <= 9`, the unregularized problem is underdetermined. It may be included for
larger `k` with explicit ridge regularization, but it must not be presented as a
like-for-like one-shot baseline.

## Proposed selection rules

All learned rules must be fitted using training/validation subjects only. A test
subject's held-out gaze error must never enter sample selection.

At each `k`, compare at least:

1. `random`: repeated random draws, with the same draw seeds reused across
   calibration methods;
2. `fixed`: a predeclared grid or temporal rule independent of model outputs;
3. `low_uncertainty`: clean/easy samples only;
4. `high_uncertainty`: a deliberately simple active-learning baseline;
5. `target_diversity`: farthest-point or k-center coverage in known stimulus
   coordinates;
6. `pose_diversity`: coverage in head yaw/pitch/translation;
7. `feature_diversity`: k-center coverage in frozen UniGaze feature space;
8. `quality_then_diversity`: reject samples below a validity threshold, then
   optimize target, pose, and/or feature coverage;
9. `oracle`: select by true held-out calibration residual only as an unattainable
   upper bound, clearly excluded from deployable comparisons.

For a candidate `x` and an already selected set `S`, a simple deployable family
is

```text
score(x | S) = validity(x)^alpha * novelty(x, S)^beta
```

with a hard validity gate for face/landmark failure, closed eyes, or severe
blur. `alpha`, `beta`, and the gate are selected on validation subjects. The
first implementation should favor transparent components over a learned policy.

## Unreliable-sample protocol

Evaluate both naturally occurring and controlled failure modes:

- blink or partial eyelid closure;
- motion/defocus blur;
- over/underexposure and strong local contrast loss;
- partial face/eye occlusion or failed landmarks;
- head pose outside the calibration envelope;
- temporal non-fixation near stimulus transitions;
- nominal target-label error that simulates a fixation miss.

Controlled corruptions give known severity and ground truth but do not prove
real-world validity. Natural failures are realistic but need a reproducible
annotation or proxy. Report the two result groups separately.

Use three possible actions instead of forcing every candidate into one scalar
weight:

- **reject and reacquire** for obvious invalid frames;
- **retain with a reliability weight** for ambiguous but potentially useful
  frames;
- **retain normally** for valid observations.

This supports a practical demo: the system can ask the user to keep looking at
the same stimulus until it obtains a valid and non-redundant observation.

## Evaluation design

### Data separation

For every held-out subject, create disjoint calibration-candidate and evaluation
sets. Prefer different sessions or temporal blocks so adjacent video frames do
not leak near-duplicates across the boundary. Any uncertainty calibration,
selection threshold, polynomial degree, ridge penalty, stopping rule, or model
checkpoint is chosen on training/validation subjects only.

### Factors

- calibration budgets: start with `k in {1, 3, 5, 9, 16}`;
- candidate-pool condition: clean, naturally noisy, and controlled corruption;
- selector: the rules above;
- calibrator: offset, weighted robust offset, affine ridge, then learned methods;
- evaluation condition: matched and shifted head pose, plus clean and difficult
  image-quality slices.

To control experiment count, first compare all selectors with the offset and
weighted-offset calibrators. Then carry only the strongest non-oracle selectors
into affine, feature-space, and MAML experiments.

### Metrics

Report:

- uncalibrated and calibrated mean/median angular error per subject;
- calibration gain and the fraction of subjects harmed by calibration;
- dispersion and confidence intervals across subjects and repeated calibration
  draws;
- performance versus `k`, pose coverage, target coverage, and corruption rate;
- rejection/reacquisition rate and effective calibration time;
- uncertainty/error rank correlation, AURC or risk-coverage, and AUSE;
- interval quality with NLL and CPE when the method outputs a predictive
  distribution;
- latency and memory for quality scoring, selection, fitting, and inference.

The primary comparison for the selection question is paired within subject,
`k`, candidate pool, calibration method, and random seed. A selector wins only
if it improves post-calibration error without hiding the cost in excessive
reacquisition or reduced coverage.

## Minimum experiment sequence

1. Complete Task 0 and freeze the UniGaze-B16 checkpoint, preprocessing, and
   subject-independent split.
2. Implement deterministic offset and ridge-affine calibration with random and
   fixed selection.
3. Add head-pose and simple image-validity measurements; validate that they
   identify injected corruptions without using test labels.
4. Compare `low_uncertainty`, `high_uncertainty`, diversity-only, and
   `quality_then_diversity` at fixed `k`.
5. Add reliability-weighted fitting and measure whether selection and weighting
   provide independent gains.
6. Only then evaluate a learned uncertainty head and a MAML or meta-residual
   calibrator.
7. Confirm the best result on a cross-dataset or real-webcam protocol with
   disjoint calibration and test sequences.

## Open decisions

- Which dataset supplies enough temporal information to mark stimulus
  transitions and fixation failures rather than only static labelled frames?
- Is the first target 3D gaze correction only, or should a separate 2D PoG
  experiment be added once camera/screen geometry is available?
- Which uncertainty method can be reproduced with the frozen UniGaze feature
  extractor without retraining two complete backbones?
- Should a labelled sample used for post-hoc uncertainty calibration count
  against the same user budget `k` as gaze personalization? The conservative
  answer is yes unless a separate validation session is explicitly available.
- What consumer device and maximum acceptable reacquisition time define the
  operational budget for the final demo?
