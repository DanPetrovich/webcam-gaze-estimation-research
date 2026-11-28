# Task 1 GazeGene error gap and temporal selection

## Results

The CPU evaluation completed on Windows on 2026-11-26.

| Target eccentricity | GazeGene share | GazeGene error | MPIIFaceGaze share | MPIIFaceGaze error |
|---|---:|---:|---:|---:|
| <10° | 4.11% | 3.1918° | 22.68% | 5.4299° |
| 10–<20° | 11.26% | 3.2993° | 55.22% | 5.3085° |
| 20–<30° | 16.33% | 3.5933° | 21.05% | 5.2558° |
| 30–<45° | 26.70% | 4.1211° | 1.04% | 6.9559° |
| ≥45° | 41.60% | 6.3626° | 0.0044% | 26.5806° |

The last MPIIFaceGaze bin contains two normalized rows. Full means are 4.8366°
on GazeGene and 5.3430° on MPIIFaceGaze for the same checkpoint. The
common-mix means are 3.4590° and 5.3528° with 32.745% overlap in bin mass.
The adjustment lowers the GazeGene estimate by 1.3776° on this reweighted
population. It highlights the effect of the distribution of gaze angles.
The fitted checkpoint has seen GazeGene training identities, which limits any
conclusion about intrinsic dataset difficulty.

On identical odd-indexed late GazeGene rows, the uncalibrated mean was
4.8420°. The mean offset fitted on all 400 early calibration candidates
achieved 4.3921° (gain 0.4499°, paired 95% CI 0.2693–0.6445°). The mean offset
fitted on 700 interleaved late rows reached 4.3083° (gain 0.5337°, CI
0.3528–0.7195°). Early and late offsets differed by 0.9509° on average in
pitch/yaw parameter space. The numerical late-fit advantage was 0.0838° and
has no paired interval in this comparison. The fit uses more samples and
temporally adjacent observations. The result shows limited gain from a single
constant offset under both tested fits even with abundant labels.

As a descriptive reference, fitting a mean offset to every duplicate-safe
MPIIFaceGaze LOPO candidate reduced 4.6143° to 2.6282° on the frozen Task 1
evaluation split across 15 subjects. There were 299–585 candidates per subject.
This calculation used `load_unique_observations`, `split_subject`,
`calibration_offsets` and `summarize_evaluation` from
`scripts/run_task1_mean_offset.py` on the local Task 0 prediction table.
These numbers use different fitted models and split constructions.

All 90 GazeGene sequences had zero eligible windows, out of 3,960 non-overlapping
windows checked. Consequently there is no temporal selection accuracy estimate.
The GazeGene authors describe randomized gaze and head-pose generation in
[section 3.2 of the dataset paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Bao_GazeGene_Large-scale_Synthetic_Gaze_Dataset_with_3D_Eyeball_Annotations_CVPR_2025_paper.pdf).
The same-checkpoint comparison and offset fits ran on all specified prediction
artifacts. A fixation-window comparison requires an ordered dataset with
repeated samples of the same fixation.

## Questions

1. How much of the apparent dataset gap remains when the checkpoint is fixed
   and both datasets are reweighted to the same coarse target-angle mix?
2. Does the early GazeGene calibration offset transfer to later frames, and
   how does it compare with an offset fitted on interleaved late frames?
3. Does median aggregation or selection by prediction stability improve
   later-frame calibration at a fixed window budget?

## Protocol

- Use the frozen GazeGene epoch-11 full checkpoint prediction artifacts.
  GazeGene uses 10 held-out identities and MPIIFaceGaze uses 15 zero-shot
  identities. GazeGene contributed 40 other identities to model training.
- The angle from camera forward to the ground-truth gaze vector is
  `acos(cos(pitch) * cos(yaw))`. The diagnostic uses fixed boundaries at
  10°, 20°, 30° and 45°. Weight subjects equally and rows equally within
  subject, then compare errors reweighted by the smaller marginal mass of
  each bin across the two datasets. The adjustment is descriptive.
- For offset drift, fit the mean pitch/yaw residual separately per GazeGene
  subject and camera on either frames 0–399 or even-indexed frames in
  600–1999. Test both fits and the baseline on the same odd-indexed late
  frames. The later fit uses many adjacent observations and is an optimistic
  diagnostic of locally correctable bias.
- For temporal selection, use non-overlapping nine-frame windows in frames
  0–399. Keep windows whose known target angles stay within a 1° pitch/yaw
  radius of the central frame. This is an offline eligibility proxy.
  Compare the center of nine random windows, the median prediction of the
  same windows, and the median prediction of the nine windows with lowest
  median distance from their prediction median. Use 30 deterministic draws
  for the two random variants and seed `20261126`.
- Each chosen window contributes one residual using its central target.
  Fit a coordinate-wise Huber offset with a fixed 3° delta, taken from the
  earlier GazeGene validation result for `k=9`. Compare all methods and the
  uncalibrated baseline on frames 600–1999 of the same eligible sequences.
  Average camera scores within identity. Bootstrap identities for paired
  gains with 10,000 replicates.

The dataset comparison holds model weights fixed. Training familiarity,
preprocessing, precision, label convention and distribution remain possible
contributors. Coarse angle matching does not control head pose, camera,
tracking conditions or label noise. Window availability and frame ordering
limit how this offline result generalizes to live fixation selection.

## Run

The research revision is `d2df9009611f1fc9e55217c3831de571b1a1d74c`.
The evaluation exports aggregate JSON. Prediction tables remain in private storage.
