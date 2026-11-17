# Task 1: GazeGene MAE-B16 full training and zero-shot transfer

## Outcome

The fixed 12-epoch `train_and_zero_shot` run completed successfully on the
Windows RTX 4070 Ti SUPER server. The completion artifact reports `PASS` at
2026-11-16 03:29:21 UTC. Checkpoint selection used only the six-subject
GazeGene validation split; MPIIFaceGaze was evaluated strictly zero-shot after
selection and was not used for fitting or checkpoint selection.

The selected checkpoint is epoch 11:

| Metric | Result |
|---|---:|
| GazeGene validation subject-macro mean | 6.4666° |
| GazeGene test subject-macro mean | **4.8366°** |
| GazeGene test 95% subject-bootstrap CI | 4.5257–5.1003° |
| GazeGene test sample median / p90 / p95 | 3.5421° / 8.8380° / 12.6798° |
| MPIIFaceGaze zero-shot subject-macro mean | **5.3430°** |
| MPIIFaceGaze zero-shot 95% subject-bootstrap CI | 4.6886–5.9725° |
| MPIIFaceGaze zero-shot sample median / p90 / p95 | 5.0260° / 8.8111° / 10.0314° |

The GazeGene test result covers 180,000 samples from 10 held-out identities.
The MPIIFaceGaze result covers 45,000 samples from all 15 identities. The two
numbers measure different domains and must not be pooled.

## Protocol

- model: UniGaze MAE-B16;
- initialization: reviewed official MAE-B16 checkpoint;
- seed: `20261112`;
- GazeGene split: subjects 01–40 train, 41–46 validation, 47–56 test;
- epochs: 12, with validation-based checkpoint selection;
- physical batch size: 24;
- gradient accumulation: 2, effective batch size 48;
- evaluation batch size: 64;
- mixed precision: AMP FP16;
- optimizer: AdamW;
- backbone/head learning rates: `3e-6` / `3e-5`;
- weight decay: `1e-4`;
- run identity: `gazegene_b16_seed_20261112`;
- zero-shot target: normalized MPIIFaceGaze `p00`–`p14`.

The reviewed GazeGene metadata fingerprint was
`ae04683f47d578417b212a3fa06b93c69665424a7776c1b93f0527e343fa3fd8`.
The normalized MPIIFaceGaze fingerprint was
`4ca3987a7c370e8a8f5f6d382e20a68e1f8a9e9935dffe4e152fd9345dd859aa`.

## Training dynamics and resources

Epoch 11 achieved the best validation subject-macro error, 6.4666°. Epoch 12
was effectively flat at 6.4699° and therefore was not selected. Its training
mean angular error was 3.6787°.

The sum of the 12 measured epoch durations is 58,362.79 seconds, or about
16 h 13 min. Peak CUDA memory was 3,326,674,944 allocated bytes and
4,269,801,472 reserved bytes (3.33/4.27 decimal GB, or 3.10/3.98 GiB). This
confirms substantial unused capacity on the 16 GB GPU. Observed utilization was
bursty, indicating an input-pipeline bottleneck rather than a memory limit.
Future runs should benchmark data-loader parallelism, pinned memory,
persistent workers, prefetching, and larger physical batches in a separate
performance experiment. The completed run must remain unchanged.

## Checkpoint provenance

The checkpoint files remain private on the compute server and are not committed
or transferred through the relay.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `best_full_model.pt` | 346,339,435 | `d24c6cee737cd12dbc66264538d28d8d4fd8a4e71caca69a24fd79d2e906a87b` |
| `best_backbone_mae_format.pth` | 346,332,549 | `3e737772dad4f28d560487bc4b106cb6de2753ab37a181e011a5b67552c4a33f` |

## Paired MPIIFaceGaze LOPO transfer result

The follow-up experiment repeated all 15 frozen Task 0 LOPO folds with the
GazeGene `best_backbone_mae_format.pth` initialization. The combined completion
artifact reports `PASS`; both workers completed and all 45,000 samples from
`p00`–`p14` are present. The backbone hash matches the recorded GazeGene
checkpoint.

| Metric | GazeGene-initialized LOPO |
|---|---:|
| Subject-macro mean | **4.6857°** |
| Subject standard deviation | 1.5079° |
| 95% subject-bootstrap CI | 3.9483–5.4158° |
| Pooled sample mean | 4.6857° |
| Pooled sample median | 4.4275° |

The subject-macro mean is 0.1058°, or 2.31%, worse than the frozen Task 0
official-MAE initialization at 4.5799°. The aggregate confidence intervals
overlap substantially. A paired fold-level confidence interval for the
difference has not yet been reconstructed locally, so this result must not be
described as a statistically significant degradation. It does show that the
tested GazeGene initialization provides no observed aggregate transfer benefit.

## Interpretation and next decision

The experiment establishes a complete GazeGene pretraining artifact and a
cross-dataset zero-shot reference. The zero-shot MPIIFaceGaze subject-macro
error of 5.3430° is 0.7631° higher than the frozen Task 0 LOPO reference of
4.5799°. This is not evidence that GazeGene pretraining is harmful: Task 0 was
fine-tuned on MPIIFaceGaze training identities in every fold, while this result
uses no MPIIFaceGaze fitting at all.

The paired LOPO experiment is now complete. Under the fixed splits,
hyperparameters, seed, and checkpoint-selection rule, the GazeGene initialization
did not improve MPIIFaceGaze subject-macro error. Further expensive transfer
training is not the immediate priority; the project should return to the central
few-shot calibration and sample-selection question.

## Operational caveat

The scientific output completed, but the supervisor job ended with phase
`timed_out` because its runtime limit expired before the full 12-epoch plus
zero-shot workload finished. A shutdown request also did not enforce the quiet
hours as intended. The output is still complete because `completion.json`
reports `PASS`, all 12 history rows exist, the selected checkpoint and both
evaluation summaries exist, and the GPU is idle. Before another long run, test
the corrected stop/timeout path independently and size the runtime limit from
the measured 16-hour duration plus evaluation margin.

## Threats to validity

- GazeGene and MPIIFaceGaze differ in subjects, cameras, capture conditions,
  labels, and preprocessing; the zero-shot gap cannot identify which domain
  factor causes the error.
- GazeGene test and MPIIFaceGaze zero-shot metrics are single-run estimates for
  one seed.
- The GazeGene validation CI is wide because it contains only six identities;
  checkpoint choice may therefore be noisy.
- MPIIFaceGaze contains upsampled observations, so 45,000 rows do not represent
  45,000 unique raw frames.
- The paired LOPO aggregate is available, but the per-fold difference artifact
  has not yet been copied into the local repository for a paired significance
  analysis.
- The run measures static-frame angular error, not calibrated screen-point
  accuracy, real-webcam latency, fixation detection, or robustness slices.
