# Task 0 UniGaze-B16 smoke test

## Outcome

The Kaggle smoke experiment completed the intended end-to-end path: public-data
acquisition, official UniGaze normalization, dataset audit, pretrained checkpoint
loading, full-model forward and backward passes, validation-based checkpoint
selection, held-out evaluation, plots, artifact persistence, and W&B logging.

This establishes that the experiment plumbing works. It does **not** establish
model quality and must not be quoted as a reproduction of a paper result.

## Executed protocol

| Field | Value |
|---|---|
| Dataset | MPIIFaceGaze |
| Preprocessing implementation | Official UniGaze `normalize_mpii.py` |
| UniGaze revision | `9c240fbe33f3d6146970a77b7c8fa06a7e60019e` |
| Initialization | Official MAE-B16 epoch-299 checkpoint |
| Training scope | Complete ViT backbone and gaze head |
| Train subject | `p02` |
| Validation subject | `p01` |
| Test subject | `p00` |
| Samples | 16 train + 16 validation + 16 test |
| Epochs | 1 |
| Batch size | 8 |
| Precision | Automatic mixed precision |
| Hardware | Kaggle T4 x2 allocation; one GPU used by the notebook |
| Split property | Identity-disjoint 1/1/1 smoke split |

The image normalization code is taken from the official UniGaze repository. Its
geometric normalization follows the established MPIIGaze/MPIIFaceGaze normalized
camera protocol; the implementation used by this project is pinned to the
revision above.

## Measured smoke outputs

The reviewed local record is
[`results/task0_unigaze_b16_smoke_kaggle_v1.json`](../results/task0_unigaze_b16_smoke_kaggle_v1.json).

| Metric | Observed value |
|---|---:|
| Training mean angular error | 20.4320° |
| Validation mean angular error | 16.4139° |
| Test mean angular error | 17.4859° |
| Test median angular error | 17.1663° |
| Test standard deviation | 9.8817° |
| Constant training-median baseline, test mean | 12.0536° |
| Fold execution time | 7.86 s |
| Peak allocated GPU memory | 1.08 GiB |

The later W&B-enabled Kaggle run also completed and produced the same reported
test mean and median. Its run ID is
`20261008T173626Z_smoke_c5e612ad8_s66cd13a3`; the dashboard is available at
[W&B](https://wandb.ai/petrovichdan/diploma-gaze-estimation/runs/20261008T173626Z_smoke_c5e612ad8_s66cd13a3).
The complete notebook execution, including dependency installation, downloads,
preprocessing, training, evaluation, and synchronization, took approximately
212 seconds.

## What the run verifies

- The official dataset archive and pinned UniGaze source can be downloaded in a
  clean Kaggle session.
- The UniGaze normalization script produces readable `224x224` HDF5 face crops
  with finite two-angle gaze labels for the smoke subjects.
- The MAE-B16 checkpoint loads and all model parameters participate in training.
- A forward pass, AMP backward pass, optimizer update, validation pass, best
  checkpoint save/load, and held-out test pass complete without fatal errors.
- Angular-error summaries, per-sample tables, learning curves, qualitative plots,
  run manifests, and checkpoints are generated.
- W&B receives aggregate training/evaluation telemetry and a small artifact
  without receiving face images, raw biometric data, or model checkpoints.

## Interpretation

The model is worse than the constant baseline in this run. That is expected to
be possible with only 16 training images and one epoch, and it is useful evidence
that the metric and baseline comparison are not hard-coded to make the model look
successful. No scientific conclusion about UniGaze follows from these numbers.

The result is not comparable to the roughly 4.8–4.9° FullFace MPIIFaceGaze
reference because the smoke protocol, sample count, training duration, model,
and validation procedure differ.

## Known issue and correction

The W&B-enabled smoke run emitted PyTorch's warning that `lr_scheduler.step()`
was called before an effective `optimizer.step()`. With AMP, `GradScaler` may
skip the optimizer update after detecting an overflow. The notebook previously
advanced `OneCycleLR` even on that skipped update.

`amp_optimizer_step()` now compares the AMP scale before and after the attempted
update. It advances `OneCycleLR` only when the optimizer update actually ran and
records `train/skipped_optimizer_steps` otherwise. The source and generated
notebook have passed syntax and structural checks, but this corrected path has
not yet been rerun on Kaggle.

## Next execution gate

Before reporting a Task 0 result:

1. Repeat the smoke run with the scheduler correction and verify that the warning
   is absent.
2. Persist the complete normalized 15-subject dataset outside ephemeral Kaggle
   storage.
3. Run one complete identity-disjoint fold and review its learning curves,
   baseline comparison, runtime, and GPU utilization.
4. Split the 15-fold LOPO evaluation into resumable jobs; one monolithic Kaggle
   run cannot be assumed to fit the platform's session limit.
5. Record the complete configuration, hashes, subject split, environment,
   failures, final per-subject metrics, and threats to validity.
