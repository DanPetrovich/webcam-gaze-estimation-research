# Task 1: GazeGene experiment protocol

## Status

The local GazeGene FaceCrops release passed the semantic audit on 2026-11-12:
56 subjects, 9 cameras per subject, 2,000 frames per subject-camera pair, and
1,008,000 labels in total. The metadata fingerprint is
`ae04683f47d578417b212a3fa06b93c69665424a7776c1b93f0527e343fa3fd8`.

Training results are not yet available. They must not be inferred from the
dataset audit or from a smoke run.

A local CPU smoke run on 2026-11-12 exercised real GazeGene JPEG and label
loading, forward/backward, optimizer update, validation, test evaluation,
full-checkpoint reload, and MAE-format backbone reload. It used random temporary
initialization and two samples per split, so its metrics are intentionally not
research results.

A separate interface smoke used 15 one-row synthetic HDF5 files to exercise the
official MPIIFaceGaze loader and all zero-shot subject aggregation paths. It did
not use real MPIIFaceGaze data and therefore provides no scientific metric.

## Fixed split and model selection

- Train: subjects 1-40 (720,000 samples).
- Validation: subjects 41-46 (108,000 samples).
- Test: subjects 47-56 (180,000 samples).
- Test identities are evaluated only after selecting the checkpoint by the
  validation subject-macro mean angular error.
- The starting point is the official UniGaze MAE-B16 checkpoint at pinned
  UniGaze revision `9c240fbe33f3d6146970a77b7c8fa06a7e60019e`.
- GazeGene `gaze_C` is converted with `vector_to_pitchyaw(-gaze_C)` to the
  UniGaze `[pitch, yaw]` convention in radians.

## Experiments

1. `GG-within`: fine-tune MAE-B16 on GazeGene train subjects, select on
   GazeGene validation subjects, evaluate once on GazeGene test subjects.
2. `GG-zero-shot-MPII`: load the full `GG-within` checkpoint and evaluate it on
   every MPIIFaceGaze subject without fitting on MPIIFaceGaze.
3. `MAE-to-MPII`: existing Task 0 LOPO baseline initialized from official
   MAE-B16.
4. `GG-to-MPII`: repeat the exact same Task 0 LOPO folds and hyperparameters,
   but initialize the ViT backbone from `GG-within`; the two-output gaze head is
   newly initialized by `MAE_Gaze` in both transfer conditions.

For `GG-to-MPII`, point the existing Task 0 notebook at
`best_backbone_mae_format.pth` with `TASK0_PRETRAINED_CHECKPOINT`. Leave it
unset for `MAE-to-MPII`.

The primary comparison for experiment 4 is the paired per-subject difference
in mean angular error between `MAE-to-MPII` and `GG-to-MPII`. Report all 15
subject values, their macro mean and dispersion, and a paired confidence
interval. Do not tune the transfer condition on MPIIFaceGaze test folds.
After both LOPO result tables exist, run `scripts/compare_task1_transfer.py`
with the two `per_subject_metrics.csv` files to produce the paired table and
bootstrap confidence interval.

## Resource plan

The reference training device is an RTX 4070 Ti Super with 16 GB VRAM. The
default effective batch is 48 (`batch_size=24`, accumulation 2). If memory is
insufficient, reduce the physical batch and increase accumulation while keeping
the effective batch fixed. Record peak allocated and reserved GPU memory.

GazeGene carries additional No Distribution terms. Keep images, raw labels,
checkpoints, and biometric-derived artifacts off Git and public cloud storage.
Transfer the dataset only to storage covered by the access terms.

## Completion gate

Task 1 is complete only after the audit report, resolved configs, environment,
full and backbone checkpoints, training history, per-subject validation/test
metrics, zero-shot MPIIFaceGaze metrics, and the paired transfer comparison are
stored together with the code revision and interpretation. Failed runs remain
in the experiment log.
