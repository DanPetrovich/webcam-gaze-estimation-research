# Task 1 server handoff

Use this handoff in the Codex installation on the personal MacBook that can
reach the Windows GPU server. The target GPU is an NVIDIA RTX 4070 Ti Super
with 16 GB VRAM.

## Preconditions

1. Resolve the existing bootstrap blocker by installing the managed Python in
   a normal per-user Windows directory rather than the restricted working
   volume that returns OS error 448.
2. Make the diploma repository, the official pinned UniGaze source, the
   official MAE-B16 checkpoint, normalized MPIIFaceGaze HDF5 files, and the
   extracted GazeGene FaceCrops directory visible on the server.
3. Do not move GazeGene through GitHub, the relay issue, or public storage. Its
   additional No Distribution terms apply, and face images are biometric data.
4. Verify `nvidia-smi`, CUDA PyTorch, free disk space, and SHA-256 of the MAE
   checkpoint before the first run.

The environment must provide `torch`, `torchvision`, `timm`, `numpy`, `pandas`,
`Pillow`, and `tqdm`. Zero-shot MPIIFaceGaze evaluation additionally requires
`h5py`, `opencv-python-headless`, and `omegaconf`. Install the CUDA-compatible
PyTorch build for the server driver rather than relying on an arbitrary CPU
wheel.

## Execution order

The server supervisor now exposes two fixed allowlisted presets:

- `task1_gazegene_mae_b16_smoke`, with a 1,800-second maximum runtime;
- `task1_gazegene_mae_b16_full`, with a 43,200-second maximum runtime.

Both presets use a server-local configuration and reject relay-supplied paths,
commands, arguments, or environment variables. The launcher verifies the
reviewed training-script and MAE-checkpoint SHA-256 values, the pinned UniGaze
revision, and the passing audit fingerprint before execution. The full preset
also requires all 15 MPIIFaceGaze subject files. On 2026-11-13 the installed
smoke preset correctly stopped at preflight because the server-local config was
not yet present; no training was started.

After transfer, run `notebooks/task1_gazegene_audit.ipynb` against the server
copy and require the same metadata fingerprint
`ae04683f47d578417b212a3fa06b93c69665424a7776c1b93f0527e343fa3fd8`.
Do not train on a partial or mismatched copy.

Then run `notebooks/task1_gazegene_b16_finetune.ipynb` with environment
variables equivalent to:

```text
TASK1_RUN_MODE=train_and_zero_shot
TASK1_GAZEGENE_ROOT=<server path to GazeGene_FaceCrops>
TASK1_UNIGAZE_ROOT=<server path to pinned UniGaze/unigaze>
TASK1_MAE_CHECKPOINT=<server path to mae_b16_checkpoint-299.pth>
TASK1_MPII_ROOT=<server path containing p00.h5 ... p14.h5>
TASK1_OUTPUT_ROOT=<private server output directory>
TASK1_RUN_ID=gazegene_b16_seed_20261112
TASK1_RESUME=true
TASK1_BATCH_SIZE=24
TASK1_EVAL_BATCH_SIZE=64
TASK1_ACCUMULATION=2
TASK1_NUM_WORKERS=4
```

If CUDA runs out of memory, change the physical/effective batch pair from
`24×2` to `12×4` or `8×6`; do not silently change the effective batch of 48.
Start with `TASK1_RUN_MODE=smoke` and require a successful forward/backward,
checkpoint reload, validation evaluation, and test evaluation before the full
run. Smoke metrics are not reportable results.

Keep the same `TASK1_RUN_ID` when resuming an interrupted full run. The notebook
writes `last_state.pt` atomically after every completed epoch and restores the
model, optimizer, scheduler, AMP scaler, history, and early-stopping state.

The full run must produce at least:

- `best_full_model.pt`;
- `best_backbone_mae_format.pth`;
- `training_history.csv`;
- GazeGene test per-subject metrics;
- MPIIFaceGaze zero-shot per-subject metrics;
- environment, resolved config, selected checkpoint, and peak GPU memory.

Then repeat the existing Task 0 LOPO experiment twice with identical folds and
hyperparameters:

1. leave `TASK0_PRETRAINED_CHECKPOINT` unset for official MAE-B16;
2. set it to `best_backbone_mae_format.pth` for GazeGene initialization.

Use separate output roots or preserve both run directories. Aggregate each LOPO
condition with `scripts/aggregate_task0_lopo.py`, then compare their
`per_subject_metrics.csv` files:

```text
python scripts/compare_task1_transfer.py \
  --mae <MAE aggregate/per_subject_metrics.csv> \
  --gazegene <GazeGene aggregate/per_subject_metrics.csv> \
  --output-dir <paired comparison output>
```

Return only aggregate metrics, configs, hashes, and logs through the relay.
Never attach dataset images, raw annotations, biometric crops, or checkpoints
to the relay issue.
