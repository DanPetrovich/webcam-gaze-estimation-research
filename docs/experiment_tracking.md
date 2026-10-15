# Experiment tracking and Kaggle workflow

## Decision

The project uses a portable, file-based experiment record as the source of
truth. Git versions code and configuration. Each training run writes an
immutable manifest plus metrics and artifacts. Kaggle is an execution platform
and artifact host, not the only experiment database.

W&B is an optional live-monitoring and comparison layer, but a run must remain
reproducible without a W&B account or Internet connection. DVC or an object-store
registry can be added if the number of datasets and checkpoints becomes
difficult to manage manually.

## What identifies one run

Every run receives a unique ID containing:

- UTC start time;
- experiment name;
- hash of the fully resolved configuration;
- hash of the source files executed by the run.

The run manifest also records the Git revision and dirty state, dataset
manifest hash, runtime information, status, and optional external identifiers
such as a Kaggle notebook slug and version. A code change therefore creates a
new run even when all hyperparameters remain unchanged.

Do not compare or resume runs only by notebook filename. Do not overwrite a
completed run directory. Do not put credentials, tokens, raw face images, or
personal paths into a manifest.

## Artifact layout

```text
outputs/<experiment>/<run_id>/
├── run_manifest.json       identity, provenance, status, external IDs
├── resolved_config.json    exact effective parameters
├── source_manifest.json    source paths, sizes, and SHA-256 hashes
├── dataset_manifest.json   split/data identity without raw images
├── summary.json            final aggregate result
├── checkpoints/            large weights; not committed to Git
├── logs/                   textual and structured training logs
├── metrics/                epoch and per-subject tables
└── plots/                  learning curves and evaluation figures
```

The entire run directory is saved as Kaggle Notebook Output. After an important
run, download the output bundle locally. Large checkpoints remain outside Git.
Small aggregate records can be indexed into `results/experiment_index.csv` and
committed after review.

## Recommended progression

1. Commit or otherwise record the exact code state before a benchmark run.
2. Run `smoke` on one fold and a small sample limit.
3. Check the dataset audit, forward/backward pass, checkpoint resume, curves,
   and final summary.
4. Run one complete fold.
5. Only then run the complete leave-one-person-out protocol.
6. Download the output bundle and rebuild the comparison index.
7. Write a short interpretation including failures and threats to validity.

Smoke runs verify plumbing only and must never be quoted as benchmark results.

## Kaggle-specific record

For a Kaggle run, record these values in `external_ids` when available:

```json
{
  "platform": "kaggle",
  "notebook_slug": "owner/notebook-name",
  "notebook_version": "3"
}
```

Kaggle notebook versioning is useful evidence, but it does not replace the
source and configuration hashes because the same notebook can depend on
different attached dataset versions.

The dataset manifest should record Kaggle input dataset slugs and versions,
subject lists, preprocessing version, file sizes, and checksums where practical.
For large normalized HDF5 files, a documented metadata hash may be used during
development; final reported experiments should use byte-level checksums.

## Building the comparison table

After installing the package in editable mode, scan any number of downloaded
output roots:

```bash
python scripts/build_experiment_index.py \
  --outputs outputs/task0_unigaze_b16 \
  --outputs downloaded_kaggle_outputs \
  --destination results/experiment_index.csv
```

The command reads every `run_manifest.json`, adds scalar fields from its
`summary.json`, and creates one row per run. Failed and cancelled runs remain in
the index; they are research evidence and should not be silently deleted.

## W&B monitoring

The Task 0 Kaggle notebook supports W&B when a Kaggle Secret named `wandb_api`
is attached. Outside Kaggle, the standard `WANDB_API_KEY` environment variable
is used. The secret value must never appear in notebook cells, committed files,
logs, manifests, or reports. When the secret is absent or W&B becomes
temporarily unavailable, local file logging and Kaggle execution continue unless
`wandb_required = true` is explicitly selected.

W&B receives batch and epoch losses, angular errors, learning rate, AMP scale,
GPU-memory use, fold summaries, final aggregate metrics, gradient histograms,
and a small aggregate evaluation artifact. It must not receive face images,
normalized-data previews, qualitative figures containing faces, checkpoints,
or per-sample prediction tables. File-based run artifacts remain the source of
truth; W&B is a dashboard, notification, and run-comparison layer.
