# %% [markdown]
# # UniGaze-B16 linear probe on MPIIFaceGaze
#
# This is **not** the paper's full fine-tuning experiment. The MAE-pretrained
# ViT is frozen; only a two-output linear layer is trained. We use the authors'
# normalization and dataset loader, then evaluate with subject-independent
# leave-one-person-out (LOPO) folds. The reported numbers must not be presented
# as a reproduction of UniGaze's published 4–5 degree result.
#
# Run as a private Kaggle notebook with GPU and Internet enabled. Raw face data,
# normalized face crops and the MAE checkpoint stay in `/kaggle/temp`, not in
# published notebook outputs. Only aggregate metrics and tiny probe heads are
# written to `/kaggle/working/unigaze_linear_probe`.

# %% [markdown]
# ## 1. Configuration and environment
#
# The first run downloads about 1 GB of source images, the authors' checkpoint,
# and the dependencies for their normalization script. Later runs can attach
# already-normalized private HDF5 data and/or a checkpoint as Kaggle inputs.

# %%
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SEED = 20260920
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    raise RuntimeError("This notebook requires a Kaggle GPU for preprocessing and feature extraction.")

ROOT = Path("/kaggle/temp/unigaze_probe") if Path("/kaggle").exists() else Path("/tmp/unigaze_probe")
OUTPUT = Path("/kaggle/working/unigaze_linear_probe") if Path("/kaggle").exists() else Path.cwd() / "unigaze_linear_probe"
INPUT = Path("/kaggle/input")
ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

SUBJECTS = [f"p{i:02d}" for i in range(15)]
FEATURE_BATCH_SIZE = 32
MAX_EPOCHS = 120
PATIENCE = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
MAE_B16_FOLDER = "https://drive.google.com/drive/folders/1vz38f90jPrMwb_lByzJfaMgH6BEtB49f"
RAW_URL = "https://collaborative-ai.org/files/datasets/MPIIFaceGaze.zip"

print({"device": str(DEVICE), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__})


def run(command: list[str], cwd: Path | None = None) -> None:
    """Run an external command and stop immediately on failure."""
    subprocess.run(command, cwd=cwd, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# %% [markdown]
# ## 2. Get the official UniGaze code and MAE-B16 encoder checkpoint
#
# This downloads the *pretrained encoder*, not an already gaze-trained model.
# The decoder from MAE pretraining is discarded by the authors' `MAE_Gaze` class.

# %%
run([sys.executable, "-m", "pip", "install", "-q", "timm==1.0.9", "face-alignment==1.4.1", "rich", "gdown", "omegaconf"])
REPO = ROOT / "UniGaze"
if not (REPO / ".git").exists():
    run(["git", "clone", "--depth", "1", "https://github.com/ut-vision/UniGaze.git", str(REPO)])
repo_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
sys.path.insert(0, str(REPO / "unigaze"))

checkpoint_candidates = list(INPUT.rglob("mae_b16_checkpoint-299.pth")) if INPUT.exists() else []
if not checkpoint_candidates:
    checkpoint_dir = ROOT / "checkpoint"
    checkpoint_dir.mkdir(exist_ok=True)
    import gdown

    gdown.download_folder(MAE_B16_FOLDER, output=str(checkpoint_dir), quiet=False, use_cookies=False)
    checkpoint_candidates = list(checkpoint_dir.rglob("mae_b16_checkpoint-299.pth"))
if len(checkpoint_candidates) != 1:
    raise FileNotFoundError(f"Expected exactly one official MAE-B16 checkpoint, found {len(checkpoint_candidates)}")
CHECKPOINT = checkpoint_candidates[0]
if CHECKPOINT.stat().st_size < 10_000_000:
    raise ValueError("Checkpoint is implausibly small; download may have failed")
print({"repo_revision": repo_revision, "checkpoint_bytes": CHECKPOINT.stat().st_size, "checkpoint_sha256": sha256(CHECKPOINT)})

# %% [markdown]
# ## 3. Prepare normalized MPIIFaceGaze with the authors' code
#
# The earlier Task 0 notebook audited the *original* images. UniGaze expects
# aligned 224×224 face patches and gaze labels transformed into the normalized
# camera frame. A simple crop or resize of Task 0's originals is not equivalent.
# `normalize_mpii.py` runs landmark detection and gaze normalization. This stage
# can take a substantial fraction of a Kaggle session.

# %%
def find_complete_h5() -> Path | None:
    roots = [ROOT / "normalized"]
    if INPUT.exists():
        roots.extend(path for path in INPUT.iterdir() if path.is_dir())
    for root in roots:
        if root.exists():
            candidates = [root, *root.glob("*"), *root.glob("*/*")]
            for candidate in candidates:
                if all((candidate / f"{subject}.h5").is_file() for subject in SUBJECTS):
                    return candidate
    return None


def find_raw_root() -> Path | None:
    roots = [ROOT / "raw"]
    if INPUT.exists():
        roots.extend(path for path in INPUT.iterdir() if path.is_dir())
    for root in roots:
        if root.exists():
            candidates = [root, *root.glob("*"), *root.glob("*/*")]
            for candidate in candidates:
                if all((candidate / subject / f"{subject}.txt").is_file() for subject in SUBJECTS):
                    return candidate
    return None


def download_raw_archive() -> Path:
    archive = ROOT / "MPIIFaceGaze.zip"
    if not archive.is_file():
        partial = archive.with_suffix(".zip.part")
        try:
            urllib.request.urlretrieve(RAW_URL, partial)
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
    raw_dir = ROOT / "raw"
    raw_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        root_resolved = raw_dir.resolve()
        for member in zipped.infolist():
            target = (raw_dir / member.filename).resolve()
            if target != root_resolved and root_resolved not in target.parents:
                raise ValueError(f"Unsafe archive path: {member.filename}")
        zipped.extractall(raw_dir)
    result = find_raw_root()
    if result is None:
        raise FileNotFoundError("The official archive did not contain all 15 MPIIFaceGaze subjects")
    return result


H5_ROOT = find_complete_h5()
raw_sha256 = None
if H5_ROOT is None:
    RAW_ROOT = find_raw_root() or download_raw_archive()
    archive = ROOT / "MPIIFaceGaze.zip"
    if archive.is_file():
        raw_sha256 = sha256(archive)
    H5_ROOT = ROOT / "normalized"
    run(
        [sys.executable, "normalize_mpii.py", "--mpiifacegaze_raw_dir", str(RAW_ROOT), "--output_dir", str(H5_ROOT)],
        cwd=REPO / "gazedata_preparation",
    )
if not all((H5_ROOT / f"{subject}.h5").is_file() for subject in SUBJECTS):
    raise RuntimeError("Normalization did not produce all 15 subject files")
print({"normalized_subjects": len(SUBJECTS), "raw_archive_sha256": raw_sha256})

# %% [markdown]
# ## 4. Audit a normalized sample and freeze the ViT
#
# The official loader converts BGR to RGB and applies ImageNet normalization.
# We explicitly verify `requires_grad=False` for every encoder parameter.

# %%
from datasets.mpiigaze import MPIIGazeDataset
from models.vit.mae_gaze import MAE_Gaze
from torch.utils.data import DataLoader

probe_dataset = MPIIGazeDataset(dataset_path=str(H5_ROOT), color_type="bgr", keys_to_use=["p00.h5"], data_name="mpii", image_size=224)
example = probe_dataset[0]
assert tuple(example["image"].shape) == (3, 224, 224)
assert np.asarray(example["gaze"]).shape == (2,)
assert np.isfinite(np.asarray(example["gaze"])).all()
print({"sample_image_shape": list(example["image"].shape), "sample_gaze_radians": np.asarray(example["gaze"]).tolist()})

model = MAE_Gaze(model_type="vit_b_16", global_pool=False, custom_pretrained_path=str(CHECKPOINT)).to(DEVICE)
for parameter in model.vit.parameters():
    parameter.requires_grad_(False)
model.vit.eval()
assert not any(parameter.requires_grad for parameter in model.vit.parameters())
print({"encoder_trainable_parameters": 0, "feature_dimension": model.vit.embed_dim})

# %% [markdown]
# ## 5. Extract frozen features once for all 15 people
#
# No test person's labels are used when fitting a head. Extracting its *fixed*
# features is allowed: no weight updates or normalization statistics are fit
# during this pass. Features and images stay under `/kaggle/temp`.

# %%
FEATURE_DIR = ROOT / "features"
FEATURE_DIR.mkdir(exist_ok=True)
feature_files: dict[str, Path] = {}

for subject in SUBJECTS:
    destination = FEATURE_DIR / f"{subject}.npz"
    feature_files[subject] = destination
    if destination.is_file():
        continue
    dataset = MPIIGazeDataset(dataset_path=str(H5_ROOT), color_type="bgr", keys_to_use=[f"{subject}.h5"], data_name="mpii", image_size=224)
    loader = DataLoader(dataset, batch_size=FEATURE_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    features, labels = [], []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(DEVICE, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                representation = model.vit.forward_features(image)
            features.append(representation.float().cpu().numpy())
            labels.append(batch["gaze"].float().numpy())
    x = np.concatenate(features).astype(np.float32)
    y = np.concatenate(labels).astype(np.float32)
    assert x.shape == (len(dataset), model.vit.embed_dim)
    assert y.shape == (len(dataset), 2)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    np.savez_compressed(destination, features=x, gaze=y)
    print(subject, len(dataset))

# Release the large encoder before training the tiny heads.
del model
torch.cuda.empty_cache()

# %% [markdown]
# ## 6. LOPO: train only a linear head; select epoch on a separate person
#
# For each test person, the next person is validation; 13 remaining people are
# used for epoch selection. Then we reinitialize and refit on all 14 non-test
# people for exactly the selected number of epochs. We also report a constant
# median-angle baseline fitted on the same 14 subjects. The test person never
# influences epoch selection, weights, or baseline statistics.

# %%
arrays = {subject: np.load(path) for subject, path in feature_files.items()}


def stack_subjects(subjects: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    x = np.concatenate([arrays[subject]["features"] for subject in subjects])
    y = np.concatenate([arrays[subject]["gaze"] for subject in subjects])
    return torch.from_numpy(x).to(DEVICE), torch.from_numpy(y).to(DEVICE)


def angular_error_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Great-circle angle from [pitch, yaw] in radians; independent of axis signs."""
    pitch_a, yaw_a = pred[:, 0], pred[:, 1]
    pitch_b, yaw_b = target[:, 0], target[:, 1]
    cosine = torch.sin(pitch_a) * torch.sin(pitch_b) + torch.cos(pitch_a) * torch.cos(pitch_b) * torch.cos(yaw_a - yaw_b)
    return torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))


def new_head(seed: int) -> torch.nn.Linear:
    torch.manual_seed(seed)
    return torch.nn.Linear(model_feature_dim, 2).to(DEVICE)


def fit_head(train_x: torch.Tensor, train_y: torch.Tensor, epochs: int, seed: int,
             validation: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.nn.Linear, int]:
    head = new_head(seed)
    optimizer = torch.optim.Adam(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_epoch, best_validation, stale = epochs, float("inf"), 0
    for epoch in range(1, epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.l1_loss(head(train_x), train_y)
        loss.backward()
        optimizer.step()
        if validation is not None:
            head.eval()
            with torch.inference_mode():
                value = torch.nn.functional.l1_loss(head(validation[0]), validation[1]).item()
            if value < best_validation - 1e-6:
                best_validation, best_epoch, stale = value, epoch, 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
    return head, best_epoch


model_feature_dim = int(arrays[SUBJECTS[0]]["features"].shape[1])
rows = []
for fold, test_subject in enumerate(SUBJECTS):
    validation_subject = SUBJECTS[(fold + 1) % len(SUBJECTS)]
    development_subjects = [subject for subject in SUBJECTS if subject not in {test_subject, validation_subject}]
    assert len(development_subjects) == 13
    dev_x, dev_y = stack_subjects(development_subjects)
    val_x, val_y = stack_subjects([validation_subject])
    _, chosen_epochs = fit_head(dev_x, dev_y, MAX_EPOCHS, SEED + fold, (val_x, val_y))
    del dev_x, dev_y, val_x, val_y

    train_subjects = [subject for subject in SUBJECTS if subject != test_subject]
    train_x, train_y = stack_subjects(train_subjects)
    test_x, test_y = stack_subjects([test_subject])
    head, _ = fit_head(train_x, train_y, chosen_epochs, SEED + fold)
    head.eval()
    with torch.inference_mode():
        errors = angular_error_deg(head(test_x), test_y).cpu().numpy()
        baseline_angles = torch.median(train_y, dim=0).values.expand_as(test_y)
        baseline = angular_error_deg(baseline_angles, test_y).cpu().numpy()
    torch.save({"weight": head.weight.detach().cpu(), "bias": head.bias.detach().cpu(),
                "test_subject": test_subject, "epochs": chosen_epochs}, OUTPUT / f"head_{test_subject}.pt")
    rows.append({"test_subject": test_subject, "validation_subject": validation_subject,
                 "train_samples": len(train_y), "test_samples": len(test_y), "epochs": chosen_epochs,
                 "mean_deg": float(errors.mean()), "median_deg": float(np.median(errors)),
                 "p90_deg": float(np.percentile(errors, 90)),
                 "constant_baseline_mean_deg": float(baseline.mean())})
    print(rows[-1])
    del train_x, train_y, test_x, test_y, head

# %% [markdown]
# ## 7. Save the measured result and provenance
#
# The mean below is the average of 15 subject means (not a pooled image mean).
# Compare it only with experiments using the same normalized data and LOPO
# protocol. It is not the published UniGaze full-fine-tuning result.

# %%
per_subject = pd.DataFrame(rows)
per_subject.to_csv(OUTPUT / "per_subject_metrics.csv", index=False)
summary = {
    "experiment": "frozen UniGaze MAE-B16 + trainable 2D linear head",
    "protocol": "15-fold LOPO; 13 train/1 validation for epoch selection, refit on 14 train",
    "subject_mean_angular_error_deg": float(per_subject["mean_deg"].mean()),
    "subject_std_angular_error_deg": float(per_subject["mean_deg"].std(ddof=1)),
    "subject_mean_constant_baseline_deg": float(per_subject["constant_baseline_mean_deg"].mean()),
    "subject_count": len(SUBJECTS),
    "sample_count": int(sum(len(arrays[subject]["gaze"]) for subject in SUBJECTS)),
    "seed": SEED,
    "repo_revision": repo_revision,
    "checkpoint_sha256": sha256(CHECKPOINT),
    "raw_archive_sha256": raw_sha256,
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(),
    "feature_batch_size": FEATURE_BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
display(per_subject)
