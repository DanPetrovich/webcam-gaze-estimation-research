# %% [markdown]
# # Task 0 — MPIIFaceGaze data audit and CPU smoke test
#
# This notebook prepares the first reproducible dataset checkpoint for the
# diploma project. It performs the following operations without requiring a GPU:
#
# 1. locates an attached MPIIFaceGaze dataset or downloads the official original
#    archive (the much larger normalized archive is opt-in);
# 2. validates the subject folders, images, 28-column annotations, calibration
#    files, and reports the availability of normalized MATLAB files;
# 3. builds a typed sample manifest and checks it for missing/non-finite values;
# 4. visualizes random faces, facial landmarks, and gaze metadata;
# 5. creates leakage-safe subject-independent development and benchmark splits;
# 6. runs a small CPU forward/backward pass through the complete image pipeline;
# 7. saves machine-readable reports under `/kaggle/working/task0_outputs`.
#
# The notebook does **not** train or benchmark a scientific model. That belongs
# to the next task after the data contract has been verified.

# %% [markdown]
# ## 1. Reproducibility and configuration
#
# MPIIFaceGaze is licensed for non-commercial scientific use. Keep the notebook
# private and do not publish or redistribute the dataset archives.
#
# When running on Kaggle, either attach the dataset as a private input or enable
# Internet access and leave `DOWNLOAD_IF_MISSING = True`. The normalized archive
# is not required for task 0 and is intentionally not downloaded on the standard
# Kaggle disk. Attach it as a private input later, or explicitly set
# `DOWNLOAD_NORMALIZED_ARCHIVE = True` in a runtime with sufficient free space.

# %%
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

SEED = 20260917
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working") if Path("/kaggle").exists() else Path.cwd()
DATA_ROOT = Path(os.environ.get("MPIIFACEGAZE_ROOT", WORK_ROOT / "task0_data"))
OUTPUT_ROOT = WORK_ROOT / "task0_outputs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

DOWNLOAD_IF_MISSING = True
DOWNLOAD_NORMALIZED_ARCHIVE = False
VERIFY_ARCHIVE_SHA256 = True
VISUAL_SAMPLES = 12
SMOKE_SAMPLES = 32
SMOKE_BATCH_SIZE = 8
CLEAN_TRANSIENT_DATA_ON_BATCH = True

OFFICIAL_ARCHIVES = {
    "MPIIFaceGaze.zip": (
        "https://collaborative-ai.org/files/datasets/MPIIFaceGaze.zip"
    ),
    "MPIIFaceGaze_normalized.zip": (
        "https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/"
        "MPIIFaceGaze_normalized.zip"
    ),
}


@dataclass(frozen=True)
class RunConfig:
    seed: int
    data_root: str
    output_root: str
    download_if_missing: bool
    download_normalized_archive: bool
    visual_samples: int
    smoke_samples: int
    clean_transient_data_on_batch: bool


RUN_CONFIG = RunConfig(
    seed=SEED,
    data_root=str(DATA_ROOT),
    output_root=str(OUTPUT_ROOT),
    download_if_missing=DOWNLOAD_IF_MISSING,
    download_normalized_archive=DOWNLOAD_NORMALIZED_ARCHIVE,
    visual_samples=VISUAL_SAMPLES,
    smoke_samples=SMOKE_SAMPLES,
    clean_transient_data_on_batch=CLEAN_TRANSIENT_DATA_ON_BATCH,
)

print(json.dumps(asdict(RUN_CONFIG), indent=2))
print(f"PyTorch: {torch.__version__}; CUDA used by task 0: False")

# %% [markdown]
# ## 2. Locate or download the official archives
#
# Downloads are streamed to disk and renamed only after completion. ZIP members
# are checked before extraction to prevent path traversal.

# %%
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "MPIIFaceGaze-academic-audit/1.0"},
    )
    print(f"Downloading {url} -> {destination}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            expected = int(response.headers.get("Content-Length", 0))
            written = 0
            with partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    written += len(chunk)
                    if expected and written % (100 * 1024 * 1024) < len(chunk):
                        print(f"  {written / expected:.1%}")
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return destination


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        zipped.extractall(destination)


def candidate_search_roots() -> list[Path]:
    roots = [DATA_ROOT]
    if KAGGLE_INPUT_ROOT.exists():
        roots.extend(path for path in KAGGLE_INPUT_ROOT.iterdir() if path.is_dir())
    return roots


def find_subject_root(roots: Iterable[Path]) -> Path | None:
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        candidates.extend(
            path.parent
            for path in root.rglob("p00.txt")
            if path.parent.name == "p00"
        )
    if not candidates:
        return None
    dataset_roots = sorted({path.parent for path in candidates})
    ranked = sorted(
        dataset_roots,
        key=lambda path: sum((path / f"p{i:02d}").is_dir() for i in range(15)),
        reverse=True,
    )
    return ranked[0]


def find_normalized_mat_files(roots: Iterable[Path]) -> list[Path]:
    files: set[Path] = set()
    expected_names = {f"p{i:02d}.mat" for i in range(15)}
    for root in roots:
        if root.exists():
            files.update(
                path.resolve()
                for path in root.rglob("*.mat")
                if path.name in expected_names
            )
    return sorted(files)


subject_root = find_subject_root(candidate_search_roots())
normalized_mat_files = find_normalized_mat_files(candidate_search_roots())
archive_records: dict[str, dict[str, object]] = {}

if DOWNLOAD_IF_MISSING and (
    subject_root is None
    or (DOWNLOAD_NORMALIZED_ARCHIVE and not normalized_mat_files)
):
    archive_root = DATA_ROOT / "archives"
    extracted_root = DATA_ROOT / "extracted"
    required_archives = []
    if subject_root is None:
        required_archives.append("MPIIFaceGaze.zip")
    if DOWNLOAD_NORMALIZED_ARCHIVE and not normalized_mat_files:
        required_archives.append("MPIIFaceGaze_normalized.zip")

    for filename in required_archives:
        url = OFFICIAL_ARCHIVES[filename]
        archive_path = archive_root / filename
        if not archive_path.exists():
            download_file(url, archive_path)
        archive_records[filename] = {
            "url": url,
            "bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path) if VERIFY_ARCHIVE_SHA256 else None,
        }
        marker = extracted_root / f".{filename}.extracted"
        if not marker.exists():
            safe_extract_zip(archive_path, extracted_root)
            marker.touch()
    subject_root = subject_root or find_subject_root([extracted_root])
    normalized_mat_files = find_normalized_mat_files(
        candidate_search_roots() + [extracted_root]
    )

if subject_root is None:
    raise FileNotFoundError(
        "MPIIFaceGaze was not found. Attach a private Kaggle input containing "
        "p00/.../p00.txt, or enable Internet and keep DOWNLOAD_IF_MISSING=True."
    )

print(f"MPIIFaceGaze subject root: {subject_root}")
print(
    "Normalized subject files: "
    f"{len(normalized_mat_files)}/15 "
    "(optional for task 0; attach as a private input for normalized-model work)"
)
if archive_records:
    print(json.dumps(archive_records, indent=2))

# %% [markdown]
# ## 3. Parse the original 28-column annotation format
#
# Coordinate contract used below:
#
# - `screen_x_px`, `screen_y_px`: target location in screen pixels;
# - `face_center_*`: face centre in camera coordinates;
# - `gaze_target_*`: 3D target in camera coordinates;
# - `gaze_*`: unit vector **from face centre to target**;
# - `landmark_*`: four eye corners and two mouth corners in image pixels.
#
# No yaw/pitch sign convention is imposed at this stage. The manifest retains
# the explicit 3D vector to avoid silently mixing conventions between papers.

# %%
LANDMARK_NAMES = (
    "right_eye_outer",
    "right_eye_inner",
    "left_eye_inner",
    "left_eye_outer",
    "mouth_right",
    "mouth_left",
)

MANIFEST_COLUMNS = [
    "subject",
    "image_rel",
    "image_path",
    "screen_x_px",
    "screen_y_px",
    *[
        f"landmark_{name}_{axis}"
        for name in LANDMARK_NAMES
        for axis in ("x", "y")
    ],
    "head_rvec_x",
    "head_rvec_y",
    "head_rvec_z",
    "head_tvec_x",
    "head_tvec_y",
    "head_tvec_z",
    "face_center_x",
    "face_center_y",
    "face_center_z",
    "gaze_target_x",
    "gaze_target_y",
    "gaze_target_z",
    "evaluation_eye",
    "gaze_x",
    "gaze_y",
    "gaze_z",
]


def resolve_image_path(subject_dir: Path, image_rel: str) -> Path:
    relative = Path(image_rel.replace("\\", "/"))
    candidates = (
        subject_dir / relative,
        subject_dir.parent / relative,
        subject_dir / relative.name,
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def parse_subject_annotations(subject_dir: Path) -> Iterator[dict[str, object]]:
    annotation_path = subject_dir / f"{subject_dir.name}.txt"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

    with annotation_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            tokens = line.split()
            if not tokens:
                continue
            if len(tokens) != 28:
                raise ValueError(
                    f"{annotation_path}:{line_number}: expected 28 columns, "
                    f"found {len(tokens)}"
                )
            numeric = np.asarray(tokens[1:27], dtype=np.float64)
            face_center = numeric[20:23]
            gaze_target = numeric[23:26]
            gaze = gaze_target - face_center
            norm = float(np.linalg.norm(gaze))
            gaze_unit = gaze / norm if norm > 0.0 else np.full(3, np.nan)
            image_path = resolve_image_path(subject_dir, tokens[0])
            values: list[object] = [
                subject_dir.name,
                tokens[0],
                str(image_path),
                *numeric.tolist(),
                tokens[27],
                *gaze_unit.tolist(),
            ]
            yield dict(zip(MANIFEST_COLUMNS, values, strict=True))


subject_dirs = sorted(
    path
    for path in subject_root.iterdir()
    if path.is_dir() and path.name.startswith("p") and path.name[1:].isdigit()
)
manifest = pd.DataFrame(
    row
    for subject_dir in subject_dirs
    for row in parse_subject_annotations(subject_dir)
)

manifest_path = OUTPUT_ROOT / "mpiifacegaze_manifest.csv"
manifest.to_csv(manifest_path, index=False)
print(f"Manifest: {len(manifest):,} rows, {len(manifest.columns)} columns")
print(f"Saved: {manifest_path}")

# %% [markdown]
# ## 4. Structural and numerical audit
#
# Fatal problems stop the notebook. Non-fatal findings are recorded in the JSON
# report so later experiments can state exactly which data version was used.

# %%
def image_probe(paths: Sequence[str]) -> tuple[list[str], list[dict[str, object]]]:
    failures: list[str] = []
    metadata: list[dict[str, object]] = []
    for value in paths:
        path = Path(value)
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                metadata.append(
                    {
                        "path": str(path),
                        "width": image.width,
                        "height": image.height,
                        "mode": image.mode,
                    }
                )
        except Exception as error:
            failures.append(f"{path}: {type(error).__name__}: {error}")
    return failures, metadata


numeric_columns = manifest.select_dtypes(include=[np.number]).columns
missing_images = manifest.loc[
    ~manifest["image_path"].map(lambda value: Path(value).is_file()),
    "image_path",
].tolist()
non_finite_rows = int((~np.isfinite(manifest[numeric_columns])).any(axis=1).sum())
duplicate_images = int(manifest["image_path"].duplicated().sum())
gaze_norms = np.linalg.norm(manifest[["gaze_x", "gaze_y", "gaze_z"]], axis=1)
invalid_gaze_norms = int((~np.isclose(gaze_norms, 1.0, atol=1e-6)).sum())

probe_count = min(300, len(manifest))
probe_paths = manifest.sample(probe_count, random_state=SEED)["image_path"].tolist()
corrupt_images, image_metadata = image_probe(probe_paths)
image_modes = pd.Series([item["mode"] for item in image_metadata]).value_counts().to_dict()
image_sizes = (
    pd.DataFrame(image_metadata)[["width", "height"]]
    .value_counts()
    .rename("count")
    .reset_index()
    .to_dict(orient="records")
    if image_metadata
    else []
)

calibration_counts = {
    subject_dir.name: len(list(subject_dir.rglob("*.mat")))
    for subject_dir in subject_dirs
}
subject_counts = manifest.groupby("subject").size().sort_index()
audit_report = {
    "config": asdict(RUN_CONFIG),
    "subject_root": str(subject_root),
    "archive_records": archive_records,
    "subjects": subject_counts.to_dict(),
    "subject_count": int(manifest["subject"].nunique()),
    "sample_count": int(len(manifest)),
    "missing_image_count": len(missing_images),
    "missing_image_examples": missing_images[:20],
    "non_finite_row_count": non_finite_rows,
    "duplicate_image_count": duplicate_images,
    "invalid_unit_gaze_count": invalid_gaze_norms,
    "probed_image_count": probe_count,
    "corrupt_probe_count": len(corrupt_images),
    "corrupt_probe_examples": corrupt_images[:20],
    "image_modes": image_modes,
    "image_sizes": image_sizes,
    "calibration_mat_counts": calibration_counts,
    "normalized_mat_count": len(normalized_mat_files),
    "normalized_data_required": DOWNLOAD_NORMALIZED_ARCHIVE,
}

report_path = OUTPUT_ROOT / "task0_audit_report.json"
report_path.write_text(json.dumps(audit_report, indent=2), encoding="utf-8")

fatal_findings = {
    "unexpected_subject_count": audit_report["subject_count"] != 15,
    "missing_images": bool(missing_images),
    "non_finite_rows": non_finite_rows > 0,
    "invalid_unit_gaze": invalid_gaze_norms > 0,
    "corrupt_probed_images": bool(corrupt_images),
    "missing_calibration_files": any(
        count < 3 for count in calibration_counts.values()
    ),
}

non_fatal_findings = {
    "normalized_subject_files_not_attached": len(normalized_mat_files) < 15,
}

display(subject_counts.rename("samples").to_frame())
display(pd.DataFrame(image_sizes).head(10))
print(json.dumps(fatal_findings, indent=2))
print(json.dumps(non_fatal_findings, indent=2))
print(f"Saved: {report_path}")

if any(fatal_findings.values()):
    raise RuntimeError(
        "Dataset audit found fatal issues. Inspect task0_audit_report.json before "
        "continuing. No split or smoke-test result should be trusted yet."
    )

# %% [markdown]
# ## 5. Qualitative image and annotation inspection
#
# The plot overlays the six released facial landmarks. The displayed 3D vector
# is the normalized face-centre-to-target direction in camera coordinates.

# %%
sample_rows = manifest.sample(
    min(VISUAL_SAMPLES, len(manifest)),
    random_state=SEED,
).reset_index(drop=True)

columns = 4
rows = math.ceil(len(sample_rows) / columns)
figure, axes = plt.subplots(rows, columns, figsize=(16, 4 * rows), squeeze=False)

for axis, (_, sample) in zip(axes.flat, sample_rows.iterrows(), strict=False):
    with Image.open(sample["image_path"]) as image:
        axis.imshow(image.convert("RGB"))
    for name in LANDMARK_NAMES:
        axis.scatter(
            sample[f"landmark_{name}_x"],
            sample[f"landmark_{name}_y"],
            s=18,
        )
    gaze = sample[["gaze_x", "gaze_y", "gaze_z"]].to_numpy(dtype=float)
    axis.set_title(
        f"{sample['subject']} | target=({sample['screen_x_px']:.0f}, "
        f"{sample['screen_y_px']:.0f}) px\n"
        f"gaze=[{gaze[0]:+.2f}, {gaze[1]:+.2f}, {gaze[2]:+.2f}]"
    )
    axis.axis("off")

for axis in axes.flat[len(sample_rows) :]:
    axis.axis("off")

figure.suptitle("MPIIFaceGaze qualitative audit", fontsize=16)
figure.tight_layout()
grid_path = OUTPUT_ROOT / "qualitative_audit.png"
figure.savefig(grid_path, dpi=160, bbox_inches="tight")
plt.show()
print(f"Saved: {grid_path}")

# %% [markdown]
# ## 6. Label distributions
#
# These yaw/pitch values are audit-only derived quantities. Definitions are
# explicit below and must not be assumed to match a model repository until its
# coordinate convention has been checked:
#
# - `yaw = atan2(gaze_x, -gaze_z)` because forward gaze points along `-Z`;
# - `pitch = asin(gaze_y)`;
# - angles are reported in degrees.

# %%
gaze_matrix = manifest[["gaze_x", "gaze_y", "gaze_z"]].to_numpy(dtype=np.float64)
manifest["audit_yaw_deg"] = np.degrees(
    np.arctan2(gaze_matrix[:, 0], -gaze_matrix[:, 2])
)
manifest["audit_pitch_deg"] = np.degrees(np.arcsin(np.clip(gaze_matrix[:, 1], -1.0, 1.0)))

figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].hist(manifest["audit_yaw_deg"], bins=60)
axes[0].set(title="Audit yaw distribution", xlabel="degrees", ylabel="samples")
axes[1].hist(manifest["audit_pitch_deg"], bins=60)
axes[1].set(title="Audit pitch distribution", xlabel="degrees", ylabel="samples")
figure.tight_layout()
distribution_path = OUTPUT_ROOT / "gaze_angle_distributions.png"
figure.savefig(distribution_path, dpi=160, bbox_inches="tight")
plt.show()

display(
    manifest[["audit_yaw_deg", "audit_pitch_deg"]]
    .describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
    .round(3)
)

# %% [markdown]
# ## 7. Leakage-safe subject-independent splits
#
# Two protocols are exported:
#
# - `development`: 13 train subjects, one validation subject, one test subject;
# - `benchmark`: 14 train subjects and one test subject. Use this only after all
#   hyperparameters and stopping rules have been fixed using development runs.
#
# Each of the 15 subjects becomes the test subject exactly once.

# %%
def build_lopo_splits(subjects: Sequence[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    ordered = sorted(subjects)
    for fold_index, test_subject in enumerate(ordered):
        validation_subject = ordered[(fold_index + 1) % len(ordered)]
        for subject in ordered:
            development_role = (
                "test"
                if subject == test_subject
                else "validation"
                if subject == validation_subject
                else "train"
            )
            benchmark_role = "test" if subject == test_subject else "train"
            records.extend(
                [
                    {
                        "protocol": "development",
                        "fold": fold_index,
                        "subject": subject,
                        "role": development_role,
                    },
                    {
                        "protocol": "benchmark",
                        "fold": fold_index,
                        "subject": subject,
                        "role": benchmark_role,
                    },
                ]
            )
    return pd.DataFrame.from_records(records)


splits = build_lopo_splits(manifest["subject"].unique().tolist())
split_path = OUTPUT_ROOT / "subject_independent_splits.csv"
splits.to_csv(split_path, index=False)

for (protocol, fold), group in splits.groupby(["protocol", "fold"]):
    role_sets = {
        role: set(role_group["subject"])
        for role, role_group in group.groupby("role")
    }
    roles = list(role_sets)
    for left_index, left_role in enumerate(roles):
        for right_role in roles[left_index + 1 :]:
            overlap = role_sets[left_role] & role_sets[right_role]
            if overlap:
                raise AssertionError(
                    f"Identity leakage in {protocol} fold {fold}: {overlap}"
                )

display(
    splits.groupby(["protocol", "fold", "role"])["subject"]
    .nunique()
    .rename("subject_count")
    .head(12)
    .to_frame()
)
print(f"Saved: {split_path}")

# %% [markdown]
# ## 8. CPU smoke test of the image-to-gaze training path
#
# This deliberately tiny CNN is not a baseline and its loss is not a research
# result. One optimizer step verifies image decoding, resizing, batching, model
# output shape, angular loss, backward propagation, and checkpoint serialization.

# %%
class MPIIFaceGazeDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, frame: pd.DataFrame, image_size: int = 96) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as image:
            rgb = image.convert("RGB").resize(
                (self.image_size, self.image_size),
                Image.Resampling.BILINEAR,
            )
            pixels = np.asarray(rgb, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
        gaze_tensor = torch.tensor(
            [row["gaze_x"], row["gaze_y"], row["gaze_z"]],
            dtype=torch.float32,
        )
        return image_tensor, gaze_tensor


class TinyGazeSmokeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(16, 3)

    def forward(self, images: Tensor) -> Tensor:
        features = self.features(images).flatten(1)
        return nn.functional.normalize(self.head(features), dim=1)


def angular_loss_degrees(prediction: Tensor, target: Tensor) -> Tensor:
    prediction = nn.functional.normalize(prediction, dim=1)
    target = nn.functional.normalize(target, dim=1)
    cosine = (prediction * target).sum(dim=1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.rad2deg(torch.acos(cosine)).mean()


smoke_frame = manifest.sample(
    min(SMOKE_SAMPLES, len(manifest)),
    random_state=SEED,
)
smoke_loader = DataLoader(
    MPIIFaceGazeDataset(smoke_frame),
    batch_size=SMOKE_BATCH_SIZE,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
    num_workers=0,
)
smoke_model = TinyGazeSmokeModel().cpu()
optimizer = torch.optim.Adam(smoke_model.parameters(), lr=1e-3)

images, targets = next(iter(smoke_loader))
optimizer.zero_grad(set_to_none=True)
predictions = smoke_model(images)
loss = angular_loss_degrees(predictions, targets)
loss.backward()
optimizer.step()

checkpoint_path = OUTPUT_ROOT / "task0_smoke_checkpoint.pt"
torch.save(
    {
        "model_state_dict": smoke_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "seed": SEED,
        "note": "Infrastructure smoke test only; not a scientific baseline.",
    },
    checkpoint_path,
)

smoke_result = {
    "batch_shape": list(images.shape),
    "target_shape": list(targets.shape),
    "prediction_shape": list(predictions.shape),
    "finite_loss": bool(torch.isfinite(loss).item()),
    "loss_degrees_before_one_update": float(loss.detach()),
    "checkpoint": str(checkpoint_path),
}
(OUTPUT_ROOT / "task0_smoke_result.json").write_text(
    json.dumps(smoke_result, indent=2),
    encoding="utf-8",
)
print(json.dumps(smoke_result, indent=2))

if not smoke_result["finite_loss"]:
    raise RuntimeError("Smoke test produced a non-finite loss")

# %% [markdown]
# ## 9. Completion summary
#
# Task 0 is complete only if this cell reports `PASS`. Before model training,
# manually inspect `qualitative_audit.png` and document any suspicious samples.

# %%
required_outputs = [
    manifest_path,
    report_path,
    grid_path,
    distribution_path,
    split_path,
    checkpoint_path,
    OUTPUT_ROOT / "task0_smoke_result.json",
]
missing_outputs = [str(path) for path in required_outputs if not path.is_file()]

completion = {
    "status": "PASS" if not missing_outputs else "FAIL",
    "dataset_samples": len(manifest),
    "subjects": int(manifest["subject"].nunique()),
    "required_outputs": [str(path) for path in required_outputs],
    "missing_outputs": missing_outputs,
    "manual_follow_up": (
        "Inspect qualitative_audit.png and record suspicious samples before "
        "starting FullFace or UniGaze training."
    ),
}
if missing_outputs:
    raise RuntimeError(f"Task 0 outputs are incomplete: {missing_outputs}")

# A Kaggle batch commit saves everything under /kaggle/working as version output.
# Remove only the transient copy downloaded by this notebook, never an attached
# /kaggle/input dataset. This avoids persisting or redistributing face images and
# keeps the committed output small; the manifest and audit artifacts remain.
batch_run = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() == "batch"
data_root_resolved = DATA_ROOT.resolve()
working_root_resolved = WORK_ROOT.resolve()
subject_root_resolved = subject_root.resolve()
remove_transient_data = (
    CLEAN_TRANSIENT_DATA_ON_BATCH
    and batch_run
    and data_root_resolved.is_relative_to(working_root_resolved)
    and subject_root_resolved.is_relative_to(data_root_resolved)
)
if remove_transient_data:
    shutil.rmtree(data_root_resolved)

completion["transient_download_removed"] = remove_transient_data
(OUTPUT_ROOT / "task0_completion.json").write_text(
    json.dumps(completion, indent=2),
    encoding="utf-8",
)
print(json.dumps(completion, indent=2))
