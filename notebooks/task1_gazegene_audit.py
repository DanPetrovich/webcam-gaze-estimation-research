# %% [markdown]
# # Task 1 — GazeGene dataset audit and UniGaze compatibility smoke test
#
# This Kaggle-ready notebook validates an **already extracted, privately attached**
# GazeGene dataset. It never downloads or redistributes GazeGene. Before running:
#
# 1. accept the official GazeGene access and license conditions;
# 2. attach an extracted GazeGene directory as a private Kaggle input;
# 3. optionally set `GAZEGENE_ROOT` to that directory.
#
# The notebook supports both `GazeGene_FaceCrops` (`labels/`, `imgs/`) and the
# geometrically normalized release (`labels_normalized/`, `imgs_normalized/`).
# It records the detected variant because results from the two preprocessing
# paths are not directly interchangeable.
#
# Outputs contain only aggregate metadata, relative paths, plots, and a small
# label-only smoke manifest. Dataset images and pickle annotations remain in the
# read-only private input. GazeGene is distributed under CC BY-NC-SA 4.0 with
# additional **No Distribution** terms. Load pickle files only from the official
# dataset: Python pickle is unsafe for untrusted inputs.

# %%
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


@dataclass(frozen=True)
class AuditConfig:
    dataset_root: str | None = os.environ.get("GAZEGENE_ROOT")
    variant: str = os.environ.get("GAZEGENE_VARIANT", "auto")
    output_root: str = (
        "/kaggle/working/task1_gazegene_audit"
        if Path("/kaggle").exists()
        else str(Path.cwd() / "outputs" / "task1_gazegene_audit")
    )
    seed: int = int(os.environ.get("GAZEGENE_AUDIT_SEED", "20261112"))
    expected_subjects: int = 56
    expected_cameras: int = 9
    expected_frames_per_camera: int = 2_000
    expected_total_samples: int = 1_008_000
    project_train_subjects: tuple[int, ...] = tuple(range(1, 41))
    project_validation_subjects: tuple[int, ...] = tuple(range(41, 47))
    project_test_subjects: tuple[int, ...] = tuple(range(47, 57))
    smoke_samples: int = int(os.environ.get("GAZEGENE_SMOKE_SAMPLES", "512"))
    montage_images: int = int(os.environ.get("GAZEGENE_MONTAGE_IMAGES", "12"))
    gaze_norm_tolerance: float = 1e-4


CONFIG = AuditConfig()
if CONFIG.variant not in {"auto", "face_crops", "normalized"}:
    raise ValueError("GAZEGENE_VARIANT must be auto, face_crops, or normalized")

OUTPUT_ROOT = Path(CONFIG.output_root).expanduser().resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(CONFIG.seed)
SOURCE_SHA256 = "__SOURCE_SHA256__"
if SOURCE_SHA256.startswith("__"):
    # The percent-to-notebook converter replaces the placeholder in .ipynb.
    # Direct .py executions compute the same provenance value at runtime.
    SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SUBJECT_PATTERN = re.compile(r"^subject0*(\d+)$", re.IGNORECASE)
CAMERA_PATTERN = re.compile(r"camera0*(\d+)", re.IGNORECASE)
GAZE_LABEL_PATTERN = "gaze_label_camera*.pkl"
REQUIRED_GAZE_KEYS = {
    "gaze_C",
    "visual_axis_L",
    "visual_axis_R",
    "optic_axis_L",
    "optic_axis_R",
    "gaze_target",
    "gaze_depth",
    "head_R_mat",
    "head_T_vec",
}

print(json.dumps(asdict(CONFIG), indent=2))


# %% [markdown]
# ## 1. Coordinate conversion used by the project
#
# GazeGene defines `gaze_C` from the head centre toward the gaze target. The
# ETH-XGaze/UniGaze HDF5 convention used by Task 0 stores `[pitch, yaw]` after
# negating that target-pointing vector. Therefore the compatibility conversion
# below is explicitly `vector_to_pitchyaw(-gaze_C)`.
#
# This cell verifies the mathematical round trip. It does not replace the later
# empirical checks of GazeGene's actual distributions and image orientation.

# %%
def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Expected [N, 3] vectors, got {array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("Vectors contain non-finite or zero norms")
    return array / norms


def gazegene_vector_to_unigaze_pitchyaw(vectors: np.ndarray) -> np.ndarray:
    """Convert head-to-target GazeGene vectors to UniGaze [pitch, yaw]."""
    unigaze_vectors = -normalize_vectors(vectors)
    pitch = np.arcsin(np.clip(unigaze_vectors[:, 1], -1.0, 1.0))
    yaw = np.arctan2(unigaze_vectors[:, 0], unigaze_vectors[:, 2])
    return np.column_stack((pitch, yaw))


def unigaze_pitchyaw_to_vector(pitchyaw: np.ndarray) -> np.ndarray:
    """Convert UniGaze [pitch, yaw] radians to its unit-vector convention."""
    angles = np.asarray(pitchyaw, dtype=np.float64)
    if angles.ndim != 2 or angles.shape[1] != 2:
        raise ValueError(f"Expected [N, 2] pitch/yaw, got {angles.shape}")
    pitch, yaw = angles[:, 0], angles[:, 1]
    return np.column_stack(
        (
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
            np.cos(pitch) * np.cos(yaw),
        )
    )


synthetic_vectors = normalize_vectors(
    np.array(
        [
            [0.0, 0.0, -1.0],
            [0.2, -0.1, -1.0],
            [-0.4, 0.3, -1.0],
        ]
    )
)
synthetic_angles = gazegene_vector_to_unigaze_pitchyaw(synthetic_vectors)
synthetic_roundtrip = unigaze_pitchyaw_to_vector(synthetic_angles)
np.testing.assert_allclose(synthetic_roundtrip, -synthetic_vectors, atol=1e-12)
print("Coordinate conversion round trip: PASS")


# %% [markdown]
# ## 2. Locate an extracted private GazeGene input
#
# The Hugging Face release is a split archive. Extract it before this audit and
# attach the extracted directory privately. The notebook deliberately does not
# download the gated dataset or expand a roughly 29 GB archive into its outputs.

# %%
def subject_number(path: Path) -> int:
    match = SUBJECT_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Not a GazeGene subject directory: {path}")
    return int(match.group(1))


def camera_number(path: Path) -> int:
    match = CAMERA_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"Cannot parse camera ID from {path.name}")
    return int(match.group(1))


def label_variant(path: Path) -> str:
    return "normalized" if "normalized" in path.name.lower() else "face_crops"


def candidate_dataset_roots(search_root: Path) -> list[Path]:
    candidates: set[Path] = set()
    # Kaggle inputs normally add one slug directory above the dataset root.
    # Use targeted shallow globs instead of rglob: recursively walking through
    # one million image files merely to find 504 labels is unnecessarily slow.
    for prefix_depth in range(1, 5):
        prefix = "*/" * prefix_depth
        for labels_dir in ("labels_normalized", "labels"):
            pattern = (
                f"{prefix}subject*/{labels_dir}/gaze_label_camera0*.pkl"
            )
            for label_path in search_root.glob(pattern):
                subject_dir = label_path.parent.parent
                if SUBJECT_PATTERN.match(subject_dir.name):
                    candidates.add(subject_dir.parent.resolve())
    return sorted(candidates)


def root_label_files(root: Path, requested_variant: str) -> list[Path]:
    files = sorted(
        path
        for subject_dir in root.glob("subject*")
        if SUBJECT_PATTERN.match(subject_dir.name)
        for labels_dir in ("labels_normalized", "labels")
        for path in (subject_dir / labels_dir).glob(GAZE_LABEL_PATTERN)
    )
    if requested_variant == "auto":
        variants = {label_variant(path) for path in files}
        if "normalized" in variants:
            return [path for path in files if label_variant(path) == "normalized"]
        return [path for path in files if label_variant(path) == "face_crops"]
    return [path for path in files if label_variant(path) == requested_variant]


def resolve_dataset_root() -> tuple[Path, list[Path]]:
    if CONFIG.dataset_root:
        root = Path(CONFIG.dataset_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"GAZEGENE_ROOT is not a directory: {root}")
        files = root_label_files(root, CONFIG.variant)
        if not files:
            raise FileNotFoundError(f"No GazeGene label files found under {root}")
        return root, files

    local_candidates = (
        Path.cwd() / "data" / "GazeGene" / "extracted" / "GazeGene_FaceCrops",
        Path.cwd() / "data" / "GazeGene_FaceCrops",
    )
    for root in local_candidates:
        if root.is_dir():
            files = root_label_files(root.resolve(), CONFIG.variant)
            if files:
                return root.resolve(), files

    search_roots = [Path("/kaggle/input")] if Path("/kaggle/input").exists() else []
    candidates: list[tuple[int, Path, list[Path]]] = []
    for search_root in search_roots:
        for root in candidate_dataset_roots(search_root):
            files = root_label_files(root, CONFIG.variant)
            if files:
                candidates.append((len(files), root, files))
    if not candidates:
        raise FileNotFoundError(
            "No extracted GazeGene labels were found. Attach the extracted private "
            "dataset and optionally set GAZEGENE_ROOT. Split archive parts alone "
            "are not a runnable dataset."
        )
    candidates.sort(key=lambda item: (-item[0], str(item[1])))
    best_count = candidates[0][0]
    best = [item for item in candidates if item[0] == best_count]
    if len(best) > 1:
        roots = "\n- ".join(str(item[1]) for item in best)
        raise RuntimeError(
            "Several equally complete GazeGene roots were found. Set "
            f"GAZEGENE_ROOT explicitly:\n- {roots}"
        )
    _, root, files = best[0]
    return root, files


DATASET_ROOT, LABEL_FILES = resolve_dataset_root()
DETECTED_VARIANT = label_variant(LABEL_FILES[0])
print(
    json.dumps(
        {
            "dataset_root": str(DATASET_ROOT),
            "detected_variant": DETECTED_VARIANT,
            "label_file_count": len(LABEL_FILES),
        },
        indent=2,
    )
)


# %% [markdown]
# ## 3. Validate annotation files and build a non-biometric manifest
#
# Every label file is loaded once. The manifest stores relative label paths and
# aggregate statistics, never raw annotations or absolute Kaggle input paths.

# %%
def load_official_pickle(path: Path) -> Mapping[str, Any]:
    # Only official GazeGene files are allowed here; pickle can execute code.
    with path.open("rb") as stream:
        value = pickle.load(stream)  # noqa: S301
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected a mapping in {path}, got {type(value).__name__}")
    return value


def array_with_shape(
    labels: Mapping[str, Any], key: str, trailing_shape: tuple[int, ...]
) -> np.ndarray:
    if key not in labels:
        raise KeyError(key)
    array = np.asarray(labels[key])
    if array.ndim != len(trailing_shape) + 1 or array.shape[1:] != trailing_shape:
        raise ValueError(f"{key} has unexpected shape {array.shape}")
    return array


manifest_rows: list[dict[str, Any]] = []
all_angles: list[np.ndarray] = []
fatal_errors: list[str] = []

for index, label_path in enumerate(LABEL_FILES, start=1):
    subject = subject_number(label_path.parent.parent)
    camera = camera_number(label_path)
    try:
        labels = load_official_pickle(label_path)
        missing_keys = sorted(REQUIRED_GAZE_KEYS - set(labels))
        if missing_keys:
            raise KeyError(f"missing keys: {missing_keys}")

        gaze = array_with_shape(labels, "gaze_C", (3,)).astype(np.float64)
        sample_count = len(gaze)
        for key in ("visual_axis_L", "visual_axis_R", "optic_axis_L", "optic_axis_R"):
            if len(array_with_shape(labels, key, (3,))) != sample_count:
                raise ValueError(f"{key} length does not match gaze_C")
        if len(array_with_shape(labels, "gaze_target", (3,))) != sample_count:
            raise ValueError("gaze_target length does not match gaze_C")
        if len(array_with_shape(labels, "head_R_mat", (3, 3))) != sample_count:
            raise ValueError("head_R_mat length does not match gaze_C")
        if len(array_with_shape(labels, "head_T_vec", (3,))) != sample_count:
            raise ValueError("head_T_vec length does not match gaze_C")
        gaze_depth = np.asarray(labels["gaze_depth"])
        if gaze_depth.shape != (sample_count,):
            raise ValueError(f"gaze_depth has unexpected shape {gaze_depth.shape}")
        if not all(
            np.isfinite(np.asarray(labels[key])).all() for key in REQUIRED_GAZE_KEYS
        ):
            raise ValueError("one or more required arrays contain non-finite values")

        norms = np.linalg.norm(gaze, axis=1)
        angles = gazegene_vector_to_unigaze_pitchyaw(gaze)
        all_angles.append(angles)
        manifest_rows.append(
            {
                "subject": subject,
                "camera": camera,
                "variant": DETECTED_VARIANT,
                "label_path": label_path.relative_to(DATASET_ROOT).as_posix(),
                "label_bytes": label_path.stat().st_size,
                "sample_count": sample_count,
                "finite": True,
                "gaze_norm_mean": float(norms.mean()),
                "gaze_norm_max_abs_error": float(np.max(np.abs(norms - 1.0))),
                "pitch_min_deg": float(np.rad2deg(angles[:, 0]).min()),
                "pitch_max_deg": float(np.rad2deg(angles[:, 0]).max()),
                "yaw_min_deg": float(np.rad2deg(angles[:, 1]).min()),
                "yaw_max_deg": float(np.rad2deg(angles[:, 1]).max()),
                "median_original_z": float(np.median(gaze[:, 2])),
            }
        )
    except (KeyError, OSError, TypeError, ValueError, pickle.UnpicklingError) as error:
        fatal_errors.append(f"{label_path.relative_to(DATASET_ROOT)}: {error}")

    if index % 50 == 0 or index == len(LABEL_FILES):
        print(f"Audited {index}/{len(LABEL_FILES)} label files", flush=True)

manifest = pd.DataFrame(manifest_rows).sort_values(["subject", "camera"])
if manifest.empty:
    raise RuntimeError("No valid GazeGene label files were found")
manifest.to_csv(OUTPUT_ROOT / "label_manifest.csv", index=False)

subject_summary = (
    manifest.groupby("subject", as_index=False)
    .agg(camera_count=("camera", "nunique"), sample_count=("sample_count", "sum"))
    .sort_values("subject")
)
subject_summary.to_csv(OUTPUT_ROOT / "subject_summary.csv", index=False)

print(manifest.head())
print(subject_summary.to_string(index=False))


# %% [markdown]
# ## 4. Fixed identity-disjoint project split
#
# The dataset card recommends subjects 1–46 for training and 47–56 for testing.
# Because test identities must not select epochs or hyperparameters, this project
# reserves subjects 41–46 as validation and uses 1–40 for training. This is a
# project protocol derived from, but not identical to, the official 46/10 split.

# %%
split = {
    "protocol": "identity_disjoint_40_train_6_validation_10_test",
    "source_recommendation": "subjects 1-46 train; subjects 47-56 test",
    "train_subjects": list(CONFIG.project_train_subjects),
    "validation_subjects": list(CONFIG.project_validation_subjects),
    "test_subjects": list(CONFIG.project_test_subjects),
}
split_subjects = (
    set(split["train_subjects"])
    | set(split["validation_subjects"])
    | set(split["test_subjects"])
)
assert len(split_subjects) == CONFIG.expected_subjects
assert not (set(split["train_subjects"]) & set(split["validation_subjects"]))
assert not (set(split["train_subjects"]) & set(split["test_subjects"]))
assert not (set(split["validation_subjects"]) & set(split["test_subjects"]))
(OUTPUT_ROOT / "split.json").write_text(
    json.dumps(split, indent=2), encoding="utf-8"
)
print(json.dumps(split, indent=2))


# %% [markdown]
# ## 5. Distribution plots and image-resolution smoke audit

# The plots use all gaze labels. The montage opens only a few images. Missing
# image directories are fatal because label-only success is not sufficient for
# a trainable dataset.

# %%
angles = np.concatenate(all_angles, axis=0)
angles_deg = np.rad2deg(angles)

figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].hist(angles_deg[:, 1], bins=90, color="tab:blue", alpha=0.85)
axes[0].set(title="GazeGene yaw", xlabel="degrees", ylabel="samples")
axes[1].hist(angles_deg[:, 0], bins=90, color="tab:orange", alpha=0.85)
axes[1].set(title="GazeGene pitch", xlabel="degrees", ylabel="samples")
figure.tight_layout()
figure.savefig(OUTPUT_ROOT / "gaze_angle_distributions.png", dpi=160)
plt.show()

camera_counts = manifest.groupby("camera")["sample_count"].sum().sort_index()
axis = camera_counts.plot.bar(figsize=(10, 4), color="tab:green")
axis.set(title="Samples per camera", xlabel="camera", ylabel="samples")
axis.figure.tight_layout()
axis.figure.savefig(OUTPUT_ROOT / "samples_per_camera.png", dpi=160)
plt.show()


def image_directory_for_label(label_path: Path) -> Path | None:
    subject_dir = label_path.parent.parent
    camera = camera_number(label_path)
    directory_names = (
        ("imgs_normalized", "imgs")
        if DETECTED_VARIANT == "normalized"
        else ("imgs", "imgs_normalized")
    )
    for directory_name in directory_names:
        candidate = subject_dir / directory_name / f"camera{camera}"
        if candidate.is_dir():
            return candidate
    labels = load_official_pickle(label_path)
    image_hint = labels.get("img_path")
    if isinstance(image_hint, (str, os.PathLike)):
        hint_parent = Path(image_hint).parent
        for candidate in (subject_dir / hint_parent, DATASET_ROOT / hint_parent):
            if candidate.is_dir():
                return candidate
    return None


sampled_label_indices = RNG.choice(
    len(LABEL_FILES), size=min(CONFIG.montage_images, len(LABEL_FILES)), replace=False
)
sampled_images: list[tuple[Path, int, int]] = []
missing_image_directories: list[str] = []
for label_index in sampled_label_indices:
    label_path = LABEL_FILES[int(label_index)]
    image_dir = image_directory_for_label(label_path)
    if image_dir is None:
        missing_image_directories.append(
            label_path.relative_to(DATASET_ROOT).as_posix()
        )
        continue
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        missing_image_directories.append(
            label_path.relative_to(DATASET_ROOT).as_posix()
        )
        continue
    image_path = image_paths[int(RNG.integers(0, len(image_paths)))]
    sampled_images.append(
        (image_path, subject_number(label_path.parent.parent), camera_number(label_path))
    )

if not sampled_images:
    raise RuntimeError(
        "No GazeGene images could be resolved from the sampled label files. "
        "Check the extracted directory layout."
    )

columns = 4
rows = int(np.ceil(len(sampled_images) / columns))
figure, axes = plt.subplots(rows, columns, figsize=(14, 3.5 * rows))
for axis in np.asarray(axes).reshape(-1):
    axis.axis("off")
image_shapes: list[tuple[int, int, int]] = []
for axis, (image_path, subject, camera) in zip(
    np.asarray(axes).reshape(-1), sampled_images, strict=False
):
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        image_shapes.append((rgb.height, rgb.width, 3))
        axis.imshow(rgb)
    axis.set_title(f"subject{subject}, camera{camera}\n{image_path.name}")
    axis.axis("off")
figure.suptitle(f"GazeGene qualitative audit — {DETECTED_VARIANT}")
figure.tight_layout()
figure.savefig(OUTPUT_ROOT / "qualitative_montage.png", dpi=150)
plt.show()

print({"sampled_image_shapes": sorted(set(image_shapes))})


# %% [markdown]
# ## 6. Label-only compatibility smoke manifest
#
# This output proves that labels can be converted to Task 0's `[pitch, yaw]`
# convention without copying images. It is not a training dataset and contains
# no absolute input paths.

# %%
smoke_parts: list[pd.DataFrame] = []
shuffled_manifest = manifest.sample(frac=1.0, random_state=CONFIG.seed)
base_per_label, extra_labels = divmod(CONFIG.smoke_samples, len(shuffled_manifest))
for position, row in enumerate(shuffled_manifest.itertuples(index=False)):
    label_path = DATASET_ROOT / row.label_path
    labels = load_official_pickle(label_path)
    gaze = np.asarray(labels["gaze_C"], dtype=np.float64)
    take = base_per_label + int(position < extra_labels)
    if take == 0:
        continue
    if take > len(gaze):
        raise RuntimeError(
            f"Cannot sample {take} unique frames from {row.label_path}"
        )
    indices = RNG.choice(len(gaze), size=take, replace=False)
    pitchyaw = gazegene_vector_to_unigaze_pitchyaw(gaze[indices])
    smoke_parts.append(
        pd.DataFrame(
            {
                "subject": int(row.subject),
                "camera": int(row.camera),
                "frame_index": indices,
                "target_pitch_rad": pitchyaw[:, 0],
                "target_yaw_rad": pitchyaw[:, 1],
            }
        )
    )

smoke_manifest = pd.concat(smoke_parts, ignore_index=True)
if len(smoke_manifest) != CONFIG.smoke_samples:
    raise RuntimeError(
        f"Requested {CONFIG.smoke_samples} smoke labels, got {len(smoke_manifest)}"
    )
if not np.isfinite(
    smoke_manifest[["target_pitch_rad", "target_yaw_rad"]].to_numpy()
).all():
    raise ValueError("Compatibility smoke manifest contains non-finite angles")
smoke_manifest.to_csv(OUTPUT_ROOT / "compatibility_smoke_manifest.csv", index=False)
print(smoke_manifest.describe().to_string())


# %% [markdown]
# ## 7. Final report and execution gate
#
# `PASS` requires the documented 56 subjects, 9 cameras per subject, 2,000
# samples per subject-camera pair, 1,008,000 total labels, unit gaze vectors,
# disjoint complete splits, and resolvable sampled images. If the attached input
# is an intentional subset, set expectations only in a separate notebook
# version; do not report a subset smoke run as a complete dataset audit.

# %%
observed_subjects = sorted(manifest["subject"].unique().tolist())
camera_counts_by_subject = manifest.groupby("subject")["camera"].nunique()
total_samples = int(manifest["sample_count"].sum())
dataset_fingerprint_payload = "\n".join(
    f"{row.label_path}\t{row.label_bytes}\t{row.sample_count}"
    for row in manifest.itertuples(index=False)
)
dataset_fingerprint = hashlib.sha256(
    dataset_fingerprint_payload.encode("utf-8")
).hexdigest()

checks = {
    "no_label_errors": not fatal_errors,
    "expected_subject_ids": observed_subjects == list(range(1, 57)),
    "expected_camera_count_per_subject": bool(
        (camera_counts_by_subject == CONFIG.expected_cameras).all()
    ),
    "expected_samples_per_label_file": bool(
        (manifest["sample_count"] == CONFIG.expected_frames_per_camera).all()
    ),
    "expected_total_samples": total_samples == CONFIG.expected_total_samples,
    "unit_gaze_vectors": bool(
        (
            manifest["gaze_norm_max_abs_error"]
            <= CONFIG.gaze_norm_tolerance
        ).all()
    ),
    "sampled_images_resolved": not missing_image_directories,
    "split_covers_all_subjects": split_subjects == set(range(1, 57)),
}
status = "PASS" if all(checks.values()) else "FAIL"
report = {
    "status": status,
    "record_type": "dataset_audit",
    "dataset": "GazeGene",
    "detected_variant": DETECTED_VARIANT,
    "observed_subject_count": len(observed_subjects),
    "observed_label_file_count": len(manifest),
    "observed_total_samples": total_samples,
    "dataset_metadata_fingerprint_sha256": dataset_fingerprint,
    "coordinate_conversion": "vector_to_pitchyaw(-gaze_C) -> [pitch, yaw] radians",
    "checks": checks,
    "fatal_errors": fatal_errors[:100],
    "missing_sampled_image_directories": missing_image_directories,
    "sampled_image_shapes": [list(shape) for shape in sorted(set(image_shapes))],
    "config": asdict(CONFIG),
    "source_sha256": SOURCE_SHA256,
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "limitations": [
        "Metadata fingerprint hashes relative paths, sizes, and sample counts; it is not a full 28.7 GB content hash.",
        "Only sampled images are decoded; label arrays are audited exhaustively.",
        "FaceCrops and normalized variants require separate experiment records.",
        "This audit does not train or evaluate a gaze model.",
    ],
}
(OUTPUT_ROOT / "audit_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
(OUTPUT_ROOT / "resolved_config.json").write_text(
    json.dumps(asdict(CONFIG), indent=2), encoding="utf-8"
)
print(json.dumps(report, indent=2))
if status != "PASS":
    failed = [name for name, passed in checks.items() if not passed]
    raise RuntimeError(f"GazeGene audit failed: {failed}")

required_outputs = {
    "audit_report.json",
    "resolved_config.json",
    "label_manifest.csv",
    "subject_summary.csv",
    "split.json",
    "compatibility_smoke_manifest.csv",
    "gaze_angle_distributions.png",
    "samples_per_camera.png",
    "qualitative_montage.png",
}
missing_outputs = sorted(
    name for name in required_outputs if not (OUTPUT_ROOT / name).is_file()
)
if missing_outputs:
    raise RuntimeError(f"Missing expected outputs: {missing_outputs}")
print(f"GazeGene audit PASS; outputs: {OUTPUT_ROOT}")
