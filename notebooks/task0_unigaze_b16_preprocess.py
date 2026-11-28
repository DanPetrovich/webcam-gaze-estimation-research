# %% [markdown]
# # Task 0 — one-time MPIIFaceGaze preprocessing cache
#
# This Kaggle notebook runs the pinned official UniGaze normalization once and
# writes the resulting 15 HDF5 files to `/kaggle/working`. Save the completed
# notebook as a private Kaggle Dataset (or attach this notebook's output) and
# point the training notebook's `normalized_data_root` at the attached
# `mpiifacegaze_unigaze` directory with `run_preprocessing=False`.
#
# The HDF5 files contain derived face images and gaze labels. Keep the output
# private and apply the source dataset's license and biometric-data controls.

# %%
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class PreprocessConfig:
    runtime_root: str = (
        "/kaggle/temp/task0_unigaze_preprocess"
        if Path("/kaggle").exists()
        else "/space/project/diploma"
    )
    output_root: str = (
        "/kaggle/working/task0_unigaze_b16_preprocessed"
        if Path("/kaggle").exists()
        else "/space/project/diploma/data/processed"
    )
    seed: int = 20261004
    target_samples_per_subject: int = 3_000


CONFIG = PreprocessConfig()
RUNTIME_ROOT = Path(CONFIG.runtime_root).expanduser().resolve()
OUTPUT_ROOT = Path(CONFIG.output_root).expanduser().resolve()
NORMALIZED_ROOT = OUTPUT_ROOT / "mpiifacegaze_unigaze"
RAW_ROOT = RUNTIME_ROOT / "data" / "raw" / "MPIIFaceGaze"
UNIGAZE_REPO = RUNTIME_ROOT / "external" / "UniGaze"
SUBJECTS = tuple(f"p{index:02d}" for index in range(15))
UNIGAZE_REVISION = "9c240fbe33f3d6146970a77b7c8fa06a7e60019e"
RAW_URL = "https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip"
RAW_ARCHIVE_BYTES = 942_343_290


def run_checked(command: Sequence[str], *, cwd: Path | None = None) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise ValueError(f"Unsafe archive path: {member.filename}")
        zipped.extractall(destination)


def raw_subject_is_complete(subject: str) -> bool:
    root = RAW_ROOT / subject
    return (
        (root / f"{subject}.txt").is_file()
        and (root / "Calibration" / "Camera.mat").is_file()
        and any(root.glob("day*/*.jpg"))
    )


def normalized_subject_is_valid(subject: str, *, require_target: bool) -> bool:
    import h5py
    import numpy as np

    path = NORMALIZED_ROOT / f"{subject}.h5"
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r", swmr=True) as h5:
            required = {"face_patch", "face_gaze", "face_head_pose"}
            if not required.issubset(h5.keys()):
                return False
            count = int(h5["face_patch"].shape[0])
            if h5["face_patch"].shape[1:] != (224, 224, 3):
                return False
            if h5["face_gaze"].shape != (count, 2):
                return False
            if require_target and count != CONFIG.target_samples_per_subject:
                return False
            return bool(np.isfinite(h5["face_gaze"][:]).all())
    except (OSError, KeyError, ValueError):
        return False


# %% [markdown]
# ## 1. Install the single missing Kaggle dependency and pin UniGaze

# %%
if importlib.util.find_spec("face_alignment") is None:
    run_checked(
        [sys.executable, "-m", "pip", "install", "-q", "face-alignment==1.4.1"]
    )

if not (UNIGAZE_REPO / ".git").exists():
    UNIGAZE_REPO.parent.mkdir(parents=True, exist_ok=True)
    run_checked(["git", "init", str(UNIGAZE_REPO)])
    run_checked(
        [
            "git",
            "-C",
            str(UNIGAZE_REPO),
            "remote",
            "add",
            "origin",
            "https://github.com/ut-vision/UniGaze.git",
        ]
    )
    run_checked(
        [
            "git",
            "-C",
            str(UNIGAZE_REPO),
            "fetch",
            "--depth",
            "1",
            "origin",
            UNIGAZE_REVISION,
        ]
    )
    run_checked(
        ["git", "-C", str(UNIGAZE_REPO), "checkout", "--detach", "FETCH_HEAD"]
    )

normalize_script = UNIGAZE_REPO / "gazedata_preparation" / "normalize_mpii.py"
upsample_script = UNIGAZE_REPO / "gazedata_preparation" / "upsample_mpii.py"
for required_path in (normalize_script, upsample_script):
    if not required_path.is_file():
        raise FileNotFoundError(required_path)


# %% [markdown]
# ## 2. Download raw data into ephemeral storage

# %%
if not all(raw_subject_is_complete(subject) for subject in SUBJECTS):
    archive = RUNTIME_ROOT / "downloads" / "MPIIFaceGaze.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        partial = archive.with_suffix(".zip.part")
        try:
            urllib.request.urlretrieve(RAW_URL, partial)
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
    if archive.stat().st_size != RAW_ARCHIVE_BYTES:
        raise ValueError(f"Unexpected MPIIFaceGaze archive size: {archive.stat().st_size}")
    safe_extract_zip(archive, RAW_ROOT.parent)

missing_raw = [subject for subject in SUBJECTS if not raw_subject_is_complete(subject)]
if missing_raw:
    raise RuntimeError(f"Raw dataset is incomplete for: {missing_raw}")


# %% [markdown]
# ## 3. Normalize only missing subjects, then upsample to 3,000
#
# Completed valid HDF5 files are retained. If a previous preprocessing Version
# stopped halfway, rerunning this notebook processes only missing or invalid
# subjects instead of discarding two hours of work.

# %%
NORMALIZED_ROOT.mkdir(parents=True, exist_ok=True)
missing_normalized = [
    subject
    for subject in SUBJECTS
    if not normalized_subject_is_valid(subject, require_target=False)
]
if missing_normalized:
    raw_view = RUNTIME_ROOT / "data" / "raw_missing_subjects"
    if raw_view.exists():
        shutil.rmtree(raw_view)
    raw_view.mkdir(parents=True)
    for subject in missing_normalized:
        invalid_output = NORMALIZED_ROOT / f"{subject}.h5"
        invalid_output.unlink(missing_ok=True)
        (raw_view / subject).symlink_to(RAW_ROOT / subject, target_is_directory=True)
    run_checked(
        [
            sys.executable,
            str(normalize_script),
            "--mpiifacegaze_raw_dir",
            str(raw_view),
            "--output_dir",
            str(NORMALIZED_ROOT),
        ],
        cwd=normalize_script.parent,
    )
else:
    print("All normalized subjects already exist; normalization skipped.")

upsample_code = (
    "import numpy as np, runpy, sys; "
    f"np.random.seed({CONFIG.seed}); "
    f"sys.argv=['upsample_mpii.py','--data_dir',{str(NORMALIZED_ROOT)!r}]; "
    f"runpy.run_path({str(upsample_script)!r}, run_name='__main__')"
)
run_checked([sys.executable, "-c", upsample_code], cwd=upsample_script.parent)


# %% [markdown]
# ## 4. Validate and fingerprint the reusable output

# %%
import h5py

manifest: list[dict[str, Any]] = []
for subject in SUBJECTS:
    if not normalized_subject_is_valid(subject, require_target=True):
        raise RuntimeError(f"Invalid normalized output for {subject}")
    path = NORMALIZED_ROOT / f"{subject}.h5"
    with h5py.File(path, "r", swmr=True) as h5:
        manifest.append(
            {
                "subject": subject,
                "file": path.name,
                "samples": int(h5["face_patch"].shape[0]),
                "face_patch_shape": list(h5["face_patch"].shape),
                "face_gaze_shape": list(h5["face_gaze"].shape),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

metadata = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source_dataset": "MPIIFaceGaze",
    "preprocessing": "official UniGaze normalize_mpii.py plus upsample_mpii.py",
    "unigaze_revision": UNIGAZE_REVISION,
    "seed": CONFIG.seed,
    "subjects": manifest,
    "usage": {
        "normalized_data_root": str(NORMALIZED_ROOT),
        "run_preprocessing": False,
    },
    "privacy": "Derived face images; keep private and follow the source license.",
}
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "manifest.json").write_text(
    json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
)
(OUTPUT_ROOT / "resolved_config.json").write_text(
    json.dumps(asdict(CONFIG), indent=2) + "\n", encoding="utf-8"
)

total_bytes = sum(item["bytes"] for item in manifest)
print(
    json.dumps(
        {
            "status": "complete",
            "subjects": len(manifest),
            "samples": sum(item["samples"] for item in manifest),
            "total_gib": total_bytes / (1024**3),
            "normalized_data_root": str(NORMALIZED_ROOT),
        },
        indent=2,
    )
)
print("Preprocessing completed successfully.")
