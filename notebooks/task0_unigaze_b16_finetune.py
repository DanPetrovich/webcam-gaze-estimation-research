# %% [markdown]
# # Task 0 — full UniGaze-B16 fine-tuning on MPIIFaceGaze
#
# This notebook is the main Task 0 experiment. It loads the official UniGaze
# MAE-B16 checkpoint, attaches the authors' two-angle gaze head, and fine-tunes
# **the complete ViT and head** on normalized MPIIFaceGaze faces. It is not the
# frozen-feature linear probe from the earlier exploratory notebook.
#
# The notebook runs on Kaggle and remains portable to the CU cluster:
#
# - Kaggle: Internet and a GPU accelerator must be enabled;
# - attach a Kaggle Secret named `wandb_api` to enable live W&B tracking;
# - Kaggle downloads and raw biometric data stay under `/kaggle/temp`;
# - only logs, checkpoints, aggregate metrics, and plots go to
#   `/kaggle/working` and become Notebook Output;
# - on the CU cluster the same code uses `/space/project/diploma` and expects
#   all inputs to be present offline.
#
# Start with `ExperimentConfig.run_mode = "smoke"`. After it finishes
# successfully, change the default to `"single_fold"`, then to `"full_lopo"`
# for the complete 15-fold run. Smoke results are deliberately stored separately
# and must never be reported as benchmark results.

# %% [markdown]
# ## 1. Configuration
#
# Kaggle downloads the official inputs automatically. Expected CU inputs:
#
# ```text
# /space/project/diploma/
# ├── external/UniGaze/                         official repository
# ├── external/wheelhouse/                      offline Python wheels
# ├── data/raw/MPIIFaceGaze/                  optional raw dataset
# ├── data/processed/mpiifacegaze_unigaze/    p00.h5 ... p14.h5
# └── models/pretrained/mae_b16_checkpoint-299.pth
# ```
#
# The normalized HDF5 files must be produced by the official UniGaze
# `gazedata_preparation/normalize_mpii.py` pipeline and then upsampled to 3,000
# samples per subject as described by the authors. Set
# `ExperimentConfig.run_preprocessing=True` only when the raw dataset and every
# offline dependency are already installed.

# %%
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from IPython.display import Image as NotebookImage
from IPython.display import display
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable: Iterable[Any], **_: Any) -> Iterable[Any]:
        return iterable


@dataclass(frozen=True)
class ExperimentConfig:
    project_root: str = (
        "/kaggle/working" if Path("/kaggle").exists() else "/space/project/diploma"
    )
    runtime_root: str = (
        "/kaggle/temp/task0_unigaze_b16"
        if Path("/kaggle").exists()
        else "/space/project/diploma"
    )
    run_mode: str = "smoke"  # smoke | single_fold | full_lopo
    seed: int = 20261004
    model_type: str = "vit_b_16"
    epochs: int = 12
    smoke_epochs: int = 1
    batch_size: int = 8
    eval_batch_size: int = 32
    num_workers: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    drop_path_rate: float = 0.1
    one_cycle_div_factor: float = 25.0
    one_cycle_final_div_factor: float = 10_000.0
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float | None = None
    use_amp: bool = True
    train_subject_limit: int = 16
    validation_subject_limit: int = 16
    test_subject_limit: int = 16
    smoke_raw_samples_per_subject: int = 16
    single_test_subject: str = "p00"
    require_3000_samples_per_subject: bool = True
    run_preprocessing: bool = True
    resume: bool = True
    keep_resume_checkpoint_after_completion: bool = False
    log_every_steps: int = 25
    bootstrap_samples: int = 2_000
    latency_warmup_steps: int = 10
    latency_measurement_steps: int = 50
    enable_wandb: bool = True
    wandb_required: bool = False
    wandb_secret_label: str = "wandb_api"
    wandb_project: str = "diploma-gaze-estimation"
    wandb_entity: str | None = None
    wandb_group: str = "task0-unigaze-b16"
    wandb_log_every_steps: int = 10
    wandb_watch_gradients: bool = True
    wandb_gradient_log_frequency: int = 100
    wandb_log_aggregate_artifact: bool = True
    wandb_alerts: bool = True


CONFIG = ExperimentConfig()
if CONFIG.run_mode not in {"smoke", "single_fold", "full_lopo"}:
    raise ValueError(f"Unsupported run_mode: {CONFIG.run_mode}")
if CONFIG.gradient_accumulation_steps < 1:
    raise ValueError("gradient_accumulation_steps must be positive")
if CONFIG.log_every_steps < 1 or CONFIG.wandb_log_every_steps < 1:
    raise ValueError("Logging intervals must be positive")

IS_KAGGLE = Path("/kaggle").is_dir()
PROJECT_ROOT = Path(CONFIG.project_root).expanduser().resolve()
RUNTIME_ROOT = Path(CONFIG.runtime_root).expanduser().resolve()
UNIGAZE_REPO = RUNTIME_ROOT / "external" / "UniGaze"
UNIGAZE_PACKAGE_ROOT = UNIGAZE_REPO / "unigaze"
OFFLINE_WHEELHOUSE = RUNTIME_ROOT / "external" / "wheelhouse"
RAW_DATA_ROOT = RUNTIME_ROOT / "data" / "raw" / "MPIIFaceGaze"
NORMALIZED_DATA_ROOT = (
    RUNTIME_ROOT / "data" / "processed" / "mpiifacegaze_unigaze"
)
MAE_CHECKPOINT = (
    RUNTIME_ROOT / "models" / "pretrained" / "mae_b16_checkpoint-299.pth"
)
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "task0_unigaze_b16"
SUBJECTS = tuple(f"p{i:02d}" for i in range(15))
SMOKE_SUBJECTS = ("p00", "p01", "p02")
ACTIVE_SUBJECTS = SMOKE_SUBJECTS if CONFIG.run_mode == "smoke" else SUBJECTS
UNIGAZE_REVISION = "9c240fbe33f3d6146970a77b7c8fa06a7e60019e"
RAW_URL = "https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip"
MAE_B16_FOLDER = "https://drive.google.com/drive/folders/1vz38f90jPrMwb_lByzJfaMgH6BEtB49f"
NOTEBOOK_SOURCE_SHA256 = "__SOURCE_SHA256__"


def stable_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


CONFIG_DICT = asdict(CONFIG)
CONFIG_HASH = stable_json_hash(CONFIG_DICT)
if NOTEBOOK_SOURCE_SHA256.startswith("__"):
    if "__file__" not in globals():
        raise RuntimeError("Notebook source hash was not injected during export")
    NOTEBOOK_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
RUN_STARTED_AT = datetime.now(timezone.utc)
RUN_ID = (
    f"{RUN_STARTED_AT.strftime('%Y%m%dT%H%M%SZ')}_{CONFIG.run_mode}_"
    f"c{CONFIG_HASH[:8]}_s{NOTEBOOK_SOURCE_SHA256[:8]}"
)
RUN_DIR = OUTPUT_BASE / RUN_ID
CHECKPOINT_ROOT = RUN_DIR / "checkpoints"
PLOT_ROOT = RUN_DIR / "plots"
METRIC_ROOT = RUN_DIR / "metrics"
LOG_ROOT = RUN_DIR / "logs"
for directory in (RUN_DIR, CHECKPOINT_ROOT, PLOT_ROOT, METRIC_ROOT, LOG_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

print(
    json.dumps(
        {
            **CONFIG_DICT,
            "config_hash": CONFIG_HASH,
            "source_sha256": NOTEBOOK_SOURCE_SHA256,
            "run_id": RUN_ID,
            "run_dir": str(RUN_DIR),
        },
        indent=2,
    )
)

# %% [markdown]
# ## 2. Reproducibility, dependencies, and official model source
#
# On Kaggle this cell installs only missing dependencies, checks out a pinned
# UniGaze revision, and downloads the official MAE-B16 checkpoint. On the
# network-isolated CU cluster it reads the same assets from persistent storage.

# %%
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CONFIG.seed)


def run_checked(command: Sequence[str], cwd: Path | None = None) -> None:
    print("$", " ".join(map(str, command)))
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def ensure_package(module: str, requirement: str) -> None:
    """Install one missing package online on Kaggle or offline on CU."""
    if importlib.util.find_spec(module) is not None:
        return
    command = [sys.executable, "-m", "pip", "install", "-q"]
    if not IS_KAGGLE:
        if not OFFLINE_WHEELHOUSE.is_dir():
            raise RuntimeError(
                f"Missing module {module!r} and offline wheelhouse "
                f"{OFFLINE_WHEELHOUSE} does not exist"
            )
        command.extend(["--no-index", "--find-links", str(OFFLINE_WHEELHOUSE)])
    command.append(requirement)
    run_checked(command)
    importlib.invalidate_caches()
    if importlib.util.find_spec(module) is None:
        raise RuntimeError(f"Installation did not provide module {module!r}")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    raise RuntimeError(
        "Task 0 fine-tuning requires a CUDA GPU. In Kaggle open "
        "Settings -> Accelerator and select a GPU before running all cells."
    )


for module, requirement in (
    ("omegaconf", "omegaconf==2.3.0"),
    ("timm", "timm==1.0.9"),
    ("face_alignment", "face-alignment==1.4.1"),
    ("rich", "rich>=13,<15"),
    ("gdown", "gdown>=5,<6"),
    ("wandb", "wandb>=0.18,<1"),
):
    ensure_package(module, requirement)


if IS_KAGGLE:
    current_revision = None
    if (UNIGAZE_REPO / ".git").is_dir():
        current_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=UNIGAZE_REPO, text=True
        ).strip()
    if current_revision != UNIGAZE_REVISION:
        if UNIGAZE_REPO.exists():
            shutil.rmtree(UNIGAZE_REPO)
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

    if not MAE_CHECKPOINT.is_file():
        import gdown

        checkpoint_download = RUNTIME_ROOT / "downloads" / "mae_b16"
        checkpoint_download.mkdir(parents=True, exist_ok=True)
        gdown.download_folder(
            MAE_B16_FOLDER,
            output=str(checkpoint_download),
            quiet=False,
            use_cookies=False,
        )
        candidates = list(checkpoint_download.rglob("mae_b16_checkpoint-299.pth"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                "Expected exactly one official MAE-B16 checkpoint, "
                f"found {len(candidates)}"
            )
        MAE_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(candidates[0]), MAE_CHECKPOINT)

    # PyTorch 2.6 changed torch.load's default to weights_only=True. The
    # authors' trusted checkpoint contains NumPy objects and their loader does
    # not yet pass the compatibility flag explicitly.
    torch_major_minor = tuple(
        int(component) for component in torch.__version__.split("+")[0].split(".")[:2]
    )
    if torch_major_minor >= (2, 6):
        mae_gaze_source = UNIGAZE_PACKAGE_ROOT / "models" / "vit" / "mae_gaze.py"
        source_text = mae_gaze_source.read_text(encoding="utf-8")
        old_call = "torch.load(custom_pretrained_path, map_location='cpu')"
        new_call = (
            "torch.load(custom_pretrained_path, map_location='cpu', "
            "weights_only=False)"
        )
        if old_call in source_text:
            mae_gaze_source.write_text(
                source_text.replace(old_call, new_call), encoding="utf-8"
            )


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)

def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


REQUIRED_MODULES = {
    "cv2": "opencv-python",
    "h5py": "h5py",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "omegaconf": "omegaconf",
    "pandas": "pandas",
    "timm": "timm",
    "torch": "torch",
    "torchvision": "torchvision",
    "wandb": "wandb",
}
if CONFIG.run_preprocessing:
    REQUIRED_MODULES["face_alignment"] = "face-alignment"
    REQUIRED_MODULES["scipy"] = "scipy"

missing_modules = [
    f"{module} ({distribution})"
    for module, distribution in REQUIRED_MODULES.items()
    if importlib.util.find_spec(module) is None
]
if missing_modules:
    raise RuntimeError(
        "Missing dependencies after setup: " + ", ".join(missing_modules)
    )

required_repository_files = [
    UNIGAZE_PACKAGE_ROOT / "models" / "vit" / "mae_gaze.py",
    UNIGAZE_PACKAGE_ROOT / "datasets" / "mpiigaze.py",
    UNIGAZE_REPO / "LICENSE.txt",
]
missing_repository_files = [str(path) for path in required_repository_files if not path.is_file()]
if missing_repository_files:
    raise FileNotFoundError(
        "Official UniGaze source is incomplete. Missing:\n- "
        + "\n- ".join(missing_repository_files)
    )

if not MAE_CHECKPOINT.is_file():
    raise FileNotFoundError(
        f"Place the official MAE-B16 checkpoint at {MAE_CHECKPOINT}"
    )
if MAE_CHECKPOINT.stat().st_size < 10_000_000:
    raise ValueError(f"Checkpoint is implausibly small: {MAE_CHECKPOINT}")


def git_revision(repository: Path) -> str | None:
    revision_file = repository / "OFFICIAL_REVISION.txt"
    if revision_file.is_file():
        return revision_file.read_text(encoding="utf-8").strip()
    if not (repository / ".git").exists():
        return None
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


environment = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    "unigaze_revision": git_revision(UNIGAZE_REPO),
    "unigaze_mae_gaze_sha256": sha256_file(
        UNIGAZE_PACKAGE_ROOT / "models" / "vit" / "mae_gaze.py"
    ),
    "checkpoint_path": str(MAE_CHECKPOINT),
    "checkpoint_sha256": sha256_file(MAE_CHECKPOINT),
    "notebook_source_sha256": NOTEBOOK_SOURCE_SHA256,
    "packages": {
        distribution: package_version(distribution)
        for distribution in sorted(set(REQUIRED_MODULES.values()))
    },
}
(RUN_DIR / "environment.json").write_text(
    json.dumps(environment, indent=2), encoding="utf-8"
)
(RUN_DIR / "resolved_config.json").write_text(
    json.dumps({**CONFIG_DICT, "config_hash": CONFIG_HASH}, indent=2),
    encoding="utf-8",
)
print(json.dumps(environment, indent=2))

initial_manifest = {
    "schema_version": 1,
    "run_id": RUN_ID,
    "experiment": "task0_unigaze_b16_finetune",
    "status": "running",
    "started_at": RUN_STARTED_AT.isoformat(),
    "finished_at": None,
    "config_sha256": CONFIG_HASH,
    "source_sha256": NOTEBOOK_SOURCE_SHA256,
    "dataset_sha256": None,
    "git": {"revision": None, "dirty": None},
    "runtime": environment,
    "external_ids": {
        "platform": "kaggle" if IS_KAGGLE else "cu_jupyterhub",
        "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
    },
}
(RUN_DIR / "run_manifest.json").write_text(
    json.dumps(initial_manifest, indent=2), encoding="utf-8"
)

# %% [markdown]
# ## 3. Official MPIIFaceGaze download and preprocessing
#
# Kaggle downloads the official 899 MB raw archive. Smoke mode creates a view
# containing 16 source images for each of three disjoint subjects, then invokes
# the authors' unchanged `normalize_mpii.py`. Full runs normalize all subjects
# and apply the authors' 3,000-sample upsampling step.

# %%
def raw_dataset_is_complete(root: Path) -> bool:
    return all(
        (root / subject / f"{subject}.txt").is_file()
        and (root / subject / "Calibration" / "Camera.mat").is_file()
        for subject in ACTIVE_SUBJECTS
    )


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise ValueError(f"Unsafe archive path: {member.filename}")
        zipped.extractall(destination)


def download_raw_dataset() -> None:
    if raw_dataset_is_complete(RAW_DATA_ROOT):
        return
    if not IS_KAGGLE:
        raise FileNotFoundError(f"Raw MPIIFaceGaze is incomplete under {RAW_DATA_ROOT}")
    archive = RUNTIME_ROOT / "downloads" / "MPIIFaceGaze.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        partial = archive.with_suffix(".zip.part")
        try:
            urllib.request.urlretrieve(RAW_URL, partial)
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
    if archive.stat().st_size != 942_343_290:
        raise ValueError(f"Unexpected MPIIFaceGaze archive size: {archive.stat().st_size}")
    safe_extract_zip(archive, RAW_DATA_ROOT.parent)
    if not raw_dataset_is_complete(RAW_DATA_ROOT):
        raise RuntimeError("Official archive did not produce the expected dataset layout")


def make_smoke_raw_view(source: Path) -> Path:
    destination = RUNTIME_ROOT / "data" / "raw_smoke"
    if destination.exists():
        shutil.rmtree(destination)
    for subject in SMOKE_SUBJECTS:
        source_subject = source / subject
        target_subject = destination / subject
        target_subject.mkdir(parents=True)
        (target_subject / f"{subject}.txt").symlink_to(
            source_subject / f"{subject}.txt"
        )
        (target_subject / "Calibration").symlink_to(
            source_subject / "Calibration", target_is_directory=True
        )
        images = sorted(source_subject.glob("day*/*.jpg"))[
            : CONFIG.smoke_raw_samples_per_subject
        ]
        if len(images) != CONFIG.smoke_raw_samples_per_subject:
            raise RuntimeError(f"Not enough raw smoke images for {subject}")
        for image in images:
            target_day = target_subject / image.parent.name
            target_day.mkdir(exist_ok=True)
            (target_day / image.name).symlink_to(image)
    return destination


if CONFIG.run_preprocessing:
    download_raw_dataset()

PREPROCESS_RAW_ROOT = (
    make_smoke_raw_view(RAW_DATA_ROOT)
    if CONFIG.run_mode == "smoke" and CONFIG.run_preprocessing
    else RAW_DATA_ROOT
)


existing_h5 = sorted(NORMALIZED_DATA_ROOT.glob("p??.h5"))
if CONFIG.run_preprocessing:
    if not raw_dataset_is_complete(PREPROCESS_RAW_ROOT):
        raise FileNotFoundError(
            f"Raw MPIIFaceGaze is incomplete under {PREPROCESS_RAW_ROOT}"
        )
    if existing_h5 and len(existing_h5) != len(ACTIVE_SUBJECTS):
        raise RuntimeError(
            "Normalization output is partial. Move it aside and restart preprocessing: "
            f"{NORMALIZED_DATA_ROOT}"
        )
    if not existing_h5:
        NORMALIZED_DATA_ROOT.mkdir(parents=True, exist_ok=True)
        run_checked(
            [
                sys.executable,
                "normalize_mpii.py",
                "--mpiifacegaze_raw_dir",
                str(PREPROCESS_RAW_ROOT),
                "--output_dir",
                str(NORMALIZED_DATA_ROOT),
            ],
            cwd=UNIGAZE_REPO / "gazedata_preparation",
        )
    if CONFIG.run_mode != "smoke":
        upsample_code = (
            "import numpy as np, runpy, sys; "
            f"np.random.seed({CONFIG.seed}); "
            f"sys.argv=['upsample_mpii.py','--data_dir',{str(NORMALIZED_DATA_ROOT)!r}]; "
            f"runpy.run_path({str(UNIGAZE_REPO / 'gazedata_preparation' / 'upsample_mpii.py')!r}, run_name='__main__')"
        )
        run_checked(
            [sys.executable, "-c", upsample_code],
            cwd=UNIGAZE_REPO / "gazedata_preparation",
        )

# %% [markdown]
# ## 4. Dataset audit and normalized-image preview
#
# The audit verifies all subjects, HDF5 keys, shapes, finite labels, image size,
# and the expected 3,000 samples per subject. The resulting manifest is saved
# beside the experiment. The manifest hash covers filenames, shapes, sizes, and
# timestamps; it is not a byte-level checksum of the large HDF5 payloads.

# %%
expected_h5 = [NORMALIZED_DATA_ROOT / f"{subject}.h5" for subject in ACTIVE_SUBJECTS]
missing_h5 = [str(path) for path in expected_h5 if not path.is_file()]
if missing_h5:
    raise FileNotFoundError(
        "Missing normalized UniGaze HDF5 files:\n- " + "\n- ".join(missing_h5)
    )

dataset_manifest: list[dict[str, Any]] = []
for subject, path in zip(ACTIVE_SUBJECTS, expected_h5, strict=True):
    with h5py.File(path, "r", swmr=True) as h5:
        required_keys = {"face_patch", "face_gaze", "face_head_pose"}
        missing_keys = sorted(required_keys.difference(h5.keys()))
        if missing_keys:
            raise KeyError(f"{path} misses HDF5 keys: {missing_keys}")
        count = int(h5["face_patch"].shape[0])
        if h5["face_patch"].shape[1:] != (224, 224, 3):
            raise ValueError(f"Unexpected face_patch shape in {path}: {h5['face_patch'].shape}")
        if h5["face_gaze"].shape != (count, 2):
            raise ValueError(f"Unexpected face_gaze shape in {path}: {h5['face_gaze'].shape}")
        if not np.isfinite(h5["face_gaze"][:]).all():
            raise ValueError(f"Non-finite gaze labels in {path}")
        if (
            CONFIG.run_mode != "smoke"
            and CONFIG.require_3000_samples_per_subject
            and count != 3_000
        ):
            raise ValueError(
                f"{subject} has {count} samples; official UniGaze preprocessing "
                "requires upsampling each MPIIFaceGaze subject to 3,000"
            )
        stat = path.stat()
        dataset_manifest.append(
            {
                "subject": subject,
                "path": str(path),
                "samples": count,
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "face_patch_shape": list(h5["face_patch"].shape),
                "face_gaze_shape": list(h5["face_gaze"].shape),
            }
        )

dataset_manifest_hash = stable_json_hash(dataset_manifest)
(RUN_DIR / "dataset_manifest.json").write_text(
    json.dumps(
        {"metadata_sha256": dataset_manifest_hash, "subjects": dataset_manifest},
        indent=2,
    ),
    encoding="utf-8",
)
initial_manifest["dataset_sha256"] = dataset_manifest_hash
(RUN_DIR / "run_manifest.json").write_text(
    json.dumps(initial_manifest, indent=2), encoding="utf-8"
)
display(pd.DataFrame(dataset_manifest))

sys.path.insert(0, str(UNIGAZE_PACKAGE_ROOT))
from datasets.mpiigaze import MPIIGazeDataset
from models.vit.mae_gaze import MAE_Gaze


def make_dataset(subjects: Sequence[str]) -> Dataset[Any]:
    return MPIIGazeDataset(
        dataset_path=str(NORMALIZED_DATA_ROOT),
        color_type="bgr",
        keys_to_use=[f"{subject}.h5" for subject in subjects],
        data_name="mpii",
        image_size=224,
        transform_type="basic_imagenet",
    )


preview_dataset = make_dataset(["p00"])
preview_rng = np.random.default_rng(CONFIG.seed)
preview_indices = preview_rng.choice(len(preview_dataset), size=9, replace=False)
mean = np.asarray([0.485, 0.456, 0.406])[:, None, None]
std = np.asarray([0.229, 0.224, 0.225])[:, None, None]
figure, axes = plt.subplots(3, 3, figsize=(10, 10))
for axis, index in zip(axes.flat, preview_indices, strict=True):
    item = preview_dataset[int(index)]
    image = item["image"].numpy() * std + mean
    image = np.clip(image.transpose(1, 2, 0), 0.0, 1.0)
    pitch, yaw = np.rad2deg(np.asarray(item["gaze"]))
    axis.imshow(image)
    axis.set_title(f"index={index}\npitch={pitch:.1f}°, yaw={yaw:.1f}°")
    axis.axis("off")
figure.suptitle("Normalized MPIIFaceGaze samples used by UniGaze")
figure.tight_layout()
preview_path = PLOT_ROOT / "normalized_sample_audit.png"
figure.savefig(preview_path, dpi=160, bbox_inches="tight")
plt.show()
del preview_dataset
gc.collect()

# %% [markdown]
# ## 5. Metrics, deterministic loaders, logging, and checkpoints
#
# Labels and predictions use `[pitch, yaw]` in radians. Angular error is the
# great-circle angle between their corresponding 3D gaze vectors and is reported
# in degrees. Test labels never select epochs or checkpoints.

# %%
def pitchyaw_to_vector(pitchyaw: Tensor) -> Tensor:
    if pitchyaw.ndim != 2 or pitchyaw.shape[1] != 2:
        raise ValueError(f"Expected [N, 2] pitch/yaw tensor, got {pitchyaw.shape}")
    sine, cosine = torch.sin(pitchyaw), torch.cos(pitchyaw)
    return torch.stack(
        (
            cosine[:, 0] * sine[:, 1],
            sine[:, 0],
            cosine[:, 0] * cosine[:, 1],
        ),
        dim=1,
    )


def angular_error_deg(prediction: Tensor, target: Tensor) -> Tensor:
    prediction_vector = F.normalize(pitchyaw_to_vector(prediction), dim=1)
    target_vector = F.normalize(pitchyaw_to_vector(target), dim=1)
    cosine = (prediction_vector * target_vector).sum(dim=1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int,
    confidence: float = 0.95,
    samples: int | None = None,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("bootstrap_mean_ci expects a non-empty 1D array")
    samples = samples or CONFIG.bootstrap_samples
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    remaining = samples
    while remaining:
        chunk = min(250, remaining)
        indices = rng.integers(0, len(values), size=(chunk, len(values)))
        means.append(values[indices].mean(axis=1))
        remaining -= chunk
    bootstrap_means = np.concatenate(means)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def summarize_errors(errors: np.ndarray, seed: int) -> dict[str, float]:
    errors = np.asarray(errors, dtype=np.float64)
    low, high = bootstrap_mean_ci(errors, seed=seed)
    return {
        "mean_deg": float(errors.mean()),
        "median_deg": float(np.median(errors)),
        "std_deg": float(errors.std(ddof=1)) if len(errors) > 1 else 0.0,
        "p90_deg": float(np.percentile(errors, 90)),
        "p95_deg": float(np.percentile(errors, 95)),
        "mean_ci95_low_deg": low,
        "mean_ci95_high_deg": high,
    }


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def limited_subset(dataset: Dataset[Any], limit: int, seed: int) -> Dataset[Any]:
    if limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def make_loader(
    dataset: Dataset[Any],
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator,
) -> DataLoader[Any]:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": CONFIG.num_workers,
        "pin_memory": True,
        "drop_last": shuffle,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": CONFIG.num_workers > 0,
    }
    if CONFIG.num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def create_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"unigaze_task0.{CONFIG_HASH}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


LOGGER = create_logger(LOG_ROOT / "run.log")


def load_wandb_api_key() -> str | None:
    """Read the W&B key without printing or persisting its value."""
    # Keep the standard W&B environment variable for non-Kaggle execution,
    # while allowing Kaggle's user-defined secret label to stay configurable.
    environment_key = os.environ.get("WANDB_API_KEY")
    if environment_key:
        return environment_key
    if not IS_KAGGLE:
        return None
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret(CONFIG.wandb_secret_label)
    except Exception as error:
        LOGGER.warning(
            "W&B secret %s is unavailable (%s)",
            CONFIG.wandb_secret_label,
            type(error).__name__,
        )
        return None


def initialize_wandb() -> Any | None:
    if not CONFIG.enable_wandb:
        LOGGER.info("W&B tracking is disabled by configuration")
        return None
    api_key = load_wandb_api_key()
    if not api_key:
        message = (
            f"W&B tracking requested but {CONFIG.wandb_secret_label!r} is not "
            "available. File and Kaggle logging remain active."
        )
        if CONFIG.wandb_required:
            raise RuntimeError(message)
        LOGGER.warning(message)
        return None

    os.environ["WANDB_API_KEY"] = api_key
    try:
        import wandb

        run = wandb.init(
            project=CONFIG.wandb_project,
            entity=CONFIG.wandb_entity,
            group=CONFIG.wandb_group,
            job_type=CONFIG.run_mode,
            name=RUN_ID,
            id=RUN_ID,
            resume="allow",
            config={
                **CONFIG_DICT,
                "config_hash": CONFIG_HASH,
                "source_sha256": NOTEBOOK_SOURCE_SHA256,
                "unigaze_revision": UNIGAZE_REVISION,
            },
            tags=["task0", "unigaze", "mpiifacegaze", CONFIG.run_mode],
            notes=(
                "Identity-disjoint Task 0 run. Smoke metrics validate plumbing "
                "only and are not benchmark results."
            ),
            dir=str(LOG_ROOT),
        )
    except Exception as error:
        message = f"W&B initialization failed ({type(error).__name__})"
        if CONFIG.wandb_required:
            raise RuntimeError(message) from error
        LOGGER.warning("%s; file and Kaggle logging remain active", message)
        return None
    run.define_metric("monitor/sequence")
    for pattern in ("train/*", "validation/*", "test/*", "system/*"):
        run.define_metric(pattern, step_metric="monitor/sequence")
    run.summary["provenance/config_sha256"] = CONFIG_HASH
    run.summary["provenance/source_sha256"] = NOTEBOOK_SOURCE_SHA256
    run.summary["provenance/checkpoint_sha256"] = environment["checkpoint_sha256"]
    run.summary["provenance/unigaze_revision"] = UNIGAZE_REVISION
    LOGGER.info("W&B tracking enabled: %s", run.url)
    return run


WANDB_RUN = initialize_wandb()
WANDB_SEQUENCE = 0
WANDB_LOGGING_ACTIVE = WANDB_RUN is not None


def wandb_log(values: dict[str, Any]) -> None:
    """Log without making a transient W&B failure fatal to training."""
    global WANDB_LOGGING_ACTIVE, WANDB_SEQUENCE
    if WANDB_RUN is None or not WANDB_LOGGING_ACTIVE:
        return
    payload = {"monitor/sequence": WANDB_SEQUENCE, **values}
    try:
        WANDB_RUN.log(payload)
        WANDB_SEQUENCE += 1
    except Exception as error:
        if CONFIG.wandb_required:
            raise
        WANDB_LOGGING_ACTIVE = False
        LOGGER.warning("W&B logging disabled after %s", type(error).__name__)


def wandb_alert(title: str, text: str, level: str = "INFO") -> None:
    if WANDB_RUN is None or not WANDB_LOGGING_ACTIVE or not CONFIG.wandb_alerts:
        return
    try:
        import wandb

        alert_level = getattr(wandb.AlertLevel, level.upper())
        WANDB_RUN.alert(
            title=title,
            text=text,
            level=alert_level,
            wait_duration=300,
        )
    except Exception as error:
        if CONFIG.wandb_required:
            raise
        LOGGER.warning("Could not send W&B alert (%s)", type(error).__name__)


def finish_wandb(exit_code: int) -> None:
    """Finish the remote run without hiding the primary experiment outcome."""
    if WANDB_RUN is None:
        return
    try:
        WANDB_RUN.finish(exit_code=exit_code)
    except Exception as error:
        if CONFIG.wandb_required:
            raise
        LOGGER.warning("Could not finish W&B run cleanly (%s)", type(error).__name__)


if WANDB_RUN is not None:
    initial_manifest["external_ids"]["wandb_run_id"] = WANDB_RUN.id
    initial_manifest["external_ids"]["wandb_run_url"] = WANDB_RUN.url
    (RUN_DIR / "run_manifest.json").write_text(
        json.dumps(initial_manifest, indent=2), encoding="utf-8"
    )
    wandb_log(
        {
            "system/gpu_memory_total_gib": environment["gpu_memory_bytes"]
            / (1024**3),
            "system/dataset_subjects": len(ACTIVE_SUBJECTS),
            "system/dataset_samples": int(
                sum(item["samples"] for item in dataset_manifest)
            ),
        }
    )


def atomic_json_write(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_csv_write(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)

# %% [markdown]
# ## 6. Full-model training and evaluation functions
#
# Every parameter is trainable. The checkpoint selector is validation mean
# angular error. Test evaluation happens once, after the best validation
# checkpoint has been restored.

# %%
def make_model() -> nn.Module:
    model = MAE_Gaze(
        model_type=CONFIG.model_type,
        global_pool=False,
        drop_path_rate=CONFIG.drop_path_rate,
        custom_pretrained_path=str(MAE_CHECKPOINT),
    ).to(DEVICE)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable != total:
        raise RuntimeError(f"Expected full fine-tuning, trainable={trainable}, total={total}")
    LOGGER.info("Model parameters: trainable=%d total=%d", trainable, total)
    return model


def make_grad_scaler() -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=CONFIG.use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=CONFIG.use_amp)


def amp_optimizer_step(
    scaler: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> bool:
    """Run one AMP optimizer step and advance the LR schedule only on success.

    ``GradScaler`` skips ``optimizer.step()`` when it detects an overflow and
    lowers its scale. Advancing ``OneCycleLR`` on such a skipped update makes
    the learning-rate schedule run ahead of the actual parameter updates and
    triggers PyTorch's scheduler-order warning on the first overflow.

    Returns ``True`` when the optimizer update was executed and ``False`` when
    AMP skipped it.
    """
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    optimizer_step_completed = float(scaler.get_scale()) >= scale_before
    if optimizer_step_completed:
        scheduler.step()
    return optimizer_step_completed


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    epoch: int,
    fold: str,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    angular_sum = 0.0
    sample_count = 0
    skipped_optimizer_steps = 0
    start = time.perf_counter()
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for step, batch in enumerate(progress, start=1):
        images = batch["image"].float().to(DEVICE, non_blocking=True)
        targets = batch["gaze"].float().to(DEVICE, non_blocking=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=CONFIG.use_amp
        ):
            predictions = model(images)["pred_gaze"]
            loss = F.l1_loss(predictions, targets)
            scaled_loss = loss / CONFIG.gradient_accumulation_steps
        if not torch.isfinite(loss):
            wandb_alert(
                "Task 0 stopped: non-finite loss",
                f"fold={fold}, epoch={epoch}, batch={step}",
                level="ERROR",
            )
            raise FloatingPointError(f"Non-finite training loss at epoch={epoch}, step={step}")
        scaler.scale(scaled_loss).backward()
        should_update = (
            step % CONFIG.gradient_accumulation_steps == 0 or step == len(loader)
        )
        if should_update:
            if CONFIG.gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), CONFIG.gradient_clip_norm
                )
            if not amp_optimizer_step(scaler, optimizer, scheduler):
                skipped_optimizer_steps += 1
                LOGGER.warning(
                    "AMP skipped optimizer update: fold=%s epoch=%d batch=%d",
                    fold,
                    epoch,
                    step,
                )
        batch_errors = angular_error_deg(predictions.detach(), targets)
        batch_size = len(targets)
        loss_sum += float(loss.detach()) * batch_size
        angular_sum += float(batch_errors.sum())
        sample_count += batch_size
        if step % CONFIG.wandb_log_every_steps == 0 or step == len(loader):
            wandb_log(
                {
                    "train/fold": fold,
                    "train/epoch": epoch,
                    "train/batch": step,
                    "train/batches_in_epoch": len(loader),
                    "train/samples_seen_in_epoch": sample_count,
                    "train/batch_loss_rad": float(loss.detach()),
                    "train/skipped_optimizer_steps": skipped_optimizer_steps,
                    "train/batch_angular_error_deg": float(batch_errors.mean()),
                    "train/running_loss_rad": loss_sum / sample_count,
                    "train/running_angular_error_deg": angular_sum / sample_count,
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/amp_scale": float(scaler.get_scale()),
                    "system/gpu_memory_allocated_gib": (
                        torch.cuda.memory_allocated() / (1024**3)
                    ),
                    "system/gpu_memory_reserved_gib": (
                        torch.cuda.memory_reserved() / (1024**3)
                    ),
                }
            )
        if step % CONFIG.log_every_steps == 0 or step == len(loader):
            LOGGER.info(
                "epoch=%d step=%d/%d train_loss=%.6f train_angle_deg=%.4f lr=%.3e",
                epoch,
                step,
                len(loader),
                loss_sum / sample_count,
                angular_sum / sample_count,
                optimizer.param_groups[0]["lr"],
            )
    return {
        "train_loss": loss_sum / sample_count,
        "train_mean_deg": angular_sum / sample_count,
        "epoch_seconds": time.perf_counter() - start,
        "skipped_optimizer_steps": float(skipped_optimizer_steps),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    subject: str,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    records: list[pd.DataFrame] = []
    for batch in tqdm(loader, desc=f"evaluate {subject}", leave=False):
        images = batch["image"].float().to(DEVICE, non_blocking=True)
        targets = batch["gaze"].float().to(DEVICE, non_blocking=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=CONFIG.use_amp
        ):
            predictions = model(images)["pred_gaze"]
            loss = F.l1_loss(predictions, targets)
        errors = angular_error_deg(predictions.float(), targets).cpu().numpy()
        predictions_np = predictions.float().cpu().numpy()
        targets_np = targets.cpu().numpy()
        indices = np.asarray(batch["index"], dtype=np.int64)
        records.append(
            pd.DataFrame(
                {
                    "subject": subject,
                    "sample_index": indices,
                    "target_pitch_rad": targets_np[:, 0],
                    "target_yaw_rad": targets_np[:, 1],
                    "pred_pitch_rad": predictions_np[:, 0],
                    "pred_yaw_rad": predictions_np[:, 1],
                    "angular_error_deg": errors,
                }
            )
        )
        loss_sum += float(loss) * len(targets)
        sample_count += len(targets)
    predictions_frame = pd.concat(records, ignore_index=True)
    metrics = summarize_errors(
        predictions_frame["angular_error_deg"].to_numpy(), seed=seed
    )
    metrics["l1_loss_rad"] = loss_sum / sample_count
    metrics["samples"] = int(sample_count)
    return metrics, predictions_frame


def training_gaze_median(subjects: Sequence[str]) -> np.ndarray:
    labels: list[np.ndarray] = []
    for subject in subjects:
        with h5py.File(NORMALIZED_DATA_ROOT / f"{subject}.h5", "r", swmr=True) as h5:
            labels.append(h5["face_gaze"][:].astype(np.float32))
    return np.median(np.concatenate(labels, axis=0), axis=0).astype(np.float32)


def baseline_errors(targets: np.ndarray, median_angles: np.ndarray) -> np.ndarray:
    target_tensor = torch.from_numpy(targets.astype(np.float32))
    baseline_tensor = torch.from_numpy(
        np.broadcast_to(median_angles, targets.shape).copy()
    )
    return angular_error_deg(baseline_tensor, target_tensor).numpy()


@torch.inference_mode()
def benchmark_inference(model: nn.Module, batch: Tensor) -> dict[str, float]:
    model.eval()
    batch = batch.to(DEVICE, non_blocking=True)
    warmup = 3 if CONFIG.run_mode == "smoke" else CONFIG.latency_warmup_steps
    steps = 10 if CONFIG.run_mode == "smoke" else CONFIG.latency_measurement_steps
    for _ in range(warmup):
        model(batch)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        model(batch)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    milliseconds = elapsed * 1_000.0 / steps
    return {
        "model_batch_size": int(len(batch)),
        "model_batch_latency_ms": milliseconds,
        "model_per_image_latency_ms": milliseconds / len(batch),
        "model_throughput_images_per_second": len(batch) * steps / elapsed,
    }


def plot_fold_history(history: pd.DataFrame, destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="train")
    axes[0].plot(history["epoch"], history["validation_loss"], marker="o", label="validation")
    axes[0].set(title="L1 loss", xlabel="epoch", ylabel="radians")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(history["epoch"], history["train_mean_deg"], marker="o", label="train")
    axes[1].plot(history["epoch"], history["validation_mean_deg"], marker="o", label="validation")
    axes[1].set(title="Angular error", xlabel="epoch", ylabel="degrees")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=170, bbox_inches="tight")
    plt.show()


def plot_qualitative_samples(
    subject: str,
    predictions: pd.DataFrame,
    destination: Path,
) -> None:
    ordered = predictions.sort_values("angular_error_deg").reset_index(drop=True)
    positions = np.linspace(0, len(ordered) - 1, 9).round().astype(int)
    chosen = ordered.iloc[positions]
    dataset = make_dataset([subject])
    figure, axes = plt.subplots(3, 3, figsize=(11, 11))
    for axis, (_, row) in zip(axes.flat, chosen.iterrows(), strict=True):
        item = dataset[int(row["sample_index"])]
        image = item["image"].numpy() * std + mean
        image = np.clip(image.transpose(1, 2, 0), 0.0, 1.0)
        target_deg = np.rad2deg([row["target_pitch_rad"], row["target_yaw_rad"]])
        pred_deg = np.rad2deg([row["pred_pitch_rad"], row["pred_yaw_rad"]])
        axis.imshow(image)
        axis.set_title(
            f"error={row['angular_error_deg']:.2f}°\n"
            f"target=({target_deg[0]:.1f}, {target_deg[1]:.1f})°\n"
            f"pred=({pred_deg[0]:.1f}, {pred_deg[1]:.1f})°"
        )
        axis.axis("off")
    figure.suptitle(f"{subject}: error quantiles from best to worst")
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.show()
    del dataset

# %% [markdown]
# ## 7. One subject-independent fold
#
# For test subject `pXX`, the next subject cyclically is validation and the other
# 13 subjects are training data. This is a strict identity-disjoint 13/1/1
# protocol. It is intentionally labelled separately from published results that
# use a different split. The validation subject selects the checkpoint; the test
# subject is evaluated only after training.

# %%
def fold_subjects(test_subject: str) -> tuple[list[str], str]:
    if CONFIG.run_mode == "smoke":
        if test_subject != "p00":
            raise ValueError("Smoke protocol is fixed to test=p00")
        return ["p02"], "p01"
    test_index = SUBJECTS.index(test_subject)
    validation_subject = SUBJECTS[(test_index + 1) % len(SUBJECTS)]
    training_subjects = [
        subject
        for subject in SUBJECTS
        if subject not in {test_subject, validation_subject}
    ]
    return training_subjects, validation_subject


def run_fold(test_subject: str) -> dict[str, Any]:
    fold_seed = CONFIG.seed + SUBJECTS.index(test_subject)
    set_seed(fold_seed)
    training_subjects, validation_subject = fold_subjects(test_subject)
    fold_dir = CHECKPOINT_ROOT / test_subject
    fold_dir.mkdir(parents=True, exist_ok=True)
    summary_path = fold_dir / "fold_summary.json"
    prediction_path = METRIC_ROOT / f"predictions_{test_subject}.csv"
    history_path = METRIC_ROOT / f"history_{test_subject}.csv"
    curve_path = PLOT_ROOT / f"learning_curves_{test_subject}.png"
    qualitative_path = PLOT_ROOT / f"qualitative_{test_subject}.png"
    resume_path = fold_dir / "resume.pt"
    best_model_path = fold_dir / "best_model.pt"

    if summary_path.is_file() and prediction_path.is_file():
        LOGGER.info("Skipping completed fold %s", test_subject)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    train_dataset: Dataset[Any] = make_dataset(training_subjects)
    validation_dataset: Dataset[Any] = make_dataset([validation_subject])
    test_dataset: Dataset[Any] = make_dataset([test_subject])
    epochs = CONFIG.epochs
    if CONFIG.run_mode == "smoke":
        train_dataset = limited_subset(train_dataset, CONFIG.train_subject_limit, fold_seed)
        validation_dataset = limited_subset(
            validation_dataset, CONFIG.validation_subject_limit, fold_seed + 1
        )
        test_dataset = limited_subset(test_dataset, CONFIG.test_subject_limit, fold_seed + 2)
        epochs = CONFIG.smoke_epochs

    loader_generator = torch.Generator().manual_seed(fold_seed)
    train_loader = make_loader(
        train_dataset, CONFIG.batch_size, shuffle=True, generator=loader_generator
    )
    validation_loader = make_loader(
        validation_dataset,
        CONFIG.eval_batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(fold_seed + 1),
    )
    test_loader = make_loader(
        test_dataset,
        CONFIG.eval_batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(fold_seed + 2),
    )

    model = make_model()
    if WANDB_RUN is not None and CONFIG.wandb_watch_gradients:
        try:
            WANDB_RUN.watch(
                model,
                log="gradients",
                log_freq=CONFIG.wandb_gradient_log_frequency,
                idx=SUBJECTS.index(test_subject),
                log_graph=False,
            )
        except Exception as error:
            if CONFIG.wandb_required:
                raise
            LOGGER.warning("W&B gradient watch disabled (%s)", type(error).__name__)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / CONFIG.gradient_accumulation_steps
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG.learning_rate,
        epochs=epochs,
        steps_per_epoch=updates_per_epoch,
        div_factor=CONFIG.one_cycle_div_factor,
        final_div_factor=CONFIG.one_cycle_final_div_factor,
    )
    scaler = make_grad_scaler()
    start_epoch = 1
    best_validation = float("inf")
    best_epoch = 0
    history_rows: list[dict[str, Any]] = []

    if CONFIG.resume and resume_path.is_file():
        checkpoint = torch_load(resume_path)
        if checkpoint["config_hash"] != CONFIG_HASH:
            raise RuntimeError(f"Configuration mismatch in {resume_path}")
        if checkpoint.get("source_sha256") != NOTEBOOK_SOURCE_SHA256:
            raise RuntimeError(f"Notebook source mismatch in {resume_path}")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        optimizer_to(optimizer, DEVICE)
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        loader_generator.set_state(checkpoint["loader_generator_state"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint["best_validation_mean_deg"])
        best_epoch = int(checkpoint["best_epoch"])
        history_rows = list(checkpoint["history"])
        LOGGER.info("Resumed %s from epoch %d", test_subject, start_epoch)

    torch.cuda.reset_peak_memory_stats()
    fold_start = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, epoch, test_subject
        )
        validation_metrics, _ = evaluate(
            model, validation_loader, validation_subject, fold_seed + epoch
        )
        row = {
            "test_subject": test_subject,
            "validation_subject": validation_subject,
            "epoch": epoch,
            **train_metrics,
            "validation_loss": validation_metrics["l1_loss_rad"],
            "validation_mean_deg": validation_metrics["mean_deg"],
            "validation_median_deg": validation_metrics["median_deg"],
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        }
        history_rows.append(row)
        atomic_csv_write(pd.DataFrame(history_rows), history_path)
        improved = validation_metrics["mean_deg"] < best_validation
        if improved:
            best_validation = validation_metrics["mean_deg"]
            best_epoch = epoch
            atomic_torch_save(
                {
                    "model_state": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "epoch": epoch,
                    "validation_mean_deg": best_validation,
                    "config_hash": CONFIG_HASH,
                    "source_sha256": NOTEBOOK_SOURCE_SHA256,
                    "test_subject": test_subject,
                    "validation_subject": validation_subject,
                },
                best_model_path,
            )
        atomic_torch_save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "loader_generator_state": loader_generator.get_state(),
                "rng_state": capture_rng_state(),
                "epoch": epoch,
                "best_validation_mean_deg": best_validation,
                "best_epoch": best_epoch,
                "history": history_rows,
                "config_hash": CONFIG_HASH,
                "source_sha256": NOTEBOOK_SOURCE_SHA256,
            },
            resume_path,
        )
        LOGGER.info(
            "fold=%s epoch=%d train=%.4f° validation=%.4f° best=%.4f°@%d",
            test_subject,
            epoch,
            train_metrics["train_mean_deg"],
            validation_metrics["mean_deg"],
            best_validation,
            best_epoch,
        )
        wandb_log(
            {
                "validation/fold": test_subject,
                "validation/subject": validation_subject,
                "validation/epoch": epoch,
                "validation/loss_rad": validation_metrics["l1_loss_rad"],
                "validation/mean_angular_error_deg": validation_metrics["mean_deg"],
                "validation/median_angular_error_deg": validation_metrics["median_deg"],
                "validation/std_angular_error_deg": validation_metrics["std_deg"],
                "validation/p90_angular_error_deg": validation_metrics["p90_deg"],
                "validation/p95_angular_error_deg": validation_metrics["p95_deg"],
                "validation/best_mean_angular_error_deg": best_validation,
                "validation/best_epoch": best_epoch,
                "validation/improved": int(improved),
                "train/epoch_loss_rad": train_metrics["train_loss"],
                "train/epoch_angular_error_deg": train_metrics["train_mean_deg"],
                "train/epoch_seconds": train_metrics["epoch_seconds"],
                "system/peak_gpu_memory_gib": (
                    torch.cuda.max_memory_allocated() / (1024**3)
                ),
            }
        )

    if not best_model_path.is_file():
        raise RuntimeError(f"No best model checkpoint was produced for {test_subject}")
    best_checkpoint = torch_load(best_model_path)
    model.load_state_dict(best_checkpoint["model_state"])
    test_metrics, predictions = evaluate(
        model, test_loader, test_subject, fold_seed + 10_000
    )

    median_angles = training_gaze_median(training_subjects)
    targets = predictions[["target_pitch_rad", "target_yaw_rad"]].to_numpy()
    constant_errors = baseline_errors(targets, median_angles)
    constant_metrics = summarize_errors(constant_errors, seed=fold_seed + 20_000)
    predictions["constant_baseline_error_deg"] = constant_errors
    atomic_csv_write(predictions, prediction_path)

    first_batch = next(iter(test_loader))["image"].float()
    latency = benchmark_inference(model, first_batch)
    history = pd.DataFrame(history_rows)
    plot_fold_history(history, curve_path)
    plot_qualitative_samples(test_subject, predictions, qualitative_path)

    fold_summary: dict[str, Any] = {
        "status": "smoke_only" if CONFIG.run_mode == "smoke" else "complete",
        "protocol": (
            "smoke-only: 1 train / 1 validation / 1 test"
            if CONFIG.run_mode == "smoke"
            else "identity-disjoint 13 train / 1 validation / 1 test"
        ),
        "test_subject": test_subject,
        "validation_subject": validation_subject,
        "training_subjects": training_subjects,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "test_samples": len(test_dataset),
        "best_epoch": best_epoch,
        "best_validation_mean_deg": best_validation,
        "test": test_metrics,
        "constant_baseline": constant_metrics,
        "latency": latency,
        "fold_seconds": time.perf_counter() - fold_start,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "best_model_path": str(best_model_path),
        "history_path": str(history_path),
        "predictions_path": str(prediction_path),
        "config_hash": CONFIG_HASH,
        "source_sha256": NOTEBOOK_SOURCE_SHA256,
        "dataset_manifest_hash": dataset_manifest_hash,
    }
    atomic_json_write(fold_summary, summary_path)
    wandb_log(
        {
            "test/fold": test_subject,
            "test/subject": test_subject,
            "test/best_epoch": best_epoch,
            "test/best_validation_mean_deg": best_validation,
            "test/mean_angular_error_deg": test_metrics["mean_deg"],
            "test/median_angular_error_deg": test_metrics["median_deg"],
            "test/std_angular_error_deg": test_metrics["std_deg"],
            "test/p90_angular_error_deg": test_metrics["p90_deg"],
            "test/p95_angular_error_deg": test_metrics["p95_deg"],
            "test/mean_ci95_low_deg": test_metrics["mean_ci95_low_deg"],
            "test/mean_ci95_high_deg": test_metrics["mean_ci95_high_deg"],
            "test/constant_baseline_mean_deg": constant_metrics["mean_deg"],
            "test/model_batch_latency_ms": latency["model_batch_latency_ms"],
            "test/model_per_image_latency_ms": latency[
                "model_per_image_latency_ms"
            ],
            "test/model_throughput_images_per_second": latency[
                "model_throughput_images_per_second"
            ],
            "system/fold_seconds": fold_summary["fold_seconds"],
            "system/peak_gpu_memory_gib": fold_summary["peak_gpu_memory_gib"],
        }
    )
    wandb_alert(
        title=f"Task 0 fold {test_subject} completed",
        text=(
            f"test mean={test_metrics['mean_deg']:.3f}°, "
            f"best validation={best_validation:.3f}° at epoch {best_epoch}"
        ),
    )
    if not CONFIG.keep_resume_checkpoint_after_completion:
        resume_path.unlink(missing_ok=True)

    if WANDB_RUN is not None and CONFIG.wandb_watch_gradients:
        try:
            WANDB_RUN.unwatch(model)
        except Exception as error:
            if CONFIG.wandb_required:
                raise
            LOGGER.warning("Could not remove W&B model hooks (%s)", type(error).__name__)

    del model, optimizer, scheduler, scaler
    del train_loader, validation_loader, test_loader
    del train_dataset, validation_dataset, test_dataset
    torch.cuda.empty_cache()
    gc.collect()
    return fold_summary

# %% [markdown]
# ## 8. Run smoke, one fold, or complete LOPO
#
# - `smoke`: one short epoch and three small identity-disjoint subsets for QA;
# - `single_fold`: all samples for `p00` as test;
# - `full_lopo`: all 15 identity-disjoint folds.
#
# Completed folds are skipped on rerun. An interrupted active fold resumes from
# its last epoch, including optimizer, scheduler, AMP scaler, loader generator,
# and random-number-generator states.

# %%
if CONFIG.run_mode in {"smoke", "single_fold"}:
    test_subjects = [CONFIG.single_test_subject]
else:
    test_subjects = list(SUBJECTS)

fold_summaries = []
try:
    for test_subject in test_subjects:
        fold_summaries.append(run_fold(test_subject))
except Exception as error:
    initial_manifest["status"] = "failed"
    initial_manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    initial_manifest["failure"] = {
        "exception_type": type(error).__name__,
        "message": str(error),
    }
    atomic_json_write(initial_manifest, RUN_DIR / "run_manifest.json")
    wandb_alert(
        "Task 0 training failed",
        f"{type(error).__name__}: {error}",
        level="ERROR",
    )
    if WANDB_RUN is not None:
        WANDB_RUN.summary["run/status"] = "failed"
        WANDB_RUN.summary["run/failure_type"] = type(error).__name__
    finish_wandb(exit_code=1)
    raise

per_subject_rows = []
prediction_frames = []
history_frames = []
for fold_summary in fold_summaries:
    subject = fold_summary["test_subject"]
    per_subject_rows.append(
        {
            "test_subject": subject,
            "validation_subject": fold_summary["validation_subject"],
            "best_epoch": fold_summary["best_epoch"],
            "samples": fold_summary["test_samples"],
            **fold_summary["test"],
            "constant_baseline_mean_deg": fold_summary["constant_baseline"]["mean_deg"],
            "fold_seconds": fold_summary["fold_seconds"],
            "peak_gpu_memory_gib": fold_summary["peak_gpu_memory_gib"],
        }
    )
    prediction_frames.append(
        pd.read_csv(METRIC_ROOT / f"predictions_{subject}.csv")
    )
    history_frames.append(pd.read_csv(METRIC_ROOT / f"history_{subject}.csv"))

per_subject_metrics = pd.DataFrame(per_subject_rows)
all_predictions = pd.concat(prediction_frames, ignore_index=True)
all_history = pd.concat(history_frames, ignore_index=True)
atomic_csv_write(per_subject_metrics, METRIC_ROOT / "per_subject_metrics.csv")
atomic_csv_write(all_predictions, METRIC_ROOT / "all_predictions.csv")
atomic_csv_write(all_history, METRIC_ROOT / "all_history.csv")

subject_mean_errors = per_subject_metrics["mean_deg"].to_numpy()
subject_ci = bootstrap_mean_ci(subject_mean_errors, seed=CONFIG.seed + 30_000)
final_summary = {
    "status": "smoke_only" if CONFIG.run_mode == "smoke" else "complete",
    "experiment": "full UniGaze MAE-B16 fine-tuning",
    "run_mode": CONFIG.run_mode,
    "protocol": (
        "smoke-only: 1 train / 1 validation / 1 test"
        if CONFIG.run_mode == "smoke"
        else "identity-disjoint 13 train / 1 validation / 1 test"
    ),
    "completed_subjects": test_subjects,
    "subject_count": len(test_subjects),
    "subject_mean_angular_error_deg": float(subject_mean_errors.mean()),
    "subject_std_angular_error_deg": (
        float(subject_mean_errors.std(ddof=1)) if len(subject_mean_errors) > 1 else 0.0
    ),
    "subject_mean_ci95_low_deg": subject_ci[0],
    "subject_mean_ci95_high_deg": subject_ci[1],
    "pooled_sample_metrics": summarize_errors(
        all_predictions["angular_error_deg"].to_numpy(), seed=CONFIG.seed + 40_000
    ),
    "constant_baseline_subject_mean_deg": float(
        per_subject_metrics["constant_baseline_mean_deg"].mean()
    ),
    "config_hash": CONFIG_HASH,
    "source_sha256": NOTEBOOK_SOURCE_SHA256,
    "dataset_manifest_hash": dataset_manifest_hash,
    "run_dir": str(RUN_DIR),
}
atomic_json_write(final_summary, RUN_DIR / "summary.json")
wandb_log(
    {
        "test/subject_mean_angular_error_deg": final_summary[
            "subject_mean_angular_error_deg"
        ],
        "test/subject_std_angular_error_deg": final_summary[
            "subject_std_angular_error_deg"
        ],
        "test/subject_mean_ci95_low_deg": final_summary[
            "subject_mean_ci95_low_deg"
        ],
        "test/subject_mean_ci95_high_deg": final_summary[
            "subject_mean_ci95_high_deg"
        ],
        "test/pooled_mean_angular_error_deg": final_summary[
            "pooled_sample_metrics"
        ]["mean_deg"],
        "test/pooled_median_angular_error_deg": final_summary[
            "pooled_sample_metrics"
        ]["median_deg"],
        "test/constant_baseline_subject_mean_deg": final_summary[
            "constant_baseline_subject_mean_deg"
        ],
    }
)
display(per_subject_metrics)
print(json.dumps(final_summary, indent=2))

# %% [markdown]
# ## 9. Final figures
#
# These figures are generated from saved CSV files, so they can be rebuilt
# without retraining. The subject plot compares the model with the constant
# training-set median-angle baseline. Distribution plots show both histogram and
# empirical CDF. The scatter plot exposes pitch/yaw bias and compression.

# %%
figure, axis = plt.subplots(figsize=(max(8, len(per_subject_metrics) * 0.65), 5))
positions = np.arange(len(per_subject_metrics))
width = 0.38
axis.bar(
    positions - width / 2,
    per_subject_metrics["mean_deg"],
    width,
    label="UniGaze-B16 fine-tuned",
)
axis.bar(
    positions + width / 2,
    per_subject_metrics["constant_baseline_mean_deg"],
    width,
    label="constant baseline",
)
axis.set_xticks(positions, per_subject_metrics["test_subject"])
axis.set(ylabel="mean angular error (degrees)", title="Per-subject test error")
axis.grid(axis="y", alpha=0.25)
axis.legend()
figure.tight_layout()
per_subject_plot = PLOT_ROOT / "per_subject_comparison.png"
figure.savefig(per_subject_plot, dpi=180, bbox_inches="tight")
plt.show()

errors = np.sort(all_predictions["angular_error_deg"].to_numpy())
figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].hist(errors, bins=50, alpha=0.85)
axes[0].set(xlabel="angular error (degrees)", ylabel="samples", title="Test error distribution")
axes[0].grid(alpha=0.2)
axes[1].plot(errors, np.arange(1, len(errors) + 1) / len(errors))
axes[1].set(xlabel="angular error (degrees)", ylabel="empirical CDF", title="Test error CDF")
axes[1].grid(alpha=0.2)
figure.tight_layout()
distribution_plot = PLOT_ROOT / "error_distribution.png"
figure.savefig(distribution_plot, dpi=180, bbox_inches="tight")
plt.show()

plot_frame = all_predictions
if len(plot_frame) > 10_000:
    plot_frame = plot_frame.sample(10_000, random_state=CONFIG.seed)
figure, axes = plt.subplots(1, 2, figsize=(12, 5))
for axis, angle in zip(axes, ("pitch", "yaw"), strict=True):
    target_deg = np.rad2deg(plot_frame[f"target_{angle}_rad"])
    pred_deg = np.rad2deg(plot_frame[f"pred_{angle}_rad"])
    low = min(float(target_deg.min()), float(pred_deg.min()))
    high = max(float(target_deg.max()), float(pred_deg.max()))
    axis.scatter(target_deg, pred_deg, s=6, alpha=0.25)
    axis.plot([low, high], [low, high], "--", color="black", linewidth=1)
    axis.set(
        xlabel=f"target {angle} (degrees)",
        ylabel=f"predicted {angle} (degrees)",
        title=f"{angle.capitalize()}: prediction vs target",
    )
    axis.grid(alpha=0.2)
figure.tight_layout()
scatter_plot = PLOT_ROOT / "prediction_scatter.png"
figure.savefig(scatter_plot, dpi=180, bbox_inches="tight")
plt.show()

print("Saved experiment artifacts:")
for path in sorted(RUN_DIR.rglob("*")):
    if path.is_file():
        print(path.relative_to(RUN_DIR), f"({path.stat().st_size / 1024:.1f} KiB)")

initial_manifest["status"] = "completed"
initial_manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
atomic_json_write(initial_manifest, RUN_DIR / "run_manifest.json")

if WANDB_RUN is not None:
    WANDB_RUN.summary["run/status"] = final_summary["status"]
    WANDB_RUN.summary["test/subject_mean_angular_error_deg"] = final_summary[
        "subject_mean_angular_error_deg"
    ]
    WANDB_RUN.summary["test/pooled_mean_angular_error_deg"] = final_summary[
        "pooled_sample_metrics"
    ]["mean_deg"]
    WANDB_RUN.summary["test/constant_baseline_subject_mean_deg"] = final_summary[
        "constant_baseline_subject_mean_deg"
    ]

    if CONFIG.wandb_log_aggregate_artifact and WANDB_LOGGING_ACTIVE:
        try:
            import wandb

            evaluation_artifact = wandb.Artifact(
                name=f"task0-evaluation-{RUN_ID}",
                type="evaluation",
                metadata={
                    "run_mode": CONFIG.run_mode,
                    "config_hash": CONFIG_HASH,
                    "source_sha256": NOTEBOOK_SOURCE_SHA256,
                    "dataset_manifest_hash": dataset_manifest_hash,
                    "privacy": (
                        "aggregate-only: no face images, checkpoints, or "
                        "per-sample predictions"
                    ),
                },
            )
            aggregate_artifact_paths = [
                RUN_DIR / "resolved_config.json",
                RUN_DIR / "environment.json",
                RUN_DIR / "source_manifest.json",
                RUN_DIR / "dataset_manifest.json",
                RUN_DIR / "run_manifest.json",
                RUN_DIR / "summary.json",
                METRIC_ROOT / "per_subject_metrics.csv",
                METRIC_ROOT / "all_history.csv",
                per_subject_plot,
                distribution_plot,
            ]
            for artifact_path in aggregate_artifact_paths:
                if artifact_path.exists():
                    evaluation_artifact.add_file(
                        str(artifact_path),
                        name=str(artifact_path.relative_to(RUN_DIR)),
                    )
            WANDB_RUN.log_artifact(
                evaluation_artifact,
                aliases=[CONFIG.run_mode, "latest"],
            )
        except Exception as error:
            if CONFIG.wandb_required:
                raise
            LOGGER.warning(
                "Could not upload aggregate W&B artifact (%s)",
                type(error).__name__,
            )

wandb_alert(
    "Task 0 run completed",
    (
        f"run={RUN_ID}, mode={CONFIG.run_mode}, "
        f"subject mean error={final_summary['subject_mean_angular_error_deg']:.3f} deg"
    ),
    level="INFO",
)
finish_wandb(exit_code=0)
print(f"Run completed: {RUN_ID}")

# %% [markdown]
# ## Interpretation checklist
#
# Before reporting a result:
#
# 1. Confirm `status == "complete"`; never report a `smoke_only` result.
# 2. State the active identity-disjoint protocol: smoke is 1/1/1; full runs
#    use 13/1/1.
# 3. Do not compare its number directly with a paper using a different split.
# 4. Report subject mean and dispersion, not only the pooled sample mean.
# 5. Compare with the constant baseline and the frozen linear probe.
# 6. Record failed or interrupted runs instead of silently discarding them.
# 7. Treat the A100 latency as model-only training-host latency, not consumer
#    webcam end-to-end latency.
