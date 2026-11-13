# %% [markdown]
# # Task 1 — MAE-B16 on GazeGene
#
# This notebook runs the fixed identity-disjoint GazeGene baseline and can then
# evaluate the selected full checkpoint zero-shot on normalized MPIIFaceGaze.
# It does not download or redistribute GazeGene. Set paths through the
# `TASK1_*` environment variables before running on Kaggle or the private GPU
# server. Full training is intended for the RTX 4070 Ti Super server.

# %%
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# UniGaze loads the trusted official MAE checkpoint without an explicit
# weights_only argument; PyTorch 2.6 otherwise changes that call's default.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable: Iterable[Any], **_: Any) -> Iterable[Any]:
        return iterable


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_root: str = os.environ.get(
        "TASK1_GAZEGENE_ROOT",
        str(Path.cwd() / "data" / "GazeGene" / "extracted" / "GazeGene_FaceCrops"),
    )
    unigaze_root: str = os.environ.get(
        "TASK1_UNIGAZE_ROOT", str(Path.cwd() / "external" / "UniGaze" / "unigaze")
    )
    mae_checkpoint: str = os.environ.get(
        "TASK1_MAE_CHECKPOINT",
        str(Path.cwd() / "models" / "pretrained" / "mae_b16_checkpoint-299.pth"),
    )
    mpii_root: str | None = os.environ.get("TASK1_MPII_ROOT")
    full_checkpoint: str | None = os.environ.get("TASK1_FULL_CHECKPOINT")
    output_root: str = os.environ.get(
        "TASK1_OUTPUT_ROOT", str(Path.cwd() / "outputs" / "task1_gazegene_b16")
    )
    run_id: str | None = os.environ.get("TASK1_RUN_ID")
    resume: bool = env_bool("TASK1_RESUME", True)
    run_mode: str = os.environ.get(
        "TASK1_RUN_MODE", "smoke"
    )  # smoke | train | zero_shot | train_and_zero_shot
    seed: int = int(os.environ.get("TASK1_SEED", "20261112"))
    epochs: int = int(os.environ.get("TASK1_EPOCHS", "12"))
    early_stopping_patience: int = int(os.environ.get("TASK1_PATIENCE", "3"))
    batch_size: int = int(os.environ.get("TASK1_BATCH_SIZE", "24"))
    eval_batch_size: int = int(os.environ.get("TASK1_EVAL_BATCH_SIZE", "64"))
    gradient_accumulation_steps: int = int(os.environ.get("TASK1_ACCUMULATION", "2"))
    num_workers: int = int(os.environ.get("TASK1_NUM_WORKERS", "4"))
    backbone_learning_rate: float = float(os.environ.get("TASK1_BACKBONE_LR", "3e-6"))
    head_learning_rate: float = float(os.environ.get("TASK1_HEAD_LR", "3e-5"))
    weight_decay: float = float(os.environ.get("TASK1_WEIGHT_DECAY", "1e-4"))
    warmup_fraction: float = float(os.environ.get("TASK1_WARMUP_FRACTION", "0.05"))
    use_amp: bool = env_bool("TASK1_USE_AMP", True)
    smoke_samples_per_split: int = int(os.environ.get("TASK1_SMOKE_SAMPLES", "256"))
    bootstrap_replicates: int = int(os.environ.get("TASK1_BOOTSTRAP_REPLICATES", "10000"))
    expected_frames_per_camera: int = 2_000
    train_subjects: tuple[int, ...] = tuple(range(1, 41))
    validation_subjects: tuple[int, ...] = tuple(range(41, 47))
    test_subjects: tuple[int, ...] = tuple(range(47, 57))


CONFIG = ExperimentConfig()
if CONFIG.run_mode not in {"smoke", "train", "zero_shot", "train_and_zero_shot"}:
    raise ValueError(f"Unsupported TASK1_RUN_MODE: {CONFIG.run_mode}")
if min(CONFIG.batch_size, CONFIG.eval_batch_size, CONFIG.gradient_accumulation_steps) < 1:
    raise ValueError("Batch sizes and accumulation must be positive")
if CONFIG.bootstrap_replicates < 1:
    raise ValueError("TASK1_BOOTSTRAP_REPLICATES must be positive")

DATASET_ROOT = Path(CONFIG.dataset_root).expanduser().resolve()
UNIGAZE_ROOT = Path(CONFIG.unigaze_root).expanduser().resolve()
MAE_CHECKPOINT = Path(CONFIG.mae_checkpoint).expanduser().resolve()
OUTPUT_ROOT = Path(CONFIG.output_root).expanduser().resolve()
RUN_STARTED = datetime.now(timezone.utc)
RUN_ID = CONFIG.run_id or f"{RUN_STARTED.strftime('%Y%m%dT%H%M%SZ')}_{CONFIG.run_mode}"
RUN_DIR = OUTPUT_ROOT / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=CONFIG.run_id is not None and CONFIG.resume)
SOURCE_SHA256 = "__SOURCE_SHA256__"
if SOURCE_SHA256.startswith("__"):
    SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


config_payload = asdict(CONFIG)
config_sha256 = hashlib.sha256(
    json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
run_identity_sha256 = hashlib.sha256(
    f"{config_sha256}:{SOURCE_SHA256}".encode()
).hexdigest()
resolved_config_path = RUN_DIR / "resolved_config.json"
resolved_config = {
    "config": config_payload,
    "config_sha256": config_sha256,
    "source_sha256": SOURCE_SHA256,
    "run_identity_sha256": run_identity_sha256,
}
if resolved_config_path.is_file():
    previous_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    if previous_config.get("run_identity_sha256") != run_identity_sha256:
        raise RuntimeError(
            "TASK1_RUN_ID already exists with a different resolved config; "
            "use a new run ID instead of mixing experiments"
        )
else:
    write_json(resolved_config_path, resolved_config)
print(json.dumps({**asdict(CONFIG), "run_dir": str(RUN_DIR)}, indent=2))

# %% [markdown]
# ## 1. Reproducibility and dependencies
#
# The notebook refuses CPU full training and records the exact initialization
# checkpoint. The official UniGaze package must already be present at the pinned
# revision `9c240fbe33f3d6146970a77b7c8fa06a7e60019e`.

# %%
for required in (DATASET_ROOT, UNIGAZE_ROOT, MAE_CHECKPOINT):
    if not required.exists():
        raise FileNotFoundError(required)

sys.path.insert(0, str(UNIGAZE_ROOT))
from models.vit.mae_gaze import MAE_Gaze

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if CONFIG.run_mode in {"train", "train_and_zero_shot"} and DEVICE.type != "cuda":
    raise RuntimeError("Full Task 1 training requires a CUDA GPU")

random.seed(CONFIG.seed)
np.random.seed(CONFIG.seed)
torch.manual_seed(CONFIG.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG.seed)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False

environment = {
    "python": sys.version,
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "device": str(DEVICE),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "gpu_memory_bytes": (
        torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
    ),
    "mae_checkpoint": str(MAE_CHECKPOINT),
    "mae_checkpoint_sha256": sha256_file(MAE_CHECKPOINT),
    "source_sha256": SOURCE_SHA256,
}
write_json(RUN_DIR / "environment.json", environment)

# %% [markdown]
# ## 2. GazeGene loader and coordinate contract
#
# Each identity has nine cameras and 2,000 frames per camera. Images are loaded
# as RGB, resized to 224×224 and ImageNet-normalized. No horizontal flip is used
# because it would require a coupled yaw transformation. Labels are converted
# from head-to-target `gaze_C` to UniGaze `[pitch, yaw]` via `-gaze_C`.

# %%
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def vector_to_pitchyaw(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    values = -values / np.linalg.norm(values, axis=1, keepdims=True)
    return np.column_stack(
        (np.arcsin(np.clip(values[:, 1], -1.0, 1.0)), np.arctan2(values[:, 0], values[:, 2]))
    ).astype(np.float32)


class GazeGeneDataset(Dataset[dict[str, Any]]):
    def __init__(self, root: Path, subjects: Sequence[int], train: bool) -> None:
        self.root = root
        self.subjects = tuple(subjects)
        self.train = train
        self.frames = CONFIG.expected_frames_per_camera
        self.cameras = 9
        self._label_cache: dict[tuple[int, int], np.ndarray] = {}
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=True),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1) if train else nn.Identity(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.subjects) * self.cameras * self.frames

    def decode_index(self, index: int) -> tuple[int, int, int]:
        per_subject = self.cameras * self.frames
        subject = self.subjects[index // per_subject]
        remainder = index % per_subject
        return subject, remainder // self.frames, remainder % self.frames

    def labels(self, subject: int, camera: int) -> np.ndarray:
        key = (subject, camera)
        if key not in self._label_cache:
            path = self.root / f"subject{subject}" / "labels" / f"gaze_label_camera{camera}.pkl"
            with path.open("rb") as stream:
                value = pickle.load(stream)  # noqa: S301 - official gated dataset only
            if not isinstance(value, Mapping) or "gaze_C" not in value:
                raise ValueError(f"Invalid official GazeGene label file: {path}")
            gaze = np.asarray(value["gaze_C"], dtype=np.float32)
            if gaze.shape != (self.frames, 3) or not np.isfinite(gaze).all():
                raise ValueError(f"Unexpected gaze_C in {path}: {gaze.shape}")
            self._label_cache[key] = vector_to_pitchyaw(gaze)
        return self._label_cache[key]

    def __getitem__(self, index: int) -> dict[str, Any]:
        subject, camera, frame = self.decode_index(index)
        path = self.root / f"subject{subject}" / "imgs" / f"camera{camera}" / f"subject{subject}_{frame:04d}.jpg"
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {
            "image": tensor,
            "gaze": torch.from_numpy(self.labels(subject, camera)[frame]),
            "subject": subject,
            "camera": camera,
            "frame": frame,
            "index": index,
        }


full_train_dataset = GazeGeneDataset(DATASET_ROOT, CONFIG.train_subjects, train=True)
full_validation_dataset = GazeGeneDataset(DATASET_ROOT, CONFIG.validation_subjects, train=False)
full_test_dataset = GazeGeneDataset(DATASET_ROOT, CONFIG.test_subjects, train=False)
train_dataset: Dataset[Any] = full_train_dataset
validation_dataset: Dataset[Any] = full_validation_dataset
test_dataset: Dataset[Any] = full_test_dataset

if CONFIG.run_mode == "smoke":
    generator = torch.Generator().manual_seed(CONFIG.seed)
    def smoke_subset(dataset: Dataset[Any]) -> Subset[Any]:
        count = min(CONFIG.smoke_samples_per_split, len(dataset))
        return Subset(dataset, torch.randperm(len(dataset), generator=generator)[:count].tolist())
    train_dataset = smoke_subset(train_dataset)
    validation_dataset = smoke_subset(validation_dataset)
    test_dataset = smoke_subset(test_dataset)


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(dataset: Dataset[Any], batch_size: int, shuffle: bool) -> DataLoader[Any]:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": CONFIG.num_workers,
        "pin_memory": DEVICE.type == "cuda",
        "drop_last": shuffle,
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(CONFIG.seed),
        "persistent_workers": CONFIG.num_workers > 0,
    }
    if CONFIG.num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


train_loader = make_loader(train_dataset, CONFIG.batch_size, True)
validation_loader = make_loader(validation_dataset, CONFIG.eval_batch_size, False)
test_loader = make_loader(test_dataset, CONFIG.eval_batch_size, False)
preview = next(iter(validation_loader))
assert preview["image"].shape[1:] == (3, 224, 224)
assert torch.isfinite(preview["gaze"]).all()
print({"train": len(train_dataset), "validation": len(validation_dataset), "test": len(test_dataset)})


def numpy_pitchyaw_to_vector(angles: np.ndarray) -> np.ndarray:
    pitch, yaw = angles[:, 0], angles[:, 1]
    return np.column_stack(
        (np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw))
    )


def bootstrap_subject_mean_ci(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("Subject bootstrap requires a non-empty 1D sequence")
    generator = np.random.default_rng(CONFIG.seed)
    indices = generator.integers(
        0, len(array), size=(CONFIG.bootstrap_replicates, len(array))
    )
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def constant_direction_baseline() -> dict[str, float]:
    direction_sum = np.zeros(3, dtype=np.float64)
    train_samples = 0
    for subject in CONFIG.train_subjects:
        for camera in range(9):
            angles = full_train_dataset.labels(subject, camera)
            direction_sum += numpy_pitchyaw_to_vector(angles).sum(axis=0)
            train_samples += len(angles)
    constant = direction_sum / np.linalg.norm(direction_sum)
    subject_means: list[float] = []
    all_errors: list[np.ndarray] = []
    for subject in CONFIG.test_subjects:
        subject_errors: list[np.ndarray] = []
        for camera in range(9):
            targets = numpy_pitchyaw_to_vector(full_test_dataset.labels(subject, camera))
            cosine = np.clip(targets @ constant, -1.0, 1.0)
            subject_errors.append(np.rad2deg(np.arccos(cosine)))
        values = np.concatenate(subject_errors)
        subject_means.append(float(values.mean()))
        all_errors.append(values)
    errors = np.concatenate(all_errors)
    ci_low, ci_high = bootstrap_subject_mean_ci(subject_means)
    result = {
        "name": "training_mean_3d_direction",
        "train_samples": train_samples,
        "test_samples": len(errors),
        "sample_micro_mean_deg": float(errors.mean()),
        "sample_micro_median_deg": float(np.median(errors)),
        "sample_micro_p90_deg": float(np.percentile(errors, 90)),
        "sample_micro_p95_deg": float(np.percentile(errors, 95)),
        "subject_macro_mean_deg": float(np.mean(subject_means)),
        "subject_macro_std_deg": float(np.std(subject_means, ddof=1)),
        "subject_macro_ci95_low_deg": ci_low,
        "subject_macro_ci95_high_deg": ci_high,
    }
    write_json(RUN_DIR / "constant_direction_baseline.json", result)
    return result


print({"constant_baseline": constant_direction_baseline()})

# %% [markdown]
# ## 3. Model, metric, optimizer and scheduler

# %%
def pitchyaw_to_vector(angles: Tensor) -> Tensor:
    pitch, yaw = angles[:, 0], angles[:, 1]
    return torch.stack((torch.cos(pitch) * torch.sin(yaw), torch.sin(pitch), torch.cos(pitch) * torch.cos(yaw)), dim=1)


def angular_error_deg(prediction: Tensor, target: Tensor) -> Tensor:
    prediction_vector = F.normalize(pitchyaw_to_vector(prediction), dim=1)
    target_vector = F.normalize(pitchyaw_to_vector(target), dim=1)
    cosine = (prediction_vector * target_vector).sum(dim=1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def make_model(initial_checkpoint: Path = MAE_CHECKPOINT) -> nn.Module:
    return MAE_Gaze(
        model_type="vit_b_16", global_pool=False, drop_path_rate=0.1,
        custom_pretrained_path=str(initial_checkpoint),
    ).to(DEVICE)


model = make_model()
optimizer = torch.optim.AdamW(
    [
        {"params": model.vit.parameters(), "lr": CONFIG.backbone_learning_rate},
        {"params": model.gaze_fc.parameters(), "lr": CONFIG.head_learning_rate},
    ],
    weight_decay=CONFIG.weight_decay,
)
updates_per_epoch = math.ceil(len(train_loader) / CONFIG.gradient_accumulation_steps)
total_updates = max(updates_per_epoch * (1 if CONFIG.run_mode == "smoke" else CONFIG.epochs), 1)
warmup_updates = round(total_updates * CONFIG.warmup_fraction)


def lr_multiplier(update: int) -> float:
    if warmup_updates and update < warmup_updates:
        return max((update + 1) / warmup_updates, 1 / warmup_updates)
    progress = (update - warmup_updates) / max(total_updates - warmup_updates, 1)
    return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
try:
    scaler: Any = torch.amp.GradScaler("cuda", enabled=CONFIG.use_amp and DEVICE.type == "cuda")
except (AttributeError, TypeError):
    scaler = torch.cuda.amp.GradScaler(enabled=CONFIG.use_amp and DEVICE.type == "cuda")


def load_trusted_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        value = torch.load(path, map_location=DEVICE)
    if not isinstance(value, Mapping) or "model" not in value:
        raise ValueError(f"Invalid checkpoint: {path}")
    return value


def atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)

# %% [markdown]
# ## 4. Train and select on validation identities

# %%
def train_epoch(epoch: int) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = error_sum = 0.0
    samples = 0
    started = time.perf_counter()
    for step, batch in enumerate(tqdm(train_loader, desc=f"train {epoch}"), start=1):
        images = batch["image"].to(DEVICE, non_blocking=True)
        targets = batch["gaze"].to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=CONFIG.use_amp and DEVICE.type == "cuda"):
            predictions = model(images)["pred_gaze"]
            loss = F.l1_loss(predictions, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={step}")
        scaler.scale(loss / CONFIG.gradient_accumulation_steps).backward()
        if step % CONFIG.gradient_accumulation_steps == 0 or step == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        count = len(targets)
        loss_sum += float(loss.detach()) * count
        error_sum += float(angular_error_deg(predictions.detach(), targets).sum())
        samples += count
    return {"train_l1_rad": loss_sum / samples, "train_mean_deg": error_sum / samples, "epoch_seconds": time.perf_counter() - started}


@torch.inference_mode()
def evaluate(loader: DataLoader[Any], split_name: str) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    rows: list[pd.DataFrame] = []
    for batch in tqdm(loader, desc=f"evaluate {split_name}"):
        images = batch["image"].to(DEVICE, non_blocking=True)
        targets = batch["gaze"].to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=CONFIG.use_amp and DEVICE.type == "cuda"):
            predictions = model(images)["pred_gaze"]
        rows.append(pd.DataFrame({
            "subject": np.asarray(batch["subject"]),
            "camera": np.asarray(batch["camera"]),
            "frame": np.asarray(batch["frame"]),
            "target_pitch_rad": targets[:, 0].cpu().numpy(),
            "target_yaw_rad": targets[:, 1].cpu().numpy(),
            "pred_pitch_rad": predictions[:, 0].float().cpu().numpy(),
            "pred_yaw_rad": predictions[:, 1].float().cpu().numpy(),
            "angular_error_deg": angular_error_deg(predictions.float(), targets).cpu().numpy(),
        }))
    frame = pd.concat(rows, ignore_index=True)
    per_subject = frame.groupby("subject")["angular_error_deg"].agg(["mean", "median", "std", "count"])
    ci_low, ci_high = bootstrap_subject_mean_ci(per_subject["mean"].tolist())
    metrics = {
        "split": split_name,
        "sample_micro_mean_deg": float(frame["angular_error_deg"].mean()),
        "sample_micro_median_deg": float(frame["angular_error_deg"].median()),
        "sample_micro_std_deg": float(frame["angular_error_deg"].std(ddof=1)) if len(frame) > 1 else 0.0,
        "sample_micro_p90_deg": float(frame["angular_error_deg"].quantile(0.90)),
        "sample_micro_p95_deg": float(frame["angular_error_deg"].quantile(0.95)),
        "subject_macro_mean_deg": float(per_subject["mean"].mean()),
        "subject_macro_std_deg": float(per_subject["mean"].std(ddof=1)) if len(per_subject) > 1 else 0.0,
        "subject_macro_ci95_low_deg": ci_low,
        "subject_macro_ci95_high_deg": ci_high,
        "samples": len(frame),
        "subjects": len(per_subject),
    }
    frame.to_csv(RUN_DIR / f"{split_name}_predictions.csv", index=False)
    per_subject.to_csv(RUN_DIR / f"{split_name}_per_subject.csv")
    write_json(RUN_DIR / f"{split_name}_metrics.json", metrics)
    return metrics, frame


history: list[dict[str, Any]] = []
best_validation = float("inf")
epochs_without_improvement = 0
epochs = 1 if CONFIG.run_mode == "smoke" else CONFIG.epochs
start_epoch = 1
resume_path = RUN_DIR / "last_state.pt"
if CONFIG.resume and resume_path.is_file():
    resume_state = load_trusted_checkpoint(resume_path)
    model.load_state_dict(resume_state["model"])
    optimizer.load_state_dict(resume_state["optimizer"])
    scheduler.load_state_dict(resume_state["scheduler"])
    scaler.load_state_dict(resume_state["scaler"])
    history = list(resume_state["history"])
    best_validation = float(resume_state["best_validation"])
    epochs_without_improvement = int(resume_state["epochs_without_improvement"])
    start_epoch = int(resume_state["epoch"]) + 1
    print(f"Resuming {RUN_ID} at epoch {start_epoch}")
if CONFIG.run_mode in {"smoke", "train", "train_and_zero_shot"}:
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_epoch(epoch)
        validation_metrics, _ = evaluate(validation_loader, f"validation_epoch_{epoch:02d}")
        record = {"epoch": epoch, **train_metrics, **validation_metrics}
        history.append(record)
        score = validation_metrics["subject_macro_mean_deg"]
        if score < best_validation:
            best_validation = score
            epochs_without_improvement = 0
            atomic_torch_save({"model": model.state_dict(), "epoch": epoch, "validation_subject_macro_mean_deg": score}, RUN_DIR / "best_full_model.pt")
            atomic_torch_save({"model": model.vit.state_dict()}, RUN_DIR / "best_backbone_mae_format.pth")
        else:
            epochs_without_improvement += 1
        pd.DataFrame(history).to_csv(RUN_DIR / "training_history.csv", index=False)
        atomic_torch_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "history": history,
                "best_validation": best_validation,
                "epochs_without_improvement": epochs_without_improvement,
            },
            resume_path,
        )
        if CONFIG.run_mode != "smoke" and epochs_without_improvement >= CONFIG.early_stopping_patience:
            break
    selected = load_trusted_checkpoint(RUN_DIR / "best_full_model.pt")
    model.load_state_dict(selected["model"])
    test_metrics, _ = evaluate(test_loader, "gazegene_test")
    write_json(RUN_DIR / "selected_checkpoint.json", {"epoch": selected["epoch"], "validation_subject_macro_mean_deg": selected["validation_subject_macro_mean_deg"], "test": test_metrics})

# %% [markdown]
# ## 5. Optional zero-shot MPIIFaceGaze evaluation
#
# Set `TASK1_RUN_MODE=zero_shot` and `TASK1_FULL_CHECKPOINT` to an existing
# GazeGene full checkpoint, or use `train_and_zero_shot` with `TASK1_MPII_ROOT`.
# No MPIIFaceGaze sample is used for fitting or checkpoint selection.

# %%
if CONFIG.run_mode in {"zero_shot", "train_and_zero_shot"}:
    if CONFIG.mpii_root is None:
        raise ValueError("TASK1_MPII_ROOT is required for zero-shot evaluation")
    mpii_root = Path(CONFIG.mpii_root).expanduser().resolve()
    if CONFIG.run_mode == "zero_shot":
        if CONFIG.full_checkpoint is None:
            raise ValueError("TASK1_FULL_CHECKPOINT is required for zero_shot mode")
        zero_shot_checkpoint_path = Path(CONFIG.full_checkpoint).expanduser().resolve()
        checkpoint = load_trusted_checkpoint(zero_shot_checkpoint_path)
        model.load_state_dict(checkpoint["model"])
    else:
        zero_shot_checkpoint_path = RUN_DIR / "best_full_model.pt"
        checkpoint = load_trusted_checkpoint(zero_shot_checkpoint_path)
        model.load_state_dict(checkpoint["model"])
    write_json(
        RUN_DIR / "zero_shot_source_checkpoint.json",
        {
            "path": str(zero_shot_checkpoint_path),
            "bytes": zero_shot_checkpoint_path.stat().st_size,
            "sha256": sha256_file(zero_shot_checkpoint_path),
            "selected_epoch": checkpoint.get("epoch"),
            "selection_validation_subject_macro_mean_deg": checkpoint.get(
                "validation_subject_macro_mean_deg"
            ),
        },
    )

    from datasets.mpiigaze import MPIIGazeDataset
    subject_frames: list[pd.DataFrame] = []
    for subject_index in range(15):
        subject = f"p{subject_index:02d}"
        dataset = MPIIGazeDataset(
            dataset_path=str(mpii_root), color_type="bgr", keys_to_use=[f"{subject}.h5"],
            data_name="mpii", image_size=224, transform_type="basic_imagenet",
        )
        loader = make_loader(dataset, CONFIG.eval_batch_size, False)
        subject_batches: list[pd.DataFrame] = []
        model.eval()
        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"zero-shot {subject}"):
                images = batch["image"].float().to(DEVICE, non_blocking=True)
                targets = batch["gaze"].float().to(DEVICE, non_blocking=True)
                predictions = model(images)["pred_gaze"]
                subject_batches.append(
                    pd.DataFrame(
                        {
                            "subject": subject,
                            "sample_index": np.asarray(batch["index"]),
                            "target_pitch_rad": targets[:, 0].cpu().numpy(),
                            "target_yaw_rad": targets[:, 1].cpu().numpy(),
                            "pred_pitch_rad": predictions[:, 0].cpu().numpy(),
                            "pred_yaw_rad": predictions[:, 1].cpu().numpy(),
                            "angular_error_deg": angular_error_deg(
                                predictions, targets
                            ).cpu().numpy(),
                        }
                    )
                )
        subject_frames.append(pd.concat(subject_batches, ignore_index=True))
    zero_shot = pd.concat(subject_frames, ignore_index=True)
    zero_shot.to_csv(RUN_DIR / "mpiifacegaze_zero_shot_predictions.csv", index=False)
    per_subject = zero_shot.groupby("subject")["angular_error_deg"].agg(["mean", "median", "std", "count"])
    per_subject.to_csv(RUN_DIR / "mpiifacegaze_zero_shot_per_subject.csv")
    ci_low, ci_high = bootstrap_subject_mean_ci(per_subject["mean"].tolist())
    write_json(RUN_DIR / "mpiifacegaze_zero_shot_metrics.json", {
        "sample_micro_mean_deg": float(zero_shot["angular_error_deg"].mean()),
        "sample_micro_median_deg": float(zero_shot["angular_error_deg"].median()),
        "sample_micro_std_deg": float(zero_shot["angular_error_deg"].std(ddof=1)),
        "sample_micro_p90_deg": float(zero_shot["angular_error_deg"].quantile(0.90)),
        "sample_micro_p95_deg": float(zero_shot["angular_error_deg"].quantile(0.95)),
        "subject_macro_mean_deg": float(per_subject["mean"].mean()),
        "subject_macro_std_deg": float(per_subject["mean"].std(ddof=1)),
        "subject_macro_ci95_low_deg": ci_low,
        "subject_macro_ci95_high_deg": ci_high,
        "subjects": len(per_subject), "samples": len(zero_shot),
    })

if torch.cuda.is_available():
    write_json(RUN_DIR / "gpu_memory.json", {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    })
checkpoint_artifacts = {}
for checkpoint_name in ("best_full_model.pt", "best_backbone_mae_format.pth"):
    checkpoint_path = RUN_DIR / checkpoint_name
    if checkpoint_path.is_file():
        checkpoint_artifacts[checkpoint_name] = {
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
        }
write_json(RUN_DIR / "checkpoint_artifacts.json", checkpoint_artifacts)
write_json(RUN_DIR / "completion.json", {"status": "PASS", "finished_at": datetime.now(timezone.utc).isoformat()})
print(f"Task 1 run complete: {RUN_DIR}")
