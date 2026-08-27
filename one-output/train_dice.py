import argparse
import atexit
import json
import logging
import os
import signal
import sys
import tomllib
import traceback
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from unet_model import UNet


dir_paths_file = Path("./training_paths.toml")
default_path_profile = "pin"
dir_checkpoint = Path("./checkpoints/")
dir_h5_spot_segmentation = (
    Path(__file__).resolve().parent.parent / "data_esrf" / "augmented_spots_train.h5"
)
dir_runs = Path("./runs/")
dir_previews = Path("./prediction_previews/")
dir_debug = Path("./debug_logs/")


def _resolve_config_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def _profile_value(profile: dict, *names: str) -> str | None:
    for name in names:
        if name in profile:
            return profile[name]
    return None


def load_training_paths(paths_file: Path, profile_name: str) -> dict[str, Path]:
    paths_file = Path(paths_file)
    paths = {
        "h5_file": dir_h5_spot_segmentation,
        "checkpoint_dir": dir_checkpoint,
        "log_dir": dir_runs,
        "preview_dir": dir_previews,
        "debug_dir": dir_debug,
    }
    if not paths_file.exists():
        return paths

    config = tomllib.loads(paths_file.read_text())
    if profile_name not in config:
        available_profiles = ", ".join(sorted(config)) or "<none>"
        raise ValueError(
            f"Path profile {profile_name!r} not found in {paths_file}. "
            f"Available profiles: {available_profiles}"
        )

    profile = config[profile_name]
    base_dir = paths_file.parent
    input_path = _profile_value(profile, "input_path", "inputpath", "h5_file")
    output_path = _profile_value(profile, "output_path", "output")

    if input_path is not None:
        paths["h5_file"] = _resolve_config_path(input_path, base_dir)

    if output_path is not None:
        output_dir = _resolve_config_path(output_path, base_dir)
        paths["checkpoint_dir"] = output_dir / "checkpoints"
        paths["log_dir"] = output_dir / "runs"
        paths["preview_dir"] = output_dir / "prediction_previews"
        paths["debug_dir"] = output_dir / "debug_logs"

    override_keys = {
        "checkpoint_dir": ("checkpoint_dir", "checkpoint_path"),
        "log_dir": ("log_dir", "runs_dir"),
        "preview_dir": ("preview_dir",),
        "debug_dir": ("debug_dir",),
    }
    for target_key, names in override_keys.items():
        value = _profile_value(profile, *names)
        if value is not None:
            paths[target_key] = _resolve_config_path(value, base_dir)

    return paths


def apply_path_overrides(args, paths: dict[str, Path]) -> dict[str, Path]:
    resolved = dict(paths)
    cli_overrides = {
        "h5_file": args.h5_file,
        "checkpoint_dir": args.checkpoint_dir,
        "log_dir": args.log_dir,
        "preview_dir": args.preview_dir,
        "debug_dir": args.debug_dir,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            resolved[key] = Path(value).expanduser()
    return resolved


class TrainingDebugRecorder:
    """Writes a small heartbeat file with the latest known training position."""

    def __init__(self, state_file: Path):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = {
            "status": "starting",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "pid": os.getpid(),
        }
        self.update(phase="process_started")

    def update(self, **kwargs) -> None:
        self.state.update(kwargs)
        self.state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        tmp_file = self.state_file.with_suffix(self.state_file.suffix + ".tmp")
        tmp_file.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n")
        tmp_file.replace(self.state_file)

    def record_exception(self, exc: BaseException) -> None:
        self.update(
            status="crashed",
            phase="exception",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )


def setup_logging(debug_dir: Path) -> tuple[Path, Path, TrainingDebugRecorder]:
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    debug_log_file = debug_dir / f"train_debug_{timestamp}.log"
    state_file = debug_dir / "train_state.json"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(debug_log_file, mode="w"),
        ],
        force=True,
    )

    recorder = TrainingDebugRecorder(state_file)
    logging.info("Debug log file: %s", debug_log_file)
    logging.info("Training state heartbeat file: %s", state_file)
    atexit.register(lambda: logging.info("Python process exiting"))
    return debug_log_file, state_file, recorder


def install_signal_logging(debug_recorder: TrainingDebugRecorder) -> None:
    def _handler(signum, _frame):
        signal_name = signal.Signals(signum).name
        logging.error("Received %s; writing debug state before exit.", signal_name)
        debug_recorder.update(status="stopped_by_signal", phase="signal", signal=signal_name)
        raise SystemExit(128 + signum)

    for signal_name in ("SIGTERM", "SIGINT"):
        if hasattr(signal, signal_name):
            signal.signal(getattr(signal, signal_name), _handler)


class H5SpotSeparationDataset(Dataset):
    """Spot separation samples stored as HDF5 groups with image/spot_images datasets."""

    def __init__(self, h5_file: Path, img_scale: float = 1.0):
        self.h5_file = Path(h5_file)
        self.img_scale = img_scale

        if not self.h5_file.exists():
            raise FileNotFoundError(f"HDF5 spot separation file not found: {self.h5_file}")

        with h5py.File(self.h5_file, "r") as f:
            self.sample_names = sorted(
                name
                for name, obj in f.items()
                if isinstance(obj, h5py.Group) and "image" in obj and "spot_images" in obj
            )

        if not self.sample_names:
            raise ValueError(
                f"No image/spot_images sample groups found in {self.h5_file}. "
                "Regenerate the augmentation archive with separated spot intensity targets."
            )

        logging.info("Found %d HDF5 spot separation samples", len(self.sample_names))

    def __len__(self):
        return len(self.sample_names)

    @staticmethod
    def normalize_image_and_targets(image: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Normalize both targets together and reconstruct their exact additive input."""
        image = image.astype(np.float32, copy=False)
        targets = targets.astype(np.float32, copy=False)
        finite_image = np.isfinite(image)
        if not finite_image.any():
            return np.zeros_like(image, dtype=np.float32), np.zeros_like(targets, dtype=np.float32)

        clean_image = np.where(finite_image, image, 0.0)
        clean_targets = np.where(np.isfinite(targets), targets, 0.0)
        np.maximum(clean_targets, 0.0, out=clean_targets)
        positive_values = clean_image[clean_image > 0]
        if positive_values.size == 0:
            return np.zeros_like(image, dtype=np.float32), np.zeros_like(targets, dtype=np.float32)
        high = float(np.percentile(positive_values, 99.9))
        if high <= 0:
            high = float(positive_values.max())
        if high <= 0:
            return np.zeros_like(image, dtype=np.float32), np.zeros_like(targets, dtype=np.float32)

        normalized_targets = np.clip(clean_targets, 0.0, high) / high
        normalized_image = normalized_targets.sum(axis=0, dtype=np.float32)
        return normalized_image.astype(np.float32, copy=False), normalized_targets.astype(np.float32, copy=False)

    def __getitem__(self, idx):
        sample_name = self.sample_names[idx]

        with h5py.File(self.h5_file, "r") as f:
            image = f[sample_name]["image"][()]
            targets = f[sample_name]["spot_images"][()]

        if image.ndim == 3:
            image = image.mean(axis=-1)
        if targets.ndim != 3 or targets.shape[0] != 2:
            raise ValueError(f"{sample_name}: expected spot_images shape (2, H, W), got {targets.shape}")

        image, targets = self.normalize_image_and_targets(image, targets)
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        target_tensor = torch.from_numpy(targets)

        if self.img_scale != 1.0:
            size = (
                max(1, int(image_tensor.shape[1] * self.img_scale)),
                max(1, int(image_tensor.shape[2] * self.img_scale)),
            )
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False
            ).squeeze(0)
            target_tensor = F.interpolate(
                target_tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False
            ).squeeze(0)

        return {"image": image_tensor, "target": target_tensor}


def spot_intensity_prediction(logits: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits) * image


def compose_spot_predictions(first_spot: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    second_spot = (image - first_spot).clamp_min(0.0)
    return torch.cat((first_spot, second_spot), dim=1)


def soft_dice_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    numerator = 2.0 * (prediction * target).sum(dim=(2, 3)) + smooth
    denominator = prediction.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth
    return numerator / denominator


def single_spot_loss_per_sample(
    prediction: torch.Tensor,
    spot_target: torch.Tensor,
    foreground_weight: float = 8.0,
    foreground_threshold: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """Loss vectors for one prediction against one candidate ground-truth spot."""
    dice = 1.0 - soft_dice_per_sample(prediction, spot_target).mean(dim=1)

    error = (prediction - spot_target).abs()
    foreground = (spot_target > foreground_threshold).to(dtype=error.dtype)
    weights = 1.0 + foreground_weight * foreground
    intensity = (error * weights).sum(dim=(1, 2, 3)) / weights.sum(dim=(1, 2, 3)).clamp_min(1.0)

    # Unlike input-background loss, this also penalizes assigning the other spot to this output.
    target_background = (spot_target <= foreground_threshold).to(dtype=prediction.dtype)
    background = (prediction.abs() * target_background).sum(dim=(1, 2, 3)) / target_background.sum(
        dim=(1, 2, 3)
    ).clamp_min(1.0)
    total = dice + 0.25 * intensity + 0.01 * background
    return {
        "total": total,
        "dice": dice,
        "intensity": intensity,
        "background": background,
    }


def selected_target_is_second(
    first_spot_prediction: torch.Tensor,
    target: torch.Tensor,
    foreground_weight: float = 8.0,
) -> torch.Tensor:
    first = single_spot_loss_per_sample(
        first_spot_prediction, target[:, 0:1], foreground_weight=foreground_weight
    )
    second = single_spot_loss_per_sample(
        first_spot_prediction, target[:, 1:2], foreground_weight=foreground_weight
    )
    return second["total"] < first["total"]


def align_spot_targets(first_spot_prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Put the selected one-spot target first and the residual target second."""
    use_second = selected_target_is_second(first_spot_prediction, target)
    aligned = target.clone()
    aligned[use_second] = target[use_second].flip(1)
    return aligned


def separation_loss_components(
    first_spot_prediction: torch.Tensor,
    target: torch.Tensor,
    image: torch.Tensor,
    foreground_weight: float = 8.0,
) -> dict[str, torch.Tensor]:
    del image  # The loss supervises one selected spot only; the other output is a residual.
    first = single_spot_loss_per_sample(
        first_spot_prediction, target[:, 0:1], foreground_weight=foreground_weight
    )
    second = single_spot_loss_per_sample(
        first_spot_prediction, target[:, 1:2], foreground_weight=foreground_weight
    )
    use_second = second["total"] < first["total"]

    selected = {
        name: torch.where(use_second, second[name], first[name]).mean()
        for name in first
    }
    return selected


def separation_metric_tensors(
    first_spot_prediction: torch.Tensor,
    target: torch.Tensor,
    image: torch.Tensor,
    threshold: float = 1e-4,
) -> dict[str, torch.Tensor]:
    aligned_target = align_spot_targets(first_spot_prediction, target)
    prediction = compose_spot_predictions(first_spot_prediction, image)
    loss_parts = separation_loss_components(first_spot_prediction, target, image)
    first_spot_target = aligned_target[:, 0:1]
    first_spot_error = (first_spot_prediction - first_spot_target).abs()
    first_spot_squared_error = (first_spot_prediction - first_spot_target).square()
    two_spot_absolute_error = (prediction - aligned_target).abs()
    two_spot_squared_error = (prediction - aligned_target).square()
    foreground = first_spot_target > threshold
    background = ~foreground

    hard_prediction = first_spot_prediction > threshold
    hard_target = first_spot_target > threshold
    true_positive = (hard_prediction & hard_target).sum(dtype=torch.float32)
    false_positive = (hard_prediction & ~hard_target).sum(dtype=torch.float32)
    false_negative = (~hard_prediction & hard_target).sum(dtype=torch.float32)
    true_negative = (~hard_prediction & ~hard_target).sum(dtype=torch.float32)

    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    recall = true_positive / (true_positive + false_negative).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    iou = true_positive / (true_positive + false_positive + false_negative).clamp_min(1.0)
    pixel_accuracy = (true_positive + true_negative) / hard_target.numel()

    per_spot_soft_dice = soft_dice_per_sample(prediction, target)
    reconstruction = prediction.sum(dim=1, keepdim=True)

    return {
        "loss_total": loss_parts["total"],
        "loss_dice": loss_parts["dice"],
        "loss_intensity": loss_parts["intensity"],
        "loss_background": loss_parts["background"],
        "soft_dice": soft_dice_per_sample(first_spot_prediction, first_spot_target).mean(),
        "soft_dice_spot_1": per_spot_soft_dice[:, 0].mean(),
        "soft_dice_spot_2": per_spot_soft_dice[:, 1].mean(),
        "two_spot_soft_dice": per_spot_soft_dice.mean(),
        "mae": first_spot_error.mean(),
        "rmse": first_spot_squared_error.mean().sqrt(),
        "two_spot_mae": two_spot_absolute_error.mean(),
        "two_spot_rmse": two_spot_squared_error.mean().sqrt(),
        "foreground_mae": first_spot_error[foreground].mean() if foreground.any() else first_spot_error.new_tensor(0.0),
        "background_mae": first_spot_error[background].mean() if background.any() else first_spot_error.new_tensor(0.0),
        "spot_1_mae": two_spot_absolute_error[:, 0].mean(),
        "spot_2_mae": two_spot_absolute_error[:, 1].mean(),
        "spot_1_rmse": two_spot_squared_error[:, 0].mean().sqrt(),
        "spot_2_rmse": two_spot_squared_error[:, 1].mean().sqrt(),
        "reconstruction_mae": (reconstruction - image).abs().mean(),
        "reconstruction_rmse": (reconstruction - image).square().mean().sqrt(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "pixel_accuracy": pixel_accuracy,
        "predicted_foreground_fraction": hard_prediction.float().mean(),
        "target_foreground_fraction": hard_target.float().mean(),
        "prediction_mean": first_spot_prediction.mean(),
        "target_mean": first_spot_target.mean(),
        "prediction_sum": first_spot_prediction.sum(),
        "target_sum": first_spot_target.sum(),
    }


def metrics_to_floats(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: value.detach().float().cpu().item() for name, value in metrics.items()}


def average_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}
    return {
        name: float(np.mean([metrics[name] for metrics in metric_dicts]))
        for name in metric_dicts[0]
    }


def log_metrics(writer: SummaryWriter, prefix: str, metrics: dict[str, float], step: int) -> None:
    for name, value in metrics.items():
        writer.add_scalar(f"{prefix}/{name}", value, step)


def separation_loss(
    first_spot_prediction: torch.Tensor,
    target: torch.Tensor,
    image: torch.Tensor,
) -> torch.Tensor:
    return separation_loss_components(first_spot_prediction, target, image)["total"]


def evaluate(model, dataloader, device, amp):
    model.eval()
    metric_dicts = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            targets = batch["target"].to(device=device, dtype=torch.float32)

            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
                logits = model(images)
                if logits.shape[2:] != targets.shape[2:]:
                    logits = F.interpolate(
                        logits, size=targets.shape[2:], mode="bilinear", align_corners=False
                    )
                prediction = spot_intensity_prediction(logits, images)
                metric_dicts.append(metrics_to_floats(separation_metric_tensors(prediction, targets, images)))

    model.train()
    return average_metric_dicts(metric_dicts)


def save_prediction_previews(
    model,
    dataloader,
    device,
    amp,
    output_dir: Path,
    epoch: int,
    max_samples: int = 4,
):
    if max_samples <= 0:
        return

    epoch_dir = output_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            targets = batch["target"].to(device=device, dtype=torch.float32)

            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
                logits = model(images)
                if logits.shape[2:] != targets.shape[2:]:
                    logits = F.interpolate(logits, size=targets.shape[2:], mode="bilinear", align_corners=False)
                first_spot_predictions = spot_intensity_prediction(logits, images)
                predictions = compose_spot_predictions(first_spot_predictions, images)
                targets = align_spot_targets(first_spot_predictions, targets)

            batch_size = images.shape[0]
            for sample_idx in range(batch_size):
                if saved >= max_samples:
                    model.train()
                    return

                image = images[sample_idx, 0].detach().cpu().numpy()
                true_spots = targets[sample_idx].detach().cpu().numpy()
                pred_spots = predictions[sample_idx].detach().cpu().numpy()
                true_sum = true_spots.sum(axis=0)
                pred_sum = pred_spots.sum(axis=0)
                residual_error = np.abs(pred_spots[1] - true_spots[1])

                fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
                panels = [
                    (image, "input", "gray", 0.0, 1.0),
                    (true_spots[0], "true spot 1", "gray", 0.0, 1.0),
                    (pred_spots[0], "pred spot 1", "gray", 0.0, 1.0),
                    (np.abs(pred_spots[0] - true_spots[0]), "error spot 1", "magma", 0.0, 1.0),
                    (true_sum, "true sum", "gray", 0.0, 1.0),
                    (true_spots[1], "true spot 2", "gray", 0.0, 1.0),
                    (pred_spots[1], "residual spot 2", "gray", 0.0, 1.0),
                    (residual_error, "error spot 2", "magma", 0.0, 1.0),
                ]
                for axis, (data, title, cmap, vmin, vmax) in zip(axes.flat, panels):
                    axis.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
                    axis.set_title(title)
                    axis.axis("off")

                file_name = epoch_dir / f"sample_{saved:02d}.png"
                fig.savefig(file_name, dpi=150)
                plt.close(fig)
                saved += 1

    model.train()


def make_run_name(h5_file: Path, epochs: int, batch_size: int, learning_rate: float, img_scale: float) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    data_name = Path(h5_file).stem
    return f"{timestamp}_{data_name}_e{epochs}_b{batch_size}_lr{learning_rate:g}_s{img_scale:g}"


def safe_model_file_name(run_name: str) -> str:
    safe_name = "".join(char if char.isalnum() or char in "._-" else "_" for char in run_name)
    return f"final_{safe_name}.pth"


def train_model(
    model,
    device,
    h5_file: Path = dir_h5_spot_segmentation,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    val_percent: float = 0.1,
    save_checkpoint: bool = True,
    img_scale: float = 0.5,
    amp: bool = False,
    weight_decay: float = 1e-8,
    gradient_clipping: float = 1.0,
    log_dir: Path = dir_runs,
    run_name: str | None = None,
    preview_dir: Path = dir_previews,
    preview_samples: int = 4,
    checkpoint_dir: Path = dir_checkpoint,
    max_samples: int | None = None,
    debug_recorder: TrainingDebugRecorder | None = None,
):
    if debug_recorder is not None:
        debug_recorder.update(status="running", phase="loading_dataset", h5_file=str(h5_file))
    dataset = H5SpotSeparationDataset(h5_file, img_scale)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError(f"--max-samples must be positive, got {max_samples}")
        if max_samples < len(dataset):
            indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))[:max_samples]
            dataset = Subset(dataset, indices.tolist())
            logging.info("Limited dataset to %d samples for this run", len(dataset))
            if debug_recorder is not None:
                debug_recorder.update(phase="dataset_limited", max_samples=max_samples, dataset_size=len(dataset))

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    if debug_recorder is not None:
        debug_recorder.update(
            phase="splitting_dataset",
            dataset_size=len(dataset),
            train_size=n_train,
            validation_size=n_val,
        )
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    loader_args = {
        "batch_size": batch_size,
        "num_workers": os.cpu_count() or 0,
        "pin_memory": device.type != "cpu",
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    run_name = run_name or make_run_name(h5_file, epochs, batch_size, learning_rate, img_scale)
    run_dir = Path(log_dir) / run_name
    writer = SummaryWriter(log_dir=str(run_dir))
    run_preview_dir = Path(preview_dir) / run_name
    if debug_recorder is not None:
        debug_recorder.update(
            phase="run_initialized",
            run_name=run_name,
            run_dir=str(run_dir),
            preview_dir=str(run_preview_dir),
            epochs=epochs,
            batch_size=batch_size,
            max_samples=max_samples,
            learning_rate=learning_rate,
            device=device.type,
            amp=amp,
        )
    logging.info("TensorBoard log directory: %s", run_dir)
    logging.info("Prediction preview directory: %s", run_preview_dir)
    logging.info("Open TensorBoard with: tensorboard --logdir %s", Path(log_dir))
    writer.add_hparams(
        {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "val_percent": val_percent,
            "save_checkpoint": save_checkpoint,
            "img_scale": img_scale,
            "amp": amp,
            "h5_file": str(h5_file),
            "run_name": run_name,
            "preview_samples": preview_samples,
            "max_samples": max_samples or 0,
            "optimizer": "AdamW",
            "loss": "PI_one_spot_dice+0.25_foreground_l1+0.01_target_background",
        },
        {"hparam/metric": 0},
    )

    logging.info(
        """Starting training:
        HDF5 file:       %s
        Epochs:          %d
        Batch size:      %d
        Learning rate:   %s
        Training size:   %d
        Validation size: %d
        Max samples:     %s
        Checkpoints:     %s
        Device:          %s
        Image scaling:   %s
        Mixed Precision: %s
    """,
        h5_file,
        epochs,
        batch_size,
        learning_rate,
        n_train,
        n_val,
        max_samples if max_samples is not None else "all",
        save_checkpoint,
        device.type,
        img_scale,
        amp,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        foreach=True,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=200)
    grad_scaler = torch.amp.GradScaler(device.type, enabled=amp)
    global_step = 0

    for epoch in range(1, epochs + 1):
        if debug_recorder is not None:
            debug_recorder.update(status="running", phase="epoch_started", epoch=epoch, epochs=epochs)
        logging.info("Starting epoch %d/%d", epoch, epochs)
        model.train()
        train_epoch_metrics = []
        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
            for batch_idx, batch in enumerate(train_loader, start=1):
                images = batch["image"]
                targets = batch["target"].to(device=device, dtype=torch.float32)

                assert images.shape[1] == model.n_channels, (
                    f"Network has {model.n_channels} input channels, "
                    f"but loaded images have {images.shape[1]} channels."
                )

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

                with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
                    logits = model(images)
                    if logits.shape[2:] != targets.shape[2:]:
                        logits = F.interpolate(
                            logits, size=targets.shape[2:], mode="bilinear", align_corners=False
                        )
                    predictions = spot_intensity_prediction(logits, images)
                    loss_parts = separation_loss_components(predictions, targets, images)
                    loss = loss_parts["total"]

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                if debug_recorder is not None and (batch_idx == 1 or batch_idx % 10 == 0):
                    debug_recorder.update(
                        phase="training_batch",
                        epoch=epoch,
                        epochs=epochs,
                        batch=batch_idx,
                        batches_per_epoch=len(train_loader),
                        global_step=global_step,
                    )
                with torch.no_grad():
                    batch_metrics = metrics_to_floats(separation_metric_tensors(predictions, targets, images))
                train_epoch_metrics.append(batch_metrics)
                log_metrics(writer, "train_batch", batch_metrics, global_step)
                writer.add_scalar("Loss/train_batch", batch_metrics["loss_total"], global_step)
                writer.add_scalar("Loss_parts/train_dice", batch_metrics["loss_dice"], global_step)
                writer.add_scalar("Loss_parts/train_intensity", batch_metrics["loss_intensity"], global_step)
                writer.add_scalar("Loss_parts/train_background", batch_metrics["loss_background"], global_step)
                pbar.set_postfix(**{"loss (batch)": batch_metrics["loss_total"]})

                division_step = n_train // (5 * batch_size)
                if division_step > 0 and global_step % division_step == 0:
                    for tag, value in model.named_parameters():
                        tag = tag.replace("/", ".")
                        if not (torch.isinf(value) | torch.isnan(value)).any():
                            writer.add_histogram(f"Weights/{tag}", value.data, global_step)
                        if value.grad is not None and not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                            writer.add_histogram(f"Gradients/{tag}", value.grad.data, global_step)

                    if debug_recorder is not None:
                        debug_recorder.update(
                            phase="validation_step",
                            epoch=epoch,
                            batch=batch_idx,
                            global_step=global_step,
                        )
                    val_metrics = evaluate(model, val_loader, device, amp)
                    val_score = val_metrics.get("loss_total", 0.0)
                    scheduler.step(val_score)

                    logging.info("Validation separation loss: %s", val_score)
                    writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], global_step)
                    writer.add_scalar("Loss/validation_step", val_score, global_step)
                    log_metrics(writer, "validation_step", val_metrics, global_step)
                    writer.add_image("input_images", images[0], global_step)
                    preview_targets = align_spot_targets(predictions[:1], targets[:1])
                    writer.add_image("spots/true_1", preview_targets[0, 0:1].float(), global_step)
                    writer.add_image("spots/true_2", preview_targets[0, 1:2].float(), global_step)
                    preview_predictions = compose_spot_predictions(predictions[:1], images[:1])
                    writer.add_image("spots/pred_1", preview_predictions[0, 0:1].float(), global_step)
                    writer.add_image("spots/residual_pred_2", preview_predictions[0, 1:2].float(), global_step)

        if debug_recorder is not None:
            debug_recorder.update(phase="epoch_validation", epoch=epoch, epochs=epochs, global_step=global_step)
        train_metrics = average_metric_dicts(train_epoch_metrics)
        val_metrics = evaluate(model, val_loader, device, amp)
        mean_epoch_loss = train_metrics.get("loss_total", 0.0)
        val_score = val_metrics.get("loss_total", 0.0)
        scheduler.step(val_score)
        if debug_recorder is not None:
            debug_recorder.update(
                phase="saving_prediction_previews",
                epoch=epoch,
                epochs=epochs,
                global_step=global_step,
                train_loss=mean_epoch_loss,
                validation_loss=val_score,
            )
        save_prediction_previews(
            model,
            val_loader,
            device,
            amp,
            run_preview_dir,
            epoch,
            max_samples=preview_samples,
        )
        writer.add_scalar("Loss/train_epoch", mean_epoch_loss, epoch)
        writer.add_scalar("Loss/validation_epoch", val_score, epoch)
        log_metrics(writer, "train_epoch", train_metrics, epoch)
        log_metrics(writer, "validation_epoch", val_metrics, epoch)
        logging.info("Epoch %d mean training loss: %s", epoch, mean_epoch_loss)
        logging.info("Epoch %d validation separation loss: %s", epoch, val_score)

        if debug_recorder is not None:
            debug_recorder.update(
                phase="epoch_finished",
                epoch=epoch,
                epochs=epochs,
                global_step=global_step,
                train_loss=mean_epoch_loss,
                validation_loss=val_score,
            )

        if save_checkpoint:
            if debug_recorder is not None:
                debug_recorder.update(phase="saving_checkpoint", epoch=epoch, epochs=epochs, global_step=global_step)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(checkpoint_dir / f"checkpoint_epoch{epoch}.pth"))
            logging.info("Checkpoint %d saved!", epoch)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_model_file = checkpoint_dir / safe_model_file_name(run_name)
    if debug_recorder is not None:
        debug_recorder.update(
            phase="saving_final_model",
            epochs=epochs,
            global_step=global_step,
            final_model_file=str(final_model_file),
        )
    torch.save(model.state_dict(), str(final_model_file))
    logging.info("Final model saved to %s", final_model_file)

    if debug_recorder is not None:
        debug_recorder.update(
            status="completed",
            phase="training_finished",
            epochs=epochs,
            global_step=global_step,
            final_model_file=str(final_model_file),
        )
    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description="Train the UNet on HDF5 spot separation data")
    parser.add_argument("--epochs", "-e", metavar="E", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", "-b", dest="batch_size", metavar="B", type=int, default=1, help="Batch size")
    parser.add_argument("--learning-rate", "-l", metavar="LR", type=float, default=1e-4, help="Learning rate", dest="lr")
    parser.add_argument("--load", "-f", type=str, default=False, help="Load model from a .pth file")
    parser.add_argument("--scale", "-s", type=float, default=0.5, help="Downscaling factor of the images")
    parser.add_argument(
        "--validation",
        "-v",
        dest="val",
        type=float,
        default=10.0,
        help="Percent of the data that is used as validation (0-100)",
    )
    parser.add_argument("--amp", action="store_true", default=False, help="Use mixed precision")
    parser.add_argument("--bilinear", action="store_true", default=False, help="Use bilinear upsampling")
    parser.add_argument("--paths-file", type=str, default=str(dir_paths_file), help="TOML file with training input/output paths")
    parser.add_argument("--path-profile", type=str, default=default_path_profile, help="Path profile to read from --paths-file")
    parser.add_argument("--h5-file", type=str, default=None, help="Override HDF5 input file from the selected path profile")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint directory from the selected path profile")
    parser.add_argument("--log-dir", type=str, default=None, help="Override TensorBoard root log directory from the selected path profile")
    parser.add_argument("--run-name", type=str, default=None, help="Optional TensorBoard run name")
    parser.add_argument("--preview-dir", type=str, default=None, help="Override prediction preview directory from the selected path profile")
    parser.add_argument("--preview-samples", type=int, default=4, help="Number of validation previews to save per epoch")
    parser.add_argument("--max-samples", type=int, default=None, help="Use only this many random samples from the HDF5 file for a debug run")
    parser.add_argument("--debug-dir", type=str, default=None, help="Override debug log directory from the selected path profile")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    training_paths = apply_path_overrides(
        args, load_training_paths(Path(args.paths_file), args.path_profile)
    )
    debug_log_file, state_file, debug_recorder = setup_logging(training_paths["debug_dir"])
    install_signal_logging(debug_recorder)
    debug_recorder.update(
        phase="arguments_parsed",
        command=" ".join(sys.argv),
        path_profile=args.path_profile,
        paths_file=str(args.paths_file),
        h5_file=str(training_paths["h5_file"]),
        checkpoint_dir=str(training_paths["checkpoint_dir"]),
        log_dir=str(training_paths["log_dir"]),
        preview_dir=str(training_paths["preview_dir"]),
        debug_log_file=str(debug_log_file),
        state_file=str(state_file),
        max_samples=args.max_samples,
    )

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        debug_recorder.update(phase="device_selected", device=str(device), cuda_available=torch.cuda.is_available())
        logging.info("Using device %s", device)

        model = UNet(n_channels=1, n_classes=1, bilinear=args.bilinear)
        model = model.to(memory_format=torch.channels_last)

        logging.info(
            "Network: %d input channel, %d output channel, %s upscaling",
            model.n_channels,
            model.n_classes,
            "Bilinear" if model.bilinear else "Transposed conv",
        )

        if args.load:
            debug_recorder.update(phase="loading_checkpoint", checkpoint=args.load)
            state_dict = torch.load(args.load, map_location=device)
            state_dict.pop("mask_values", None)
            model_state = model.state_dict()
            compatible_state = {
                name: value
                for name, value in state_dict.items()
                if name in model_state and value.shape == model_state[name].shape
            }
            skipped_keys = sorted(set(state_dict) - set(compatible_state))
            model.load_state_dict(compatible_state, strict=False)
            if skipped_keys:
                logging.info("Skipped incompatible checkpoint keys: %s", ", ".join(skipped_keys))
            logging.info("Model loaded from %s", args.load)

        model.to(device=device)

        try:
            train_model(
                model=model,
                device=device,
                h5_file=training_paths["h5_file"],
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                img_scale=args.scale,
                val_percent=args.val / 100,
                amp=args.amp,
                log_dir=training_paths["log_dir"],
                run_name=args.run_name,
                preview_dir=training_paths["preview_dir"],
                preview_samples=args.preview_samples,
                checkpoint_dir=training_paths["checkpoint_dir"],
                max_samples=args.max_samples,
                debug_recorder=debug_recorder,
            )
        except torch.cuda.OutOfMemoryError as exc:
            logging.exception(
                "Detected OutOfMemoryError. Enabling checkpointing to reduce memory usage. "
                "Consider enabling AMP (--amp) for faster and more memory-efficient training."
            )
            debug_recorder.update(status="oom_retrying", phase="cuda_out_of_memory", error=str(exc))
            torch.cuda.empty_cache()
            model.use_checkpointing()
            train_model(
                model=model,
                device=device,
                h5_file=training_paths["h5_file"],
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                img_scale=args.scale,
                val_percent=args.val / 100,
                amp=args.amp,
                log_dir=training_paths["log_dir"],
                run_name=args.run_name,
                preview_dir=training_paths["preview_dir"],
                preview_samples=args.preview_samples,
                checkpoint_dir=training_paths["checkpoint_dir"],
                max_samples=args.max_samples,
                debug_recorder=debug_recorder,
            )
    except Exception as exc:
        logging.exception("Training failed with an unhandled exception.")
        debug_recorder.record_exception(exc)
        raise
