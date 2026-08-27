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
default_path_profile = "twooutputs"
dir_checkpoint = Path("./checkpoints/")
dir_h5_spot_segmentation = Path("./data/augmented_spot_patches.h5")
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
        "train_path": Path(__file__).resolve(),
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
    train_path = _profile_value(profile, "train_path", "train_file")

    if input_path is not None:
        paths["h5_file"] = _resolve_config_path(input_path, base_dir)

    if output_path is not None:
        output_dir = _resolve_config_path(output_path, base_dir)
        paths["checkpoint_dir"] = output_dir / "checkpoints"
        paths["log_dir"] = output_dir / "runs"
        paths["preview_dir"] = output_dir / "prediction_previews"
        paths["debug_dir"] = output_dir / "debug_logs"

    if train_path is not None:
        paths["train_path"] = _resolve_config_path(train_path, base_dir)

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
        "train_path": args.train_path,
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
        image = image.astype(np.float32, copy=False)
        targets = targets.astype(np.float32, copy=False)
        finite = np.isfinite(image)
        if not finite.any():
            return np.zeros_like(image, dtype=np.float32), np.zeros_like(targets, dtype=np.float32)

        values = image[finite]
        lo, hi = np.percentile(values, [1, 99.9])
        if hi <= lo:
            lo = float(values.min())
            hi = float(values.max())
        if hi <= lo:
            return np.zeros_like(image, dtype=np.float32), np.zeros_like(targets, dtype=np.float32)

        image = np.clip(image, lo, hi)
        image = (image - lo) / (hi - lo)
        image[~finite] = 0.0

        targets = np.clip(targets, lo, hi)
        targets = (targets - lo) / (hi - lo)
        targets[~np.isfinite(targets)] = 0.0
        return image.astype(np.float32, copy=False), targets.astype(np.float32, copy=False)

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


def spot_intensity_prediction(logits: torch.Tensor) -> torch.Tensor:
    return F.softplus(logits)


def l1_loss_per_channel_pixel(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Absolute intensity difference for every sample, output channel, and pixel."""
    return (prediction - target).abs()


def l1_loss_per_pixel(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute intensity difference at each pixel, averaged over output channels."""
    return l1_loss_per_channel_pixel(prediction, target).mean(dim=1)


def l1_loss_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute pixel intensity difference for each sample."""
    return l1_loss_per_channel_pixel(prediction, target).mean(dim=(1, 2, 3))


def weighted_l1_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    foreground_weight: float = 8.0,
    foreground_threshold: float = 1e-4,
) -> torch.Tensor:
    """L1 with extra weight on nonzero target pixels, normalized per sample."""
    error = (prediction - target).abs()
    foreground = (target > foreground_threshold).to(dtype=error.dtype)
    weights = 1.0 + foreground_weight * foreground
    numerator = (error * weights).sum(dim=(1, 2, 3))
    denominator = weights.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return numerator / denominator


def permutation_invariant_weighted_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    foreground_weight: float = 8.0,
) -> torch.Tensor:
    direct = weighted_l1_per_sample(prediction, target, foreground_weight=foreground_weight)
    swapped = weighted_l1_per_sample(prediction, target.flip(1), foreground_weight=foreground_weight)
    return torch.minimum(direct, swapped).mean()


def reconstruction_l1_loss(prediction: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(prediction.sum(dim=1, keepdim=True), image)


def background_l1_loss(
    prediction: torch.Tensor,
    image: torch.Tensor,
    background_threshold: float = 1e-4,
) -> torch.Tensor:
    background = image <= background_threshold
    if not background.any():
        return prediction.new_tensor(0.0)
    return prediction.masked_select(background.expand_as(prediction)).abs().mean()


def overlap_exclusivity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    foreground_threshold: float = 1e-4,
) -> torch.Tensor:
    target_overlap = (target[:, 0:1] > foreground_threshold) & (target[:, 1:2] > foreground_threshold)
    disallowed_overlap = ~target_overlap
    if not disallowed_overlap.any():
        return prediction.new_tensor(0.0)
    overlap_intensity = prediction[:, 0:1] * prediction[:, 1:2]
    return overlap_intensity.masked_select(disallowed_overlap).mean()


def best_l1_assignment(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align predictions to targets by the lower two-channel L1 loss.

    Returns aligned predictions, per-channel pixel losses, and combined per-pixel losses.
    """
    direct = l1_loss_per_sample(prediction, target)
    swapped = l1_loss_per_sample(prediction, target.flip(1))
    use_swapped = swapped < direct

    aligned = prediction.clone()
    aligned[use_swapped] = prediction[use_swapped].flip(1)
    channel_loss = l1_loss_per_channel_pixel(aligned, target)
    pixel_loss = channel_loss.mean(dim=1)
    return aligned, channel_loss, pixel_loss


def separation_loss_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    image: torch.Tensor,
    foreground_weight: float = 8.0,
) -> dict[str, torch.Tensor]:
    spot = permutation_invariant_weighted_l1_loss(
        prediction,
        target,
        foreground_weight=foreground_weight,
    )
    reconstruction = reconstruction_l1_loss(prediction, image)
    background = background_l1_loss(prediction, image)
    overlap = overlap_exclusivity_loss(prediction, target)
    total = spot + 0.3 * reconstruction + 0.1 * background + 0.3 * overlap
    return {
        "total": total,
        "spot": spot,
        "reconstruction": reconstruction,
        "background": background,
        "overlap": overlap,
    }


def separation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    image: torch.Tensor,
) -> torch.Tensor:
    return separation_loss_components(prediction, target, image)["total"]


def align_prediction_channels(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return best_l1_assignment(prediction, target)[0]


def save_loss_detail_preview(
    file_name: Path,
    image: np.ndarray,
    channel_loss: np.ndarray,
    pixel_loss: np.ndarray,
    sum_error: np.ndarray,
    loss_vmax: float,
) -> None:
    """Save a focused loss-map PNG with colorbars for per-pixel L1 errors."""
    threshold = np.percentile(pixel_loss, 95)
    top_loss_mask = pixel_loss > threshold
    mean_loss = float(pixel_loss.mean())
    max_loss = float(pixel_loss.max())

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    panels = [
        (image, "input image", "gray", 0.0, 1.0, None),
        (channel_loss[0], f"spot 1 loss\nmean={channel_loss[0].mean():.4f}", "magma", 0.0, loss_vmax, "absolute L1 error"),
        (channel_loss[1], f"spot 2 loss\nmean={channel_loss[1].mean():.4f}", "magma", 0.0, loss_vmax, "absolute L1 error"),
        (sum_error, f"sum error\nmean={sum_error.mean():.4f}", "magma", 0.0, loss_vmax, "absolute intensity error"),
        (pixel_loss, f"pixel loss avg over spots\nmean={mean_loss:.4f}, max={max_loss:.4f}", "magma", 0.0, loss_vmax, "absolute L1 error"),
        (top_loss_mask, f"top 5% pixel loss\nthreshold>{threshold:.4f}", "gray", 0.0, 1.0, None),
    ]

    for axis, (data, title, cmap, vmin, vmax, colorbar_label) in zip(axes.flat, panels):
        image_artist = axis.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
        if colorbar_label is not None:
            colorbar = fig.colorbar(image_artist, ax=axis, fraction=0.046, pad=0.04)
            colorbar.set_label(colorbar_label)

    fig.suptitle("Per-pixel L1 loss: dark = small error, bright = large error", fontsize=12)
    fig.savefig(file_name, dpi=150)
    plt.close(fig)


def evaluate(model, dataloader, device, amp):
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            targets = batch["target"].to(device=device, dtype=torch.float32)

            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
                logits = model(images)
                if logits.shape != targets.shape:
                    logits = F.interpolate(
                        logits, size=targets.shape[2:], mode="bilinear", align_corners=False
                    )
                prediction = spot_intensity_prediction(logits)
                losses.append(separation_loss(prediction, targets, images))

    model.train()
    if not losses:
        return 0.0
    return torch.stack(losses).mean().item()


def save_prediction_previews(
    model,
    dataloader,
    device,
    amp,
    output_dir: Path,
    epoch: int,
    max_samples: int = 5,
):
    if max_samples <= 0:
        return

    dataset_size = len(dataloader.dataset)
    if dataset_size <= 0:
        return

    epoch_dir = output_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng()
    selected_indices = set(
        rng.choice(dataset_size, size=min(max_samples, dataset_size), replace=False).tolist()
    )
    model.eval()
    saved = 0
    seen = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            targets = batch["target"].to(device=device, dtype=torch.float32)

            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
                logits = model(images)
                if logits.shape != targets.shape:
                    logits = F.interpolate(logits, size=targets.shape[2:], mode="bilinear", align_corners=False)
                predictions, channel_losses, pixel_losses = best_l1_assignment(
                    spot_intensity_prediction(logits), targets
                )

            batch_size = images.shape[0]
            for sample_idx in range(batch_size):
                current_index = seen
                seen += 1
                if current_index not in selected_indices:
                    continue

                image = images[sample_idx, 0].detach().cpu().numpy()
                true_spots = targets[sample_idx].detach().cpu().numpy()
                pred_spots = predictions[sample_idx].detach().cpu().numpy()
                channel_loss = channel_losses[sample_idx].detach().cpu().numpy()
                pixel_loss = pixel_losses[sample_idx].detach().cpu().numpy()
                true_sum = true_spots.sum(axis=0)
                pred_sum = pred_spots.sum(axis=0)
                sum_error = np.abs(pred_sum - true_sum)
                loss_vmax = max(float(channel_loss.max()), float(pixel_loss.max()), float(sum_error.max()), 1e-6)

                fig, axes = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
                panels = [
                    (image, "input", "gray", 0.0, 1.0),
                    (true_spots[0], "true spot 1", "gray", 0.0, 1.0),
                    (pred_spots[0], "pred spot 1", "gray", 0.0, 1.0),
                    (channel_loss[0], "loss spot 1", "magma", 0.0, loss_vmax),
                    (true_sum, "true sum", "gray", 0.0, 1.0),
                    (true_spots[1], "true spot 2", "gray", 0.0, 1.0),
                    (pred_spots[1], "pred spot 2", "gray", 0.0, 1.0),
                    (channel_loss[1], "loss spot 2", "magma", 0.0, loss_vmax),
                    (pred_sum, "pred sum", "gray", 0.0, 1.0),
                    (sum_error, "sum error", "magma", 0.0, loss_vmax),
                    (pixel_loss, "pixel loss", "magma", 0.0, loss_vmax),
                    (pixel_loss > np.percentile(pixel_loss, 95), "top 5% loss", "gray", 0.0, 1.0),
                ]
                for axis, (data, title, cmap, vmin, vmax) in zip(axes.flat, panels):
                    axis.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
                    axis.set_title(title)
                    axis.axis("off")

                file_name = epoch_dir / f"sample_{saved:02d}_val_{current_index:05d}.png"
                fig.savefig(file_name, dpi=150)
                plt.close(fig)

                loss_file_name = epoch_dir / f"sample_{saved:02d}_val_{current_index:05d}_loss.png"
                save_loss_detail_preview(
                    loss_file_name,
                    image=image,
                    channel_loss=channel_loss,
                    pixel_loss=pixel_loss,
                    sum_error=sum_error,
                    loss_vmax=loss_vmax,
                )
                np.savez_compressed(
                    file_name.with_suffix(".npz"),
                    image=image,
                    true_spots=true_spots,
                    pred_spots=pred_spots,
                    channel_loss=channel_loss,
                    pixel_loss=pixel_loss,
                    sum_error=sum_error,
                )
                saved += 1
                if saved >= len(selected_indices):
                    model.train()
                    return

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
    preview_samples: int = 5,
    checkpoint_dir: Path = dir_checkpoint,
    max_samples: int | None = None,
    early_stopping_patience: int = 10,
    debug_recorder: TrainingDebugRecorder | None = None,
):
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")
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
            "early_stopping_patience": early_stopping_patience,
            "base_features": model.base_features,
            "optimizer": "AdamW",
            "loss": "PI_full_image_l1",
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
        Base features:   %d
        Early stopping:  %s
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
        model.base_features,
        f"patience {early_stopping_patience}" if early_stopping_patience else "disabled",
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
    best_val_score = float("inf")
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        if debug_recorder is not None:
            debug_recorder.update(status="running", phase="epoch_started", epoch=epoch, epochs=epochs)
        logging.info("Starting epoch %d/%d", epoch, epochs)
        model.train()
        epoch_loss = 0.0
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
                    if logits.shape != targets.shape:
                        logits = F.interpolate(
                            logits, size=targets.shape[2:], mode="bilinear", align_corners=False
                        )
                    predictions = spot_intensity_prediction(logits)
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
                total_loss = loss_parts["total"].detach().cpu().item()
                epoch_loss += total_loss
                writer.add_scalar("Loss/train_batch", total_loss, global_step)
                for loss_name, loss_value in loss_parts.items():
                    writer.add_scalar(
                        f"Loss_parts/train_{loss_name}",
                        loss_value.detach().cpu().item(),
                        global_step,
                    )
                pbar.set_postfix(**{"loss (batch)": total_loss})

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
                    val_score = evaluate(model, val_loader, device, amp)
                    scheduler.step(val_score)

                    logging.info("Validation separation loss: %s", val_score)
                    writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], global_step)
                    writer.add_scalar("Loss/validation_step", val_score, global_step)
                    writer.add_image("input_images", images[0], global_step)
                    writer.add_image("spots/true_1", targets[0, 0:1].float(), global_step)
                    writer.add_image("spots/true_2", targets[0, 1:2].float(), global_step)
                    aligned = align_prediction_channels(predictions[:1], targets[:1])
                    writer.add_image("spots/pred_1", aligned[0, 0:1].float(), global_step)
                    writer.add_image("spots/pred_2", aligned[0, 1:2].float(), global_step)

        if debug_recorder is not None:
            debug_recorder.update(phase="epoch_validation", epoch=epoch, epochs=epochs, global_step=global_step)
        mean_epoch_loss = epoch_loss / max(1, len(train_loader))
        val_score = evaluate(model, val_loader, device, amp)
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
        logging.info("Epoch %d mean training loss: %s", epoch, mean_epoch_loss)
        logging.info("Epoch %d validation separation loss: %s", epoch, val_score)

        if n_val > 0 and val_score < best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            epochs_without_improvement = 0
            best_model_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            logging.info("New best validation loss at epoch %d: %s", epoch, val_score)
        elif n_val > 0:
            epochs_without_improvement += 1
            logging.info("Validation loss did not improve for %d epoch(s)", epochs_without_improvement)

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

        if early_stopping_patience and n_val > 0 and epochs_without_improvement >= early_stopping_patience:
            logging.info(
                "Early stopping at epoch %d; best validation loss was %s at epoch %d",
                epoch, best_val_score, best_epoch,
            )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logging.info("Restored best model weights from epoch %d", best_epoch)

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
    parser.add_argument("--base-features", type=int, default=32, help="Number of features in the first U-Net level")
    parser.add_argument(
        "--early-stopping-patience", type=int, default=10,
        help="Stop after this many epochs without validation improvement (0 disables)",
    )
    parser.add_argument("--paths-file", type=str, default=str(dir_paths_file), help="TOML file with training input/output paths")
    parser.add_argument("--path-profile", type=str, default=default_path_profile, help="Path profile to read from --paths-file")
    parser.add_argument("--h5-file", type=str, default=None, help="Override HDF5 input file from the selected path profile")
    parser.add_argument("--train-path", type=str, default=None, help="Override train.py path recorded from the selected path profile")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint directory from the selected path profile")
    parser.add_argument("--log-dir", type=str, default=None, help="Override TensorBoard root log directory from the selected path profile")
    parser.add_argument("--run-name", type=str, default=None, help="Optional TensorBoard run name")
    parser.add_argument("--preview-dir", type=str, default=None, help="Override prediction preview directory from the selected path profile")
    parser.add_argument("--preview-samples", type=int, default=5, help="Number of random validation previews to save per epoch")
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
        train_path=str(training_paths["train_path"]),
        debug_log_file=str(debug_log_file),
        state_file=str(state_file),
        max_samples=args.max_samples,
    )

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        debug_recorder.update(phase="device_selected", device=str(device), cuda_available=torch.cuda.is_available())
        logging.info("Using device %s", device)

        model = UNet(n_channels=1, n_classes=2, bilinear=args.bilinear, base_features=args.base_features)
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
            model.load_state_dict(state_dict)
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
                early_stopping_patience=args.early_stopping_patience,
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
                early_stopping_patience=args.early_stopping_patience,
                debug_recorder=debug_recorder,
            )
    except Exception as exc:
        logging.exception("Training failed with an unhandled exception.")
        debug_recorder.record_exception(exc)
        raise
