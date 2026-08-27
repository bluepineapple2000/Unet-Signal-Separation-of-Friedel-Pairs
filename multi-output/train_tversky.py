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
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from unet_model import UNet


dir_checkpoint = Path("./checkpoints/")
dir_h5_spot_segmentation = Path("./data/augmented_spot_patches_with_masks.h5")
dir_runs = Path("./runs/")
dir_previews = Path("./prediction_previews/")
dir_loss_diagnostics = Path("./loss_diagnostics/")
dir_debug = Path("./debug_logs/")
dir_project_paths = Path("./project_paths.toml")


def load_project_paths(paths_file: Path) -> dict:
    if not paths_file.exists():
        return {}
    with paths_file.open("rb") as file:
        return tomllib.load(file)


def configured_input_h5(paths_file: Path) -> Path:
    config = load_project_paths(paths_file)
    input_h5 = config.get("data", {}).get("input_h5")
    if input_h5 is None:
        return dir_h5_spot_segmentation
    input_path = Path(input_h5).expanduser()
    if input_path.is_absolute():
        return input_path
    return paths_file.parent / input_pathsasdfil


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
    """Spot separation samples stored as HDF5 groups with image/spot_images/spot_masks datasets."""

    def __init__(self, h5_file: Path, img_scale: float = 1.0):
        self.h5_file = Path(h5_file)
        self.img_scale = img_scale

        if not self.h5_file.exists():
            raise FileNotFoundError(f"HDF5 spot separation file not found: {self.h5_file}")

        with h5py.File(self.h5_file, "r") as f:
            self.sample_names = sorted(
                name
                for name, obj in f.items()
                if isinstance(obj, h5py.Group)
                and "image" in obj
                and "spot_images" in obj
                and "spot_masks" in obj
            )

        if not self.sample_names:
            raise ValueError(
                f"No image/spot_images/spot_masks sample groups found in {self.h5_file}. "
                "Run the augmentation conversion notebook to add separated spot mask targets."
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
            masks = f[sample_name]["spot_masks"][()]

        if image.ndim == 3:
            image = image.mean(axis=-1)
        if targets.ndim != 3 or targets.shape[0] != 2:
            raise ValueError(f"{sample_name}: expected spot_images shape (2, H, W), got {targets.shape}")
        if masks.ndim != 3 or masks.shape[0] != 2:
            raise ValueError(f"{sample_name}: expected spot_masks shape (2, H, W), got {masks.shape}")
        if masks.shape != targets.shape:
            raise ValueError(f"{sample_name}: spot_masks shape {masks.shape} does not match spot_images {targets.shape}")

        image, targets = self.normalize_image_and_targets(image, targets)
        masks = (masks > 0).astype(np.float32, copy=False)
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        target_tensor = torch.from_numpy(targets)
        mask_tensor = torch.from_numpy(masks)

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
            mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=size, mode="nearest").squeeze(0)

        return {"image": image_tensor, "target": target_tensor, "mask": mask_tensor}


def spot_separation_prediction(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    if logits.shape[1] != 4:
        raise ValueError(f"Expected 4 output channels, got {logits.shape[1]}")

    mask_logits = logits[:, 0:2]
    intensity_logits = logits[:, 2:4]
    masks = torch.sigmoid(mask_logits)
    intensities = torch.sigmoid(intensity_logits)
    return {
        "mask_logits": mask_logits,
        "intensity_logits": intensity_logits,
        "masks": masks,
        "intensities": intensities,
    }


def soft_dice_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    numerator = 2.0 * (prediction * target).sum(dim=(2, 3)) + smooth
    denominator = prediction.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth
    return numerator / denominator


def weighted_tversky_loss_per_spot(
    prediction: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    background_weight: float = 0.02,
    wrong_spot_weight: float = 2.0,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Weighted Tversky mask loss for already matched two-channel masks."""
    pred_mask = prediction.clamp(0.0, 1.0)
    target_mask = (target > 0.5).to(dtype=target.dtype)
    other_mask = target_mask.flip(1)
    other_only = (other_mask > 0.5) & (target_mask <= 0.5)
    background = (other_mask <= 0.5) & (target_mask <= 0.5)

    false_positive_weights = torch.ones_like(pred_mask)
    false_positive_weights = torch.where(
        background,
        torch.full_like(false_positive_weights, background_weight),
        false_positive_weights,
    )
    false_positive_weights = torch.where(
        other_only,
        torch.full_like(false_positive_weights, wrong_spot_weight),
        false_positive_weights,
    )

    true_positive = (pred_mask * target_mask).sum(dim=(2, 3))
    false_positive = (pred_mask * (1.0 - target_mask) * false_positive_weights).sum(dim=(2, 3))
    false_negative = ((1.0 - pred_mask) * target_mask).sum(dim=(2, 3))
    score = (true_positive + smooth) / (
        true_positive + alpha * false_positive + beta * false_negative + smooth
    )
    return 1.0 - score


def matched_separation_loss_per_sample(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    mask_target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    mask_losses = weighted_tversky_loss_per_spot(outputs["masks"], mask_target)
    intensity_losses = (outputs["intensities"] - target).abs().mean(dim=(2, 3))
    mask_loss = mask_losses.mean(dim=1)
    intensity_loss = intensity_losses.mean(dim=1)
    total = mask_loss + intensity_loss
    return {
        "total": total,
        "mask_tversky": mask_loss,
        "intensity_l1": intensity_loss,
        "mask_1": mask_losses[:, 0],
        "mask_2": mask_losses[:, 1],
        "intensity_1": intensity_losses[:, 0],
        "intensity_2": intensity_losses[:, 1],
    }


def separation_loss_components(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    mask_target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    direct = matched_separation_loss_per_sample(outputs, target, mask_target)
    swapped = matched_separation_loss_per_sample(outputs, target.flip(1), mask_target.flip(1))
    use_swapped = swapped["total"] < direct["total"]
    return {
        name: torch.where(use_swapped, swapped[name], direct[name]).mean()
        for name in direct
    }


def separation_metric_tensors(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    mask_target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    mask_prediction = outputs["masks"]
    loss_parts = separation_loss_components(outputs, target, mask_target)
    direct = matched_separation_loss_per_sample(outputs, target, mask_target)
    swapped = matched_separation_loss_per_sample(outputs, target.flip(1), mask_target.flip(1))
    use_swapped = swapped["total"] < direct["total"]
    matched_mask_target = torch.where(
        use_swapped[:, None, None, None],
        mask_target.flip(1),
        mask_target,
    )
    per_spot_mask_dice = soft_dice_per_sample(mask_prediction, matched_mask_target)

    return {
        "loss_total": loss_parts["total"],
        "loss_mask_tversky": loss_parts["mask_tversky"],
        "loss_intensity_l1": loss_parts["intensity_l1"],
        "loss_mask_1": loss_parts["mask_1"],
        "loss_mask_2": loss_parts["mask_2"],
        "loss_intensity_1": loss_parts["intensity_1"],
        "loss_intensity_2": loss_parts["intensity_2"],
        "mask_soft_dice_spot_1": per_spot_mask_dice[:, 0].mean(),
        "mask_soft_dice_spot_2": per_spot_mask_dice[:, 1].mean(),
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
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    mask_target: torch.Tensor,
) -> torch.Tensor:
    return separation_loss_components(outputs, target, mask_target)["total"]


def evaluate(model, dataloader, device, amp):
    model.eval()
    metric_dicts = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            targets = batch["target"].to(device=device, dtype=torch.float32)
            masks = batch["mask"].to(device=device, dtype=torch.float32)

            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
                logits = model(images)
                if logits.shape[2:] != targets.shape[2:]:
                    logits = F.interpolate(
                        logits, size=targets.shape[2:], mode="bilinear", align_corners=False
                    )
                outputs = spot_separation_prediction(logits)
                metric_dicts.append(metrics_to_floats(separation_metric_tensors(outputs, targets, masks)))

    model.train()
    return average_metric_dicts(metric_dicts)




def separation_pixel_loss_maps(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    mask_target: torch.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    background_weight: float = 0.02,
    wrong_spot_weight: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Pixelwise diagnostic map matching the weighted mask/L1 loss ingredients."""
    pred_mask = outputs["masks"].clamp(0.0, 1.0)
    target_mask = (mask_target > 0.5).to(dtype=mask_target.dtype)
    other_mask = target_mask.flip(1)
    other_only = (other_mask > 0.5) & (target_mask <= 0.5)
    background = (other_mask <= 0.5) & (target_mask <= 0.5)

    false_positive_weights = torch.ones_like(pred_mask)
    false_positive_weights = torch.where(
        background,
        torch.full_like(false_positive_weights, background_weight),
        false_positive_weights,
    )
    false_positive_weights = torch.where(
        other_only,
        torch.full_like(false_positive_weights, wrong_spot_weight),
        false_positive_weights,
    )

    false_positive_map = pred_mask * (1.0 - target_mask) * false_positive_weights
    false_negative_map = (1.0 - pred_mask) * target_mask
    mask_map = alpha * false_positive_map + beta * false_negative_map
    intensity_map = (outputs["intensities"] - target).abs()
    total_map = mask_map.mean(dim=1) + intensity_map.mean(dim=1)
    return {
        "total": total_map,
        "mask": mask_map.mean(dim=1),
        "intensity": intensity_map.mean(dim=1),
    }


def matched_separation_pixel_loss_maps(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    mask_target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    direct = matched_separation_loss_per_sample(outputs, target, mask_target)
    swapped = matched_separation_loss_per_sample(outputs, target.flip(1), mask_target.flip(1))
    use_swapped = swapped["total"] < direct["total"]

    direct_maps = separation_pixel_loss_maps(outputs, target, mask_target)
    swapped_maps = separation_pixel_loss_maps(outputs, target.flip(1), mask_target.flip(1))
    return {
        name: torch.where(use_swapped[:, None, None], swapped_maps[name], direct_maps[name])
        for name in direct_maps
    }


def save_loss_diagnostic(
    model,
    dataloader,
    device,
    amp,
    output_dir: Path,
    epoch: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    best = None

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            targets = batch["target"].to(device=device, dtype=torch.float32)
            masks = batch["mask"].to(device=device, dtype=torch.float32)

            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
                logits = model(images)
                if logits.shape[2:] != targets.shape[2:]:
                    logits = F.interpolate(logits, size=targets.shape[2:], mode="bilinear", align_corners=False)
                outputs = spot_separation_prediction(logits)
                pixel_maps = matched_separation_pixel_loss_maps(outputs, targets, masks)

            total_maps = pixel_maps["total"]
            flat_values = total_maps.flatten(start_dim=1)
            sample_values, flat_indices = flat_values.max(dim=1)
            batch_best_idx = int(sample_values.argmax().detach().cpu().item())
            batch_best_value = float(sample_values[batch_best_idx].detach().cpu().item())
            if best is not None and batch_best_value <= best["value"]:
                continue

            height, width = total_maps.shape[-2:]
            del height
            flat_index = int(flat_indices[batch_best_idx].detach().cpu().item())
            row, col = divmod(flat_index, width)
            best = {
                "value": batch_best_value,
                "row": row,
                "col": col,
                "image": images[batch_best_idx, 0].detach().cpu().numpy(),
                "target_sum": targets[batch_best_idx].sum(dim=0).detach().cpu().numpy(),
                "prediction_sum": outputs["intensities"][batch_best_idx].sum(dim=0).detach().cpu().numpy(),
                "total_map": pixel_maps["total"][batch_best_idx].detach().cpu().numpy(),
                "mask_map": pixel_maps["mask"][batch_best_idx].detach().cpu().numpy(),
                "intensity_map": pixel_maps["intensity"][batch_best_idx].detach().cpu().numpy(),
            }

    if best is None:
        model.train()
        return

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    panels = [
        (best["image"], "input", "gray", None),
        (best["total_map"], f"total pixel loss max={best['value']:.4g}", "magma", "loss"),
        (best["mask_map"], "weighted mask error", "magma", "mask"),
        (best["target_sum"], "target intensity sum", "gray", None),
        (best["prediction_sum"], "predicted intensity sum", "gray", None),
        (best["intensity_map"], "intensity L1 error", "magma", "L1"),
    ]
    for axis, (data, title, cmap, colorbar_label) in zip(axes.flat, panels):
        image = axis.imshow(data, cmap=cmap)
        axis.scatter([best["col"]], [best["row"]], s=90, facecolors="none", edgecolors="cyan", linewidths=1.8)
        axis.set_title(title)
        axis.set_xlabel(f"max pixel: row {best['row']}, col {best['col']}")
        axis.set_xticks([])
        axis.set_yticks([])
        if colorbar_label is not None:
            colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            colorbar.set_label(colorbar_label)

    file_name = output_dir / f"epoch_{epoch:03d}_loss_diagnostic.png"
    fig.savefig(file_name, dpi=150)
    plt.close(fig)
    logging.info(
        "Saved loss diagnostic for epoch %d to %s; max pixel row=%d col=%d value=%s",
        epoch,
        file_name,
        best["row"],
        best["col"],
        best["value"],
    )
    model.train()

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
            masks = batch["mask"].to(device=device, dtype=torch.float32)

            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
                logits = model(images)
                if logits.shape[2:] != targets.shape[2:]:
                    logits = F.interpolate(logits, size=targets.shape[2:], mode="bilinear", align_corners=False)
                outputs = spot_separation_prediction(logits)
                predictions = outputs["intensities"]
                mask_predictions = outputs["masks"]

            batch_size = images.shape[0]
            for sample_idx in range(batch_size):
                if saved >= max_samples:
                    model.train()
                    return

                image = images[sample_idx, 0].detach().cpu().numpy()
                true_spots = targets[sample_idx].detach().cpu().numpy()
                pred_spots = predictions[sample_idx].detach().cpu().numpy()
                true_masks = masks[sample_idx].detach().cpu().numpy()
                pred_masks = mask_predictions[sample_idx].detach().cpu().numpy()
                true_sum = true_spots.sum(axis=0)
                pred_sum = pred_spots.sum(axis=0)

                fig, axes = plt.subplots(4, 4, figsize=(12, 12), constrained_layout=True)
                panels = [
                    (image, "input", "gray", 0.0, 1.0),
                    (true_spots[0], "true spot 1", "gray", 0.0, 1.0),
                    (pred_spots[0], "pred spot 1", "gray", 0.0, 1.0),
                    (np.abs(pred_spots[0] - true_spots[0]), "error spot 1", "magma", 0.0, 1.0),
                    (true_sum, "true sum", "gray", 0.0, 1.0),
                    (true_spots[1], "true spot 2", "gray", 0.0, 1.0),
                    (pred_spots[1], "pred spot 2", "gray", 0.0, 1.0),
                    (np.abs(pred_spots[1] - true_spots[1]), "error spot 2", "magma", 0.0, 1.0),
                    (pred_sum, "pred sum", "gray", 0.0, 1.0),
                    (true_masks[0], "true mask 1", "gray", 0.0, 1.0),
                    (pred_masks[0], "pred mask 1", "gray", 0.0, 1.0),
                    (np.abs(pred_masks[0] - true_masks[0]), "mask error 1", "magma", 0.0, 1.0),
                    (np.abs(pred_sum - image), "recon error", "magma", 0.0, 1.0),
                    (true_masks[1], "true mask 2", "gray", 0.0, 1.0),
                    (pred_masks[1], "pred mask 2", "gray", 0.0, 1.0),
                    (np.abs(pred_masks[1] - true_masks[1]), "mask error 2", "magma", 0.0, 1.0),
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
    loss_diagnostic_dir: Path = dir_loss_diagnostics,
    loss_diagnostic_interval: int = 50,
    early_stopping_patience: int = 10,
    debug_recorder: TrainingDebugRecorder | None = None,
):
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    if debug_recorder is not None:
        debug_recorder.update(status="running", phase="loading_dataset", h5_file=str(h5_file))
    dataset = H5SpotSeparationDataset(h5_file, img_scale)

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
    run_loss_diagnostic_dir = Path(loss_diagnostic_dir) / run_name
    if debug_recorder is not None:
        debug_recorder.update(
            phase="run_initialized",
            run_name=run_name,
            run_dir=str(run_dir),
            preview_dir=str(run_preview_dir),
            loss_diagnostic_dir=str(run_loss_diagnostic_dir),
            loss_diagnostic_interval=loss_diagnostic_interval,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device.type,
            amp=amp,
        )
    logging.info("TensorBoard log directory: %s", run_dir)
    logging.info("Prediction preview directory: %s", run_preview_dir)
    logging.info("Loss diagnostic directory: %s", run_loss_diagnostic_dir)
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
            "loss_diagnostic_interval": loss_diagnostic_interval,
            "early_stopping_patience": early_stopping_patience,
            "base_features": model.base_features,
            "optimizer": "AdamW",
            "loss": "permutation_invariant_weighted_Tversky(masks)+L1(intensities)",
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
        train_epoch_metrics = []
        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
            for batch_idx, batch in enumerate(train_loader, start=1):
                images = batch["image"]
                targets = batch["target"].to(device=device, dtype=torch.float32)
                masks = batch["mask"].to(device=device, dtype=torch.float32)

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
                    outputs = spot_separation_prediction(logits)
                    predictions = outputs["intensities"]
                    loss_parts = separation_loss_components(outputs, targets, masks)
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
                    batch_metrics = metrics_to_floats(separation_metric_tensors(outputs, targets, masks))
                train_epoch_metrics.append(batch_metrics)
                log_metrics(writer, "train_batch", batch_metrics, global_step)
                writer.add_scalar("Loss/train_batch", batch_metrics["loss_total"], global_step)
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
                    writer.add_image("spots/true_1", targets[0, 0:1].float(), global_step)
                    writer.add_image("spots/true_2", targets[0, 1:2].float(), global_step)
                    writer.add_image("spots/pred_1", predictions[0, 0:1].float(), global_step)
                    writer.add_image("spots/pred_2", predictions[0, 1:2].float(), global_step)
                    writer.add_image("masks/true_1", masks[0, 0:1].float(), global_step)
                    writer.add_image("masks/true_2", masks[0, 1:2].float(), global_step)
                    writer.add_image("masks/pred_1", outputs["masks"][0, 0:1].float(), global_step)
                    writer.add_image("masks/pred_2", outputs["masks"][0, 1:2].float(), global_step)

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
        if loss_diagnostic_interval > 0 and epoch % loss_diagnostic_interval == 0:
            if debug_recorder is not None:
                debug_recorder.update(
                    phase="saving_loss_diagnostic",
                    epoch=epoch,
                    epochs=epochs,
                    global_step=global_step,
                )
            save_loss_diagnostic(
                model,
                val_loader,
                device,
                amp,
                run_loss_diagnostic_dir,
                epoch,
            )
        writer.add_scalar("Loss/train_epoch", mean_epoch_loss, epoch)
        writer.add_scalar("Loss/validation_epoch", val_score, epoch)
        log_metrics(writer, "train_epoch", train_metrics, epoch)
        log_metrics(writer, "validation_epoch", val_metrics, epoch)
        logging.info("Epoch %d mean training loss: %s", epoch, mean_epoch_loss)
        logging.info("Epoch %d validation separation loss: %s", epoch, val_score)

        if n_val > 0 and val_score < best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            epochs_without_improvement = 0
            best_model_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            logging.info("New best validation loss at epoch %d: %s", epoch, val_score)
        elif n_val > 0:
            epochs_without_improvement += 1
            logging.info(
                "Validation loss did not improve for %d epoch(s)",
                epochs_without_improvement,
            )

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
            dir_checkpoint.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / f"checkpoint_epoch{epoch}.pth"))
            logging.info("Checkpoint %d saved!", epoch)

        if (
            early_stopping_patience
            and n_val > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            logging.info(
                "Early stopping at epoch %d; best validation loss was %s at epoch %d",
                epoch,
                best_val_score,
                best_epoch,
            )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logging.info("Restored best model weights from epoch %d", best_epoch)

    dir_checkpoint.mkdir(parents=True, exist_ok=True)
    final_model_file = dir_checkpoint / safe_model_file_name(run_name)
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
    parser.add_argument(
        "--base-features",
        type=int,
        default=32,
        help="Number of features in the first U-Net level",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Stop after this many epochs without validation improvement (0 disables)",
    )
    parser.add_argument(
        "--paths-file",
        type=str,
        default=str(dir_project_paths),
        help="Path to TOML file with project paths",
    )
    parser.add_argument(
        "--h5-file",
        type=str,
        default=None,
        help="Override input HDF5 file from project_paths.toml",
    )
    parser.add_argument("--log-dir", type=str, default=str(dir_runs), help="TensorBoard root log directory")
    parser.add_argument("--run-name", type=str, default=None, help="Optional TensorBoard run name")
    parser.add_argument("--preview-dir", type=str, default=str(dir_previews), help="Directory for saved prediction PNG previews")
    parser.add_argument("--preview-samples", type=int, default=4, help="Number of validation previews to save per epoch")
    parser.add_argument(
        "--loss-diagnostic-dir",
        type=str,
        default=str(dir_loss_diagnostics),
        help="Directory for per-pixel loss diagnostic PNGs",
    )
    parser.add_argument(
        "--loss-diagnostic-interval",
        type=int,
        default=50,
        help="Save a per-pixel loss diagnostic every N epochs; use 0 to disable",
    )
    parser.add_argument("--debug-dir", type=str, default=str(dir_debug), help="Directory for durable debug logs and heartbeat state")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    paths_file = Path(args.paths_file)
    input_h5_file = Path(args.h5_file).expanduser() if args.h5_file else configured_input_h5(paths_file)
    debug_log_file, state_file, debug_recorder = setup_logging(Path(args.debug_dir))
    install_signal_logging(debug_recorder)
    debug_recorder.update(
        phase="arguments_parsed",
        command=" ".join(sys.argv),
        debug_log_file=str(debug_log_file),
        state_file=str(state_file),
        paths_file=str(paths_file),
        input_h5_file=str(input_h5_file),
    )

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        debug_recorder.update(phase="device_selected", device=str(device), cuda_available=torch.cuda.is_available())
        logging.info("Using device %s", device)

        model = UNet(
            n_channels=1,
            n_classes=4,
            bilinear=args.bilinear,
            base_features=args.base_features,
        )
        model = model.to(memory_format=torch.channels_last)

        logging.info(
            "Network: %d input channel, %d output channels, %s upscaling",
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
                h5_file=input_h5_file,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                img_scale=args.scale,
                val_percent=args.val / 100,
                amp=args.amp,
                log_dir=Path(args.log_dir),
                run_name=args.run_name,
                preview_dir=Path(args.preview_dir),
                preview_samples=args.preview_samples,
                loss_diagnostic_dir=Path(args.loss_diagnostic_dir),
                loss_diagnostic_interval=args.loss_diagnostic_interval,
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
                h5_file=input_h5_file,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                img_scale=args.scale,
                val_percent=args.val / 100,
                amp=args.amp,
                log_dir=Path(args.log_dir),
                run_name=args.run_name,
                preview_dir=Path(args.preview_dir),
                preview_samples=args.preview_samples,
                loss_diagnostic_dir=Path(args.loss_diagnostic_dir),
                loss_diagnostic_interval=args.loss_diagnostic_interval,
                early_stopping_patience=args.early_stopping_patience,
                debug_recorder=debug_recorder,
            )
    except Exception as exc:
        logging.exception("Training failed with an unhandled exception.")
        debug_recorder.record_exception(exc)
        raise
