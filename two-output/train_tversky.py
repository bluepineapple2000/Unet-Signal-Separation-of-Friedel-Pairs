import argparse
import atexit
import json
import logging
import os
import signal
import sys
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
dir_h5_spot_segmentation = Path("./data/augmented_spot_patches.h5")
dir_runs = Path("./runs/")
dir_previews = Path("./prediction_previews/")
dir_debug = Path("./debug_logs/")


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


def target_masks_from_intensity(
    target: torch.Tensor,
    foreground_threshold: float = 1e-4,
) -> torch.Tensor:
    return (target > foreground_threshold).to(dtype=target.dtype)


def predicted_presence_from_intensity(prediction: torch.Tensor) -> torch.Tensor:
    # Targets are normalized to roughly [0, 1], so clamp intensity predictions into a soft mask range.
    return prediction.clamp(0.0, 1.0)


def weighted_tversky_loss_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    foreground_threshold: float = 1e-4,
    alpha: float = 0.7,
    beta: float = 0.3,
    background_weight: float = 0.02,
    wrong_spot_weight: float = 2.0,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Tversky mask loss for already matched two-channel predictions.

    Own-spot and true overlap pixels are positives. Other-spot-only pixels are weighted
    false positives. Plain background false positives are still visible but very weak.
    """
    pred_mask = predicted_presence_from_intensity(prediction)
    target_mask = target_masks_from_intensity(target, foreground_threshold=foreground_threshold)
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
    return 1.0 - score.mean(dim=1)


def foreground_intensity_l1_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    foreground_threshold: float = 1e-4,
    overlap_weight: float = 0.5,
) -> torch.Tensor:
    """Intensity loss only where the matched target spot exists, including true overlap."""
    target_mask = target_masks_from_intensity(target, foreground_threshold=foreground_threshold)
    overlap = (target_mask > 0.5) & (target_mask.flip(1) > 0.5)
    weights = torch.where(overlap, torch.full_like(target_mask, overlap_weight), target_mask)
    error = (prediction - target).abs() * weights
    return error.sum(dim=(1, 2, 3)) / weights.sum(dim=(1, 2, 3)).clamp_min(1.0)


def matched_separation_loss_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask_loss_weight: float = 1.0,
    intensity_loss_weight: float = 0.2,
) -> dict[str, torch.Tensor]:
    mask = weighted_tversky_loss_per_sample(prediction, target)
    intensity = foreground_intensity_l1_per_sample(prediction, target)
    total = mask_loss_weight * mask + intensity_loss_weight * intensity
    return {
        "total": total,
        "mask_tversky": mask,
        "intensity": intensity,
    }


def separation_loss_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    image: torch.Tensor,
) -> dict[str, torch.Tensor]:
    del image  # Separation is supervised by the two target spot channels, not by reconstruction.
    direct = matched_separation_loss_per_sample(prediction, target)
    swapped = matched_separation_loss_per_sample(prediction, target.flip(1))
    use_swapped = swapped["total"] < direct["total"]
    selected = {
        name: torch.where(use_swapped, swapped[name], direct[name]).mean()
        for name in direct
    }
    return selected


def separation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    image: torch.Tensor,
) -> torch.Tensor:
    return separation_loss_components(prediction, target, image)["total"]


def align_prediction_channels(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    direct = matched_separation_loss_per_sample(prediction, target)["total"]
    swapped = matched_separation_loss_per_sample(prediction, target.flip(1))["total"]
    use_swapped = swapped < direct
    aligned = prediction.clone()
    aligned[use_swapped] = prediction[use_swapped].flip(1)
    return aligned


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
                if logits.shape != targets.shape:
                    logits = F.interpolate(logits, size=targets.shape[2:], mode="bilinear", align_corners=False)
                predictions = align_prediction_channels(spot_intensity_prediction(logits), targets)

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
                error = np.abs(pred_sum - true_sum)

                fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
                panels = [
                    (image, "input", "gray", 0.0, 1.0),
                    (true_spots[0], "true spot 1", "gray", 0.0, 1.0),
                    (pred_spots[0], "pred spot 1", "gray", 0.0, 1.0),
                    (np.abs(pred_spots[0] - true_spots[0]), "error spot 1", "magma", 0.0, 1.0),
                    (true_sum, "true sum", "gray", 0.0, 1.0),
                    (true_spots[1], "true spot 2", "gray", 0.0, 1.0),
                    (pred_spots[1], "pred spot 2", "gray", 0.0, 1.0),
                    (error, "sum error", "magma", 0.0, 1.0),
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
    debug_recorder: TrainingDebugRecorder | None = None,
):
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
    if debug_recorder is not None:
        debug_recorder.update(
            phase="run_initialized",
            run_name=run_name,
            run_dir=str(run_dir),
            preview_dir=str(run_preview_dir),
            epochs=epochs,
            batch_size=batch_size,
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
            "optimizer": "AdamW",
            "loss": "PI_weighted_tversky_mask+0.2_foreground_intensity_l1",
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
                loss_values = (
                    torch.stack(
                        (
                            loss_parts["total"],
                            loss_parts["mask_tversky"],
                            loss_parts["intensity"],
                        )
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )
                total_loss, mask_tversky_loss, intensity_loss = loss_values
                epoch_loss += total_loss
                writer.add_scalar("Loss/train_batch", total_loss, global_step)
                writer.add_scalar("Loss_parts/train_mask_tversky", mask_tversky_loss, global_step)
                writer.add_scalar("Loss_parts/train_intensity", intensity_loss, global_step)
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
    parser.add_argument("--h5-file", type=str, default=str(dir_h5_spot_segmentation), help="Path to HDF5 file")
    parser.add_argument("--log-dir", type=str, default=str(dir_runs), help="TensorBoard root log directory")
    parser.add_argument("--run-name", type=str, default=None, help="Optional TensorBoard run name")
    parser.add_argument("--preview-dir", type=str, default=str(dir_previews), help="Directory for saved prediction PNG previews")
    parser.add_argument("--preview-samples", type=int, default=4, help="Number of validation previews to save per epoch")
    parser.add_argument("--debug-dir", type=str, default=str(dir_debug), help="Directory for durable debug logs and heartbeat state")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    debug_log_file, state_file, debug_recorder = setup_logging(Path(args.debug_dir))
    install_signal_logging(debug_recorder)
    debug_recorder.update(
        phase="arguments_parsed",
        command=" ".join(sys.argv),
        debug_log_file=str(debug_log_file),
        state_file=str(state_file),
    )

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        debug_recorder.update(phase="device_selected", device=str(device), cuda_available=torch.cuda.is_available())
        logging.info("Using device %s", device)

        model = UNet(n_channels=1, n_classes=2, bilinear=args.bilinear)
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
                h5_file=Path(args.h5_file),
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
                h5_file=Path(args.h5_file),
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
                debug_recorder=debug_recorder,
            )
    except Exception as exc:
        logging.exception("Training failed with an unhandled exception.")
        debug_recorder.record_exception(exc)
        raise
