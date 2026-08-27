import argparse
import logging
import os
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
dir_h5_spot_segmentation = Path("../data_esrf/augmented_spot_patches.h5")
dir_runs = Path("./runs/")
dir_previews = Path("./prediction_previews/")


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
        lo, hi = np.percentile(values, [1, 99.8])
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


def permutation_invariant_l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    direct = F.l1_loss(prediction, target, reduction="none").mean(dim=(1, 2, 3))
    swapped = F.l1_loss(prediction, target.flip(1), reduction="none").mean(dim=(1, 2, 3))
    return torch.minimum(direct, swapped).mean()


def align_prediction_channels(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    direct = F.l1_loss(prediction, target, reduction="none").mean(dim=(1, 2, 3))
    swapped = F.l1_loss(prediction, target.flip(1), reduction="none").mean(dim=(1, 2, 3))
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
                losses.append(permutation_invariant_l1_loss(prediction, targets))

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


def train_model(
    model,
    device,
    h5_file: Path = dir_h5_spot_segmentation,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    val_percent: float = 0.1,
    save_checkpoint: bool = True,
    img_scale: float = 0.5,
    amp: bool = False,
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    gradient_clipping: float = 1.0,
    log_dir: Path = dir_runs,
    run_name: str | None = None,
    preview_dir: Path = dir_previews,
    preview_samples: int = 4,
):
    dataset = H5SpotSeparationDataset(h5_file, img_scale)

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
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

    optimizer = optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        foreach=True,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)
    grad_scaler = torch.amp.GradScaler(device.type, enabled=amp)
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
            for batch in train_loader:
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
                    loss = permutation_invariant_l1_loss(predictions, targets)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                writer.add_scalar("Loss/train_batch", loss.item(), global_step)
                pbar.set_postfix(**{"loss (batch)": loss.item()})

                division_step = n_train // (5 * batch_size)
                if division_step > 0 and global_step % division_step == 0:
                    for tag, value in model.named_parameters():
                        tag = tag.replace("/", ".")
                        if not (torch.isinf(value) | torch.isnan(value)).any():
                            writer.add_histogram(f"Weights/{tag}", value.data, global_step)
                        if value.grad is not None and not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                            writer.add_histogram(f"Gradients/{tag}", value.grad.data, global_step)

                    val_score = evaluate(model, val_loader, device, amp)
                    scheduler.step(val_score)

                    logging.info("Validation separation L1 loss: %s", val_score)
                    writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], global_step)
                    writer.add_scalar("Loss/validation_step", val_score, global_step)
                    writer.add_image("input_images", images[0], global_step)
                    writer.add_image("spots/true_1", targets[0, 0:1].float(), global_step)
                    writer.add_image("spots/true_2", targets[0, 1:2].float(), global_step)
                    aligned = align_prediction_channels(predictions[:1], targets[:1])
                    writer.add_image("spots/pred_1", aligned[0, 0:1].float(), global_step)
                    writer.add_image("spots/pred_2", aligned[0, 1:2].float(), global_step)

        mean_epoch_loss = epoch_loss / max(1, len(train_loader))
        val_score = evaluate(model, val_loader, device, amp)
        scheduler.step(val_score)
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
        logging.info("Epoch %d validation separation L1 loss: %s", epoch, val_score)

        if save_checkpoint:
            dir_checkpoint.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / f"checkpoint_epoch{epoch}.pth"))
            logging.info("Checkpoint %d saved!", epoch)

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description="Train the UNet on HDF5 spot separation data")
    parser.add_argument("--epochs", "-e", metavar="E", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", "-b", dest="batch_size", metavar="B", type=int, default=1, help="Batch size")
    parser.add_argument("--learning-rate", "-l", metavar="LR", type=float, default=1e-5, help="Learning rate", dest="lr")
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
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        )
    except torch.cuda.OutOfMemoryError:
        logging.error(
            "Detected OutOfMemoryError. Enabling checkpointing to reduce memory usage. "
            "Consider enabling AMP (--amp) for faster and more memory-efficient training."
        )
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
        )
