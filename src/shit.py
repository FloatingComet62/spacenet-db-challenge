
#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.dataset import BuildingDataset
from training.metrics import batch_metrics, calculate_scores


CHECKPOINT = ROOT / "checkpoints" / "noop.pt"
DATA_DIR = ROOT / "data" / "building_dataset"
BATCH_SIZE = 4
NUM_WORKERS = 2


def dice_loss(logits, targets, eps=1e-6):
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )
    dice = (2 * intersection + eps) / (denominator + eps)
    return 1 - dice.mean()


def combined_loss(logits, targets):
    return nn.functional.binary_cross_entropy_with_logits(
        logits, targets
    ) + dice_loss(logits, targets)


def resolve_files(files):
    resolved = []

    for file in files:
        path = Path(file)

        if not path.is_absolute():
            path = ROOT / path

        if not path.exists():
            # Check whether the checkpoint stores only a filename.
            path = DATA_DIR / Path(file).name

        if not path.exists():
            raise FileNotFoundError(
                f"Validation tile not found: {file}"
            )

        resolved.append(path)

    return resolved


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Device: {device}")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    if "val_files" not in checkpoint:
        raise KeyError(
            "Checkpoint has no 'val_files'. "
            "Cannot reproduce the original validation split."
        )

    val_files = resolve_files(checkpoint["val_files"])
    print(f"Validation tiles: {len(val_files)}")

    dataset = BuildingDataset(val_files)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(device)

    # Support common checkpoint key names.
    state_dict = None
    for key in ("model_state_dict", "model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            if all(torch.is_tensor(v) for v in value.values()):
                state_dict = value
                break

    if state_dict is None:
        raise KeyError(
            "Could not find model weights in the checkpoint. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    model.load_state_dict(state_dict)
    model.eval()

    total_loss = 0.0
    total_samples = 0
    totals = [0, 0, 0, 0]

    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            loss = combined_loss(logits, masks)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            metrics = batch_metrics(logits, masks)
            totals = [
                old + new
                for old, new in zip(totals, metrics)
            ]

    iou, dice, precision, recall = calculate_scores(totals)
    avg_loss = total_loss / max(total_samples, 1)

    print("\nValidation results")
    print("-" * 40)
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation patches: {total_samples}")
    print(f"Validation loss: {avg_loss:.6f}")
    print(f"IoU:              {iou:.4f}")
    print(f"Dice:             {dice:.4f}")
    print(f"Precision:        {precision:.4f}")
    print(f"Recall:           {recall:.4f}")


if __name__ == "__main__":
    main()
