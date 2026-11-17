
#!/usr/bin/env python3

from pathlib import Path
import sys

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.dataset import BuildingDataset
from training.metrics import batch_metrics, calculate_scores

CHECKPOINT = ROOT / "checkpoints" / "noop.pt"
BATCH_SIZE = 4
NUM_WORKERS = 2


def combined_loss(logits, targets):
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets
    )

    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )
    dice = (2 * intersection + 1e-6) / (denominator + 1e-6)

    return bce + (1 - dice.mean())


def resolve_files(files):
    resolved = []

    for file in files:
        path = Path(file)
        if not path.is_absolute():
            path = ROOT / "data" / "building_dataset" / path

        if not path.exists():
            raise FileNotFoundError(f"Tile not found: {path}")

        resolved.append(path)

    return resolved


def evaluate(model, files, device, split_name):
    dataset = BuildingDataset(files)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    total_loss = 0.0
    total_samples = 0
    totals = [0, 0, 0, 0]

    model.eval()

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
                old + new for old, new in zip(totals, metrics)
            ]

    iou, dice, precision, recall = calculate_scores(totals)
    avg_loss = total_loss / max(total_samples, 1)

    print(f"\n{split_name} results")
    print("-" * 45)
    print(f"Tiles:            {len(files)}")
    print(f"Patches:          {total_samples}")
    print(f"Loss:             {avg_loss:.6f}")
    print(f"IoU:              {iou:.4f}")
    print(f"Dice:             {dice:.4f}")
    print(f"Precision:        {precision:.4f}")
    print(f"Recall:           {recall:.4f}")

    return {
        "loss": avg_loss,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
    }


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

    print(f"Saved epoch: {checkpoint.get('epoch', 'unknown')}")

    for key in ("train_files", "val_files"):
        if key not in checkpoint:
            raise KeyError(
                f"Checkpoint is missing '{key}'. "
                f"Available keys: {list(checkpoint.keys())}"
            )

    train_files = resolve_files(checkpoint["train_files"])
    val_files = resolve_files(checkpoint["val_files"])

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(device)

    state_dict = None
    for key in ("model_state_dict", "model", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict) and candidate:
            if all(torch.is_tensor(v) for v in candidate.values()):
                state_dict = candidate
                break

    if state_dict is None:
        raise KeyError(
            f"Could not find model weights. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    model.load_state_dict(state_dict)

    print(f"Training tiles: {len(train_files)}")
    print(f"Validation tiles: {len(val_files)}")

    train_results = evaluate(model, train_files, device, "TRAINING")
    val_results = evaluate(model, val_files, device, "VALIDATION")

    print("\nComparison")
    print("=" * 45)
    for metric in ("loss", "iou", "dice", "precision", "recall"):
        print(
            f"{metric:12s} "
            f"train={train_results[metric]:.4f}  "
            f"val={val_results[metric]:.4f}"
        )


if __name__ == "__main__":
    main()
