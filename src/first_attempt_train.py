from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

DATA_DIR = Path("data/building_dataset")
OUTPUT_DIR = Path("checkpoints")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 111
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.2
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


from training.dataset import *

all_files = sorted(DATA_DIR.glob("*.npz"))

if len(all_files) < 2:
    raise RuntimeError(
        f"Need at least 2 NPZ files in {DATA_DIR.resolve()}; "
        f"found {len(all_files)}."
    )

rng = random.Random(SEED)
shuffled = all_files.copy()
rng.shuffle(shuffled)

n_val = max(1, round(len(shuffled) * VAL_FRACTION))
n_val = min(n_val, len(shuffled) - 1)

val_files = shuffled[:n_val]
train_files = shuffled[n_val:]

print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'}")
print(f"Image tiles: {len(all_files)}")
print(f"Train tiles: {len(train_files)}")
print(f"Validation tiles: {len(val_files)}")

train_dataset = BuildingDataset(train_files)
val_dataset = BuildingDataset(val_files)

print(f"Training patches: {len(train_dataset)}")
print(f"Validation patches: {len(val_dataset)}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)


model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None,
).to(DEVICE)

bce_loss = nn.BCEWithLogitsLoss()

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
    return bce_loss(logits, targets) + dice_loss(logits, targets)


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

use_amp = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

from training.metrics import *

best_val_iou = -1.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss_sum = 0.0

    for images, masks in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = combined_loss(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss_sum += loss.item() * images.size(0)

    train_loss = train_loss_sum / len(train_dataset)

    # Evaluate without gradient tracking.
    model.eval()
    val_loss_sum = 0.0
    totals = [0, 0, 0, 0]

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = combined_loss(logits, masks)

            val_loss_sum += loss.item() * images.size(0)

            batch = batch_metrics(logits.float(), masks)
            totals = [a + b for a, b in zip(totals, batch)]

    val_loss = val_loss_sum / len(val_dataset)
    iou, dice, precision, recall = calculate_scores(totals)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"IoU={iou:.4f} | Dice={dice:.4f} | "
        f"Precision={precision:.4f} | Recall={recall:.4f}"
    )

    if iou <= best_val_iou:
        continue
    best_val_iou = iou

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_iou": iou,
            "val_dice": dice,
            "seed": SEED,
            "train_files": [p.name for p in train_files],
            "val_files": [p.name for p in val_files],
        },
        OUTPUT_DIR / "best_unet_resnet34.pt",
    )

    print("  Saved best checkpoint.")

print(f"\nBest validation IoU: {best_val_iou:.4f}")
print(f"Checkpoint: {OUTPUT_DIR / 'best_unet_resnet34.pt'}")
