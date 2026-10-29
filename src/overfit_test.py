from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import segmentation_models_pytorch as smp

from training.dataset import BuildingDataset
from training.metrics import batch_metrics, calculate_scores

DATA_DIR = Path("data/building_dataset")
SEED = 111
PATCHES_TO_FIT = 8
BATCH_SIZE = 2
EPOCHS = 50
LEARNING_RATE = 1e-4

random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
files = sorted(DATA_DIR.glob("*.npz"))
dataset = BuildingDataset(files)

# Identify patches with at least one positive pixel.
positive_indices = []

for i, (path, y, x) in enumerate(dataset.samples):
    import numpy as np

    with np.load(path) as data:
        mask = data["mask"][y:y + 512, x:x + 512]

    if mask.any():
        positive_indices.append(i)

if len(positive_indices) < PATCHES_TO_FIT:
    raise RuntimeError("Not enough positive patches")

# Use patches with the largest building coverage for this diagnostic.
def positive_count(index):
    path, y, x = dataset.samples[index]
    import numpy as np

    with np.load(path) as data:
        return np.count_nonzero(data["mask"][y:y + 512, x:x + 512])

selected = sorted(
    positive_indices,
    key=positive_count,
    reverse=True,
)[:PATCHES_TO_FIT]

subset = Subset(dataset, selected)
loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None,
).to(DEVICE)

bce = nn.BCEWithLogitsLoss()

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1 - ((2 * intersection + eps) / (denominator + eps)).mean()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(1, EPOCHS + 1):
    model.train()
    loss_sum = 0.0
    totals = [0, 0, 0, 0]

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = bce(logits, masks) + dice_loss(logits, masks)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * images.size(0)
        batch = batch_metrics(logits.detach(), masks)
        totals = [a + b for a, b in zip(totals, batch)]

    iou, dice, precision, recall = calculate_scores(totals)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"loss={loss_sum / len(subset):.4f} | "
        f"IoU={iou:.4f} | Dice={dice:.4f} | "
        f"Precision={precision:.4f} | Recall={recall:.4f}"
    )
