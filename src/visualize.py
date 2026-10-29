from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=Path("checkpoints/best_unet_resnet34.pt"),
)
args = parser.parse_args()

DATA_DIR = Path("data/building_dataset")
CHECKPOINT = args.checkpoint


PATCH_SIZE = 512
THRESHOLD = 0.5
NUM_IMAGES = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def patch_starts(length, patch_size):
    if length <= patch_size:
        return [0]

    starts = list(range(0, length - patch_size + 1, patch_size))
    last = length - patch_size

    if starts[-1] != last:
        starts.append(last)

    return starts


# Load model and checkpoint.
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Device:", DEVICE)
print("Checkpoint epoch:", checkpoint["epoch"])
print("Validation IoU:", checkpoint["val_iou"])


def predict_mask(image):
    """Predict a full image using overlapping patches."""
    height, width = image.shape[:2]

    # Accumulate logits where patches overlap.
    logits_sum = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    for y in patch_starts(height, PATCH_SIZE):
        for x in patch_starts(width, PATCH_SIZE):
            patch = image[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE,
            ].astype(np.float32)

            # Same normalization used during training.
            patch /= 255.0
            patch = (patch - MEAN) / STD
            patch = np.moveaxis(patch, -1, 0).copy()

            tensor = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                with torch.amp.autocast(
                    device_type=DEVICE.type,
                    dtype=torch.float16,
                    enabled=(DEVICE.type == "cuda"),
                ):
                    logits = model(tensor)

            logits = logits.float().squeeze().cpu().numpy()

            h = min(PATCH_SIZE, height - y)
            w = min(PATCH_SIZE, width - x)

            logits_sum[y:y+h, x:x+w] += logits[:h, :w]
            counts[y:y+h, x:x+w] += 1

    averaged_logits = logits_sum / np.maximum(counts, 1)
    probabilities = 1.0 / (1.0 + np.exp(-averaged_logits))

    return probabilities >= THRESHOLD


def overlay(image, mask, color):
    """Blend a binary mask over an RGB image."""
    result = image.astype(np.float32).copy()
    color = np.asarray(color, dtype=np.float32)

    result[mask] = (
        0.5 * result[mask] + 0.5 * color
    )

    return np.clip(result, 0, 255).astype(np.uint8)


# Prefer validation tiles saved in the checkpoint.
val_names = checkpoint.get("val_files", [])
files = [DATA_DIR / name for name in val_names]
files = [p for p in files if p.exists()]

# Fallback for older checkpoints.
if not files:
    files = sorted(DATA_DIR.glob("*.npz"))

if not files:
    raise RuntimeError(f"No .npz files found in {DATA_DIR.resolve()}")

files = files[:NUM_IMAGES]
index = 0

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
plt.subplots_adjust(bottom=0.14)


def show_image():
    global index

    path = files[index]

    with np.load(path) as data:
        image = data["image"].copy()
        mask = data["mask"].astype(bool).copy()

    prediction = predict_mask(image)

    # Green = ground truth; magenta = model prediction.
    gt_overlay = overlay(image, mask, [0, 255, 0])
    pred_overlay = overlay(image, prediction, [255, 0, 255])

    for ax in axes:
        ax.clear()
        ax.axis("off")

    axes[0].imshow(image)
    axes[0].set_title("Original image")

    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground truth (green)")

    axes[2].imshow(pred_overlay)
    axes[2].set_title("Prediction (magenta)")

    intersection = np.logical_and(mask, prediction).sum()
    union = np.logical_or(mask, prediction).sum()
    iou = intersection / union if union else 1.0

    fig.suptitle(
        f"{index + 1}/{len(files)} — {path.stem} | "
        f"Tile IoU: {iou:.3f}",
        fontsize=12,
    )

    fig.canvas.draw_idle()


def on_key(event):
    global index

    if event.key == " ":
        index = (index + 1) % len(files)
        show_image()
    elif event.key == "backspace":
        index = (index - 1) % len(files)
        show_image()


fig.canvas.mpl_connect("key_press_event", on_key)

print("Space: next image | Backspace: previous image")
show_image()
plt.show()
