#!/usr/bin/env python3

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.dataset import BuildingDataset, PATCH_SIZE


CHECKPOINT = ROOT / "checkpoints" / "noop.pt"
BATCH_SIZE = 1
THRESHOLD = 0.5


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


def load_model(device):
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

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
            f"Could not find model weights. Keys: {list(checkpoint.keys())}"
        )

    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint


def make_samples(files):
    dataset = BuildingDataset(files)
    return dataset


def denormalize(image_tensor):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * std + mean

    return np.clip(image, 0, 1)


class PredictionViewer:
    def __init__(self, model, checkpoint, device):
        self.model = model
        self.checkpoint = checkpoint
        self.device = device
        self.split_name = "validation"
        self.index = 0
        self.threshold = THRESHOLD

        self.datasets = {
            "training": make_samples(
                resolve_files(checkpoint["train_files"])
            ),
            "validation": make_samples(
                resolve_files(checkpoint["val_files"])
            ),
        }

        self.fig, self.axes = plt.subplots(1, 4, figsize=(17, 5))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()

        print("\nControls:")
        print("  Right / n : next patch")
        print("  Left / p  : previous patch")
        print("  t         : toggle training / validation")
        print("  + / -     : adjust prediction threshold")
        print("  q / Esc   : quit")

        plt.show()

    def on_key(self, event):
        if event.key in ("right", "n"):
            self.index += 1
        elif event.key in ("left", "p"):
            self.index -= 1
        elif event.key == "t":
            self.split_name = (
                "training"
                if self.split_name == "validation"
                else "validation"
            )
            self.index = 0
        elif event.key in ("+", "="):
            self.threshold = min(0.95, self.threshold + 0.05)
        elif event.key == "-":
            self.threshold = max(0.05, self.threshold - 0.05)
        elif event.key in ("q", "escape"):
            plt.close(self.fig)
            return
        else:
            return

        dataset = self.datasets[self.split_name]
        self.index = max(0, min(self.index, len(dataset) - 1))
        self.draw()

    def draw(self):
        dataset = self.datasets[self.split_name]
        image_tensor, mask_tensor = dataset[self.index]

        with torch.inference_mode():
            logits = self.model(
                image_tensor.unsqueeze(0).to(self.device)
            )
            probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

        image = denormalize(image_tensor)
        ground_truth = mask_tensor[0].numpy() >= 0.5
        prediction = probabilities >= self.threshold

        intersection = np.logical_and(prediction, ground_truth).sum()
        union = np.logical_or(prediction, ground_truth).sum()
        iou = intersection / union if union else 1.0

        pred_pixels = int(prediction.sum())
        true_pixels = int(ground_truth.sum())

        overlay = image.copy()
        # Green: true positive; red: false positive; blue: false negative.
        tp = prediction & ground_truth
        fp = prediction & ~ground_truth
        fn = ~prediction & ground_truth

        overlay[tp] = (
            0.45 * overlay[tp] + 0.55 * np.array([0.0, 1.0, 0.0])
        )
        overlay[fp] = (
            0.45 * overlay[fp] + 0.55 * np.array([1.0, 0.0, 0.0])
        )
        overlay[fn] = (
            0.45 * overlay[fn] + 0.55 * np.array([0.0, 0.4, 1.0])
        )

        titles = [
            "Satellite image",
            "Ground truth",
            f"Prediction (threshold={self.threshold:.2f})",
            f"Overlay (IoU={iou:.3f})",
        ]
        images = [image, ground_truth, prediction, overlay]
        cmaps = [None, "gray", "gray", None]

        for ax, title, img, cmap in zip(
            self.axes, titles, images, cmaps
        ):
            ax.clear()
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis("off")

        self.fig.suptitle(
            f"{self.split_name.upper()} | "
            f"patch {self.index + 1}/{len(dataset)} | "
            f"GT pixels: {true_pixels:,} | "
            f"predicted pixels: {pred_pixels:,}"
        )
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Device: {device}")

    model, checkpoint = load_model(device)
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    PredictionViewer(model, checkpoint, device)


if __name__ == "__main__":
    main()
